"""
sunrgbd.py — core utilities for working with the SUN RGB-D dataset.

Covers:
  * scene discovery in the arbitrary SUN RGB-D directory layout;
  * correct decoding of 16-bit depth maps (bit shift!);
  * reading camera intrinsics;
  * parsing 2D polygon annotations from annotation2Dfinal/index.json;
  * class-name normalization;
  * back-projection of pixels into 3D (used in chapter 3 of the thesis).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as mask_utils


MAX_DEPTH_M = 8.0

# Categories that are NOT object instances ("stuff" rather than "things").
# Excluded from instance segmentation: they have no well-defined instance
# boundaries.
STUFF_CLASSES = {
    "wall", "floor", "ceiling", "room", "background", "unknown",
    "wall decoration", "floor mat", "ceiling light",
}

# The standard 10-class SUN RGB-D list (used by most detection/segmentation
# publications on this dataset — keeps results comparable).
STANDARD_10_CLASSES = [
    "bed", "chair", "sofa", "table", "desk",
    "toilet", "dresser", "nightstand", "bookshelf", "bathtub",
]

SYNONYMS = {
    "chairs": "chair", "office chair": "chair", "armchair": "chair",
    "desk chair": "chair", "swivel chair": "chair",
    "tables": "table", "coffee table": "table",
    "dining table": "table", "side table": "table", "end table": "table",
    "desks": "desk", "office desk": "desk", "computer desk": "desk",
    "writing desk": "desk", "study desk": "desk",
    "beds": "bed", "double bed": "bed", "bunk bed": "bed",
    "toilets": "toilet", "toilet bowl": "toilet",
    "bath tub": "bathtub", "bathtubs": "bathtub", "tub": "bathtub",
    "dressers": "dresser", "chest of drawers": "dresser",
    "bookcase": "bookshelf",
    "night stand": "nightstand", "bedside table": "nightstand",
    "night table": "nightstand", "nightstands": "nightstand",
    "monitor": "monitor", "computer monitor": "monitor", "screen": "monitor",
    "tv": "television", "tv screen": "television",
    "garbage bin": "garbage bin", "trash can": "garbage bin",
    "trash bin": "garbage bin", "garbage can": "garbage bin",
    "waste basket": "garbage bin", "bin": "garbage bin",
    "book shelf": "bookshelf", "bookshelves": "bookshelf",
    "shelves": "shelf", "shelfs": "shelf",
    "cabinets": "cabinet", "kitchen cabinet": "cabinet",
    "couch": "sofa", "sofa chair": "sofa",
    "pillows": "pillow", "cushion": "pillow",
    "lamps": "lamp", "table lamp": "lamp", "desk lamp": "lamp",
    "picture frame": "picture", "pictures": "picture",
    "books": "book", "papers": "paper", "bottles": "bottle",
    "boxes": "box", "cups": "cup", "mug": "cup",
    "computer": "computer", "pc": "computer", "cpu": "computer",
    "keyboard ": "keyboard", "white board": "whiteboard",
    "door ": "door", "doors": "door", "window ": "window", "windows": "window",
    # frequent typos found in the raw annotations
    "chiar": "chair", "cabnet": "cabinet", "bathtube": "bathtub",
}

_SQUASHED_SYNONYMS: dict[str, str] = {}
for _alias, _canon in SYNONYMS.items():
    _SQUASHED_SYNONYMS.setdefault(_alias.strip().replace(" ", "").replace("-", ""), _canon)
for _canon in set(SYNONYMS.values()):
    _SQUASHED_SYNONYMS.setdefault(_canon.replace(" ", "").replace("-", ""), _canon)

_CANONICAL_NAMES = frozenset(SYNONYMS.values())

_TRAILING_NUM = re.compile(r"[\s_\-#]*\d+\s*$")
_MULTISPACE = re.compile(r"\s+")

@dataclass
class Scene:
    """A single scene (frame) of the SUN RGB-D dataset."""

    scene_dir: Path
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path | None
    annotation_path: Path | None
    sensor: str                 # kv1 / kv2 / xtion / realsense
    collection: str             # NYUdata / b3dodata / sun3ddata / ...
    rel_key: str                # unique scene key relative to the dataset root
    group_key: str = ""         # grouping key for the split (see assign_groups)
    scene_type: str = ""        # room type from scene.txt

    @property
    def name(self) -> str:
        return self.scene_dir.name


@dataclass
class ObjectAnnotation:
    """One annotated object (polygon) in a scene."""

    raw_name: str
    name: str                                  # normalized name
    polygons: list[list[float]] = field(default_factory=list)  # [x1,y1,x2,y2,...]


def _first_file(directory: Path, patterns: Sequence[str]) -> Path | None:
    """Returns the first file in the directory matching one of the patterns."""
    if not directory.is_dir():
        return None
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    return None


def _infer_sensor_and_collection(rel_parts: Sequence[str]) -> tuple[str, str]:
    """Infers the sensor and sub-collection from a scene's relative path."""
    sensors = {"kv1", "kv2", "xtion", "realsense"}
    sensor, collection = "unknown", "unknown"
    for i, part in enumerate(rel_parts):
        if part in sensors:
            sensor = part
            if i + 1 < len(rel_parts):
                collection = rel_parts[i + 1]
            break
    if sensor == "unknown" and rel_parts:
        collection = rel_parts[0]
    return sensor, collection


def find_scenes(root: Path, require_annotation: bool = True) -> list[Scene]:
    """Recursively finds all SUN RGB-D scenes under `root`."""
    root = Path(root).expanduser().resolve()
    scenes: list[Scene] = []

    for image_dir in root.rglob("image"):
        if not image_dir.is_dir():
            continue
        scene_dir = image_dir.parent

        rgb = _first_file(image_dir, ["*.jpg", "*.png", "*.jpeg"])
        if rgb is None:
            continue

        depth = _first_file(scene_dir / "depth_bfx", ["*.png"])
        depth_source = "depth_bfx"
        if depth is None:
            depth = _first_file(scene_dir / "depth", ["*.png"])
            depth_source = "depth"
        if depth is None:
            continue

        intr = scene_dir / "intrinsics.txt"
        intr = intr if intr.is_file() else None

        ann = scene_dir / "annotation2Dfinal" / "index.json"
        if not ann.is_file():
            alt = scene_dir / "annotation2D" / "index.json"
            ann = alt if alt.is_file() else None
        if require_annotation and ann is None:
            continue

        rel = scene_dir.relative_to(root)
        sensor, collection = _infer_sensor_and_collection(rel.parts)

        scene_type = ""
        scene_txt = scene_dir / "scene.txt"
        if scene_txt.is_file():
            try:
                scene_type = scene_txt.read_text(errors="ignore").strip()
            except OSError:
                pass

        scenes.append(
            Scene(
                scene_dir=scene_dir,
                rgb_path=rgb,
                depth_path=depth,
                intrinsics_path=intr,
                annotation_path=ann,
                sensor=sensor,
                collection=collection,
                rel_key=rel.as_posix(),
                scene_type=scene_type,
            )
        )
        scenes[-1].depth_source = depth_source  

    scenes.sort(key=lambda s: s.rel_key)
    return scenes


def assign_groups(scenes: Iterable[Scene], flat_threshold: int = 30) -> None:
    """Assigns `group_key` for a leak-free train/val/test split."""
    scenes = list(scenes)
    parent_counts: dict[str, int] = {}
    for s in scenes:
        parent = s.scene_dir.parent.as_posix()
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    for s in scenes:
        parent = s.scene_dir.parent.as_posix()
        if parent_counts[parent] >= flat_threshold:
            s.group_key = s.rel_key           # independent frame
        else:
            s.group_key = parent              # frame sequence

def decode_depth(raw: np.ndarray, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Decodes a 16-bit SUN RGB-D depth map into meters.
    Returns a float32 array in meters; unreliable values (0 or > max) = 0.
    """
    d = np.asarray(raw).astype(np.uint16)
    high = (d >> 3).astype(np.uint16)
    low = ((d.astype(np.uint32) << 13) & 0xFFFF).astype(np.uint16)
    decoded = np.bitwise_or(high, low).astype(np.float32) / 1000.0
    decoded[~np.isfinite(decoded)] = 0.0
    decoded[decoded > max_depth_m] = 0.0
    decoded[decoded < 0.0] = 0.0
    return decoded


def read_depth(path: Path, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Reads and decodes a depth map into meters."""   
    with Image.open(path) as img:
        raw = np.array(img)
    if raw.ndim == 3:                      # 3-channel PNGs do occur
        raw = raw[..., 0]
    return decode_depth(raw, max_depth_m=max_depth_m)


def read_rgb(path: Path) -> np.ndarray:
    """Reads an RGB image as a uint8 array (H, W, 3)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def read_intrinsics(path: Path | None) -> Intrinsics | None:
    """Reads the camera intrinsics matrix from intrinsics.txt."""
    if path is None or not Path(path).is_file():
        return None
    try:
        values = [float(v) for v in Path(path).read_text().split()]
    except (OSError, ValueError):
        return None
    if len(values) < 9:
        return None
    k = np.array(values[:9], dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if not all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        return None
    return Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def backproject(
    depth_m: np.ndarray,
    intr: Intrinsics,
    mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-projection: (u, v, depth) -> 3D coordinates (X, Y, Z).

    Classic pinhole camera model:
        Z = depth(v, u)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Returns (points Nx3, colors Nx3 or None). Only pixels with reliable
    depth (> 0) and, if given, inside the mask are used.
    """
    h, w = depth_m.shape[:2]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]

    valid = z > 0
    if mask is not None:
        valid &= mask[::stride, ::stride].astype(bool)
    if not valid.any():
        return np.empty((0, 3), np.float32), (None if rgb is None else np.empty((0, 3), np.float32))

    z = z[valid].astype(np.float32)
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)

    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=1)

    colors = None
    if rgb is not None:
        colors = (rgb[::stride, ::stride][valid].astype(np.float32) / 255.0)
    return points, colors


def normalize_name(raw: str) -> str:
    """Normalizes a class name to its canonical form."""
    name = str(raw).strip().lower()
    name = name.replace("_", " ")
    if name and not name[0].isdigit():
        name = _TRAILING_NUM.sub("", name)
    name = _MULTISPACE.sub(" ", name).strip()
    name = SYNONYMS.get(name, name)
    if name not in _CANONICAL_NAMES:
        squashed = name.replace(" ", "").replace("-", "")
        canon = _SQUASHED_SYNONYMS.get(squashed)
        if canon is None and squashed.endswith("s"):
            canon = _SQUASHED_SYNONYMS.get(squashed[:-1])
        if canon is not None:
            name = canon
    return name


def _coerce_coords(value) -> list[float]:
    """Coordinates in index.json may be numbers, strings, or nested lists."""
    out: list[float] = []
    if value is None:
        return out
    if not isinstance(value, (list, tuple)):
        value = [value]
    for item in value:
        if isinstance(item, (list, tuple)):
            out.extend(_coerce_coords(item))
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def load_annotations(path: Path | None) -> list[ObjectAnnotation]:
    """Parses annotation2Dfinal/index.json into a list of polygon objects.

    Format (LabelMe-like):
        {"objects": [{"name": "chair"}, null, ...],
         "frames":  [{"polygon": [{"object": 0, "x": [...], "y": [...]}, ...]}]}

    Some files are corrupted or contain empty/deleted objects —
    all such cases are skipped without raising.
    """
    if path is None or not Path(path).is_file():
        return []
    try:
        data = json.loads(Path(path).read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    objects = data.get("objects") or []
    if not isinstance(objects, list):
        return []

    names: list[str | None] = []
    for obj in objects:
        if isinstance(obj, dict):
            names.append(obj.get("name"))
        else:
            names.append(None)

    collected: dict[int, ObjectAnnotation] = {}

    def _add(obj_idx: int, xs: list[float], ys: list[float]) -> None:
        if obj_idx < 0 or obj_idx >= len(names):
            return
        raw = names[obj_idx]
        if not raw:
            return
        n = min(len(xs), len(ys))
        if n < 3:                               
            return
        flat: list[float] = []
        for i in range(n):
            flat.extend((xs[i], ys[i]))
        entry = collected.get(obj_idx)
        if entry is None:
            entry = ObjectAnnotation(raw_name=str(raw), name=normalize_name(raw))
            collected[obj_idx] = entry
        entry.polygons.append(flat)

    frames = data.get("frames") or []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for poly in frame.get("polygon") or []:
                if not isinstance(poly, dict):
                    continue
                obj_idx = poly.get("object")
                if not isinstance(obj_idx, int):
                    continue
                _add(obj_idx, _coerce_coords(poly.get("x")), _coerce_coords(poly.get("y")))

    if not collected:
        for idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            poly = obj.get("polygon")
            if isinstance(poly, dict):
                _add(idx, _coerce_coords(poly.get("x")), _coerce_coords(poly.get("y")))
            elif isinstance(poly, list):
                for p in poly:
                    if isinstance(p, dict):
                        _add(idx, _coerce_coords(p.get("x")), _coerce_coords(p.get("y")))

    return [collected[k] for k in sorted(collected)]

def clip_polygon(poly: Sequence[float], width: int, height: int) -> list[float]:
    """Clips polygon coordinates to the image bounds."""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts.reshape(-1).tolist()


def polygons_to_mask(polygons: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Rasterizes a set of polygons into a single binary mask (uint8)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def _area_via_pycocotools(
    polygons: Sequence[Sequence[float]], width: int, height: int
) -> float | None:
    """Computes the area exactly the way COCOeval does (if pycocotools exists)."""
    try:
        rles = mask_utils.frPyObjects([list(p) for p in polygons], height, width)
        rle = mask_utils.merge(rles)
        return float(mask_utils.area(rle))
    except Exception:  
        return None


def polygons_bbox_area(
    polygons: Sequence[Sequence[float]], width: int, height: int
) -> tuple[list[float], float]:
    """Computes the COCO bbox [x, y, w, h] and the mask area in pixels."""
    all_pts = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            all_pts.append(pts)
    if not all_pts:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    stacked = np.vstack(all_pts)
    x0 = float(np.clip(stacked[:, 0].min(), 0, width - 1))
    y0 = float(np.clip(stacked[:, 1].min(), 0, height - 1))
    x1 = float(np.clip(stacked[:, 0].max(), 0, width - 1))
    y1 = float(np.clip(stacked[:, 1].max(), 0, height - 1))

    bw, bh = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
    if bw < 1 or bh < 1:
        return [x0, y0, bw, bh], 0.0

    area = _area_via_pycocotools(polygons, width, height)
    if area is None:
        canvas_w, canvas_h = int(np.ceil(bw)) + 1, int(np.ceil(bh)) + 1
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for pts in all_pts:
            shifted = pts - np.array([x0, y0])
            cv2.fillPoly(canvas, [np.round(shifted).astype(np.int32)], 1)
        area = float(canvas.sum())
    return [x0, y0, bw, bh], area

def iter_batched(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
