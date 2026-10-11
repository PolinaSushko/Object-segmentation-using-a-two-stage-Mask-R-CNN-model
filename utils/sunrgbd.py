"""
sunrgbd.py — core utilities for working with the SUN RGB-D dataset.

Covers:
  * scene discovery in the arbitrary SUN RGB-D directory layout;
  * leak-free scene grouping for the train/val/test split (assign_groups);
  * correct decoding of 16-bit depth maps (bit shift!);
  * reading camera intrinsics;
  * parsing 2D polygon annotations from annotation2Dfinal/index.json and
    clipping them to the image (clip_polygon);
  * class-name normalization;
  * back-projection of pixels into 3D (used in chapter 3 of the thesis).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
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

# --- Leak-free grouping (dataset v2, see assign_groups) ----------------------
# Collections made of video scans: every parent directory is one sequence of
# near-duplicate frames, never a flat collection.
SEQUENCE_COLLECTIONS = frozenset({"sun3ddata"})
# Collections without any capture metadata (sequential file names only):
# grouped by visual near-duplicate links between neighbouring frames.
NO_METADATA_COLLECTIONS = frozenset({"NYUdata", "b3dodata"})
# Frames of timestamped collections (kv2, xtion_align_data, realsense) less
# than this many seconds apart belong to one capture session (one room).
CAPTURE_GAP_S = 120
# Visual near-duplicate test (ORB + RANSAC fundamental-matrix inliers) for
# NO_METADATA_COLLECTIONS: frames at most VISUAL_WINDOW positions apart with
# >= VISUAL_MIN_INLIERS inliers show the same place (unrelated frames reach
# at most ~22 inliers).
VISUAL_WINDOW = 5
VISUAL_MIN_INLIERS = 40

# capture time (+ device serial) encoded in the scene directory names
_CAPTURE_PATTERNS = (
    # kv2: [000002_][long_]2014-05-26_14-23-37[-RobotInRoom]_260595134347[_rgbf...]
    re.compile(r"^(?:\d+_)?(?:long_)?(\d{4})-(\d\d)-(\d\d)_(\d\d)-(\d\d)-(\d\d)"
               r"(?:-[A-Za-z]+)?_(\d+)"),
    # realsense: 2014_10_24-09_14_34-1311000073
    re.compile(r"^(\d{4})_(\d\d)_(\d\d)-(\d\d)_(\d\d)_(\d\d)-(\d+)$"),
    # xtion_align_data: 2014_12_18_11_31_55 (single device, no serial)
    re.compile(r"^(\d{4})_(\d\d?)_(\d\d?)_(\d\d?)_(\d\d?)_(\d\d?)$"),
)

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

# Dataset v2: compound names and typos of the std10 / min400 classes that the
# rules above miss (reviewed one by one, decisions and reasons for every
# candidate name: notebooks/outputs/class_analysis/name_recovery_v2.csv).
# Only names whose meaning is unambiguously the target class are listed;
# desk-like "X table" (computer / study / office table), desk/table combos,
# dressing table, sofa set / sofa bed, stools, benches, ottomans, counters,
# object parts and groups are deliberately NOT mapped.
SYNONYMS.update({
    # chair (functional types; "sofa chair" spellings follow the sofa rule)
    "computer chair": "chair", "comptuer chair": "chair", "deschair": "chair",
    "dining chair": "chair", "dinning chair": "chair", "doining chair": "chair",
    "dinning room chair": "chair", "study chair": "chair",
    "student chair": "chair", "school chair": "chair", "lounge chair": "chair",
    "rocking chair": "chair", "recliner": "chair", "recliner chair": "chair",
    "reclining chair": "chair", "revolving chair": "chair",
    "revalving chair": "chair", "rolling chair": "chair",
    "rooling chair": "chair", "executive chair": "chair",
    "accent chair": "chair", "side chair": "chair", "captains chair": "chair",
    "papasan chair": "chair", "bowl chair": "chair", "gliding chair": "chair",
    "cushion chair": "chair", "cusion chair": "chair", "sponge chair": "chair",
    "kids chair": "chair", "background chair": "chair",
    "waiting room chair": "chair", "sitting chair": "chair",
    "bentwood chair": "chair", "cozy chair": "chair", "soft chair": "chair",
    "designer chair": "chair", "counter chair": "chair",
    # sofa
    "loveseat": "sofa", "love seat": "sofa", "futon": "sofa",
    "sectional": "sofa", "sectional sofa": "sofa", "sectional couch": "sofa",
    "single sofa chair": "sofa", "sofa chair single": "sofa",
    "shopa chair": "sofa", "shoba chair": "sofa", "soda chair": "sofa",
    "single seater sofa": "sofa", "sofa single seater": "sofa",
    "singla sofa": "sofa", "3 seater sofa": "sofa", "sofa 3 sitter": "sofa",
    "couch sofa": "sofa", "sofa couch": "sofa", "loveseat/sofa": "sofa",
    "waiting couch": "sofa", "sitting couch": "sofa", "cushion sofa": "sofa",
    # table
    "round table": "table", "tabel": "table", "tabels": "table",
    "tabla": "table", "tablr": "table", "conference table": "table",
    "discussion table": "table", "tea table": "table",
    "dinning table": "table", "dinner table": "table", "cofee table": "table",
    "coffe table": "table", "coffie table": "table", "coffee tabel": "table",
    "kichen table": "table", "banquet table": "table", "buffet table": "table",
    "cocktail table": "table", "pedestal table": "table",
    "serving table": "table", "restaurant table": "table",
    "lamp table": "table", "center table": "table", "centre table": "table",
    "glasstop table": "table", "glass tabel": "table", "lab table": "table",
    "work table": "table", "working table": "table",
    # desk
    "school desk": "desk", "student desk": "desk", "work desk": "desk",
    "cubicle desk": "desk", "teacher desk": "desk", "teacher's desk": "desk",
    "techer desk": "desk", "reception desk": "desk", "editor's desk": "desk",
    "combuter desk": "desk", "professor computer desk": "desk",
    # bed
    "daybed": "bed", "loft bed": "bed", "toddler bed": "bed",
    "four poster bed": "bed", "princess bed": "bed", "full bed": "bed",
    "unmade bed": "bed",
    # toilet / bathtub
    "western toilet": "toilet",
    "bathdub": "bathtub", "bathtab": "bathtub", "bathtun": "bathtub",
    "bothtub": "bathtub", "bathing tub": "bathtub", "jetted tub": "bathtub",
    "shower tub": "bathtub",
    # dresser / nightstand / bookshelf
    "dresser with mirror": "dresser", "dresser with morror": "dresser",
    "dresser with mirroe": "dresser", "ornate dresser with mirror": "dresser",
    "mirrored dresser": "dresser", "chester drawers": "dresser",
    "vanity chest of drawers": "dresser", "7-drawer dresser": "dresser",
    "hallway dresser": "dresser",
    "nightsyand": "nightstand", "nightstan": "nightstand",
    "nightsatnd": "nightstand", "bedside stand": "nightstand",
    "bedside chest": "nightstand",
    "bookself": "bookshelf", "bookshelve": "bookshelf",
    "bookshelfe": "bookshelf", "bookshelp": "bookshelf",
    "bookshelf containing books": "bookshelf", "books shelf": "bookshelf",
    # min400 classes (canonical names before SEMANTIC_MERGES)
    "storage cabinet": "cabinet", "display cabinet": "cabinet",
    "medicine cabinet": "cabinet", "medine cabinet": "cabinet",
    "china cabinet": "cabinet", "corner china cabinet": "cabinet",
    "curio cabinet": "cabinet", "overhead cabinet": "cabinet",
    "over-head cabinets": "cabinet", "trophy cabinet": "cabinet",
    "bar cabinet": "cabinet", "wine cabinet": "cabinet",
    "entertainment cabinet": "cabinet", "tv cabinet": "cabinet",
    "cabinent": "cabinet", "cabint": "cabinet", "cabinates": "cabinet",
    "ccabinet": "cabinet", "cupboards": "cupboard", "ccupboard": "cupboard",
    "filecabnet": "file cabinet", "filecabinent": "file cabinet",
    "fliecabinet": "file cabinet", "fillingcabinet": "file cabinet",
    "filingcabnet": "file cabinet", "filingcabnets": "file cabinet",
    "filecabnets": "file cabinet", "filescabinets": "file cabinet",
    "large file cabinet": "file cabinet", "filing cabinet": "file cabinet",
    "throw pillow": "pillow", "decorative pillow": "pillow",
    "decor pillow": "pillow", "sofa pillow": "pillow", "couch pillow": "pillow",
    "bed pillow": "pillow", "neck pillow": "pillow", "neckroll pillow": "pillow",
    "cylinder pillow": "pillow", "cylindrical pillow": "pillow",
    "light gray throw pillow": "pillow", "polkadotted pillow": "pillow",
    "decorative square bed pillow": "pillow",
    "decorative round bed pillow": "pillow", "pillowes": "pillow",
    "roiund pillow": "pillow",
    "computer screen": "monitor", "desktop screen": "monitor",
    "back computer screen": "monitor", "backleft computer screen": "monitor",
    "pc monitor": "monitor", "desktop monitor": "monitor",
    "apple monitor": "monitor", "flatscreen monitor": "monitor",
    "system monitor": "monitor", "system display monitor": "monitor",
    "ststem displat monitor": "monitor", "computer moniter": "monitor",
    "computter monitor": "monitor", "computers monitor": "monitor",
    "compuer monitors": "monitor", "comptuer monitor": "monitor",
    "monitor screen": "monitor", "lcd display": "monitor",
    "tv monitor": "television", "led monitor tv": "television",
    "flatscreen tv": "television", "wall-mounted tv": "television",
    "wallmount tv": "television", "led tv": "television",
    "lcd tv": "television", "lcd television": "television",
    "hdtv": "television", "widescreen tv": "television",
    "shelving unit": "shelf", "shelf unit": "shelf", "lack shelf": "shelf",
    "lack wall shelf": "shelf", "storage shelves": "shelf",
    "metal storage shelf": "shelf", "3-tier shelf": "shelf",
    "hanging shelf": "shelf", "office shelf": "shelf",
    "shelve": "shelf", "shellf": "shelf", "shelfes": "shelf",
    "floor lamp": "lamp", "night lamp": "lamp", "bed lamp": "lamp",
    "bedside lamp": "lamp", "standing lamp": "lamp", "pole lamp": "lamp",
    "office lamp": "lamp", "buffet lamp": "lamp", "wall lamp": "lamp",
    "wall mounted lamp": "lamp", "atble lamp": "lamp", "bed lamb": "lamp",
    "light lamp": "lamp", "table light": "lamp",
    "recycle bin": "garbage bin", "recycling bin": "garbage bin",
    "recyle bin": "garbage bin", "recyling bin": "garbage bin",
    "recylcing bin": "garbage bin", "receycle bin": "garbage bin",
    "rcycle bin": "garbage bin", "paper recycle bin": "garbage bin",
    "recycle trashcan": "garbage bin", "recyclable trashcan": "garbage bin",
    "recycable trashbin": "garbage bin", "garbage and recycling bin": "garbage bin",
    "waste bin": "garbage bin", "wast bin": "garbage bin",
    "waist bin": "garbage bin", "wastepaper bin": "garbage bin",
    "waste can": "garbage bin", "waste bascket": "garbage bin",
    "wast basket": "garbage bin", "waaste basket": "garbage bin",
    "waste pasket": "garbage bin", "trash basket": "garbage bin",
    "rubbish bin": "garbage bin", "refuse bin": "garbage bin",
    "trash receptacle": "garbage bin", "waste receptacle": "garbage bin",
    "waste recepticle": "garbage bin", "trash receptical": "garbage bin",
    "garbeg bin": "garbage bin", "garbage bean": "garbage bin",
    "dustbin": "garbage bin", "dust bin": "garbage bin",
    "dusbin": "garbage bin", "dustbeen": "garbage bin",
    "office trashcan": "garbage bin",
    "computer keyboard": "keyboard", "conmputer keyboard": "keyboard",
    "desktop keyboard": "keyboard", "keybord": "keyboard",
    "keyborad": "keyboard", "keyboad": "keyboard", "keybored": "keyboard",
    "keybooard": "keyboard", "keyboaed": "keyboard",
    "computer tower": "computer", "pc tower": "computer",
    "cpu tower": "computer", "tower computer": "computer",
    "desktop computer": "computer", "desktop computer tower": "computer",
    "desktop pc": "computer", "personal computer": "computer",
    "pc computer": "computer", "computer cpu": "computer",
    "apple computer": "computer", "trird computer": "computer",
    "sink basin": "sink", "utility sink": "sink", "washing sink": "sink",
    "water sink": "sink", "kitchen sink": "sink", "bathroom sink": "sink",
    "washbasin": "sink", "wash basin": "sink", "washing basin": "sink",
    "washbasine": "sink", "wsashbasin": "sink", "hand basin": "sink",
    "potted plant": "plant", "house plant": "plant", "indoor plant": "plant",
    "decorative plant": "plant", "ornamental plant": "plant",
    "money plant": "plant", "fake plant": "plant", "artificial plant": "plant",
    "flower plant": "plant", "plants": "plant", "plant with pot": "plant",
    "plant in planter": "plant", "plant in flowerpot": "plant",
})

_SQUASHED_SYNONYMS: dict[str, str] = {}
for _alias, _canon in SYNONYMS.items():
    _SQUASHED_SYNONYMS.setdefault(_alias.strip().replace(" ", "").replace("-", ""), _canon)
for _canon in set(SYNONYMS.values()):
    _SQUASHED_SYNONYMS.setdefault(_canon.replace(" ", "").replace("-", ""), _canon)

_CANONICAL_NAMES = frozenset(SYNONYMS.values())

# Dataset v2: "<modifier(s)><class>" names (white chair, smallcoffeetable,
# 2ndchair, woodenroundchair, ...) are the class itself. Applied only to the
# std10 / min400 classes and only for purely descriptive modifiers (colour,
# material, size, shape, position, ordinal, article); heads that are
# ambiguous on their own (bin, tub, screen, pc, cpu) are never completed.
_MODIFIABLE_CLASSES = frozenset(STANDARD_10_CLASSES) | {
    "cabinet", "cupboard", "file cabinet", "pillow", "monitor", "television",
    "shelf", "lamp", "garbage bin", "keyboard", "computer", "sink", "plant",
}
_AMBIGUOUS_HEADS = frozenset({"bin", "tub", "screen", "pc", "cpu"})
# single boards of a shelving unit (parts), not shelf furniture
_MODIFIER_EXCEPTIONS = frozenset({
    "uppershelf", "uppershelve", "lowershelf", "middleshelf", "middlerightshelf",
    "lowerrightshelf",
})
_MODIFIABLE_HEADS = {
    alias: canon for alias, canon in _SQUASHED_SYNONYMS.items()
    if canon in _MODIFIABLE_CLASSES and alias not in _AMBIGUOUS_HEADS
}
_MODIFIERS = sorted("""
    white black brown grey gray red blue green yellow orange purple pink beige
    navy teal silver gold golden dark lightgray lightgrey mint charcoal cream
    colour color
    wood wooden wodden metal metallic steel iron plastic leather glass granite
    marble wicker rattan bamboo
    small smaller big bigger large larger long tall short mini low little huge
    medium tiny round rounded square rectangular rectangle circular circle oval
    left right middle center centre front far upper lower corner side wall
    first second third fourth fifth sixth seventh 1st 2nd 3rd 4th 5th 6th 7th
    a an another other extra single double twin
    folding foldable rolling revolving swivel reclining rocking padded
    kitchen dining dinning bathroom bedroom
""".split(), key=len, reverse=True)
_MODIFIER_PREFIX = re.compile("|".join(_MODIFIERS))
_EDGE_PUNCT = " .,;:'\"`|\\[]{}"

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


def parse_capture_time(rel_key: str) -> tuple[datetime, str] | None:
    """(capture time, device) encoded in a scene directory name, or None.

    kv2 (align_kv2, kinect2data) and realsense names carry the device serial;
    xtion_align_data has a single device ("<sensor>/<collection>" is used).
    """
    parts = rel_key.split("/")
    for pattern in _CAPTURE_PATTERNS:
        m = pattern.match(parts[-1])
        if m:
            g = m.groups()
            try:
                t = datetime(*map(int, g[:6]))
            except ValueError:
                return None
            device = g[6] if len(g) > 6 else "/".join(parts[:2])
            return t, device
    return None


def sun3d_site(rel_key: str) -> str | None:
    """Physical site of a SUN3D frame: the location directory with a trailing
    scan index removed (brown_cs_4 -> brown_cs, harvard_tea_2 -> harvard_tea),
    because repeated scans of one building are stored as separate locations
    (visually confirmed matches between brown_cs_1/2, 4/5, 8/9 and
    brown_cogsci_2/7). None for frames of other collections."""
    parts = rel_key.split("/")
    if len(parts) < 4 or parts[1] not in SEQUENCE_COLLECTIONS:
        return None
    return re.sub(r"_\d+$", "", parts[2])


def _orb_features(path: Path, max_side: int = 640):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    scale = max_side / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    kp, des = cv2.ORB_create(nfeatures=1500).detectAndCompute(img, None)
    if des is None or len(kp) < 10:
        return None
    return np.float32([k.pt for k in kp]), des


def _ransac_inliers(fa, fb) -> int:
    """Number of feature matches consistent with one epipolar geometry."""
    if fa is None or fb is None:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(fa[1], fb[1], k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return 0
    pa = np.float32([fa[0][m.queryIdx] for m in good])
    pb = np.float32([fb[0][m.trainIdx] for m in good])
    # plain RANSAC through the USAC API: its explicit random state makes every
    # call reproducible (FM_RANSAC keeps hidden state between calls)
    params = cv2.UsacParams()
    params.threshold, params.confidence, params.maxIterations = 2.0, 0.999, 5000
    params.randomGeneratorState, params.isParallel = 0, False
    params.sampler, params.score = cv2.SAMPLING_UNIFORM, cv2.SCORE_METHOD_RANSAC
    params.loMethod = cv2.LOCAL_OPTIM_NULL
    try:
        _, inl = cv2.findFundamentalMat(pa, pb, params)
    except cv2.error:
        return 0
    return 0 if inl is None else int(inl.sum())


def visual_links(frames: Iterable[tuple[str, Path]], window: int = VISUAL_WINDOW,
                 min_inliers: int = VISUAL_MIN_INLIERS,
                 workers: int | None = None) -> list[tuple[str, str]]:
    """Near-duplicate frame pairs (rel_key, rel_key), e.g. two views of one NYU room.

    `frames` are (rel_key, RGB path); frames of the same parent directory
    whose frame numbers (NYU0001, img_0063, ...) differ by at most `window`
    are compared — by number, not list position, so the compared pairs do
    not depend on which frames are present. Deterministic: fixed ORB / RANSAC
    settings and seed, OpenCV threading off.
    """
    from concurrent.futures import ThreadPoolExecutor

    by_parent: dict[str, list[tuple[int, str]]] = {}
    paths: dict[str, Path] = {}
    for pos, (key, path) in enumerate(sorted(frames)):
        digits = re.findall(r"\d+", key.rsplit("/", 1)[-1])
        by_parent.setdefault(key.rsplit("/", 1)[0], []).append(
            (int(digits[-1]) if digits else pos, key))
        paths[key] = Path(path)
    pairs = []
    for seq in by_parent.values():
        seq.sort()
        for i, (num, key) in enumerate(seq):
            for num2, key2 in seq[i + 1:]:
                if num2 - num > window:
                    break
                pairs.append((key, key2))
    # OpenCV's own threading is switched off (ORB keypoint order could vary
    # with it); images are processed in parallel Python threads instead.
    n_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(workers) as pool:
            feats = dict(zip(paths, pool.map(_orb_features, paths.values())))
        return [(a, b) for a, b in pairs
                if _ransac_inliers(feats[a], feats[b]) >= min_inliers]
    finally:
        cv2.setNumThreads(n_threads)


def assign_groups(scenes: Iterable[Scene], flat_threshold: int = 30,
                  capture_gap_s: float = CAPTURE_GAP_S,
                  visual: bool = True) -> None:
    """Assigns `group_key` for a leak-free train/val/test split (dataset v2).

    Frames that may show the same physical place are put into one group
    (union of all rules below), so no place can occur in two subsets:

    (a) SUN3D (SEQUENCE_COLLECTIONS): every frame of one site — all scans of
        the same apartment / lab / building (`sun3d_site`), not only one scan;
    (b)+(c)+(d) timestamped collections (kv2 align_kv2 / kinect2data,
        xtion_align_data, realsense lg / sa / sh / shr): frames sorted by the
        capture time in their directory name (`parse_capture_time`) form ONE
        stream across devices, split into capture sessions wherever two
        consecutive frames are more than `capture_gap_s` apart. This keeps
        together frame sequences of one room taken by the same device
        (typical cadence 10-60 s) and the simultaneous multi-sensor captures
        of 2014-12-18 (kv2 / xtion / realsense in the same rooms, clocks
        within seconds to a few minutes);
    (e) NO_METADATA_COLLECTIONS (kv1 NYUdata / b3dodata — sequential ids
        only): visual near-duplicates among neighbouring frames
        (`visual_links`; skipped with visual=False);
    (f) any other layout (fallback, pre-v2 rule): a parent directory with
        >= `flat_threshold` scenes is a flat collection of independent
        frames, a smaller one is a frame sequence.

    Group keys are relative to the dataset root: "<sensor>/sun3ddata/<site>",
    "capture/<session start>", or the rel_key of the first frame.
    """
    scenes = sorted(scenes, key=lambda s: s.rel_key)
    parent: dict[str, str] = {s.rel_key: s.rel_key for s in scenes}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # root = smallest rel_key

    names: dict[str, str] = {}                  # root -> readable group key
    sites: dict[str, str] = {}
    timed: list[tuple[datetime, str]] = []
    fallback: list[Scene] = []
    for s in scenes:
        site = sun3d_site(s.rel_key)
        capture = parse_capture_time(s.rel_key)
        if site is not None:                                        # (a)
            key = f"{s.sensor}/{s.collection}/{site}"
            union(sites.setdefault(key, s.rel_key), s.rel_key)
            names[sites[key]] = key
        elif capture is not None:                                   # (b-d)
            timed.append((capture[0], s.rel_key))
        elif s.collection not in NO_METADATA_COLLECTIONS:           # (f)
            fallback.append(s)

    timed.sort()
    start = None
    for i, (t, key) in enumerate(timed):
        if i == 0 or (t - timed[i - 1][0]).total_seconds() > capture_gap_s:
            start = key
            names[key] = f"capture/{t:%Y-%m-%d_%H-%M-%S}"
        union(start, key)

    if visual:                                                      # (e)
        for a, b in visual_links([(s.rel_key, s.rgb_path) for s in scenes
                                  if s.collection in NO_METADATA_COLLECTIONS]):
            union(a, b)

    parent_counts: dict[str, int] = {}                              # (f)
    for s in fallback:
        p = s.scene_dir.parent.as_posix()
        parent_counts[p] = parent_counts.get(p, 0) + 1
    first_in: dict[str, str] = {}
    for s in fallback:                          # scenes are sorted by rel_key
        p = s.scene_dir.parent.as_posix()
        if parent_counts[p] < flat_threshold:
            union(first_in.setdefault(p, s.rel_key), s.rel_key)

    group_name: dict[str, str] = {}
    for key, name in sorted(names.items(), key=lambda kv: kv[1]):
        group_name.setdefault(find(key), name)
    for s in scenes:
        root = find(s.rel_key)
        s.group_key = group_name.get(root, root)

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
    """Normalizes a class name to its canonical form.

    Lower case, edge punctuation and trailing indices removed, then SYNONYMS
    (also matched without spaces / hyphens and without a plural "s"), then
    the descriptive-modifier rule for the std10 / min400 classes (v2).
    """
    name = str(raw).strip().lower()
    name = name.replace("_", " ").strip(_EDGE_PUNCT)      # "chair." / "|keyboard"
    if name and not name[0].isdigit():
        name = _TRAILING_NUM.sub("", name)
    name = _MULTISPACE.sub(" ", name).strip()
    name = SYNONYMS.get(name, name)
    if name not in _CANONICAL_NAMES:
        squashed = name.replace(" ", "").replace("-", "")
        canon = _SQUASHED_SYNONYMS.get(squashed)
        if canon is None and squashed.endswith("s"):
            canon = _SQUASHED_SYNONYMS.get(squashed[:-1])
        # "<descriptive modifiers><class>" (v2): strip modifiers one by one
        rest = squashed
        while canon is None and squashed not in _MODIFIER_EXCEPTIONS:
            m = _MODIFIER_PREFIX.match(rest)
            if m is None or m.end() == len(rest):
                break
            rest = rest[m.end():]
            canon = _MODIFIABLE_HEADS.get(rest)
            if canon is None and rest.endswith("s"):
                canon = _MODIFIABLE_HEADS.get(rest[:-1])
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
    """Clips a polygon to the image rectangle [0, width] x [0, height].

    Sutherland–Hodgman clipping against the four image borders: the result is
    the part of the polygon inside the image (a polygon again, so the COCO /
    YOLO formats are unchanged). Clamping each vertex independently — the
    pre-v2 behaviour — is NOT clipping: it moves vertices along the border
    and can cut off (or add) large parts of objects that leave the frame.

    Returns [] when nothing of the polygon remains (fewer than 3 vertices or
    zero area); polygons fully inside the image are returned unchanged.
    """
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return []
    inside_img = ((pts[:, 0] >= 0) & (pts[:, 0] <= width)
                  & (pts[:, 1] >= 0) & (pts[:, 1] <= height))
    if not inside_img.all():
        # (axis, bound, keep the side <= bound?) for left, right, top, bottom
        for axis, bound, keep_le in ((0, 0.0, False), (0, float(width), True),
                                     (1, 0.0, False), (1, float(height), True)):
            if len(pts) == 0:
                break
            ins = pts[:, axis] <= bound if keep_le else pts[:, axis] >= bound
            out = []
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                if ins[i]:
                    out.append(pts[i])
                if ins[i] != ins[j]:              # the edge crosses the border
                    t = (bound - pts[i, axis]) / (pts[j, axis] - pts[i, axis])
                    p = pts[i] + t * (pts[j] - pts[i])
                    p[axis] = bound
                    out.append(p)
            pts = np.asarray(out, dtype=np.float64).reshape(-1, 2)
        if len(pts):                              # drop repeated vertices
            keep = np.any(np.abs(pts - np.roll(pts, 1, axis=0)) > 1e-9, axis=1)
            pts = pts[keep] if keep.any() else pts[:1]
    if len(pts) < 3:
        return []
    x, y = pts[:, 0], pts[:, 1]
    if abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 1e-6:
        return []                                 # zero-area (degenerate) polygon
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
    x0 = float(np.clip(stacked[:, 0].min(), 0, width))
    y0 = float(np.clip(stacked[:, 1].min(), 0, height))
    x1 = float(np.clip(stacked[:, 0].max(), 0, width))
    y1 = float(np.clip(stacked[:, 1].max(), 0, height))

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
