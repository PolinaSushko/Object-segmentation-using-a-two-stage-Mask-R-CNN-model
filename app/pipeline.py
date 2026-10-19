"""Inference pipeline of the web demo.

RGB image -> Mask R-CNN instance masks -> monocular depth (Depth Anything V2,
no depth sensor at inference time) -> per-object point clouds (mask refinement,
voxel + SOR cleaning — `utils/pointcloud.py`, shared with notebooks 05/07/08)
-> 3D characteristics.

Depth backends (env `DEPTH_BACKEND`; notebook 06 fits them, notebook 08
evaluates them end to end with predicted masks):
  * "metric_scaled" (default): metric indoor model (metres out of the box) +
    one global scale factor fitted on validation — the FOV mismatch between
    its training renders and real Kinect-class cameras leaves a constant bias;
  * "metric_focal": the same metric model, bias corrected from the camera
    intrinsics: depth = raw * (fx / long_side) / F0, where F0 is the focal
    length (per unit image long side) the model implicitly assumes;
  * "relative_global": relative model (disparity output) + the global affine
    alignment in disparity space fitted on validation, depth = 1/(s*raw + t).

Resolution: the mask-refinement constants of `utils/pointcloud.py` are in
pixels (boundary erosion, discontinuity window) and were tuned on SUN RGB-D
frames (long side 561–730 px). Inputs outside that range are resized to a
640 px long side before *every* stage, and the intrinsics are rescaled with
them; otherwise large photos re-introduce the depth "curtains" the filter
removes (e.g. a chair's OBB volume x22 at 4x resolution).

Segmentation model: Mask R-CNN ResNet-50-FPN v2 (torchvision) trained on SUN
RGB-D dataset v2, 20 min400 classes (run `v2_maskrcnn_torchvision_min400`,
notebook 02): val / test mask AP 0.212 / 0.226 on the 20 classes; on the 10
standard classes test mask AP 0.266. The thesis selects, by validation mask AP
on the 10 standard classes, the Detectron2 R101-FPN model trained on the same
20 classes (val 0.257, test 0.267); on the 10-class test set the two are equal
(difference 0.0002, paired bootstrap p = 0.98; notebook 10), and the
torchvision model keeps the demo pure PyTorch (no Detectron2 build on CPU
Spaces) — see README.md.

Weights: the segmentation model is loaded from a weights-only file written by
`export_weights.py` (state_dict + class names + meta, `torch.load(...,
weights_only=True)`), found via env MASKRCNN_CKPT (local path), env
MASKRCNN_HF_REPO (+ MASKRCNN_HF_FILENAME, Hugging Face Hub) or
`weights/maskrcnn_min400.pt` next to this file — see `resolve_weights()`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

APP_DIR = Path(__file__).resolve().parent


def _add_utils_to_path() -> Path:
    """Makes `utils/` importable in both supported layouts:
    the repository (app/ next to utils/) and a Hugging Face Space whose root
    is the content of app/ plus a copy of utils/."""
    for root in (APP_DIR, APP_DIR.parent):
        if (root / "utils" / "pointcloud.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise ImportError(
        f"utils/pointcloud.py not found next to {APP_DIR} or in its parent. "
        "In a Space, copy the repository's utils/ folder into the Space root.")


PROJECT_ROOT = _add_utils_to_path()

import utils.pointcloud as pc  # noqa: E402
import utils.sunrgbd as sd     # noqa: E402

# segmentation model

# fallback when a weights file carries no class names: category_id order of the
# std10 COCO files (index 0 is the background class)
STD10_FALLBACK = ["bed", "chair", "sofa", "table", "desk",
                  "toilet", "dresser", "nightstand", "bookshelf", "bathtub"]

# builders of the supported architectures (meta["arch"] of the weights file);
# head replacement as in training notebook 02
SEGMENTATION_ARCHS = {
    "maskrcnn_resnet50_fpn_v2": maskrcnn_resnet50_fpn_v2,
    "maskrcnn_resnet50_fpn": maskrcnn_resnet50_fpn,
}
DEFAULT_ARCH = "maskrcnn_resnet50_fpn_v2"
# shown in the UI header (the weights file's meta records source run / val AP)
MODEL_DESCRIPTION = ("Mask R-CNN ResNet-50-FPN v2 (torchvision), SUN RGB-D dataset v2, "
                     "20 classes; test mask AP 0.226 (20 classes), 0.266 on the 10 "
                     "standard classes")

WEIGHTS_FORMAT = "vkr-maskrcnn-weights/1"
DEFAULT_WEIGHTS_FILE = APP_DIR / "weights" / "maskrcnn_min400.pt"
DEFAULT_HF_FILENAME = "maskrcnn_min400.pt"

# duplicate suppression across classes (torchvision NMS is per class, so the
# same object may come out as e.g. "table" and "desk", or "shelf" and
# "bookshelf"). On the min400 train+val ground truth of dataset v2 only 84 of
# 165,776 pairs of objects in the same image overlap with mask IoU >= 0.7
# across classes (46 of them pillow/sofa; std10: 5 of 97,833) and 51 within a
# class, so such pairs among the predictions are near-certainly duplicates;
# same-class fragments lying >= 90% inside a higher-scoring mask of the same
# class are dropped too (128 GT pairs, 0.08%, mostly nested chair / cabinet /
# shelf annotations).
DUP_MASK_IOU = 0.7
DUP_SAME_CLASS_CONTAINMENT = 0.9

# depth backends (constants fitted on 200 sensor-stratified validation images,
# notebook 06, dataset v2; test AbsRel / delta1: relative_global 0.274 / 0.52,
# metric_scaled 0.233 / 0.77, metric_focal 0.259 / 0.68). `thesis_3d` is the
# end-to-end result with predicted masks over all 20 classes the demo shows
# (notebook 08, dataset v2, min400 test split, 3586 matched objects of the
# selected Mask R-CNN; medians vs GT mask + sensor depth; on the 10 standard
# classes alone: 15.2 / 20.4 / 25.0 cm).
DEPTH_BACKENDS = {
    "metric_scaled": {
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        # depth = scale * raw_depth
        "scale": 0.8573,
        "label": "Depth Anything V2 Metric-Indoor Small + global scale",
        "thesis_3d": "median Chamfer 15.8 cm (18.8 cm on the 10 non-standard "
                     "classes), OBB volume bias −14%, median |volume error| 49%",
    },
    "metric_focal": {
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        # depth = (fx / long_side) / F0 * raw_depth
        "F0": 1.0278,
        "label": "Depth Anything V2 Metric-Indoor Small + focal-aware scale",
        "thesis_3d": "median Chamfer 21.7 cm (27.0 cm on the 10 non-standard "
                     "classes), OBB volume bias −30%, median |volume error| 59%",
    },
    "relative_global": {
        "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        # depth = 1 / (s * raw_disparity + t)
        "s": 0.0894, "t": 0.2743,
        "label": "Depth Anything V2 Small (relative) + global disparity alignment",
        "thesis_3d": "median Chamfer 28.0 cm (37.4 cm on the 10 non-standard "
                     "classes), OBB volume bias −63%, median |volume error| 71%",
    },
}
# default = the best sensor-free backend of notebooks 06/08 at the same latency as
# the relative model (metric Base is better still, but 2.4x slower)
DEPTH_BACKEND = os.environ.get("DEPTH_BACKEND", "metric_scaled")

# monocular 3D is usable up to ~3.5 m (notebook 08, dataset v2, default backend,
# all 20 classes: median Chamfer 13 cm below 2.5 m, 20 cm at 2.5-3.5 m, 28-49 cm
# beyond); farther objects are flagged
RELIABLE_RANGE_M = 3.5

MAX_DEPTH_M = pc.MAX_DEPTH_M
MIN_DEPTH_M = pc.MIN_DEPTH_M
MIN_POINTS = pc.MIN_POINTS

# processing resolution (see the module docstring): SUN RGB-D frames have a
# long side of 561 (kv1) .. 730 (kv2) px and are processed natively; anything
# else is resized to the 640 px long side of the 640x480 sensor-class frames
NATIVE_LONG_SIDE = (561, 730)
PROC_LONG_SIDE = 640

# default intrinsics when nothing better is known: fx = fy = 0.925 * long side
# (median fx / width of the SUN RGB-D test split, ~57 deg horizontal FOV of a
# Kinect-class camera), principal point at the image centre
DEFAULT_FX_PER_LONG_SIDE = 0.925
FILM_35MM_DIAGONAL_MM = math.hypot(36.0, 24.0)   # 43.27 mm


# small helpers

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_cpus() -> int:
    """CPUs this process may actually use: the cgroup quota (containers,
    e.g. a 2-vCPU Space whose host reports dozens of cores), then the
    affinity mask, then os.cpu_count()."""
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def configure_cpu_threads(device: torch.device) -> None:
    """On CPU, one intra-op thread per usable CPU (env TORCH_NUM_THREADS
    overrides): torch's default counts the host's cores, which oversubscribes
    a quota-limited container."""
    if device.type != "cpu":
        return
    n = int(os.environ.get("TORCH_NUM_THREADS", 0)) or available_cpus()
    torch.set_num_threads(n)


def sha1_of_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# camera intrinsics

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def default_for(cls, width: int, height: int) -> "Intrinsics":
        """Kinect-like assumption (see DEFAULT_FX_PER_LONG_SIDE); uses the long
        side, so portrait photos get the same field of view as landscape."""
        f = DEFAULT_FX_PER_LONG_SIDE * max(width, height)
        return cls(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0)

    @classmethod
    def from_35mm_equivalent(cls, f35_mm: float, width: int, height: int) -> "Intrinsics":
        """EXIF FocalLengthIn35mmFilm -> pixels. The 35 mm equivalent is
        defined by matching the angle of view of a 36x24 mm frame along the
        diagonal (the convention phone and camera makers use for any aspect
        ratio), so f_px / diagonal_px = f35 / 43.27 mm. For a 3:2 image this
        equals f35 / 36 mm * long side; for 4:3 it is 4% longer."""
        f = f35_mm * math.hypot(width, height) / FILM_35MM_DIAGONAL_MM
        return cls(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0)

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        """Intrinsics of the image resized by (sx, sy); pixel centres map as
        u' + 0.5 = sx * (u + 0.5)."""
        return Intrinsics(fx=self.fx * sx, fy=self.fy * sy,
                          cx=(self.cx + 0.5) * sx - 0.5,
                          cy=(self.cy + 0.5) * sy - 0.5)

    def to_sd(self) -> sd.Intrinsics:
        return sd.Intrinsics(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


class ExampleIntrinsics:
    """Sensor intrinsics of the bundled SUN RGB-D examples, matched by the
    SHA-1 of the file bytes (not by file name): an unrelated upload named
    like an example does not inherit them, while Gradio's copies of the
    example files (byte-identical) still match."""

    def __init__(self, examples_dir: Path):
        self.by_hash: dict[str, tuple[str, Intrinsics]] = {}
        table_path = examples_dir / "intrinsics.json"
        if not table_path.is_file():
            return
        for name, k in json.loads(table_path.read_text()).items():
            path = examples_dir / name
            if path.is_file():
                self.by_hash[sha1_of_file(path)] = (name, Intrinsics(**k))

    def lookup(self, path: str | Path) -> tuple[str, Intrinsics] | None:
        return self.by_hash.get(sha1_of_file(path))


# image loading

@dataclass
class LoadedImage:
    rgb: np.ndarray                 # uint8 [H, W, 3], orientation corrected
    size: tuple[int, int]           # (W, H) of the full-resolution image
    f35_mm: float | None            # EXIF FocalLengthIn35mmFilm, if plausible
    notes: list[str] = field(default_factory=list)


def processing_size(width: int, height: int) -> tuple[int, int]:
    """(W, H) the pipeline works at — native for SUN RGB-D-like frames,
    otherwise a 640 px long side with the aspect ratio kept."""
    long_side = max(width, height)
    if NATIVE_LONG_SIDE[0] <= long_side <= NATIVE_LONG_SIDE[1]:
        return width, height
    s = PROC_LONG_SIDE / long_side
    return max(1, round(width * s)), max(1, round(height * s))


def _to_rgb(im: Image.Image) -> Image.Image:
    """Any PIL mode -> RGB. Transparency is composited over white; 16/32-bit
    and float single-channel images are min-max stretched to 8 bit."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.nanmin(a), np.nanmax(a)
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
        im = Image.fromarray(np.nan_to_num(a).astype(np.uint8), "L")
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return im.convert("RGB")


def load_image(path: str | Path, target_long_side: int | None = None) -> LoadedImage:
    """Reads an image file: EXIF orientation applied, converted to RGB,
    EXIF 35 mm-equivalent focal length extracted. Large JPEGs are decoded
    directly at a reduced DCT scale when `target_long_side` allows it."""
    notes = []
    with Image.open(path) as im:
        im.seek(0)                                   # first frame of animations
        f35 = None
        try:
            exif = im.getexif()
            f35 = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.FocalLengthIn35mmFilm)
            orientation = exif.get(ExifTags.Base.Orientation, 1)
        except Exception:                            # malformed EXIF
            exif, orientation = None, 1
        f35 = float(f35) if f35 and 8 <= float(f35) <= 1200 else None
        stored_w, stored_h = im.size
        swap = orientation in (5, 6, 7, 8)
        size = (stored_h, stored_w) if swap else (stored_w, stored_h)
        if target_long_side and im.format == "JPEG":
            s = target_long_side / max(size)
            if s < 0.5:                              # DCT scaling: 1/2, 1/4, 1/8
                im.draft("RGB", (math.ceil(stored_w * s), math.ceil(stored_h * s)))
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            notes.append("EXIF orientation could not be applied")
        rgb = np.asarray(_to_rgb(im))
    return LoadedImage(rgb=rgb, size=size, f35_mm=f35, notes=notes)


def resolve_intrinsics(path: str | Path, loaded: LoadedImage,
                       examples: ExampleIntrinsics | None,
                       ) -> tuple[Intrinsics, str, str]:
    """Intrinsics of `loaded.rgb` + (source, human-readable note).
    Priority: bundled example (exact sensor calibration) > EXIF 35 mm
    equivalent focal length > default assumption. Derived at the full file
    resolution, then rescaled if the JPEG was decoded at a reduced scale."""
    w, h = loaded.size
    hit = examples.lookup(path) if examples is not None else None
    if hit is not None:
        name, intr = hit
        source, note = "example", f"exact sensor intrinsics of example `{name}`"
    elif loaded.f35_mm is not None:
        intr = Intrinsics.from_35mm_equivalent(loaded.f35_mm, w, h)
        source, note = "exif", (f"from EXIF 35 mm-equivalent focal length "
                                f"{loaded.f35_mm:.0f} mm (fx = fy = {intr.fx:.0f} px "
                                f"at {w}×{h})")
    else:
        intr = Intrinsics.default_for(w, h)
        source, note = "default", (
            f"**assumed** (no calibration or EXIF focal length): Kinect-like "
            f"fx = fy = {DEFAULT_FX_PER_LONG_SIDE} × long side = {intr.fx:.0f} px "
            f"at {w}×{h}, centred principal point — absolute sizes scale with "
            f"this guess")
    rh, rw = loaded.rgb.shape[:2]
    if (rw, rh) != (w, h):
        intr = intr.scaled(rw / w, rh / h)
    return intr, source, note


# segmentation model wrapper (the only place that knows the model specifics)

def resolve_weights() -> Path:
    """Local path of the segmentation weights: env MASKRCNN_CKPT, else
    env MASKRCNN_HF_REPO (+ MASKRCNN_HF_FILENAME) downloaded from the
    Hugging Face Hub, else weights/maskrcnn_min400.pt next to this file."""
    local = os.environ.get("MASKRCNN_CKPT")
    if local:
        p = Path(local).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MASKRCNN_CKPT={local!r} does not exist.")
        return p
    repo = os.environ.get("MASKRCNN_HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        filename = os.environ.get("MASKRCNN_HF_FILENAME", DEFAULT_HF_FILENAME)
        try:
            return Path(hf_hub_download(repo_id=repo, filename=filename,
                                        repo_type=os.environ.get("MASKRCNN_HF_REPO_TYPE", "model"),
                                        token=os.environ.get("HF_TOKEN")))
        except Exception as e:
            raise RuntimeError(
                f"Could not download {filename!r} from the Hugging Face repo "
                f"{repo!r} ({type(e).__name__}: {e}). Check MASKRCNN_HF_REPO / "
                f"MASKRCNN_HF_FILENAME and, for a private repo, HF_TOKEN.") from e
    if DEFAULT_WEIGHTS_FILE.is_file():
        return DEFAULT_WEIGHTS_FILE
    raise FileNotFoundError(
        "No segmentation weights found. Either set MASKRCNN_CKPT to a local "
        "weights file, or MASKRCNN_HF_REPO (+ MASKRCNN_HF_FILENAME, default "
        f"{DEFAULT_HF_FILENAME!r}) to a Hugging Face model repo, or place the "
        f"file at {DEFAULT_WEIGHTS_FILE}. Create it from a training checkpoint "
        "with: python app/export_weights.py <best.pth> <out.pt>")


def _class_names(raw, num_classes: int) -> list[str]:
    """Normalises stored class names to a list indexed by label id
    (0 = background); falls back to std10 or generic names."""
    names = None
    if isinstance(raw, dict):                       # {category_id: name}
        names = ["__background__"] + [str(raw[k]) for k in sorted(raw, key=int)]
    elif isinstance(raw, (list, tuple)):
        names = [str(n) for n in raw]
        if len(names) == num_classes - 1:
            names = ["__background__"] + names
    if names is not None and len(names) == num_classes:
        return names
    if num_classes == len(STD10_FALLBACK) + 1:
        print("[pipeline] warning: no usable class names in the weights file; "
              "assuming the std10 order")
        return ["__background__"] + STD10_FALLBACK
    print("[pipeline] warning: no usable class names in the weights file")
    return ["__background__"] + [f"class_{i}" for i in range(1, num_classes)]


class Segmenter:
    """Instance segmentation model behind a minimal interface:
    `Segmenter.load(path, device)` and `predict(rgb, conf)`. Swapping in
    another torchvision Mask R-CNN only needs a weights file with a
    different meta["arch"] (see SEGMENTATION_ARCHS) and class names."""

    def __init__(self, model: torch.nn.Module, class_names: list[str],
                 device: torch.device, meta: dict | None = None):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.meta = meta or {}
        self._lock = threading.Lock()              # score_thresh is model state

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "Segmenter":
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"{path} cannot be loaded with weights_only=True — probably a "
                "full training checkpoint (optimizer, scheduler, numpy objects). "
                "Convert it once with: python app/export_weights.py "
                f"{path} <out.pt>  ({type(e).__name__})") from e
        if isinstance(blob, dict) and "state_dict" in blob:
            state = blob["state_dict"]
            meta = dict(blob.get("meta") or {})
            raw_names = blob.get("class_names")
        else:                                       # bare state_dict
            state, meta, raw_names = blob, {}, None
        arch = meta.get("arch", DEFAULT_ARCH)
        if arch not in SEGMENTATION_ARCHS:
            raise ValueError(f"Unsupported architecture {arch!r}; "
                             f"supported: {sorted(SEGMENTATION_ARCHS)}")
        num_classes = int(state["roi_heads.box_predictor.cls_score.weight"].shape[0])
        model = SEGMENTATION_ARCHS[arch](weights=None, weights_backbone=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
        model.load_state_dict(state)
        meta["arch"] = arch
        return cls(model, _class_names(raw_names, num_classes), device, meta)

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, conf: float) -> list[dict]:
        """Instances with score >= conf, sorted by score: {mask, label, score}.
        The threshold is applied inside the ROI heads, so only kept detections
        get their masks pasted at image resolution (torchvision's default
        0.05 would paste up to 100 masks)."""
        tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)
        with self._lock:
            self.model.roi_heads.score_thresh = max(float(conf) - 1e-6, 0.0)
            out = self.model([tensor.to(self.device)])[0]
        keep = out["scores"] >= conf
        masks = (out["masks"][keep, 0] >= 0.5).cpu().numpy()
        labels = out["labels"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        return [{"mask": masks[i], "label": self.class_names[int(labels[i])],
                 "score": float(scores[i])} for i in order if masks[i].any()]


def suppress_duplicates(instances: list[dict], iou_thr: float = DUP_MASK_IOU,
                        containment_thr: float = DUP_SAME_CLASS_CONTAINMENT,
                        ) -> tuple[list[dict], int]:
    """Greedy class-agnostic mask NMS over score-sorted instances: drops an
    instance whose mask IoU with an already kept one is >= iou_thr (any
    class), or which lies >= containment_thr inside a kept mask of the same
    class. Returns (kept, number removed)."""
    kept, flat, areas = [], [], []
    for inst in instances:
        m = inst["mask"].ravel()
        a = int(m.sum())
        dup = False
        for k, km, ka in zip(kept, flat, areas):
            inter = int(np.count_nonzero(m & km))
            if inter == 0:
                continue
            if inter / (a + ka - inter) >= iou_thr or (
                    k["label"] == inst["label"] and inter / a >= containment_thr):
                dup = True
                break
        if not dup:
            kept.append(inst); flat.append(m); areas.append(a)
    return kept, len(instances) - len(kept)


# full pipeline

@dataclass
class DetectedObject:
    label: str
    score: float
    mask: np.ndarray            # bool [h, w] at processing resolution
    points: np.ndarray          # [N, 3] cleaned cloud, camera frame, metres
    colors: np.ndarray          # [N, 3] RGB in 0..1 of the source pixels
    obb: tuple | None           # (corners [8, 3], edges) of the oriented box
    metrics: dict

    @property
    def approximate(self) -> bool:
        """Beyond the usable range of monocular 3D (RELIABLE_RANGE_M)."""
        d = self.metrics.get("dist_to_camera_m", np.nan)
        return bool(np.isfinite(d) and d > RELIABLE_RANGE_M)


@dataclass
class SceneResult:
    objects: list[DetectedObject]
    scene: tuple[np.ndarray, np.ndarray] | None   # sparse coloured scene cloud
    rgb: np.ndarray                               # image at processing resolution
    intrinsics: Intrinsics                        # ... and its intrinsics
    input_size: tuple[int, int]                   # (W, H) given to process()
    n_detections: int = 0                         # after the confidence threshold
    n_duplicates: int = 0                         # removed by suppress_duplicates
    n_too_sparse: int = 0                         # clouds below MIN_POINTS
    timings: dict = field(default_factory=dict)

    @property
    def resized(self) -> bool:
        return self.input_size != (self.rgb.shape[1], self.rgb.shape[0])


class DemoPipeline:
    def __init__(self, weights_path: str | Path | None = None,
                 device: torch.device | None = None,
                 depth_backend: str = DEPTH_BACKEND):
        if depth_backend not in DEPTH_BACKENDS:
            raise ValueError(f"Unknown depth backend {depth_backend!r} (env "
                             f"DEPTH_BACKEND); valid options: "
                             f"{', '.join(DEPTH_BACKENDS)}")
        self.device = device or pick_device()
        configure_cpu_threads(self.device)
        self.weights_path = Path(weights_path) if weights_path else resolve_weights()
        self.segmenter = Segmenter.load(self.weights_path, self.device)

        self.depth_backend = depth_backend
        self.depth_cfg = DEPTH_BACKENDS[depth_backend]
        self.depth_proc = AutoImageProcessor.from_pretrained(self.depth_cfg["model_id"])
        self.depth_model = (AutoModelForDepthEstimation
                            .from_pretrained(self.depth_cfg["model_id"])
                            .to(self.device).eval())

    @property
    def class_names(self) -> list[str]:
        return self.segmenter.class_names[1:]

    # stage 1: instance segmentation (+ cross-class duplicate suppression)
    def segment(self, rgb: np.ndarray, conf: float = 0.5) -> tuple[list[dict], int]:
        return suppress_duplicates(self.segmenter.predict(rgb, conf))

    # stage 2: monocular metric depth
    @torch.no_grad()
    def depth(self, rgb: np.ndarray, intr: Intrinsics | sd.Intrinsics | None = None) -> np.ndarray:
        h, w = rgb.shape[:2]
        inputs = self.depth_proc(images=Image.fromarray(rgb),
                                 return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)
        post = self.depth_proc.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)])
        raw = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
        if self.depth_backend == "relative_global":
            disp = self.depth_cfg["s"] * raw + self.depth_cfg["t"]
            z = np.where(disp > 1e-6, 1.0 / np.maximum(disp, 1e-6), 0.0)
        elif self.depth_backend == "metric_focal":
            # fitted on landscape frames where width = long side
            fx = intr.fx if intr is not None else DEFAULT_FX_PER_LONG_SIDE * max(w, h)
            z = (fx / max(w, h)) / self.depth_cfg["F0"] * raw
        else:
            z = self.depth_cfg["scale"] * raw
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, 0.0, MAX_DEPTH_M).astype(np.float32)

    # all stages
    def process(self, rgb: np.ndarray, conf: float = 0.5,
                intrinsics: Intrinsics | None = None,
                scene_stride: int | None = None) -> SceneResult:
        """rgb at any resolution with its intrinsics (default assumption if
        None). The image is first brought to the processing resolution
        (`processing_size`) and the intrinsics are rescaled with it; masks,
        clouds and the returned image are at that resolution. If scene_stride
        is set, a sparse coloured cloud of the whole scene is returned too."""
        h0, w0 = rgb.shape[:2]
        intr0 = intrinsics or Intrinsics.default_for(w0, h0)
        w, h = processing_size(w0, h0)
        if (w, h) != (w0, h0):
            interp = cv2.INTER_AREA if w < w0 else cv2.INTER_CUBIC
            rgb = cv2.resize(rgb, (w, h), interpolation=interp)
            intr = intr0.scaled(w / w0, h / h0)
        else:
            intr = intr0
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        intr_sd = intr.to_sd()

        timings = {}
        t = time.perf_counter()
        instances, n_dup = self.segment(rgb, conf=conf)
        timings["segmentation"] = time.perf_counter() - t
        t = time.perf_counter()
        depth = self.depth(rgb, intr)
        timings["depth"] = time.perf_counter() - t

        t = time.perf_counter()
        objects, n_sparse = [], 0
        for inst in instances:
            pcd = pc.build_cloud(depth, intr_sd, inst["mask"], rgb)
            if len(pcd.points) < MIN_POINTS:
                n_sparse += 1
                continue
            objects.append(DetectedObject(
                label=inst["label"], score=inst["score"], mask=inst["mask"],
                points=np.asarray(pcd.points), colors=np.asarray(pcd.colors),
                obb=pc.obb_wireframe(pcd), metrics=pc.object_3d_metrics(pcd)))

        scene = None
        if scene_stride:
            valid = pc.usable_depth(depth) & ~pc.depth_discontinuities(depth)
            scene = sd.backproject(depth, intr_sd, mask=valid, rgb=rgb, stride=scene_stride)
        timings["3d"] = time.perf_counter() - t
        return SceneResult(objects=objects, scene=scene, rgb=rgb, intrinsics=intr,
                           input_size=(w0, h0), n_detections=len(instances) + n_dup,
                           n_duplicates=n_dup, n_too_sparse=n_sparse, timings=timings)
