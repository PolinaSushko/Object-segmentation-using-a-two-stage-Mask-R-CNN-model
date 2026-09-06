"""Inference pipeline of the web demo.

RGB image -> Mask R-CNN instance masks -> monocular depth (Depth Anything V2,
no depth sensor at inference time) -> per-object point clouds (mask refinement,
voxel + SOR cleaning — `utils/pointcloud.py`, shared with notebooks 05/07/08)
-> 3D characteristics.

Two sensor-free depth backends are implemented (notebook 06 compares them):
  * "relative_global": relative model (disparity output) + the global affine
    alignment in disparity space fitted on validation, depth = 1/(s*raw + t);
  * "metric_scaled": metric indoor model (metres out of the box) + one global
    scale factor fitted on validation — the FOV mismatch between its training
    renders and real Kinect-class cameras leaves a constant bias;
  * "metric_focal": the same metric model, bias corrected from the camera
    intrinsics: depth = raw * (fx / W) / F0, where F0 is the focal length
    (per unit image width) the model implicitly assumes (fitted on validation).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils.pointcloud as pc  # noqa: E402
import utils.sunrgbd as sd     # noqa: E402

# segmentation model
# category_id order of the std10 COCO files; index 0 is the background class
CLASSES = ["__background__", "bed", "chair", "sofa", "table", "desk",
           "toilet", "dresser", "nightstand", "bookshelf", "bathtub"]
NUM_CLASSES = len(CLASSES)

DEFAULT_CKPT = Path(os.environ.get(
    "MASKRCNN_CKPT",
    "~/Downloads/maskrcnn_torchvision_std10/best.pth")).expanduser()

# depth backends (constants fitted on 200 sensor-stratified validation images,
# notebook 06; test AbsRel / delta1: relative_global 0.279 / 0.56,
# metric_scaled 0.231 / 0.75, metric_focal 0.245 / 0.69)
DEPTH_BACKENDS = {
    "relative_global": {
        "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        # depth = 1 / (s * raw_disparity + t)
        "s": 0.0811, "t": 0.2618,
        "label": "Depth Anything V2 Small (relative) + global disparity alignment",
    },
    "metric_scaled": {
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        # depth = scale * raw_depth
        "scale": 0.8592,
        "label": "Depth Anything V2 Metric-Indoor Small + global scale",
    },
    "metric_focal": {
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        # depth = (fx / W) / F0 * raw_depth
        "F0": 1.0442,
        "label": "Depth Anything V2 Metric-Indoor Small + focal-aware scale",
    },
}
# default = the best sensor-free backend of notebooks 06/08 at the same latency as
# the relative model (metric Base is better still, but 2.6x slower)
DEPTH_BACKEND = os.environ.get("DEPTH_BACKEND", "metric_scaled")

MAX_DEPTH_M = pc.MAX_DEPTH_M
MIN_DEPTH_M = pc.MIN_DEPTH_M
MIN_POINTS = pc.MIN_POINTS

# median fx / image_width over the SUN RGB-D test split; used when an uploaded
# image carries no sensor metadata
DEFAULT_FX_PER_WIDTH = 0.925


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def default_for(cls, width: int, height: int) -> "Intrinsics":
        f = DEFAULT_FX_PER_WIDTH * width
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)

    def to_sd(self) -> sd.Intrinsics:
        return sd.Intrinsics(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


@dataclass
class DetectedObject:
    label: str
    score: float
    mask: np.ndarray            # bool [H, W], raw predicted mask
    points: np.ndarray          # [N, 3] cleaned cloud, camera frame, metres
    colors: np.ndarray          # [N, 3] RGB in 0..1 of the source pixels
    obb: tuple | None           # (corners [8, 3], edges) of the oriented box
    metrics: dict


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_maskrcnn(num_classes: int):
    """Same head replacement as the training notebook 02."""
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


class DemoPipeline:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT,
                 device: torch.device | None = None,
                 depth_backend: str = DEPTH_BACKEND):
        self.device = device or pick_device()

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        self.segm = build_maskrcnn(NUM_CLASSES)
        self.segm.load_state_dict(state)
        self.segm.to(self.device).eval()

        self.depth_backend = depth_backend
        self.depth_cfg = DEPTH_BACKENDS[depth_backend]
        self.depth_proc = AutoImageProcessor.from_pretrained(self.depth_cfg["model_id"])
        self.depth_model = (AutoModelForDepthEstimation
                            .from_pretrained(self.depth_cfg["model_id"])
                            .to(self.device).eval())

    # stage 1: instance segmentation
    @torch.no_grad()
    def segment(self, rgb: np.ndarray, conf: float = 0.5) -> list[dict]:
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        out = self.segm([tensor.to(self.device)])[0]
        keep = out["scores"] >= conf
        results = []
        for mask, label, score in zip(out["masks"][keep, 0].cpu().numpy(),
                                      out["labels"][keep].cpu().numpy(),
                                      out["scores"][keep].cpu().numpy()):
            results.append({"mask": mask >= 0.5, "label": CLASSES[int(label)],
                            "score": float(score)})
        return results

    # stage 2: monocular metric depth
    @torch.no_grad()
    def depth(self, rgb: np.ndarray, intr: sd.Intrinsics | None = None) -> np.ndarray:
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
            fx_per_width = (intr.fx if intr is not None else DEFAULT_FX_PER_WIDTH * w) / w
            z = fx_per_width / self.depth_cfg["F0"] * raw
        else:
            z = self.depth_cfg["scale"] * raw
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, 0.0, MAX_DEPTH_M).astype(np.float32)

    # stage 3: point clouds and 3D characteristics (utils/pointcloud.py)
    def process(self, rgb: np.ndarray, conf: float = 0.5,
                intrinsics: Intrinsics | None = None,
                scene_stride: int | None = None,
                ) -> tuple[list[DetectedObject], tuple[np.ndarray, np.ndarray] | None]:
        """Returns detected objects and, if scene_stride is set, a sparse
        coloured cloud (points, colors) of the whole scene for 3D context."""
        h, w = rgb.shape[:2]
        intr = (intrinsics or Intrinsics.default_for(w, h)).to_sd()
        instances = self.segment(rgb, conf=conf)
        depth = self.depth(rgb, intr)

        objects = []
        for inst in instances:
            pcd = pc.build_cloud(depth, intr, inst["mask"], rgb)
            if len(pcd.points) < MIN_POINTS:
                continue
            metrics = pc.object_3d_metrics(pcd)
            objects.append(DetectedObject(
                label=inst["label"], score=inst["score"], mask=inst["mask"],
                points=np.asarray(pcd.points), colors=np.asarray(pcd.colors),
                obb=pc.obb_wireframe(pcd), metrics=metrics))

        scene = None
        if scene_stride:
            valid = pc.usable_depth(depth) & ~pc.depth_discontinuities(depth)
            scene = sd.backproject(depth, intr, mask=valid, rgb=rgb, stride=scene_stride)
        return objects, scene
