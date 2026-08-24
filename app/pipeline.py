"""Inference pipeline of the web demo.

RGB image -> Mask R-CNN instance masks -> monocular depth (Depth Anything V2
Small, GLOBAL disparity alignment - no depth sensor at inference time) ->
per-object point clouds (voxel + SOR cleaning) -> 3D characteristics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# segmentation model 
# category_id order of the std10 COCO files; index 0 is the background class
CLASSES = ["__background__", "bed", "chair", "sofa", "table", "desk",
           "toilet", "dresser", "nightstand", "bookshelf", "bathtub"]
NUM_CLASSES = len(CLASSES)

DEFAULT_CKPT = Path(os.environ.get(
    "MASKRCNN_CKPT",
    "~/Downloads/maskrcnn_torchvision_std10/best.pth")).expanduser()

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# depth: global lstsq_disp alignment (notebook 06) 
# depth = 1 / (s * raw_disparity + t); fitted on 150 validation images.
GLOBAL_DISP_S = 0.0861
GLOBAL_DISP_T = 0.2090
MAX_DEPTH_M = 8.0
MIN_DEPTH_M = 0.3

# cloud pipeline constants (must mirror notebooks 05/07) 
VOXEL_SIZE = 0.02
SOR_NEIGHBORS = 20
SOR_STD_RATIO = 2.0
MIN_POINTS = 100

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


@dataclass
class DetectedObject:
    label: str
    score: float
    mask: np.ndarray            # bool [H, W]
    points: np.ndarray          # [N, 3] cleaned cloud, camera frame, metres
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
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device: torch.device | None = None):
        self.device = device or pick_device()

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        self.segm = build_maskrcnn(NUM_CLASSES)
        self.segm.load_state_dict(state)
        self.segm.to(self.device).eval()

        self.depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = (AutoModelForDepthEstimation
                            .from_pretrained(DEPTH_MODEL_ID)
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
    def depth(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        inputs = self.depth_proc(images=Image.fromarray(rgb),
                                 return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)
        post = self.depth_proc.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)])
        raw = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
        disp = GLOBAL_DISP_S * raw + GLOBAL_DISP_T
        z = np.where(disp > 1e-6, 1.0 / np.maximum(disp, 1e-6), 0.0)
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, 0.0, MAX_DEPTH_M).astype(np.float32)

    # stage 3: point clouds and 3D characteristics
    @staticmethod
    def _backproject(depth: np.ndarray, intr: Intrinsics,
                     mask: np.ndarray) -> np.ndarray:
        v, u = np.nonzero(mask)
        z = depth[v, u]
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _clean(points: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if len(pcd.points) == 0:
            return pcd
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        if len(pcd.points) >= SOR_NEIGHBORS:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NEIGHBORS,
                                                    std_ratio=SOR_STD_RATIO)
        return pcd

    @staticmethod
    def _metrics(pcd: o3d.geometry.PointCloud) -> dict:
        pts = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)
        m = {"n_points": len(pts),
             "distance_m": float(np.linalg.norm(centroid)),
             "centroid": centroid}
        try:
            obb = pcd.get_oriented_bounding_box()
            e1, e2, e3 = sorted(obb.extent, reverse=True)
            m.update(obb_extent_1=e1, obb_extent_2=e2, obb_extent_3=e3,
                     obb_volume_m3=float(np.prod(obb.extent)))
        except RuntimeError:
            m.update(obb_extent_1=np.nan, obb_extent_2=np.nan,
                     obb_extent_3=np.nan, obb_volume_m3=np.nan)
        try:
            hull, _ = pcd.compute_convex_hull()
            m["hull_volume_m3"] = float(hull.get_volume())
        except RuntimeError:
            m["hull_volume_m3"] = np.nan
        return m

    # full pipeline
    def process(self, rgb: np.ndarray, conf: float = 0.5,
                intrinsics: Intrinsics | None = None,
                scene_stride: int | None = None,
                ) -> tuple[list[DetectedObject], np.ndarray | None]:
        """Returns detected objects and, if scene_stride is set, a sparse cloud
        of the whole scene (every scene_stride-th pixel) for 3D context."""
        h, w = rgb.shape[:2]
        intr = intrinsics or Intrinsics.default_for(w, h)
        instances = self.segment(rgb, conf=conf)
        depth = self.depth(rgb)
        valid = (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M)

        objects = []
        for inst in instances:
            pts = self._backproject(depth, intr, inst["mask"] & valid)
            pcd = self._clean(pts)
            if len(pcd.points) < MIN_POINTS:
                continue
            objects.append(DetectedObject(
                label=inst["label"], score=inst["score"], mask=inst["mask"],
                points=np.asarray(pcd.points), metrics=self._metrics(pcd)))

        scene_points = None
        if scene_stride:
            sub = np.zeros_like(valid)
            sub[::scene_stride, ::scene_stride] = True
            scene_points = self._backproject(depth, intr, valid & sub)
        return objects, scene_points
