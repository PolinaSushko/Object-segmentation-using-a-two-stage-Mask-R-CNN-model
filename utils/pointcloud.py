"""
pointcloud.py — the instance-mask -> point-cloud pipeline.

Single source of truth for notebooks 05/07/08 and the web demo (app/pipeline.py):

  1. mask refinement — drops the pixels whose depth cannot be trusted for
     *this* object: the mask boundary (mixed foreground/background depth),
     depth discontinuities (the "flying pixels" / curtains that connect an
     object to the wall behind it) and per-object depth outliers;
  2. back-projection through the pinhole model (`sunrgbd.backproject`);
  3. cloud cleaning in Open3D (voxel downsampling + statistical outlier removal);
  4. per-object 3D characteristics (centroid, distance, AABB/OBB extents,
     OBB / convex-hull volumes) and the Chamfer distance between clouds.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import ndimage

from . import sunrgbd as sd

# depth validity range (metres) — sensor range; also used for monocular maps
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = sd.MAX_DEPTH_M          # 8 m

# cloud cleaning
VOXEL_SIZE = 0.02                     # m, downsampling resolution
SOR_NEIGHBORS = 20                    # statistical outlier removal: neighbours...
SOR_STD_RATIO = 2.0                   # ...and the std-dev threshold
MIN_POINTS = 100                      # clouds with fewer points are discarded

# mask refinement (boundary / discontinuity filtering)
ERODE_PX = 2                          # boundary band removed from every mask
EDGE_WINDOW = 5                       # px, window of the local depth-range test
EDGE_TOL_M = 0.15                     # m, discontinuity threshold (absolute)...
EDGE_TOL_REL = 0.05                   # ...or relative to depth, whichever is larger
DEPTH_MAD_K = 3.0                     # per-object depth outliers: |z - med| > k * MAD
DEPTH_MAD_FLOOR_M = 0.05              # m, lower bound of the MAD scale


# 1. mask refinement

def usable_depth(depth: np.ndarray, min_depth: float = MIN_DEPTH_M,
                 max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Pixels with a trustworthy metric depth value."""
    return (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)


def depth_discontinuities(depth: np.ndarray, window: int = EDGE_WINDOW,
                          tol_m: float = EDGE_TOL_M,
                          tol_rel: float = EDGE_TOL_REL) -> np.ndarray:
    """True where the local depth range (window x window, valid pixels only)
    exceeds max(tol_m, tol_rel * depth) — the depth discontinuities on which
    boundary pixels, sensor flying pixels and the smooth ramps of monocular
    maps produce points floating between an object and its background."""
    valid = usable_depth(depth)
    d_max = ndimage.maximum_filter(np.where(valid, depth, -np.inf), size=window)
    d_min = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=window)
    local_range = np.where(valid, d_max - d_min, 0.0)
    return valid & (local_range > np.maximum(tol_m, tol_rel * depth))


def refine_mask(mask: np.ndarray, depth: np.ndarray, erode_px: int = ERODE_PX,
                edge_filter: bool = True, mad_k: float | None = DEPTH_MAD_K,
                ) -> np.ndarray:
    """Restricts an instance mask to the pixels whose depth can be attributed
    to the object: erodes the mask boundary, removes depth discontinuities and
    per-object depth outliers (robust MAD test). Any step can be disabled;
    with erode_px=0, edge_filter=False, mad_k=None this is just
    `mask & usable_depth(depth)` — the pipeline of the first version."""
    m = mask.astype(bool) & usable_depth(depth)
    if erode_px > 0:
        m &= ndimage.binary_erosion(mask.astype(bool), iterations=erode_px)
    if edge_filter:
        m &= ~depth_discontinuities(depth)
    if mad_k is not None and m.sum() >= MIN_POINTS:
        z = depth[m]
        med = np.median(z)
        scale = max(1.4826 * np.median(np.abs(z - med)), DEPTH_MAD_FLOOR_M)
        keep = np.zeros_like(m)
        keep[m] = np.abs(z - med) <= mad_k * scale
        m = keep
    return m


# 2 + 3. back-projection and cleaning

def clean_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Voxel downsample + statistical outlier removal."""
    if len(pcd.points) == 0:
        return pcd
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    if len(pcd.points) >= SOR_NEIGHBORS:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NEIGHBORS,
                                                std_ratio=SOR_STD_RATIO)
    return pcd


def build_cloud(depth: np.ndarray, intr: sd.Intrinsics, mask: np.ndarray,
                rgb: np.ndarray | None = None, refine: bool = True,
                ) -> o3d.geometry.PointCloud:
    """Instance mask + depth map -> cleaned Open3D cloud (camera frame, metres).
    `refine=True` applies `refine_mask` with the module defaults; pass an
    already refined mask with `refine=False` (paired protocols of notebooks
    07/08 intersect the refined masks of two depth maps first)."""
    pixels = refine_mask(mask, depth) if refine else (mask.astype(bool) & usable_depth(depth))
    points, colors = sd.backproject(depth, intr, mask=pixels, rgb=rgb)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return clean_cloud(pcd)


# 4. object characteristics and cloud comparison

def object_3d_metrics(pcd: o3d.geometry.PointCloud) -> dict:
    """Per-object 3D characteristics of a cleaned cloud (camera frame, metres).
    OBB extents are reported sorted (largest first) — tilt-invariant up to noise."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    aabb = pcd.get_axis_aligned_bounding_box().get_extent()
    m = {
        "n_points": len(pts),
        "centroid": centroid,
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "dist_to_camera_m": float(np.linalg.norm(centroid)),
        "aabb_w": float(aabb[0]), "aabb_h": float(aabb[1]), "aabb_d": float(aabb[2]),
    }
    # OBB and convex hull need enough non-degenerate points
    try:
        obb = pcd.get_oriented_bounding_box()
        m["obb_extent_1"], m["obb_extent_2"], m["obb_extent_3"] = \
            (float(e) for e in sorted(obb.extent, reverse=True))
        m["obb_volume_m3"] = float(np.prod(obb.extent))
    except RuntimeError:
        m["obb_extent_1"] = m["obb_extent_2"] = m["obb_extent_3"] = np.nan
        m["obb_volume_m3"] = np.nan
    try:
        hull, _ = pcd.compute_convex_hull()
        m["hull_volume_m3"] = float(hull.get_volume())
    except RuntimeError:
        m["hull_volume_m3"] = np.nan
    return m


def obb_wireframe(pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Corners (8, 3) and the 12 edges of the oriented bounding box, for plots."""
    try:
        obb = pcd.get_oriented_bounding_box()
    except RuntimeError:
        return None
    corners = np.asarray(obb.get_box_points())
    # Open3D corner order: 0-1-7-2 is the bottom face, 3-6-4-5 the top face
    edges = [(0, 1), (1, 7), (7, 2), (2, 0), (3, 6), (6, 4), (4, 5), (5, 3),
             (0, 3), (1, 6), (7, 4), (2, 5)]
    return corners, edges


def chamfer_cm(pcd_a: o3d.geometry.PointCloud, pcd_b: o3d.geometry.PointCloud) -> float:
    """Symmetric Chamfer-L1 distance between two clouds, in centimetres."""
    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    return float(100.0 * 0.5 * (d_ab.mean() + d_ba.mean()))


def pct_err(est: float, ref: float) -> float:
    """Signed percentage error of an estimate against the reference value."""
    if not np.isfinite(est) or not np.isfinite(ref) or ref <= 0:
        return np.nan
    return 100.0 * (est - ref) / ref
