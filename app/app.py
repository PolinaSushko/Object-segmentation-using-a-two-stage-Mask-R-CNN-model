"""Gradio demo: instance segmentation -> monocular depth -> 3D object analysis.

Run locally (repository root):  python app/app.py
Environment (see README.md and pipeline.py):
  MASKRCNN_CKPT         local weights file written by export_weights.py
  MASKRCNN_HF_REPO      ... or a Hugging Face model repo holding it
  MASKRCNN_HF_FILENAME  file name in that repo (default maskrcnn_std10.pt)
  DEPTH_BACKEND         metric_scaled (default) | metric_focal | relative_global
  TORCH_NUM_THREADS     CPU threads (default: CPUs available to the process)
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

import pipeline as P

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLE_INTRINSICS = P.ExampleIntrinsics(EXAMPLES_DIR)

MAX_PLOT_POINTS = 40_000   # total budget for the object clouds (CPU-friendly)
MAX_SCENE_POINTS = 25_000  # extra budget for the scene-context cloud
FONT_SIZE = 15

# distinct object colors (RGB 0-255), reused across the overlay and the 3D plot
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (0, 128, 128), (220, 190, 255), (170, 110, 40)]
APPROX = "≈"               # marks objects beyond P.RELIABLE_RANGE_M


@lru_cache(maxsize=1)
def get_pipeline() -> P.DemoPipeline:
    print("Loading models...")
    t0 = time.time()
    pipe = P.DemoPipeline()
    print(f"Models ready on {pipe.device} ({time.time() - t0:.1f} s); weights: "
          f"{pipe.weights_path.name} ({pipe.segmenter.meta.get('arch')}, "
          f"{len(pipe.class_names)} classes); depth backend: {pipe.depth_cfg['label']}")
    return pipe


@lru_cache(maxsize=1)
def _font():
    """DejaVu Sans if the system has it, else Pillow's built-in scalable font
    (Pillow >= 10.1), else the legacy bitmap font."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=FONT_SIZE)
    except TypeError:
        return ImageFont.load_default()


def _label_anchor(mask: np.ndarray) -> tuple[int, int]:
    """Innermost mask pixel (maximum of the distance transform): always
    inside the object, unlike the centroid of a concave mask."""
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x) - 1, int(y) - 1


def _overlay(rgb: np.ndarray, objects) -> np.ndarray:
    """Blends instance masks into the image and labels each object; objects
    beyond the reliable range get a hatched fill and the ≈ mark."""
    out = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = ((xx + yy) // 6) % 2 == 0
    for i, obj in enumerate(objects):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        fill = obj.mask & hatch if obj.approximate else obj.mask
        out[fill] = 0.55 * out[fill] + 0.45 * color
        contours, _ = cv2.findContours(obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color.tolist(), 1)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font = _font()
    for i, obj in enumerate(objects):
        tag = f"{i + 1}: {obj.label} {obj.score:.2f}" + (f" {APPROX}" if obj.approximate else "")
        x0, y0, x1, y1 = draw.textbbox((0, 0), tag, font=font)
        tw, th = x1 - x0, y1 - y0
        cx, cy = _label_anchor(obj.mask)
        x = int(np.clip(cx - tw / 2, 2, max(2, w - tw - 2)))
        y = int(np.clip(cy - th / 2, 2, max(2, h - th - 2)))
        draw.rectangle((x - 2, y - 2, x + tw + 2, y + th + 2), fill=(0, 0, 0))
        draw.text((x - x0, y - y0), tag, fill=PALETTE[i % len(PALETTE)], font=font)
    return np.asarray(img)


def _subsample(n: int, budget: int) -> np.ndarray:
    if n <= budget:
        return np.arange(n)
    return np.random.default_rng(0).choice(n, budget, replace=False)


def _rgb_strings(colors01: np.ndarray, alpha: float | None = None) -> list[str]:
    c = (np.clip(colors01, 0, 1) * 255).astype(int)
    if alpha is None:
        return [f"rgb({r},{g},{b})" for r, g, b in c]
    return [f"rgba({r},{g},{b},{alpha})" for r, g, b in c]


def _cloud_figure(objects, scene=None, color_mode: str = "RGB") -> go.Figure:
    """Per-object clouds (one legend entry each, with its OBB wireframe and
    label) plus an optional scene-context cloud. Plot axes follow the camera
    frame — X right, Z forward (depth), -Y up — and the initial viewpoint is
    the camera's own, so the 3D view opens looking like the photograph.
    Objects beyond the reliable range have dashed boxes and the ≈ mark."""
    fig = go.Figure()
    if scene is not None and len(scene[0]):
        pts, cols = scene
        idx = _subsample(len(pts), MAX_SCENE_POINTS)
        pts, cols = pts[idx], cols[idx]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1], mode="markers",
            marker=dict(size=1.2, color=_rgb_strings(cols, alpha=0.45)
                        if color_mode == "RGB" else "rgba(128,128,128,0.3)"),
            name="scene context", hoverinfo="skip"))

    total = sum(len(o.points) for o in objects) or 1
    for i, obj in enumerate(objects):
        r, g, b = PALETTE[i % len(PALETTE)]
        own = f"rgb({r},{g},{b})"
        idx = _subsample(len(obj.points), max(500, int(MAX_PLOT_POINTS * len(obj.points) / total)))
        pts = obj.points[idx]
        name = f"{i + 1}: {obj.label}" + (f" {APPROX}" if obj.approximate else "")
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1], mode="markers",
            marker=dict(size=2.2, color=_rgb_strings(obj.colors[idx])
                        if color_mode == "RGB" else own),
            name=name, legendgroup=name,
            hovertemplate=f"{name}<extra></extra>"))
        if obj.obb is not None:
            corners, edges = obj.obb
            xs, ys, zs = [], [], []
            for a, b_ in edges:
                xs += [corners[a, 0], corners[b_, 0], None]
                ys += [corners[a, 2], corners[b_, 2], None]
                zs += [-corners[a, 1], -corners[b_, 1], None]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=own, width=3, dash="dash" if obj.approximate else "solid"),
                name=name, legendgroup=name, showlegend=False, hoverinfo="skip"))
            top = corners[np.argmin(corners[:, 1])]      # highest corner (-Y up)
            fig.add_trace(go.Scatter3d(
                x=[top[0]], y=[top[2]], z=[-top[1] + 0.05], mode="text",
                text=[f"{i + 1}" + (APPROX if obj.approximate else "")],
                textfont=dict(color=own, size=14),
                name=name, legendgroup=name, showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        scene=dict(xaxis_title="X, m", yaxis_title="Z (depth), m",
                   zaxis_title="-Y (up), m", aspectmode="data",
                   # camera behind the sensor, slightly above, looking along +Z
                   camera=dict(eye=dict(x=0.0, y=-1.9, z=0.35),
                               up=dict(x=0, y=0, z=1),
                               center=dict(x=0, y=0, z=0))),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0), height=600)
    return fig


def _fmt(value, digits: int) -> str:
    """Fixed-point number, or an em dash for NaN / missing values."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{digits}f}" if np.isfinite(v) else "—"


def _table(objects) -> pd.DataFrame:
    rows = []
    for i, obj in enumerate(objects):
        m = obj.metrics
        rows.append({
            "#": i + 1, "class": obj.label, "score": _fmt(obj.score, 2),
            "distance, m": _fmt(m.get("dist_to_camera_m"), 2),
            "size L×W×H, m": " × ".join(_fmt(m.get(f"obb_extent_{k}"), 2) for k in (1, 2, 3)),
            "OBB volume, m³": _fmt(m.get("obb_volume_m3"), 3),
            "hull volume, m³": _fmt(m.get("hull_volume_m3"), 3),
            "points": m.get("n_points", "—"),
            "reliability": (f"{APPROX} approximate (> {P.RELIABLE_RANGE_M:g} m)"
                            if obj.approximate else "within range")})
    return pd.DataFrame(rows)


def analyse(image_path: str | None, conf: float, show_context: bool, color_mode: str):
    if not image_path:
        raise gr.Error("Upload an image or pick an example first.")
    pipe = get_pipeline()
    try:
        loaded = P.load_image(image_path, target_long_side=P.PROC_LONG_SIDE)
    except Exception as e:
        raise gr.Error(f"Could not read the image ({type(e).__name__}: {e}).")
    intr, intr_source, intr_note = P.resolve_intrinsics(image_path, loaded, EXAMPLE_INTRINSICS)

    t0 = time.time()
    res = pipe.process(loaded.rgb, conf=conf, intrinsics=intr,
                       scene_stride=3 if show_context else None)
    dt = time.time() - t0
    objects = res.objects
    tm = res.timings

    lines = []
    if objects:
        lines.append(f"**{len(objects)} object(s)** in {dt:.1f} s on {pipe.device} "
                     f"(segmentation {tm['segmentation']:.1f} s, depth {tm['depth']:.1f} s, "
                     f"3D {tm['3d']:.1f} s).")
    else:
        lines.append(f"**No objects** of the {len(pipe.class_names)} supported classes "
                     f"({', '.join(pipe.class_names)}) at confidence ≥ {conf:.2f} "
                     f"({dt:.1f} s). Try a lower threshold.")
    if res.n_duplicates:
        lines.append(f"{res.n_duplicates} duplicate detection(s) of the same object removed "
                     f"(mask IoU ≥ {P.DUP_MASK_IOU} across classes).")
    if res.n_too_sparse:
        lines.append(f"{res.n_too_sparse} detection(s) skipped: fewer than "
                     f"{P.MIN_POINTS} reliable 3D points after filtering.")
    w0, h0 = loaded.size
    h, w = res.rgb.shape[:2]
    if (w, h) != (w0, h0):
        lines.append(f"Input {w0}×{h0} processed at {w}×{h} (the 3D filters work in "
                     f"pixels and are tuned for SUN RGB-D frames).")
    lines.append(f"Intrinsics [{intr_source}]: {intr_note}.")
    lines += [f"Note: {n}." for n in loaded.notes]
    n_far = sum(o.approximate for o in objects)
    lines.append(
        f"Depth: {pipe.depth_cfg['label']} (no sensor). Clouds show the *visible "
        f"surface* of each object (single view); layout and shapes are reliable, "
        f"absolute sizes are approximate — thesis evaluation of this backend: "
        f"{pipe.depth_cfg['thesis_3d']}; usable range ≈ up to "
        f"{P.RELIABLE_RANGE_M:g} m" + (f" ({n_far} object(s) beyond it marked {APPROX})."
                                       if n_far else "."))
    status = "\n\n".join(lines)

    if not objects:
        return res.rgb, _cloud_figure([], res.scene, color_mode), pd.DataFrame(), status
    return (_overlay(res.rgb, objects), _cloud_figure(objects, res.scene, color_mode),
            _table(objects), status)


with gr.Blocks(title="RGB → 3D object analysis") as demo:
    gr.Markdown(
        "# From a single RGB photo to 3D object characteristics\n"
        "Mask R-CNN (SUN RGB-D, 10 indoor classes) → Depth Anything V2 "
        "(monocular depth, no sensor) → per-object point clouds "
        "(Open3D) → spatial characteristics.")
    with gr.Row():
        with gr.Column(scale=1):
            # image_mode=None: the original file reaches the pipeline unchanged
            # (EXIF orientation / focal length, byte hash of the examples)
            image_in = gr.Image(type="filepath", image_mode=None,
                                label="Indoor scene (RGB)")
            conf_in = gr.Slider(0.3, 0.9, value=0.5, step=0.05,
                                label="Detection confidence threshold")
            context_in = gr.Checkbox(value=True,
                                     label="Show scene context in 3D")
            color_in = gr.Radio(["RGB", "by object"], value="RGB",
                                label="Point colours in 3D")
            run_btn = gr.Button("Analyse scene", variant="primary")
            gr.Examples(
                examples=sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")),
                inputs=image_in, label="SUN RGB-D test scenes (never seen in training)")
            status_out = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Tab("Instance masks"):
                masks_out = gr.Image(label="Detected objects")
            with gr.Tab("3D point cloud"):
                cloud_out = gr.Plot(label="Per-object clouds with oriented boxes (camera frame)")
            with gr.Tab("Object characteristics"):
                table_out = gr.Dataframe(label="Per-object 3D characteristics")
    # one request at a time: the models are shared (and the segmenter's
    # score threshold is per-call model state)
    run_btn.click(analyse, inputs=[image_in, conf_in, context_in, color_in],
                  outputs=[masks_out, cloud_out, table_out, status_out],
                  concurrency_limit=1)

if __name__ == "__main__":
    get_pipeline()          # fail fast (missing weights, bad DEPTH_BACKEND)
    demo.launch()
