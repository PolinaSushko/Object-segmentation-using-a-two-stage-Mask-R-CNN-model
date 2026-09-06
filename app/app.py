"""Gradio demo: instance segmentation -> monocular depth -> 3D object analysis.

Run locally:  python app/app.py
Environment:  MASKRCNN_CKPT (checkpoint path), DEPTH_BACKEND
              ("relative_global" | "metric_scaled", see pipeline.py)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from pipeline import DemoPipeline, Intrinsics

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLE_INTRINSICS = json.loads((EXAMPLES_DIR / "intrinsics.json").read_text())

MAX_PLOT_POINTS = 40_000   # total budget for the object clouds (CPU-friendly)
MAX_SCENE_POINTS = 25_000  # extra budget for the scene-context cloud

# distinct object colors (RGB 0-255), reused across the overlay and the 3D plot
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (0, 128, 128), (220, 190, 255), (170, 110, 40)]

print("Loading models...")
t0 = time.time()
PIPELINE = DemoPipeline()
print(f"Models ready on {PIPELINE.device} ({time.time() - t0:.1f} s); "
      f"depth backend: {PIPELINE.depth_cfg['label']}")


def _overlay(rgb: np.ndarray, objects) -> np.ndarray:
    """Blends instance masks into the image and numbers each object."""
    out = rgb.astype(np.float32)
    for i, obj in enumerate(objects):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        out[obj.mask] = 0.55 * out[obj.mask] + 0.45 * color
    img = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, obj in enumerate(objects):
        ys, xs = np.nonzero(obj.mask)
        tag = f"{i + 1}: {obj.label} {obj.score:.2f}"
        x, y = float(xs.mean()), float(ys.mean())
        draw.text((x + 1, y + 1), tag, fill=(0, 0, 0), font=font)
        draw.text((x, y), tag, fill=(255, 255, 255), font=font)
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
    the camera's own, so the 3D view opens looking like the photograph."""
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
        name = f"{i + 1}: {obj.label}"
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
                x=xs, y=ys, z=zs, mode="lines", line=dict(color=own, width=3),
                name=name, legendgroup=name, showlegend=False, hoverinfo="skip"))
            top = corners[np.argmin(corners[:, 1])]      # highest corner (-Y up)
            fig.add_trace(go.Scatter3d(
                x=[top[0]], y=[top[2]], z=[-top[1] + 0.05], mode="text",
                text=[f"{i + 1}"], textfont=dict(color=own, size=14),
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


def _table(objects) -> pd.DataFrame:
    rows = []
    for i, obj in enumerate(objects):
        m = obj.metrics
        rows.append({
            "#": i + 1, "class": obj.label, "score": round(obj.score, 2),
            "distance, m": round(m["dist_to_camera_m"], 2),
            "size L×W×H, m": " × ".join(
                f"{m[f'obb_extent_{k}']:.2f}" for k in (1, 2, 3)),
            "OBB volume, m³": round(m["obb_volume_m3"], 3),
            "hull volume, m³": round(m["hull_volume_m3"], 3),
            "points": m["n_points"]})
    return pd.DataFrame(rows)


def analyse(image_path: str | None, conf: float, show_context: bool, color_mode: str):
    if not image_path:
        raise gr.Error("Upload an image or pick an example first.")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = rgb.shape[:2]

    known = EXAMPLE_INTRINSICS.get(Path(image_path).name)
    intr = Intrinsics(**known) if known else Intrinsics.default_for(w, h)
    intr_note = ("sensor intrinsics of the example scene" if known
                 else "no sensor metadata: Kinect-like intrinsics assumed "
                      f"(fx = fy = {intr.fx:.0f} px)")

    t0 = time.time()
    objects, scene = PIPELINE.process(
        rgb, conf=conf, intrinsics=intr, scene_stride=3 if show_context else None)
    dt = time.time() - t0

    if not objects:
        status = (f"No objects of the 10 supported classes found at "
                  f"confidence ≥ {conf:.2f} ({dt:.1f} s). Try a lower threshold.")
        return rgb, _cloud_figure([], scene, color_mode), pd.DataFrame(), status

    status = (f"{len(objects)} object(s) in {dt:.1f} s on {PIPELINE.device} — "
              f"{intr_note}. Depth: {PIPELINE.depth_cfg['label']} (no sensor). "
              f"Clouds show the *visible surface* of each object (single view); "
              f"layout and shapes are reliable, absolute sizes are approximate "
              f"(thesis: |volume error| ≈ 50%, usable range < 3.5 m).")
    return (_overlay(rgb, objects), _cloud_figure(objects, scene, color_mode),
            _table(objects), status)


with gr.Blocks(title="RGB → 3D object analysis") as demo:
    gr.Markdown(
        "# From a single RGB photo to 3D object characteristics\n"
        "Mask R-CNN (SUN RGB-D, 10 indoor classes) → Depth Anything V2 "
        "(monocular depth, no sensor) → per-object point clouds "
        "(Open3D) → spatial characteristics.")
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="Indoor scene (RGB)")
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
    run_btn.click(analyse, inputs=[image_in, conf_in, context_in, color_in],
                  outputs=[masks_out, cloud_out, table_out, status_out])

if __name__ == "__main__":
    demo.launch()
