"""Gradio demo: instance segmentation -> monocular depth -> 3D object analysis.

Run locally:  python app/app.py
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

MAX_PLOT_POINTS = 40_000   # total budget for the 3D scatter (CPU-friendly)

# distinct object colors (RGB 0-255), reused across the overlay and the 3D plot
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (0, 128, 128), (220, 190, 255), (170, 110, 40)]

print("Loading models...")
t0 = time.time()
PIPELINE = DemoPipeline()
print(f"Models ready on {PIPELINE.device} ({time.time() - t0:.1f} s)")


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


MAX_SCENE_POINTS = 20_000  # extra budget for the gray scene-context cloud


def _cloud_figure(objects, scene_points=None) -> go.Figure:
    """One Scatter3d trace per object plus an optional gray scene-context
    trace; axes follow the camera frame (X right, Z forward, -Y up)."""
    fig = go.Figure()
    if scene_points is not None and len(scene_points):
        pts = scene_points
        if len(pts) > MAX_SCENE_POINTS:
            idx = np.random.default_rng(0).choice(len(pts), MAX_SCENE_POINTS,
                                                  replace=False)
            pts = pts[idx]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1],
            mode="markers",
            marker=dict(size=1.0, color="rgba(128,128,128,0.35)"),
            name="scene context", hoverinfo="skip"))
    total = sum(len(o.points) for o in objects) or 1
    for i, obj in enumerate(objects):
        pts = obj.points
        budget = max(500, int(MAX_PLOT_POINTS * len(pts) / total))
        if len(pts) > budget:
            idx = np.random.default_rng(0).choice(len(pts), budget, replace=False)
            pts = pts[idx]
        r, g, b = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1],
            mode="markers",
            marker=dict(size=1.6, color=f"rgb({r},{g},{b})"),
            name=f"{i + 1}: {obj.label}",
            hovertemplate=f"{i + 1}: {obj.label}<extra></extra>"))
    fig.update_layout(
        scene=dict(xaxis_title="X, m", yaxis_title="Z (depth), m",
                   zaxis_title="-Y (up), m", aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0), height=560)
    return fig


def _table(objects) -> pd.DataFrame:
    rows = []
    for i, obj in enumerate(objects):
        m = obj.metrics
        rows.append({
            "#": i + 1, "class": obj.label, "score": round(obj.score, 2),
            "distance, m": round(m["distance_m"], 2),
            "size L×W×H, m": " × ".join(
                f"{m[f'obb_extent_{k}']:.2f}" for k in (1, 2, 3)),
            "OBB volume, m³": round(m["obb_volume_m3"], 3),
            "hull volume, m³": round(m["hull_volume_m3"], 3),
            "points": m["n_points"]})
    return pd.DataFrame(rows)


def analyse(image_path: str | None, conf: float, show_context: bool):
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
    objects, scene_points = PIPELINE.process(
        rgb, conf=conf, intrinsics=intr, scene_stride=4 if show_context else None)
    dt = time.time() - t0

    if not objects:
        status = (f"No objects of the 10 supported classes found at "
                  f"confidence ≥ {conf:.2f} ({dt:.1f} s). Try a lower threshold.")
        return rgb, _cloud_figure([], scene_points), pd.DataFrame(), status

    status = (f"{len(objects)} object(s) in {dt:.1f} s on {PIPELINE.device} — "
              f"{intr_note}. Depth is monocular (global scale alignment): "
              f"shapes and layout are reliable, absolute sizes are approximate "
              f"(thesis: |volume error| ≈ 50%, usable range < 3.5 m).")
    return (_overlay(rgb, objects), _cloud_figure(objects, scene_points),
            _table(objects), status)


with gr.Blocks(title="RGB → 3D object analysis") as demo:
    gr.Markdown(
        "# From a single RGB photo to 3D object characteristics\n"
        "Mask R-CNN (SUN RGB-D, 10 indoor classes) → Depth Anything V2 "
        "(monocular depth, global scale alignment) → per-object point clouds "
        "(Open3D) → spatial characteristics. No depth sensor required.")
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="Indoor scene (RGB)")
            conf_in = gr.Slider(0.3, 0.9, value=0.5, step=0.05,
                                label="Detection confidence threshold")
            context_in = gr.Checkbox(value=True,
                                     label="Show scene context in 3D (gray cloud)")
            run_btn = gr.Button("Analyse scene", variant="primary")
            gr.Examples(
                examples=sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")),
                inputs=image_in, label="SUN RGB-D test scenes (never seen in training)")
            status_out = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Tab("Instance masks"):
                masks_out = gr.Image(label="Detected objects")
            with gr.Tab("3D point cloud"):
                cloud_out = gr.Plot(label="Per-object clouds (camera frame)")
            with gr.Tab("Object characteristics"):
                table_out = gr.Dataframe(label="Per-object 3D characteristics")
    run_btn.click(analyse, inputs=[image_in, conf_in, context_in],
                  outputs=[masks_out, cloud_out, table_out, status_out])

if __name__ == "__main__":
    demo.launch()
