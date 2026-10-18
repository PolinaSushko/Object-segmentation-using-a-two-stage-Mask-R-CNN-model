---
title: RGB to 3D Object Analysis
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.25.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Mask R-CNN + monocular depth -> 3D object characteristics
---

# RGB → 3D object analysis (web demo)

Single RGB photo → Mask R-CNN instance masks (SUN RGB-D, 10 indoor classes) →
Depth Anything V2 monocular depth (no sensor) → per-object point clouds
(Open3D) → distance, oriented-box size and volume of every object.

## Run locally (repository layout)

```bash
# once: training checkpoint (model + optimizer, 367 MB) -> weights-only file (176 MB)
python app/export_weights.py ~/Downloads/v2_maskrcnn_torchvision_std10/best.pth \
    app/weights/maskrcnn_std10.pt \
    --note "dataset v2 (2026-10-11), std10; val mask AP 0.2459, test mask AP 0.2609"
python app/app.py              # finds app/weights/maskrcnn_std10.pt automatically
```

The app imports `utils/` (the shared point-cloud pipeline of the thesis) from
the repository root; in a Space it is a copy next to `app.py` (see below).

## Segmentation model

`weights/maskrcnn_std10.pt` is **Mask R-CNN ResNet-50-FPN v2 (torchvision)
trained on SUN RGB-D dataset v2**, standard 10 classes (run
`v2_maskrcnn_torchvision_std10`, notebook 02, best epoch 10). Its `meta`
records the source checkpoint, its SHA-1 and the validation mask AP (0.2459).

Why torchvision and not the thesis's selected model: the model selected by
validation mask AP on the 10 standard classes is Mask R-CNN Detectron2
R101-FPN trained on the 20 min400 classes (val 0.2570, test 0.2665 on the
std10 classes). The torchvision model is statistically equivalent on the test
split — Δ test mask AP 0.006 (0.2609), paired bootstrap p = 0.54 (notebook 10)
— and it keeps the demo pure PyTorch: `pip install torch torchvision` on a CPU
Space, no Detectron2 build. The end-to-end 3D numbers below are notebook 08's,
measured with the selected model's predictions; segmentation error costs under
1 cm Chamfer there, so for an equivalent segmenter they describe the demo as
well.

The bundled `examples/` are four test-split scenes (one per sensor); how they
were chosen is documented in `examples/README.md`.

## Configuration (environment variables)

| variable | meaning |
|---|---|
| `MASKRCNN_CKPT` | local weights file written by `export_weights.py` |
| `MASKRCNN_HF_REPO` | otherwise: Hugging Face model repo holding that file (downloaded with `hf_hub_download`; `HF_TOKEN` for a private repo) |
| `MASKRCNN_HF_FILENAME` | file name in that repo (default `maskrcnn_std10.pt`) |
| `DEPTH_BACKEND` | `metric_scaled` (default) \| `metric_focal` \| `relative_global` |
| `TORCH_NUM_THREADS` | CPU threads (default: CPUs available to the container) |

Without `MASKRCNN_CKPT` / `MASKRCNN_HF_REPO` the app looks for
`weights/maskrcnn_std10.pt` next to `app.py`. The weights are loaded with
`torch.load(..., weights_only=True)`; a raw training checkpoint is rejected
with the conversion command.

## Hugging Face Space layout

The Space root is the content of `app/` plus a copy of the repository's `utils/`:

```
app.py  pipeline.py  export_weights.py  requirements.txt  packages.txt
README.md  examples/  utils/  (weights/maskrcnn_std10.pt — optional)
```

`pipeline.py` finds `utils/` either next to itself (Space) or one level up
(repository). Put the weights in a model repo and set `MASKRCNN_HF_REPO` in
the Space settings, or commit them under `weights/` with Git LFS.
`requirements.txt` pins the verified versions and pulls CPU-only PyTorch;
`packages.txt` adds the system libraries Open3D needs.

## What the numbers mean

* Inputs are processed at SUN RGB-D-like resolution (long side 561–730 px
  natively, otherwise resized to 640 px) because the mask-refinement filters
  work in pixels; intrinsics are rescaled with the image.
* Intrinsics: exact sensor calibration for the bundled examples (matched by
  file content), else the EXIF 35 mm-equivalent focal length, else an
  explicit Kinect-like assumption (fx = 0.925 × long side). The status line
  states which one was used.
* Thesis evaluation per depth backend (notebook 08, dataset v2, test split,
  predicted masks of the selected Mask R-CNN, 2580 matched objects; medians
  against GT mask + sensor depth):

  | `DEPTH_BACKEND` | median Chamfer | OBB volume bias | median \|volume error\| |
  |---|---|---|---|
  | `metric_scaled` (default) | 15.2 cm | −12% | 49% |
  | `metric_focal` | 20.4 cm | −28% | 58% |
  | `relative_global` | 25.0 cm | −61% | 69% |

  For reference: the same masks on sensor depth give 0.8 cm, i.e. the error
  is almost entirely the monocular depth. Usable range ≈ 3.5 m (median Chamfer
  of the default backend 12 cm below 2.5 m, 19 cm at 2.5–3.5 m, 28–49 cm
  beyond) — farther objects are marked ≈ (approximate).
