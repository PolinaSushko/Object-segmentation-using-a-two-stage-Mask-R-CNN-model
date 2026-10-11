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
python app/export_weights.py ~/Downloads/maskrcnn_torchvision_std10/best.pth \
    app/weights/maskrcnn_std10.pt
python app/app.py              # finds app/weights/maskrcnn_std10.pt automatically
```

The app imports `utils/` (the shared point-cloud pipeline of the thesis) from
the repository root; in a Space it is a copy next to `app.py` (see below).

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
* Thesis evaluation of the default backend (notebook 08, predicted masks):
  median Chamfer 15.8 cm, OBB volume bias −7%, median |volume error| 53%;
  usable range ≈ 3.5 m — farther objects are marked ≈ (approximate).
