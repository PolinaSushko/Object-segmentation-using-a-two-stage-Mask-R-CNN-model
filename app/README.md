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

Single RGB photo → Mask R-CNN instance masks (SUN RGB-D, 20 indoor classes) →
Depth Anything V2 monocular depth (no sensor) → per-object point clouds
(Open3D) → distance, oriented-box size and volume of every object.

## Run locally (repository layout)

```bash
# once: training checkpoint (model + optimizer, 367 MB) -> weights-only file (176 MB)
python app/export_weights.py ~/Downloads/v2_maskrcnn_torchvision_min400/best.pth \
    app/weights/maskrcnn_min400.pt \
    --note "dataset v2, run v2_maskrcnn_torchvision_min400 (20 classes, min400)"
python app/app.py              # finds app/weights/maskrcnn_min400.pt automatically
```

The app imports `utils/` (the shared point-cloud pipeline of the thesis) from
the repository root; in a Space it is a copy next to `app.py` (see below).

## Segmentation model

`weights/maskrcnn_min400.pt` is **Mask R-CNN ResNet-50-FPN v2 (torchvision)
trained on SUN RGB-D dataset v2**, 20 classes — the 10 standard ones (bed,
chair, sofa, table, desk, toilet, dresser, nightstand, bookshelf, bathtub)
plus cabinet, pillow, display, shelf, lamp, garbage bin, computer, keyboard,
sink, plant (the min400 configuration: every class with ≥ 400 instances; run
`v2_maskrcnn_torchvision_min400`, notebook 02, best epoch 11). Its `meta`
records the source checkpoint, its SHA-1 and the validation mask AP (0.2121
on the 20 classes). Test mask AP: 0.2258 on the 20 classes, 0.2663 on the 10
standard classes (notebook 10). The weakest classes are shelf (0.02), desk
(0.06) and computer (0.08); toilet (0.67), garbage bin and keyboard are the
best.

Why torchvision and not the thesis's selected model: the model selected by
validation mask AP on the 10 standard classes is Mask R-CNN Detectron2
R101-FPN trained on the same 20 classes (val 0.2570, test 0.2665 on the std10
classes). The torchvision model is equal on the 10-class test set — Δ test
mask AP 0.0002, paired bootstrap p = 0.98; on the 20 classes Δ 0.006, p = 0.21
(notebook 10) — and it keeps the demo pure PyTorch: `pip install torch
torchvision` on a CPU Space, no Detectron2 build. The previous 10-class demo
weights (`maskrcnn_std10.pt`, run `v2_maskrcnn_torchvision_std10`) still load
through `MASKRCNN_CKPT`: the class names are read from the weights file.

The end-to-end 3D numbers below are notebook 08's, measured with the selected
model's predictions over all 20 classes (min400 test split). Segmentation error
costs under 1 cm Chamfer there, so for an equivalent segmenter they describe the
demo as well. The 10 additional classes (mostly smaller objects: lamps,
pillows, displays, keyboards, plants) come out worse than the furniture with
monocular depth: 18.8 vs 15.1 cm median Chamfer for the default backend. They lie
farther away, and the Small depth model is weak on desk-top electronics
(display 29 cm, keyboard 25 cm) and lamps (24 cm).

The bundled `examples/` are four test-split scenes (one per sensor); how they
were chosen is documented in `examples/README.md`.

## Configuration (environment variables)

| variable | meaning |
|---|---|
| `MASKRCNN_CKPT` | local weights file written by `export_weights.py` |
| `MASKRCNN_HF_REPO` | otherwise: Hugging Face model repo holding that file (downloaded with `hf_hub_download`; `HF_TOKEN` for a private repo) |
| `MASKRCNN_HF_FILENAME` | file name in that repo (default `maskrcnn_min400.pt`) |
| `DEPTH_BACKEND` | `metric_scaled` (default) \| `metric_focal` \| `relative_global` |
| `TORCH_NUM_THREADS` | CPU threads (default: CPUs available to the container) |

Without `MASKRCNN_CKPT` / `MASKRCNN_HF_REPO` the app looks for
`weights/maskrcnn_min400.pt` next to `app.py`. The weights are loaded with
`torch.load(..., weights_only=True)`; a raw training checkpoint is rejected
with the conversion command.

## Hugging Face Space layout

The Space root is the content of `app/` plus a copy of the repository's `utils/`:

```
app.py  pipeline.py  export_weights.py  requirements.txt  packages.txt
README.md  examples/  utils/  (weights/maskrcnn_min400.pt — optional)
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
* Thesis evaluation per depth backend (notebook 08, dataset v2, min400 test
  split, predicted masks of the selected Mask R-CNN, 3586 matched objects of
  all 20 classes; medians against GT mask + sensor depth):

  | `DEPTH_BACKEND` | median Chamfer: 20 classes (10 standard / 10 extra) | OBB volume bias | median \|volume error\| |
  |---|---|---|---|
  | `metric_scaled` (default) | 15.8 cm (15.1 / 18.8) | −14% | 49% |
  | `metric_focal` | 21.7 cm (20.4 / 27.0) | −30% | 59% |
  | `relative_global` | 28.0 cm (25.0 / 37.4) | −63% | 71% |

  The thesis' main table (10 standard classes, std10 test split, 2580 objects)
  gives 15.2 / 20.4 / 25.0 cm. For reference: the same masks on sensor depth
  give 0.8 cm, i.e. the error is almost entirely the monocular depth. Usable
  range ≈ 3.5 m (median Chamfer of the default backend 13 cm below 2.5 m,
  20 cm at 2.5–3.5 m, 28–49 cm beyond) — farther objects are marked ≈
  (approximate).
