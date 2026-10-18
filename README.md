# Instance segmentation with a two-stage Mask R-CNN on SUN RGB-D and 2D → 3D object representation

Master's thesis project. A Mask R-CNN model segments indoor furniture
instances in RGB images of [SUN RGB-D](https://rgbd.cs.princeton.edu/); its masks
are combined with depth (sensor or monocular) to back-project each object into a
point cloud and measure its spatial properties (position, distance, oriented
bounding box, size, volume). A Gradio web demo runs the whole pipeline on a
single RGB photo.

```
RGB ──► Mask R-CNN ──► instance masks ─┐
  │                                    ├─► mask refinement ─► back-projection ─► point cloud ─► 3D metrics
  └──► depth (sensor | Depth Anything V2) ┘     (pinhole intrinsics)            (Open3D)       (OBB, distance, volume)
```

## Results (dataset v2, test split: 1,287 images, 5,043 instances, 10 classes)

| Model | val mask AP | test mask AP [95% CI] | AP50 | macro P / R / F1 @0.5 | T4 ms/img |
|---|---|---|---|---|---|
| **Mask R-CNN, Detectron2 R101-FPN, trained on 20 classes²** (selected by val AP) | **0.257** | **0.267** [0.246, 0.302] | 0.427 | 0.49 / 0.46 / 0.46 | 224 |
| Mask R-CNN, Detectron2 R101-FPN | 0.248 | 0.263 [0.243, 0.297] | **0.431** | 0.46 / 0.50 / 0.47 | — |
| Mask R-CNN, torchvision R50-FPN v2 (demo model) | 0.245 | 0.261 [0.242, 0.294] | 0.426 | 0.37 / 0.51 / 0.42 | 231 |
| Mask R-CNN, Detectron2 R50-FPN | 0.244 | 0.247 [0.226, 0.283] | 0.405 | 0.47 / 0.46 / 0.46 | 183 |
| YOLOv8s-seg | 0.241 | 0.243 [0.222, 0.277] | 0.391 | 0.63 / 0.33 / 0.42 | 22 |
| SAM ViT-B + GT-box prompts¹ | — | 0.398 (class-agnostic) | 0.753 | — | — |

¹ Not directly comparable: ground-truth boxes as prompts, class-agnostic — an upper
bound of mask quality given perfect detection.
² Trained on the `min400` configuration (the 10 standard classes + 10 more frequent
classes, ids 1–10 shared) and scored on the 10 standard classes after its predictions
of the extra classes are dropped. The class configuration is treated as a
hyperparameter, so the model is selected by the same rule (validation mask AP on the
10-class validation split). On its own 20 classes it reaches test mask AP 0.232
(val 0.224).

Paired bootstrap over test images (notebook 10): selected Mask R-CNN − YOLOv8s-seg
ΔAP = +0.024 [0.002, 0.045], p = 0.03; ΔAP50 = +0.037 [0.010, 0.064], p = 0.006.
Among the Mask R-CNN variants no test difference is significant: 20 vs 10 training
classes (same R101) ΔAP = +0.004, p = 0.66; Detectron2 vs torchvision ΔAP = +0.002,
p = 0.76; R101 vs R50 ΔAP = +0.016, p = 0.09 (significant only for AP50).
Mask R-CNN finds more objects (macro recall 0.46 vs 0.33); YOLO is ~10× faster and
draws slightly tighter masks (mean matched IoU 0.815 vs 0.790).

Error analysis of the selected model (notebook 09, TIDE): localization (ΔAP 6.1)
and background false positives (5.8) dominate, then missed objects (4.6) and class
confusion (2.6, mostly desk ↔ table and chair ↔ sofa). A visual audit showed that
most confident "background" detections are real but unannotated furniture — the
absolute AP is limited by annotation completeness of SUN RGB-D.

Monocular depth for the demo (notebook 06): Depth Anything V2 Metric-Indoor Small
with one global scale constant — AbsRel 0.233, δ<1.25 0.766 against sensor depth.

End-to-end 3D with the selected model's masks (notebook 08, 2,580 matched test
objects, median Chamfer distance to the GT-mask + sensor-depth reference): 0.8 cm
with sensor depth — segmentation error is almost free in 3D; with monocular depth
15.2 cm for the demo backend (metric Small + global scale, OBB volume bias −12%),
13.2 cm for metric Base, 25.0 cm for the relative model with a global alignment and
8.9 cm for the per-image oracle alignment.

## Repository layout

| Path | Content |
|---|---|
| `utils/sunrgbd.py` | SUN RGB-D utilities: scene discovery, depth decoding (bit shift), per-scene intrinsics, annotation parsing, class-name normalization, polygon clipping, leak-free scene grouping, back-projection |
| `utils/prepare_dataset.py` | CLI: polygons → COCO, stratified group split, leak check, dataset version file |
| `utils/pack_for_colab.py` | packs the images/depth of the selected splits into one zip for Colab |
| `utils/evaluate_predictions.py` | the single evaluator for every model: COCO AP, per-class AP, P/R/F1, confusion matrix, mean matched IoU (`--agnostic` for SAM) |
| `utils/error_analysis.py` | TIDE error decomposition, bootstrap CIs, paired tests, galleries |
| `utils/coco_to_yolo.py` | COCO splits → YOLO-seg format |
| `utils/sam_gt_boxes.py` | SAM zero-shot baseline with GT-box prompts |
| `utils/pointcloud.py` | the shared point-cloud pipeline: mask refinement, back-projection, voxel + outlier filtering, 3D metrics, Chamfer distance |
| `notebooks/01_data_preparation.ipynb` | dataset overview, depth decoding, annotation parsing, clipping, leak-free grouping, split tables |
| `notebooks/01b_class_analysis.ipynb` | class vocabulary analysis, name normalization, extended class configs |
| `notebooks/02–04_*.ipynb` | training on Colab: Mask R-CNN (torchvision), Mask R-CNN (Detectron2), YOLOv8s-seg — one shared prediction and timing protocol |
| `notebooks/05_pointclouds_sensor.ipynb` | masks + sensor depth → point clouds → 3D object metrics |
| `notebooks/06_monocular_depth.ipynb` | monocular depth (Depth Anything V2): scale alignment methods, metric models |
| `notebooks/07_eval3d_mono_vs_sensor.ipynb` | 3D reconstruction from monocular vs sensor depth |
| `notebooks/08_eval3d_pred_masks.ipynb` | end-to-end 3D with predicted masks and six depth backends |
| `notebooks/09_error_analysis.ipynb` | error analysis of the selected model |
| `notebooks/10_model_comparison.ipynb` | final comparison table, significance tests, training curves, qualitative examples |
| `notebooks/outputs/` | generated splits, tables and figures |
| `app/` | Gradio web demo (see `app/README.md`) |

## Data

SUN RGB-D (10,335 RGB-D frames from 4 sensors: Kinect v1/v2, Xtion, RealSense) is
not included; download it from the official page and point `--root` to it. File
paths inside the COCO files are relative to the dataset root.

Dataset v2 used for all reported results:

- **Classes:** the standard 10 SUN RGB-D classes (bed, chair, sofa, table, desk,
  toilet, dresser, nightstand, bookshelf, bathtub); `min400` (20 classes) is an
  extended configuration for the class-count experiment.
- **Masks:** annotation polygons are clipped to the image (Sutherland–Hodgman);
  compound/misspelled raw names (roundtable, loveseat, tabel, …) are mapped to
  their class (`notebooks/outputs/class_analysis/name_recovery_v2.csv`).
- **Leak-free split:** frames are grouped before splitting — SUN3D by site,
  timestamped frames by capture session across sensors, Kinect v1 near-duplicates
  by feature matching — then a 70/15/15 group split stratified by class and sensor
  (seed 42). `notebooks/outputs/splits/leak_check.txt` documents zero leaks.

## Reproducing

Local environment: Python 3.11, `pip install -r notebooks/requirements.txt`
(plus the PyTorch packages listed there for notebook 06 and SAM).

```bash
# 1. COCO annotations + leak-free split (std10), then the extended config
python utils/prepare_dataset.py --root /path/to/SUNRGBD --out notebooks/outputs \
    --classes standard10
python utils/prepare_dataset.py --coco notebooks/outputs/sunrgbd_instances_all.json \
    --out notebooks/outputs --min-instances 400 --splits-name splits_min400 \
    --base-assignment notebooks/outputs/splits/split_assignment.csv

# 2. archive for Colab (upload to Google Drive)
python utils/pack_for_colab.py --root /path/to/SUNRGBD \
    --splits notebooks/outputs/splits notebooks/outputs/splits_min400 --out sunrgbd_data.zip
```

3. Train on Google Colab (T4): notebooks 02 (torchvision), 03 (Detectron2,
   `BACKBONE` switch) and 04 (YOLOv8s-seg). Set `DRIVE_DIR` to your Drive folder.
   Training resumes automatically after a session drop; each run folder
   (`vkr_runs/v2_*`) receives `val_predictions.json`, `test_predictions.json`,
   `timing.json` and logs.
4. Download the run folders to `~/Downloads/` (or set `VKR_RUNS_ROOT`) and run
   notebooks 10 and 09 locally.
5. 3D part: notebook 06 (writes monocular depth maps of the test split), then
   05, 07 and 08.

Any single predictions file can be evaluated with the shared evaluator:

```bash
python utils/evaluate_predictions.py --gt notebooks/outputs/splits/instances_test.json \
    --pred test_predictions.json --out eval_dir --name my_model
```

## Web demo

```bash
python app/export_weights.py <best.pth> app/weights/maskrcnn_std10.pt   # model-only weights
python app/app.py
```

Upload an RGB photo (or pick one of the bundled test-split examples): the demo
shows instance masks, an interactive 3D point cloud with oriented bounding boxes
and a table of object properties. Details, depth backends and accuracy notes —
`app/README.md`.
