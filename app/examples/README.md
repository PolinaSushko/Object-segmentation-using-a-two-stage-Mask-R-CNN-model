# Demo examples

Four SUN RGB-D scenes, one per sensor, all from the **test split of dataset v2**
(`notebooks/outputs/splits_min400/instances_test.json`; all four are in the
std10 test file as well; none of them, nor any frame of their capture groups,
was used for training or model selection). The files are byte-exact copies of
the dataset RGB frames; `intrinsics.json` holds each scene's exact calibration
(the `fx, fy, cx, cy` of its COCO image record, read from the scene's
`intrinsics.txt`). The app matches an input to this table by the SHA-1 of the
file bytes, so a re-encoded or edited copy falls back to the EXIF / default
intrinsics.

| file | image_id | sensor | scene type | GT objects | demo-model TP / FP / FN | per-image F1 | dataset path (relative to the dataset root) |
|---|---|---|---|---|---|---|---|
| `kv1_dining_room.jpg` | 1700 | kv1 | dining_room | 5 (chair, table) | 5 / 0 / 0 | 1.00 | `kv1/b3dodata/img_0313/image/img_0313.jpg` |
| `kv2_bedroom.jpg` | 4357 | kv2 | bedroom | 7 (bed, lamp, nightstand, pillow) | 6 / 0 / 1 | 0.92 | `kv2/kinect2data/002156_2014-06-27_23-54-19_260595134347_rgbf000104-resize/image/0000104.jpg` |
| `xtion_lounge.jpg` | 10137 | xtion | idk (unlabelled; a lounge corner) | 4 (plant, sofa, table) | 3 / 0 / 1 | 0.86 | `xtion/xtion_align_data/2014_12_18_12_5_29/image/0000025.jpg` |
| `realsense_rest_space.jpg` | 6756 | realsense | rest_space | 4 (chair, lamp, table) | 4 / 0 / 0 | 1.00 | `realsense/sa/2014_10_24-21_17_38-1311000073/image/0000065.jpg` |

## Selection rule

The examples are meant to show what the pipeline does when segmentation works,
so they were chosen among the scenes where the demo model performs well. They
are **not** representative of average accuracy: the median per-image F1 over
the candidate pool below is 0.50, and the model's test mask AP is 0.226 on
its 20 classes.

1. **Pool:** test-split images (20-class min400 GT) with ≥ 4 GT objects,
   ≥ 2 classes, and ≥ 80 % of the GT objects within 4 m (per-object median
   *sensor* depth), because monocular 3D is reliable up to ≈ 3.5 m.
   537 images passed (kv1 93, kv2 221, xtion 162, realsense 61).
2. **Score:** per-image F1 of the demo model (torchvision Mask R-CNN v2,
   20 classes, `~/Downloads/v2_maskrcnn_torchvision_min400/test_predictions.json`)
   at confidence ≥ 0.5, with the matching protocol of
   `utils/evaluate_predictions.py`. Detections are matched greedily by score
   at class-agnostic mask IoU ≥ 0.5, and a misclassified match counts as
   FP + FN. Scenes are ranked by F1, then by the number of GT objects.
3. **Visual check:** the top candidates per sensor were run through the demo
   pipeline (`metric_scaled` backend, all three backends for the final four).
   Their overlays were inspected, and the monocular distances were compared
   with the sensor distances of the same masks. A scene was rejected if it
   had a merged or duplicate mask on screen, or if the 3D result was clearly
   broken. Among the rest, scenes with room types and classes that the other
   examples do not show were preferred. The examples should also show the
   classes beyond the standard ten.

Top 5 per sensor (chosen rows in bold):

| sensor | image_id | scene type | GT | classes | TP/FP/FN | F1 |
|---|---|---|---|---|---|---|
| kv1 | **1700** | dining_room | 5 | chair, table | 5/0/0 | 1.00 |
| kv1 | 1685 | dining_room | 4 | chair, table | 3/0/1 | 0.86 |
| kv1 | 1610 | office | 8 | chair, display, keyboard, lamp, table | 7/3/1 | 0.78 |
| kv1 | 28 | office | 4 | garbage bin, sofa, table | 3/1/1 | 0.75 |
| kv1 | 672 | bathroom | 4 | bathtub, cabinet, garbage bin, sink | 3/1/1 | 0.75 |
| kv2 | 3718 | conference_room | 7 | chair, table | 7/0/0 | 1.00 |
| kv2 | 2041 | conference_room | 4 | bookshelf, cabinet, garbage bin, table | 4/0/0 | 1.00 |
| kv2 | **4357** | bedroom | 7 | bed, lamp, nightstand, pillow | 6/0/1 | 0.92 |
| kv2 | 2071 | rest_space | 6 | chair, plant, sofa, table | 6/1/0 | 0.92 |
| kv2 | 3704 | rest_space | 5 | chair, garbage bin, table | 5/1/0 | 0.91 |
| xtion | 8387 | conference_room | 4 | chair, table | 4/1/0 | 0.89 |
| xtion | 10071 | conference_room | 10 | chair, table | 9/2/1 | 0.86 |
| xtion | **10137** | idk | 4 | plant, sofa, table | 3/0/1 | 0.86 |
| xtion | 10238 | classroom | 4 | chair, table | 3/0/1 | 0.86 |
| xtion | 8385 | conference_room | 10 | chair, table | 8/1/2 | 0.84 |
| realsense | **6756** | rest_space | 4 | chair, lamp, table | 4/0/0 | 1.00 |
| realsense | 6757 | library | 4 | chair, table | 4/0/0 | 1.00 |
| realsense | 6811 | rest_space | 6 | chair, table | 5/0/1 | 0.91 |
| realsense | 6347 | discussion_area | 4 | chair, table | 4/1/0 | 0.89 |
| realsense | 6912 | classroom | 6 | chair, table | 6/2/0 | 0.86 |

* **kv1 1700** ranks first and is kept from the previous example set.
* **kv2 4357** (rank 3) was taken over the two scenes with F1 1.00. It is the
  only bedroom in the set, with four classes, two of them beyond std10 (lamp,
  pillow); 3718 shows only chairs and tables, and 2041 is a sparse corner
  of a conference room.
* **xtion 10137** (rank 3): 8387 and 10071 show only chairs and tables, like
  kv1 1700; 10137 adds a sofa and a plant. The plant is 3.7 m away, so the
  scene also shows the "≈ approximate" flag for far objects.
* **realsense 6756** ranks first (tied with 6757). 6757 was rejected because
  monocular depth halves its distances: the chairs come out at 1.2–1.4 m
  against 2.9–3.3 m from the sensor. One of its objects is also too sparse to
  give a cloud. 6756 comes from a different capture session (October 2014)
  than xtion 10137. 6811, the previous realsense example, would have been a
  second view of the same place: the realsense frames 6811–6833 and the
  xtion frames 10090–10149, 10137 among them, were captured in one session
  (2014-12-18, 11:49–12:13).

## History

The previous example set (10-class demo model) was `kv1_dining_room` (1700),
`kv2_furniture_store` (3964), `xtion_classroom` (10256) and
`realsense_rest_space` (6811), chosen with the same rule. With the 20-class
model and GT, 3964 dropped to rank 57 among the kv2 candidates (F1 0.67: its
lamp and pillows add GT objects and detections), and 10256 dropped to rank 8.
The set before that (`kv1_classroom`, `kv2_library`, an older
`xtion_classroom` and test image 6728) was replaced because the first three
are in the v2 **train** split.
