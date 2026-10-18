# Demo examples

Four SUN RGB-D scenes, one per sensor, all from the **test split of dataset v2**
(`notebooks/outputs/splits/instances_test.json`; none of them, nor any frame
of their capture groups, was used for training or model selection). The files
are byte-exact copies of the dataset RGB frames; `intrinsics.json` holds each
scene's exact calibration (read from the scene's `intrinsics.txt` with
`utils.sunrgbd.read_intrinsics`). The app matches an input to this table by
the SHA-1 of the file bytes, so a re-encoded or edited copy falls back to the
EXIF / default intrinsics.

| file | image_id | sensor | scene type | GT objects | demo-model TP / FP / FN | per-image F1 | dataset path (relative to the dataset root) |
|---|---|---|---|---|---|---|---|
| `kv1_dining_room.jpg` | 1700 | kv1 | dining_room | 5 (chair, table) | 5 / 0 / 0 | 1.00 | `kv1/b3dodata/img_0313/image/img_0313.jpg` |
| `kv2_furniture_store.jpg` | 3964 | kv2 | furniture_store (bedroom set) | 4 (bed, dresser, nightstand) | 4 / 0 / 0 | 1.00 | `kv2/kinect2data/001735_2014-06-26_19-46-05_260595134347_rgbf000166-resize/image/0000166.jpg` |
| `xtion_classroom.jpg` | 10256 | xtion | classroom | 4 (chair, table) | 4 / 1 / 0 | 0.89 | `xtion/xtion_align_data/2014_12_18_15_50_8/image/0000099.jpg` |
| `realsense_rest_space.jpg` | 6811 | realsense | rest_space | 6 (chair, table) | 5 / 0 / 1 | 0.91 | `realsense/shr/2014_12_18-11_49_13-1311000073/image/0000063.jpg` |

## Selection rule

The examples are meant to show what the pipeline does when segmentation works,
so they were chosen among the scenes where the demo model performs well. They
are **not** representative of average accuracy: the median per-image F1 over
the candidate pool below is 0.56, and the model's test mask AP is 0.261.

1. **Pool:** test-split images with ≥ 4 GT objects, ≥ 2 classes, and ≥ 80 % of
   the GT objects within 4 m (per-object median *sensor* depth) — monocular 3D
   is reliable up to ≈ 3.5 m. 325 images passed (kv1 37, kv2 141, xtion 100,
   realsense 47).
2. **Score:** per-image F1 of the demo model (torchvision Mask R-CNN v2,
   `~/Downloads/v2_maskrcnn_torchvision_std10/test_predictions.json`) at
   confidence ≥ 0.5, with the matching protocol of `utils/evaluate_predictions.py`
   (greedy by score, class-agnostic mask IoU ≥ 0.5, a misclassified match counts
   as FP + FN). Ranked by F1, then by number of GT objects.
3. **Visual check:** the top 8–10 per sensor were run through the demo
   pipeline (`metric_scaled` backend) and their overlays inspected. Rejected:
   scenes with a merged or duplicate mask on screen (e.g. xtion 10071, rank 1:
   a 0.61 "chair" blob spanning three chairs; xtion 8386 / 10143, realsense 5964),
   and xtion 10107, which shows the same building and session as realsense 6811.
   Among the rest, different room types were preferred.

Top 5 per sensor (chosen rows in bold):

| sensor | image_id | scene type | GT | classes | TP/FP/FN | F1 |
|---|---|---|---|---|---|---|
| kv1 | **1700** | dining_room | 5 | chair, table | 5/0/0 | 1.00 |
| kv1 | 40 | office | 5 | chair, table | 4/0/1 | 0.89 |
| kv1 | 1719 | dining_room | 4 | chair, table | 4/2/0 | 0.80 |
| kv1 | 1395 | dining_room | 7 | chair, table | 6/3/1 | 0.75 |
| kv1 | 1869 | dining_room | 4 | chair, table | 3/1/1 | 0.75 |
| kv2 | 2538 | furniture_store | 4 | bookshelf, chair, table | 4/0/0 | 1.00 |
| kv2 | 3704 | rest_space | 4 | chair, table | 4/0/0 | 1.00 |
| kv2 | **3964** | furniture_store | 4 | bed, dresser, nightstand | 4/0/0 | 1.00 |
| kv2 | 2015 | rest_space | 4 | chair, table | 4/0/0 | 1.00 |
| kv2 | 2938 | office | 7 | chair, table | 7/1/0 | 0.93 |
| xtion | 10071 | conference_room | 10 | chair, table | 9/1/1 | 0.90 |
| xtion | **10256** | classroom | 4 | chair, table | 4/1/0 | 0.89 |
| xtion | 8390 | conference_room | 4 | chair, table | 4/1/0 | 0.89 |
| xtion | 10143 | rest_space | 7 | chair, table | 7/2/0 | 0.88 |
| xtion | 8385 | conference_room | 10 | chair, table | 9/2/1 | 0.86 |
| realsense | **6811** | rest_space | 6 | chair, table | 5/0/1 | 0.91 |
| realsense | 5964 | rest_space | 4 | chair, table | 4/1/0 | 0.89 |
| realsense | 6351 | discussion_area | 4 | chair, table | 3/0/1 | 0.86 |
| realsense | 6726 | study_space | 4 | chair, sofa, table | 3/0/1 | 0.86 |
| realsense | 6739 | study_space | 4 | chair, sofa, table | 3/0/1 | 0.86 |

kv2 3964 was taken from the four kv2 scenes tied at F1 1.00 because it is the
only one with bedroom classes (bed, dresser, nightstand); the other three
examples show chairs and tables. The xtion classroom has three objects beyond
3.5 m, so it also shows the "≈ approximate" flag for far objects.

The earlier example set (`kv1_classroom`, `kv2_library`, `xtion_classroom` —
different frames — and `realsense_rest_space` = test image 6728) was replaced:
the first three are in the v2 **train** split. Image 6728 is in the test split
but fails the 4 m rule (71 % of its objects within 4 m), has F1 0.50, and the
model misses the large chair in the foreground.
