"""
coco_to_yolo.py — converts the prepared COCO splits to the YOLO segmentation
format (for YOLOv8-seg fine-tuning).

Output layout (Ultralytics convention):
    <out>/
      data.yaml                  # dataset config for `model.train(data=...)`
      images/{train,val,test}/   # symlinks (default) or copies of the RGB images
      labels/{train,val,test}/   # one .txt per image: "cls x1 y1 x2 y2 ..." in [0,1]
      image_mapping.csv          # split, image_id, original file_name

Example:
    python coco_to_yolo.py \
        --split-dir notebooks/outputs/splits \
        --root "/path/to/SUNRGBD" \
        --out yolo_std10 --mode symlink
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

SUBSETS = ("train", "val", "test")


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace area of an (N, 2) polygon."""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _splice(polygons: list[np.ndarray]) -> np.ndarray:
    """Splices disjoint (N, 2) polygons into one via zero-width bridges at the
    closest pair of vertices."""
    parts = sorted(polygons, key=_polygon_area, reverse=True)
    merged = parts[0]
    for cand in parts[1:]:
        d2 = ((merged[:, None, :] - cand[None, :, :]) ** 2).sum(axis=-1)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        merged = np.concatenate(
            [merged[: i + 1], cand[j:], cand[: j + 1], merged[i:]])
    return merged


def _polygon_iou(pts: np.ndarray, target_rle: dict, width: int, height: int) -> float:
    """IoU between a rasterized (N, 2) polygon and a target RLE mask."""
    flat = np.clip(pts, [0, 0], [width - 1e-3, height - 1e-3]).reshape(-1).tolist()
    rle = mask_utils.merge(mask_utils.frPyObjects([flat], height, width))
    return float(mask_utils.iou([rle], [target_rle], [0])[0][0])


def merge_polygons(polygons: list[np.ndarray], width: int, height: int
                   ) -> np.ndarray | None:
    """Merges the polygons of one instance into a single equivalent polygon.

    First splices the raw parts (exact when they do not overlap) and verifies
    the result against the union mask; overlapping parts make the spliced
    polygon self-intersect and lose the intersection to the even-odd fill
    rule — such cases fall back to splicing the external contours of the
    union mask, and the better of the two candidates is returned.
    """
    union = mask_utils.merge(mask_utils.frPyObjects(
        [p.reshape(-1).tolist() for p in polygons], height, width))

    raw = _splice(polygons)
    raw_iou = _polygon_iou(raw, union, width, height)
    if raw_iou >= 0.99:
        return raw

    mask = mask_utils.decode(union)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # +0.5: cv2 contour coords are pixel centers; shift to continuous space
    parts = [c.reshape(-1, 2).astype(np.float64) + 0.5
             for c in contours if len(c) >= 3]
    if not parts:
        return raw
    cont = _splice(parts)
    return cont if _polygon_iou(cont, union, width, height) > raw_iou else raw


def convert_split(ann_file: Path, root: Path, out: Path, subset: str,
                  mode: str) -> tuple[int, int, int, list[dict]]:
    """Converts one instances_*.json; returns (images, instances, merged, mapping)."""
    coco = json.loads(ann_file.read_text(encoding="utf-8"))

    cat_ids = sorted(c["id"] for c in coco["categories"])
    assert cat_ids == list(range(1, len(cat_ids) + 1)), \
        f"{ann_file}: category ids are not contiguous 1..K"

    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_dir = out / "images" / subset
    lbl_dir = out / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_inst, n_merged, mapping = 0, 0, []
    for im in coco["images"]:
        src = root / im["file_name"]
        if not src.is_file():
            raise SystemExit(f"Image not found: {src}")
        dst = img_dir / f"{im['id']}{src.suffix.lower()}"
        if not dst.exists():
            if mode == "copy":
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

        w, h = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(im["id"], []):
            polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                     for p in a["segmentation"] if len(p) >= 6]
            if not polys:
                continue
            if len(polys) > 1:
                pts = merge_polygons(polys, w, h)
                if pts is None:
                    continue
                n_merged += 1
            else:
                pts = polys[0]
            pts = pts / np.array([w, h], dtype=np.float64)
            pts = np.clip(pts, 0.0, 1.0)
            coords = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
            lines.append(f"{a['category_id'] - 1} {coords}")
            n_inst += 1
        (lbl_dir / f"{im['id']}.txt").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        mapping.append({"split": subset, "image_id": im["id"],
                        "file_name": im["file_name"]})
    return len(coco["images"]), n_inst, n_merged, mapping


def main() -> int:
    p = argparse.ArgumentParser(description="COCO splits -> YOLO-seg dataset")
    p.add_argument("--split-dir", type=Path, required=True,
                   help="directory with instances_{train,val,test}.json")
    p.add_argument("--root", type=Path, required=True,
                   help="dataset root the COCO file_name paths are relative to")
    p.add_argument("--out", type=Path, required=True,
                   help="output YOLO dataset directory")
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink",
                   help="symlink (default, no extra disk) or copy the images")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    train_json = json.loads(
        (args.split_dir / "instances_train.json").read_text(encoding="utf-8"))
    names = {c["id"] - 1: c["name"]
             for c in sorted(train_json["categories"], key=lambda c: c["id"])}

    all_mapping: list[dict] = []
    for subset in SUBSETS:
        ann_file = args.split_dir / f"instances_{subset}.json"
        n_img, n_inst, n_merged, mapping = convert_split(
            ann_file, root, out, subset, args.mode)
        all_mapping.extend(mapping)
        print(f"[✓] {subset:<6} {n_img:>6} images, {n_inst:>7} instances "
              f"({n_merged} multi-polygon instances merged)")

    with (out / "image_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["split", "image_id", "file_name"])
        wr.writeheader()
        wr.writerows(all_mapping)

    yaml_lines = [f"path: {out}", "train: images/train", "val: images/val",
                  "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in names.items()]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[✓] {len(names)} classes -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
