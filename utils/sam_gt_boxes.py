"""
sam_gt_boxes.py — zero-shot SAM evaluation with ground-truth boxes as prompts.

Evaluate the output with the shared evaluator in class-agnostic mode:
    python utils/evaluate_predictions.py --gt .../instances_test.json \
        --pred sam_predictions.json --name sam_vit_b --agnostic --out ...

Example:
    python sam_gt_boxes.py \
        --gt notebooks/outputs/splits/instances_test.json \
        --root "/path/to/SUNRGBD" \
        --out sam_predictions_std10.json \
        --model facebook/sam-vit-base

    # quick smoke test on 5 images:
    python sam_gt_boxes.py --gt ... --root ... --out smoke.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from transformers import SamModel, SamProcessor 


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(
        description="SAM zero-shot masks from GT-box prompts -> COCO predictions")
    p.add_argument("--gt", type=Path, required=True, help="COCO GT (instances_*.json)")
    p.add_argument("--root", type=Path, required=True,
                   help="dataset root the COCO file_name paths are relative to")
    p.add_argument("--out", type=Path, required=True,
                   help="output predictions JSON (COCO results format)")
    p.add_argument("--model", type=str, default="facebook/sam-vit-base",
                   help="HF model id (sam-vit-base | sam-vit-large | sam-vit-huge)")
    p.add_argument("--device", type=str, default="auto",
                   help="auto (default) | cuda | mps | cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N images (smoke test)")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[*] Loading {args.model} on {device} ...")
    model = SamModel.from_pretrained(args.model).to(device).eval()
    processor = SamProcessor.from_pretrained(args.model)

    root = args.root.expanduser().resolve()
    coco_gt = COCO(str(args.gt))
    img_ids = coco_gt.getImgIds()
    if args.limit:
        img_ids = img_ids[: args.limit]

    predictions: list[dict] = []
    t0 = time.time()
    for n, img_id in enumerate(img_ids, 1):
        info = coco_gt.loadImgs(img_id)[0]
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, iscrowd=False))
        if not anns:
            continue
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])

        image = Image.open(root / info["file_name"]).convert("RGB")
        inputs = processor(image, input_boxes=[boxes], return_tensors="pt")
        # float64 -> float32 before moving: MPS has no float64 support
        inputs = {k: ((v.float() if torch.is_floating_point(v) else v).to(device)
                      if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)

        # [n_boxes, 3, H, W] at the original resolution
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]
        iou_scores = outputs.iou_scores.cpu()[0]          # [n_boxes, 3]
        best = iou_scores.argmax(dim=-1)                  # best hypothesis per box

        for bi, a in enumerate(anns):
            m = masks[bi, best[bi]].numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            predictions.append({
                "image_id": img_id,
                "category_id": a["category_id"],          # copied from GT (see docstring)
                "bbox": [float(v) for v in a["bbox"]],    # the prompt box
                "score": float(iou_scores[bi, best[bi]]),
                "segmentation": rle,
            })

        if n % 25 == 0 or n == len(img_ids):
            speed = n / (time.time() - t0)
            eta_min = (len(img_ids) - n) / speed / 60
            print(f"    {n}/{len(img_ids)} images "
                  f"({speed:.2f} img/s, ETA {eta_min:.0f} min)", end="\r")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"[✓] {len(predictions)} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
