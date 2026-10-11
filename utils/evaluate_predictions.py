"""
evaluate_predictions.py — unified evaluation of instance-segmentation predictions.

Input: a COCO GT file (instances_*.json) and a predictions file in the COCO
results format — a JSON list of
    {"image_id": int, "category_id": int, "score": float,
     "segmentation": RLE (or polygon list), "bbox": [x, y, w, h] (optional)}
which is exactly what training notebooks 02-04 save (test_predictions.json,
eval_test/coco_instances_results.json).

Metrics:
  * COCO AP (IoU 0.50:0.95), AP50, AP75, APs/APm/APl — segm and (optionally) bbox;
  * per-class segm AP;
  * per-class precision / recall / F1 at a fixed confidence threshold
    (the way YOLO reports are usually presented) + micro/macro averages;
  * confusion matrix (rows = GT + background, cols = predicted + missed);
  * mean IoU of matched masks (mask quality in isolation from classification).

Examples:
    python evaluate_predictions.py \
        --gt notebooks/outputs/splits/instances_test.json \
        --pred runs/maskrcnn_torchvision_std10/test_predictions.json \
        --name maskrcnn_tv_std10 --out notebooks/outputs/comparison

    python evaluate_predictions.py --gt ... --pred sam_predictions.json \
        --name sam_vit_b --agnostic --out notebooks/outputs/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCOEVAL_STATS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def load_predictions(pred_path: Path, coco_gt: COCO) -> list[dict]:
    """Loads COCO-format predictions; converts polygon segmentations to RLE."""
    preds = json.loads(Path(pred_path).read_text())
    if not isinstance(preds, list):
        raise SystemExit(f"{pred_path}: expected a JSON list of predictions")

    gt_img_ids = set(coco_gt.getImgIds())
    gt_cat_ids = set(coco_gt.getCatIds())
    kept, alien_img, alien_cat = [], 0, 0
    for p in preds:
        if "segmentation" not in p:
            continue
        if p["image_id"] not in gt_img_ids:
            alien_img += 1
            continue
        if p["category_id"] not in gt_cat_ids:
            alien_cat += 1
            continue
        seg = p["segmentation"]
        if isinstance(seg, list):  # polygons -> RLE
            info = coco_gt.loadImgs(p["image_id"])[0]
            rles = mask_utils.frPyObjects(seg, info["height"], info["width"])
            p = dict(p)
            p["segmentation"] = mask_utils.merge(rles)
        kept.append(p)
    if alien_img or alien_cat:
        print(f"[!] Dropped predictions outside GT: {alien_img} by image_id, "
              f"{alien_cat} by category_id")
    return kept


def coco_ap(coco_gt: COCO, preds: list[dict], iou_type: str,
            class_names: dict[int, str], agnostic: bool = False,
            ) -> tuple[dict, dict]:
    """COCOeval on all predictions (no score filtering, per the COCO protocol)."""
    if not preds:
        return {s: 0.0 for s in COCOEVAL_STATS}, {}
    coco_dt = coco_gt.loadRes([dict(p) for p in preds])
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    if agnostic:
        ev.params.useCats = 0
    ev.evaluate(); ev.accumulate(); ev.summarize()

    metrics = {name: float(ev.stats[i]) for i, name in enumerate(COCOEVAL_STATS)}
    per_class = {}
    if not agnostic:
        # precision shape: [IoU, recall, class, area, maxDet]
        prec = ev.eval["precision"]
        for k, cat_id in enumerate(ev.params.catIds):
            p = prec[:, :, k, 0, -1]
            per_class[class_names[cat_id]] = (
                float(p[p > -1].mean()) if (p > -1).any() else float("nan"))
    return metrics, per_class


def pr_f1_confusion(coco_gt: COCO, preds: list[dict], conf_thresh: float,
                    iou_thresh: float,
                    ) -> tuple[pd.DataFrame, np.ndarray, list[str], float]:
    """Greedy class-agnostic matching by descending score: a prediction that localizes a GT
    object but misclassifies it counts as FP for the predicted class and FN
    for the true one."""
    preds = [p for p in preds if p["score"] >= conf_thresh]

    cat_ids = sorted(coco_gt.getCatIds())
    cat_names = [coco_gt.loadCats(c)[0]["name"] for c in cat_ids]
    cat_idx = {c: i for i, c in enumerate(cat_ids)}
    K = len(cat_ids)

    # rows = GT classes (+ background), cols = predicted (+ missed)
    conf = np.zeros((K + 1, K + 1), dtype=int)
    tp, fp, fn = np.zeros(K, int), np.zeros(K, int), np.zeros(K, int)
    matched_ious: list[float] = []

    preds_by_img: dict[int, list[dict]] = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    for img_id in coco_gt.getImgIds():
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, iscrowd=False))
        dts = sorted(preds_by_img.get(img_id, []), key=lambda p: -p["score"])
        gt_rles = [coco_gt.annToRLE(a) for a in gts]
        dt_rles = [d["segmentation"] for d in dts]
        ious = (mask_utils.iou(dt_rles, gt_rles, [0] * len(gt_rles))
                if gt_rles and dt_rles else np.zeros((len(dt_rles), len(gt_rles))))

        gt_used: set[int] = set()
        for di, d in enumerate(dts):
            best_gi, best_iou = -1, iou_thresh
            for gi in range(len(gts)):
                if gi not in gt_used and ious[di, gi] >= best_iou:
                    best_gi, best_iou = gi, ious[di, gi]
            dk = cat_idx[d["category_id"]]
            if best_gi >= 0:
                gt_used.add(best_gi)
                gk = cat_idx[gts[best_gi]["category_id"]]
                conf[gk, dk] += 1
                matched_ious.append(best_iou)
                if gk == dk:
                    tp[dk] += 1
                else:                     # localized but misclassified
                    fp[dk] += 1; fn[gk] += 1
            else:                         # no GT match -> false positive
                conf[K, dk] += 1
                fp[dk] += 1
        for gi, g in enumerate(gts):      # unmatched GT -> missed object
            if gi not in gt_used:
                gk = cat_idx[g["category_id"]]
                conf[gk, K] += 1
                fn[gk] += 1

    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    table = pd.DataFrame({"precision": prec, "recall": rec, "F1": f1,
                          "TP": tp, "FP": fp, "FN": fn}, index=cat_names)
    mp = tp.sum() / max(tp.sum() + fp.sum(), 1)
    mr = tp.sum() / max(tp.sum() + fn.sum(), 1)
    table.loc["micro avg"] = [mp, mr, 2 * mp * mr / max(mp + mr, 1e-9),
                              tp.sum(), fp.sum(), fn.sum()]
    table.loc["macro avg"] = [prec.mean(), rec.mean(), f1.mean(),
                              tp.sum(), fp.sum(), fn.sum()]
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return table.round(4), conf, cat_names, mean_iou


def plot_confusion(conf: np.ndarray, cat_names: list[str], iou_thresh: float,
                   conf_thresh: float, out_png: Path) -> None:
    K = len(cat_names)
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    side = max(8, K * 0.45)
    fig, ax = plt.subplots(figsize=(side + 1, side))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K + 1))
    ax.set_xticklabels(cat_names + ["missed"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K + 1))
    ax.set_yticklabels(cat_names + ["background"], fontsize=8)
    for i in range(K + 1):
        for j in range(K + 1):
            if conf[i, j]:
                ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion matrix (mask IoU >= {iou_thresh}, conf >= {conf_thresh})")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    matplotlib.use("Agg")      # headless CLI; importing the module keeps inline plots
    p = argparse.ArgumentParser(
        description="Unified evaluation of COCO-format instance predictions")
    p.add_argument("--gt", type=Path, required=True, help="COCO GT (instances_*.json)")
    p.add_argument("--pred", type=Path, required=True,
                   help="predictions in COCO results format")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--name", type=str, default=None,
                   help="model label, used as the output file prefix "
                        "(default: predictions file stem)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="confidence threshold for P/R/F1 and the confusion matrix")
    p.add_argument("--iou", type=float, default=0.5,
                   help="mask IoU threshold for matching predictions to GT")
    p.add_argument("--agnostic", action="store_true",
                   help="class-agnostic mode (SAM with GT-box prompts): "
                        "COCOeval with useCats=0, no classification tables")
    p.add_argument("--no-bbox", action="store_true", help="skip bbox COCOeval")
    args = p.parse_args()

    name = args.name or args.pred.stem
    args.out.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(args.gt))
    class_names = {c["id"]: c["name"] for c in
                   coco_gt.loadCats(coco_gt.getCatIds())}
    preds = load_predictions(args.pred, coco_gt)
    n_imgs_with_preds = len({p["image_id"] for p in preds})
    print(f"[*] {name}: {len(preds)} predictions on {n_imgs_with_preds}/"
          f"{len(coco_gt.getImgIds())} GT images")
    if not preds:
        raise SystemExit("No usable predictions — nothing to evaluate.")

    summary: dict = {"name": name, "gt": str(args.gt), "pred": str(args.pred),
                     "n_predictions": len(preds), "agnostic": args.agnostic,
                     "conf_thresh": args.conf, "iou_thresh": args.iou}

    # COCO AP
    rows = {}
    print("\n[*] COCOeval segm" + (" (class-agnostic)" if args.agnostic else ""))
    segm_metrics, per_class = coco_ap(coco_gt, preds, "segm", class_names,
                                      agnostic=args.agnostic)
    rows["segm"] = segm_metrics
    if not args.no_bbox:
        print("\n[*] COCOeval bbox")
        bbox_metrics, _ = coco_ap(coco_gt, preds, "bbox", class_names,
                                  agnostic=args.agnostic)
        rows["bbox"] = bbox_metrics

    coco_df = pd.DataFrame(rows).round(4)
    coco_df.to_csv(args.out / f"{name}_coco_metrics.csv")
    summary["coco"] = {k: {m: round(v, 4) for m, v in d.items()}
                       for k, d in rows.items()}
    print(f"\n{name} — COCO metrics:")
    print(coco_df.to_string())

    if per_class:
        per_class_df = (pd.Series(per_class, name="mask AP")
                        .sort_values(ascending=False).to_frame().round(4))
        per_class_df.to_csv(args.out / f"{name}_per_class_ap.csv")
        summary["per_class_segm_AP"] = {k: round(v, 4)
                                        for k, v in per_class.items()}
        print(f"\n{name} — per-class segm AP:")
        print(per_class_df.to_string())

    # P/R/F1 + confusion matrix (skipped in class-agnostic mode)
    table, conf, cat_names, mean_iou = pr_f1_confusion(
        coco_gt, preds, args.conf, args.iou)
    summary["mean_matched_iou"] = round(mean_iou, 4)
    print(f"\nMean IoU of matched masks (conf >= {args.conf}, "
          f"IoU >= {args.iou}): {mean_iou:.4f}")

    if not args.agnostic:
        table.to_csv(args.out / f"{name}_pr_f1.csv")
        pd.DataFrame(conf, index=cat_names + ["background"],
                     columns=cat_names + ["missed"]).to_csv(
            args.out / f"{name}_confusion_matrix.csv")
        plot_confusion(conf, cat_names, args.iou, args.conf,
                       args.out / f"{name}_confusion_matrix.png")
        summary["pr_f1"] = {
            "micro": table.loc["micro avg", ["precision", "recall", "F1"]].to_dict(),
            "macro": table.loc["macro avg", ["precision", "recall", "F1"]].to_dict()}
        print(f"\n{name} — P/R/F1 (conf >= {args.conf}, IoU >= {args.iou}):")
        print(table.to_string())

    (args.out / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[✓] Results written to {args.out}/{name}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
