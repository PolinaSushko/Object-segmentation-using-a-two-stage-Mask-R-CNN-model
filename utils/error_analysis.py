"""
error_analysis.py — reusable analysis helpers for notebooks 09 (error analysis)
and 10 (model comparison).

All headline metrics still come from the single evaluator
`utils/evaluate_predictions.py` (COCOeval wrappers, P/R/F1, confusion matrix);
this module adds what the evaluator does not provide:

  * run discovery — Colab run folders downloaded from Drive (`v2_<model>_<config>...`),
    their timing.json, training logs and parameter counts;
  * a per-detection match table taken from COCOeval's own `evalImgs`, from which
    AP is recomputed for any image weighting — exact for weights of 1, hence
    image-level bootstrap CIs and the paired bootstrap test of an AP difference
    (both models resampled on the same images);
  * a TIDE-style error decomposition (Bolya et al., ECCV 2020) with pycocotools
    mask IoU: cls / loc / both / dupe / bkg / missed, each fixed in isolation by
    an oracle, AP recomputed — the AP difference is the cost of that error type.
    `TideData.compute_ap()` reproduces COCOeval mask AP to float precision
    (checked by `check_tide_vs_cocoeval`);
  * renderers for error galleries and qualitative comparisons (PNG montages).

The TIDE code is a cleaned port of the one-off v1 analysis (vkr_work/v1_error_analysis).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:                                   # imported as `utils.error_analysis`
    from . import evaluate_predictions as ep
except ImportError:                    # imported with utils/ on sys.path
    import evaluate_predictions as ep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/Users/polina_sushko/My_work/University/ВКР/SUNRGBD")
DEFAULT_RUNS_ROOT = Path.home() / "Downloads"

IOU_THRS = np.linspace(0.5, 0.95, 10)      # COCOeval defaults
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100
T_B = 0.1                                   # TIDE background threshold
TIDE_ERRORS = ["cls", "loc", "both", "dupe", "bkg", "miss"]
SIZE_BUCKETS = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2),
                "large": (96 ** 2, 1e10)}   # COCO area ranges (px^2)
SENSORS = ["kv1", "kv2", "xtion", "realsense"]

# Nominal parameter counts (millions) used only when no checkpoint is in the run
# folder; heads with COCO classes, so the true value differs by < 1%.
NOMINAL_PARAMS_M = {
    "R_50_FPN_3x": 44.2, "R_101_FPN_3x": 63.2, "X_101_32x8d_FPN_3x": 107.3,
    "R50-FPN v2": 46.4, "YOLOv8s-seg": 11.8,
}

# Categorical palette (fixed order, follows the entity, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"


def set_plot_style() -> None:
    """Recessive axes / grid, thin marks; colours come from PALETTE in fixed order."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2, "legend.frameon": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold"})


@contextlib.contextmanager
def quiet():
    """Silences pycocotools' prints (index creation, COCOeval summaries)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# GT / predictions
# ---------------------------------------------------------------------------

def load_coco(path: Path) -> COCO:
    with quiet():
        return COCO(str(path))


def coco_from_dataset(dataset: dict) -> COCO:
    """Builds an in-memory COCO object from a dataset dict."""
    coco = COCO()
    coco.dataset = dataset
    with quiet():
        coco.createIndex()
    return coco


def subset_coco(coco_gt: COCO, img_ids: Iterable[int] | None = None,
                cat_relabel: dict[int, int] | None = None) -> COCO:
    """GT restricted to `img_ids` (all when None), categories optionally relabeled."""
    d = coco_gt.dataset
    keep = set(coco_gt.getImgIds() if img_ids is None else img_ids)
    anns = [a for a in d["annotations"] if a["image_id"] in keep]
    if cat_relabel:
        anns = [dict(a, category_id=cat_relabel.get(a["category_id"], a["category_id"]))
                for a in anns]
    return coco_from_dataset({
        "images": [im for im in d["images"] if im["id"] in keep],
        "annotations": anns,
        "categories": d["categories"]})


def load_preds(pred_path: Path, coco_gt: COCO,
               max_category_id: int | None = None) -> list[dict]:
    """Predictions via the evaluator's loader (polygons -> RLE; predictions whose
    image_id / category_id are not in the GT file are dropped and counted).

    That filter is exactly what evaluating a min400 model (ids 1–20) on the std10
    GT needs: ids 1–10 are identical in both configs, so predictions with
    category_id > 10 and those on min400-only images fall away. `max_category_id`
    only makes the intent explicit (checked against the GT categories).
    """
    if max_category_id is not None:
        assert max(coco_gt.getCatIds()) == max_category_id, \
            "max_category_id must equal the largest category id of the GT file"
    preds = ep.load_predictions(Path(pred_path), coco_gt)
    print(f"[i] {Path(pred_path).parent.name}/{Path(pred_path).name}: "
          f"{len(preds)} predictions on {len({p['image_id'] for p in preds})}/"
          f"{len(coco_gt.getImgIds())} GT images")
    return preds


def class_names(coco_gt: COCO) -> dict[int, str]:
    return {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}


def image_sensor(info: dict) -> str:
    """Sensor of an image: the `sensor` field, else the file_name prefix."""
    return info.get("sensor") or info["file_name"].split("/")[0]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

RUN_NAME_RE = re.compile(
    r"^(?P<version>v\d+)_(?P<family>maskrcnn_detectron2|maskrcnn_torchvision|yolov8seg)"
    r"_(?P<config>std10|min400|min100)"
    r"(?:_(?P<backbone>R_50_FPN_3x|R_101_FPN_3x|X_101_32x8d_FPN_3x))?"
    r"(?P<suffix>.*)$")

FAMILY_LABEL = {"maskrcnn_detectron2": "Mask R-CNN (Detectron2)",
                "maskrcnn_torchvision": "Mask R-CNN (torchvision)",
                "yolov8seg": "YOLOv8s-seg"}
CONFIG_SPLITS = {"std10": "splits", "min400": "splits_min400", "min100": "splits_min100"}
CONFIG_N_CLASSES = {"std10": 10, "min400": 20, "min100": 41}

# Run folders the comparison expects (glob patterns relative to the runs root).
EXPECTED_RUNS = [
    "v2_maskrcnn_detectron2_std10_R_50_FPN_3x*",
    "v2_maskrcnn_detectron2_std10_R_101_FPN_3x*",
    "v2_maskrcnn_detectron2_min400_*",
    "v2_maskrcnn_torchvision_std10*",
    "v2_yolov8seg_std10*",
]


@dataclass
class Run:
    name: str
    dir: Path
    family: str
    config: str
    backbone: str
    suffix: str = ""
    files: dict = field(default_factory=dict)

    @property
    def is_maskrcnn(self) -> bool:
        return self.family.startswith("maskrcnn")

    @property
    def label(self) -> str:
        bb = f" {self.backbone}" if self.family == "maskrcnn_detectron2" else ""
        sfx = self.suffix.strip("_")
        return f"{FAMILY_LABEL[self.family]}{bb} {self.config}" + (f" [{sfx}]" if sfx else "")

    def gt_path(self, split: str, config: str | None = None) -> Path:
        cfg = config or self.config
        return (PROJECT_ROOT / "notebooks" / "outputs" / CONFIG_SPLITS[cfg]
                / f"instances_{split}.json")


def parse_run(path: Path) -> Run | None:
    m = RUN_NAME_RE.match(path.name)
    if not m:
        return None
    fam = m["family"]
    backbone = (m["backbone"] or "") if fam == "maskrcnn_detectron2" else (
        "R50-FPN v2" if fam == "maskrcnn_torchvision" else "YOLOv8s-seg")
    run = Run(name=path.name, dir=path, family=fam, config=m["config"],
              backbone=backbone, suffix=m["suffix"] or "")
    for key, rel in [("test", "test_predictions.json"), ("val", "val_predictions.json"),
                     ("timing", "timing.json"), ("d2_metrics", "metrics.json"),
                     ("tv_history", "history.csv"), ("yolo_results", "train/results.csv")]:
        if (path / rel).exists():
            run.files[key] = path / rel
    return run


def discover_runs(runs_root: Path, version: str = "v2", verbose: bool = True) -> list[Run]:
    """All run folders `<version>_*` under `runs_root` that hold test predictions."""
    runs_root = Path(runs_root).expanduser()
    runs, skipped = [], []
    for d in sorted(runs_root.glob(f"{version}_*")):
        if not d.is_dir():
            continue
        run = parse_run(d)
        if run is None:
            continue
        if "test" not in run.files:
            skipped.append(f"{d.name} (no test_predictions.json)")
            continue
        runs.append(run)
    if verbose:
        for pat in EXPECTED_RUNS:
            if not any(r.dir.match(pat) for r in runs):
                print(f"[note] expected run not found, skipped: {runs_root / pat}")
        for s in skipped:
            print(f"[note] run folder skipped: {s}")
        print(f"[*] {len(runs)} run(s) found under {runs_root}")
    return runs


def read_timing(run: Run) -> dict:
    if "timing" not in run.files:
        return {}
    return json.loads(run.files["timing"].read_text())


def count_params(run: Run) -> tuple[float, str]:
    """(parameters in millions, source). From a checkpoint in the run folder when
    one is present and loadable, otherwise the nominal value of the architecture."""
    candidates = ["model_best.pth", "model_final.pth", "best.pth", "last.pth",
                  "train/weights/best.pt"]
    for rel in candidates:
        path = run.dir / rel
        if not path.exists():
            continue
        try:
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck.get("model", ck) if isinstance(ck, dict) else ck
            if hasattr(sd, "parameters"):          # pickled nn.Module (YOLO)
                n = sum(p.numel() for p in sd.parameters())
            else:
                n = sum(int(np.prod(v.shape)) for k, v in sd.items()
                        if hasattr(v, "shape") and not any(
                            s in k for s in ("running_mean", "running_var",
                                             "num_batches_tracked")))
            return n / 1e6, f"checkpoint {rel}"
        except Exception as e:                       # noqa: BLE001
            print(f"[note] {run.name}: cannot count parameters from {rel} ({e!r:.80})")
    nominal = NOMINAL_PARAMS_M.get(run.backbone)
    return (nominal, "nominal") if nominal else (float("nan"), "n/a")


def load_val_curve(run: Run) -> pd.DataFrame:
    """Validation mask AP per epoch from the run's training log.

    Detectron2 metrics.json — `segm/AP` (percent) at every eval (EVAL_PERIOD = one
    nominal epoch); torchvision history.csv — `val_segm_AP`; YOLO results.csv —
    Ultralytics' own `metrics/mAP50-95(M)` (not COCOeval, so only the trend is
    comparable)."""
    if "d2_metrics" in run.files:
        rows = []
        for line in run.files["d2_metrics"].read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "segm/AP" in r:
                rows.append((r["iteration"], r["segm/AP"] / 100.0))
        if not rows:
            return pd.DataFrame()
        df = (pd.DataFrame(rows, columns=["iteration", "val_mask_AP"])
              .drop_duplicates("iteration", keep="last").sort_values("iteration"))
        period = df.iteration.iloc[0] + 1
        df["epoch"] = (df.iteration + 1) / period
        df["source"] = "COCOeval (Detectron2)"
        return df[["epoch", "val_mask_AP", "source"]]
    if "tv_history" in run.files:
        h = pd.read_csv(run.files["tv_history"])
        return pd.DataFrame({"epoch": h.epoch, "val_mask_AP": h.val_segm_AP,
                             "source": "COCOeval (torchvision)"})
    if "yolo_results" in run.files:
        h = pd.read_csv(run.files["yolo_results"])
        h.columns = h.columns.str.strip()
        return pd.DataFrame({"epoch": h.epoch, "val_mask_AP": h["metrics/mAP50-95(M)"],
                             "source": "Ultralytics mAP50-95(M)"})
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Evaluation through the shared evaluator
# ---------------------------------------------------------------------------

def evaluate(coco_gt: COCO, preds: list[dict], agnostic: bool = False,
             bbox: bool = True, conf: float = 0.5, iou: float = 0.5,
             pr_f1: bool = True) -> dict:
    """COCO segm (+bbox) AP, per-class AP, P/R/F1, confusion, mean matched IoU —
    all computed by utils/evaluate_predictions.py."""
    names = class_names(coco_gt)
    out: dict = {}
    with quiet():
        out["segm"], out["per_class"] = ep.coco_ap(coco_gt, preds, "segm", names,
                                                   agnostic=agnostic)
        out["bbox"] = (ep.coco_ap(coco_gt, preds, "bbox", names, agnostic=agnostic)[0]
                       if bbox else None)
    if pr_f1:
        table, cm, cat_names, mean_iou = ep.pr_f1_confusion(coco_gt, preds, conf, iou)
        out.update(pr_f1=table, confusion=cm, cat_names=cat_names, mean_iou=mean_iou)
    return out


def segm_ap(coco_gt: COCO, preds: list[dict], agnostic: bool = False) -> dict:
    """Only the COCO segm stats (fast path for validation / subsets)."""
    with quiet():
        return ep.coco_ap(coco_gt, preds, "segm", class_names(coco_gt),
                          agnostic=agnostic)[0]


def select_best(table: pd.DataFrame, candidates: pd.Series,
                metric: str = "val_mask_AP") -> str:
    """Model selection rule of the thesis: the highest *validation* mask AP among
    `candidates`; test metrics are never looked at here."""
    assert not metric.startswith("test"), "model selection must not use the test split"
    pool = table[candidates & table[metric].notna()]
    if pool.empty:
        raise ValueError("no candidate with a validation score")
    return pool[metric].idxmax()


# ---------------------------------------------------------------------------
# Match table from COCOeval -> AP for any image weighting (bootstrap)
# ---------------------------------------------------------------------------

@dataclass
class MatchTable:
    """Per-detection outcome of COCOeval matching (area 'all', maxDets 100).

    For every evaluated class k: image index, score, TP/FP flags per IoU threshold,
    in the exact order COCOeval.accumulate() uses; plus non-ignored GT counts per
    image. AP for image weights w (bootstrap counts) equals COCOeval on the
    resampled set; for w = 1 it equals COCOeval exactly."""
    img_ids: list[int]
    cat_ids: list[int]
    det_img: dict[int, np.ndarray]
    det_score: dict[int, np.ndarray]
    det_tp: dict[int, np.ndarray]           # (T, n) bool
    det_fp: dict[int, np.ndarray]           # (T, n) bool
    gt_count: np.ndarray                    # (K, n_img) non-ignored GT
    cocoeval_stats: np.ndarray = None

    @property
    def n_images(self) -> int:
        return len(self.img_ids)


def build_match_table(coco_gt: COCO, preds: list[dict], iou_type: str = "segm",
                      agnostic: bool = False) -> MatchTable:
    with quiet():
        coco_dt = coco_gt.loadRes([dict(p) for p in preds])
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        if agnostic:
            ev.params.useCats = 0
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    p = ev.params
    img_ids = list(p.imgIds)
    img_pos = {i: n for n, i in enumerate(img_ids)}
    cat_ids = list(p.catIds) if p.useCats else [-1]
    max_det = p.maxDets[-1]
    n_a, n_i = len(p.areaRng), len(img_ids)
    det_img, det_score, det_tp, det_fp = {}, {}, {}, {}
    gt_count = np.zeros((len(cat_ids), n_i))
    for k, cat in enumerate(cat_ids):
        base = k * n_a * n_i                     # area index 0 = 'all'
        E = [ev.evalImgs[base + i] for i in range(n_i)]
        E = [e for e in E if e is not None]
        for e in E:
            gt_count[k, img_pos[e["image_id"]]] = np.count_nonzero(e["gtIgnore"] == 0)
        if not E:
            det_img[cat] = np.zeros(0, int); det_score[cat] = np.zeros(0)
            det_tp[cat] = det_fp[cat] = np.zeros((len(p.iouThrs), 0), bool)
            continue
        scores = np.concatenate([e["dtScores"][:max_det] for e in E])
        imgs = np.concatenate([[img_pos[e["image_id"]]] * len(e["dtScores"][:max_det])
                               for e in E]).astype(int)
        order = np.argsort(-scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in E], axis=1)[:, order]
        dtig = np.concatenate([e["dtIgnore"][:, :max_det] for e in E], axis=1)[:, order]
        det_img[cat] = imgs[order]
        det_score[cat] = scores[order]
        det_tp[cat] = np.logical_and(dtm, np.logical_not(dtig))
        det_fp[cat] = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
    return MatchTable(img_ids, cat_ids, det_img, det_score, det_tp, det_fp,
                      gt_count, cocoeval_stats=np.array(ev.stats))


def _class_precision(tp_w: np.ndarray, fp_w: np.ndarray, npig: float) -> np.ndarray:
    """COCOeval.accumulate for one class and one weighting: (T,) mean interpolated
    precision over the 101 recall thresholds."""
    T = tp_w.shape[0]
    out = np.zeros(T)
    if tp_w.shape[1] == 0:
        return out
    tp = np.cumsum(tp_w, axis=1)
    fp = np.cumsum(fp_w, axis=1)
    rc = tp / npig
    pr = tp / (fp + tp + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]   # precision envelope
    n = pr.shape[1]
    for t in range(T):
        inds = np.searchsorted(rc[t], REC_THRS, side="left")
        q = np.where(inds < n, pr[t, np.minimum(inds, n - 1)], 0.0)
        out[t] = q.mean()
    return out


def ap_from_weights(mt: MatchTable, weights: np.ndarray | None = None,
                    per_class: bool = False):
    """COCO AP (IoU .50:.95) and AP50 for image weights (n_img,) — integer
    bootstrap counts or 0/1 subset indicators. Classes without (weighted) GT are
    excluded, exactly as COCOeval does."""
    w = np.ones(mt.n_images) if weights is None else np.asarray(weights, float)
    cls_prec = {}
    for k, cat in enumerate(mt.cat_ids):
        npig = float(mt.gt_count[k] @ w)
        if npig == 0:
            continue
        wd = w[mt.det_img[cat]]
        cls_prec[cat] = _class_precision(mt.det_tp[cat] * wd, mt.det_fp[cat] * wd, npig)
    if not cls_prec:
        return (np.nan, np.nan, {}) if per_class else (np.nan, np.nan)
    P = np.array(list(cls_prec.values()))            # (K_valid, T)
    ap, ap50 = float(P.mean()), float(P[:, 0].mean())
    if per_class:
        return ap, ap50, {c: float(v.mean()) for c, v in cls_prec.items()}
    return ap, ap50


def bootstrap_weights(n_images: int, n_boot: int, seed: int = 0,
                      subset: np.ndarray | None = None) -> np.ndarray:
    """(n_boot, n_images) resampling counts; with `subset` (bool mask) only those
    images are resampled (others get weight 0)."""
    rng = np.random.default_rng(seed)
    pool = np.arange(n_images) if subset is None else np.flatnonzero(subset)
    W = np.zeros((n_boot, n_images))
    for b in range(n_boot):
        idx = rng.choice(pool, size=len(pool), replace=True)
        W[b] = np.bincount(idx, minlength=n_images)
    return W


def bootstrap_ap(mt: MatchTable, n_boot: int = 1000, seed: int = 0,
                 subset: np.ndarray | None = None) -> pd.DataFrame:
    """Bootstrap distribution of (AP, AP50) over images."""
    W = bootstrap_weights(mt.n_images, n_boot, seed, subset)
    return pd.DataFrame([ap_from_weights(mt, w) for w in W], columns=["AP", "AP50"])


def ci_summary(point: float, boot: Sequence[float], level: float = 0.95) -> dict:
    """Percentile CI (ci_lo/ci_hi) plus the bias of the bootstrap distribution and
    the basic (bias-corrected, 2θ − q) interval. The bootstrap distribution of AP
    is shifted upwards (the precision envelope is a running max, so resampling
    noise inflates it), which makes the percentile CI asymmetric around θ."""
    boot = np.asarray(boot)
    a = (1 - level) / 2
    lo, hi = float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))
    return {"point": point, "boot_se": float(boot.std(ddof=1)),
            "ci_lo": lo, "ci_hi": hi, "boot_bias": float(boot.mean() - point),
            "basic_lo": 2 * point - hi, "basic_hi": 2 * point - lo}


def paired_bootstrap(mt_a: MatchTable, mt_b: MatchTable, n_boot: int = 1000,
                     seed: int = 0, metric: str = "AP") -> dict:
    """Paired bootstrap test of AP(a) − AP(b): both models evaluated on the SAME
    resampled images. Two-sided p-value = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))."""
    assert mt_a.img_ids == mt_b.img_ids, "match tables must share the image list"
    j = 0 if metric == "AP" else 1
    W = bootstrap_weights(mt_a.n_images, n_boot, seed)
    point = ap_from_weights(mt_a)[j] - ap_from_weights(mt_b)[j]
    d = np.array([ap_from_weights(mt_a, w)[j] - ap_from_weights(mt_b, w)[j] for w in W])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"metric": metric, "delta": point, "boot_se": float(d.std(ddof=1)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p_value": p, "n_boot": n_boot, "n_images": mt_a.n_images}


def per_image_stats(mt: MatchTable, conf: float = 0.5, iou_index: int = 0) -> pd.DataFrame:
    """Per-image TP/FP/FN/F1 at score >= conf and IoU >= IOU_THRS[iou_index]
    (class-aware COCOeval matching; greedy by score, so thresholding afterwards
    is exact)."""
    n = mt.n_images
    tp, fp = np.zeros(n), np.zeros(n)
    for cat in mt.cat_ids:
        keep = mt.det_score[cat] >= conf
        imgs = mt.det_img[cat][keep]
        np.add.at(tp, imgs, mt.det_tp[cat][iou_index, keep])
        np.add.at(fp, imgs, mt.det_fp[cat][iou_index, keep])
    n_gt = mt.gt_count.sum(0)
    fn = n_gt - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0)
    return pd.DataFrame({"image_id": mt.img_ids, "n_gt": n_gt, "tp": tp, "fp": fp,
                         "fn": fn, "f1": f1}).set_index("image_id")


# ---------------------------------------------------------------------------
# TIDE-style error decomposition
# ---------------------------------------------------------------------------

class TideData:
    """Per-image detections (top MAX_DETS per image and class, as COCOeval), GT
    and the full det x GT mask-IoU matrix (class-agnostic)."""

    def __init__(self, coco_gt: COCO, preds: list[dict], max_dets: int = MAX_DETS):
        self.coco_gt = coco_gt
        self.preds = preds
        self.cat_ids = sorted(coco_gt.getCatIds())
        self.img_ids = sorted(coco_gt.getImgIds())
        self.names = class_names(coco_gt)
        self.gts, self.dets, self.iou = {}, {}, {}
        by_img = defaultdict(list)
        for i, p in enumerate(preds):
            by_img[p["image_id"]].append(i)
        for img in self.img_ids:
            anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img, iscrowd=False))
            rles = [coco_gt.annToRLE(a) for a in anns]
            idx = sorted(by_img.get(img, []), key=lambda i: -preds[i]["score"])
            per_cat = defaultdict(int)
            kept = []
            for i in idx:                           # maxDets per (image, class)
                c = preds[i]["category_id"]
                if per_cat[c] < max_dets:
                    per_cat[c] += 1
                    kept.append(i)
            drles = [preds[i]["segmentation"] for i in kept]
            # gt tuple: (ann_id, cat, area, ignored)
            self.gts[img] = [(a["id"], a["category_id"], a["area"], False) for a in anns]
            self.dets[img] = [{"score": preds[i]["score"], "cat": preds[i]["category_id"],
                               "pidx": i, "suppressed": False} for i in kept]
            self.iou[img] = (mask_utils.iou(drles, rles, [0] * len(rles))
                             if rles and drles else np.zeros((len(drles), len(rles))))

    # -- matching ------------------------------------------------------------
    @staticmethod
    def match_image(dets, gts, iou, t):
        """COCOeval greedy matching per class at threshold t (ties -> later GT)."""
        det_match = np.full(len(dets), -1)
        gt_matched = np.zeros(len(gts), bool)
        if not dets or not gts:
            return det_match, gt_matched
        gcat = np.array([g[1] for g in gts])
        gign = np.array([g[3] for g in gts])
        thr = min(t, 1 - 1e-10)
        order = np.argsort([-d["score"] for d in dets], kind="mergesort")
        for di in order:
            d = dets[di]
            if d["suppressed"]:
                continue
            cand = (~gt_matched) & (gcat == d["cat"]) & (~gign) & (iou[di] >= thr)
            if not cand.any():
                continue
            idx = np.flatnonzero(cand)
            vals = iou[di, idx]
            best = idx[len(idx) - 1 - int(np.argmax(vals[::-1]))]
            det_match[di] = best
            gt_matched[best] = True
        return det_match, gt_matched

    def compute_ap(self, t_list=IOU_THRS, dets=None, gts=None, iou=None,
                   img_ids=None, per_class=False):
        """COCO AP (101-point, per-class mean) over thresholds `t_list`, with
        optional oracle overrides of dets / gts / iou."""
        dets = dets or self.dets
        gts = gts or self.gts
        iou = iou or self.iou
        img_ids = self.img_ids if img_ids is None else img_ids
        per_cls = {c: [] for c in self.cat_ids}
        for t in t_list:
            scores, tps = defaultdict(list), defaultdict(list)
            npos = defaultdict(int)
            for img in img_ids:
                g, d = gts[img], dets[img]
                for x in g:
                    if not x[3]:
                        npos[x[1]] += 1
                dm, _ = self.match_image(d, g, iou[img], t)
                for di, x in enumerate(d):
                    if x["suppressed"]:
                        continue
                    scores[x["cat"]].append(x["score"])
                    tps[x["cat"]].append(dm[di] >= 0)
            for c in self.cat_ids:
                if npos[c] == 0:
                    per_cls[c].append(np.nan)
                    continue
                s = np.array(scores[c])
                tp = np.array(tps[c], bool)[np.argsort(-s, kind="mergesort")]
                per_cls[c].append(_class_precision(tp[None].astype(float),
                                                   (~tp)[None].astype(float),
                                                   float(npos[c]))[0])
        cls_ap = {c: float(np.mean(v)) for c, v in per_cls.items() if not np.isnan(v).any()}
        ap = float(np.mean(list(cls_ap.values()))) if cls_ap else np.nan
        return (ap, cls_ap) if per_class else ap

    # -- error classification -----------------------------------------------
    def classify_errors(self, t_f: float = 0.5, t_b: float = T_B):
        """One error type per non-TP detection at foreground threshold t_f.
        Returns (records DataFrame, gt_status {(img, gi): 'tp'|'miss'|'target'})."""
        records, gt_status = [], {}
        for img in self.img_ids:
            dets, gts, iou = self.dets[img], self.gts[img], self.iou[img]
            dm, gm = self.match_image(dets, gts, iou, t_f)
            gcat = np.array([g[1] for g in gts]) if gts else np.zeros(0, int)
            targeted = set()
            for di, d in enumerate(dets):
                row = {"img": img, "di": di, "cat": d["cat"], "score": d["score"],
                       "pidx": d["pidx"], "type": None, "gi": -1, "iou": 0.0}
                if dm[di] >= 0:
                    row.update(type="tp", gi=int(dm[di]), iou=float(iou[di, dm[di]]))
                elif not gts:
                    row["type"] = "bkg"
                else:
                    same = gcat == d["cat"]
                    ious = iou[di]
                    iou_same = np.where(same, ious, -1)
                    iou_other = np.where(~same, ious, -1)
                    js, jo = int(iou_same.argmax()), int(iou_other.argmax())
                    if t_b <= iou_same[js] < t_f:
                        row.update(type="loc", gi=js, iou=float(iou_same[js]))
                        targeted.add(js)
                    elif iou_other[jo] >= t_f:
                        row.update(type="cls", gi=jo, iou=float(iou_other[jo]))
                        targeted.add(jo)
                    elif iou_same[js] >= t_f:
                        row.update(type="dupe", gi=js, iou=float(iou_same[js]))
                    elif ious.max() < t_b:
                        row.update(type="bkg", gi=int(ious.argmax()), iou=float(ious.max()))
                    else:
                        row.update(type="both", gi=jo, iou=float(iou_other[jo]))
                records.append(row)
            for gi in range(len(gts)):
                gt_status[(img, gi)] = ("tp" if gm[gi] else
                                        "target" if gi in targeted else "miss")
        rec = pd.DataFrame(records)
        rec["cls_name"] = rec.cat.map(self.names)
        rec["gt_cat"] = [self.gts[r.img][r.gi][1] if r.gi >= 0 else -1
                         for r in rec.itertuples()]
        rec["gt_name"] = rec.gt_cat.map(self.names)
        return rec, gt_status

    def fixed_inputs(self, rec: pd.DataFrame, gt_status: dict, fix: str,
                     score_band: tuple[float, float] | None = None):
        """dets / gts / iou overrides with one error type fixed by an oracle:
        dupe/bkg/both -> detection removed; cls -> class corrected; loc -> mask
        replaced by the GT mask (IoU 1); miss -> GT removed. cls/loc fixes that
        hit an already-matched GT are removed instead (they would be dupes)."""
        dets_o = {img: [dict(d) for d in self.dets[img]] for img in self.img_ids}
        gts_o = {img: list(self.gts[img]) for img in self.img_ids}
        iou_o = {img: self.iou[img].copy() for img in self.img_ids}
        used = {k for k, v in gt_status.items() if v == "tp"}
        newly = set()
        sel = rec[rec.type == fix]
        if score_band is not None:
            sel = sel[(sel.score >= score_band[0]) & (sel.score < score_band[1])]
        for r in sel.sort_values("score", ascending=False).itertuples():
            img, di, gi = r.img, r.di, r.gi
            if fix in ("dupe", "bkg", "both"):
                dets_o[img][di]["suppressed"] = True
            elif fix in ("cls", "loc"):
                key = (img, gi)
                if key in used or key in newly:
                    dets_o[img][di]["suppressed"] = True
                    continue
                newly.add(key)
                if fix == "cls":
                    dets_o[img][di]["cat"] = gts_o[img][gi][1]
                else:
                    iou_o[img][di, gi] = 1.0
        if fix == "miss":
            for (img, gi), st in gt_status.items():
                if st == "miss":
                    g = gts_o[img][gi]
                    gts_o[img][gi] = (g[0], g[1], g[2], True)
        return dets_o, gts_o, iou_o


def check_tide_vs_cocoeval(data: TideData, tol: float = 1e-9) -> pd.DataFrame:
    """TIDE's own AP / AP50 vs pycocotools on the same predictions."""
    with quiet():
        stats, per_cls = ep.coco_ap(data.coco_gt, data.preds, "segm", data.names)
    own, own_cls = data.compute_ap(per_class=True)
    own50 = data.compute_ap([0.5])
    df = pd.DataFrame({"pycocotools": [stats["AP"], stats["AP50"]],
                       "tide_lib": [own, own50]}, index=["AP", "AP50"])
    df["abs_diff"] = (df.pycocotools - df.tide_lib).abs()
    cls_diff = max(abs(own_cls[c] - per_cls[data.names[c]]) for c in own_cls)
    df.attrs["max_per_class_diff"] = cls_diff
    df.attrs["exact"] = bool(df.abs_diff.max() < tol and cls_diff < tol)
    return df


def run_tide(data: TideData, t_f: float = 0.5, verbose: bool = True) -> dict:
    """Full decomposition. Returns dict with
    summary  — per error type: count, count with score > 0.5, ΔAP50, ΔAP;
    per_class — per class ΔAP50 / ΔAP per error type + counts;
    records, gt_status, base AP values."""
    base, base_cls = data.compute_ap(per_class=True)
    base50, base50_cls = data.compute_ap([0.5], per_class=True)
    rec, gt_status = data.classify_errors(t_f=t_f)
    n_miss = sum(v == "miss" for v in gt_status.values())
    rows, pc_rows = [], defaultdict(dict)
    for fix in TIDE_ERRORS:
        d_o, g_o, i_o = data.fixed_inputs(rec, gt_status, fix)
        ap50, ap50_cls = data.compute_ap([0.5], d_o, g_o, i_o, per_class=True)
        ap, ap_cls = data.compute_ap(IOU_THRS, d_o, g_o, i_o, per_class=True)
        is_det = fix != "miss"
        rows.append({"error": fix,
                     "count": int((rec.type == fix).sum()) if is_det else n_miss,
                     "count_score>0.5": (int(((rec.type == fix) & (rec.score > 0.5)).sum())
                                         if is_det else np.nan),
                     "dAP50": ap50 - base50, "dAP": ap - base})
        for c in base_cls:
            pc_rows[c][f"dAP50_{fix}"] = ap50_cls.get(c, np.nan) - base50_cls[c]
            pc_rows[c][f"dAP_{fix}"] = ap_cls.get(c, np.nan) - base_cls[c]
        if verbose:
            print(f"  {fix:5s} dAP50 {rows[-1]['dAP50']:+.4f}  dAP {rows[-1]['dAP']:+.4f}")
    # mask-quality oracle: every TP (at IoU 0.5) gets the perfect GT mask
    i_o = {img: data.iou[img].copy() for img in data.img_ids}
    for r in rec[rec.type == "tp"].itertuples():
        i_o[r.img][r.di, r.gi] = 1.0
    ap_pm = data.compute_ap(IOU_THRS, iou=i_o)
    rows.append({"error": "mask quality (TP masks -> GT)", "count": int((rec.type == "tp").sum()),
                 "count_score>0.5": int(((rec.type == "tp") & (rec.score > 0.5)).sum()),
                 "dAP50": 0.0, "dAP": ap_pm - base})
    summary = pd.DataFrame(rows)
    summary.attrs.update(base_AP=base, base_AP50=base50)

    pc = []
    for c in data.cat_ids:
        if c not in base_cls:
            continue
        sub = rec[rec.cat == c]
        n_gt = sum(1 for img in data.img_ids for g in data.gts[img] if g[1] == c)
        n_miss_c = sum(1 for (img, gi), v in gt_status.items()
                       if v == "miss" and data.gts[img][gi][1] == c)
        row = {"class": data.names[c], "n_gt": n_gt, "AP": base_cls[c],
               "AP50": base50_cls[c], "tp@0.5": int((sub.type == "tp").sum()),
               "miss": n_miss_c, "mean_tp_iou": sub[sub.type == "tp"].iou.mean()}
        for t in TIDE_ERRORS[:-1]:
            row[f"n_{t}>0.3"] = int(((sub.type == t) & (sub.score > 0.3)).sum())
        row.update(pc_rows[c])
        pc.append(row)
    return {"summary": summary, "per_class": pd.DataFrame(pc).set_index("class"),
            "records": rec, "gt_status": gt_status, "base_AP": base, "base_AP50": base50}


def bkg_by_score(data: TideData, rec: pd.DataFrame, gt_status: dict,
                 bands=((0.8, 1.01), (0.5, 0.8), (0.3, 0.5), (0.0, 0.3), (0.5, 1.01))
                 ) -> pd.DataFrame:
    """AP cost of background FPs by score band (only that band removed)."""
    base, base50 = data.compute_ap(), data.compute_ap([0.5])
    rows = []
    for lo, hi in bands:
        d_o, g_o, i_o = data.fixed_inputs(rec, gt_status, "bkg", score_band=(lo, hi))
        n = int(((rec.type == "bkg") & (rec.score >= lo) & (rec.score < hi)).sum())
        rows.append({"score_band": f"[{lo:g}, {min(hi, 1):g})", "n_bkg": n,
                     "dAP50": data.compute_ap([0.5], d_o, g_o, i_o) - base50,
                     "dAP": data.compute_ap(IOU_THRS, d_o, g_o, i_o) - base})
    return pd.DataFrame(rows)


def confusion_pairs(rec: pd.DataFrame, min_score: float = 0.3) -> pd.DataFrame:
    """(predicted, GT) class pairs of TIDE cls errors with score > min_score,
    plus the symmetric pair total."""
    ce = rec[(rec.type == "cls") & (rec.score > min_score)]
    pairs = (ce.groupby(["cls_name", "gt_name"]).size().rename("n")
             .reset_index().rename(columns={"cls_name": "predicted", "gt_name": "ground_truth"}))
    pairs["pair"] = [" <-> ".join(sorted([a, b])) for a, b in
                     zip(pairs.predicted, pairs.ground_truth)]
    pairs["pair_total"] = pairs.groupby("pair").n.transform("sum")
    return pairs.sort_values(["pair_total", "n"], ascending=False).reset_index(drop=True)


def bkg_vs_full_coco(data: TideData, rec: pd.DataFrame, full_coco: COCO,
                     min_score: float = 0.5) -> pd.DataFrame:
    """What lies under background / both FPs in the all-class COCO: the best
    overlapping annotation of a class OUTSIDE the evaluated set (IoU and the
    fraction of the detection it covers). A high IoU means the model found a real
    object that the class vocabulary (or the name mapping) does not cover; no
    overlap at all may still be an unlabeled object (needs visual audit)."""
    eval_names = set(data.names.values())
    fnames = class_names(full_coco)
    rows = []
    sel = rec[rec.type.isin(["bkg", "both"]) & (rec.score >= min_score)]
    for r in sel.itertuples():
        det = data.preds[r.pidx]["segmentation"]
        anns = [a for a in full_coco.loadAnns(full_coco.getAnnIds(imgIds=r.img))
                if fnames[a["category_id"]] not in eval_names]
        best_name, best_iou, best_iod = None, 0.0, 0.0
        if anns:
            rles = [full_coco.annToRLE(a) for a in anns]
            ious = mask_utils.iou([det], rles, [0] * len(rles))[0]
            inter = np.array([mask_utils.area(mask_utils.merge([det, g], intersect=True))
                              for g in rles])
            j = int(ious.argmax())
            best_name, best_iou = fnames[anns[j]["category_id"]], float(ious[j])
            best_iod = float(inter.max() / max(mask_utils.area(det), 1))
        rows.append({"img": r.img, "pidx": r.pidx, "type": r.type, "pred": r.cls_name,
                     "score": r.score, "other_name": best_name, "other_iou": best_iou,
                     "max_iod_other": best_iod})
    df = pd.DataFrame(rows)
    if len(df):
        df["covered_by"] = np.select(
            [df.other_iou >= 0.5, df.other_iou >= 0.1],
            ["other-class GT, IoU >= 0.5", "other-class GT, IoU 0.1-0.5"],
            "nothing annotated")
    return df


def merged_pair_ap(coco_gt: COCO, preds: list[dict],
                   pairs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Label-ambiguity check: AP of a class pair when both GT and predictions
    are relabeled to one class, vs the separate per-class APs."""
    names = class_names(coco_gt)
    n2i = {v: k for k, v in names.items()}
    with quiet():
        _, base = ep.coco_ap(coco_gt, preds, "segm", names)
    counts = pd.Series([a["category_id"] for a in coco_gt.dataset["annotations"]]).value_counts()
    rows = []
    for a, b in pairs:
        if a not in n2i or b not in n2i:
            continue
        ia, ib = n2i[a], n2i[b]
        g = subset_coco(coco_gt, cat_relabel={ia: ib})
        pr = [dict(p, category_id=ib) if p["category_id"] == ia else p for p in preds]
        with quiet():
            _, merged = ep.coco_ap(g, pr, "segm", names)
        na, nb = counts.get(ia, 0), counts.get(ib, 0)
        rows.append({"pair": f"{a}+{b}", "AP_a": base[a], "AP_b": base[b],
                     "AP_instance_weighted": (base[a] * na + base[b] * nb) / max(na + nb, 1),
                     "AP_merged": merged[b]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


CLASS_BGR = [_hex_bgr(c) for c in PALETTE] + [(180, 119, 31), (75, 25, 230)]


def read_image(info: dict, root: Path = DATASET_ROOT) -> np.ndarray:
    """BGR image resized to the COCO width/height (grey canvas if missing)."""
    im = cv2.imread(str(Path(root) / info["file_name"]))
    if im is None:
        return np.full((info["height"], info["width"], 3), 90, np.uint8)
    if im.shape[:2] != (info["height"], info["width"]):
        im = cv2.resize(im, (info["width"], info["height"]))
    return im


def draw_outline(img, rle, color, thick=2):
    m = mask_utils.decode(rle).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, thick)


def draw_fill(img, rle, color, alpha=0.45):
    m = mask_utils.decode(rle).astype(bool)
    img[m] = ((1 - alpha) * img[m] + alpha * np.array(color)).astype(np.uint8)


def put_label(img, text, xy, color, scale=0.45):
    x, y = int(xy[0]), max(int(xy[1]), 12)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _finish_tile(canvas, title, size=(560, 420)):
    canvas = cv2.resize(canvas, size)
    cv2.rectangle(canvas, (0, 0), (size[0], 22), (0, 0, 0), -1)
    cv2.putText(canvas, title[:95], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_error_tile(coco_gt: COCO, preds: list[dict], img_id: int, pidx: int = -1,
                      gi: int = -1, kind: str = "", full_coco: COCO | None = None,
                      root: Path = DATASET_ROOT) -> np.ndarray:
    """One inspected error: red fill = the detection, green = GT of the evaluated
    classes, yellow = the GT targeted by the error (missed / loc target),
    thin grey = annotations of all other classes (full COCO)."""
    names = class_names(coco_gt)
    info = coco_gt.loadImgs(img_id)[0]
    canvas = read_image(info, root)
    if full_coco is not None and img_id in full_coco.imgs:
        fnames = class_names(full_coco)
        for a in full_coco.loadAnns(full_coco.getAnnIds(imgIds=img_id)):
            n = fnames[a["category_id"]]
            if n in names.values():
                continue
            draw_outline(canvas, full_coco.annToRLE(a), (160, 160, 160), 1)
            x, y, w, h = a["bbox"]
            put_label(canvas, n, (x + 2, y + h / 2), (210, 210, 210), 0.38)
    gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, iscrowd=False))
    for j, a in enumerate(gts):
        col = (0, 255, 255) if j == gi else (0, 210, 0)
        draw_outline(canvas, coco_gt.annToRLE(a), col, 3 if j == gi else 2)
        x, y, w, h = a["bbox"]
        put_label(canvas, "GT " + names[a["category_id"]], (x + 2, y + 14), col)
    title = f"img {img_id} {image_sensor(info)}"
    if pidx >= 0:
        p = preds[pidx]
        draw_fill(canvas, p["segmentation"], (0, 0, 255))
        x, y, w, h = mask_utils.toBbox(p["segmentation"])
        put_label(canvas, f"PRED {names[p['category_id']]} {p['score']:.2f}",
                  (x + 2, y + h - 4), (90, 90, 255))
        title += f" | {kind} pred {names[p['category_id']]} {p['score']:.2f}"
    if kind == "miss" and gi >= 0:
        a = gts[gi]
        title += f" | MISSED {names[a['category_id']]} area {a['area']:.0f}"
    return _finish_tile(canvas, title)


def render_predictions_tile(coco_gt: COCO, preds_img: list[dict], img_id: int,
                            title: str, conf: float = 0.5, show_gt: bool = False,
                            root: Path = DATASET_ROOT) -> np.ndarray:
    """Masks of one image coloured by class: GT annotations (show_gt) or the
    predictions with score >= conf."""
    names = class_names(coco_gt)
    cat_idx = {c: k for k, c in enumerate(sorted(names))}
    info = coco_gt.loadImgs(img_id)[0]
    canvas = read_image(info, root)
    if show_gt:
        items = [(coco_gt.annToRLE(a), a["category_id"], None) for a in
                 coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, iscrowd=False))]
    else:
        items = [(p["segmentation"], p["category_id"], p["score"])
                 for p in sorted(preds_img, key=lambda p: p["score"]) if p["score"] >= conf]
    for rle, cat, score in items:
        col = CLASS_BGR[cat_idx[cat] % len(CLASS_BGR)]
        draw_fill(canvas, rle, col, 0.4)
        draw_outline(canvas, rle, col, 2)
    for rle, cat, score in items:
        x, y, w, h = mask_utils.toBbox(rle)
        txt = names[cat] + ("" if score is None else f" {score:.2f}")
        put_label(canvas, txt, (x + 2, y + 14), (255, 255, 255))
    return _finish_tile(canvas, title)


def save_montage(tiles: list[np.ndarray], path: Path, ncols: int = 2) -> Path:
    """Grid of equally sized tiles -> PNG."""
    if not tiles:
        return path
    blank = np.zeros_like(tiles[0])
    rows = []
    for k in range(0, len(tiles), ncols):
        row = tiles[k:k + ncols]
        row += [blank] * (ncols - len(row))
        rows.append(np.hstack(row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path
