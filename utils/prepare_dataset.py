"""
Helper script: runs steps 1.2, 1.6 and 1.7 without any visualization.

The notebook `01_data_preparation.ipynb` is the primary tool — it shows plots
and intermediate checks. This script is only needed when the data has already
been inspected visually and the splits must be regenerated quickly with
different parameters (e.g. a different number of classes or random seed)
without re-running the whole notebook.

Examples:
    # full pipeline
    python prepare_dataset.py --root ~/datasets/SUNRGBD --out outputs --top-k 10

    # only rebuild the splits from an existing COCO file (fast)
    python prepare_dataset.py --coco outputs/sunrgbd_instances_all.json \\
        --out outputs --top-k 15 --seed 123

    # dataset v2 from scratch: new stratified std10 split, then the extended
    # configs inheriting it (see CLAUDE.md for the exact commands)
    python prepare_dataset.py --root ~/datasets/SUNRGBD --out outputs \
        --classes standard10 --fresh-split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sunrgbd as sd  

SEMANTIC_MERGES = {
    "cupboard": "cabinet", "file cabinet": "cabinet", "filing cabinet": "cabinet",
    "filecabinet": "cabinet", "filingcabinet": "cabinet",
    "monitor": "display", "television": "display",
    "washbasin": "sink", "wash basin": "sink", "bathroom sink": "sink",
    "kitchen sink": "sink",
    "refrigerator": "fridge", "mini fridge": "fridge",
    "blackboard": "whiteboard", "chalkboard": "whiteboard",
    "countertop": "counter", "kitchen counter": "counter",
    "dustbin": "garbage bin", "dust bin": "garbage bin",
    "phone": "telephone", "stuffed toy": "toy",
    "electrical outlet": "outlet", "electricaloutlet": "outlet",
    "water bottle": "bottle", "waterbottle": "bottle",
    "cardboard box": "box", "cardboardbox": "box",
    "photo": "picture", "column": "pillar", "notice": "sign",
    "doorframe": "door", "doorway": "door",
}

# Classes never auto-selected:
# planar/"stuff"-like, tiny (mostly < 32^2 px), amorphous, furniture parts
# (nested masks inside dresser/bed), and person.
EXCLUDED_CLASSES = {
    # planar / stuff
    "window", "door", "curtain", "blinds", "mirror", "whiteboard", "picture",
    "poster", "painting", "carpet", "rug", "counter", "baseboard", "pillar",
    "board", "sign",
    # tiny
    "mouse", "outlet", "cup", "bottle", "faucet", "plate", "bowl", "glass",
    "vase", "tag", "pricetag", "speaker", "light", "jar", "clock", "shoe",
    "switch", "lightswitch",
    # amorphous
    "clothes", "cloth", "towel", "paper", "box", "bag", "book", "blanket",
    "pipe", "wire", "cord", "container",
    # furniture/room parts (already covered by the whole object's class)
    "drawer", "headboard", "mattress", "bed frame", "bedframe",
    "tabletop", "windowsill", "handle", "frame", "vent", "air vent",
    # too generic / meaningless as a category
    "object", "wood", "tile", "machine", "file",
    # out of scope
    "person",
}


SUBSETS = ("train", "val", "test")

# Written into every split dir (dataset_version.json); the training notebooks
# check it before training.
DATASET_VERSION = "v2"
DATASET_NOTES = ("proper polygon clipping, recovered names, leak-free grouping "
                 "(sun3d site, multi-sensor captures, kv2 sessions, kv1 "
                 "near-duplicates), stratified group split")


def load_image_ids(paths: list[Path]) -> dict[str, int]:
    """rel_key -> image_id from earlier outputs (COCO json or split_assignment.csv)."""
    ids: dict[str, int] = {}
    for path in paths:
        if path.suffix == ".json":
            images = json.loads(path.read_text(encoding="utf-8"))["images"]
            pairs = [(im["rel_key"], im["id"]) for im in images]
        else:
            prev = pd.read_csv(path)
            pairs = list(zip(prev.rel_key, prev.image_id))
        for key, img_id in pairs:
            if ids.setdefault(key, int(img_id)) != int(img_id):
                raise SystemExit(f"Conflicting image_id for {key}: {ids[key]} vs "
                                 f"{img_id} in {path}")
    if len(set(ids.values())) != len(ids):
        raise SystemExit("The same image_id is used by several scenes in "
                         + ", ".join(map(str, paths)))
    return ids


def build_coco(root: Path, out_dir: Path, min_area: float, min_side: float,
               require_intrinsics: bool,
               id_map: dict[str, int] | None = None) -> Path:
    """Step 1.2: SUN RGB-D polygons -> COCO.

    `id_map` (rel_key -> image_id of an earlier build) keeps image ids stable:
    known scenes reuse their id, new ones get max+1, ... Without it, the id is
    the 1-based position in the sorted scene list (so adding a scene would
    shift every later id).
    """
    print(f"[*] Scanning {root} ...")
    scenes = [s for s in sd.find_scenes(root, require_annotation=True)]
    if not scenes:
        raise SystemExit("No annotated scenes found — check the path.")
    sd.assign_groups(scenes)
    sizes = Counter(Counter(s.group_key for s in scenes).values())
    print(f"[*] Scenes to process: {len(scenes)} in {sum(sizes.values())} leak-free "
          f"groups (largest: {max(sizes)} scenes)")

    images, annotations = [], []
    cat_ids, skipped = {}, Counter()
    ann_id = 1
    id_map = id_map or {}
    next_id = max(id_map.values(), default=0) + 1
    n_new_ids = 0

    for pos, scene in enumerate(scenes, start=1):
        if pos % 250 == 0 or pos == len(scenes):
            print(f"    {pos}/{len(scenes)}", end="\r", flush=True)
        img_id = id_map.get(scene.rel_key, next_id) if id_map else pos

        intr = sd.read_intrinsics(scene.intrinsics_path)
        if require_intrinsics and intr is None:
            skipped["missing camera intrinsics"] += 1
            continue
        try:
            with Image.open(scene.rgb_path) as im:
                width, height = im.size
        except OSError:
            skipped["unreadable RGB"] += 1
            continue

        kept = 0
        for obj in sd.load_annotations(scene.annotation_path):
            if not obj.name or obj.name in sd.STUFF_CLASSES:
                skipped["stuff category"] += 1
                continue
            polys = [sd.clip_polygon(p, width, height) for p in obj.polygons]
            polys = [p for p in polys if len(p) >= 6]
            if not polys:
                skipped["degenerate polygon"] += 1
                continue
            bbox, area = sd.polygons_bbox_area(polys, width, height)
            if area < min_area or bbox[2] < min_side or bbox[3] < min_side:
                skipped["object too small"] += 1
                continue

            cat_ids.setdefault(obj.name, len(cat_ids) + 1)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_ids[obj.name],
                "segmentation": [[round(v, 2) for v in p] for p in polys],
                "bbox": [round(v, 2) for v in bbox], "area": round(area, 2), "iscrowd": 0,
            })
            ann_id += 1
            kept += 1

        if kept == 0:
            skipped["scene with no usable objects"] += 1
            continue
        if id_map and img_id == next_id:   # new scene: take the next free id
            next_id += 1
            n_new_ids += 1

        images.append({
            "id": img_id, "file_name": scene.rgb_path.relative_to(root).as_posix(),
            "width": width, "height": height,
            "depth_file": scene.depth_path.relative_to(root).as_posix(),
            "depth_source": getattr(scene, "depth_source", ""),
            "sensor": scene.sensor, "collection": scene.collection,
            "scene_type": scene.scene_type, "rel_key": scene.rel_key,
            "group_key": scene.group_key,
            "fx": intr.fx if intr else None, "fy": intr.fy if intr else None,
            "cx": intr.cx if intr else None, "cy": intr.cy if intr else None,
        })
    print()

    valid = {a["image_id"] for a in annotations}
    images = [im for im in images if im["id"] in valid]
    coco = {
        "info": {"description": "SUN RGB-D instance segmentation", "version": DATASET_VERSION,
                 "source_root": str(root)},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": "object"}
                       for n, i in sorted(cat_ids.items(), key=lambda kv: kv[1])],
    }
    path = out_dir / "sunrgbd_instances_all.json"
    path.write_text(json.dumps(coco), encoding="utf-8")

    print(f"[✓] {len(images)} images, {len(annotations)} instances, {len(cat_ids)} categories")
    if id_map:
        print(f"    image ids: {len(images) - n_new_ids} reused, {n_new_ids} new")
    if skipped:
        for reason, cnt in skipped.most_common():
            print(f"    filtered out — {reason}: {cnt}")
    return path


def stratified_group_assignment(groups: dict[str, list[dict]], anns: list[dict],
                                fractions: dict[str, float], seed: int,
                                fixed: dict[str, str] | None = None,
                                max_passes: int = 50) -> dict[str, str]:
    """Stratified assignment of whole groups to subsets.

    Every group is described by its per-class instance counts, per-sensor
    image counts and its image count. The cost is the sum over subsets and
    features of the squared deviation from the target count, relative to the
    target ((count - fraction * total) / (fraction * total))^2, so a rare class
    in val/test weighs as much as a frequent one in train.
      1. greedy: groups are placed one by one (largest relative weight first —
         groups with rare classes / big groups go early, ties broken by a
         seeded shuffle) where the cost grows the least;
      2. refinement: single-group moves and two-group swaps between subsets
         that lower the cost, until none is left.
    `fixed` groups (inherited from a base config) are counted but never moved.
    """
    fixed = fixed or {}
    keys = list(groups)
    cats = sorted({a["category_id"] for a in anns})
    sensors = sorted({im.get("sensor", "") for g in groups.values() for im in g})
    col = {("cat", c): i for i, c in enumerate(cats)}
    col.update({("sensor", s): len(cats) + i for i, s in enumerate(sensors)})
    img_group = {im["id"]: gi for gi, g in enumerate(keys) for im in groups[g]}

    feats = np.zeros((len(keys), len(col) + 1))
    for a in anns:
        feats[img_group[a["image_id"]], col[("cat", a["category_id"])]] += 1
    for gi, g in enumerate(keys):
        for im in groups[g]:
            feats[gi, col[("sensor", im.get("sensor", ""))]] += 1
        feats[gi, -1] = len(groups[g])

    subsets = list(fractions)
    total = np.maximum(feats.sum(0), 1.0)
    target = np.array([fractions[s] for s in subsets])[:, None] * total[None, :]
    scale = np.maximum(target, 1.0) ** 2
    current = np.zeros_like(target)
    where = np.full(len(keys), -1)
    for gi, g in enumerate(keys):
        if g in fixed:
            where[gi] = subsets.index(fixed[g])
            current[where[gi]] += feats[gi]

    def add_cost(f: np.ndarray) -> np.ndarray:      # cost growth per subset
        return ((f * (2 * (current - target) + f)) / scale).sum(1)

    free = [gi for gi, g in enumerate(keys) if g not in fixed]
    random.Random(seed).shuffle(free)
    free.sort(key=lambda gi: -(feats[gi] / total).max())      # stable sort
    for gi in free:
        where[gi] = int(np.argmin(add_cost(feats[gi])))
        current[where[gi]] += feats[gi]

    free_arr = np.array(free, dtype=int)
    for _ in range(max_passes):
        changed = 0
        for gi in free:                                   # single-group moves
            f, s = feats[gi], where[gi]
            current[s] -= f
            gain = add_cost(f)
            best = int(np.argmin(gain))
            if gain[best] < gain[s] - 1e-12:
                s, changed = best, changed + 1
            where[gi] = s
            current[s] += f
        for gi in free:                                   # two-group swaps
            s1 = where[gi]
            for s2 in range(len(subsets)):
                if s2 == s1:
                    continue
                cand = free_arr[where[free_arr] == s2]
                if not len(cand):
                    continue
                d = feats[cand] - feats[gi]               # s1 gets d, s2 loses d
                e1, e2 = current[s1] - target[s1], current[s2] - target[s2]
                delta = (((2 * e1 + d) * d) / scale[s1]).sum(1) \
                    + (((d - 2 * e2) * d) / scale[s2]).sum(1)
                k = int(np.argmin(delta))
                if delta[k] < -1e-12:
                    gj = cand[k]
                    current[s1] += d[k]
                    current[s2] -= d[k]
                    where[gi], where[gj] = s2, s1
                    changed += 1
                    break
        if not changed:
            break
    return {g: subsets[where[gi]] for gi, g in enumerate(keys)}


def split_report(split: dict[str, list[dict]], anns: list[dict],
                 categories: list[dict]) -> pd.DataFrame:
    """Per-class instance and per-sensor image counts in every subset, with
    the relative deviation of each subset's share from the train share."""
    sub_of = {im["id"]: s for s, imgs in split.items() for im in imgs}
    id2name = {c["id"]: c["name"] for c in categories}
    rows = Counter()
    for a in anns:
        rows[("class", id2name[a["category_id"]], sub_of[a["image_id"]])] += 1
    for s, imgs in split.items():
        for im in imgs:
            rows[("sensor", im.get("sensor", ""), s)] += 1
    sensors = sorted({im.get("sensor", "") for imgs in split.values() for im in imgs})
    index = pd.MultiIndex.from_tuples([("class", c["name"]) for c in categories]
                                      + [("sensor", s) for s in sensors],
                                      names=["kind", "name"])
    df = (pd.Series(rows).rename_axis(["kind", "name", "subset"]).unstack("subset")
          .reindex(index=index, columns=list(SUBSETS)).fillna(0).astype(int))
    df["total"] = df[list(SUBSETS)].sum(axis=1)
    for s in SUBSETS:
        share = df[s] / df.groupby(level="kind")[s].transform("sum")
        train = df["train"] / df.groupby(level="kind")["train"].transform("sum")
        df[f"{s}_share"] = share.round(4)
        if s != "train":
            df[f"{s}_rel_dev"] = ((share - train) / train).round(3)
    return df


def check_leaks(split: dict[str, list[dict]], source_root: str | None = None,
                visual: bool = True) -> tuple[dict[str, int], list[str]]:
    """Independent leak audit of a split (rules a-e of sd.assign_groups).

    Re-derives the "same place" relations from the raw metadata instead of
    trusting group_key: (0) a group spanning subsets; (a) a SUN3D scan / site
    in several subsets; (c) simultaneous multi-sensor captures (different
    devices, <= 5 s apart) in different subsets; (d) frames of the same device
    <= 60 s apart and any timestamped frames <= CAPTURE_GAP_S apart in
    different subsets; (e) kv1 visual near-duplicates in different subsets
    (needs the images: `source_root`). Returns (counts, report lines).
    """
    sub = {im["rel_key"]: s for s, imgs in split.items() for im in imgs}
    imgs = [im for imgs in split.values() for im in imgs]
    counts: dict[str, int] = {}

    per_group: dict[str, set] = defaultdict(set)
    for im in imgs:
        per_group[im.get("group_key") or im["rel_key"]].add(sub[im["rel_key"]])
    counts["groups spanning subsets"] = sum(len(v) > 1 for v in per_group.values())

    scans, sites = defaultdict(set), defaultdict(set)
    for im in imgs:
        site = sd.sun3d_site(im["rel_key"])
        if site is not None:
            scans[im["rel_key"].rsplit("/", 1)[0]].add(sub[im["rel_key"]])
            sites[site].add(sub[im["rel_key"]])
    counts["(a) SUN3D scans in >1 subset"] = sum(len(v) > 1 for v in scans.values())
    counts["(a) SUN3D sites in >1 subset"] = sum(len(v) > 1 for v in sites.values())

    timed = sorted((t, dev, im["rel_key"]) for im in imgs
                   for t, dev in [sd.parse_capture_time(im["rel_key"]) or (None, None)]
                   if t is not None)
    c_pairs = d_pairs = gap_pairs = 0
    near_train: set[str] = set()
    for i, (t, dev, key) in enumerate(timed):
        for t2, dev2, key2 in timed[i + 1:]:
            dt = (t2 - t).total_seconds()
            if dt > sd.CAPTURE_GAP_S:
                break
            if sub[key] == sub[key2]:
                continue
            gap_pairs += 1
            c_pairs += dev != dev2 and dt <= 5
            if dev == dev2 and dt <= 60:
                d_pairs += 1
                for a, b in ((key, key2), (key2, key)):
                    if sub[a] == "test" and sub[b] == "train":
                        near_train.add(a)
    counts["(c) multi-sensor pairs <= 5 s in different subsets"] = c_pairs
    counts["(d) same-device pairs <= 60 s in different subsets"] = d_pairs
    counts["(d) test images with a train frame of the same device <= 60 s"] = len(near_train)
    counts[f"(b-d) timestamped pairs <= {sd.CAPTURE_GAP_S} s in different subsets"] = gap_pairs

    if visual and source_root:
        frames = [(im["rel_key"], Path(source_root) / im["file_name"]) for im in imgs
                  if im.get("collection") in sd.NO_METADATA_COLLECTIONS]
        counts["(e) kv1 near-duplicate pairs in different subsets"] = sum(
            sub[a] != sub[b] for a, b in sd.visual_links(frames))

    lines = [f"{'OK  ' if v == 0 else 'LEAK'} {k}: {v}" for k, v in counts.items()]
    return counts, lines


def make_splits(coco_path: Path, out_dir: Path, top_k: int, manual: list[str],
                val: float, test: float, seed: int,
                min_instances: int | None = None,
                splits_name: str = "splits",
                base_assignment: Path | None = None,
                fresh: bool = False, visual_check: bool = True,
                stratify_min_instances: int | None = 400) -> Path:
    """Steps 1.6-1.7: class selection and grouped, stratified train/val/test split.

    Selection modes (in priority order):
      * `manual`        — an explicit class list (e.g. standard10);
      * `min_instances` — every class with >= N instances after SEMANTIC_MERGES,
                          excluding EXCLUDED_CLASSES; the standard 10 are always
                          kept (ids 1-10, same as in the standard10 config);
      * otherwise       — the top_k most frequent classes.

    Assignment (whole groups, see sd.assign_groups):
      * an existing <split_dir>/split_assignment.csv is a FROZEN split: its
        rel_key -> subset rows are kept, new frames join their group's subset
        (ignored with `fresh=True`);
      * else `base_assignment` (the split of a base config) is inherited the
        same way;
      * all remaining groups get a seeded stratified assignment
        (`stratified_group_assignment`). Besides the selected classes it also
        balances every class with >= `stratify_min_instances` instances (the
        min400 class set), so configs inheriting this split stay balanced.
    Writes instances_{train,val,test}.json, split_assignment.csv,
    split_class_distribution.csv, leak_check.txt and dataset_version.json
    into out_dir/splits_name; returns that directory.
    """
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    merges = SEMANTIC_MERGES if min_instances is not None else {}
    id2name = {c["id"]: merges.get(c["name"], c["name"])
               for c in coco["categories"]}
    counts = Counter(id2name[a["category_id"]] for a in coco["annotations"])

    if manual:
        keep = list(manual)
    elif min_instances is not None:
        extra = sorted(
            (n for n, c in counts.items()
             if c >= min_instances and n not in EXCLUDED_CLASSES
             and n not in sd.STANDARD_10_CLASSES),
            key=lambda n: -counts[n])
        keep = list(sd.STANDARD_10_CLASSES) + extra
    else:
        keep = [n for n, _ in counts.most_common(top_k)]
    missing = [n for n in keep if n not in counts]
    if missing:
        raise SystemExit(f"Unknown classes: {', '.join(missing)}")

    coverage = sum(counts[n] for n in keep) / max(sum(counts.values()), 1) * 100
    print(f"[*] Selected {len(keep)} categories, coverage {coverage:.1f}%")

    new_cats = [{"id": i, "name": n, "supercategory": "object"} for i, n in enumerate(keep, 1)]
    name2id = {c["name"]: c["id"] for c in new_cats}

    anns = []
    for a in coco["annotations"]:
        name = id2name[a["category_id"]]
        if name in name2id:
            b = dict(a); b["category_id"] = name2id[name]
            anns.append(b)

    keep_ids = {a["image_id"] for a in anns}
    imgs = [im for im in coco["images"] if im["id"] in keep_ids]

    groups = defaultdict(list)
    for im in imgs:
        groups[im.get("group_key") or im["rel_key"]].append(im)

    split_dir = out_dir / splits_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # A frozen split is preserved row by row (rel_key -> subset); a group
    # with known frames keeps their (majority) subset for its new frames.
    assign_csv = split_dir / "split_assignment.csv"
    frozen_src = None
    if assign_csv.is_file() and not fresh:
        frozen_src = assign_csv
        if base_assignment is not None:
            print(f"[!] {assign_csv} already exists and is used as the frozen "
                  f"split; --base-assignment {base_assignment} is IGNORED "
                  f"(use --fresh-split to re-inherit from the base config)")
    elif base_assignment is not None:
        if not Path(base_assignment).is_file():
            raise SystemExit(f"Base assignment not found: {base_assignment}")
        frozen_src = Path(base_assignment)
    existing: dict[str, str] = {}
    if frozen_src is not None:
        prev = pd.read_csv(frozen_src)
        existing = dict(zip(prev.rel_key, prev.subset))
        prev_ids = dict(zip(prev.rel_key, prev.image_id))
        moved = [im["rel_key"] for im in imgs
                 if im["rel_key"] in prev_ids and int(prev_ids[im["rel_key"]]) != im["id"]]
        if moved:
            raise SystemExit(f"{len(moved)} image ids differ from {frozen_src} "
                             f"(e.g. {moved[0]}) — rebuild the COCO file with "
                             f"the old ids before splitting")
        print(f"[*] Frozen split: using assignments from {frozen_src}")

    fixed: dict[str, str] = {}
    for g, members in groups.items():
        known = [existing[im["rel_key"]] for im in members if im["rel_key"] in existing]
        if known:
            fixed[g] = Counter(known).most_common(1)[0][0]
    strat_anns = list(anns)
    if stratify_min_instances is not None:      # also balance the min400 classes
        merged = {c["id"]: SEMANTIC_MERGES.get(c["name"], c["name"])
                  for c in coco["categories"]}
        n_merged = Counter(merged[a["category_id"]] for a in coco["annotations"])
        extra = {n for n, c in n_merged.items() if c >= stratify_min_instances
                 and n not in EXCLUDED_CLASSES and n not in name2id}
        extra_id = {n: -i for i, n in enumerate(sorted(extra), 1)}
        strat_anns += [{"image_id": a["image_id"],
                        "category_id": extra_id[merged[a["category_id"]]]}
                       for a in coco["annotations"]
                       if a["image_id"] in keep_ids and merged[a["category_id"]] in extra]
    fractions = {"train": 1.0 - val - test, "val": val, "test": test}
    group_subset = stratified_group_assignment(groups, strat_anns, fractions, seed, fixed)

    split = {s: [] for s in SUBSETS}
    n_joined = n_kept = 0
    for g in sorted(groups):
        for im in sorted(groups[g], key=lambda im: im["rel_key"]):
            subset = existing.get(im["rel_key"])
            if subset is None:
                subset = group_subset[g]
                n_joined += g in fixed
            else:
                n_kept += 1
            split[subset].append(im)
    if existing:
        print(f"[*] Frozen split: kept {n_kept} image assignments ({len(fixed)} "
              f"groups), {n_joined} new images joined their group, "
              f"{len(groups) - len(fixed)} new groups assigned (stratified)")
    else:
        print(f"[*] Stratified split of {len(groups)} groups (seed {seed})")

    rows = []
    for subset, subset_imgs in split.items():
        ids = {im["id"] for im in subset_imgs}
        sub_anns = [a for a in anns if a["image_id"] in ids]
        (split_dir / f"instances_{subset}.json").write_text(
            json.dumps({"info": {"subset": subset, "version": DATASET_VERSION},
                        "licenses": [], "images": subset_imgs,
                        "annotations": sub_anns, "categories": new_cats}), encoding="utf-8")
        print(f"[✓] {subset:<6} {len(subset_imgs):>6} images, {len(sub_anns):>7} instances")
        rows += [{"rel_key": im["rel_key"], "image_id": im["id"],
                  "group_key": im.get("group_key", ""), "subset": subset} for im in subset_imgs]

    pd.DataFrame(rows).sort_values(["subset", "rel_key"]).to_csv(assign_csv, index=False)

    report = split_report(split, anns, new_cats)
    report.to_csv(split_dir / "split_class_distribution.csv")
    dev = report[[f"{s}_rel_dev" for s in ("val", "test")]].abs()
    print(f"[*] Max |relative deviation| of val/test shares from train: "
          f"classes {dev.loc['class'].max().max():.1%}, "
          f"sensors {dev.loc['sensor'].max().max():.1%}")

    leak_counts, leak_lines = check_leaks(split, coco.get("info", {}).get("source_root"),
                                          visual=visual_check)
    sizes = pd.Series([len(v) for v in groups.values()])
    header = [f"Leak check — {split_dir.name} (dataset {DATASET_VERSION})",
              f"images {len(imgs)}, groups {len(groups)}; group size: median "
              f"{sizes.median():.0f}, p99 {sizes.quantile(0.99):.0f}, max {sizes.max()} "
              f"({sizes.max() / len(imgs):.1%} of images)", ""]
    (split_dir / "leak_check.txt").write_text("\n".join(header + leak_lines) + "\n",
                                               encoding="utf-8")
    for line in leak_lines:
        print("    " + line)
    leaks = {k: v for k, v in leak_counts.items() if v}
    if leaks and not existing:
        raise SystemExit(f"[!] Leak between subsets detected: {leaks}")
    print("[✓] No leak between subsets." if not leaks
          else f"[!] WARNING: the frozen split leaks: {leaks}")

    version = {
        "version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "config": split_dir.name,
        "classes": len(new_cats),
        "class_names": [c["name"] for c in new_cats],
        "n_images": {s: len(split[s]) for s in SUBSETS},
        "n_instances": {s: int(report.loc["class", s].sum()) for s in SUBSETS},
        "seed": seed,
        "split_source": str(frozen_src) if frozen_src else "fresh stratified split",
        "notes": DATASET_NOTES,
    }
    (split_dir / "dataset_version.json").write_text(json.dumps(version, indent=2) + "\n",
                                                     encoding="utf-8")
    return split_dir


def main() -> int:
    p = argparse.ArgumentParser(description="SUN RGB-D preparation (steps 1.2, 1.6, 1.7)")
    p.add_argument("--root", type=Path, help="SUN RGB-D dataset root")
    p.add_argument("--coco", type=Path, help="existing COCO file (skips step 1.2)")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--classes", type=str, default="",
                   help="comma-separated class list or 'standard10' "
                        "(the standard 10-class SUN RGB-D benchmark)")
    p.add_argument("--min-instances", type=int, default=None,
                   help="extended config: keep every class with >= N instances "
                        "(after SEMANTIC_MERGES, minus EXCLUDED_CLASSES; "
                        "the standard 10 are always kept)")
    p.add_argument("--splits-name", type=str, default="splits",
                   help="subdirectory of --out for the split files")
    p.add_argument("--base-assignment", type=Path, default=None,
                   help="split_assignment.csv of a base config to inherit "
                        "frozen scene-to-subset assignments from")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-area", type=float, default=200.0)
    p.add_argument("--min-side", type=float, default=8.0)
    p.add_argument("--allow-missing-intrinsics", action="store_true")
    p.add_argument("--fresh-split", action="store_true",
                   help="ignore an existing split_assignment.csv of this config "
                        "(new stratified split, or re-inherit --base-assignment)")
    p.add_argument("--stratify-min-instances", type=int, default=400,
                   help="also balance every class with >= N instances (after "
                        "SEMANTIC_MERGES) in the split; 0 = only the selected classes")
    p.add_argument("--no-visual-check", action="store_true",
                   help="skip the kv1 near-duplicate part of the leak check "
                        "(needs the images; ~1 min)")
    args = p.parse_args()

    if not args.root and not args.coco:
        p.error("provide --root or --coco")
    args.out.mkdir(parents=True, exist_ok=True)

    coco_path = args.coco
    if coco_path is None:
        # Keep image ids of the previous build / frozen splits stable.
        id_sources = [args.out / "sunrgbd_instances_all.json",
                      *sorted(args.out.glob("*/split_assignment.csv")),
                      *([args.base_assignment] if args.base_assignment else [])]
        id_sources = [path for path in id_sources if path.is_file()]
        id_map = load_image_ids(id_sources)
        if id_map:
            print(f"[*] Reusing image ids of {len(id_map)} scenes from: "
                  + ", ".join(map(str, id_sources)))
        coco_path = build_coco(args.root.expanduser().resolve(), args.out,
                               args.min_area, args.min_side,
                               not args.allow_missing_intrinsics, id_map=id_map)

    if args.classes.strip() == "standard10":
        manual = list(sd.STANDARD_10_CLASSES)
    else:
        manual = [c.strip() for c in args.classes.split(",") if c.strip()]
    make_splits(coco_path, args.out, args.top_k, manual, args.val, args.test,
                args.seed, min_instances=args.min_instances,
                splits_name=args.splits_name,
                base_assignment=args.base_assignment,
                fresh=args.fresh_split, visual_check=not args.no_visual_check,
                stratify_min_instances=args.stratify_min_instances or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
