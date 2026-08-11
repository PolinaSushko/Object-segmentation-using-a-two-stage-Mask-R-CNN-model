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
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

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


def build_coco(root: Path, out_dir: Path, min_area: float, min_side: float,
               require_intrinsics: bool) -> Path:
    """Step 1.2: SUN RGB-D polygons -> COCO."""
    print(f"[*] Scanning {root} ...")
    scenes = [s for s in sd.find_scenes(root, require_annotation=True)]
    if not scenes:
        raise SystemExit("No annotated scenes found — check the path.")
    sd.assign_groups(scenes)
    print(f"[*] Scenes to process: {len(scenes)}")

    images, annotations = [], []
    cat_ids, skipped = {}, Counter()
    ann_id = 1

    for img_id, scene in enumerate(scenes, start=1):
        if img_id % 250 == 0 or img_id == len(scenes):
            print(f"    {img_id}/{len(scenes)}", end="\r", flush=True)

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
        "info": {"description": "SUN RGB-D instance segmentation", "source_root": str(root)},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": "object"}
                       for n, i in sorted(cat_ids.items(), key=lambda kv: kv[1])],
    }
    path = out_dir / "sunrgbd_instances_all.json"
    path.write_text(json.dumps(coco), encoding="utf-8")

    print(f"[✓] {len(images)} images, {len(annotations)} instances, {len(cat_ids)} categories")
    if skipped:
        for reason, cnt in skipped.most_common():
            print(f"    filtered out — {reason}: {cnt}")
    return path


def make_splits(coco_path: Path, out_dir: Path, top_k: int, manual: list[str],
                val: float, test: float, seed: int,
                min_instances: int | None = None,
                splits_name: str = "splits",
                base_assignment: Path | None = None) -> None:
    """Steps 1.6-1.7: class selection and grouped train/val/test split.

    Selection modes (in priority order):
      * `manual`        — an explicit class list (e.g. standard10);
      * `min_instances` — every class with >= N instances after SEMANTIC_MERGES,
                          excluding EXCLUDED_CLASSES; the standard 10 are always
                          kept (ids 1-10, same as in the standard10 config);
      * otherwise       — the top_k most frequent classes.
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
    
    assign_csv = split_dir / "split_assignment.csv"
    existing: dict[str, str] = {}
    if not assign_csv.is_file() and base_assignment is not None:
        if not Path(base_assignment).is_file():
            raise SystemExit(f"Base assignment not found: {base_assignment}")
        prev = pd.read_csv(base_assignment)
        existing = dict(zip(prev.group_key, prev.subset))
        print(f"[*] Inherited frozen split from {base_assignment}")
    elif assign_csv.is_file():
        prev = pd.read_csv(assign_csv)
        existing = dict(zip(prev.group_key, prev.subset))

    new_keys = sorted(g for g in groups if g not in existing)
    rng = random.Random(seed)
    rng.shuffle(new_keys)
    n_test, n_val = int(round(len(new_keys) * test)), int(round(len(new_keys) * val))

    assign = {g: existing[g] for g in groups if g in existing}
    for i, g in enumerate(new_keys):
        assign[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    if existing:
        print(f"[*] Frozen split found: kept {len(assign) - len(new_keys)} "
              f"existing group assignments, assigned {len(new_keys)} new groups")

    split = {"train": [], "val": [], "test": []}
    for g, subset in assign.items():
        split[subset].extend(groups[g])
    rows = []
    for subset, subset_imgs in split.items():
        ids = {im["id"] for im in subset_imgs}
        sub_anns = [a for a in anns if a["image_id"] in ids]
        (split_dir / f"instances_{subset}.json").write_text(
            json.dumps({"info": {"subset": subset}, "licenses": [], "images": subset_imgs,
                        "annotations": sub_anns, "categories": new_cats}), encoding="utf-8")
        print(f"[✓] {subset:<6} {len(subset_imgs):>6} images, {len(sub_anns):>7} instances")
        rows += [{"rel_key": im["rel_key"], "image_id": im["id"],
                  "group_key": im.get("group_key", ""), "subset": subset} for im in subset_imgs]

    pd.DataFrame(rows).sort_values(["subset", "rel_key"]).to_csv(
        split_dir / "split_assignment.csv", index=False)

    seen, leaks = {}, set()
    for subset, subset_imgs in split.items():
        for im in subset_imgs:
            g = im.get("group_key", "")
            if g in seen and seen[g] != subset:
                leaks.add(g)
            seen[g] = subset
    print("[!] Leak between subsets detected!" if leaks
          else "[✓] No leak between subsets.")


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
    args = p.parse_args()

    if not args.root and not args.coco:
        p.error("provide --root or --coco")
    args.out.mkdir(parents=True, exist_ok=True)

    coco_path = args.coco
    if coco_path is None:
        coco_path = build_coco(args.root.expanduser().resolve(), args.out,
                               args.min_area, args.min_side,
                               not args.allow_missing_intrinsics)

    if args.classes.strip() == "standard10":
        manual = list(sd.STANDARD_10_CLASSES)
    else:
        manual = [c.strip() for c in args.classes.split(",") if c.strip()]
    make_splits(coco_path, args.out, args.top_k, manual, args.val, args.test,
                args.seed, min_instances=args.min_instances,
                splits_name=args.splits_name,
                base_assignment=args.base_assignment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
