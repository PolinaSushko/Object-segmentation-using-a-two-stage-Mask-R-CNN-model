#!/usr/bin/env python3
"""
Допоміжний скрипт: виконує кроки 1.2, 1.6 і 1.7 без візуалізації.

Ноутбук `01_data_preparation.ipynb` є основним інструментом — він показує
графіки й проміжні перевірки. Цей скрипт потрібен лише тоді, коли дані вже
перевірено візуально і треба швидко перегенерувати вибірки з іншими
параметрами (наприклад, змінити кількість класів або зерно генератора),
не запускаючи весь ноутбук.

Приклади:
    # повний цикл
    python prepare_dataset.py --root ~/datasets/SUNRGBD --out outputs --top-k 10

    # лише перебудувати вибірки з готового COCO-файлу (швидко)
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import sunrgbd as sd  # noqa: E402


def build_coco(root: Path, out_dir: Path, min_area: float, min_side: float,
               require_intrinsics: bool) -> Path:
    """Крок 1.2: полігони SUN RGB-D -> COCO."""
    print(f"[*] Сканую {root} ...")
    scenes = [s for s in sd.find_scenes(root, require_annotation=True)]
    if not scenes:
        raise SystemExit("Сцен з анотаціями не знайдено — перевірте шлях.")
    sd.assign_groups(scenes)
    print(f"[*] Сцен до обробки: {len(scenes)}")

    images, annotations = [], []
    cat_ids, skipped = {}, Counter()
    ann_id = 1

    for img_id, scene in enumerate(scenes, start=1):
        if img_id % 250 == 0 or img_id == len(scenes):
            print(f"    {img_id}/{len(scenes)}", end="\r", flush=True)

        intr = sd.read_intrinsics(scene.intrinsics_path)
        if require_intrinsics and intr is None:
            skipped["без параметрів камери"] += 1
            continue
        try:
            with Image.open(scene.rgb_path) as im:
                width, height = im.size
        except OSError:
            skipped["не читається RGB"] += 1
            continue

        kept = 0
        for obj in sd.load_annotations(scene.annotation_path):
            if not obj.name or obj.name in sd.STUFF_CLASSES:
                skipped["категорія stuff"] += 1
                continue
            polys = [sd.clip_polygon(p, width, height) for p in obj.polygons]
            polys = [p for p in polys if len(p) >= 6]
            if not polys:
                skipped["вироджений полігон"] += 1
                continue
            bbox, area = sd.polygons_bbox_area(polys, width, height)
            if area < min_area or bbox[2] < min_side or bbox[3] < min_side:
                skipped["замалий об'єкт"] += 1
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
            skipped["сцена без придатних об'єктів"] += 1
            continue

        images.append({
            "id": img_id, "file_name": str(scene.rgb_path),
            "width": width, "height": height,
            "depth_file": str(scene.depth_path),
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

    print(f"[✓] {len(images)} зображень, {len(annotations)} екземплярів, {len(cat_ids)} категорій")
    if skipped:
        for reason, cnt in skipped.most_common():
            print(f"    відфільтровано — {reason}: {cnt}")
    return path


def make_splits(coco_path: Path, out_dir: Path, top_k: int, manual: list[str],
                val: float, test: float, seed: int) -> None:
    """Кроки 1.6-1.7: відбір класів і згрупований поділ вибірок."""
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(id2name[a["category_id"]] for a in coco["annotations"])

    keep = manual or [n for n, _ in counts.most_common(top_k)]
    missing = [n for n in keep if n not in counts]
    if missing:
        raise SystemExit(f"Немає таких класів: {', '.join(missing)}")

    coverage = sum(counts[n] for n in keep) / max(sum(counts.values()), 1) * 100
    print(f"[*] Обрано {len(keep)} категорій, покриття {coverage:.1f}%")

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

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test, n_val = int(round(len(keys) * test)), int(round(len(keys) * val))

    split = {"train": [], "val": [], "test": []}
    for i, g in enumerate(keys):
        subset = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        split[subset].extend(groups[g])

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset, subset_imgs in split.items():
        ids = {im["id"] for im in subset_imgs}
        sub_anns = [a for a in anns if a["image_id"] in ids]
        (split_dir / f"instances_{subset}.json").write_text(
            json.dumps({"info": {"subset": subset}, "licenses": [], "images": subset_imgs,
                        "annotations": sub_anns, "categories": new_cats}), encoding="utf-8")
        print(f"[✓] {subset:<6} {len(subset_imgs):>6} зображень, {len(sub_anns):>7} екземплярів")
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
    print("[!] Виявлено витік між вибірками!" if leaks
          else "[✓] Витоку між вибірками немає.")


def main() -> int:
    p = argparse.ArgumentParser(description="Підготовка SUN RGB-D (кроки 1.2, 1.6, 1.7)")
    p.add_argument("--root", type=Path, help="корінь набору даних SUN RGB-D")
    p.add_argument("--coco", type=Path, help="готовий COCO-файл (пропустити крок 1.2)")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--classes", type=str, default="", help="перелік класів через кому")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-area", type=float, default=200.0)
    p.add_argument("--min-side", type=float, default=8.0)
    p.add_argument("--allow-missing-intrinsics", action="store_true")
    args = p.parse_args()

    if not args.root and not args.coco:
        p.error("вкажіть --root або --coco")
    args.out.mkdir(parents=True, exist_ok=True)

    coco_path = args.coco
    if coco_path is None:
        coco_path = build_coco(args.root.expanduser().resolve(), args.out,
                               args.min_area, args.min_side,
                               not args.allow_missing_intrinsics)

    manual = [c.strip() for c in args.classes.split(",") if c.strip()]
    make_splits(coco_path, args.out, args.top_k, manual, args.val, args.test, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
