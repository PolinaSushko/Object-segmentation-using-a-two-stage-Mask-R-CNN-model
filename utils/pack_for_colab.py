"""
Packs the prepared SUN RGB-D subset for training in Google Colab.

Takes one or more split directories (each with instances_train/val/test.json)
and bundles into a single zip:
  * <splits_dir_name>/*.json — COCO annotations for every class config
    (paths inside are already relative) and its dataset_version.json (the
    training notebooks check the dataset version before training);
  * the union of all RGB images and depth maps referenced by the splits,
    preserving the relative directory structure.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

SUBSETS = ("train", "val", "test")


def main() -> int:
    p = argparse.ArgumentParser(description="Pack data for Colab")
    p.add_argument("--root", type=Path, required=True, help="SUN RGB-D root")
    p.add_argument("--splits", type=Path, required=True, nargs="+",
                   help="one or more directories with instances_{train,val,test}.json; "
                        "each is stored in the zip under its own basename")
    p.add_argument("--out", type=Path, default=Path("sunrgbd_data.zip"))
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    names = [d.name for d in args.splits]
    if len(set(names)) != len(names):
        raise SystemExit("Split directories must have unique basenames")

    files: set[str] = set()          # relative paths of data files (union)
    for split_dir in args.splits:
        for subset in SUBSETS:
            split_path = split_dir / f"instances_{subset}.json"
            if not split_path.is_file():
                raise SystemExit(f"Not found: {split_path}")
            coco = json.loads(split_path.read_text(encoding="utf-8"))
            for im in coco["images"]:
                files.add(im["file_name"])
                files.add(im["depth_file"])
        version_path = split_dir / "dataset_version.json"
        if not version_path.is_file():
            raise SystemExit(f"Not found: {version_path} (regenerate the split with "
                             f"utils/prepare_dataset.py)")
        version = json.loads(version_path.read_text(encoding="utf-8"))
        print(f"[*] {split_dir.name}: dataset {version['version']}, "
              f"{version['classes']} classes, images {version['n_images']}")

    missing = [f for f in sorted(files) if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, first: {missing[0]}")

    print(f"[*] Split configs: {', '.join(names)}")
    print(f"[*] Data files (union): {len(files)}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for split_dir in args.splits:
            for subset in SUBSETS:
                sp = split_dir / f"instances_{subset}.json"
                zf.write(sp, f"{split_dir.name}/instances_{subset}.json")
            zf.write(split_dir / "dataset_version.json",
                     f"{split_dir.name}/dataset_version.json")
        for i, rel in enumerate(sorted(files), 1):
            if i % 2000 == 0 or i == len(files):
                print(f"    {i}/{len(files)}", end="\r", flush=True)
            zf.write(root / rel, rel)
    print()

    size_gb = args.out.stat().st_size / 1e9
    print(f"[✓] Done: {args.out}  ({size_gb:.2f} GB)")
    print("    Upload the archive to Google Drive; in Colab unpack it to /content/sunrgbd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
