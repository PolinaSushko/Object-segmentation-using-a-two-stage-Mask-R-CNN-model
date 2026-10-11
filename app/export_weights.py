"""Converts a training checkpoint of notebook 02 (model + optimizer + scheduler
+ history, ~367 MB, needs weights_only=False) into the weights-only file the
demo loads with `torch.load(..., weights_only=True)`:

    {"format": "vkr-maskrcnn-weights/1",
     "state_dict": {...},                 # model weights only
     "class_names": ["bed", ...],         # label id i -> class_names[i - 1]
     "meta": {"arch": ..., "num_classes": ..., "source": ..., ...}}

Usage:
    python app/export_weights.py ~/Downloads/maskrcnn_torchvision_std10/best.pth \
        app/weights/maskrcnn_std10.pt [--arch maskrcnn_resnet50_fpn_v2] [--fp16]

Only run it on checkpoints you trust: reading the training checkpoint
unpickles arbitrary objects (weights_only=False).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

FORMAT = "vkr-maskrcnn-weights/1"
STD10 = ["bed", "chair", "sofa", "table", "desk",
         "toilet", "dresser", "nightstand", "bookshelf", "bathtub"]


def _plain(value):
    """Converts numpy scalars & co. to plain Python for weights_only loading."""
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="training checkpoint (.pth)")
    ap.add_argument("dst", type=Path, help="output weights file (.pt)")
    ap.add_argument("--arch", default="maskrcnn_resnet50_fpn_v2",
                    help="torchvision builder the weights belong to")
    ap.add_argument("--fp16", action="store_true",
                    help="store float tensors in half precision (half the size; "
                         "the app casts back to float32)")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise SystemExit(f"{args.src}: no model state_dict found")
    if args.fp16:
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    state = {k: v.contiguous() for k, v in state.items()}

    num_classes = int(state["roi_heads.box_predictor.cls_score.weight"].shape[0])
    raw = ck.get("class_names") if isinstance(ck, dict) else None
    if isinstance(raw, dict):                     # {category_id: name}
        names = [str(raw[k]) for k in sorted(raw, key=int)]
    elif isinstance(raw, (list, tuple)):
        names = [str(n) for n in raw if n != "__background__"]
    else:
        names = STD10 if num_classes == len(STD10) + 1 else \
            [f"class_{i}" for i in range(1, num_classes)]
        print(f"warning: no class names in the checkpoint, using {names}")
    if len(names) != num_classes - 1:
        raise SystemExit(f"{len(names)} class names for {num_classes - 1} classes")

    meta = {"arch": args.arch, "num_classes": num_classes,
            "source": args.src.name,
            "source_sha1": hashlib.sha1(args.src.read_bytes()).hexdigest(),
            "dtype": "float16" if args.fp16 else "float32"}
    if isinstance(ck, dict):
        for key in ("epoch", "best_ap"):
            if key in ck:
                meta[key] = _plain(ck[key])

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "state_dict": state, "class_names": names,
                "meta": meta}, args.dst)

    # the app loads it with weights_only=True — make sure that works
    back = torch.load(args.dst, map_location="cpu", weights_only=True)
    assert back["class_names"] == names and len(back["state_dict"]) == len(state)
    size_mb = args.dst.stat().st_size / 2**20
    print(f"wrote {args.dst} ({size_mb:.0f} MB): {num_classes - 1} classes "
          f"{names}, meta {meta}")


if __name__ == "__main__":
    main()
