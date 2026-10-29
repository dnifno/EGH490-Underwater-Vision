#!/usr/bin/env python3
"""Turn a CVAT "Segmentation mask 1.1" export into class-index masks.

Run from the zeroshot/ folder (the notebooks do this automatically):

    python ../utils/prepare_cvat_export.py --export-dir ../data/cvat_annotations

Expects, inside --export-dir:  JPEGImages/ (or images/),  SegmentationClass/,  labelmap.txt
Writes to --out-dir (default: the data.own_dir folder in the config):
    images/ -> copies of the annotated frames
    masks/ -> single-channel PNGs holding the class index (255 = not annotated)
    summary.json -> the frames kept and skipped

CVAT label names are translated to project classes through cvat.aliases in the config.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from labels import load_config

IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_labelmap(path):
    """labelmap.txt -> {lower-case label name: (r, g, b)}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, colour = line.split(":")[:2]
            out[name.strip().lower()] = tuple(int(c) for c in colour.split(","))
    return out


def colour_key(rgb):
    """Pack an (H, W, 3) uint8 array into one integer per pixel."""
    return (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)


def prepare(config_path, export_dir, out_dir=None):
    """Convert the export. Returns the list of kept frame stems."""
    cfg = load_config(config_path)
    names, ign = cfg["classes"]["names"], cfg["classes"]["ignore_index"]
    aliases = {k.lower(): v for k, v in cfg["cvat"]["aliases"].items()}
    exp = Path(export_dir)
    img_dir = next((exp / d for d in ("JPEGImages", "images") if (exp / d).is_dir()), None)
    seg_dir = exp / "SegmentationClass"
    if img_dir is None or not seg_dir.is_dir() or not (exp / "labelmap.txt").exists():
        sys.exit("Export must contain JPEGImages/ (or images/), SegmentationClass/ and labelmap.txt")

    lmap = parse_labelmap(exp / "labelmap.txt")
    print("labelmap.txt in the export:")
    colour2cls = {}
    for n, c in lmap.items():
        if n not in aliases:
            sys.exit(f"CVAT label '{n}' has no entry in cvat.aliases in the config. "
                     f"Add  '{n}: <class name or ignore>'.")
        target = aliases[n]
        cls = ign if target == "ignore" else names.index(target)
        colour2cls[(c[0] << 16) | (c[1] << 8) | c[2]] = cls
        print(f"  {n:20s} {c!s:16s} -> {target}")

    out = Path(out_dir or cfg["data"]["own_dir"])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    counts = np.zeros(256, np.int64)
    kept, skipped = [], []
    for seg in sorted(seg_dir.glob("*.png")):
        stem = seg.stem
        img = next((img_dir / f"{stem}{e}" for e in IMG_EXTS if (img_dir / f"{stem}{e}").exists()), None)
        if img is None:
            print(f"  skip {stem}: no matching image")
            skipped.append(stem)
            continue
        k = colour_key(np.array(Image.open(seg).convert("RGB")))
        idx = np.full(k.shape, ign, np.uint8)
        for ck, cls in colour2cls.items():
            idx[k == ck] = cls
        unknown = ~np.isin(k, list(colour2cls))
        if unknown.mean() > 0.01:
            print(f"  WARNING {stem}: {unknown.mean():.1%} of pixels have colours not in labelmap.txt (set to ignore)")
        if (idx != ign).sum() == 0:
            print(f"  skip {stem}: nothing annotated")
            skipped.append(stem)
            continue
        shutil.copy2(img, out / "images" / img.name)
        Image.fromarray(idx).save(out / "masks" / f"{stem}.png")
        counts += np.bincount(idx.ravel(), minlength=256)[:256]
        kept.append(stem)

    total_px = counts.sum()
    labelled = counts[:len(names)].sum()
    print(f"\n{len(kept)} frames kept, {len(skipped)} skipped")
    print(f"labelled share of all pixels: {100 * labelled / max(total_px, 1):.1f}%")
    print("labelled pixel share per class (class imbalance check):")
    for i, n in enumerate(names):
        print(f"  {n:15s} {counts[i]:>12d}  {100 * counts[i] / max(labelled, 1):5.1f}%")
    print(f"  not annotated   {counts[ign]:>12d}")
    with open(out / "summary.json", "w") as f:
        json.dump({"kept": kept, "skipped": skipped}, f, indent=2)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="defaults to data.own_dir in the config")
    args = ap.parse_args()
    prepare(args.config, args.export_dir, args.out_dir)


if __name__ == "__main__":
    main()
