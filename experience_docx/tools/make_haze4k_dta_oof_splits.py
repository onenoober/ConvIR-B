#!/usr/bin/env python3
"""Create deterministic Haze4K train-derived OOF splits for DTA route selection."""

import argparse
import json
import os
import random
from pathlib import Path

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def first_existing_dir(root, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of {names} found under {root}")


def list_images(data_dir):
    input_dir = first_existing_dir(Path(data_dir) / "train", ("IN", "haze", "hazy"))
    return sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith(IMG_EXTENSIONS) and (input_dir / name).is_file()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    names = list_images(args.data_dir)
    rng = random.Random(args.seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    folds = [[] for _ in range(args.folds)]
    for idx, name in enumerate(shuffled):
        folds[idx % args.folds].append(name)
    splits = {}
    all_names = set(names)
    for fold_idx, val_names in enumerate(folds):
        val = sorted(val_names)
        train = sorted(all_names.difference(val))
        splits[f"fold{fold_idx}_train"] = train
        splits[f"fold{fold_idx}_val"] = val
    payload = {
        "meta": {
            "data_dir": args.data_dir,
            "source_split": "train",
            "seed": args.seed,
            "folds": args.folds,
            "total_count": len(names),
            "fold_val_counts": [len(fold) for fold in folds],
            "note": "Use foldN_train for training and foldN_val for internal validation. Locked Haze4K test remains reserved for final fixed-config confirmation.",
        },
        "splits": splits,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2))
    print(f"wrote {output}")
    print("DTA_OOF_SPLITS_OK")


if __name__ == "__main__":
    main()
