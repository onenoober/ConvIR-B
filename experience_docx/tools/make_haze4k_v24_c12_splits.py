#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def read_c8_names(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"name": r["name"], "split": r["split"]} for r in rows]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--c8-per-image", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    train_haze = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    train_gt = first_dir(args.data_dir / "train", ("GT", "gt"))
    train_names = sorted(p.name for p in train_haze.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    gt_stems = {p.stem for p in train_gt.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}

    c8 = read_c8_names(args.c8_per_image)
    val_names = [r["name"] for r in c8]
    val_set = set(val_names)
    if len(val_names) != 600 or len(val_set) != 600:
        raise RuntimeError(f"expected 600 unique C8 val rows, got rows={len(val_names)} unique={len(val_set)}")
    missing = sorted(val_set - set(train_names))
    if missing:
        raise RuntimeError(f"C8 val names missing from train haze: {missing[:10]}")

    train_core = []
    skipped_no_gt = []
    for name in train_names:
        if name in val_set:
            continue
        stem = Path(name).stem
        base = stem.split("_")[0] if "_" in stem else stem
        if base not in gt_stems:
            skipped_no_gt.append(name)
            continue
        train_core.append(name)

    split_payload = {
        "route": "Haze4K v2.4 C12 WD0375 distillation",
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "data_dir": str(args.data_dir),
        "train_haze_dir": str(train_haze),
        "train_gt_dir": str(train_gt),
        "source_c8_per_image": str(args.c8_per_image),
        "source_c8_sha256": sha256(args.c8_per_image),
        "counts": {
            "train_haze_total": len(train_names),
            "val_regular": sum(r["split"] == "val_regular" for r in c8),
            "val_hard": sum(r["split"] == "val_hard" for r in c8),
            "val_total": len(val_names),
            "train_core": len(train_core),
            "skipped_no_gt": len(skipped_no_gt),
        },
        "val": c8,
        "train_core": train_core,
        "skipped_no_gt": skipped_no_gt,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "v24_c12_split_manifest.json", split_payload)
    (args.out_dir / "v24_c12_0_no_locked_status.txt").write_text(
        "locked_test_touched=false\n"
        "locked_per_image_read=false\n"
        "locked_informed_tuning=false\n"
        "distillation_target=WD0375_train_core_only\n",
        encoding="utf-8",
    )
    print("C12_SPLIT_MANIFEST_OK", json.dumps(split_payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
