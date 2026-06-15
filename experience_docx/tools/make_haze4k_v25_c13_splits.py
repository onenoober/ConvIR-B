#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from c13_common import first_dir, list_image_names, load_split_manifest, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--c12-split-manifest", type=Path, required=True)
    ap.add_argument("--c12-teacher-metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    c8 = load_split_manifest(args.c12_split_manifest)
    train_haze = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    train_gt = first_dir(args.data_dir / "train", ("GT", "gt"))
    train_names = list_image_names(train_haze)
    gt_stems = {p.stem for p in train_gt.iterdir() if p.is_file()}
    val_names = [row["name"] for row in c8["val"]]
    val_set = set(val_names)
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

    payload = {
        "route": "Haze4K v2.5 C13 A0-frozen residual distillation",
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "data_dir": str(args.data_dir),
        "train_haze_dir": str(train_haze),
        "train_gt_dir": str(train_gt),
        "source_c12_split_manifest": str(args.c12_split_manifest),
        "source_c12_counts": c8["counts"],
        "source_c12_teacher_metrics": str(args.c12_teacher_metrics),
        "counts": {
            "train_haze_total": len(train_names),
            "val_total": len(val_names),
            "val_regular": sum(row["split"] == "val_regular" for row in c8["val"]),
            "val_hard": sum(row["split"] == "val_hard" for row in c8["val"]),
            "train_core": len(train_core),
            "skipped_no_gt": len(skipped_no_gt),
        },
        "val": c8["val"],
        "train_core": train_core,
        "skipped_no_gt": skipped_no_gt,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "v25_c13_split_manifest.json", payload)
    (args.out_dir / "v25_c13_0_no_locked_status.txt").write_text(
        "locked_test_touched=false\n"
        "locked_per_image_read=false\n"
        "locked_informed_tuning=false\n"
        "teacher_source=C12_train_core_WD0375_cache_only\n",
        encoding="utf-8",
    )
    print("C13_SPLIT_MANIFEST_OK", json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
