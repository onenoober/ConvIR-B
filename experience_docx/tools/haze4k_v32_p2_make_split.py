#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _sha256_lines(values):
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label_path(source_train, image_name):
    stem = image_name.split("_")[0]
    candidates = [
        source_train / "gt" / image_name,
        source_train / "gt" / f"{stem}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT label for {image_name}; tried {candidates}")


def _link_file(src, dst):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == str(src):
            return
        raise FileExistsError(f"Refusing to overwrite existing split file: {dst}")
    os.symlink(src, dst)


def _materialize_split(split_root, split_name, rows, source_train):
    haze_dir = split_root / split_name / "haze"
    gt_dir = split_root / split_name / "gt"
    haze_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    linked_haze = 0
    linked_gt = set()
    for row in rows:
        image_name = row["image_name"]
        haze_src = source_train / "haze" / image_name
        if not haze_src.is_file():
            raise FileNotFoundError(f"Missing haze image: {haze_src}")
        gt_src = _label_path(source_train, image_name)
        gt_name = gt_src.name
        _link_file(haze_src, haze_dir / image_name)
        if gt_name not in linked_gt:
            _link_file(gt_src, gt_dir / gt_name)
            linked_gt.add(gt_name)
        linked_haze += 1
    return {"haze_files": linked_haze, "gt_files": len(linked_gt)}


def _compact_split(rows):
    names = [row["image_name"] for row in rows]
    return {
        "count": len(names),
        "names_sha256": _sha256_lines(names),
        "first10": names[:10],
        "last10": names[-10:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31_csv", required=True)
    parser.add_argument("--source_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--val_fold", default="0")
    args = parser.parse_args()

    v31_csv = Path(args.v31_csv)
    source_train = Path(args.source_data_dir) / "train"
    split_root = Path(args.output_dir)
    summary_output = Path(args.summary_output)

    rows = list(csv.DictReader(v31_csv.open(newline="", encoding="utf-8")))
    rows = sorted(rows, key=lambda row: row["image_name"])
    if len(rows) != 600:
        raise ValueError(f"Expected 600 v3.1 rows, got {len(rows)}")

    val_rows = [row for row in rows if row["fold_id"] == args.val_fold]
    train_rows = [row for row in rows if row["fold_id"] != args.val_fold]
    if len(val_rows) != 120 or len(train_rows) != 480:
        raise ValueError(
            f"Unexpected P2 split sizes: train={len(train_rows)} val={len(val_rows)}"
        )

    overlap = {row["image_name"] for row in train_rows} & {row["image_name"] for row in val_rows}
    if overlap:
        raise ValueError(f"Train/val overlap: {sorted(overlap)[:5]}")

    train_link = _materialize_split(split_root, "train", train_rows, source_train)
    val_link = _materialize_split(split_root, "test", val_rows, source_train)

    summary = {
        "route_id": "haze4k_v3_2_convir_wd_full_model_line_20260707",
        "phase": "P2 train-derived split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "v31_csv": str(v31_csv),
        "source_data_dir": str(Path(args.source_data_dir)),
        "output_dir": str(split_root),
        "val_fold": args.val_fold,
        "train": _compact_split(train_rows),
        "validation": _compact_split(val_rows),
        "materialized": {
            "train": train_link,
            "test": val_link,
        },
        "locked_test_touched": False,
        "note": "Split uses original Haze4K train only; output_dir/test is a train-derived validation directory.",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
