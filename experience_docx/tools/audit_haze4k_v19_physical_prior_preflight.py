#!/usr/bin/env python3
"""Inventory Haze4K physical-prior assets for v1.9.

The v1.9 route prefers using transmission and atmospheric-light evidence when
the cloud dataset exposes it. This preflight records what is actually present
instead of assuming those priors exist in the runtime workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz", ".mat"}
TRANSMISSION_HINTS = ("trans", "transmission", "tmap", "t_map", "t")
AIRLIGHT_HINTS = ("air", "airlight", "atmos", "atmospheric", "a_map", "light")
HAZY_HINTS = ("IN", "haze", "hazy")
GT_HINTS = ("GT", "gt", "clean", "clear")


def list_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    ]


def score_dir(path: Path, hints: tuple[str, ...]) -> int:
    text = "/".join(part.lower() for part in path.parts)
    return sum(1 for hint in hints if hint.lower() in text)


def candidate_dirs(root: Path, hints: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    if not root.is_dir():
        return rows
    for path, dirs, _files in os.walk(root):
        current = Path(path)
        score = score_dir(current, hints)
        if score <= 0:
            continue
        files = list_files(current)
        rows.append(
            {
                "path": str(current),
                "score": score,
                "file_count": len(files),
                "sample_files": ";".join(str(item.relative_to(current)) for item in files[:5]),
            }
        )
    rows.sort(key=lambda row: (int(row["score"]), int(row["file_count"])), reverse=True)
    return rows


def count_split_images(data_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "test"):
        split_root = data_dir / split
        split_info: dict[str, Any] = {}
        for label, hints in (("hazy", HAZY_HINTS), ("gt", GT_HINTS)):
            dirs = candidate_dirs(split_root, hints)
            best = dirs[0] if dirs else None
            split_info[label] = best
        out[split] = split_info
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--extra_roots", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    roots = [data_dir] + [Path(item) for item in args.extra_roots]
    transmission_rows: list[dict[str, Any]] = []
    airlight_rows: list[dict[str, Any]] = []
    for root in roots:
        transmission_rows.extend({**row, "search_root": str(root)} for row in candidate_dirs(root, TRANSMISSION_HINTS))
        airlight_rows.extend({**row, "search_root": str(root)} for row in candidate_dirs(root, AIRLIGHT_HINTS))

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "v19_transmission_candidates.csv", transmission_rows)
    write_csv(output_dir / "v19_airlight_candidates.csv", airlight_rows)

    best_transmission = transmission_rows[0] if transmission_rows else None
    best_airlight = airlight_rows[0] if airlight_rows else None
    status = "PHYSICAL_PRIORS_AVAILABLE"
    if not best_transmission or not best_airlight:
        status = "PHYSICAL_PRIOR_BLOCKED_MISSING_TRANSMISSION_OR_AIRLIGHT"
    payload = {
        "route": "ConvIR-Dehaze-v1.9-ConditionalTeacherGuided",
        "stage": "physical prior preflight",
        "locked_test_touched": False,
        "data_dir": str(data_dir),
        "extra_roots": args.extra_roots,
        "split_image_dirs": count_split_images(data_dir),
        "best_transmission_candidate": best_transmission,
        "best_airlight_candidate": best_airlight,
        "transmission_candidate_count": len(transmission_rows),
        "airlight_candidate_count": len(airlight_rows),
        "decision": status,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v19_physical_prior_preflight_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
