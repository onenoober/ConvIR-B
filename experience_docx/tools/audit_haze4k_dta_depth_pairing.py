#!/usr/bin/env python3
"""Write deterministic image-depth pairing audits for DTA eval controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.data_load import DeblurDataset


def density_from_depth(path: str, invert: bool) -> float:
    depth = np.load(path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    if invert:
        depth = 1.0 - depth
    return float(np.mean(depth))


def quantile_bins(values: list[float], bins: int) -> list[int]:
    if not values:
        return []
    order = np.argsort(np.asarray(values), kind="mergesort")
    out = [0] * len(values)
    for rank, idx in enumerate(order):
        out[int(idx)] = min(bins - 1, int(rank * bins / len(values)))
    return out


def choose_source(idx: int, mode: str, bins: list[int], offset: int) -> int:
    n = len(bins)
    if mode in ("true", "normal", "invert", "zero"):
        return idx
    if mode == "shuffle_eval_fixed_perm":
        return (idx + offset) % n
    if mode == "shuffle_eval_same_density_bin":
        target_bin = bins[idx]
        for step in range(offset, offset + n):
            cand = (idx + step) % n
            if cand != idx and bins[cand] == target_bin:
                return cand
        return (idx + offset) % n
    if mode == "shuffle_eval_cross_density_bin":
        target_bin = max(bins) - bins[idx]
        for step in range(offset, offset + n):
            cand = (idx + step) % n
            if cand != idx and bins[cand] == target_bin:
                return cand
        return (idx + offset) % n
    raise ValueError(f"Unsupported pairing mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--root_split", default="train", choices=["train", "test"])
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--mode", required=True, choices=[
        "true",
        "normal",
        "invert",
        "zero",
        "shuffle_eval_fixed_perm",
        "shuffle_eval_same_density_bin",
        "shuffle_eval_cross_density_bin",
    ])
    parser.add_argument("--offset", type=int, default=137)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset = DeblurDataset(
        str(Path(args.data_dir) / args.root_split),
        "Haze4K",
        is_test=True,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    names = list(dataset.image_list)
    raw_density = [density_from_depth(dataset._depth_path(name), invert=False) for name in names]
    inv_density = [density_from_depth(dataset._depth_path(name), invert=True) for name in names]
    active_density = inv_density if args.mode in ("invert", "true") else raw_density
    bins = quantile_bins(active_density, args.bins)

    rows = []
    same_image = 0
    same_bin = 0
    for idx, name in enumerate(names):
        src_idx = choose_source(idx, args.mode, bins, args.offset)
        src_name = names[src_idx]
        if src_idx == idx:
            same_image += 1
        if bins[src_idx] == bins[idx]:
            same_bin += 1
        rows.append(
            {
                "image_index": idx,
                "image_name": name,
                "depth_index_used": src_idx,
                "depth_name_used": src_name,
                "depth_source_mode": args.mode,
                "same_image_depth": str(src_idx == idx).lower(),
                "density_bin": bins[idx],
                "depth_density_bin_used": bins[src_idx],
                "density_bin_match": str(bins[src_idx] == bins[idx]).lower(),
                "image_depth_density_raw": raw_density[idx],
                "source_depth_density_raw": raw_density[src_idx],
                "image_depth_density_invert": inv_density[idx],
                "source_depth_density_invert": inv_density[src_idx],
            }
        )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": args.mode,
        "count": len(rows),
        "same_image_count": same_image,
        "same_image_ratio": same_image / len(rows) if rows else 0.0,
        "density_bin_match_count": same_bin,
        "density_bin_match_ratio": same_bin / len(rows) if rows else 0.0,
        "offset": args.offset,
        "bins": args.bins,
        "split_json": args.split_json,
        "split_name": args.split_name,
        "outputs": {"csv": str(output_csv), "json": str(output_json)},
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("DTA_DEPTH_PAIRING_AUDIT_OK")


if __name__ == "__main__":
    main()
