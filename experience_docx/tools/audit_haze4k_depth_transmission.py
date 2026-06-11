#!/usr/bin/env python3
"""Audit Haze4K cached depth against ground-truth transmission maps.

This tool is evidence-only: it reads Haze4K haze/transmission pairs and cached
Depth Anything arrays, then writes per-image and summary CSV/JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Missing one of {names} under {root}")


def list_images(data_dir: Path, split: str) -> list[str]:
    split_dir = data_dir / split
    haze_dir = first_existing_dir(split_dir, ("IN", "haze", "hazy"))
    return sorted(
        name for name in os.listdir(haze_dir)
        if name.lower().endswith(IMG_EXTENSIONS) and (haze_dir / name).is_file()
    )


def label_base_name(image_name: str) -> list[str]:
    stem, ext = os.path.splitext(image_name)
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    return list(dict.fromkeys(candidates))


def trans_path(data_dir: Path, split: str, image_name: str) -> Path:
    trans_dir = first_existing_dir(data_dir / split, ("trans", "Trans", "transmission"))
    for candidate in label_base_name(image_name):
        path = trans_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing transmission for {image_name} in {trans_dir}")


def haze_path(data_dir: Path, split: str, image_name: str) -> Path:
    haze_dir = first_existing_dir(data_dir / split, ("IN", "haze", "hazy"))
    return haze_dir / image_name


def depth_path(depth_cache_dir: Path, split: str, image_name: str) -> Path:
    candidates = [
        depth_cache_dir / split / f"{image_name.replace('/', '__')}.npy",
        depth_cache_dir / split / f"{image_name}.npy",
        depth_cache_dir / f"{image_name.replace('/', '__')}.npy",
        depth_cache_dir / f"{image_name}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing depth cache for {split}/{image_name}; tried={candidates}")


def parse_airlight_beta(image_name: str) -> tuple[float | None, float | None]:
    parts = Path(image_name).stem.split("_")
    airlight = None
    beta = None
    if len(parts) >= 2:
        try:
            airlight = float(parts[1])
        except ValueError:
            airlight = None
    if len(parts) >= 3:
        try:
            beta = float(parts[2])
        except ValueError:
            beta = None
    return airlight, beta


def read_depth(path: Path, size: tuple[int, int]) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    depth_img = Image.fromarray(depth, mode="F")
    if depth_img.size != size:
        depth_img = depth_img.resize(size, resample=Image.BICUBIC)
    depth = np.asarray(depth_img, dtype=np.float32)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    return np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)


def read_trans(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, resample=Image.BICUBIC)
    trans = np.asarray(img, dtype=np.float32)
    if float(np.max(trans)) > 1.5:
        trans = trans / 255.0
    return np.clip(trans, 1e-4, 1.0)


def read_haze_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    n = len(values)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def maybe_sample(arrays: list[np.ndarray], max_pixels: int, rng: np.random.Generator) -> list[np.ndarray]:
    total = arrays[0].size
    flat = [arr.reshape(-1) for arr in arrays]
    if max_pixels <= 0 or total <= max_pixels:
        return flat
    idx = rng.choice(total, size=max_pixels, replace=False)
    return [arr[idx] for arr in flat]


def best_alpha_metrics(depth: np.ndarray, trans: np.ndarray, alphas: np.ndarray) -> dict[str, float]:
    best = {"alpha": float("nan"), "l1": float("inf"), "rmse": float("inf")}
    for alpha in alphas:
        proxy = np.exp(-float(alpha) * depth)
        diff = proxy - trans
        l1 = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if rmse < best["rmse"]:
            best = {"alpha": float(alpha), "l1": l1, "rmse": rmse}
    return best


def masked_mean_abs(proxy: np.ndarray, trans: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return float("nan")
    return float(np.mean(np.abs(proxy[mask] - trans[mask])))


def audit_one(data_dir: Path, depth_cache_dir: Path, split: str, image_name: str, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    h_path = haze_path(data_dir, split, image_name)
    with Image.open(h_path) as img:
        size = img.size
    d_path = depth_path(depth_cache_dir, split, image_name)
    t_path = trans_path(data_dir, split, image_name)
    depth_full = read_depth(d_path, size)
    trans_full = read_trans(t_path, size)
    haze_gray = read_haze_gray(h_path, size)

    depth, trans = maybe_sample([depth_full, trans_full], args.max_pixels_per_image, rng)
    inv_depth = 1.0 - depth
    thickness = -np.log(np.clip(trans, 1e-4, 1.0))
    haze_density = 1.0 - trans
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps, dtype=np.float64)
    normal_best = best_alpha_metrics(depth, trans, alphas)
    invert_best = best_alpha_metrics(inv_depth, trans, alphas)

    normal_proxy_full = np.exp(-normal_best["alpha"] * depth_full)
    invert_proxy_full = np.exp(-invert_best["alpha"] * (1.0 - depth_full))
    use_invert = invert_best["rmse"] < normal_best["rmse"]
    best_proxy_full = invert_proxy_full if use_invert else normal_proxy_full

    grad_x = np.zeros_like(haze_gray)
    grad_y = np.zeros_like(haze_gray)
    grad_x[:, :-1] = np.abs(haze_gray[:, 1:] - haze_gray[:, :-1])
    grad_y[:-1, :] = np.abs(haze_gray[1:, :] - haze_gray[:-1, :])
    grad = grad_x + grad_y
    low_texture_cut = np.percentile(grad, args.low_texture_percentile)
    bright_cut = np.percentile(haze_gray, args.bright_percentile)
    low_texture_mask = grad <= low_texture_cut
    bright_mask = haze_gray >= bright_cut
    dense_cut = np.percentile(trans_full, args.dense_trans_percentile)
    dense_mask = trans_full <= dense_cut

    airlight, beta = parse_airlight_beta(image_name)
    orientation = "invert" if use_invert else "normal"
    corr_normal = spearman(depth, thickness)
    corr_invert = spearman(inv_depth, thickness)
    if not math.isnan(corr_invert) and not math.isnan(corr_normal):
        if corr_invert > corr_normal + args.orientation_margin:
            orientation = "invert"
        elif corr_normal > corr_invert + args.orientation_margin:
            orientation = "normal"
        else:
            orientation = "ambiguous"

    return {
        "split": split,
        "name": image_name,
        "airlight": airlight,
        "beta": beta,
        "width": size[0],
        "height": size[1],
        "depth_path": str(d_path),
        "trans_path": str(t_path),
        "spearman_depth_neglogt": corr_normal,
        "spearman_invert_depth_neglogt": corr_invert,
        "pearson_depth_1minus_t": pearson(depth, haze_density),
        "pearson_invert_depth_1minus_t": pearson(inv_depth, haze_density),
        "normal_best_alpha": normal_best["alpha"],
        "normal_best_l1": normal_best["l1"],
        "normal_best_rmse": normal_best["rmse"],
        "invert_best_alpha": invert_best["alpha"],
        "invert_best_l1": invert_best["l1"],
        "invert_best_rmse": invert_best["rmse"],
        "best_orientation": orientation,
        "best_proxy_l1_full": float(np.mean(np.abs(best_proxy_full - trans_full))),
        "best_proxy_rmse_full": float(np.sqrt(np.mean((best_proxy_full - trans_full) ** 2))),
        "low_texture_proxy_l1": masked_mean_abs(best_proxy_full, trans_full, low_texture_mask),
        "bright_region_proxy_l1": masked_mean_abs(best_proxy_full, trans_full, bright_mask),
        "dense_region_proxy_l1": masked_mean_abs(best_proxy_full, trans_full, dense_mask),
        "trans_mean": float(np.mean(trans_full)),
        "trans_std": float(np.std(trans_full)),
        "depth_mean": float(np.mean(depth_full)),
        "depth_std": float(np.std(depth_full)),
    }


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            out.append(float(value))
    return out


def summarize(rows: list[dict[str, Any]], split: str, args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {"split": split, "count": len(rows)}
    for key in [
        "spearman_depth_neglogt",
        "spearman_invert_depth_neglogt",
        "pearson_depth_1minus_t",
        "normal_best_l1",
        "normal_best_rmse",
        "invert_best_l1",
        "invert_best_rmse",
        "best_proxy_l1_full",
        "best_proxy_rmse_full",
        "low_texture_proxy_l1",
        "bright_region_proxy_l1",
        "dense_region_proxy_l1",
    ]:
        values = finite_values(rows, key)
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
    orientations: dict[str, int] = {}
    for row in rows:
        orientations[row["best_orientation"]] = orientations.get(row["best_orientation"], 0) + 1
    summary["orientation_counts"] = orientations
    corr = summary.get("spearman_depth_neglogt_median", float("nan"))
    inv_corr = summary.get("spearman_invert_depth_neglogt_median", float("nan"))
    if math.isfinite(inv_corr) and inv_corr > corr + args.orientation_margin:
        recommendation = "invert_depth"
    elif math.isfinite(corr) and corr >= 0.45:
        recommendation = "normal_depth_strong"
    elif math.isfinite(corr) and corr >= 0.15:
        recommendation = "weak_signal_use_confidence_gate"
    else:
        recommendation = "weak_or_bad_depth_prior_require_controls"
    summary["recommendation"] = recommendation
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "name", "airlight", "beta", "width", "height",
        "spearman_depth_neglogt", "spearman_invert_depth_neglogt",
        "pearson_depth_1minus_t", "pearson_invert_depth_1minus_t",
        "normal_best_alpha", "normal_best_l1", "normal_best_rmse",
        "invert_best_alpha", "invert_best_l1", "invert_best_rmse",
        "best_orientation", "best_proxy_l1_full", "best_proxy_rmse_full",
        "low_texture_proxy_l1", "bright_region_proxy_l1", "dense_region_proxy_l1",
        "trans_mean", "trans_std", "depth_mean", "depth_std",
        "depth_path", "trans_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images_per_split", type=int, default=0)
    parser.add_argument("--max_pixels_per_image", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=5.0)
    parser.add_argument("--alpha_steps", type=int, default=100)
    parser.add_argument("--orientation_margin", type=float, default=0.03)
    parser.add_argument("--low_texture_percentile", type=float, default=20.0)
    parser.add_argument("--bright_percentile", type=float, default=80.0)
    parser.add_argument("--dense_trans_percentile", type=float, default=25.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    depth_cache_dir = Path(args.depth_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    summaries = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        names = list_images(data_dir, split)
        if args.max_images_per_split > 0:
            names = names[: args.max_images_per_split]
        rows: list[dict[str, Any]] = []
        for idx, name in enumerate(names, start=1):
            try:
                rows.append(audit_one(data_dir, depth_cache_dir, split, name, args, rng))
            except Exception as exc:  # keep evidence for dataset mismatches
                errors.append({"split": split, "name": name, "error": repr(exc)})
            if idx % 250 == 0:
                print(f"audit_progress split={split} processed={idx}/{len(names)}", flush=True)
        write_csv(output_dir / f"dta_depth_transmission_audit_{split}.csv", rows)
        all_rows.extend(rows)
        summaries.append(summarize(rows, split, args))

    write_csv(output_dir / "dta_depth_transmission_audit_all.csv", all_rows)
    payload = {
        "stage": "dta_v2_depth_transmission_audit",
        "data_dir": str(data_dir),
        "depth_cache_dir": str(depth_cache_dir),
        "splits": args.splits,
        "row_count": len(all_rows),
        "error_count": len(errors),
        "summaries": summaries,
        "overall": summarize(all_rows, "all", args),
        "config": vars(args),
        "outputs": {
            "all_csv": "dta_depth_transmission_audit_all.csv",
            "summary_json": "dta_depth_transmission_audit_summary.json",
            "errors_csv": "dta_depth_transmission_audit_errors.csv" if errors else None,
        },
    }
    if errors:
        with (output_dir / "dta_depth_transmission_audit_errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "name", "error"])
            writer.writeheader()
            writer.writerows(errors)
    (output_dir / "dta_depth_transmission_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("DTA_DEPTH_TRANSMISSION_AUDIT_OK")


if __name__ == "__main__":
    main()
