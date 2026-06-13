#!/usr/bin/env python3
"""Extract deployable Haze4K image quality/color/texture features for DTA-v3.7."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from select_haze4k_dta_v37_u_tqs_mix_phase_a import read_csv_rows, write_csv


def percentile(arr: np.ndarray, pct: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def image_path(data_dir: Path, image_id: str) -> Path:
    path = data_dir / "train" / "haze" / image_id
    if path.exists():
        return path
    path = data_dir / "test" / "haze" / image_id
    if path.exists():
        return path
    raise FileNotFoundError(image_id)


def trans_path(data_dir: Path, image_id: str) -> Path | None:
    base = image_id.split("_", 1)[0] + ".png"
    for split in ("train", "test"):
        path = data_dir / split / "trans" / base
        if path.exists():
            return path
    return None


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_gray(path: Path, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def gradient_mag(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    return np.sqrt(gx * gx + gy * gy)


def laplacian(y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:-1, 1:-1] = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    return out


def features_for(data_dir: Path, image_id: str, max_side: int) -> dict[str, Any]:
    rgb = load_rgb(image_path(data_dir, image_id), max_side=max_side)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    sat = (cmax - cmin) / np.maximum(cmax, 1e-6)
    dark = cmin
    grad = gradient_mag(y)
    lap = laplacian(y)
    high_bright = y > 0.85
    low_texture = grad < 0.015
    sky_proxy = high_bright & (sat < 0.25) & low_texture
    color_means = [float(np.mean(ch)) for ch in (r, g, b)]
    color_stds = [float(np.std(ch)) for ch in (r, g, b)]
    out = {
        "image_id": image_id,
        "q_luma_mean": float(np.mean(y)),
        "q_luma_std": float(np.std(y)),
        "q_luma_p05": percentile(y, 5),
        "q_luma_p50": percentile(y, 50),
        "q_luma_p95": percentile(y, 95),
        "q_contrast_p95_p05": percentile(y, 95) - percentile(y, 5),
        "q_saturation_mean": float(np.mean(sat)),
        "q_saturation_std": float(np.std(sat)),
        "dark_channel_mean": float(np.mean(dark)),
        "dark_channel_p05": percentile(dark, 5),
        "dark_channel_p50": percentile(dark, 50),
        "edge_grad_mean": float(np.mean(grad)),
        "edge_grad_p90": percentile(grad, 90),
        "edge_laplacian_var": float(np.var(lap)),
        "texture_low_ratio": float(np.mean(low_texture)),
        "sky_highbright_lowtex_ratio": float(np.mean(sky_proxy)),
        "highlight_ratio": float(np.mean(high_bright)),
        "color_r_mean": color_means[0],
        "color_g_mean": color_means[1],
        "color_b_mean": color_means[2],
        "color_r_std": color_stds[0],
        "color_g_std": color_stds[1],
        "color_b_std": color_stds[2],
        "color_cast_abs_rg": abs(color_means[0] - color_means[1]),
        "color_cast_abs_rb": abs(color_means[0] - color_means[2]),
        "color_cast_abs_gb": abs(color_means[1] - color_means[2]),
    }
    tpath = trans_path(data_dir, image_id)
    if tpath is not None:
        trans = load_gray(tpath, max_side=max_side)
        out.update({
            "trans_file_mean": float(np.mean(trans)),
            "trans_file_std": float(np.std(trans)),
            "trans_file_p10": percentile(trans, 10),
            "trans_file_p90": percentile(trans, 90),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_action_table", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--max_side", type=int, default=384)
    args = parser.parse_args()

    rows = read_csv_rows(args.input_action_table)
    image_ids = sorted({str(row["image_id"]) for row in rows})
    out = []
    missing = []
    for idx, image_id in enumerate(image_ids, start=1):
        try:
            out.append(features_for(args.data_dir, image_id, args.max_side))
        except Exception as exc:  # keep the audit explicit instead of hiding missing files
            missing.append({"image_id": image_id, "error": str(exc)})
        if idx % 250 == 0:
            print(f"FEATURE_PROGRESS {idx}/{len(image_ids)}", flush=True)
    write_csv(args.output_csv, out)
    if missing:
        write_csv(args.output_csv.with_suffix(".missing.csv"), missing)
    print(f"DTA_V3_7_QUALITY_FEATURES_OK images={len(out)} missing={len(missing)} output={args.output_csv}")


if __name__ == "__main__":
    main()
