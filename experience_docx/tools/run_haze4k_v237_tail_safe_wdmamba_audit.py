#!/usr/bin/env python3
"""Haze4K v2.37 tail-safe same-context WDMamba audit helpers."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROUTE_ID = "haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706"
ALPHAS = (0.125, 0.25, 0.375, 0.5)
BASE_ALPHA = 0.5
EASY_ALPHA = 0.25
SEVERE_THRESHOLD = -0.30
STRONG_REFERENCE_REGRESSION_THRESHOLD = -0.05


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cvar_low(values: list[float], frac: float = 0.05) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(len(ordered) * frac))
    return statistics.mean(ordered[:k])


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def alpha_label(alpha: float) -> str:
    return f"a{str(alpha).replace('.', 'p')}"


def load_tensor(path: str | Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict) and "tensor" in tensor:
        tensor = tensor["tensor"]
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor at {path}")
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected CHW/NCHW tensor at {path}, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(tensor, 0.0, 1.0)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    if tuple(pred.shape) != tuple(label.shape):
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} label={tuple(label.shape)}")
    mse = (torch.clamp(pred, 0, 1) - torch.clamp(label, 0, 1)).pow(2).flatten(1).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse).item())


def blend(a0: torch.Tensor, expert: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(a0 + alpha * (expert - a0), 0.0, 1.0)


def lowpass(tensor: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def tensor_rms(tensor: torch.Tensor) -> float:
    return float(tensor.pow(2).mean().clamp_min(1e-12).sqrt().item())


def luma(tensor: torch.Tensor) -> torch.Tensor:
    return 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]


def edge_energy(gray: torch.Tensor) -> float:
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
    return float((dx + dy).item())


def image_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    gray = luma(tensor)
    saturation = tensor.max(dim=1, keepdim=True).values - tensor.min(dim=1, keepdim=True).values
    dark = tensor.min(dim=1, keepdim=True).values
    return {
        f"{prefix}_luma_mean": float(gray.mean().item()),
        f"{prefix}_luma_std": float(gray.std(unbiased=False).item()),
        f"{prefix}_saturation_mean": float(saturation.mean().item()),
        f"{prefix}_dark_channel_mean": float(dark.mean().item()),
        f"{prefix}_edge_energy": edge_energy(gray),
        f"{prefix}_lowfreq_energy": tensor_rms(lowpass(gray)),
    }


def pair_stats(input_tensor: torch.Tensor, a0: torch.Tensor) -> dict[str, float]:
    diff = a0 - input_tensor
    low = lowpass(diff)
    high = diff - low
    return {
        "input_A0_abs_mean": float(diff.abs().mean().item()),
        "input_A0_lowfreq_abs_mean": float(low.abs().mean().item()),
        "input_A0_hf_abs_mean": float(high.abs().mean().item()),
    }


def parse_haze_params(image_id: str) -> tuple[str, float | None, float | None]:
    parts = image_id.rsplit("_", 2)
    if len(parts) != 3:
        return "", None, None
    try:
        return f"{parts[1]}_{parts[2]}", float(parts[1]), float(parts[2])
    except ValueError:
        return "", None, None


def hardness_bucket(row: dict[str, Any]) -> str:
    bucket = str(row.get("hardness_bucket") or row.get("selection_source_bucket") or "")
    if bucket.startswith("hard"):
        return "hard"
    if bucket.startswith("easy") or bucket.startswith("strong"):
        return "easy"
    return "mid"


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    hard = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "hard"]
    easy = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "easy"]
    mid = [float(row[key]) for row in rows if row.get(key) not in ("", None) and hardness_bucket(row) == "mid"]
    strong = [
        float(row[key])
        for row in rows
        if row.get(key) not in ("", None) and row.get("strong_reference_bucket") == "strong_reference"
    ]
    severe = [value for value in values if value <= SEVERE_THRESHOLD]
    strong_regress = [value for value in strong if value < STRONG_REFERENCE_REGRESSION_THRESHOLD]
    return {
        "mean": mean(values),
        "hard": mean(hard),
        "easy": mean(easy),
        "mid": mean(mid),
        "p05": percentile(values, 5),
        "CVaR5": cvar_low(values),
        "negative_count": sum(1 for value in values if value < 0),
        "severe_count": len(severe),
        "severe_rate": len(severe) / len(values) if values else None,
        "strong_reference_count": len(strong),
        "strong_reference_regression_count": len(strong_regress),
        "strong_reference_regression_rate": len(strong_regress) / len(strong) if strong else None,
        "sample_count": len(values),
        "severe_threshold_dB": SEVERE_THRESHOLD,
        "strong_reference_regression_threshold_dB": STRONG_REFERENCE_REGRESSION_THRESHOLD,
    }


def gate_tail_stats(stats: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.30)
        and (stats.get("hard") is not None and stats["hard"] >= 0.50)
        and (stats.get("easy") is not None and stats["easy"] >= -0.03)
        and (stats.get("p05") is not None and stats["p05"] >= -0.05)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.10)
        and stats.get("severe_rate") == 0
        and stats.get("strong_reference_regression_rate") is not None
        and stats["strong_reference_regression_rate"] <= 0.02
    )


def gate_p0(stats: dict[str, Any], cache_sha_coverage: float, fold_pass_count: int) -> bool:
    return bool(
        stats.get("sample_count") == 600
        and cache_sha_coverage == 1.0
        and gate_tail_stats(stats)
        and fold_pass_count == 5
    )


def gate_p2(stats: dict[str, Any], meta: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 1.00)
        and (stats.get("hard") is not None and stats["hard"] >= 2.00)
        and (stats.get("easy") is not None and stats["easy"] >= -0.01)
        and (stats.get("p05") is not None and stats["p05"] >= -0.01)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.01)
        and stats.get("severe_rate") == 0
        and stats.get("strong_reference_regression_rate") == 0
        and int(meta.get("fold_pass_count", 0)) == 5
        and int(meta.get("eligible_count", 0)) >= 300
        and (meta.get("hard_eligible_rate") is not None and meta["hard_eligible_rate"] >= 0.80)
        and meta.get("negative_preservation_rate") == 1.0
        and meta.get("severe_preservation_rate") == 1.0
    )


def gate_train_mask(stats: dict[str, Any], meta: dict[str, Any], eligible_min: int) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 1.00)
        and (stats.get("hard") is not None and stats["hard"] >= 2.00)
        and (stats.get("easy") is not None and stats["easy"] >= -0.01)
        and (stats.get("p05") is not None and stats["p05"] >= -0.01)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.01)
        and stats.get("severe_rate") == 0
        and stats.get("strong_reference_regression_rate") == 0
        and int(meta.get("eligible_count", 0)) >= eligible_min
        and (meta.get("hard_eligible_rate") is not None and meta["hard_eligible_rate"] >= 0.80)
        and meta.get("negative_preservation_rate") == 1.0
        and meta.get("severe_preservation_rate") == 1.0
    )


def gate_p3_holdout(stats: dict[str, Any], meta: dict[str, Any], eligible_min: int) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.80)
        and (stats.get("hard") is not None and stats["hard"] >= 1.50)
        and (stats.get("easy") is not None and stats["easy"] >= -0.01)
        and (stats.get("p05") is not None and stats["p05"] >= -0.01)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.01)
        and stats.get("severe_rate") == 0
        and stats.get("strong_reference_regression_rate") == 0
        and int(meta.get("eligible_count", 0)) >= eligible_min
    )


def assign_buckets(rows: list[dict[str, Any]]) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(str(row["image_id"]), row)
    unique = list(by_id.values())
    sorted_rows = sorted(unique, key=lambda row: float(row["A0_same_context_psnr"]))
    bucket_count = max(1, len(sorted_rows) // 4)
    hard_ids = {str(row["image_id"]) for row in sorted_rows[:bucket_count]}
    easy_ids = {str(row["image_id"]) for row in sorted_rows[-bucket_count:]}
    strong_cut = percentile([float(row["A0_same_context_psnr"]) for row in unique], 75) or 0.0
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in hard_ids:
            row["hardness_bucket"] = "hard"
        elif image_id in easy_ids:
            row["hardness_bucket"] = "easy"
        else:
            row["hardness_bucket"] = "mid"
        row["easy_bucket"] = "easy_top25" if image_id in easy_ids else "not_easy_top25"
        row["strong_reference_cut_psnr"] = strong_cut
        row["strong_reference_bucket"] = (
            "strong_reference"
            if float(row["A0_same_context_psnr"]) >= strong_cut
            else "not_strong_reference"
        )


def fold_map_from_manifest(path: Path | None, image_ids: list[str]) -> dict[str, int]:
    if path and path.exists():
        out: dict[str, int] = {}
        for row in read_csv(path):
            image_id = row.get("image_id") or Path(row.get("image_name", "")).stem
            fold = fnum(row.get("fold_id"))
            if image_id and fold is not None:
                out[str(image_id)] = int(fold)
        if out:
            return out
    return {image_id: index % 5 for index, image_id in enumerate(sorted(image_ids))}


def unique_manifest_rows(v235_manifest: Path) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    existing: dict[tuple[str, float], dict[str, str]] = {}
    for row in read_csv(v235_manifest):
        image_id = row.get("image_id") or Path(row.get("image_name", "")).stem
        if not image_id:
            continue
        by_id.setdefault(image_id, row)
        alpha = fnum(row.get("blend_alpha"))
        if alpha is not None:
            existing[(image_id, round(alpha, 6))] = row
    rows = []
    for image_id, row in by_id.items():
        out = dict(row)
        for alpha in ALPHAS:
            rec = existing.get((image_id, round(alpha, 6)))
            if rec:
                label = alpha_label(alpha)
                out[f"{label}_manifest_blend_full_PSNR"] = rec.get("blend_full_PSNR", "")
                out[f"{label}_manifest_blend_output_path"] = rec.get("blend_output_path", "")
                out[f"{label}_manifest_blend_output_sha256"] = rec.get("blend_output_sha256", "")
        rows.append(out)
    rows.sort(key=lambda item: item.get("image_id", ""))
    return rows


def run_p0_alpha_sweep(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = unique_manifest_rows(args.v235_manifest)
    fold_map = fold_map_from_manifest(args.fold_manifest, [row["image_id"] for row in source_rows])
    rows: list[dict[str, Any]] = []
    cache_ready = 0
    existing_diffs: dict[str, list[float]] = {alpha_label(alpha): [] for alpha in ALPHAS}
    for index, rec in enumerate(source_rows, start=1):
        image_id = rec["image_id"]
        a0_path = rec.get("A0_full_output_path", "")
        wd_path = rec.get("WDMamba_full_output_path", "")
        gt_path = rec.get("gt_path", "")
        required = [
            a0_path,
            wd_path,
            gt_path,
            rec.get("A0_full_output_sha256", ""),
            rec.get("WDMamba_full_output_sha256", ""),
            rec.get("gt_sha256", ""),
            rec.get("input_sha256", ""),
        ]
        if all(required) and Path(a0_path).exists() and Path(wd_path).exists() and Path(gt_path).exists():
            cache_ready += 1
        a0 = load_tensor(a0_path)
        wd = load_tensor(wd_path)
        gt = load_image_tensor(gt_path)
        a0_psnr = psnr(a0, gt)
        wd_psnr = psnr(wd, gt)
        for alpha in ALPHAS:
            label = alpha_label(alpha)
            teacher = blend(a0, wd, alpha)
            teacher_psnr = psnr(teacher, gt)
            existing_psnr = fnum(rec.get(f"{label}_manifest_blend_full_PSNR"))
            if existing_psnr is not None:
                existing_diffs[label].append(abs(existing_psnr - teacher_psnr))
            delta = teacher_psnr - a0_psnr
            rows.append({
                "image_id": image_id,
                "image_name": rec.get("image_name", ""),
                "sample_index": rec.get("sample_index", ""),
                "table_split": rec.get("table_split", ""),
                "alpha": alpha,
                "alpha_label": label,
                "context_contract": "full_image_same_context",
                "A0_same_context_psnr": a0_psnr,
                "WDMamba_full_psnr": wd_psnr,
                "teacher_same_context_psnr": teacher_psnr,
                "teacher_delta_vs_A0": delta,
                "input_path": rec.get("input_path", ""),
                "gt_path": gt_path,
                "A0_full_output_path": a0_path,
                "WDMamba_full_output_path": wd_path,
                "input_sha256": rec.get("input_sha256", ""),
                "gt_sha256": rec.get("gt_sha256", ""),
                "A0_full_output_sha256": rec.get("A0_full_output_sha256", ""),
                "WDMamba_full_output_sha256": rec.get("WDMamba_full_output_sha256", ""),
                "existing_blend_output_path": rec.get(f"{label}_manifest_blend_output_path", ""),
                "existing_blend_output_sha256": rec.get(f"{label}_manifest_blend_output_sha256", ""),
                "existing_blend_recompute_abs_diff": abs(existing_psnr - teacher_psnr) if existing_psnr is not None else "",
                "fold_id": fold_map.get(image_id, (index - 1) % 5),
                "negative": delta < 0,
                "severe": delta <= SEVERE_THRESHOLD,
                "locked_test_touched": False,
            })
        print(f"v237_p0_progress {index}/{len(source_rows)} {image_id}", flush=True)
    assign_buckets(rows)
    for row in rows:
        row["strong_reference_regression"] = (
            row["strong_reference_bucket"] == "strong_reference"
            and float(row["teacher_delta_vs_A0"]) < STRONG_REFERENCE_REGRESSION_THRESHOLD
        )

    fields = [
        "image_id",
        "image_name",
        "sample_index",
        "table_split",
        "alpha",
        "alpha_label",
        "context_contract",
        "A0_same_context_psnr",
        "WDMamba_full_psnr",
        "teacher_same_context_psnr",
        "teacher_delta_vs_A0",
        "hardness_bucket",
        "easy_bucket",
        "strong_reference_bucket",
        "strong_reference_cut_psnr",
        "fold_id",
        "negative",
        "severe",
        "strong_reference_regression",
        "input_path",
        "gt_path",
        "A0_full_output_path",
        "WDMamba_full_output_path",
        "input_sha256",
        "gt_sha256",
        "A0_full_output_sha256",
        "WDMamba_full_output_sha256",
        "existing_blend_output_path",
        "existing_blend_output_sha256",
        "existing_blend_recompute_abs_diff",
        "locked_test_touched",
    ]
    write_csv(args.out_dir / "v237_p0_alpha_safety_sweep_per_image.csv", rows, fields)

    cache_sha_coverage = cache_ready / len(source_rows) if source_rows else 0.0
    alpha_summaries = []
    passing_alphas = []
    for alpha in ALPHAS:
        subset = [row for row in rows if float(row["alpha"]) == alpha]
        stats = summarize_values(subset, "teacher_delta_vs_A0")
        fold_summaries = []
        fold_pass_count = 0
        for fold_id in range(5):
            fold_subset = [row for row in subset if int(row["fold_id"]) == fold_id]
            fold_stats = summarize_values(fold_subset, "teacher_delta_vs_A0")
            fold_pass = gate_tail_stats(fold_stats)
            fold_pass_count += int(fold_pass)
            fold_summaries.append({"fold_id": fold_id, "gate_pass": fold_pass, **fold_stats})
        gate_pass = gate_p0(stats, cache_sha_coverage, fold_pass_count)
        if gate_pass:
            passing_alphas.append(alpha)
        label = alpha_label(alpha)
        alpha_summaries.append({
            "alpha": alpha,
            "alpha_label": label,
            "cache_sha_coverage": cache_sha_coverage,
            "fold_pass": f"{fold_pass_count}/5",
            "fold_pass_count": fold_pass_count,
            "gate_pass": gate_pass,
            "existing_blend_recompute_max_abs_diff": max(existing_diffs[label]) if existing_diffs[label] else None,
            "existing_blend_recompute_mean_abs_diff": mean(existing_diffs[label]),
            "summary": stats,
            "fold_summary": fold_summaries,
        })
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P0 alpha/blend safety sweep",
        "source_manifest": str(args.v235_manifest),
        "fold_manifest": str(args.fold_manifest) if args.fold_manifest else "",
        "locked_test_touched": False,
        "image_count": len(source_rows),
        "row_count": len(rows),
        "cache_sha_coverage": cache_sha_coverage,
        "gate": {
            "image_count": 600,
            "cache_sha_coverage": 1.0,
            "mean_delta_min_dB": 0.30,
            "hard_delta_min_dB": 0.50,
            "easy_delta_min_dB": -0.03,
            "p05_min_dB": -0.05,
            "CVaR5_min_dB": -0.10,
            "severe_rate": 0,
            "strong_reference_regression_rate_max": 0.02,
            "fold_pass": "5/5",
        },
        "alpha_summaries": alpha_summaries,
        "passing_alphas": passing_alphas,
        "any_alpha_pass": bool(passing_alphas),
        "decision": "P0_PASS_UNMASKED_ALPHA_SUBSTRATE" if passing_alphas else "P0_FAIL_ALL_UNMASKED_ALPHA_SUBSTRATES",
    }
    write_json(args.out_dir / "v237_p0_alpha_safety_sweep_summary.json", payload)
    write_json(args.out_dir / "v237_p0_closeout.json", {
        "route_id": ROUTE_ID,
        "phase": "P0",
        "decision": payload["decision"],
        "gate_pass": bool(passing_alphas),
        "passing_alphas": passing_alphas,
        "p1_tail_atlas_authorized": not bool(passing_alphas),
        "p2_mask_preservation_authorized": not bool(passing_alphas),
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_P0_ALPHA_SAFETY_SWEEP_OK")


def rows_by_alpha(p0_rows: list[dict[str, str]]) -> dict[str, dict[float, dict[str, str]]]:
    out: dict[str, dict[float, dict[str, str]]] = {}
    for row in p0_rows:
        alpha = fnum(row.get("alpha"))
        if alpha is not None:
            out.setdefault(row["image_id"], {})[round(alpha, 6)] = row
    return out


def run_p1_tail_atlas(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p0_rows = read_csv(args.p0_csv)
    grouped = rows_by_alpha(p0_rows)
    atlas: list[dict[str, Any]] = []
    for index, (image_id, alpha_rows) in enumerate(sorted(grouped.items()), start=1):
        base_row = alpha_rows[round(BASE_ALPHA, 6)]
        input_tensor = load_image_tensor(base_row["input_path"])
        a0 = load_tensor(base_row["A0_full_output_path"])
        wd = load_tensor(base_row["WDMamba_full_output_path"])
        haze_text, haze_beta, haze_airlight = parse_haze_params(image_id)
        base_features: dict[str, Any] = {"image_id": image_id, "image_name": base_row.get("image_name", "")}
        base_features.update(image_stats(input_tensor, "input"))
        base_features.update(image_stats(a0, "A0"))
        base_features.update(pair_stats(input_tensor, a0))
        base_features.update({
            "haze_params_from_id_if_available": haze_text,
            "haze_beta_from_id": haze_beta if haze_beta is not None else "",
            "haze_airlight_from_id": haze_airlight if haze_airlight is not None else "",
        })
        for alpha in ALPHAS:
            row = alpha_rows[round(alpha, 6)]
            delta_tensor = blend(a0, wd, alpha) - a0
            low = lowpass(delta_tensor)
            high = delta_tensor - low
            atlas.append({
                **base_features,
                "alpha": alpha,
                "alpha_label": row.get("alpha_label", ""),
                "A0_same_context_psnr": fnum(row.get("A0_same_context_psnr")),
                "teacher_same_context_psnr": fnum(row.get("teacher_same_context_psnr")),
                "teacher_delta_vs_A0": fnum(row.get("teacher_delta_vs_A0")),
                "hardness_bucket": row.get("hardness_bucket", ""),
                "strong_reference_bucket": row.get("strong_reference_bucket", ""),
                "fold_id": row.get("fold_id", ""),
                "is_negative": boolish(row.get("negative")),
                "is_severe": boolish(row.get("severe")),
                "is_strong_reference_regression": boolish(row.get("strong_reference_regression")),
                "teacher_minus_A0_LL_energy": tensor_rms(low),
                "teacher_minus_A0_HF_energy": tensor_rms(high),
                "locked_test_touched": False,
            })
        if index % 25 == 0 or index == len(grouped):
            print(f"v237_p1_progress {index}/{len(grouped)}", flush=True)

    fields = [
        "image_id",
        "image_name",
        "alpha",
        "alpha_label",
        "A0_same_context_psnr",
        "teacher_same_context_psnr",
        "teacher_delta_vs_A0",
        "hardness_bucket",
        "strong_reference_bucket",
        "fold_id",
        "is_negative",
        "is_severe",
        "is_strong_reference_regression",
        "input_luma_mean",
        "input_luma_std",
        "input_saturation_mean",
        "input_dark_channel_mean",
        "input_edge_energy",
        "input_lowfreq_energy",
        "A0_luma_mean",
        "A0_luma_std",
        "A0_saturation_mean",
        "A0_dark_channel_mean",
        "A0_edge_energy",
        "A0_lowfreq_energy",
        "input_A0_abs_mean",
        "input_A0_lowfreq_abs_mean",
        "input_A0_hf_abs_mean",
        "teacher_minus_A0_LL_energy",
        "teacher_minus_A0_HF_energy",
        "haze_params_from_id_if_available",
        "haze_beta_from_id",
        "haze_airlight_from_id",
        "locked_test_touched",
    ]
    write_csv(args.out_dir / "v237_p1_tail_failure_atlas.csv", atlas, fields)
    summary_by_alpha = []
    for alpha in ALPHAS:
        subset = [row for row in atlas if float(row["alpha"]) == alpha]
        negatives = [row for row in subset if row["is_negative"]]
        severe = [row for row in subset if row["is_severe"]]
        strong_bad = [row for row in subset if row["is_strong_reference_regression"]]
        summary_by_alpha.append({
            "alpha": alpha,
            "alpha_label": alpha_label(alpha),
            "row_count": len(subset),
            "negative_count": len(negatives),
            "severe_count": len(severe),
            "strong_reference_regression_count": len(strong_bad),
            "negative_easy_or_strong_reference_count": sum(
                1 for row in negatives
                if row["hardness_bucket"] == "easy" or row["strong_reference_bucket"] == "strong_reference"
            ),
            "severe_easy_or_strong_reference_count": sum(
                1 for row in severe
                if row["hardness_bucket"] == "easy" or row["strong_reference_bucket"] == "strong_reference"
            ),
            "negative_A0_top25_count": sum(1 for row in negatives if row["strong_reference_bucket"] == "strong_reference"),
            "mean_delta": mean([float(row["teacher_delta_vs_A0"]) for row in subset]),
            "worst_delta": min([float(row["teacher_delta_vs_A0"]) for row in subset]) if subset else None,
        })
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P1 tail failure atlas",
        "source_p0_csv": str(args.p0_csv),
        "locked_test_touched": False,
        "atlas_rows": len(atlas),
        "summary_by_alpha": summary_by_alpha,
        "decision": "P1_DONE_TAIL_FAILURE_ATLAS",
    }
    write_json(args.out_dir / "v237_p1_tail_failure_summary.json", payload)
    write_json(args.out_dir / "v237_p1_closeout.json", {
        "route_id": ROUTE_ID,
        "phase": "P1",
        "decision": "P1_DONE_TAIL_FAILURE_ATLAS",
        "p2_mask_preservation_authorized": True,
        "locked_test_touched": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_P1_TAIL_FAILURE_ATLAS_OK")


def mask_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_count = sum(1 for row in rows if boolish(row.get("eligible")))
    hard_rows = [row for row in rows if hardness_bucket(row) == "hard"]
    hard_eligible = sum(1 for row in hard_rows if boolish(row.get("eligible")))
    base_negative = [row for row in rows if fnum(row.get("base_alpha0p5_delta"), 0.0) < 0]
    base_severe = [row for row in rows if fnum(row.get("base_alpha0p5_delta"), 0.0) <= SEVERE_THRESHOLD]
    return {
        "eligible_count": eligible_count,
        "hard_count": len(hard_rows),
        "hard_eligible_rate": hard_eligible / len(hard_rows) if hard_rows else None,
        "base_negative_count": len(base_negative),
        "base_severe_count": len(base_severe),
        "negative_preservation_rate": (
            sum(1 for row in base_negative if not boolish(row.get("eligible")) and abs(float(row["masked_delta"])) <= 1e-12)
            / len(base_negative)
            if base_negative
            else 1.0
        ),
        "severe_preservation_rate": (
            sum(1 for row in base_severe if not boolish(row.get("eligible")) and abs(float(row["masked_delta"])) <= 1e-12)
            / len(base_severe)
            if base_severe
            else 1.0
        ),
    }


def rule_ids() -> list[str]:
    return [
        "M0_oracle_positive",
        "M1_margin_positive",
        "M2_strong_margin",
        "M3_hard_mid_positive",
        "M4_preserve_strong_reference",
        "M5_soft_easy_veto",
        "M6_alpha_bucketed",
    ]


def rule_delta(rule: str, alpha_rows: dict[float, dict[str, str]]) -> tuple[bool, float, float | None, str]:
    base = alpha_rows[round(BASE_ALPHA, 6)]
    easy = alpha_rows[round(EASY_ALPHA, 6)]
    d05 = float(base["teacher_delta_vs_A0"])
    d025 = float(easy["teacher_delta_vs_A0"])
    bucket = hardness_bucket(base)
    strong = base.get("strong_reference_bucket") == "strong_reference"
    if rule == "M0_oracle_positive":
        eligible = d05 >= 0.0
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "delta_ge_0" if eligible else "preserve_nonpositive"
    if rule == "M1_margin_positive":
        eligible = d05 >= 0.05
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "delta_ge_0p05" if eligible else "preserve_margin"
    if rule == "M2_strong_margin":
        eligible = d05 >= 0.10
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "delta_ge_0p10" if eligible else "preserve_strong_margin"
    if rule == "M3_hard_mid_positive":
        eligible = d05 >= 0.05 and bucket != "easy"
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "hard_mid_delta_ge_0p05" if eligible else "preserve_easy_or_margin"
    if rule == "M4_preserve_strong_reference":
        eligible = d05 >= 0.05 and not strong
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "nonstrong_delta_ge_0p05" if eligible else "preserve_strong_or_margin"
    if rule == "M5_soft_easy_veto":
        eligible = d05 >= 0.05 and not (bucket == "easy" and d05 < 0.50)
        return eligible, d05 if eligible else 0.0, BASE_ALPHA if eligible else None, "soft_easy_veto_pass" if eligible else "preserve_soft_easy_veto"
    if rule == "M6_alpha_bucketed":
        if bucket != "easy" and d05 > 0.0:
            return True, d05, BASE_ALPHA, "hard_mid_alpha0p5_positive"
        if (bucket == "easy" or strong) and d025 > 0.0:
            return True, d025, EASY_ALPHA, "easy_strong_alpha0p25_positive"
        return False, 0.0, None, "preserve_bucketed_nonpositive"
    raise KeyError(rule)


def run_p2_mask_sweep(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alpha_map = rows_by_alpha(read_csv(args.p0_csv))
    per_image: list[dict[str, Any]] = []
    for image_id, alpha_rows in sorted(alpha_map.items()):
        if round(BASE_ALPHA, 6) not in alpha_rows or round(EASY_ALPHA, 6) not in alpha_rows:
            continue
        base = alpha_rows[round(BASE_ALPHA, 6)]
        for rule in rule_ids():
            eligible, masked_delta, selected_alpha, reason = rule_delta(rule, alpha_rows)
            per_image.append({
                "image_id": image_id,
                "image_name": base.get("image_name", ""),
                "rule_id": rule,
                "fold_id": base.get("fold_id", ""),
                "hardness_bucket": base.get("hardness_bucket", ""),
                "strong_reference_bucket": base.get("strong_reference_bucket", ""),
                "A0_same_context_psnr": base.get("A0_same_context_psnr", ""),
                "base_alpha0p5_delta": base.get("teacher_delta_vs_A0", ""),
                "alpha0p25_delta": alpha_rows[round(EASY_ALPHA, 6)].get("teacher_delta_vs_A0", ""),
                "eligible": eligible,
                "selected_alpha": selected_alpha if selected_alpha is not None else "",
                "masked_delta": masked_delta,
                "mask_reason": reason,
                "negative": masked_delta < 0,
                "severe": masked_delta <= SEVERE_THRESHOLD,
                "strong_reference_regression": (
                    base.get("strong_reference_bucket") == "strong_reference"
                    and masked_delta < STRONG_REFERENCE_REGRESSION_THRESHOLD
                ),
                "locked_test_touched": False,
            })
    write_csv(args.out_dir / "v237_p2_mask_preservation_sweep_per_image.csv", per_image)
    summary_rows = []
    summary_by_rule = []
    passing_rules = []
    for rule in rule_ids():
        subset = [row for row in per_image if row["rule_id"] == rule]
        stats = summarize_values(subset, "masked_delta")
        fold_summary = []
        fold_pass_count = 0
        for fold_id in range(5):
            fold_subset = [row for row in subset if int(row["fold_id"]) == fold_id]
            fold_stats = summarize_values(fold_subset, "masked_delta")
            fold_meta = mask_meta(fold_subset)
            fold_gate = gate_train_mask(fold_stats, fold_meta, eligible_min=60)
            fold_pass_count += int(fold_gate)
            fold_summary.append({"fold_id": fold_id, "gate_pass": fold_gate, **fold_meta, **fold_stats})
        meta = mask_meta(subset)
        meta["fold_pass_count"] = fold_pass_count
        meta["fold_pass"] = f"{fold_pass_count}/5"
        gate_pass = gate_p2(stats, meta)
        if gate_pass:
            passing_rules.append(rule)
        item = {"rule_id": rule, "gate_pass": gate_pass, **meta, **stats}
        summary_rows.append(item)
        summary_by_rule.append({**item, "fold_summary": fold_summary})
    write_csv(args.out_dir / "v237_p2_mask_preservation_sweep_summary.csv", summary_rows)
    selected = None
    if passing_rules:
        candidates = [row for row in summary_rows if row["gate_pass"]]
        selected = max(candidates, key=lambda row: (float(row["mean"]), float(row["hard"]), int(row["eligible_count"])))
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P2 teacher-positive + A0-preservation mask sweep",
        "source_p0_csv": str(args.p0_csv),
        "locked_test_touched": False,
        "rule_count": len(rule_ids()),
        "image_count": len(alpha_map),
        "passing_rules": passing_rules,
        "selected_rule": selected["rule_id"] if selected else None,
        "gate_pass": bool(selected),
        "decision": "P2_PASS_MASK_PRESERVATION_SUBSTRATE" if selected else "P2_FAIL_MASK_PRESERVATION_SUBSTRATE",
        "summary_by_rule": summary_by_rule,
    }
    write_json(args.out_dir / "v237_p2_mask_preservation_sweep_summary.json", payload)
    write_json(args.out_dir / "v237_p2_closeout.json", {
        "route_id": ROUTE_ID,
        "phase": "P2",
        "decision": payload["decision"],
        "gate_pass": bool(selected),
        "selected_rule": selected["rule_id"] if selected else None,
        "p3_oof_mask_selection_authorized": bool(selected),
        "locked_test_touched": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_P2_MASK_PRESERVATION_SWEEP_OK")


def select_rule_from_train(train_rows: list[dict[str, str]], eligible_min: int) -> dict[str, Any]:
    candidates = []
    for rule in rule_ids():
        subset = [row for row in train_rows if row["rule_id"] == rule]
        stats = summarize_values(subset, "masked_delta")
        meta = mask_meta(subset)
        train_pass = gate_train_mask(stats, meta, eligible_min)
        candidates.append({"rule_id": rule, "train_gate_pass": train_pass, **meta, **stats})
    return max(candidates, key=lambda row: (int(row["train_gate_pass"]), float(row["mean"] or -999), float(row["hard"] or -999), int(row["eligible_count"])))


def run_p3_oof_mask(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.p2_csv)
    per_image: list[dict[str, Any]] = []
    fold_summaries = []
    heldout_pass_count = 0
    train_pass_count = 0
    for fold_id in range(5):
        train_rows = [row for row in rows if int(row["fold_id"]) != fold_id]
        heldout_all = [row for row in rows if int(row["fold_id"]) == fold_id]
        selected = select_rule_from_train(train_rows, eligible_min=args.train_eligible_min)
        train_pass_count += int(selected["train_gate_pass"])
        heldout = [row for row in heldout_all if row["rule_id"] == selected["rule_id"]]
        stats = summarize_values(heldout, "masked_delta")
        meta = mask_meta(heldout)
        heldout_gate = gate_p3_holdout(stats, meta, args.heldout_eligible_min)
        heldout_pass_count += int(heldout_gate)
        fold_summaries.append({
            "heldout_fold_id": fold_id,
            "selected_rule": selected["rule_id"],
            "selected_train_gate_pass": selected["train_gate_pass"],
            "heldout_gate_pass": heldout_gate,
            "train_mean": selected["mean"],
            "train_hard": selected["hard"],
            "train_eligible_count": selected["eligible_count"],
            **{f"heldout_{k}": v for k, v in {**meta, **stats}.items()},
        })
        for row in heldout:
            out = dict(row)
            out["heldout_fold_id"] = fold_id
            out["selected_rule"] = selected["rule_id"]
            out["selected_train_gate_pass"] = selected["train_gate_pass"]
            out["heldout_gate_pass"] = heldout_gate
            per_image.append(out)
    write_csv(args.out_dir / "v237_p3_oof_mask_selection_per_image.csv", per_image)
    write_csv(args.out_dir / "v237_p3_oof_mask_selection_summary.csv", fold_summaries)
    gate_pass = heldout_pass_count == 5 and train_pass_count == 5
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P3 fold-stable mask selection / OOF substrate audit",
        "source_p2_csv": str(args.p2_csv),
        "locked_test_touched": False,
        "heldout_fold_pass": f"{heldout_pass_count}/5",
        "heldout_fold_pass_count": heldout_pass_count,
        "selected_train_gate_pass": f"{train_pass_count}/5",
        "selected_train_gate_pass_count": train_pass_count,
        "gate_pass": gate_pass,
        "decision": "P3_PASS_FOLD_STABLE_MASK_SELECTION" if gate_pass else "P3_FAIL_FOLD_STABLE_MASK_SELECTION",
        "p4_target_only_separability_authorized": gate_pass,
        "fold_summary": fold_summaries,
    }
    write_json(args.out_dir / "v237_p3_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_P3_OOF_MASK_SELECTION_OK")


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [score for label, score in zip(labels, scores) if label == 1]
    neg = [score for label, score in zip(labels, scores) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def threshold_at_fpr(labels: list[int], scores: list[float], max_fpr: float) -> tuple[float, float, float]:
    negatives = sum(1 for label in labels if label == 0)
    positives = sum(1 for label in labels if label == 1)
    best = (math.inf, 0.0, 0.0)
    for threshold in sorted(set(scores), reverse=True):
        pred = [score >= threshold for score in scores]
        fp = sum(1 for p, y in zip(pred, labels) if p and y == 0)
        tp = sum(1 for p, y in zip(pred, labels) if p and y == 1)
        fpr = fp / negatives if negatives else 0.0
        recall = tp / positives if positives else 0.0
        if fpr <= max_fpr and recall >= best[1]:
            best = (threshold, recall, fpr)
    return best


def train_logistic(train_x: torch.Tensor, train_y: torch.Tensor, epochs: int, lr: float) -> torch.nn.Linear:
    model = torch.nn.Linear(train_x.shape[1], 1)
    positives = float(train_y.sum().item())
    negatives = float(train_y.numel() - positives)
    pos_weight = torch.tensor([negatives / positives]) if positives > 0 else torch.tensor([1.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
    return model


def run_p4_target_only(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(237)
    atlas_by_id = {
        row["image_id"]: row
        for row in read_csv(args.p1_atlas_csv)
        if abs((fnum(row.get("alpha")) or -1.0) - BASE_ALPHA) < 1e-9
    }
    p3_rows = read_csv(args.p3_csv)
    feature_cols = [
        "input_luma_mean",
        "input_luma_std",
        "input_saturation_mean",
        "input_dark_channel_mean",
        "input_edge_energy",
        "input_lowfreq_energy",
        "A0_luma_mean",
        "A0_luma_std",
        "A0_saturation_mean",
        "A0_dark_channel_mean",
        "A0_edge_energy",
        "A0_lowfreq_energy",
        "input_A0_abs_mean",
        "input_A0_lowfreq_abs_mean",
        "input_A0_hf_abs_mean",
    ]
    rows = []
    for row in p3_rows:
        atlas = atlas_by_id.get(row["image_id"])
        if not atlas:
            continue
        base_delta = fnum(row.get("base_alpha0p5_delta"), 0.0) or 0.0
        strong = row.get("strong_reference_bucket") == "strong_reference"
        out = {
            "image_id": row["image_id"],
            "fold_id": int(row["fold_id"]),
            "hardness_bucket": row.get("hardness_bucket", ""),
            "strong_reference_bucket": row.get("strong_reference_bucket", ""),
            "selected_rule": row.get("selected_rule", ""),
            "eligible_label": int(boolish(row.get("eligible"))),
            "noop_label": int(not boolish(row.get("eligible"))),
            "unsafe_label": int(base_delta < 0 or base_delta <= SEVERE_THRESHOLD or (strong and base_delta < STRONG_REFERENCE_REGRESSION_THRESHOLD)),
            "severe_label": int(base_delta <= SEVERE_THRESHOLD),
            "strong_reference_unsafe_label": int(strong and base_delta < STRONG_REFERENCE_REGRESSION_THRESHOLD),
            "base_alpha0p5_delta": base_delta,
        }
        for col in feature_cols:
            out[col] = fnum(atlas.get(col), 0.0) or 0.0
        rows.append(out)
    if not rows:
        raise RuntimeError("no P4 rows after joining P1 atlas and P3 OOF mask output")

    matrix = torch.tensor([[float(row[col]) for col in feature_cols] for row in rows], dtype=torch.float32)
    labels = torch.tensor([float(row["unsafe_label"]) for row in rows], dtype=torch.float32)
    scores = [0.0 for _ in rows]
    per_fold = []
    for fold_id in range(5):
        train_idx = [i for i, row in enumerate(rows) if int(row["fold_id"]) != fold_id]
        test_idx = [i for i, row in enumerate(rows) if int(row["fold_id"]) == fold_id]
        train_x = matrix[train_idx]
        test_x = matrix[test_idx]
        mu = train_x.mean(dim=0, keepdim=True)
        sigma = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = (train_x - mu) / sigma
        test_x = (test_x - mu) / sigma
        train_y = labels[train_idx]
        if float(train_y.sum().item()) == 0 or float(train_y.sum().item()) == float(train_y.numel()):
            fold_scores = [float(train_y.mean().item()) for _ in test_idx]
        else:
            model = train_logistic(train_x, train_y, args.epochs, args.lr)
            with torch.no_grad():
                fold_scores = torch.sigmoid(model(test_x).squeeze(1)).tolist()
        for idx, score in zip(test_idx, fold_scores):
            scores[idx] = float(score)
        fold_labels = [int(rows[idx]["unsafe_label"]) for idx in test_idx]
        fold_auc = auroc(fold_labels, [float(score) for score in fold_scores])
        fold_pr = auprc(fold_labels, [float(score) for score in fold_scores])
        per_fold.append({
            "fold_id": fold_id,
            "sample_count": len(test_idx),
            "unsafe_count": sum(fold_labels),
            "unsafe_AUROC": fold_auc,
            "unsafe_AUPRC": fold_pr,
            "fold_pass": bool(fold_auc is not None and fold_auc >= 0.85 and fold_pr is not None and fold_pr >= 0.50),
        })
    for row, score in zip(rows, scores):
        row["unsafe_oof_score"] = score
    labels_int = [int(row["unsafe_label"]) for row in rows]
    threshold, unsafe_recall, unsafe_fpr = threshold_at_fpr(labels_int, scores, 0.10)
    severe_rows = [i for i, row in enumerate(rows) if row["severe_label"]]
    strong_rows = [i for i, row in enumerate(rows) if row["strong_reference_unsafe_label"]]
    predicted = [score >= threshold for score in scores]
    severe_recall = sum(1 for i in severe_rows if predicted[i]) / len(severe_rows) if severe_rows else None
    strong_recall = sum(1 for i in strong_rows if predicted[i]) / len(strong_rows) if strong_rows else None
    easy_pred = [i for i, row in enumerate(rows) if row["hardness_bucket"] == "easy" and predicted[i]]
    easy_noop_precision = sum(1 for i in easy_pred if rows[i]["noop_label"]) / len(easy_pred) if easy_pred else None
    fold_pass_count = sum(1 for row in per_fold if row["fold_pass"])
    unsafe_auc = auroc(labels_int, scores)
    unsafe_pr = auprc(labels_int, scores)
    gate_pass = bool(
        unsafe_auc is not None and unsafe_auc >= 0.85
        and unsafe_pr is not None and unsafe_pr >= 0.50
        and severe_recall is not None and severe_recall >= 0.80
        and strong_recall is not None and strong_recall >= 0.80
        and easy_noop_precision is not None and easy_noop_precision >= 0.90
        and fold_pass_count >= 4
    )
    write_csv(args.out_dir / "v237_p4_target_only_eligibility_features.csv", rows)
    write_csv(args.out_dir / "v237_p4_target_only_eligibility_per_fold.csv", per_fold)
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P4 target-only no-op / unsafe separability audit",
        "source_p1_atlas_csv": str(args.p1_atlas_csv),
        "source_p3_csv": str(args.p3_csv),
        "locked_test_touched": False,
        "feature_columns": feature_cols,
        "forbidden_runtime_features_used": False,
        "unsafe_count": sum(labels_int),
        "unsafe_base_rate": sum(labels_int) / len(labels_int),
        "unsafe_detection_AUROC": unsafe_auc,
        "unsafe_AUPRC": unsafe_pr,
        "threshold_at_FPR_0p10": threshold,
        "unsafe_recall_at_FPR_0p10": unsafe_recall,
        "unsafe_FPR_at_threshold": unsafe_fpr,
        "severe_recall_at_FPR_0p10": severe_recall,
        "strong_reference_unsafe_recall_at_FPR_0p10": strong_recall,
        "easy_noop_precision": easy_noop_precision,
        "fold_pass": f"{fold_pass_count}/5",
        "fold_pass_count": fold_pass_count,
        "gate_pass": gate_pass,
        "decision": "P4_PASS_TARGET_ONLY_SEPARABILITY" if gate_pass else "P4_FAIL_TARGET_ONLY_SEPARABILITY",
        "p5_masked_free_tensor_projection_authorized": gate_pass,
        "gate": {
            "unsafe_detection_AUROC_min": 0.85,
            "unsafe_AUPRC_min": 0.50,
            "severe_recall_at_FPR_0p10_min": 0.80,
            "strong_reference_unsafe_recall_at_FPR_0p10_min": 0.80,
            "easy_noop_precision_min": 0.90,
            "fold_pass_min": "4/5",
        },
        "fold_summary": per_fold,
    }
    write_json(args.out_dir / "v237_p4_target_only_eligibility_oof_summary.json", payload)
    write_json(args.out_dir / "v237_p4_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_P4_TARGET_ONLY_ELIGIBILITY_OK")


def run_closeout(args: argparse.Namespace) -> None:
    out = args.out_dir
    p0 = json.loads((out / "v237_p0_closeout.json").read_text(encoding="utf-8")) if (out / "v237_p0_closeout.json").exists() else {}
    p2 = json.loads((out / "v237_p2_closeout.json").read_text(encoding="utf-8")) if (out / "v237_p2_closeout.json").exists() else {}
    p3 = json.loads((out / "v237_p3_closeout.json").read_text(encoding="utf-8")) if (out / "v237_p3_closeout.json").exists() else {}
    p4 = json.loads((out / "v237_p4_closeout.json").read_text(encoding="utf-8")) if (out / "v237_p4_closeout.json").exists() else {}
    if p0.get("gate_pass"):
        decision = "P0_PASS_UNMASKED_ALPHA_SUBSTRATE_STOP_BEFORE_TRAINING"
    elif p2 and not p2.get("gate_pass"):
        decision = "P2_FAIL_STOP_NO_TAIL_SAFE_MASK_SUBSTRATE"
    elif p3 and not p3.get("gate_pass"):
        decision = "P3_FAIL_STOP_NO_FOLD_STABLE_MASK"
    elif p4 and not p4.get("gate_pass"):
        decision = "P4_FAIL_STOP_TARGET_ONLY_NOOP_UNSAFE_NOT_SEPARABLE"
    elif p4 and p4.get("gate_pass"):
        decision = "P4_PASS_P5_MASKED_FREE_TENSOR_PROJECTION_AUTHORIZED"
    else:
        decision = "OPEN_PARTIAL"
    payload = {
        "route_id": ROUTE_ID,
        "inherited_reference": "v2.36 P0_FAIL_STOP_BEFORE_BRIDGE_TRAINING",
        "primary_question": "Can the full600 WDMamba same-context teacher distribution be converted into a fold-stable tail-safe teacher-positive plus A0-preservation substrate?",
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
        "direct_crop_contract_authorized": False,
        "crop256_fullslice_target_authorized": False,
        "p0_alpha_sweep_pass": p0.get("gate_pass") if p0 else None,
        "p1_tail_failure_atlas_done": (out / "v237_p1_closeout.json").exists(),
        "p2_mask_preservation_pass": p2.get("gate_pass") if p2 else None,
        "p3_oof_mask_selection_pass": p3.get("gate_pass") if p3 else None,
        "p4_target_only_separability_pass": p4.get("gate_pass") if p4 else None,
        "p5_masked_free_tensor_projection_pass": None,
        "selected_alpha": (p0.get("passing_alphas") or [None])[0] if p0.get("passing_alphas") else None,
        "selected_mask_rule": p2.get("selected_rule") if p2 else None,
        "decision": decision,
    }
    write_json(out / "v237_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V237_CLOSEOUT_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    p0 = sub.add_parser("p0-alpha-sweep")
    p0.add_argument("--out-dir", type=Path, required=True)
    p0.add_argument("--v235-manifest", type=Path, required=True)
    p0.add_argument("--fold-manifest", type=Path, default=None)

    p1 = sub.add_parser("p1-tail-atlas")
    p1.add_argument("--out-dir", type=Path, required=True)
    p1.add_argument("--p0-csv", type=Path, required=True)

    p2 = sub.add_parser("p2-mask-sweep")
    p2.add_argument("--out-dir", type=Path, required=True)
    p2.add_argument("--p0-csv", type=Path, required=True)

    p3 = sub.add_parser("p3-oof-mask")
    p3.add_argument("--out-dir", type=Path, required=True)
    p3.add_argument("--p2-csv", type=Path, required=True)
    p3.add_argument("--train-eligible-min", type=int, default=240)
    p3.add_argument("--heldout-eligible-min", type=int, default=60)

    p4 = sub.add_parser("p4-target-only")
    p4.add_argument("--out-dir", type=Path, required=True)
    p4.add_argument("--p1-atlas-csv", type=Path, required=True)
    p4.add_argument("--p3-csv", type=Path, required=True)
    p4.add_argument("--epochs", type=int, default=800)
    p4.add_argument("--lr", type=float, default=0.05)

    closeout = sub.add_parser("closeout")
    closeout.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "p0-alpha-sweep":
        run_p0_alpha_sweep(args)
    elif args.phase == "p1-tail-atlas":
        run_p1_tail_atlas(args)
    elif args.phase == "p2-mask-sweep":
        run_p2_mask_sweep(args)
    elif args.phase == "p3-oof-mask":
        run_p3_oof_mask(args)
    elif args.phase == "p4-target-only":
        run_p4_target_only(args)
    elif args.phase == "closeout":
        run_closeout(args)
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
