#!/usr/bin/env python3
"""Haze4K v2.38 micro-alpha same-context WDMamba audit helpers."""
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
from PIL import Image

ROUTE_ID = "haze4k_v2_38_microalpha_same_context_wdmamba_safe_substrate_projection_20260706"
ALPHAS = (0.015625, 0.03125, 0.046875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125)
SEVERE_THRESHOLD = -0.30
STRONG_REFERENCE_REGRESSION_THRESHOLD = -0.05
SEARCH_MAX_ALPHA = 0.5
SEARCH_STEP = 1.0 / 1024.0


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


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return "a" + text.replace(".", "p")


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
        "worst_delta": min(values) if values else None,
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


def gate_micro_stats(stats: dict[str, Any]) -> bool:
    return bool(
        (stats.get("mean") is not None and stats["mean"] >= 0.30)
        and (stats.get("hard") is not None and stats["hard"] >= 0.50)
        and (stats.get("easy") is not None and stats["easy"] >= 0.05)
        and (stats.get("p05") is not None and stats["p05"] >= 0.00)
        and (stats.get("CVaR5") is not None and stats["CVaR5"] >= -0.01)
        and (stats.get("worst_delta") is not None and stats["worst_delta"] >= -0.05)
        and stats.get("severe_count") == 0
        and stats.get("strong_reference_regression_count") == 0
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
            "strong_reference" if float(row["A0_same_context_psnr"]) >= strong_cut else "not_strong_reference"
        )


def unique_source_rows(source_p0_csv: Path) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in read_csv(source_p0_csv):
        image_id = row.get("image_id")
        if image_id and image_id not in by_id:
            by_id[image_id] = row
    rows = list(by_id.values())
    rows.sort(key=lambda item: item.get("image_id", ""))
    return rows


def fold_pass_count(rows: list[dict[str, Any]], key: str, folds: list[int] | None = None) -> tuple[int, list[dict[str, Any]]]:
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    count = 0
    summaries = []
    for fold_id in folds:
        subset = [row for row in rows if int(row["fold_id"]) == fold_id]
        stats = summarize_values(subset, key)
        passed = gate_micro_stats(stats)
        count += int(passed)
        summaries.append({"fold_id": fold_id, "gate_pass": passed, **stats})
    return count, summaries


def gate_p0(stats: dict[str, Any], cache_sha_coverage: float, folds_passed: int) -> bool:
    return bool(stats.get("sample_count") == 600 and cache_sha_coverage == 1.0 and gate_micro_stats(stats) and folds_passed == 5)


def run_p0(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = unique_source_rows(args.source_p0_csv)
    rows: list[dict[str, Any]] = []
    cache_ready = 0
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
            teacher = blend(a0, wd, alpha)
            teacher_psnr = psnr(teacher, gt)
            delta = teacher_psnr - a0_psnr
            rows.append({
                "image_id": image_id,
                "image_name": rec.get("image_name", ""),
                "sample_index": rec.get("sample_index", ""),
                "table_split": rec.get("table_split", ""),
                "alpha": alpha,
                "alpha_label": alpha_label(alpha),
                "context_contract": "full_image_same_context",
                "A0_same_context_psnr": a0_psnr,
                "WDMamba_full_psnr": wd_psnr,
                "teacher_alpha_psnr": teacher_psnr,
                "delta_vs_A0": delta,
                "worst_delta": delta,
                "fold_id": int(float(rec.get("fold_id", (index - 1) % 5))),
                "negative": delta < 0,
                "severe": delta <= SEVERE_THRESHOLD,
                "input_path": rec.get("input_path", ""),
                "gt_path": gt_path,
                "A0_full_output_path": a0_path,
                "WDMamba_full_output_path": wd_path,
                "input_sha256": rec.get("input_sha256", ""),
                "gt_sha256": rec.get("gt_sha256", ""),
                "A0_full_output_sha256": rec.get("A0_full_output_sha256", ""),
                "WDMamba_full_output_sha256": rec.get("WDMamba_full_output_sha256", ""),
                "locked_test_touched": False,
            })
        if index % 25 == 0 or index == len(source_rows):
            print(f"v238_p0_progress {index}/{len(source_rows)} {image_id}", flush=True)
    assign_buckets(rows)
    for row in rows:
        row["strong_reference_regression"] = (
            row["strong_reference_bucket"] == "strong_reference"
            and float(row["delta_vs_A0"]) < STRONG_REFERENCE_REGRESSION_THRESHOLD
        )
    fields = [
        "image_id", "image_name", "sample_index", "table_split", "alpha", "alpha_label",
        "context_contract", "A0_same_context_psnr", "WDMamba_full_psnr", "teacher_alpha_psnr",
        "delta_vs_A0", "hardness_bucket", "easy_bucket", "strong_reference_bucket",
        "strong_reference_cut_psnr", "fold_id", "negative", "severe", "strong_reference_regression",
        "worst_delta", "input_path", "gt_path", "A0_full_output_path", "WDMamba_full_output_path",
        "input_sha256", "gt_sha256", "A0_full_output_sha256", "WDMamba_full_output_sha256",
        "locked_test_touched",
    ]
    write_csv(args.out_dir / "v238_p0_microalpha_safety_sweep_per_image.csv", rows, fields)
    cache_sha_coverage = cache_ready / len(source_rows) if source_rows else 0.0
    alpha_summaries = []
    passing_alphas = []
    for alpha in ALPHAS:
        subset = [row for row in rows if abs(float(row["alpha"]) - alpha) < 1e-12]
        stats = summarize_values(subset, "delta_vs_A0")
        fp_count, fold_summary = fold_pass_count(subset, "delta_vs_A0")
        passed = gate_p0(stats, cache_sha_coverage, fp_count)
        if passed:
            passing_alphas.append(alpha)
        alpha_summaries.append({
            "alpha": alpha,
            "alpha_label": alpha_label(alpha),
            "cache_sha_coverage": cache_sha_coverage,
            "fold_pass": f"{fp_count}/5",
            "fold_pass_count": fp_count,
            "gate_pass": passed,
            "summary": stats,
            "fold_summary": fold_summary,
        })
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P0 fine micro-alpha safety sweep",
        "source_p0_csv": str(args.source_p0_csv),
        "locked_test_touched": False,
        "image_count": len(source_rows),
        "row_count": len(rows),
        "cache_sha_coverage": cache_sha_coverage,
        "gate": {
            "image_count": 600,
            "cache_sha_coverage": 1.0,
            "mean_delta_min_dB": 0.30,
            "hard_delta_min_dB": 0.50,
            "easy_delta_min_dB": 0.05,
            "p05_min_dB": 0.00,
            "CVaR5_min_dB": -0.01,
            "worst_delta_min_dB": -0.05,
            "severe_count": 0,
            "strong_reference_regression_count": 0,
            "fold_pass": "5/5",
        },
        "alpha_summaries": alpha_summaries,
        "passing_alphas": passing_alphas,
        "any_alpha_pass": bool(passing_alphas),
        "decision": "P0_PASS_MICROALPHA_CANDIDATES" if passing_alphas else "P0_FAIL_NO_MICROALPHA_SAFE_SUBSTRATE",
    }
    write_json(args.out_dir / "v238_p0_microalpha_safety_sweep_summary.json", payload)
    write_json(args.out_dir / "v238_p0_closeout.json", {
        "route_id": ROUTE_ID,
        "phase": "P0",
        "decision": payload["decision"],
        "gate_pass": bool(passing_alphas),
        "passing_alphas": passing_alphas,
        "p1_oof_selection_authorized": bool(passing_alphas),
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V238_P0_MICROALPHA_SWEEP_OK")


def rows_by_alpha(rows: list[dict[str, str]], alpha: float) -> list[dict[str, str]]:
    return [row for row in rows if abs((fnum(row.get("alpha")) or -1.0) - alpha) < 1e-12]


def select_alpha_from_train(train_rows: list[dict[str, str]]) -> dict[str, Any]:
    candidates = []
    train_folds = sorted({int(float(row["fold_id"])) for row in train_rows})
    for alpha in ALPHAS:
        subset = rows_by_alpha(train_rows, alpha)
        stats = summarize_values(subset, "delta_vs_A0")
        fp_count, fold_summary = fold_pass_count(subset, "delta_vs_A0", train_folds)
        passed = gate_micro_stats(stats) and fp_count == len(train_folds)
        candidates.append({
            "alpha": alpha,
            "alpha_label": alpha_label(alpha),
            "train_gate_pass": passed,
            "train_fold_pass": f"{fp_count}/{len(train_folds)}",
            "train_fold_pass_count": fp_count,
            "train_fold_summary": fold_summary,
            **{f"train_{k}": v for k, v in stats.items()},
        })
    passing = [row for row in candidates if row["train_gate_pass"]]
    if not passing:
        return max(candidates, key=lambda row: (int(row["train_gate_pass"]), row["alpha"]))
    return max(passing, key=lambda row: float(row["alpha"]))


def run_p1(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.p0_csv)
    per_fold = []
    per_image = []
    selected_alphas: list[float] = []
    heldout_pass_count = 0
    for fold_id in range(5):
        train_rows = [row for row in rows if int(float(row["fold_id"])) != fold_id]
        heldout_all = [row for row in rows if int(float(row["fold_id"])) == fold_id]
        selected = select_alpha_from_train(train_rows)
        selected_alpha = float(selected["alpha"])
        if selected["train_gate_pass"]:
            selected_alphas.append(selected_alpha)
        heldout = [row for row in heldout_all if abs((fnum(row.get("alpha")) or -1.0) - selected_alpha) < 1e-12]
        stats = summarize_values(heldout, "delta_vs_A0")
        heldout_gate = bool(selected["train_gate_pass"] and gate_micro_stats(stats))
        heldout_pass_count += int(heldout_gate)
        summary = {
            "heldout_fold_id": fold_id,
            "selected_alpha": selected_alpha,
            "selected_alpha_label": alpha_label(selected_alpha),
            "selected_train_gate_pass": selected["train_gate_pass"],
            "heldout_gate_pass": heldout_gate,
            **{k: v for k, v in selected.items() if k.startswith("train_") and k != "train_fold_summary"},
            **{f"heldout_{k}": v for k, v in stats.items()},
        }
        per_fold.append(summary)
        for row in heldout:
            out = dict(row)
            out["heldout_fold_id"] = fold_id
            out["selected_alpha"] = selected_alpha
            out["selected_alpha_label"] = alpha_label(selected_alpha)
            out["selected_train_gate_pass"] = selected["train_gate_pass"]
            out["heldout_gate_pass"] = heldout_gate
            per_image.append(out)
    selected_indices = [ALPHAS.index(alpha) for alpha in selected_alphas if alpha in ALPHAS]
    variation_steps = (max(selected_indices) - min(selected_indices)) if selected_indices else None
    selected_min = min(selected_alphas) if selected_alphas else None
    heldout_stats = summarize_values(per_image, "delta_vs_A0")
    stable_variation = variation_steps is not None and variation_steps <= 1
    gate_pass = bool(
        heldout_pass_count == 5
        and len(selected_alphas) == 5
        and selected_min is not None and selected_min >= 0.03125
        and stable_variation
        and heldout_stats.get("mean") is not None and heldout_stats["mean"] >= 0.30
        and heldout_stats.get("hard") is not None and heldout_stats["hard"] >= 0.50
        and heldout_stats.get("easy") is not None and heldout_stats["easy"] >= 0.05
        and heldout_stats.get("strong_reference_regression_count") == 0
        and heldout_stats.get("severe_count") == 0
        and heldout_stats.get("CVaR5") is not None and heldout_stats["CVaR5"] >= -0.01
    )
    selected_alpha_for_projection = selected_min if gate_pass else None
    write_csv(args.out_dir / "v238_p1_oof_microalpha_selection_per_fold.csv", per_fold)
    write_csv(args.out_dir / "v238_p1_oof_microalpha_selection_summary.csv", [{
        "heldout_fold_pass": f"{heldout_pass_count}/5",
        "heldout_fold_pass_count": heldout_pass_count,
        "selected_alpha_min": selected_min,
        "selected_alpha_max": max(selected_alphas) if selected_alphas else None,
        "selected_alpha_variation_grid_steps": variation_steps,
        "selected_alpha_for_projection": selected_alpha_for_projection,
        "gate_pass": gate_pass,
        **heldout_stats,
    }])
    write_csv(args.out_dir / "v238_p1_oof_microalpha_selection_per_image.csv", per_image)
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P1 OOF micro-alpha selection",
        "source_p0_csv": str(args.p0_csv),
        "locked_test_touched": False,
        "heldout_fold_pass": f"{heldout_pass_count}/5",
        "heldout_fold_pass_count": heldout_pass_count,
        "selected_alphas": selected_alphas,
        "selected_alpha_min": selected_min,
        "selected_alpha_max": max(selected_alphas) if selected_alphas else None,
        "selected_alpha_variation_grid_steps": variation_steps,
        "selected_alpha_for_projection": selected_alpha_for_projection,
        "gate_pass": gate_pass,
        "decision": "P1_PASS_OOF_MICROALPHA_SELECTION" if gate_pass else "P1_FAIL_OOF_MICROALPHA_SELECTION",
        "p2_safety_margin_authorized": gate_pass,
        "heldout_summary": heldout_stats,
        "fold_summary": per_fold,
        "gate": {
            "heldout_fold_pass": "5/5",
            "selected_alpha_min": 0.03125,
            "selected_alpha_variation_max_grid_steps": 1,
            "heldout_mean_delta_min_dB": 0.30,
            "heldout_hard_delta_min_dB": 0.50,
            "heldout_easy_delta_min_dB": 0.05,
            "heldout_strong_reference_regression_count": 0,
            "heldout_severe_count": 0,
            "heldout_CVaR5_min_dB": -0.01,
        },
    }
    write_json(args.out_dir / "v238_p1_oof_microalpha_selection_summary.json", payload)
    write_json(args.out_dir / "v238_p1_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V238_P1_OOF_MICROALPHA_SELECTION_OK")


def delta_curve_coeffs(a0: torch.Tensor, wd: torch.Tensor, gt: torch.Tensor) -> tuple[float, float, float]:
    a = (a0 - gt).double().flatten()
    b = (wd - a0).double().flatten()
    c0 = float((a * a).mean().item())
    c1 = float((a * b).mean().item())
    c2 = float((b * b).mean().item())
    return max(c0, 1e-12), c1, c2


def delta_for_alpha(c0: float, c1: float, c2: float, alpha: float) -> float:
    mse = max(c0 + 2.0 * alpha * c1 + alpha * alpha * c2, 1e-12)
    return float(10.0 * math.log10(c0 / mse))


def max_safe_alpha(c0: float, c1: float, c2: float, threshold: float) -> float:
    max_alpha = 0.0
    steps = int(round(SEARCH_MAX_ALPHA / SEARCH_STEP))
    for step in range(steps + 1):
        alpha = step * SEARCH_STEP
        if delta_for_alpha(c0, c1, c2, alpha) >= threshold:
            max_alpha = alpha
    return max_alpha


def run_p2(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = json.loads(args.p1_summary.read_text(encoding="utf-8"))
    selected_alpha = fnum(p1.get("selected_alpha_for_projection"))
    if selected_alpha is None:
        raise RuntimeError("P1 did not provide selected_alpha_for_projection")
    p0_unique = unique_source_rows(args.p0_csv)
    p0_by_id = {row["image_id"]: row for row in p0_unique}
    p1_rows = read_csv(args.p1_per_image_csv)
    selected_rows = {row["image_id"]: row for row in p1_rows}
    rows: list[dict[str, Any]] = []
    for index, image_id in enumerate(sorted(selected_rows), start=1):
        rec = p0_by_id[image_id]
        sel = selected_rows[image_id]
        a0 = load_tensor(rec["A0_full_output_path"])
        wd = load_tensor(rec["WDMamba_full_output_path"])
        gt = load_image_tensor(rec["gt_path"])
        c0, c1, c2 = delta_curve_coeffs(a0, wd, gt)
        max_neg = max_safe_alpha(c0, c1, c2, 0.0)
        max_severe = max_safe_alpha(c0, c1, c2, SEVERE_THRESHOLD)
        is_strong = sel.get("strong_reference_bucket") == "strong_reference"
        max_strong = max_safe_alpha(c0, c1, c2, STRONG_REFERENCE_REGRESSION_THRESHOLD) if is_strong else None
        candidates = [("negative", max_neg), ("severe", max_severe)]
        if max_strong is not None:
            candidates.append(("strong_reference", max_strong))
        first_type, nearest = min(candidates, key=lambda item: item[1])
        margin = nearest - selected_alpha
        rows.append({
            "image_id": image_id,
            "fold_id": int(float(sel["fold_id"])),
            "selected_alpha": selected_alpha,
            "oof_selected_alpha": fnum(sel.get("selected_alpha")),
            "max_safe_alpha_negative_threshold": max_neg,
            "max_safe_alpha_strong_reference_threshold": max_strong if max_strong is not None else "",
            "max_safe_alpha_severe_threshold": max_severe,
            "margin_to_nearest_failure": margin,
            "failure_type_first": first_type,
            "hardness_bucket": sel.get("hardness_bucket", ""),
            "strong_reference_bucket": sel.get("strong_reference_bucket", ""),
            "selected_delta_vs_A0": delta_for_alpha(c0, c1, c2, selected_alpha),
            "locked_test_touched": False,
        })
        if index % 25 == 0 or index == len(selected_rows):
            print(f"v238_p2_progress {index}/{len(selected_rows)} {image_id}", flush=True)
    write_csv(args.out_dir / "v238_p2_critical_alpha_margin_per_image.csv", rows)
    strong_safe = [float(row["max_safe_alpha_strong_reference_threshold"]) for row in rows if row["max_safe_alpha_strong_reference_threshold"] not in ("", None)]
    severe_safe = [float(row["max_safe_alpha_severe_threshold"]) for row in rows]
    negative_safe = [float(row["max_safe_alpha_negative_threshold"]) for row in rows]
    margins = [float(row["margin_to_nearest_failure"]) for row in rows]
    fold_margins = []
    low_margin_folds = []
    for fold_id in range(5):
        fold_values = [float(row["margin_to_nearest_failure"]) for row in rows if int(row["fold_id"]) == fold_id]
        fold_min = min(fold_values) if fold_values else None
        bad = fold_min is not None and fold_min < 0.01
        if bad:
            low_margin_folds.append(fold_id)
        fold_margins.append({"fold_id": fold_id, "min_margin_to_nearest_failure": fold_min, "gate_pass": not bad})
    p05_strong = percentile(strong_safe, 5)
    p05_severe = percentile(severe_safe, 5)
    gate_pass = bool(
        p05_strong is not None and selected_alpha <= 0.75 * p05_strong
        and p05_severe is not None and selected_alpha <= 0.75 * p05_severe
        and not low_margin_folds
    )
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P2 critical-alpha safety margin atlas",
        "source_p0_csv": str(args.p0_csv),
        "source_p1_summary": str(args.p1_summary),
        "locked_test_touched": False,
        "selected_alpha": selected_alpha,
        "image_count": len(rows),
        "p05_max_safe_alpha_negative": percentile(negative_safe, 5),
        "p05_max_safe_alpha_strong_reference": p05_strong,
        "p05_max_safe_alpha_severe": p05_severe,
        "min_margin_to_nearest_failure": min(margins) if margins else None,
        "p05_margin_to_nearest_failure": percentile(margins, 5),
        "low_margin_folds": low_margin_folds,
        "fold_margin_summary": fold_margins,
        "gate_pass": gate_pass,
        "decision": "P2_PASS_CRITICAL_ALPHA_MARGIN" if gate_pass else "P2_FAIL_CRITICAL_ALPHA_MARGIN",
        "p3_microalpha_projection_authorized": gate_pass,
        "gate": {
            "selected_alpha_lte_0p75_p05_strong_reference_safe_alpha": True,
            "selected_alpha_lte_0p75_p05_severe_safe_alpha": True,
            "no_fold_margin_lt_0p01": True,
        },
    }
    write_json(args.out_dir / "v238_p2_critical_alpha_margin_summary.json", payload)
    write_json(args.out_dir / "v238_p2_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V238_P2_CRITICAL_ALPHA_MARGIN_OK")


def run_closeout(args: argparse.Namespace) -> None:
    out = args.out_dir
    p0 = json.loads((out / "v238_p0_closeout.json").read_text(encoding="utf-8")) if (out / "v238_p0_closeout.json").exists() else {}
    p1 = json.loads((out / "v238_p1_closeout.json").read_text(encoding="utf-8")) if (out / "v238_p1_closeout.json").exists() else {}
    p2 = json.loads((out / "v238_p2_closeout.json").read_text(encoding="utf-8")) if (out / "v238_p2_closeout.json").exists() else {}
    p3 = json.loads((out / "v238_p3_closeout.json").read_text(encoding="utf-8")) if (out / "v238_p3_closeout.json").exists() else {}
    if p0 and not p0.get("gate_pass"):
        decision = "P0_FAIL_STOP_NO_MICROALPHA_SAFE_SUBSTRATE"
    elif p1 and not p1.get("gate_pass"):
        decision = "P1_FAIL_STOP_OOF_ALPHA_SELECTION_NOT_STABLE"
    elif p2 and not p2.get("gate_pass"):
        decision = "P2_FAIL_STOP_CRITICAL_ALPHA_MARGIN_TOO_SMALL"
    elif p3 and p3.get("gate_pass"):
        decision = "P3_PASS_OPEN_V239_MICROALPHA_WLFBRIDGE_S4S6"
    elif p3 and not p3.get("gate_pass"):
        decision = "P3_FAIL_MICROALPHA_NOT_REPRESENTABLE_IN_CURRENT_CARRIER"
    elif p2 and p2.get("gate_pass"):
        decision = "P2_PASS_P3_MICROALPHA_PROJECTION_AUTHORIZED"
    else:
        decision = "OPEN_PARTIAL"
    payload = {
        "route_id": ROUTE_ID,
        "inherited_reference": "v2.37 P4_FAIL_STOP_TARGET_ONLY_NOOP_UNSAFE_NOT_SEPARABLE",
        "primary_question": "Can a smaller global same-context WDMamba alpha produce a fold-stable tail-safe unmasked teacher substrate that does not require runtime no-op selection?",
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
        "oracle_mask_runtime_authorized": False,
        "direct_crop_contract_authorized": False,
        "crop256_fullslice_target_authorized": False,
        "selected_alpha": p2.get("selected_alpha") if p2 else p1.get("selected_alpha_for_projection"),
        "p0_microalpha_sweep_pass": p0.get("gate_pass") if p0 else None,
        "p1_oof_alpha_selection_pass": p1.get("gate_pass") if p1 else None,
        "p2_safety_margin_pass": p2.get("gate_pass") if p2 else None,
        "p3_microalpha_projection_pass": p3.get("gate_pass") if p3 else None,
        "decision": decision,
    }
    write_json(out / "v238_closeout.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V238_CLOSEOUT_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    p0 = sub.add_parser("p0")
    p0.add_argument("--out-dir", type=Path, required=True)
    p0.add_argument("--source-p0-csv", type=Path, required=True)
    p1 = sub.add_parser("p1")
    p1.add_argument("--out-dir", type=Path, required=True)
    p1.add_argument("--p0-csv", type=Path, required=True)
    p2 = sub.add_parser("p2")
    p2.add_argument("--out-dir", type=Path, required=True)
    p2.add_argument("--p0-csv", type=Path, required=True)
    p2.add_argument("--p1-summary", type=Path, required=True)
    p2.add_argument("--p1-per-image-csv", type=Path, required=True)
    closeout = sub.add_parser("closeout")
    closeout.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "p0":
        run_p0(args)
    elif args.phase == "p1":
        run_p1(args)
    elif args.phase == "p2":
        run_p2(args)
    elif args.phase == "closeout":
        run_closeout(args)
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
