#!/usr/bin/env python3
"""Evaluate actual DTA-v3.7 D1-TAU soft-blend oracles on train folds.

Phase D2 used table-scaled D1 deltas for alpha-shrunk actions. This tool does
the decisive D3 check: render A0 and D1 TAU candidates, blend tensors as
A0 + alpha * (candidate - A0), and compute real PSNR/SSIM deltas on the
train-derived quick5full screen only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader
from models.ConvIR import build_net as build_convir_net


VARIANT_ORDER = [
    "u3_tau_l2_s002_g025_a006",
    "u2_tau_l3_s004_g015_a006",
    "u1_tau_l1_s004_g025_a006",
]

VARIANT_LABEL = {
    "u3_tau_l2_s002_g025_a006": "U3_TAU_tail_safe",
    "u2_tau_l3_s004_g015_a006": "U2_TAU_balanced",
    "u1_tau_l1_s004_g025_a006": "U1_TAU_high_gain",
}

VARIANT_CONFIG = {
    "u1_tau_l1_s004_g025_a006": {
        "feature_strength": 0.04,
        "feature_gate_limit": 0.25,
        "feature_gate_bias": -2.0,
    },
    "u3_tau_l2_s002_g025_a006": {
        "feature_strength": 0.02,
        "feature_gate_limit": 0.25,
        "feature_gate_bias": -2.0,
    },
    "u2_tau_l3_s004_g015_a006": {
        "feature_strength": 0.04,
        "feature_gate_limit": 0.15,
        "feature_gate_bias": -2.0,
    },
}

BANK_SPECS = [
    ("A0_U2_full", ["u2_tau_l3_s004_g015_a006"], [1.0], True),
    ("A0_U2_shrink", ["u2_tau_l3_s004_g015_a006"], [0.25, 0.50, 0.75, 1.0], True),
    ("A0_U3_U2_U1_full", VARIANT_ORDER, [1.0], True),
    ("A0_U3_U2_U1_shrink", VARIANT_ORDER, [0.25, 0.50, 0.75, 1.0], True),
    ("A0_U3_U2_U1_micro_shrink", VARIANT_ORDER, [0.10, 0.25, 0.50, 0.75, 1.0], True),
    ("forced_U3_U2_U1_shrink_no_A0", VARIANT_ORDER, [0.25, 0.50, 0.75, 1.0], False),
]

UTILITY_MODES = ("max_dpsnr", "tail_averse", "ssim_guarded", "high_positive_tail_averse")
CONTROL_MODES = ("zero", "shuffle", "normal")
RUN_ID_RE = re.compile(
    r"^v35_fdf_rcs_(?P<variant>.+)_seed(?P<seed>\d+)_f(?P<fold>\d+)_(?P<stage>[^_]+)(?:_(?P<tag>.+))?$"
)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: list[float], default: float = float("nan")) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else default


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _is_name_field(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str)
    )


def unpack_batch(data: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    values = list(data)
    name = values[-1] if values and _is_name_field(values[-1]) else ""
    if name:
        values = values[:-1]
        if isinstance(name, (list, tuple)):
            name = name[0]
    input_img, label_img = values[0], values[1]
    depth = values[2] if len(values) > 2 and torch.is_tensor(values[2]) else None
    return input_img, label_img, depth, str(name)


def load_model_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_state(model: torch.nn.Module, checkpoint: Path, device: torch.device, arch: str) -> None:
    state = load_model_state(checkpoint, device)
    if arch != "dta_v3":
        model.load_state_dict(state)
        return
    result = model.load_state_dict(state, strict=False)
    allowed_missing = (
        "DTA.trans_uncertainty_head.",
        "DTA.airlight_head.",
        "DTA.airlight_uncertainty_head.",
        "DTA.safe_residual_head.",
        "DTA.safe_gate_head.",
        "DTA.router_image_head.",
        "DTA.router_patch_head.",
        "DTA.feature_fusion",
    )
    missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected DTA-v3 checkpoint load result for {checkpoint}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def build_a0(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_convir_net("base", "Haze4K", "original", arch="official_convir").to(device)
    load_state(model, checkpoint, device, "official_convir")
    model.eval()
    return model


def build_dta(variant: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    cfg = VARIANT_CONFIG[variant]
    model = build_convir_net(
        "base",
        "Haze4K",
        "original",
        arch="dta_v3",
        dta_variant="v3",
        dta_prior_channels=32,
        dta_gate_bias=-2.0,
        dta_gate_limit=0.25,
        dta_gamma_limit=0.20,
        dta_beta_limit=0.10,
        dta_alpha_init=1.0,
        dta_depth_mode="invert",
        dta_confidence_floor=0.30,
        dta_confidence_local_scale=6.0,
        dta_r0_residual_scale=0.0,
        dta_depth_residual_scale=0.0,
        dta_depth_mask_easy_budget=0.0,
        dta_depth_mask_dense_budget=0.0,
        dta_depth_mask_density_thresh=0.35,
        dta_depth_mask_bias=-4.0,
        dta_phys_t_min=0.10,
        dta_phase="depth",
        dta_ablation="full",
        dta_safe_mix_enabled=False,
        dta_feature_fusion_enabled=True,
        dta_feature_fusion_strength=cfg["feature_strength"],
        dta_feature_fusion_gate_limit=cfg["feature_gate_limit"],
        dta_feature_fusion_gate_bias=cfg["feature_gate_bias"],
    ).to(device)
    load_state(model, checkpoint, device, "dta_v3")
    model.eval()
    return model


def read_d1_run_ids(action_table_csv: Path, fold: int, seed: int, include_run_substring: str) -> dict[str, str]:
    run_ids: dict[str, str] = {}
    with action_table_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("fold")) != str(fold) or str(row.get("seed")) != str(seed):
                continue
            run_id = str(row.get("run_id", ""))
            if include_run_substring and include_run_substring not in run_id:
                continue
            variant = str(row.get("variant", ""))
            if variant in VARIANT_ORDER:
                previous = run_ids.get(variant)
                if previous is not None and previous != run_id:
                    raise ValueError(
                        f"Multiple run_ids for variant={variant} fold={fold} seed={seed}: "
                        f"{previous} vs {run_id}"
                    )
                run_ids[variant] = run_id
    missing = [variant for variant in VARIANT_ORDER if variant not in run_ids]
    if missing:
        raise ValueError(
            f"Missing D1 quick5full run_ids for fold={fold} seed={seed}: {missing}. "
            f"source={action_table_csv}"
        )
    return run_ids


def model_name_from_run_id(run_id: str) -> str:
    match = RUN_ID_RE.match(run_id)
    if not match:
        raise ValueError(f"Unsupported D1 TAU run_id format: {run_id}")
    tag = match.group("tag")
    model_name = (
        f"ConvIR-Haze4K-DTA-v3-7-TAU-{match.group('variant')}-"
        f"seed{match.group('seed')}_f{match.group('fold')}-{match.group('stage')}"
    )
    if tag:
        model_name += f"-{tag}"
    return model_name


def checkpoint_path(args: argparse.Namespace, run_ids: dict[str, str], variant: str) -> Path:
    model_name = model_name_from_run_id(run_ids[variant])
    return (
        Path(args.checkpoint_root)
        / "Dehazing"
        / "ITS"
        / "results"
        / model_name
        / "Training-Results"
        / "Final.pkl"
    )


def pad_to_factor(input_img: torch.Tensor, depth: torch.Tensor | None, factor: int = 32) -> tuple[torch.Tensor, torch.Tensor | None, int, int, int, int]:
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    padded_img = F.pad(input_img, (0, padw, 0, padh), "reflect")
    padded_depth = F.pad(depth, (0, padw, 0, padh), "reflect") if depth is not None else None
    return padded_img, padded_depth, h, w, padded_h, padded_w


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def ssim_value(pred: torch.Tensor, target: torch.Tensor, padded_h: int, padded_w: int) -> float:
    down_ratio = max(1, round(min(padded_h, padded_w) / 256))
    pooled_pred = F.adaptive_avg_pool2d(pred, (int(padded_h / down_ratio), int(padded_w / down_ratio)))
    pooled_target = F.adaptive_avg_pool2d(target, (int(padded_h / down_ratio), int(padded_w / down_ratio)))
    return float(ssim(pooled_pred, pooled_target, data_range=1, size_average=False).mean().item())


def forward_a0(model: torch.nn.Module, padded: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return torch.clamp(model(padded)[2][:, :, :h, :w], 0.0, 1.0)


def forward_dta(
    model: torch.nn.Module,
    padded: torch.Tensor,
    depth: torch.Tensor,
    h: int,
    w: int,
    depth_mode: str,
) -> torch.Tensor:
    model.DTA.depth_mode = depth_mode
    return torch.clamp(model(padded, depth, airlight=None)[2][:, :, :h, :w], 0.0, 1.0)


def action_label(variant: str, alpha: float) -> str:
    if variant == "A0":
        return "A0@0"
    return f"{VARIANT_LABEL.get(variant, variant)}@{alpha:g}"


def utility(action: dict[str, Any], mode: str) -> float:
    dpsnr = finite_float(action.get("dPSNR"), 0.0)
    dssim = finite_float(action.get("dSSIM"), 0.0)
    severe_penalty = max(0.0, -0.20 - dpsnr)
    ssim_penalty = max(0.0, -0.000005 - dssim)
    if mode == "max_dpsnr":
        return dpsnr
    if mode == "tail_averse":
        return dpsnr + 0.15 * max(dpsnr, 0.0) - 4.0 * severe_penalty - 200.0 * ssim_penalty
    if mode == "ssim_guarded":
        return dpsnr if dssim >= -0.000005 else dpsnr - 0.10
    if mode == "high_positive_tail_averse":
        positive_bonus = 0.030 if dpsnr > 0.0 else -0.030
        high_gain_bonus = 0.050 * max(dpsnr, 0.0)
        return dpsnr + positive_bonus + high_gain_bonus - 6.0 * severe_penalty - 250.0 * ssim_penalty
    raise ValueError(f"Unknown utility mode: {mode}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [finite_float(row.get("dPSNR"), 0.0) for row in rows]
    ssim_deltas = [finite_float(row.get("dSSIM"), 0.0) for row in rows]
    a0_psnr = [finite_float(row.get("A0_PSNR"), 0.0) for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: a0_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(a0_psnr, 75.0)
    strong_idx = [idx for idx, value in enumerate(a0_psnr) if value >= strong_cut]
    worst_count = sum(delta <= -0.20 for delta in deltas)
    strong_count = sum(delta <= -0.05 for delta in deltas)
    n = len(rows)

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    def surplus(control_key: str) -> float:
        return statistics.mean(
            finite_float(row.get("dPSNR"), 0.0) - finite_float(row.get(control_key), 0.0)
            for row in rows
        )

    return {
        "count": n,
        "coverage": 1.0,
        "mean_dPSNR": statistics.mean(deltas),
        "hard_bottom25_dPSNR": mean_at(hard_idx),
        "easy_top25_dPSNR": mean_at(easy_idx),
        "dSSIM": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / n,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "strong_count_le_-0.05": strong_count,
        "strong_per_600": strong_count * 600.0 / n,
        "worst_count_le_-0.20": worst_count,
        "worst_per_600": worst_count * 600.0 / n,
        "true_vs_zero": surplus("zero_delta_psnr"),
        "true_vs_shuffle": surplus("shuffle_delta_psnr"),
        "true_vs_normal": surplus("normal_delta_psnr"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def make_action(
    image_id: str,
    fold: int,
    seed: int,
    a0_psnr: float,
    a0_ssim: float,
    variant: str,
    alpha: float,
    dpsnr: float,
    dssim: float,
    controls: dict[str, float],
) -> dict[str, Any]:
    row = {
        "image_id": image_id,
        "fold": fold,
        "seed": seed,
        "A0_PSNR": a0_psnr,
        "A0_SSIM": a0_ssim,
        "chosen_variant": variant,
        "chosen_variant_label": VARIANT_LABEL.get(variant, variant),
        "chosen_alpha": alpha,
        "chosen_action": action_label(variant, alpha),
        "dPSNR": dpsnr,
        "dSSIM": dssim,
    }
    for mode in CONTROL_MODES:
        row[f"{mode}_delta_psnr"] = controls.get(mode, 0.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint_root", required=True, type=Path)
    parser.add_argument("--action_table_csv", required=True, type=Path)
    parser.add_argument("--include_run_substring", default="quick5full")
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", default="quick5full")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--depth_shuffle_offset", type=int, default=137)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_ids = read_d1_run_ids(
        args.action_table_csv,
        args.fold,
        args.seed,
        args.include_run_substring,
    )
    checkpoints = {variant: checkpoint_path(args, run_ids, variant) for variant in VARIANT_ORDER}
    missing = [str(path) for path in [args.a0_checkpoint, *checkpoints.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    print(
        f"DTA_V3_7_D3_TAU_REAL_BLEND_GROUP_START fold={args.fold} seed={args.seed} "
        f"device={device} max_images={args.max_images} run_ids={run_ids}",
        flush=True,
    )
    start_time = time.time()
    a0_model = build_a0(args.a0_checkpoint, device)
    models = {variant: build_dta(variant, path, device) for variant, path in checkpoints.items()}

    split_name = f"fold{args.fold}_val"
    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split="train",
        root_split="train",
        return_trans=False,
        return_meta=False,
        split_json=args.split_json,
        split_name=split_name,
    )
    factor = 32
    selected_rows: list[dict[str, Any]] = []
    single_action_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, label_img, depth, image_id = unpack_batch(data)
            if depth is None:
                raise ValueError("DTA real-blend verification requires depth tensors.")
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device)
            padded, padded_depth, h, w, padded_h, padded_w = pad_to_factor(input_img, depth, factor=factor)
            assert padded_depth is not None

            shuffle_idx = (idx + args.depth_shuffle_offset) % len(dataloader.dataset)
            _, _, shuffled_depth, _ = unpack_batch(dataloader.dataset[shuffle_idx])
            if shuffled_depth is None:
                raise ValueError("Missing shuffled depth tensor.")
            shuffled_depth = shuffled_depth.unsqueeze(0).to(device)
            _, padded_shuffle_depth, _, _, _, _ = pad_to_factor(input_img, shuffled_depth, factor=factor)
            assert padded_shuffle_depth is not None

            a0_pred = forward_a0(a0_model, padded, h, w)
            a0_psnr = psnr(a0_pred, label_img)
            a0_ssim = ssim_value(a0_pred, label_img, padded_h, padded_w)

            variant_outputs: dict[str, dict[str, torch.Tensor]] = {}
            for variant, model in models.items():
                variant_outputs[variant] = {
                    "true": forward_dta(model, padded, padded_depth, h, w, "invert"),
                    "zero": forward_dta(model, padded, padded_depth, h, w, "zero"),
                    "shuffle": forward_dta(model, padded, padded_shuffle_depth, h, w, "shuffle"),
                    "normal": forward_dta(model, padded, padded_depth, h, w, "normal"),
                }

            action_map: dict[tuple[str, float], dict[str, Any]] = {
                ("A0", 0.0): make_action(
                    image_id,
                    args.fold,
                    args.seed,
                    a0_psnr,
                    a0_ssim,
                    "A0",
                    0.0,
                    0.0,
                    0.0,
                    {mode: 0.0 for mode in CONTROL_MODES},
                )
            }
            for variant in VARIANT_ORDER:
                for alpha in [0.10, 0.25, 0.50, 0.75, 1.0]:
                    true_blend = torch.clamp(
                        a0_pred + alpha * (variant_outputs[variant]["true"] - a0_pred),
                        0.0,
                        1.0,
                    )
                    true_psnr = psnr(true_blend, label_img)
                    true_ssim = ssim_value(true_blend, label_img, padded_h, padded_w)
                    controls: dict[str, float] = {}
                    for mode in CONTROL_MODES:
                        control_blend = torch.clamp(
                            a0_pred + alpha * (variant_outputs[variant][mode] - a0_pred),
                            0.0,
                            1.0,
                        )
                        controls[mode] = psnr(control_blend, label_img) - a0_psnr
                    action = make_action(
                        image_id,
                        args.fold,
                        args.seed,
                        a0_psnr,
                        a0_ssim,
                        variant,
                        alpha,
                        true_psnr - a0_psnr,
                        true_ssim - a0_ssim,
                        controls,
                    )
                    action_map[(variant, alpha)] = action
                    single_action_rows.append({
                        "image_id": image_id,
                        "fold": args.fold,
                        "seed": args.seed,
                        "action": action["chosen_action"],
                        "variant": variant,
                        "alpha": alpha,
                        "A0_PSNR": a0_psnr,
                        "dPSNR": action["dPSNR"],
                        "dSSIM": action["dSSIM"],
                        "zero_delta_psnr": action["zero_delta_psnr"],
                        "shuffle_delta_psnr": action["shuffle_delta_psnr"],
                        "normal_delta_psnr": action["normal_delta_psnr"],
                    })

            for bank_name, variants, alphas, include_a0 in BANK_SPECS:
                available_actions = [action_map[("A0", 0.0)]] if include_a0 else []
                for variant in variants:
                    for alpha in alphas:
                        available_actions.append(action_map[(variant, alpha)])
                for utility_mode in UTILITY_MODES:
                    chosen = sorted(
                        available_actions,
                        key=lambda item: (utility(item, utility_mode), -finite_float(item.get("chosen_alpha"), 0.0)),
                        reverse=True,
                    )[0]
                    out_row = dict(chosen)
                    out_row.update({
                        "bank_name": bank_name,
                        "utility_mode": utility_mode,
                        "include_a0": include_a0,
                        "bank_variants": ";".join(variants),
                        "bank_alphas": ";".join(f"{alpha:g}" for alpha in alphas),
                        "oracle_utility": utility(chosen, utility_mode),
                    })
                    selected_rows.append(out_row)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"DTA_V3_7_D3_TAU_REAL_BLEND_GROUP_PROGRESS fold={args.fold} seed={args.seed} "
                    f"images={idx + 1}/{len(dataloader)} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    selected_csv = args.output_dir / f"v37_tau_real_blend_selected_seed{args.seed}_f{args.fold}.csv"
    single_action_csv = args.output_dir / f"v37_tau_real_blend_single_actions_seed{args.seed}_f{args.fold}.csv"
    write_csv(selected_csv, selected_rows)
    write_csv(single_action_csv, single_action_rows)

    bank_metrics = []
    for bank_name, _, _, _ in BANK_SPECS:
        for utility_mode in UTILITY_MODES:
            rows = [
                row for row in selected_rows
                if row["bank_name"] == bank_name and row["utility_mode"] == utility_mode
            ]
            metrics = summarize(rows)
            metrics.update({
                "bank_name": bank_name,
                "utility_mode": utility_mode,
                "fold": args.fold,
                "seed": args.seed,
                "chosen_action_counts": dict(Counter(row["chosen_action"] for row in rows)),
                "chosen_variant_counts": dict(Counter(row["chosen_variant"] for row in rows)),
                "intervention_rate": sum(row["chosen_variant"] != "A0" for row in rows) / len(rows) if rows else float("nan"),
                "mean_chosen_alpha": safe_mean([finite_float(row.get("chosen_alpha")) for row in rows]),
            })
            bank_metrics.append(metrics)

    action_metrics = []
    for action in sorted({row["action"] for row in single_action_rows}):
        rows = [row for row in single_action_rows if row["action"] == action]
        action_metrics.append({
            "action": action,
            "fold": args.fold,
            "seed": args.seed,
            **summarize([
                {
                    "A0_PSNR": row["A0_PSNR"],
                    "dPSNR": row["dPSNR"],
                    "dSSIM": row["dSSIM"],
                    "zero_delta_psnr": row["zero_delta_psnr"],
                    "shuffle_delta_psnr": row["shuffle_delta_psnr"],
                    "normal_delta_psnr": row["normal_delta_psnr"],
                }
                for row in rows
            ]),
        })

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D3_tau_real_soft_blend_group",
        "fold": args.fold,
        "seed": args.seed,
        "include_run_substring": args.include_run_substring,
        "d1_run_ids": run_ids,
        "split_name": split_name,
        "count_images": len({row["image_id"] for row in selected_rows}),
        "selected_rows": len(selected_rows),
        "single_action_rows": len(single_action_rows),
        "selected_csv": str(selected_csv),
        "single_action_csv": str(single_action_csv),
        "a0_checkpoint": str(args.a0_checkpoint),
        "candidate_checkpoints": {variant: str(path) for variant, path in checkpoints.items()},
        "bank_metrics": bank_metrics,
        "single_action_metrics": action_metrics,
        "elapsed_sec": time.time() - start_time,
        "peak_cuda_mem_mib": peak_mem,
        "locked_test_touched": False,
    }
    summary_path = args.output_dir / f"v37_tau_real_blend_summary_seed{args.seed}_f{args.fold}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"DTA_V3_7_D3_TAU_REAL_BLEND_GROUP_OK fold={args.fold} seed={args.seed} "
        f"images={summary['count_images']} selected_rows={len(selected_rows)} "
        f"elapsed_sec={summary['elapsed_sec']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
