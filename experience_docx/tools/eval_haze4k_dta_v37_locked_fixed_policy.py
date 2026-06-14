#!/usr/bin/env python3
"""One-shot locked Haze4K confirmation for the sealed DTA-v3.7 policy.

The policy is fixed before locked-test access: train the D8 ridge scorer on the
train-derived D8 action/feature table, then apply the frozen score rule to the
predeclared locked-test outer groups.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
TOOL_DIR = TOOL_PATH.parent
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(TOOL_DIR), str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader  # noqa: E402
from eval_haze4k_dta_v37_tau_real_blend_oracle import (  # noqa: E402
    CONTROL_MODES,
    VARIANT_LABEL,
    VARIANT_ORDER,
    action_label,
    build_a0,
    build_dta,
    checkpoint_path,
    finite_float,
    forward_a0,
    forward_dta,
    pad_to_factor,
    psnr,
    read_d1_run_ids,
    ssim_value,
    unpack_batch,
    write_csv,
)
from extract_haze4k_dta_v37_outputdiff_features import (  # noqa: E402
    OUTPUT_PREFIXES,
    outputdiff_features,
    resize_for_features,
    tensor_to_rgb,
)
from train_haze4k_dta_v37_d3_high_positive_policy import (  # noqa: E402
    BANKS,
    STRICT_GATES,
    VARIANTS,
    design_matrix,
    fit_ridge,
    gate_checks,
    pred_ridge,
    read_csv,
    summarize,
)
from train_haze4k_dta_v37_d5_targeted_intervention_policy import SCORE_MODES  # noqa: E402
from train_haze4k_dta_v37_d6_outputdiff_policy import (  # noqa: E402
    add_disagreement_features,
    feature_groups,
    join_outputdiff_features,
    make_base_actions,
)


POLICY_ID = "primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100"


def score_mode_by_name(name: str) -> dict[str, float]:
    for mode in SCORE_MODES:
        if mode["name"] == name:
            return mode
    raise KeyError(f"Unknown score mode: {name}")


def parse_outer_groups(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        fold_s, seed_s = item.split(":", 1)
        out.append((int(fold_s), int(seed_s)))
    if not out:
        raise ValueError("At least one outer group must be provided.")
    return out


def tensor_mean(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().mean().cpu())


def tensor_std(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().std(unbiased=False).cpu())


def image_texture_mean(image: torch.Tensor) -> float:
    brightness = image.detach().float().mean(dim=1, keepdim=True)
    dx = torch.abs(brightness[:, :, :, 1:] - brightness[:, :, :, :-1])
    dy = torch.abs(brightness[:, :, 1:, :] - brightness[:, :, :-1, :])
    return float(0.5 * (dx.mean() + dy.mean()).cpu())


def base_features(input_img: torch.Tensor, depth: torch.Tensor) -> dict[str, float]:
    fallback = F.adaptive_max_pool2d(input_img.detach().float().clamp(0.0, 1.0), 1)
    return {
        "input_brightness_mean": tensor_mean(input_img),
        "input_texture_mean": image_texture_mean(input_img),
        "airlight_fallback_mean": tensor_mean(fallback),
        "depth_mean": tensor_mean(depth),
        "depth_std": tensor_std(depth),
    }


def make_action_row(
    image_id: str,
    fold: int,
    seed: int,
    variant: str,
    alpha: float,
    a0_psnr: float,
    a0_ssim: float,
    dpsnr: float,
    dssim: float,
    controls: dict[str, float],
    base: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_id": image_id,
        "fold": fold,
        "seed": seed,
        "action": action_label(variant, alpha),
        "variant": variant,
        "alpha": alpha,
        "A0_PSNR": a0_psnr,
        "dPSNR": dpsnr,
        "dSSIM": dssim,
        **base,
    }
    for mode in CONTROL_MODES:
        row[f"{mode}_delta_psnr"] = controls.get(mode, 0.0)
    return row


def add_zero_output_features(row: dict[str, Any], output_cols: list[str]) -> None:
    for col in output_cols:
        row[col] = 0.0


def prepare_train_policy(args: argparse.Namespace) -> tuple[list[str], dict[str, np.ndarray], dict[str, float]]:
    actions = make_base_actions(
        read_csv(args.train_single_actions_csv),
        read_csv(args.feature_action_table_csv),
        args.include_run_substring,
    )
    actions = join_outputdiff_features(actions, read_csv(args.train_outputdiff_features_csv))
    add_disagreement_features(actions)
    groups = feature_groups(set().union(*(row.keys() for row in actions)), actions)
    cols = groups.get(args.feature_group, [])
    if not cols:
        raise ValueError(f"Feature group {args.feature_group!r} is empty.")

    bank = BANKS[args.action_bank]
    train = [row for row in actions if round(finite_float(row.get("alpha")), 2) in bank]
    all_cols = cols + ["alpha_float", "alpha_ge_050", "alpha_is_full"] + [f"variant_is_{v}" for v in VARIANTS]
    x_train = design_matrix(train, all_cols)
    y = np.asarray([finite_float(row.get("dPSNR")) for row in train], dtype=np.float64)
    models = {
        "gain": fit_ridge(x_train, y, alpha=3.0),
        "pos": fit_ridge(x_train, (y > 0).astype(float), alpha=3.0),
        "strong": fit_ridge(x_train, (y <= -0.05).astype(float), alpha=3.0),
        "severe": fit_ridge(x_train, (y <= -0.20).astype(float), alpha=3.0),
        "ssim_bad": fit_ridge(
            x_train,
            np.asarray([finite_float(row.get("dSSIM")) < -0.000005 for row in train], dtype=float),
            alpha=3.0,
        ),
    }
    meta = {
        "train_rows": len(train),
        "train_image_groups": len({(row["image_id"], row["fold"], row["seed"]) for row in actions}),
        "feature_count": len(cols),
    }
    return all_cols, models, meta


def render_group_actions(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    output_cols: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    ns = argparse.Namespace(checkpoint_root=args.checkpoint_root)
    run_ids = read_d1_run_ids(args.feature_action_table_csv, fold, seed, args.include_run_substring)
    checkpoints = {variant: checkpoint_path(ns, run_ids, variant) for variant in VARIANT_ORDER}
    missing = [str(path) for path in [args.a0_checkpoint, *checkpoints.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s) for fold={fold} seed={seed}: {missing}")

    a0_model = build_a0(args.a0_checkpoint, device)
    models = {variant: build_dta(variant, path, device) for variant, path in checkpoints.items()}
    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split="test",
        root_split="test",
        return_trans=False,
        return_meta=False,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, label_img, depth, image_id = unpack_batch(data)
            if depth is None:
                raise ValueError("Locked DTA policy evaluation requires test depth tensors.")
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device)
            padded, padded_depth, h, w, padded_h, padded_w = pad_to_factor(input_img, depth, factor=32)
            assert padded_depth is not None
            shuffle_idx = (idx + args.depth_shuffle_offset) % len(dataloader.dataset)
            _, _, shuffled_depth, _ = unpack_batch(dataloader.dataset[shuffle_idx])
            if shuffled_depth is None:
                raise ValueError("Missing shuffled depth tensor.")
            shuffled_depth = shuffled_depth.unsqueeze(0).to(device)
            _, padded_shuffle_depth, _, _, _, _ = pad_to_factor(input_img, shuffled_depth, factor=32)
            assert padded_shuffle_depth is not None

            a0_pred = forward_a0(a0_model, padded, h, w)
            a0_psnr = psnr(a0_pred, label_img)
            a0_ssim = ssim_value(a0_pred, label_img, padded_h, padded_w)
            base = base_features(input_img, depth)
            hazy_small = tensor_to_rgb(resize_for_features(input_img, args.feature_max_side))
            a0_small = tensor_to_rgb(resize_for_features(a0_pred, args.feature_max_side))

            a0_row = make_action_row(
                image_id, fold, seed, "A0", 0.0, a0_psnr, a0_ssim, 0.0, 0.0,
                {mode: 0.0 for mode in CONTROL_MODES}, base,
            )
            add_zero_output_features(a0_row, output_cols)
            rows.append(a0_row)

            for variant, model in models.items():
                true_out = forward_dta(model, padded, padded_depth, h, w, "invert")
                controls_out = {
                    "zero": forward_dta(model, padded, padded_depth, h, w, "zero"),
                    "shuffle": forward_dta(model, padded, padded_shuffle_depth, h, w, "shuffle"),
                    "normal": forward_dta(model, padded, padded_depth, h, w, "normal"),
                }
                cand_small = tensor_to_rgb(resize_for_features(true_out, args.feature_max_side))
                for alpha in (0.10, 0.25, 0.50, 0.75, 1.0):
                    blend = torch.clamp(a0_pred + alpha * (true_out - a0_pred), 0.0, 1.0)
                    blend_small = tensor_to_rgb(resize_for_features(blend, args.feature_max_side))
                    controls: dict[str, float] = {}
                    for mode, control in controls_out.items():
                        control_blend = torch.clamp(a0_pred + alpha * (control - a0_pred), 0.0, 1.0)
                        controls[mode] = psnr(control_blend, label_img) - a0_psnr
                    row = make_action_row(
                        image_id,
                        fold,
                        seed,
                        variant,
                        alpha,
                        a0_psnr,
                        a0_ssim,
                        psnr(blend, label_img) - a0_psnr,
                        ssim_value(blend, label_img, padded_h, padded_w) - a0_ssim,
                        controls,
                        base,
                    )
                    row.update(outputdiff_features(hazy_small, a0_small, cand_small, blend_small))
                    rows.append(row)
            if (idx + 1) % 100 == 0:
                print(
                    f"DTA_V3_7_D9_LOCKED_GROUP_PROGRESS fold={fold} seed={seed} "
                    f"images={idx + 1}/{len(dataloader)} rows={len(rows)}",
                    flush=True,
                )
    del models
    del a0_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def apply_fixed_policy(
    rows: list[dict[str, Any]],
    all_cols: list[str],
    models: dict[str, np.ndarray],
    score_mode: dict[str, float],
    target: float,
    action_bank: str,
) -> list[dict[str, Any]]:
    bank = BANKS[action_bank]
    test = [row for row in rows if round(finite_float(row.get("alpha")), 2) in bank]
    x_test = design_matrix(test, all_cols)
    pred_gain = pred_ridge(models["gain"], x_test)
    pred_pos = np.clip(pred_ridge(models["pos"], x_test), 0, 1)
    pred_strong = np.clip(pred_ridge(models["strong"], x_test), 0, 1)
    pred_severe = np.clip(pred_ridge(models["severe"], x_test), 0, 1)
    pred_ssim_bad = np.clip(pred_ridge(models["ssim_bad"], x_test), 0, 1)

    by_image: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(test):
        out = dict(row)
        score = (
            pred_gain[idx]
            + score_mode["pos"] * pred_pos[idx]
            - score_mode["strong"] * pred_strong[idx]
            - score_mode["severe"] * pred_severe[idx]
            - score_mode["ssim"] * pred_ssim_bad[idx]
            + score_mode["alpha"] * finite_float(row.get("alpha"))
        )
        if row.get("variant") == "A0":
            score = -1e9
        out.update(
            {
                "pred_gain": float(pred_gain[idx]),
                "pred_pos_prob": float(pred_pos[idx]),
                "pred_strong_prob": float(pred_strong[idx]),
                "pred_severe_prob": float(pred_severe[idx]),
                "pred_ssim_bad_prob": float(pred_ssim_bad[idx]),
                "policy_score": float(score),
                "fixed_policy_id": POLICY_ID,
            }
        )
        by_image[(str(row["image_id"]), str(row["fold"]), str(row["seed"]))].append(out)

    a0_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    best_non_a0: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
    for key, candidates in by_image.items():
        a0 = [row for row in candidates if row.get("variant") == "A0"]
        non_a0 = [row for row in candidates if row.get("variant") != "A0"]
        if not a0 or not non_a0:
            raise ValueError(f"Incomplete action bank for {key}: a0={len(a0)} non_a0={len(non_a0)}")
        a0_rows[key] = a0[0]
        best_non_a0.append(
            (
                key,
                max(
                    non_a0,
                    key=lambda row: (
                        finite_float(row.get("policy_score")),
                        finite_float(row.get("pred_gain")),
                        finite_float(row.get("alpha")),
                    ),
                ),
            )
        )
    best_non_a0.sort(key=lambda item: finite_float(item[1].get("policy_score")), reverse=True)
    take_n = int(round(target * len(best_non_a0)))
    take = {key for key, _ in best_non_a0[:take_n]}
    return [row if key in take else a0_rows[key] for key, row in best_non_a0]


def add_locked_metric_aliases(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    worst = sum(finite_float(row.get("dPSNR")) <= -0.20 for row in rows)
    strong = sum(finite_float(row.get("dPSNR")) <= -0.05 for row in rows)
    metrics["worst_per_1000"] = worst * 1000.0 / n
    metrics["strong_per_1000"] = strong * 1000.0 / n
    metrics["test_image_count"] = len({row["image_id"] for row in rows})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint_root", required=True, type=Path)
    parser.add_argument("--feature_action_table_csv", required=True, type=Path)
    parser.add_argument("--train_single_actions_csv", required=True, type=Path)
    parser.add_argument("--train_outputdiff_features_csv", required=True, type=Path)
    parser.add_argument("--include_run_substring", default="d8formal")
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--outer_groups", default="0:3407,0:3411,1:3407,1:3411")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--output_prefix", default="v37_d9_locked_fixed_policy")
    parser.add_argument("--feature_group", default="outputdiff_plus_Q")
    parser.add_argument("--action_bank", default="micro_shrink")
    parser.add_argument("--score_mode", default="pred_gain")
    parser.add_argument("--target_intervention", type=float, default=1.0)
    parser.add_argument("--feature_max_side", type=int, default=384)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--depth_shuffle_offset", type=int, default=137)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    score_mode = score_mode_by_name(args.score_mode)
    all_cols, models, train_meta = prepare_train_policy(args)
    output_cols = [col for col in all_cols if col.startswith(OUTPUT_PREFIXES)]
    outer_groups = parse_outer_groups(args.outer_groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"DTA_V3_7_D9_LOCKED_FIXED_POLICY_START policy={POLICY_ID} "
        f"outer_groups={outer_groups} device={device} max_images={args.max_images}",
        flush=True,
    )
    all_actions: list[dict[str, Any]] = []
    for fold, seed in outer_groups:
        all_actions.extend(render_group_actions(args, fold, seed, output_cols, device))
    add_disagreement_features(all_actions)
    selected = apply_fixed_policy(
        all_actions,
        all_cols,
        models,
        score_mode,
        args.target_intervention,
        args.action_bank,
    )

    aggregate = add_locked_metric_aliases(summarize(selected), selected)
    aggregate.update(
        {
            "policy_id": POLICY_ID,
            "feature_group": args.feature_group,
            "action_bank": args.action_bank,
            "score_mode": args.score_mode,
            "target_intervention": args.target_intervention,
            "outer_groups": len(outer_groups),
            "selected_rows": len(selected),
            "strict_gate_checks": gate_checks(aggregate),
            "locked_test_touched": True,
            "one_shot_locked_confirmation": True,
            "post_test_tuning_allowed": False,
            **train_meta,
        }
    )
    aggregate["strict_gate_pass"] = all(aggregate["strict_gate_checks"].values())
    aggregate["decision"] = (
        "D9_LOCKED_FIXED_POLICY_PASS" if aggregate["strict_gate_pass"] else "D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING"
    )
    aggregate["mean_alpha"] = statistics.mean(finite_float(row.get("alpha")) for row in selected)
    aggregate["intervention_rate"] = sum(row.get("variant") != "A0" for row in selected) / max(1, len(selected))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["fold"]), str(row["seed"]))].append(row)
    per_outer: list[dict[str, Any]] = []
    for (fold, seed), rows in sorted(grouped.items()):
        metrics = add_locked_metric_aliases(summarize(rows), rows)
        metrics.update({"fold": fold, "seed": seed, "policy_id": POLICY_ID})
        metrics["strict_gate_checks"] = gate_checks(metrics)
        metrics["strict_gate_pass"] = all(metrics["strict_gate_checks"].values())
        per_outer.append(metrics)

    selected_path = args.output_dir / f"{args.output_prefix}_selected_actions.csv"
    aggregate_path = args.output_dir / f"{args.output_prefix}_aggregate.csv"
    per_outer_path = args.output_dir / f"{args.output_prefix}_per_outer.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    write_csv(selected_path, selected)
    write_csv(aggregate_path, [aggregate])
    write_csv(per_outer_path, per_outer)
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D9_one_shot_locked_fixed_policy_confirmation",
        "policy_id": POLICY_ID,
        "decision": aggregate["decision"],
        "aggregate": aggregate,
        "per_outer": per_outer,
        "outer_groups": [{"fold": fold, "seed": seed} for fold, seed in outer_groups],
        "selected_actions_csv": str(selected_path),
        "aggregate_csv": str(aggregate_path),
        "per_outer_csv": str(per_outer_path),
        "feature_columns": all_cols,
        "strict_gates": STRICT_GATES,
        "locked_test_touched": True,
        "one_shot_locked_confirmation": True,
        "post_test_tuning_allowed": False,
        "elapsed_sec": time.time() - start,
        "peak_cuda_mem_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"DTA_V3_7_D9_LOCKED_FIXED_POLICY_OK decision={aggregate['decision']} "
        f"rows={len(selected)} images={aggregate['test_image_count']} "
        f"mean={aggregate['mean_dPSNR']:.6f} hard={aggregate['hard_bottom25_dPSNR']:.6f} "
        f"positive={aggregate['positive_ratio']:.6f} worst_per_600={aggregate['worst_per_600']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
