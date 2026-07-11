#!/usr/bin/env python3
"""v3l-A1 oracle granularity audit on frozen context operators.

This diagnostic reads the v3l-A0 cloud-only direct-head artifacts and asks how
much safe-step headroom is available when the correction step size is chosen by
privileged ground-truth oracles at image, block, or pixel granularity. It does
not train, save feature tensors, use route-confirm for strategy selection, touch
locked test, or authorize canary expansion.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import chd_rm_v3j_b_direct_correction_oof as v3j_b  # noqa: E402
import chd_rm_v3l_a0_canonical_operator as v3l_a0  # noqa: E402
from chd_rm_v3i_a_teacher_compressibility_audit import (  # noqa: E402
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    forward_final,
    load_pair,
    metric_pair,
    pad_to_factor,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from chd_rm_v3i_b_full_context_probe import full_context_maps  # noqa: E402
from chd_rm_v3j_a_bounded_action_audit import (  # noqa: E402
    names_from_manifest,
    output_gate_from_action_gate,
)

ROUTE_ID = "haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711"
A0_PASS_DECISION = "V3L_A0_CANONICAL_OPERATOR_REPLAY_PASS_AUTHORIZE_A1_ORACLE_GRANULARITY_AUDIT"
SEVERE_DB = -0.2
HARD_SEVERE_DB = -0.5


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def direction_class(alpha_star, energy, eps):
    if energy <= eps or not math.isfinite(alpha_star):
        return "no_step_or_degenerate"
    if alpha_star <= 0.0:
        return "wrong_direction"
    if alpha_star < 0.5:
        return "harmful_overshoot"
    if alpha_star < 1.0:
        return "beneficial_but_oversized"
    return "full_step_conservative_or_ok"


def tensor_direction(base_pred, direct_pred, label, eps):
    residual = label - base_pred
    step = direct_pred - base_pred
    dot = float(torch.sum(residual * step).item())
    energy = float(torch.sum(step * step).item())
    residual_energy = float(torch.sum(residual * residual).item())
    alpha_star = dot / energy if energy > eps else float("nan")
    denom = math.sqrt(max(residual_energy, 0.0) * max(energy, 0.0))
    cosine = dot / denom if denom > eps else float("nan")
    a0_mse = float(torch.mean((base_pred - label) ** 2).item())
    direct_mse = float(torch.mean((direct_pred - label) ** 2).item())
    return {
        "dot_r_d": dot,
        "step_energy": energy,
        "residual_energy": residual_energy,
        "alpha_star": alpha_star,
        "cosine_r_d": cosine,
        "norm_d_over_norm_r": math.sqrt(energy / max(residual_energy, eps)),
        "a0_mse": a0_mse,
        "direct_mse": direct_mse,
        "relative_excess_mse": direct_mse / max(a0_mse, eps) - 1.0,
        "direction_class": direction_class(alpha_star, energy, eps),
    }


def block_gain_stats(base_pred, direct_pred, label, block_size):
    residual = label - base_pred
    step = direct_pred - base_pred
    _, _, height, width = residual.shape
    gains = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            r = residual[:, :, y0 : y0 + block_size, x0 : x0 + block_size]
            d = step[:, :, y0 : y0 + block_size, x0 : x0 + block_size]
            gains.append(float((2.0 * torch.sum(r * d) - torch.sum(d * d)).item()))
    gains_np = np.asarray(gains, dtype=np.float64)
    negative_loss = np.maximum(-gains_np, 0.0)

    def worst_share(frac):
        if not np.any(negative_loss > 0.0):
            return 0.0
        take = max(1, int(math.ceil(gains_np.size * frac)))
        return float(np.sort(negative_loss)[::-1][:take].sum() / max(negative_loss.sum(), 1e-30))

    return {
        "block_size": block_size,
        "block_count": int(gains_np.size),
        "negative_block_count": int(np.sum(gains_np < 0.0)),
        "negative_block_fraction": float(np.mean(gains_np < 0.0)),
        "gain_sum": float(gains_np.sum()),
        "gain_mean": float(gains_np.mean()) if gains_np.size else 0.0,
        "gain_min": float(gains_np.min()) if gains_np.size else 0.0,
        "negative_gain_sum": float(gains_np[gains_np < 0.0].sum()) if np.any(gains_np < 0.0) else 0.0,
        "worst_5pct_negative_loss_share": worst_share(0.05),
        "worst_10pct_negative_loss_share": worst_share(0.10),
    }


def add_policy_row(rows, operator, seed, split, fold, index, name, policy, psnr_delta, selected=None):
    row = {
        "operator_label": operator,
        "seed": int(seed),
        "split": split,
        "fold": int(fold),
        "index": int(index),
        "name": name,
        "policy": policy,
        "psnr_delta": float(psnr_delta),
    }
    if selected is not None:
        row.update(selected)
    rows.append(row)


def apply_alpha(base_pred, output_step, alpha):
    return torch.clamp(base_pred + float(alpha) * output_step, 0.0, 1.0)


def image_oracle_grid(base_pred, output_step, label, alphas):
    best_alpha = 0.0
    best_mse = None
    best_pred = None
    for alpha in alphas:
        pred = apply_alpha(base_pred, output_step, alpha)
        mse = float(torch.mean((pred - label) ** 2).item())
        if best_mse is None or mse < best_mse:
            best_alpha = float(alpha)
            best_mse = mse
            best_pred = pred
    return best_alpha, best_pred


def block_oracle_grid(base_pred, output_step, label, alphas, block_size):
    _, _, height, width = base_pred.shape
    alpha_map = torch.zeros((1, 1, height, width), device=base_pred.device, dtype=base_pred.dtype)
    chosen = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            sl = (slice(None), slice(None), slice(y0, min(y0 + block_size, height)), slice(x0, min(x0 + block_size, width)))
            best_alpha = 0.0
            best_mse = None
            base_block = base_pred[sl]
            step_block = output_step[sl]
            label_block = label[sl]
            for alpha in alphas:
                pred_block = torch.clamp(base_block + float(alpha) * step_block, 0.0, 1.0)
                mse = float(torch.mean((pred_block - label_block) ** 2).item())
                if best_mse is None or mse < best_mse:
                    best_alpha = float(alpha)
                    best_mse = mse
            alpha_map[:, :, y0 : min(y0 + block_size, height), x0 : min(x0 + block_size, width)] = best_alpha
            chosen.append(best_alpha)
    pred = torch.clamp(base_pred + alpha_map * output_step, 0.0, 1.0)
    return np.asarray(chosen, dtype=np.float64), pred


def pixel_scalar_oracle(base_pred, output_step, label, eps):
    residual = label - base_pred
    numerator = torch.sum(residual * output_step, dim=1, keepdim=True)
    denominator = torch.sum(output_step * output_step, dim=1, keepdim=True)
    alpha = torch.where(denominator > eps, numerator / denominator.clamp_min(eps), torch.zeros_like(denominator))
    alpha = torch.clamp(alpha, 0.0, 1.0)
    pred = torch.clamp(base_pred + alpha * output_step, 0.0, 1.0)
    return alpha, pred


def alpha_selection_stats(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "selected_alpha_mean": float(arr.mean()),
        "selected_alpha_p10": percentile(arr, 10),
        "selected_alpha_p50": percentile(arr, 50),
        "selected_alpha_p90": percentile(arr, 90),
        "selected_alpha_zero_fraction": float(np.mean(arr <= 1e-12)),
        "selected_alpha_one_fraction": float(np.mean(arr >= 1.0 - 1e-12)),
    }


def summarize_policy_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["operator_label"], row["split"], row["policy"])].append(row)
    out = []
    for (operator, split, policy), group in sorted(grouped.items()):
        vals = [float(row["psnr_delta"]) for row in group]
        summary = {
            "operator_label": operator,
            "split": split,
            "policy": policy,
            "n": len(vals),
            "mean_psnr_delta": float(np.mean(vals)),
            "median_psnr_delta": percentile(vals, 50),
            "p10_psnr_delta": percentile(vals, 10),
            "p05_psnr_delta": percentile(vals, 5),
            "worst_psnr_delta": float(np.min(vals)),
            "best_psnr_delta": float(np.max(vals)),
            "positive_ratio": float(np.mean(np.asarray(vals) > 0.0)),
            "regression_le_0p2_count": int(np.sum(np.asarray(vals) <= SEVERE_DB)),
            "regression_le_0p5_count": int(np.sum(np.asarray(vals) <= HARD_SEVERE_DB)),
        }
        selected = [row for row in group if "selected_alpha_mean" in row]
        if selected:
            for key in (
                "selected_alpha_mean",
                "selected_alpha_p10",
                "selected_alpha_p50",
                "selected_alpha_p90",
                "selected_alpha_zero_fraction",
                "selected_alpha_one_fraction",
            ):
                summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in selected]))
        out.append(summary)
    return out


def summarize_direction_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["operator_label"], row["split"])].append(row)
    out = []
    for (operator, split), group in sorted(grouped.items()):
        alpha = np.asarray([float(row["alpha_star"]) for row in group], dtype=np.float64)
        finite = alpha[np.isfinite(alpha)]
        counts = defaultdict(int)
        for row in group:
            counts[row["direction_class"]] += 1
        out.append(
            {
                "operator_label": operator,
                "split": split,
                "n": len(group),
                "alpha_star_mean": float(finite.mean()) if finite.size else float("nan"),
                "alpha_star_p10": percentile(finite, 10) if finite.size else float("nan"),
                "alpha_star_p50": percentile(finite, 50) if finite.size else float("nan"),
                "alpha_star_p90": percentile(finite, 90) if finite.size else float("nan"),
                "wrong_direction_count": counts["wrong_direction"],
                "harmful_overshoot_count": counts["harmful_overshoot"],
                "beneficial_but_oversized_count": counts["beneficial_but_oversized"],
                "full_step_conservative_or_ok_count": counts["full_step_conservative_or_ok"],
                "no_step_or_degenerate_count": counts["no_step_or_degenerate"],
                "mean_relative_excess_mse": float(np.mean([float(row["relative_excess_mse"]) for row in group])),
            }
        )
    return out


def summarize_block_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["operator_label"], row["split"], int(row["block_size"]))].append(row)
    out = []
    for (operator, split, block_size), group in sorted(grouped.items()):
        neg_frac = np.asarray([float(row["negative_block_fraction"]) for row in group], dtype=np.float64)
        worst5 = np.asarray([float(row["worst_5pct_negative_loss_share"]) for row in group], dtype=np.float64)
        worst10 = np.asarray([float(row["worst_10pct_negative_loss_share"]) for row in group], dtype=np.float64)
        gain = np.asarray([float(row["gain_sum"]) for row in group], dtype=np.float64)
        out.append(
            {
                "operator_label": operator,
                "split": split,
                "block_size": block_size,
                "n": len(group),
                "mean_negative_block_fraction": float(neg_frac.mean()),
                "p90_negative_block_fraction": percentile(neg_frac, 90),
                "mean_worst_5pct_negative_loss_share": float(worst5.mean()),
                "mean_worst_10pct_negative_loss_share": float(worst10.mean()),
                "negative_total_gain_images": int(np.sum(gain < 0.0)),
                "mean_gain_sum": float(gain.mean()),
                "p10_gain_sum": percentile(gain, 10),
            }
        )
    return out


def paired_bootstrap(candidate, reference, seed, draws):
    ref_by_key = {
        (row["operator_label"], row["split"], int(row["index"]), row["name"]): float(row["psnr_delta"])
        for row in reference
    }
    diffs = []
    for row in candidate:
        key = (row["operator_label"], row["split"], int(row["index"]), row["name"])
        if key in ref_by_key:
            diffs.append(float(row["psnr_delta"]) - ref_by_key[key])
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return {"n": 0, "mean_lift": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        boot.append(float(np.mean(diffs[idx])))
    return {
        "n": int(diffs.size),
        "mean_lift": float(np.mean(diffs)),
        "ci95_low": percentile(boot, 2.5),
        "ci95_high": percentile(boot, 97.5),
    }


def oracle_gates(rows, summary_rows, args):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["operator_label"], row["split"], row["policy"])].append(row)
    summary = {(row["operator_label"], row["split"], row["policy"]): row for row in summary_rows}
    gate_rows = []
    candidate_policies = sorted({row["policy"] for row in rows if row["policy"].startswith("ORACLE_")})
    for operator in sorted({row["operator_label"] for row in rows}):
        ref_key = (operator, "OOF", "FIXED_ALPHA_0.125")
        if ref_key not in grouped:
            continue
        reference = grouped[ref_key]
        ref_summary = summary[ref_key]
        for policy in candidate_policies:
            key = (operator, "OOF", policy)
            if key not in grouped:
                continue
            candidate = grouped[key]
            cand_summary = summary[key]
            stable_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(f"{operator}:{policy}"))
            boot = paired_bootstrap(candidate, reference, args.seed + stable_offset, args.bootstrap_draws)
            gate = {
                "operator_label": operator,
                "split": "OOF",
                "policy": policy,
                "reference_policy": "FIXED_ALPHA_0.125",
                **boot,
                "mean_lift_ge_min": bool(boot["mean_lift"] >= args.min_oracle_mean_lift_db),
                "ci95_low_gt_0": bool(boot["ci95_low"] > 0.0),
                "p10_ge_reference": bool(cand_summary["p10_psnr_delta"] >= ref_summary["p10_psnr_delta"] - args.metric_eps_db),
                "severe_le_reference": bool(
                    cand_summary["regression_le_0p2_count"] <= ref_summary["regression_le_0p2_count"]
                ),
                "hard_severe_le_reference": bool(
                    cand_summary["regression_le_0p5_count"] <= ref_summary["regression_le_0p5_count"]
                ),
                "candidate_mean_psnr_delta": cand_summary["mean_psnr_delta"],
                "reference_mean_psnr_delta": ref_summary["mean_psnr_delta"],
                "candidate_p10_psnr_delta": cand_summary["p10_psnr_delta"],
                "reference_p10_psnr_delta": ref_summary["p10_psnr_delta"],
                "candidate_severe_le_0p2_count": cand_summary["regression_le_0p2_count"],
                "reference_severe_le_0p2_count": ref_summary["regression_le_0p2_count"],
                "route_confirm_used_for_strategy_selection": False,
            }
            gate["meaningful_oracle_escalation_pass"] = all(
                gate[k]
                for k in (
                    "mean_lift_ge_min",
                    "ci95_low_gt_0",
                    "p10_ge_reference",
                    "severe_le_reference",
                    "hard_severe_le_reference",
                )
            )
            gate_rows.append(gate)
    return gate_rows


def summarize_dual_operator_gates(gates):
    by_policy = defaultdict(list)
    for row in gates:
        by_policy[row["policy"]].append(row)
    out = []
    for policy, group in sorted(by_policy.items()):
        operators = sorted({row["operator_label"] for row in group})
        passed = [row for row in group if row["meaningful_oracle_escalation_pass"]]
        out.append(
            {
                "policy": policy,
                "operator_count": len(operators),
                "operators": ",".join(operators),
                "dual_operator_pass": len(operators) >= 2 and len(passed) == len(operators),
                "min_mean_lift": float(np.min([float(row["mean_lift"]) for row in group])),
                "min_ci95_low": float(np.min([float(row["ci95_low"]) for row in group])),
            }
        )
    return out


def validate_authorization(args):
    if args.source_split.lower() != "train":
        raise ValueError(f"v3l-A1 is train-derived only; got source_split={args.source_split!r}")
    for key in (args.train_key, args.confirm_key):
        if "test" in key.lower():
            raise ValueError(f"locked-test-like split key is forbidden for v3l-A1: {key!r}")
    closeout = read_json(args.a0_closeout)
    if closeout.get("decision") != A0_PASS_DECISION:
        raise RuntimeError(f"v3l-A1 requires A0 pass; got {closeout.get('decision')}")
    if closeout.get("locked_test_touched"):
        raise RuntimeError("A0 closeout says locked test was touched")
    if closeout.get("canary_authorized"):
        raise RuntimeError("A0 closeout unexpectedly authorized canary")
    if closeout.get("route_confirm_used_for_strategy_selection"):
        raise RuntimeError("A0 closeout used route-confirm for strategy selection")
    artifacts = []
    for result in closeout.get("results", []):
        if result.get("operator_label") not in set(args.operator_labels):
            continue
        path = Path(result["artifact_path"])
        if not path.is_file():
            raise FileNotFoundError(f"A0 artifact missing: {path}")
        sha = v3l_a0.sha256_path(path)
        if sha != result.get("artifact_sha256"):
            raise RuntimeError(f"A0 artifact sha mismatch for {path}: {sha} != {result.get('artifact_sha256')}")
        artifacts.append(
            {
                "operator_label": result["operator_label"],
                "seed": int(result["seed"]),
                "artifact_path": str(path),
                "artifact_sha256": sha,
            }
        )
    labels = sorted(item["operator_label"] for item in artifacts)
    missing = sorted(set(args.operator_labels) - set(labels))
    if missing:
        raise RuntimeError(f"A0 artifacts missing requested labels: {missing}")
    return closeout, artifacts


def process_split(args, artifact_info, artifact, cache, split, names, folds_by_image, base, action_model, gate_producer, device):
    rows = []
    direction_rows = []
    block_rows = []
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    operator = artifact_info["operator_label"]
    seed = artifact_info["seed"]
    fixed_alphas = tuple(float(v) for v in args.fixed_alphas)
    oracle_alphas = tuple(float(v) for v in args.oracle_alphas)
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        fold = int(folds_by_image[index])
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            _, base_psnr = metric_pair(base_pred, label)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
            hard_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate)
            _, hard_psnr = metric_pair(hard_pred, label)
            fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
            model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF" if split == "OOF" else "FINAL", fold)
            pred_low = v3j_b.score_map("context", model, fmap, mean, std, bound)
            output_step = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:]) * F.interpolate(
                pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False
            )

            add_policy_row(rows, operator, seed, split, fold, index, name, "A0", 0.0)
            add_policy_row(rows, operator, seed, split, fold, index, name, "HARD_D7C_ALPHA1", hard_psnr - base_psnr)

            direct_pred = apply_alpha(base_pred, output_step, 1.0)
            _, direct_psnr = metric_pair(direct_pred, label)
            add_policy_row(rows, operator, seed, split, fold, index, name, "DIRECT_CONTEXT_ALPHA1", direct_psnr - base_psnr)

            direction = tensor_direction(base_pred, direct_pred, label, args.energy_eps)
            direction.update(
                {
                    "operator_label": operator,
                    "seed": int(seed),
                    "split": split,
                    "fold": fold,
                    "index": index,
                    "name": name,
                    "policy": "DIRECT_CONTEXT_ALPHA1",
                    "psnr_delta": float(direct_psnr - base_psnr),
                }
            )
            direction_rows.append(direction)
            for block_size in args.block_sizes:
                block = block_gain_stats(base_pred, direct_pred, label, int(block_size))
                block.update(
                    {
                        "operator_label": operator,
                        "seed": int(seed),
                        "split": split,
                        "fold": fold,
                        "index": index,
                        "name": name,
                        "policy": "DIRECT_CONTEXT_ALPHA1",
                        "psnr_delta": float(direct_psnr - base_psnr),
                    }
                )
                block_rows.append(block)

            for alpha in fixed_alphas:
                pred = apply_alpha(base_pred, output_step, alpha)
                _, psnr = metric_pair(pred, label)
                add_policy_row(rows, operator, seed, split, fold, index, name, f"FIXED_ALPHA_{alpha:g}", psnr - base_psnr)

            selected_alpha, image_pred = image_oracle_grid(base_pred, output_step, label, oracle_alphas)
            _, image_psnr = metric_pair(image_pred, label)
            add_policy_row(
                rows,
                operator,
                seed,
                split,
                fold,
                index,
                name,
                "ORACLE_IMAGE_GRID",
                image_psnr - base_psnr,
                alpha_selection_stats([selected_alpha]),
            )

            for block_size in args.oracle_block_sizes:
                chosen, block_pred = block_oracle_grid(base_pred, output_step, label, oracle_alphas, int(block_size))
                _, block_psnr = metric_pair(block_pred, label)
                add_policy_row(
                    rows,
                    operator,
                    seed,
                    split,
                    fold,
                    index,
                    name,
                    f"ORACLE_BLOCK{int(block_size)}_GRID",
                    block_psnr - base_psnr,
                    alpha_selection_stats(chosen),
                )

            pixel_alpha, pixel_pred = pixel_scalar_oracle(base_pred, output_step, label, args.energy_eps)
            _, pixel_psnr = metric_pair(pixel_pred, label)
            add_policy_row(
                rows,
                operator,
                seed,
                split,
                fold,
                index,
                name,
                "ORACLE_PIXEL_SCALAR_CONTINUOUS",
                pixel_psnr - base_psnr,
                alpha_selection_stats(pixel_alpha.detach().cpu().numpy()),
            )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3l_a1_{operator}_{split.lower()} {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_gate, fmap, output_step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, direction_rows, block_rows


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "v3l_a1_oracle_granularity_summary.json"
    if summary_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing A1 summary: {summary_path}")
    closeout, artifacts = validate_authorization(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = read_json(args.fresh_split_manifest)
    train_names = names_from_manifest(manifest, args.train_key, args.max_train_samples)
    confirm_names = names_from_manifest(manifest, args.confirm_key, args.max_confirm_samples)
    train_folds, id_to_fold = v3j_b.fold_assignments(train_names, args.fold_count)
    confirm_folds = np.zeros(len(confirm_names), dtype=np.int64)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)

    policy_rows = []
    direction_rows = []
    block_rows = []
    for info in artifacts:
        artifact = torch.load(info["artifact_path"], map_location=device)
        if int(artifact.get("seed")) != int(info["seed"]):
            raise RuntimeError(f"artifact seed mismatch for {info['artifact_path']}")
        cache = v3l_a0.build_model_cache(artifact, args, device)
        oof_rows, oof_direction, oof_blocks = process_split(
            args, info, artifact, cache, "OOF", train_names, train_folds, base, action_model, gate_producer, device
        )
        policy_rows.extend(oof_rows)
        direction_rows.extend(oof_direction)
        block_rows.extend(oof_blocks)
        if args.include_confirm_audit:
            confirm_rows, confirm_direction, confirm_blocks = process_split(
                args,
                info,
                artifact,
                cache,
                "CONFIRM_AUDIT_ONLY",
                confirm_names,
                confirm_folds,
                base,
                action_model,
                gate_producer,
                device,
            )
            policy_rows.extend(confirm_rows)
            direction_rows.extend(confirm_direction)
            block_rows.extend(confirm_blocks)

    policy_summary = summarize_policy_rows(policy_rows)
    direction_summary = summarize_direction_rows(direction_rows)
    block_summary = summarize_block_rows(block_rows)
    gates = oracle_gates(policy_rows, policy_summary, args)
    dual_gates = summarize_dual_operator_gates(gates)
    dual_pass_policies = [row["policy"] for row in dual_gates if row["dual_operator_pass"]]
    decision = (
        "V3L_A1_ORACLE_GRANULARITY_PASS_AUTHORIZE_B_PHYSICS_RISK_AUDIT_ONLY"
        if dual_pass_policies
        else "V3L_A1_ORACLE_GRANULARITY_FAIL_NO_PHYSICS_RISK_AUTHORIZATION"
    )

    compact_policy_rows = [row for row in policy_rows if row["split"] == "OOF"]
    write_csv(output_dir / "v3l_a1_oracle_policy_rows_cloud_only.csv", policy_rows)
    write_csv(output_dir / "v3l_a1_oracle_policy_oof_rows_cloud_only.csv", compact_policy_rows)
    write_csv(output_dir / "v3l_a1_oracle_policy_summary.csv", policy_summary)
    write_csv(output_dir / "v3l_a1_direct_direction_rows_cloud_only.csv", direction_rows)
    write_csv(output_dir / "v3l_a1_direct_direction_summary.csv", direction_summary)
    write_csv(output_dir / "v3l_a1_direct_block_rows_cloud_only.csv", block_rows)
    write_csv(output_dir / "v3l_a1_direct_block_summary.csv", block_summary)
    write_csv(output_dir / "v3l_a1_oracle_escalation_gates.csv", gates)
    write_csv(output_dir / "v3l_a1_oracle_dual_operator_gates.csv", dual_gates)

    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3l-A1 oracle granularity audit on frozen A0 operators",
        "decision": decision,
        "locked_test_touched": False,
        "canary_authorized": False,
        "training_authorized": False,
        "new_model_search_authorized": False,
        "route_confirm_used_for_strategy_selection": False,
        "route_confirm_role": "confirm-audit-only output; OOF-only gate selection",
        "next_stage_authorized": "v3l-B physics risk audit only" if dual_pass_policies else "none",
        "dual_operator_pass_policies": dual_pass_policies,
        "operator_artifacts_read_cloud_only": artifacts,
        "raw_feature_tensors_saved": False,
        "large_per_image_tables_cloud_only": [
            "v3l_a1_oracle_policy_rows_cloud_only.csv",
            "v3l_a1_oracle_policy_oof_rows_cloud_only.csv",
            "v3l_a1_direct_direction_rows_cloud_only.csv",
            "v3l_a1_direct_block_rows_cloud_only.csv",
        ],
        "sample_counts": {
            "controller_train_images": len(train_names),
            "confirm_audit_images": len(confirm_names) if args.include_confirm_audit else 0,
            "clean_reference_count": len(id_to_fold),
        },
        "metric_contract": {
            "baseline": "A0 PSNR on train-derived clean-reference grouped OOF",
            "fixed_safe_reference": "FIXED_ALPHA_0.125 per frozen operator on OOF",
            "direct_step": "A0 + output_gate * bilinear_upsampled_context_residual, clamped to [0, 1]",
            "oracle_grid": list(args.oracle_alphas),
            "oracle_policies": [
                "ORACLE_IMAGE_GRID",
                *[f"ORACLE_BLOCK{int(v)}_GRID" for v in args.oracle_block_sizes],
                "ORACLE_PIXEL_SCALAR_CONTINUOUS",
            ],
            "meaningful_oracle_escalation_gate": {
                "mean_lift_vs_alpha0125_db_min": args.min_oracle_mean_lift_db,
                "paired_bootstrap_ci95_low": "> 0",
                "p10": ">= fixed alpha 0.125",
                "severe_le_0p2": "<= fixed alpha 0.125",
                "severe_le_0p5": "<= fixed alpha 0.125",
                "required_operators": list(args.operator_labels),
                "route_confirm_used": False,
            },
            "direction_thresholds": {
                "wrong_direction": "alpha_star <= 0",
                "harmful_overshoot": "0 < alpha_star < 0.5",
                "beneficial_but_oversized": "0.5 <= alpha_star < 1",
                "full_step_conservative_or_ok": "alpha_star >= 1",
                "critical_harmful_full_step_threshold": 0.5,
            },
        },
        "policy_summary": policy_summary,
        "direction_summary": direction_summary,
        "block_summary": block_summary,
        "oracle_escalation_gates": gates,
        "oracle_dual_operator_gates": dual_gates,
        "a0_closeout": args.a0_closeout,
        "a0_closeout_sha256": sha256_file(args.a0_closeout),
        "a0_decision": closeout.get("decision"),
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "fresh_split_manifest": args.fresh_split_manifest,
        "v3j_a_bounds": args.v3j_a_bounds,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--a0_closeout", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_confirm_samples", type=int, default=600)
    parser.add_argument("--operator_labels", nargs="+", default=["D_ref", "D_rep"])
    parser.add_argument("--fixed_alphas", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0])
    parser.add_argument("--oracle_alphas", type=float, nargs="+", default=[i / 32.0 for i in range(33)])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--oracle_block_sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--energy_eps", type=float, default=1e-12)
    parser.add_argument("--metric_eps_db", type=float, default=1e-9)
    parser.add_argument("--min_oracle_mean_lift_db", type=float, default=0.02)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--include_confirm_audit", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
