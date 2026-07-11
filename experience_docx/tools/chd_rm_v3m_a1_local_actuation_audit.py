#!/usr/bin/env python3
"""v3m-A1 frozen block16 oracle observability audit without controller training."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import chd_rm_v3l_a0_canonical_operator as v3l_a0
import chd_rm_v3l_a1_oracle_granularity_audit as v3l_a1
from chd_rm_v3i_a_teacher_compressibility_audit import (
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    load_pair,
    metric_pair,
    pad_to_factor,
    read_json,
)
from chd_rm_v3i_b_full_context_probe import full_context_maps
from chd_rm_v3j_a_bounded_action_audit import names_from_manifest, output_gate_from_action_gate


ROUTE_ID = "haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711"
OPERATORS = ("D_ref", "D_rep")
COMMON_ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
FIXED_ALPHA = 0.125
SIGNALS = (
    "d7c_score_mean",
    "direct_step_energy",
    "d7c_score_times_step_energy",
    "alpha1_clip_fraction",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha(label, path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {label}: expected {expected}, got {actual}")
    return actual


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_fixed_reference(path):
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "OOF" or row["policy"] != "FIXED_ALPHA_0.125":
                continue
            key = (row["operator_label"], row["name"])
            if key in rows:
                raise ValueError(f"duplicate fixed-alpha reference row: {key}")
            rows[key] = float(row["psnr_delta"])
    if not rows:
        raise ValueError("fixed-alpha reference table is empty")
    return rows


def roc_auc(scores, labels):
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    positive = int(np.sum(label_array == 1))
    negative = int(np.sum(label_array == 0))
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[label_array == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def bootstrap_interval(values, draws, seed):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def block_rows_for_image(base_pred, output_step, score_full, label, block_size):
    height, width = base_pred.shape[-2:]
    score_map = F.interpolate(score_full, size=(height, width), mode="bilinear", align_corners=False)
    raw_alpha1 = base_pred + output_step
    clip_map = ((raw_alpha1 < 0.0) | (raw_alpha1 > 1.0)).any(dim=1, keepdim=True).to(base_pred.dtype)
    rows = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            sl = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
            base_block = base_pred[sl]
            step_block = output_step[sl]
            label_block = label[sl]
            fixed_mse = None
            best_alpha = None
            best_mse = None
            for alpha in COMMON_ALPHAS:
                pred = torch.clamp(base_block + alpha * step_block, 0.0, 1.0)
                mse = float(torch.mean((pred - label_block) ** 2).item())
                if alpha == FIXED_ALPHA:
                    fixed_mse = mse
                if best_mse is None or mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            if fixed_mse is None:
                raise RuntimeError("fixed alpha is absent from the common action ladder")
            score_mean = float(torch.mean(score_map[:, :, y0:y1, x0:x1]).item())
            step_energy = float(torch.mean(step_block * step_block).item())
            clip_fraction = float(torch.mean(clip_map[:, :, y0:y1, x0:x1]).item())
            rows.append(
                {
                    "block_y": y0 // block_size,
                    "block_x": x0 // block_size,
                    "oracle_alpha": best_alpha,
                    "oracle_escalate_beyond_0p125": int(best_alpha > FIXED_ALPHA),
                    "oracle_mse_gain_vs_fixed": fixed_mse - best_mse,
                    "d7c_score_mean": score_mean,
                    "direct_step_energy": step_energy,
                    "d7c_score_times_step_energy": score_mean * step_energy,
                    "alpha1_clip_fraction": clip_fraction,
                }
            )
    return rows


def verify_input_contract(args):
    hashes = {
        "fresh_split_manifest": verify_sha(
            "fresh_split_manifest", args.fresh_split_manifest, args.expected_fresh_split_manifest_sha256
        ),
        "parent_a0_closeout": verify_sha(
            "parent_a0_closeout", args.a0_closeout, args.expected_parent_a0_closeout_sha256
        ),
        "parent_operator_manifest": verify_sha(
            "parent_operator_manifest", args.operator_artifact_manifest, args.expected_parent_operator_manifest_sha256
        ),
        "v3m_fixed_reference_oof_rows": verify_sha(
            "v3m_fixed_reference_oof_rows", args.reference_oof_rows, args.expected_reference_oof_rows_sha256
        ),
        "v3m_a0_source_manifest": verify_sha(
            "v3m_a0_source_manifest", args.v3m_a0_source_manifest, args.expected_v3m_a0_source_manifest_sha256
        ),
        "density_artifact": verify_sha("density_artifact", args.density_artifact, args.expected_density_artifact_sha256),
        "d7c_artifact": verify_sha("d7c_artifact", args.d7c_artifact, args.expected_d7c_artifact_sha256),
        "a0_checkpoint": verify_sha("a0_checkpoint", args.a0_checkpoint, args.expected_a0_checkpoint_sha256),
        "control_checkpoint": verify_sha(
            "control_checkpoint", args.control_checkpoint, args.expected_control_checkpoint_sha256
        ),
    }
    source = read_json(args.v3m_a0_source_manifest)
    if source["common_action_set"] != list(COMMON_ALPHAS):
        raise ValueError("v3m A0 source manifest does not pin the common five-level action ladder")
    if source["locked_test_touched"] or source["canary_authorized"] or source["training_authorized"]:
        raise ValueError("v3m A0 source manifest has forbidden authorization flags")
    return hashes


def run(args):
    if args.source_split.lower() != "train":
        raise ValueError("v3m-A1 is train-derived only")
    if args.block_size != 16:
        raise ValueError("v3m-A1 is fixed to block16")
    if tuple(args.common_alphas) != COMMON_ALPHAS:
        raise ValueError("v3m-A1 common action ladder must be exactly [0, .125, .25, .5, 1]")
    if args.run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be smoke or formal")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": output_dir / f"{args.run_tag}_block_rows_cloud_only.csv",
        "replay": output_dir / f"{args.run_tag}_replay_summary.csv",
        "signals": output_dir / f"{args.run_tag}_signal_summary.csv",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite A1 outputs: " + ", ".join(existing))

    input_hashes = verify_input_contract(args)
    closeout, artifacts = v3l_a1.validate_authorization(args)
    manifest = read_json(args.fresh_split_manifest)
    names = names_from_manifest(manifest, args.train_key, args.max_train_samples)
    if args.run_mode == "formal" and len(names) != args.formal_sample_count:
        raise ValueError(f"formal run requires {args.formal_sample_count} names, got {len(names)}")
    if args.run_mode == "smoke" and len(names) != args.smoke_sample_count:
        raise ValueError(f"smoke run requires {args.smoke_sample_count} names, got {len(names)}")
    folds, _ = v3l_a1.v3j_b.fold_assignments(names, args.fold_count)
    reference = load_fixed_reference(args.reference_oof_rows)
    expected_reference_keys = {(operator, name) for operator in args.operator_labels for name in names}
    if not expected_reference_keys.issubset(reference):
        raise ValueError("fixed-alpha reference table does not cover all requested operator/name pairs")

    device = torch.device(args.device)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    image_aucs = defaultdict(lambda: defaultdict(list))
    valid_counts = defaultdict(lambda: defaultdict(int))
    positive_counts = defaultdict(int)
    block_counts = defaultdict(int)
    replay_rows = []
    raw_fields = [
        "operator_label", "seed", "fold", "index", "name", "block_y", "block_x",
        "oracle_alpha", "oracle_escalate_beyond_0p125", "oracle_mse_gain_vs_fixed",
        *SIGNALS,
    ]

    with outputs["raw"].open("w", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields, lineterminator="\n")
        raw_writer.writeheader()
        for artifact_info in artifacts:
            operator = artifact_info["operator_label"]
            if operator not in set(args.operator_labels):
                continue
            artifact = torch.load(artifact_info["artifact_path"], map_location=device)
            if int(artifact.get("seed")) != int(artifact_info["seed"]):
                raise RuntimeError(f"operator artifact seed mismatch for {operator}")
            cache = v3l_a0.build_model_cache(artifact, args, device)
            max_replay_difference = 0.0
            for index, name in enumerate(names):
                input_img, label = load_pair(args.data_dir, args.source_split, name)
                input_img = input_img.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                padded, height, width = pad_to_factor(input_img)
                label = label[:, :, :height, :width]
                fold = int(folds[index])
                action_shape = action_shape_for_input(padded)
                with torch.no_grad():
                    base_pred = v3l_a1.forward_final(base, padded, height, width)
                    _, base_psnr = metric_pair(base_pred, label)
                    gate_full, score_full, _ = gate_producer(padded)
                    hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
                    fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
                    model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                    pred_low = v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound)
                    output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                    output_step = output_gate * F.interpolate(
                        pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False
                    )
                    fixed_pred = v3l_a1.apply_alpha(base_pred, output_step, FIXED_ALPHA)
                    _, fixed_psnr = metric_pair(fixed_pred, label)
                    fixed_delta = float(fixed_psnr - base_psnr)
                    reference_delta = reference[(operator, name)]
                    replay_difference = abs(fixed_delta - reference_delta)
                    if replay_difference > args.replay_tolerance_db:
                        raise RuntimeError(
                            f"fixed-alpha replay mismatch for {operator}/{name}: {fixed_delta} vs {reference_delta}"
                        )
                    max_replay_difference = max(max_replay_difference, replay_difference)
                    blocks = block_rows_for_image(base_pred, output_step, score_full, label, args.block_size)
                labels = [row["oracle_escalate_beyond_0p125"] for row in blocks]
                positive_counts[operator] += int(sum(labels))
                block_counts[operator] += len(labels)
                for signal in SIGNALS:
                    auc = roc_auc([row[signal] for row in blocks], labels)
                    if auc is not None:
                        image_aucs[operator][signal].append(float(auc))
                        valid_counts[operator][signal] += 1
                for row in blocks:
                    raw_writer.writerow(
                        {
                            "operator_label": operator,
                            "seed": artifact_info["seed"],
                            "fold": fold,
                            "index": index,
                            "name": name,
                            **row,
                        }
                    )
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
                del input_img, label, padded, base_pred, hard_gate, fmap, output_step, fixed_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            replay_rows.append(
                {
                    "operator_label": operator,
                    "seed": artifact_info["seed"],
                    "image_count": len(names),
                    "fixed_alpha_max_abs_psnr_delta_diff_db": max_replay_difference,
                    "fixed_alpha_replay_pass": True,
                }
            )

    signal_rows = []
    passing_signals = []
    for signal_index, signal in enumerate(SIGNALS):
        signal_pass = True
        for operator_index, operator in enumerate(args.operator_labels):
            values = image_aucs[operator][signal]
            valid_fraction = len(values) / max(len(names), 1)
            ci_low, ci_high = bootstrap_interval(values, args.bootstrap_draws, args.seed + signal_index * 100 + operator_index)
            auc_mean = float(np.mean(values)) if values else float("nan")
            passes = bool(
                valid_fraction >= args.min_valid_image_fraction
                and math.isfinite(ci_low)
                and ci_low >= args.min_auc_ci95_low
            )
            signal_pass = signal_pass and passes
            signal_rows.append(
                {
                    "operator_label": operator,
                    "signal": signal,
                    "image_count": len(names),
                    "valid_image_count": len(values),
                    "valid_image_fraction": valid_fraction,
                    "oracle_escalate_positive_block_count": positive_counts[operator],
                    "oracle_escalate_positive_fraction": positive_counts[operator] / max(block_counts[operator], 1),
                    "mean_image_auroc": auc_mean,
                    "mean_image_auroc_ci95_low": ci_low,
                    "mean_image_auroc_ci95_high": ci_high,
                    "auroc_ci95_low_ge_0p56": bool(math.isfinite(ci_low) and ci_low >= args.min_auc_ci95_low),
                    "valid_image_fraction_ge_0p80": valid_fraction >= args.min_valid_image_fraction,
                    "local_signal_pair_pass": passes,
                }
            )
        if signal_pass:
            passing_signals.append(signal)

    if args.run_mode == "smoke":
        decision = "V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY"
        next_stage = "v3m-A1 formal 1200-image OOF audit only"
    elif passing_signals:
        decision = "V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY"
        next_stage = "v3m-A2 OOF calibration audit only"
    else:
        decision = "V3M_A1_LOCAL_ACTION_OBSERVABILITY_WEAK_STOP_NO_CONTROLLER"
        next_stage = "none"
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "input_sha256": input_hashes,
        "v3m_a1_script_sha256": sha256_file(__file__),
        "locked_test_touched": False,
        "route_confirm_used_for_strategy_selection": False,
        "canary_authorized": False,
        "training_authorized": False,
        "raw_block_table_cloud_only": str(outputs["raw"]),
    }
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3m-A1 frozen block16 local-actuation observability audit",
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "decision": decision,
        "next_stage_authorized": next_stage,
        "locked_test_touched": False,
        "route_confirm_used_for_strategy_selection": False,
        "canary_authorized": False,
        "training_authorized": False,
        "metric_contract": {
            "source_split": "train-derived clean-reference OOF only",
            "block_size": args.block_size,
            "common_action_ladder": list(COMMON_ALPHAS),
            "oracle_target": "block16 oracle alpha > 0.125",
            "signals": list(SIGNALS),
            "signal_direction": "high score predicts oracle escalation",
            "minimum_valid_image_fraction": args.min_valid_image_fraction,
            "minimum_grouped_auc_ci95_low": args.min_auc_ci95_low,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.seed,
            "fixed_alpha_replay_tolerance_db": args.replay_tolerance_db,
        },
        "sample_counts": {"images_per_operator": len(names), "block_count_by_operator": dict(block_counts)},
        "fixed_alpha_replay": replay_rows,
        "signal_rows": signal_rows,
        "passing_signals": passing_signals,
        "source_manifest": source_manifest,
    }
    write_rows(outputs["replay"], replay_rows)
    write_rows(outputs["signals"], signal_rows)
    write_json(outputs["summary"], summary)
    write_json(outputs["source"], source_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--a0_closeout", required=True)
    parser.add_argument("--operator_artifact_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--reference_oof_rows", required=True)
    parser.add_argument("--v3m_a0_source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--run_mode", required=True)
    parser.add_argument("--expected_fresh_split_manifest_sha256", required=True)
    parser.add_argument("--expected_parent_a0_closeout_sha256", required=True)
    parser.add_argument("--expected_parent_operator_manifest_sha256", required=True)
    parser.add_argument("--expected_reference_oof_rows_sha256", required=True)
    parser.add_argument("--expected_v3m_a0_source_manifest_sha256", required=True)
    parser.add_argument("--expected_density_artifact_sha256", required=True)
    parser.add_argument("--expected_d7c_artifact_sha256", required=True)
    parser.add_argument("--expected_a0_checkpoint_sha256", required=True)
    parser.add_argument("--expected_control_checkpoint_sha256", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, required=True)
    parser.add_argument("--smoke_sample_count", type=int, default=32)
    parser.add_argument("--formal_sample_count", type=int, default=1200)
    parser.add_argument("--operator_labels", nargs="+", default=list(OPERATORS))
    parser.add_argument("--common_alphas", type=float, nargs="+", default=list(COMMON_ALPHAS))
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--min_valid_image_fraction", type=float, default=0.80)
    parser.add_argument("--min_auc_ci95_low", type=float, default=0.56)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if sorted(args.operator_labels) != sorted(OPERATORS):
        raise ValueError("v3m-A1 requires exactly D_ref and D_rep")
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    run(args)


if __name__ == "__main__":
    main()
