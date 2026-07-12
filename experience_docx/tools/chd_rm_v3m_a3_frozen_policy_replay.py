#!/usr/bin/env python3
"""v3m-A3 frozen A2-calibrated block16 policy replay."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import chd_rm_v3m_a1_local_actuation_audit as a1
import chd_rm_v3l_a0_canonical_operator as v3l_a0
import chd_rm_v3l_a1_oracle_granularity_audit as v3l_a1
from chd_rm_v3i_a_teacher_compressibility_audit import (
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
ACTION_LADDER = (0.0, 0.125, 0.25, 0.5, 1.0)
FIXED_ALPHA = 0.125
FIXED_POLICY = "FIXED_ALPHA_0.125"
ORACLE_POLICY = "ORACLE_BLOCK16_GRID"
REPLAY_POLICY = "A2_MONOTONE_BLOCK16_POLICY"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha(label, path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
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


def percentile(values, fraction):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, fraction, method="linear"))


def bootstrap_mean_interval(values, draws, seed):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, array.size, size=(draws, array.size))
    means = array[sampled].mean(axis=1)
    return percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_retention_interval(policy_lifts, oracle_lifts, draws, seed):
    policy = np.asarray(policy_lifts, dtype=np.float64)
    oracle = np.asarray(oracle_lifts, dtype=np.float64)
    if policy.size == 0 or policy.size != oracle.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, policy.size, size=(draws, policy.size))
    numerators = policy[sampled].mean(axis=1)
    denominators = oracle[sampled].mean(axis=1)
    valid = denominators > 0.0
    if int(np.sum(valid)) < int(draws * 0.99):
        raise RuntimeError("too many non-positive block16 oracle bootstrap denominators")
    ratios = numerators[valid] / denominators[valid]
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def tail_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p10": percentile(array, 0.10),
        "worst": float(np.min(array)),
        "severe_le_0p2_count": int(np.sum(array <= -0.2)),
        "hard_le_0p5_count": int(np.sum(array <= -0.5)),
    }


def load_reference_policies(path):
    references = defaultdict(dict)
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != "OOF":
                continue
            if row["policy"] not in {FIXED_POLICY, ORACLE_POLICY}:
                continue
            key = (row["operator_label"], row["name"])
            references[key][row["policy"]] = float(row["psnr_delta"])
    return references


def validate_a2_inputs(args):
    hashes = {
        "a2_summary": verify_sha("a2_summary", args.a2_summary, args.expected_a2_summary_sha256),
        "a2_source_manifest": verify_sha(
            "a2_source_manifest", args.a2_source_manifest, args.expected_a2_source_manifest_sha256
        ),
        "a2_fold_summary": verify_sha(
            "a2_fold_summary", args.a2_fold_summary, args.expected_a2_fold_summary_sha256
        ),
        "a2_calibration_bins": verify_sha(
            "a2_calibration_bins", args.a2_calibration_bins, args.expected_a2_calibration_bins_sha256
        ),
    }
    with Path(args.a2_summary).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("decision") != "V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY":
        raise ValueError("A2 summary does not authorize A3")
    for flag in ("training_authorized", "canary_authorized", "locked_test_touched"):
        if summary.get(flag):
            raise ValueError(f"A2 summary has forbidden flag set: {flag}")
    if summary.get("route_confirm_used_for_strategy_selection"):
        raise ValueError("A2 summary used route-confirm for strategy selection")
    for row in summary.get("operator_summaries", []):
        if row.get("operator_label") in OPERATORS and not row.get("operator_gate_pass"):
            raise ValueError(f"A2 operator gate did not pass: {row.get('operator_label')}")
    return hashes


def load_calibration(path):
    rows_by_key = defaultdict(list)
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator = row["operator_label"]
            fold = int(row["holdout_fold"])
            rows_by_key[(operator, fold)].append(row)
    calibration = {}
    for key, rows in rows_by_key.items():
        rows = sorted(rows, key=lambda item: int(item["bin_index"]))
        boundaries = []
        action_indices = []
        for index, row in enumerate(rows):
            if int(row["bin_index"]) != index:
                raise ValueError(f"non-contiguous calibration bins for {key}")
            upper = float(row["score_upper_inclusive"])
            if math.isfinite(upper):
                boundaries.append(upper)
            action_indices.append(int(row["monotone_action_index"]))
        if len(action_indices) != len(boundaries) + 1:
            raise ValueError(f"calibration boundary/map mismatch for {key}")
        calibration[key] = {
            "boundaries": np.asarray(boundaries, dtype=np.float64),
            "actions": np.asarray(action_indices, dtype=np.int8),
        }
    expected = {(operator, fold) for operator in OPERATORS for fold in range(5)}
    missing = sorted(expected.difference(calibration))
    if missing:
        raise ValueError(f"missing calibration maps: {missing}")
    return calibration


def apply_calibrated_policy(base_pred, output_step, block_size, calibration):
    boundaries = calibration["boundaries"]
    action_indices = calibration["actions"]
    height, width = base_pred.shape[-2:]
    policy_pred = torch.empty_like(base_pred)
    selected_indices = []
    selected_alphas = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            step_block = output_step[:, :, y0:y1, x0:x1]
            score = float(torch.mean(step_block * step_block).item())
            bin_index = int(np.searchsorted(boundaries, score, side="right"))
            action_index = int(action_indices[bin_index])
            alpha = ACTION_LADDER[action_index]
            pred_block = torch.clamp(base_pred[:, :, y0:y1, x0:x1] + alpha * step_block, 0.0, 1.0)
            policy_pred[:, :, y0:y1, x0:x1] = pred_block
            selected_indices.append(action_index)
            selected_alphas.append(alpha)
    return policy_pred, np.asarray(selected_indices, dtype=np.int8), np.asarray(selected_alphas, dtype=np.float64)


def summarize_operator(operator, rows, args):
    fixed = np.asarray([row["fixed_psnr_delta"] for row in rows], dtype=np.float64)
    policy = np.asarray([row["policy_psnr_delta"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle_block16_psnr_delta"] for row in rows], dtype=np.float64)
    policy_lift = policy - fixed
    oracle_lift = oracle - fixed
    lift_low, lift_high = bootstrap_mean_interval(policy_lift, args.bootstrap_draws, args.seed)
    retention_low, retention_high = bootstrap_retention_interval(policy_lift, oracle_lift, args.bootstrap_draws, args.seed)
    fixed_tail = tail_stats(fixed)
    policy_tail = tail_stats(policy)
    oracle_tail = tail_stats(oracle)
    action_counts = Counter()
    for row in rows:
        for index in range(len(ACTION_LADDER)):
            action_counts[index] += int(row[f"selected_action_{index}_count"])
    selected_total = sum(action_counts.values())
    summary = {
        "operator_label": operator,
        "image_count": len(rows),
        "fixed_alpha_max_abs_psnr_delta_diff_db": float(
            max(abs(row["fixed_psnr_delta"] - row["reference_fixed_psnr_delta"]) for row in rows)
        ),
        "policy_mean_lift_vs_fixed_db": float(np.mean(policy_lift)),
        "policy_mean_lift_vs_fixed_ci95_low_db": lift_low,
        "policy_mean_lift_vs_fixed_ci95_high_db": lift_high,
        "oracle_block16_mean_lift_vs_fixed_db": float(np.mean(oracle_lift)),
        "retention_vs_block16_oracle": float(np.mean(policy_lift) / np.mean(oracle_lift)),
        "retention_vs_block16_oracle_ci95_low": retention_low,
        "retention_vs_block16_oracle_ci95_high": retention_high,
        "paired_lift_p10_db": percentile(policy_lift, 0.10),
        "paired_lift_worst_db": float(np.min(policy_lift)),
        "fixed_p10_psnr_delta_db": fixed_tail["p10"],
        "policy_p10_psnr_delta_db": policy_tail["p10"],
        "oracle_p10_psnr_delta_db": oracle_tail["p10"],
        "fixed_worst_psnr_delta_db": fixed_tail["worst"],
        "policy_worst_psnr_delta_db": policy_tail["worst"],
        "oracle_worst_psnr_delta_db": oracle_tail["worst"],
        "fixed_severe_le_0p2_count": fixed_tail["severe_le_0p2_count"],
        "policy_severe_le_0p2_count": policy_tail["severe_le_0p2_count"],
        "fixed_hard_le_0p5_count": fixed_tail["hard_le_0p5_count"],
        "policy_hard_le_0p5_count": policy_tail["hard_le_0p5_count"],
        "selected_alpha_mean": float(np.mean([row["selected_alpha_mean"] for row in rows])),
    }
    for index, alpha in enumerate(ACTION_LADDER):
        summary[f"selected_action_{index}_alpha_{str(alpha).replace('.', 'p')}_fraction"] = (
            action_counts[index] / selected_total if selected_total else float("nan")
        )
    summary["gate_fixed_replay_pass"] = summary["fixed_alpha_max_abs_psnr_delta_diff_db"] <= args.replay_tolerance_db
    summary["gate_mean_lift_ci95_low_gt_0p05"] = (
        summary["policy_mean_lift_vs_fixed_ci95_low_db"] > args.minimum_mean_lift_ci95_low_db
    )
    summary["gate_retention_ci95_low_ge_0p45"] = (
        summary["retention_vs_block16_oracle_ci95_low"] >= args.minimum_retention_ci95_low
    )
    summary["gate_paired_lift_p10_ge_neg0p02"] = summary["paired_lift_p10_db"] >= args.minimum_paired_lift_p10_db
    summary["gate_severe_count_no_higher_than_fixed"] = (
        summary["policy_severe_le_0p2_count"] <= summary["fixed_severe_le_0p2_count"]
    )
    summary["gate_hard_count_no_higher_than_fixed"] = (
        summary["policy_hard_le_0p5_count"] <= summary["fixed_hard_le_0p5_count"]
    )
    summary["operator_gate_pass"] = all(
        bool(summary[key])
        for key in (
            "gate_fixed_replay_pass",
            "gate_mean_lift_ci95_low_gt_0p05",
            "gate_retention_ci95_low_ge_0p45",
            "gate_paired_lift_p10_ge_neg0p02",
            "gate_severe_count_no_higher_than_fixed",
            "gate_hard_count_no_higher_than_fixed",
        )
    )
    return summary


def run(args):
    if args.source_split.lower() != "train":
        raise ValueError("v3m-A3 is train-derived only")
    if args.block_size != 16:
        raise ValueError("v3m-A3 is fixed to block16")
    if tuple(args.common_alphas) != ACTION_LADDER:
        raise ValueError("v3m-A3 common action ladder must be exactly [0, .125, .25, .5, 1]")
    if args.run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be smoke or formal")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": output_dir / f"{args.run_tag}_policy_replay_rows_cloud_only.csv",
        "operators": output_dir / f"{args.run_tag}_operator_summary.csv",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite A3 outputs: " + ", ".join(existing))

    input_hashes = a1.verify_input_contract(args)
    input_hashes.update(validate_a2_inputs(args))
    closeout, artifacts = v3l_a1.validate_authorization(args)
    calibration = load_calibration(args.a2_calibration_bins)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = read_json(args.fresh_split_manifest)
    full_names = names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    if len(full_names) != args.formal_sample_count:
        raise ValueError(f"expected {args.formal_sample_count} full OOF names, got {len(full_names)}")
    names = full_names[: args.max_train_samples]
    if args.run_mode == "formal" and len(names) != args.formal_sample_count:
        raise ValueError(f"formal run requires {args.formal_sample_count} names, got {len(names)}")
    if args.run_mode == "smoke" and len(names) != args.smoke_sample_count:
        raise ValueError(f"smoke run requires {args.smoke_sample_count} names, got {len(names)}")
    full_folds, _ = v3l_a1.v3j_b.fold_assignments(full_names, args.fold_count)
    fold_by_name = dict(zip(full_names, full_folds.tolist()))
    folds = np.asarray([fold_by_name[name] for name in names], dtype=np.int64)
    references = load_reference_policies(args.reference_oof_rows)
    expected_reference_keys = {(operator, name) for operator in args.operator_labels for name in names}
    if not expected_reference_keys.issubset(references):
        raise ValueError("reference policy table does not cover all requested operator/name pairs")
    for key in expected_reference_keys:
        if not {FIXED_POLICY, ORACLE_POLICY}.issubset(references[key]):
            raise ValueError(f"reference policy table lacks fixed/oracle rows for {key}")

    device = torch.device(args.device)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    raw_rows = []
    rows_by_operator = defaultdict(list)

    for artifact_info in artifacts:
        operator = artifact_info["operator_label"]
        if operator not in set(args.operator_labels):
            continue
        artifact = torch.load(artifact_info["artifact_path"], map_location=device)
        if int(artifact.get("seed")) != int(artifact_info["seed"]):
            raise RuntimeError(f"operator artifact seed mismatch for {operator}")
        cache = v3l_a0.build_model_cache(artifact, args, device)
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
                gate_full, _, _ = gate_producer(padded)
                hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
                fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
                model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                pred_low = v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound)
                output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                output_step = output_gate * F.interpolate(
                    pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False
                )
                fixed_pred = v3l_a1.apply_alpha(base_pred, output_step, FIXED_ALPHA)
                policy_pred, selected_indices, selected_alphas = apply_calibrated_policy(
                    base_pred, output_step, args.block_size, calibration[(operator, fold)]
                )
                _, fixed_psnr = metric_pair(fixed_pred, label)
                _, policy_psnr = metric_pair(policy_pred, label)
                fixed_delta = float(fixed_psnr - base_psnr)
                policy_delta = float(policy_psnr - base_psnr)
            reference_fixed = references[(operator, name)][FIXED_POLICY]
            replay_difference = abs(fixed_delta - reference_fixed)
            if replay_difference > args.replay_tolerance_db:
                raise RuntimeError(
                    f"fixed-alpha replay mismatch for {operator}/{name}: {fixed_delta} vs {reference_fixed}"
                )
            counts = np.bincount(selected_indices.astype(np.int64), minlength=len(ACTION_LADDER))
            row = {
                "operator_label": operator,
                "seed": artifact_info["seed"],
                "split": "OOF",
                "fold": fold,
                "index": index,
                "name": name,
                "policy": REPLAY_POLICY,
                "base_psnr": float(base_psnr),
                "fixed_psnr_delta": fixed_delta,
                "reference_fixed_psnr_delta": reference_fixed,
                "oracle_block16_psnr_delta": references[(operator, name)][ORACLE_POLICY],
                "policy_psnr_delta": policy_delta,
                "policy_lift_vs_fixed": policy_delta - reference_fixed,
                "oracle_block16_lift_vs_fixed": references[(operator, name)][ORACLE_POLICY] - reference_fixed,
                "fixed_replay_abs_diff_db": replay_difference,
                "selected_alpha_mean": float(np.mean(selected_alphas)),
                "selected_alpha_p10": percentile(selected_alphas, 0.10),
                "selected_alpha_p50": percentile(selected_alphas, 0.50),
                "selected_alpha_p90": percentile(selected_alphas, 0.90),
                "selected_action_0_count": int(counts[0]),
                "selected_action_1_count": int(counts[1]),
                "selected_action_2_count": int(counts[2]),
                "selected_action_3_count": int(counts[3]),
                "selected_action_4_count": int(counts[4]),
            }
            raw_rows.append(row)
            rows_by_operator[operator].append(row)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
            del input_img, label, padded, base_pred, hard_gate, fmap, output_step, fixed_pred, policy_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    operator_rows = [summarize_operator(operator, rows_by_operator[operator], args) for operator in args.operator_labels]
    if args.run_mode == "smoke":
        decision = "V3M_A3_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY"
        next_stage = "v3m-A3 formal 1200-image OOF frozen-policy replay only"
    elif all(row["operator_gate_pass"] for row in operator_rows):
        decision = "V3M_A3_FROZEN_POLICY_REPLAY_PASS_AUTHORIZE_A4_ROUTE_CONFIRM_AUDIT_ONLY"
        next_stage = "v3m-A4 fixed-policy route-confirm audit only"
    else:
        decision = "V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM"
        next_stage = "none"

    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "phase": "v3m-A3 frozen A2-calibrated policy replay",
        "input_sha256": input_hashes,
        "v3m_a3_script_sha256": sha256_file(__file__),
        "raw_policy_replay_rows_cloud_only": str(outputs["raw"]),
        "locked_test_touched": False,
        "route_confirm_used_for_strategy_selection": False,
        "canary_authorized": False,
        "training_authorized": False,
    }
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3m-A3 frozen A2-calibrated policy replay",
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
            "common_action_ladder": list(ACTION_LADDER),
            "fixed_policy": FIXED_POLICY,
            "candidate_policy": REPLAY_POLICY,
            "oracle_reference_policy": ORACLE_POLICY,
            "minimum_mean_lift_ci95_low_db": args.minimum_mean_lift_ci95_low_db,
            "minimum_retention_ci95_low": args.minimum_retention_ci95_low,
            "minimum_paired_lift_p10_db": args.minimum_paired_lift_p10_db,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.seed,
            "fixed_alpha_replay_tolerance_db": args.replay_tolerance_db,
        },
        "operator_summaries": operator_rows,
        "source_manifest": source_manifest,
    }
    write_rows(outputs["raw"], raw_rows)
    write_rows(outputs["operators"], operator_rows)
    write_json(outputs["source"], source_manifest)
    write_json(outputs["summary"], summary)
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
    parser.add_argument("--a2_summary", required=True)
    parser.add_argument("--a2_source_manifest", required=True)
    parser.add_argument("--a2_fold_summary", required=True)
    parser.add_argument("--a2_calibration_bins", required=True)
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
    parser.add_argument("--expected_a2_summary_sha256", required=True)
    parser.add_argument("--expected_a2_source_manifest_sha256", required=True)
    parser.add_argument("--expected_a2_fold_summary_sha256", required=True)
    parser.add_argument("--expected_a2_calibration_bins_sha256", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, required=True)
    parser.add_argument("--smoke_sample_count", type=int, default=32)
    parser.add_argument("--formal_sample_count", type=int, default=1200)
    parser.add_argument("--operator_labels", nargs="+", default=list(OPERATORS))
    parser.add_argument("--common_alphas", type=float, nargs="+", default=list(ACTION_LADDER))
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, default=a1.D7C_THRESHOLD)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--minimum_mean_lift_ci95_low_db", type=float, default=0.05)
    parser.add_argument("--minimum_retention_ci95_low", type=float, default=0.45)
    parser.add_argument("--minimum_paired_lift_p10_db", type=float, default=-0.02)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if sorted(args.operator_labels) != sorted(OPERATORS):
        raise ValueError("v3m-A3 requires exactly D_ref and D_rep")
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    run(args)


if __name__ == "__main__":
    main()
