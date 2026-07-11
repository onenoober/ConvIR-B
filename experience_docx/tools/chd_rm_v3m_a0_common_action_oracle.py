#!/usr/bin/env python3
"""v3m-A0a common-action granularity lock on frozen v3l operators.

This audit is deliberately non-training. It replays the two frozen v3l
operators on the train-derived grouped OOF split and compares image, block32,
block16, and pixel oracles on exactly the same discrete alpha ladder. The
route-confirm split is emitted for audit only and is never used for a gate.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import chd_rm_v3l_a1_oracle_granularity_audit as v3l_a1
from chd_rm_v3i_a_teacher_compressibility_audit import read_json, write_csv, write_json
from chd_rm_v3j_a_bounded_action_audit import names_from_manifest


ROUTE_ID = "haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711"
PIXEL_CONTINUOUS_NAME = "ORACLE_PIXEL_SCALAR_CONTINUOUS"
PIXEL_GRID_NAME = "ORACLE_PIXEL_GRID"
COMMON_POLICIES = (
    "ORACLE_IMAGE_GRID",
    "ORACLE_BLOCK16_GRID",
    "ORACLE_BLOCK32_GRID",
    PIXEL_GRID_NAME,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    write_csv(path, rows)


def apply_grid_pixel_oracle(base_pred, output_step, label, _eps):
    """Select one common-ladder alpha per pixel with exact clamped candidates."""
    alphas = tuple(float(value) for value in apply_grid_pixel_oracle.alphas)
    candidates = torch.stack(
        [v3l_a1.apply_alpha(base_pred, output_step, alpha) for alpha in alphas], dim=0
    )
    errors = torch.mean((candidates - label.unsqueeze(0)) ** 2, dim=2, keepdim=True)
    selected_index = torch.argmin(errors, dim=0)
    gather_index = selected_index.expand(-1, base_pred.shape[1], -1, -1)
    selected_pred = torch.gather(candidates, 0, gather_index.unsqueeze(0)).squeeze(0)
    alpha_values = torch.as_tensor(alphas, dtype=base_pred.dtype, device=base_pred.device)
    selected_alpha = alpha_values[selected_index.squeeze(1)].unsqueeze(1)
    return selected_alpha, selected_pred


def replace_pixel_policy_names(directory):
    for path in Path(directory).glob("*"):
        if path.suffix not in {".csv", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if PIXEL_CONTINUOUS_NAME in text:
            path.write_text(text.replace(PIXEL_CONTINUOUS_NAME, PIXEL_GRID_NAME), encoding="utf-8")


def grouped_values(rows, policy):
    values = defaultdict(list)
    for row in rows:
        if row["policy"] == policy:
            values[row["name"]].append(float(row["psnr_delta"]))
    return {name: float(np.mean(items)) for name, items in values.items()}


def percentile(values, fraction):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, fraction, method="linear"))


def bootstrap_statistics(candidate, reference, seed, draws):
    groups = sorted(set(candidate) & set(reference))
    if not groups or set(candidate) != set(reference):
        raise RuntimeError("candidate/reference clean-reference groups do not match")
    candidate_values = np.asarray([candidate[group] for group in groups], dtype=np.float64)
    reference_values = np.asarray([reference[group] for group in groups], dtype=np.float64)
    lifts = candidate_values - reference_values
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    mean_lifts = np.mean(lifts[sampled], axis=1)
    return {
        "group_count": len(groups),
        "mean_lift": float(np.mean(lifts)),
        "mean_lift_ci95_low": percentile(mean_lifts, 0.025),
        "mean_lift_ci95_high": percentile(mean_lifts, 0.975),
    }


def retention_statistics(candidate, pixel, reference, seed, draws):
    groups = sorted(set(candidate) & set(pixel) & set(reference))
    if not groups or set(candidate) != set(reference) or set(pixel) != set(reference):
        raise RuntimeError("retention policies do not share the same clean-reference groups")
    candidate_lift = np.asarray([candidate[group] - reference[group] for group in groups], dtype=np.float64)
    pixel_lift = np.asarray([pixel[group] - reference[group] for group in groups], dtype=np.float64)
    denominator = float(np.mean(pixel_lift))
    if denominator <= 0.0:
        raise RuntimeError("pixel-grid oracle has non-positive mean lift")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    numerators = np.mean(candidate_lift[sampled], axis=1)
    denominators = np.mean(pixel_lift[sampled], axis=1)
    valid = denominators > 0.0
    if int(np.sum(valid)) < int(draws * 0.99):
        raise RuntimeError("too many non-positive pixel-grid bootstrap denominators")
    ratios = numerators[valid] / denominators[valid]
    return {
        "pixel_grid_mean_lift": denominator,
        "retention_ratio": float(np.mean(candidate_lift) / denominator),
        "retention_ci95_low": percentile(ratios, 0.025),
        "retention_ci95_high": percentile(ratios, 0.975),
    }


def policy_tail_statistics(values):
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "mean_psnr_delta": float(np.mean(array)),
        "p10_psnr_delta": percentile(array, 0.10),
        "worst_psnr_delta": float(np.min(array)),
        "severe_le_0p2_count": int(np.sum(array <= -0.2)),
        "hard_le_0p5_count": int(np.sum(array <= -0.5)),
    }


def correlation(left, right):
    if len(left) < 2:
        return float("nan")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def operator_agreement(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row["split"] != "OOF" or row["policy"] not in COMMON_POLICIES:
            continue
        key = (row["policy"], row["name"])
        grouped[key][row["operator_label"]] = float(row.get("mean_selected_alpha_mean", "nan"))
    output = []
    for policy in COMMON_POLICIES:
        pairs = [values for (item_policy, _), values in grouped.items() if item_policy == policy]
        pairs = [values for values in pairs if {"D_ref", "D_rep"}.issubset(values)]
        left = [values["D_ref"] for values in pairs]
        right = [values["D_rep"] for values in pairs]
        output.append(
            {
                "policy": policy,
                "split": "OOF",
                "clean_reference_group_count": len(pairs),
                "selected_alpha_mean_pearson": correlation(left, right),
                "selected_alpha_mean_abs_diff": float(np.mean(np.abs(np.asarray(left) - np.asarray(right)))) if pairs else float("nan"),
            }
        )
    return output


def run_common_oracle(args):
    raw_dir = Path(args.output_dir) / "cloud_only_raw_common_action"
    if raw_dir.exists():
        raise FileExistsError(f"refusing to overwrite cloud-only raw directory: {raw_dir}")
    raw_dir.mkdir(parents=True)

    apply_grid_pixel_oracle.alphas = tuple(float(value) for value in args.common_alphas)
    original_pixel_oracle = v3l_a1.pixel_scalar_oracle
    v3l_a1.pixel_scalar_oracle = apply_grid_pixel_oracle
    try:
        nested = argparse.Namespace(**vars(args))
        nested.output_dir = str(raw_dir)
        nested.oracle_alphas = list(args.common_alphas)
        nested.fixed_alphas = list(args.common_alphas)
        nested.allow_overwrite = False
        v3l_a1.run(nested)
    finally:
        v3l_a1.pixel_scalar_oracle = original_pixel_oracle

    replace_pixel_policy_names(raw_dir)
    return raw_dir


def summarize_common_action(args, raw_dir):
    policy_rows = read_rows(raw_dir / "v3l_a1_oracle_policy_oof_rows_cloud_only.csv")
    all_policy_rows = read_rows(raw_dir / "v3l_a1_oracle_policy_rows_cloud_only.csv")
    by_operator = defaultdict(list)
    for row in policy_rows:
        by_operator[row["operator_label"]].append(row)

    retention_rows = []
    gate_rows = []
    for operator in args.operator_labels:
        rows = by_operator[operator]
        reference = grouped_values(rows, "FIXED_ALPHA_0.125")
        pixel = grouped_values(rows, PIXEL_GRID_NAME)
        for policy in ("ORACLE_IMAGE_GRID", "ORACLE_BLOCK32_GRID", "ORACLE_BLOCK16_GRID"):
            candidate = grouped_values(rows, policy)
            bootstrap = bootstrap_statistics(candidate, reference, args.seed + len(retention_rows), args.bootstrap_draws)
            retention = retention_statistics(candidate, pixel, reference, args.seed + 1000 + len(retention_rows), args.bootstrap_draws)
            candidate_tail = policy_tail_statistics(candidate)
            reference_tail = policy_tail_statistics(reference)
            row = {
                "operator_label": operator,
                "split": "OOF",
                "policy": policy,
                "reference_policy": "FIXED_ALPHA_0.125",
                "pixel_denominator_policy": PIXEL_GRID_NAME,
                **bootstrap,
                **retention,
                **{f"candidate_{key}": value for key, value in candidate_tail.items()},
                **{f"reference_{key}": value for key, value in reference_tail.items()},
            }
            row["p10_ge_reference"] = bool(candidate_tail["p10_psnr_delta"] >= reference_tail["p10_psnr_delta"])
            row["worst_ge_reference"] = bool(candidate_tail["worst_psnr_delta"] >= reference_tail["worst_psnr_delta"])
            row["severe_le_reference"] = bool(candidate_tail["severe_le_0p2_count"] <= reference_tail["severe_le_0p2_count"])
            row["mean_lift_ci95_low_gt_0"] = bool(row["mean_lift_ci95_low"] > 0.0)
            row["block16_retention_ci95_low_ge_0p80"] = bool(row["retention_ci95_low"] >= args.block16_retention_min)
            row["block16_common_action_pass"] = bool(
                policy == "ORACLE_BLOCK16_GRID"
                and row["block16_retention_ci95_low_ge_0p80"]
                and row["mean_lift_ci95_low_gt_0"]
                and row["p10_ge_reference"]
                and row["worst_ge_reference"]
                and row["severe_le_reference"]
            )
            retention_rows.append(row)
            if policy == "ORACLE_BLOCK16_GRID":
                gate_rows.append(row)

    dual_pass = len(gate_rows) == len(args.operator_labels) and all(
        row["block16_common_action_pass"] for row in gate_rows
    )
    summary_rows = [
        row
        for row in read_rows(raw_dir / "v3l_a1_oracle_policy_summary.csv")
        if row["policy"] in {*COMMON_POLICIES, "FIXED_ALPHA_0.125"}
    ]
    confirm_summary_rows = [row for row in summary_rows if row["split"] == "CONFIRM_AUDIT_ONLY"]
    return retention_rows, gate_rows, summary_rows, confirm_summary_rows, operator_agreement(all_policy_rows), dual_pass


def run(args):
    if args.source_split.lower() != "train":
        raise ValueError("v3m-A0 is train-derived only")
    if sorted(float(value) for value in args.common_alphas) != [0.0, 0.125, 0.25, 0.5, 1.0]:
        raise ValueError("v3m-A0 common action set must be exactly [0, 0.125, 0.25, 0.5, 1]")
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "v3m_a0_common_action_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite A0 summary: {summary_path}")

    manifest = read_json(args.fresh_split_manifest)
    train_names = names_from_manifest(manifest, args.train_key, args.max_train_samples)
    confirm_names = names_from_manifest(manifest, args.confirm_key, args.max_confirm_samples)
    raw_dir = output_dir / "cloud_only_raw_common_action"
    if args.summarize_existing_raw:
        required = [
            raw_dir / "v3l_a1_oracle_policy_rows_cloud_only.csv",
            raw_dir / "v3l_a1_oracle_policy_oof_rows_cloud_only.csv",
            raw_dir / "v3l_a1_oracle_policy_summary.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("cannot rebuild summary; missing raw files: " + ", ".join(missing))
    else:
        raw_dir = run_common_oracle(args)
    retention_rows, gate_rows, policy_summary, confirm_summary, agreement_rows, dual_pass = summarize_common_action(args, raw_dir)

    decision = (
        "V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY"
        if dual_pass
        else "V3M_A0_BLOCK16_GRANULARITY_LOCK_FAIL_NO_BLOCK16_CONTROLLER"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "v3m_a0_granularity_retention_group_bootstrap.csv", retention_rows)
    write_rows(output_dir / "v3m_a0_common_action_gate.csv", gate_rows)
    write_rows(output_dir / "v3m_a0_common_action_oracle_summary.csv", policy_summary)
    write_rows(output_dir / "v3m_a0_confirm_audit_only_policy_summary.csv", confirm_summary)
    write_rows(output_dir / "v3m_a0_operator_agreement.csv", agreement_rows)

    source_manifest = {
        "route_id": ROUTE_ID,
        "parent_evidence_commit": args.parent_evidence_commit,
        "parent_cloud_worktree": args.parent_cloud_worktree,
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "v3m_script_sha256": sha256_file(__file__),
        "reused_v3l_a1_source_sha256": sha256_file(Path(v3l_a1.__file__)),
        "a0_closeout_sha256": sha256_file(args.a0_closeout),
        "fresh_split_manifest_sha256": sha256_file(args.fresh_split_manifest),
        "operator_artifact_manifest_sha256": sha256_file(args.operator_artifact_manifest),
        "row_order_sha256": sha256_lines(train_names),
        "confirm_row_order_sha256": sha256_lines(confirm_names),
        "clean_reference_group_manifest_sha256": sha256_file(args.fresh_split_manifest),
        "common_action_set": list(args.common_alphas),
        "locked_test_touched": False,
        "canary_authorized": False,
        "training_authorized": False,
        "raw_per_image_output": str(raw_dir),
    }
    write_json(output_dir / "v3m_a0_source_manifest.json", source_manifest)
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3m-A0a common-action granularity lock",
        "decision": decision,
        "next_stage_authorized": "v3m-A0b dense-grid and continuous-pixel mechanism audit only" if dual_pass else "none",
        "parent_evidence_commit": args.parent_evidence_commit,
        "locked_test_touched": False,
        "canary_authorized": False,
        "training_authorized": False,
        "new_model_search_authorized": False,
        "route_confirm_used_for_strategy_selection": False,
        "route_confirm_role": "audit-only; OOF-only gate selection",
        "metric_contract": {
            "baseline": "FIXED_ALPHA_0.125 on the same train-derived clean-reference grouped OOF rows",
            "common_action_set": list(args.common_alphas),
            "common_oracles": list(COMMON_POLICIES),
            "pixel_continuous_role": "not run in A0a; reserved for A0b diagnostic ceiling",
            "block16_retention_denominator": PIXEL_GRID_NAME,
            "block16_retention_ci95_low_min": args.block16_retention_min,
            "required_operator_labels": list(args.operator_labels),
            "gate_requirements": [
                "paired grouped mean-lift CI95 low > 0",
                "block16 retention CI95 low >= threshold",
                "block16 p10 and worst >= fixed alpha 0.125",
                "block16 severe count <= fixed alpha 0.125",
            ],
        },
        "sample_counts": {
            "oof_clean_reference_groups": len(train_names),
            "confirm_audit_only_images": len(confirm_names) if args.include_confirm_audit else 0,
        },
        "dual_operator_gate_rows": gate_rows,
        "retention_rows": retention_rows,
        "operator_agreement": agreement_rows,
        "source_manifest": source_manifest,
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
    parser.add_argument("--operator_artifact_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parent_evidence_commit", required=True)
    parser.add_argument("--parent_cloud_worktree", required=True)
    parser.add_argument("--source_branch", required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_confirm_samples", type=int, default=600)
    parser.add_argument("--operator_labels", nargs="+", default=["D_ref", "D_rep"])
    parser.add_argument("--common_alphas", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--oracle_block_sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--energy_eps", type=float, default=1e-12)
    parser.add_argument("--metric_eps_db", type=float, default=1e-9)
    parser.add_argument("--min_oracle_mean_lift_db", type=float, default=0.0)
    parser.add_argument("--block16_retention_min", type=float, default=0.80)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=None)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--include_confirm_audit", action="store_true")
    parser.add_argument("--summarize_existing_raw", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()
    if args.d7c_threshold is None:
        args.d7c_threshold = v3l_a1.D7C_THRESHOLD
    run(args)


if __name__ == "__main__":
    main()
