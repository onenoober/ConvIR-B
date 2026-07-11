#!/usr/bin/env python3
"""Read-only v3m-A0b quantization-gap audit for frozen v3l/v3m OOF rows."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROUTE_ID = "haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711"
OPERATORS = ("D_ref", "D_rep")
FIXED_POLICY = "FIXED_ALPHA_0.125"
PIXEL_CONTINUOUS_POLICY = "ORACLE_PIXEL_SCALAR_CONTINUOUS"
PAIR_SPECS = (
    ("image_dense_vs_common", "ORACLE_IMAGE_GRID", "ORACLE_IMAGE_GRID"),
    ("block16_dense_vs_common", "ORACLE_BLOCK16_GRID", "ORACLE_BLOCK16_GRID"),
    ("block32_dense_vs_common", "ORACLE_BLOCK32_GRID", "ORACLE_BLOCK32_GRID"),
    ("pixel_continuous_vs_common", PIXEL_CONTINUOUS_POLICY, "ORACLE_PIXEL_GRID"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_sha(label, path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {label}: expected {expected}, got {actual}")
    return actual


def policy_values(rows, operator_label, policy):
    values = {}
    seeds = set()
    for row in rows:
        if row["split"] != "OOF" or row["operator_label"] != operator_label or row["policy"] != policy:
            continue
        name = row["name"]
        if name in values:
            raise ValueError(f"duplicate OOF row for {operator_label}/{policy}/{name}")
        values[name] = float(row["psnr_delta"])
        seeds.add(row["seed"])
    if not values:
        raise ValueError(f"missing OOF rows for {operator_label}/{policy}")
    if len(seeds) != 1:
        raise ValueError(f"unexpected seed multiplicity for {operator_label}/{policy}: {sorted(seeds)}")
    return values, next(iter(seeds))


def summary_row(rows, operator_label, policy):
    matches = [
        row
        for row in rows
        if row["split"] == "OOF" and row["operator_label"] == operator_label and row["policy"] == policy
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one summary row for {operator_label}/{policy}, got {len(matches)}")
    return matches[0]


def summary_mean(rows, operator_label, policy):
    return float(summary_row(rows, operator_label, policy)["mean_psnr_delta"])


def paired_bootstrap(values, draws, seed):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap requires a nonempty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def assert_summary_consistency(raw_values, summary_rows, operator_label, policy, source):
    raw_mean = float(np.mean(list(raw_values.values())))
    compact_mean = summary_mean(summary_rows, operator_label, policy)
    difference = abs(raw_mean - compact_mean)
    if difference > 1e-12:
        raise ValueError(
            f"{source} summary mismatch for {operator_label}/{policy}: {raw_mean} vs {compact_mean}"
        )
    return raw_mean, compact_mean, difference


def tail_safety_against_fixed(summary_rows, operator_label, policy):
    candidate = summary_row(summary_rows, operator_label, policy)
    reference = summary_row(summary_rows, operator_label, FIXED_POLICY)
    p10_pass = float(candidate["p10_psnr_delta"]) >= float(reference["p10_psnr_delta"])
    severe_pass = int(candidate["regression_le_0p2_count"]) <= int(reference["regression_le_0p2_count"])
    return p10_pass and severe_pass, p10_pass, severe_pass


def verify_contract_json(parent_granularity_summary, v3m_source_manifest):
    parent = json.loads(Path(parent_granularity_summary).read_text(encoding="utf-8"))
    v3m = json.loads(Path(v3m_source_manifest).read_text(encoding="utf-8"))
    if parent["locked_test_touched"] or parent["canary_authorized"] or parent["training_authorized"]:
        raise ValueError("parent v3l A1 authorization flags are incompatible with A0b contract")
    expected_dense = [index / 32.0 for index in range(33)]
    if parent["metric_contract"]["oracle_grid"] != expected_dense:
        raise ValueError("parent dense action grid is not the preregistered 33-level ladder")
    expected_common = [0.0, 0.125, 0.25, 0.5, 1.0]
    if v3m["common_action_set"] != expected_common:
        raise ValueError("v3m common action ladder does not match the A0a contract")
    if v3m["locked_test_touched"] or v3m["canary_authorized"] or v3m["training_authorized"]:
        raise ValueError("v3m A0a authorization flags are incompatible with A0b contract")
    return {"parent_v3l_a1_decision": parent["decision"], "v3m_common_action_set": expected_common}


def run(args):
    output_dir = Path(args.output_dir)
    outputs = (
        output_dir / f"{args.run_id}_quantization_gap_summary.csv",
        output_dir / f"{args.run_id}_cross_audit.json",
        output_dir / f"{args.run_id}_source_manifest.json",
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite A0b outputs: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "parent_operator_manifest": require_sha(
            "parent_operator_manifest", args.parent_operator_manifest, args.expected_parent_operator_manifest_sha256
        ),
        "parent_a0_closeout": require_sha(
            "parent_a0_closeout", args.parent_a0_closeout, args.expected_parent_a0_closeout_sha256
        ),
        "parent_oof_rows": require_sha(
            "parent_oof_rows", args.parent_oof_rows, args.expected_parent_oof_rows_sha256
        ),
        "parent_policy_summary": require_sha(
            "parent_policy_summary", args.parent_policy_summary, args.expected_parent_policy_summary_sha256
        ),
        "parent_granularity_summary": require_sha(
            "parent_granularity_summary", args.parent_granularity_summary, args.expected_parent_granularity_summary_sha256
        ),
        "v3m_oof_rows": require_sha("v3m_oof_rows", args.v3m_oof_rows, args.expected_v3m_oof_rows_sha256),
        "v3m_policy_summary": require_sha(
            "v3m_policy_summary", args.v3m_policy_summary, args.expected_v3m_policy_summary_sha256
        ),
        "v3m_source_manifest": require_sha(
            "v3m_source_manifest", args.v3m_source_manifest, args.expected_v3m_source_manifest_sha256
        ),
    }
    contract_flags = verify_contract_json(args.parent_granularity_summary, args.v3m_source_manifest)
    parent_rows = read_rows(args.parent_oof_rows)
    parent_summary_rows = read_rows(args.parent_policy_summary)
    v3m_rows = read_rows(args.v3m_oof_rows)
    v3m_summary_rows = read_rows(args.v3m_policy_summary)

    summaries = []
    fixed_replay = []
    overall_pass = True
    for operator_index, operator_label in enumerate(OPERATORS):
        parent_fixed, parent_seed = policy_values(parent_rows, operator_label, FIXED_POLICY)
        v3m_fixed, v3m_seed = policy_values(v3m_rows, operator_label, FIXED_POLICY)
        if parent_seed != v3m_seed:
            raise ValueError(f"seed mismatch for {operator_label}: {parent_seed} vs {v3m_seed}")
        if set(parent_fixed) != set(v3m_fixed):
            raise ValueError(f"fixed-alpha OOF names mismatch for {operator_label}")
        fixed_diffs = [abs(parent_fixed[name] - v3m_fixed[name]) for name in sorted(parent_fixed)]
        fixed_max_abs = max(fixed_diffs)
        fixed_pass = fixed_max_abs <= args.fixed_replay_tolerance_db
        fixed_replay.append(
            {
                "operator_label": operator_label,
                "seed": parent_seed,
                "group_count": len(parent_fixed),
                "fixed_alpha_max_abs_psnr_delta_diff_db": fixed_max_abs,
                "fixed_alpha_exact_replay_pass": fixed_pass,
            }
        )
        overall_pass = overall_pass and fixed_pass

        for pair_index, (pair_name, parent_policy, v3m_policy) in enumerate(PAIR_SPECS):
            parent_values, _ = policy_values(parent_rows, operator_label, parent_policy)
            v3m_values, _ = policy_values(v3m_rows, operator_label, v3m_policy)
            if set(parent_values) != set(v3m_values) or set(parent_values) != set(parent_fixed):
                raise ValueError(f"OOF names mismatch for {operator_label}/{pair_name}")
            parent_mean, parent_compact_mean, parent_summary_diff = assert_summary_consistency(
                parent_values, parent_summary_rows, operator_label, parent_policy, "parent"
            )
            v3m_mean, v3m_compact_mean, v3m_summary_diff = assert_summary_consistency(
                v3m_values, v3m_summary_rows, operator_label, v3m_policy, "v3m"
            )
            names = sorted(parent_values)
            gaps = np.asarray([parent_values[name] - v3m_values[name] for name in names], dtype=np.float64)
            ci95_low, ci95_high = paired_bootstrap(
                gaps, args.bootstrap_draws, args.seed + operator_index * 100 + pair_index
            )
            pointwise_dominance_applicable = parent_policy != PIXEL_CONTINUOUS_POLICY
            monotonic_pass = bool(
                not pointwise_dominance_applicable
                or float(np.min(gaps)) >= -args.grid_monotonic_tolerance_db
            )
            ci_pass = bool(ci95_high <= args.max_mean_gap_db)
            parent_tail_pass, parent_p10_pass, parent_severe_pass = tail_safety_against_fixed(
                parent_summary_rows, operator_label, parent_policy
            )
            v3m_tail_pass, v3m_p10_pass, v3m_severe_pass = tail_safety_against_fixed(
                v3m_summary_rows, operator_label, v3m_policy
            )
            tail_safety_pass = parent_tail_pass and v3m_tail_pass
            pair_pass = monotonic_pass and ci_pass and tail_safety_pass
            overall_pass = overall_pass and pair_pass
            summaries.append(
                {
                    "operator_label": operator_label,
                    "seed": parent_seed,
                    "pair_name": pair_name,
                    "dense_or_continuous_policy": parent_policy,
                    "common_five_level_policy": v3m_policy,
                    "group_count": len(names),
                    "dense_or_continuous_mean_psnr_delta": parent_mean,
                    "common_mean_psnr_delta": v3m_mean,
                    "mean_gap_db": float(np.mean(gaps)),
                    "mean_gap_ci95_low_db": ci95_low,
                    "mean_gap_ci95_high_db": ci95_high,
                    "gap_p50_db": float(np.quantile(gaps, 0.5)),
                    "gap_p90_db": float(np.quantile(gaps, 0.9)),
                    "gap_p95_db": float(np.quantile(gaps, 0.95)),
                    "gap_max_db": float(np.max(gaps)),
                    "gap_min_db": float(np.min(gaps)),
                    "gap_gt_0p005_fraction": float(np.mean(gaps > args.max_mean_gap_db)),
                    "parent_summary_abs_difference_db": parent_summary_diff,
                    "v3m_summary_abs_difference_db": v3m_summary_diff,
                    "pointwise_dominance_check_applicable": pointwise_dominance_applicable,
                    "grid_monotonicity_pass": monotonic_pass,
                    "mean_gap_ci95_high_le_0p005": ci_pass,
                    "parent_policy_p10_ge_fixed": parent_p10_pass,
                    "parent_policy_severe_le_fixed": parent_severe_pass,
                    "v3m_policy_p10_ge_fixed": v3m_p10_pass,
                    "v3m_policy_severe_le_fixed": v3m_severe_pass,
                    "both_policy_tail_safety_pass": tail_safety_pass,
                    "quantization_gap_pair_pass": pair_pass,
                }
            )

    decision = (
        "V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY"
        if overall_pass
        else "V3M_A0B_QUANTIZATION_GAP_OR_PROVENANCE_FAIL_NO_A1"
    )
    source_manifest = {
        "route_id": ROUTE_ID,
        "audit_mode": "read_only_existing_frozen_rows_no_inference",
        "run_id": args.run_id,
        "input_sha256": input_hashes,
        "locked_test_touched": False,
        "canary_authorized": False,
        "training_authorized": False,
        "raw_per_image_outputs_written": False,
    }
    report = {
        "route_id": ROUTE_ID,
        "phase": f"{args.run_id} dense-grid and continuous-pixel mechanism cross-audit",
        "decision": decision,
        "next_stage_authorized": "v3m-A1 feasible local actuation audit only" if overall_pass else "none",
        "locked_test_touched": False,
        "canary_authorized": False,
        "training_authorized": False,
        "route_confirm_used_for_strategy_selection": False,
        "metric_contract": {
            "source_split": "train-derived clean-reference OOF only",
            "fixed_reference": FIXED_POLICY,
            "dense_grid": [index / 32.0 for index in range(33)],
            "common_grid": [0.0, 0.125, 0.25, 0.5, 1.0],
            "mean_gap_ci95_high_max_db": args.max_mean_gap_db,
            "fixed_replay_tolerance_db": args.fixed_replay_tolerance_db,
            "grid_monotonic_tolerance_db": args.grid_monotonic_tolerance_db,
            "continuous_pointwise_dominance_check": "not applicable after output clamp",
            "paired_bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.seed,
        },
        "contract_flags": contract_flags,
        "fixed_alpha_replay": fixed_replay,
        "quantization_gap_rows": summaries,
        "source_manifest": source_manifest,
    }
    write_rows(outputs[0], summaries)
    write_json(outputs[1], report)
    write_json(outputs[2], source_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_operator_manifest", required=True)
    parser.add_argument("--parent_a0_closeout", required=True)
    parser.add_argument("--parent_oof_rows", required=True)
    parser.add_argument("--parent_policy_summary", required=True)
    parser.add_argument("--parent_granularity_summary", required=True)
    parser.add_argument("--v3m_oof_rows", required=True)
    parser.add_argument("--v3m_policy_summary", required=True)
    parser.add_argument("--v3m_source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", default="v3m_a0b")
    parser.add_argument("--expected_parent_operator_manifest_sha256", required=True)
    parser.add_argument("--expected_parent_a0_closeout_sha256", required=True)
    parser.add_argument("--expected_parent_oof_rows_sha256", required=True)
    parser.add_argument("--expected_parent_policy_summary_sha256", required=True)
    parser.add_argument("--expected_parent_granularity_summary_sha256", required=True)
    parser.add_argument("--expected_v3m_oof_rows_sha256", required=True)
    parser.add_argument("--expected_v3m_policy_summary_sha256", required=True)
    parser.add_argument("--expected_v3m_source_manifest_sha256", required=True)
    parser.add_argument("--max_mean_gap_db", type=float, default=0.005)
    parser.add_argument("--fixed_replay_tolerance_db", type=float, default=1e-12)
    parser.add_argument("--grid_monotonic_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--monotonic_tolerance_db", type=float, default=None)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    if args.max_mean_gap_db <= 0.0:
        raise ValueError("max_mean_gap_db must be positive")
    if args.monotonic_tolerance_db is not None:
        args.grid_monotonic_tolerance_db = args.monotonic_tolerance_db
    run(args)


if __name__ == "__main__":
    main()
