#!/usr/bin/env python3
"""v3p-A1 read-only reconstruction of the frozen v3m-A3 policy failure."""

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROUTE_ID = "haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712"
LADDER = (0.0, 0.125, 0.25, 0.5, 1.0)
KEY_BY_ALPHA = {0.0: "0", 0.125: "0p125", 0.25: "0p25", 0.5: "0p5", 1.0: "1"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def parse_bound(value):
    if value == "-inf":
        return -math.inf
    if value == "inf":
        return math.inf
    return float(value)


def read_bins(path):
    result = defaultdict(lambda: defaultdict(list))
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["operator_label"]][int(row["holdout_fold"])].append(row)
    for operator in result:
        for fold in result[operator]:
            rows = sorted(result[operator][fold], key=lambda item: int(item["bin_index"]))
            boundaries = []
            actions = []
            for index, row in enumerate(rows):
                if int(row["bin_index"]) != index:
                    raise ValueError(f"non-contiguous calibration bins for {operator}/fold={fold}")
                upper = parse_bound(row["score_upper_inclusive"])
                if math.isfinite(upper):
                    boundaries.append(upper)
                actions.append(int(row["monotone_action_index"]))
            if len(actions) != len(boundaries) + 1:
                raise ValueError(f"calibration boundary/action mismatch for {operator}/fold={fold}")
            result[operator][fold] = (boundaries, actions)
    return result


def selected_alpha(bins, operator, fold, energy):
    boundaries, actions = bins[operator][fold]
    action_index = actions[bisect.bisect_right(boundaries, energy)]
    return LADDER[action_index]


def read_policy_rows(path):
    policy = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["operator_label"], row["name"])
            if key in policy:
                raise ValueError(f"duplicate policy row: {key}")
            policy[key] = row
    return policy


def psnr_from_sse(sse, pixel_count):
    return 10.0 * math.log10(pixel_count / max(sse, 1e-30))


def summarize_path(values):
    return {
        "net_sse": math.fsum(values),
        "beneficial_sse": math.fsum(max(value, 0.0) for value in values),
        "harmful_sse": math.fsum(max(-value, 0.0) for value in values),
        "count": len(values),
    }


def finalize_image(state, policy_row, output_rows, path_values, executor_values):
    key = state["key"]
    fixed_sse = math.fsum(state["fixed"])
    hard_sse = math.fsum(state["hard"])
    pixel_count = state["pixel_count"]
    fixed_psnr = psnr_from_sse(fixed_sse, pixel_count)
    hard_psnr = psnr_from_sse(hard_sse, pixel_count)
    base_psnr = float(policy_row["base_psnr"])
    rendered_psnr = base_psnr + float(policy_row["policy_psnr_delta"])
    rendered_sse = pixel_count * (10.0 ** (-rendered_psnr / 10.0))
    expected_counts = [int(policy_row[f"selected_action_{index}_count"]) for index in range(len(LADDER))]
    observed_counts = [state["counts"][alpha] for alpha in LADDER]
    count_match = expected_counts == observed_counts
    for alpha, values in state["paths"].items():
        path_values[(key[0], alpha)].extend(values)
    executor_values[key[0]].append(rendered_sse - hard_sse)
    output_rows.append({
        "operator_label": key[0],
        "name": key[1],
        "fold": state["fold"],
        "pixel_count": pixel_count,
        "fixed_psnr_delta_canonical": fixed_psnr - base_psnr,
        "reference_fixed_psnr_delta": float(policy_row["reference_fixed_psnr_delta"]),
        "fixed_replay_abs_diff_db": abs(fixed_psnr - base_psnr - float(policy_row["reference_fixed_psnr_delta"])),
        "selector_hard_lift_vs_fixed_db": hard_psnr - fixed_psnr,
        "rendered_policy_lift_vs_fixed_db": rendered_psnr - fixed_psnr,
        "executor_interaction_sse": rendered_sse - hard_sse,
        "selected_action_counts_match": count_match,
        **{f"selected_alpha_{KEY_BY_ALPHA[alpha]}_count": state["counts"][alpha] for alpha in LADDER},
    })
    return count_match


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "closeout": output_dir / f"{args.run_tag}_closeout.json",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "paths": output_dir / f"{args.run_tag}_action_path_decomposition.csv",
        "images": output_dir / f"{args.run_tag}_image_reconstruction_cloud_only.csv",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite v3p-A1 outputs: " + ", ".join(existing))
    a0 = json.loads(Path(args.a0_closeout).read_text(encoding="utf-8"))
    if a0.get("decision") != "V3P_A0_CANONICAL_NUMERICAL_PASS_AUTHORIZE_A1_RECONSTRUCTION_ONLY":
        raise ValueError("v3p-A1 requires the A0 canonical numerical pass closeout")
    bins = read_bins(args.calibration_bins)
    policy = read_policy_rows(args.policy_rows)
    image_rows = []
    path_values = defaultdict(list)
    executor_values = defaultdict(list)
    seen_policy = set()
    unmatched_bins = 0
    count_mismatches = 0
    state = None

    def flush_state():
        nonlocal state, count_mismatches
        if state is None:
            return
        key = state["key"]
        if key not in policy:
            raise ValueError(f"canonical block image missing v3m policy row: {key}")
        seen_policy.add(key)
        if not finalize_image(state, policy[key], image_rows, path_values, executor_values):
            count_mismatches += 1
        state = None

    with Path(args.canonical_blocks).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["operator_label"], row["name"])
            if state is None or key != state["key"]:
                flush_state()
                state = {
                    "key": key,
                    "fold": int(row["fold"]),
                    "pixel_count": 0,
                    "fixed": [],
                    "hard": [],
                    "counts": Counter(),
                    "paths": defaultdict(list),
                }
            try:
                alpha = selected_alpha(bins, key[0], state["fold"], float(row["direct_step_energy"]))
            except ValueError:
                unmatched_bins += 1
                raise
            alpha_key = KEY_BY_ALPHA.get(alpha)
            if alpha_key is None:
                raise ValueError(f"unexpected selected alpha: {alpha}")
            fixed = float(row["candidate_sse_0p125"])
            selected = float(row[f"candidate_sse_{alpha_key}"])
            state["pixel_count"] += int(row["candidate_pixel_count"])
            state["fixed"].append(fixed)
            state["hard"].append(selected)
            state["counts"][alpha] += 1
            if alpha != 0.125:
                state["paths"][alpha].append(fixed - selected)
    flush_state()

    missing_policy = sorted(set(policy) - seen_policy)
    fixed_replay_max = max(row["fixed_replay_abs_diff_db"] for row in image_rows)
    path_rows = []
    for (operator, alpha), values in sorted(path_values.items()):
        row = {"operator_label": operator, "selected_alpha": alpha, **summarize_path(values)}
        path_rows.append(row)
    operator_rows = []
    for operator in sorted({row["operator_label"] for row in image_rows}):
        rows = [row for row in image_rows if row["operator_label"] == operator]
        selector = [row["selector_hard_lift_vs_fixed_db"] for row in rows]
        rendered = [row["rendered_policy_lift_vs_fixed_db"] for row in rows]
        executor = executor_values[operator]
        operator_rows.append({
            "operator_label": operator,
            "image_count": len(rows),
            "selector_mean_lift_db": math.fsum(selector) / len(selector),
            "rendered_mean_lift_db": math.fsum(rendered) / len(rendered),
            "selector_p10_lift_db": sorted(selector)[int(0.1 * (len(selector) - 1))],
            "rendered_p10_lift_db": sorted(rendered)[int(0.1 * (len(rendered) - 1))],
            "executor_interaction_sse_total": math.fsum(executor),
            "executor_interaction_sse_positive_images": sum(value > 0.0 for value in executor),
        })
    hard_fail = bool(missing_policy) or unmatched_bins or count_mismatches or fixed_replay_max > args.replay_tolerance_db
    decision = "V3P_A1_RECONSTRUCTION_PASS_AUTHORIZE_A2_CONSTRAINED_G1_ORACLE_ONLY" if not hard_fail else "V3P_A1_RECONSTRUCTION_INTEGRITY_FAIL_STOP"
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3p-A1-reconstruction",
        "state": "COMPLETED_GATE_PASS" if not hard_fail else "COMPLETED_GATE_FAIL",
        "gate_type": "structural_integrity",
        "decision": decision,
        "metric_contract": "reconstruct frozen A2 bins on v3p canonical block SSE; compare action counts and fixed replay to v3m A3",
        "authorizes": "v3p-A2 constrained G1 oracle only" if not hard_fail else "none",
        "reason": "read-only action-path reconstruction and executor interaction decomposition",
        "canonical_image_count": len(image_rows),
        "v3m_policy_image_count": len(policy),
        "missing_policy_rows": len(missing_policy),
        "unmatched_calibration_bins": unmatched_bins,
        "action_count_mismatches": count_mismatches,
        "fixed_replay_max_abs_diff_db": fixed_replay_max,
        "operator_rows": operator_rows,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    source_manifest = {
        "route_id": ROUTE_ID,
        "canonical_blocks_sha256": sha256_file(args.canonical_blocks),
        "policy_rows_sha256": sha256_file(args.policy_rows),
        "calibration_bins_sha256": sha256_file(args.calibration_bins),
        "a0_closeout_sha256": sha256_file(args.a0_closeout),
        "raw_image_reconstruction_cloud_only": str(outputs["images"]),
    }
    summary = {**closeout, "action_path_rows": path_rows, "source_manifest": source_manifest}
    write_rows(outputs["images"], image_rows)
    write_rows(outputs["paths"], path_rows)
    write_json(outputs["closeout"], closeout)
    write_json(outputs["summary"], summary)
    write_json(outputs["source"], source_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_blocks", required=True)
    parser.add_argument("--policy_rows", required=True)
    parser.add_argument("--calibration_bins", required=True)
    parser.add_argument("--a0_closeout", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", default="v3p_a1")
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
