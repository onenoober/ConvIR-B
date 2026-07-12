#!/usr/bin/env python3
"""v3m-A2 fold-separated OOF calibration audit for direct-step energy."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROUTE_ID = "haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711"
OPERATORS = ("D_ref", "D_rep")
ACTION_LADDER = (0.0, 0.125, 0.25, 0.5, 1.0)
ACTION_TO_INDEX = {f"{value:.3f}".rstrip("0").rstrip("."): idx for idx, value in enumerate(ACTION_LADDER)}
ACTION_TO_INDEX.update({"0.0": 0, "0.125": 1, "0.25": 2, "0.5": 3, "1.0": 4})
FIXED_INDEX = 1
PRIMARY_SIGNAL = "direct_step_energy"


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


def parse_action_index(value):
    key = str(value).strip()
    if key in ACTION_TO_INDEX:
        return ACTION_TO_INDEX[key]
    numeric = float(key)
    distances = [abs(numeric - action) for action in ACTION_LADDER]
    idx = int(np.argmin(distances))
    if distances[idx] > 1e-9:
        raise ValueError(f"oracle alpha is not in the common ladder: {value}")
    return idx


def bootstrap_interval(values, draws, seed):
    array = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


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


def average_precision(scores, labels):
    label_array = np.asarray(labels, dtype=np.int64)
    positive = int(np.sum(label_array == 1))
    if positive == 0:
        return None
    score_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score_array, kind="mergesort")
    sorted_labels = label_array[order]
    true_positive_cumsum = np.cumsum(sorted_labels)
    precision_at_rank = true_positive_cumsum / (np.arange(sorted_labels.size) + 1.0)
    return float(np.sum(precision_at_rank[sorted_labels == 1]) / positive)


def spearman_with_bin_index(values):
    clean = [(idx, value) for idx, value in enumerate(values) if not math.isnan(value)]
    if len(clean) < 2:
        return float("nan")
    x = np.asarray([item[0] for item in clean], dtype=np.float64)
    y = np.asarray([item[1] for item in clean], dtype=np.float64)
    y_order = np.argsort(y, kind="mergesort")
    y_sorted = y[y_order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y_sorted.size:
        end = start + 1
        while end < y_sorted.size and y_sorted[end] == y_sorted[start]:
            end += 1
        ranks[y_order[start:end]] = (start + 1 + end) / 2.0
        start = end
    x = x - x.mean()
    ranks = ranks - ranks.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(ranks * ranks)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * ranks) / denom)


def monotonic_violations(values):
    previous = None
    violations = 0
    for value in values:
        if math.isnan(value):
            continue
        if previous is not None and value < previous:
            violations += 1
        previous = value
    return violations


def lower_median_action(labels):
    counts = np.bincount(labels.astype(np.int64), minlength=len(ACTION_LADDER))
    target = (int(np.sum(counts)) + 1) // 2
    cumulative = np.cumsum(counts)
    return int(np.searchsorted(cumulative, target, side="left"))


def quantile_boundaries(scores, bin_count):
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    raw = np.quantile(scores, quantiles, method="linear")
    boundaries = []
    for value in raw.tolist():
        if not math.isfinite(value):
            continue
        if boundaries and value <= boundaries[-1]:
            continue
        boundaries.append(float(value))
    return np.asarray(boundaries, dtype=np.float64)


def validate_a1_contract(args):
    hashes = {
        "a1_block_rows_cloud_only": verify_sha(
            "a1_block_rows_cloud_only", args.a1_block_rows, args.expected_a1_block_rows_sha256
        ),
        "a1_summary": verify_sha("a1_summary", args.a1_summary, args.expected_a1_summary_sha256),
        "a1_signal_summary": verify_sha(
            "a1_signal_summary", args.a1_signal_summary, args.expected_a1_signal_summary_sha256
        ),
        "a1_source_manifest": verify_sha(
            "a1_source_manifest", args.a1_source_manifest, args.expected_a1_source_manifest_sha256
        ),
    }
    with Path(args.a1_summary).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("decision") != "V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY":
        raise ValueError("A1 summary does not authorize A2")
    for flag in ("training_authorized", "canary_authorized", "locked_test_touched"):
        if summary.get(flag):
            raise ValueError(f"A1 summary has forbidden flag set: {flag}")
    if summary.get("route_confirm_used_for_strategy_selection"):
        raise ValueError("A1 summary used route-confirm for strategy selection")
    if PRIMARY_SIGNAL not in summary.get("passing_signals", []):
        raise ValueError("A1 summary does not list direct_step_energy as a passing signal")
    line_count = 0
    with Path(args.a1_block_rows).open("r", encoding="utf-8", newline="") as handle:
        for line_count, _ in enumerate(handle, start=1):
            pass
    if line_count != args.expected_a1_block_rows_line_count:
        raise ValueError(
            f"A1 block row line count mismatch: expected {args.expected_a1_block_rows_line_count}, got {line_count}"
        )
    return hashes


def load_block_rows(path):
    data = {
        operator: {
            "scores": [],
            "labels": [],
            "folds": [],
            "image_ids": [],
            "names": [],
            "name_to_id": {},
        }
        for operator in OPERATORS
    }
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "operator_label",
            "fold",
            "name",
            "oracle_alpha",
            PRIMARY_SIGNAL,
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"A1 block table missing required columns: {sorted(missing)}")
        for row in reader:
            operator = row["operator_label"]
            if operator not in data:
                raise ValueError(f"unexpected operator label: {operator}")
            name_to_id = data[operator]["name_to_id"]
            name = row["name"]
            if name not in name_to_id:
                name_to_id[name] = len(data[operator]["names"])
                data[operator]["names"].append(name)
            data[operator]["scores"].append(float(row[PRIMARY_SIGNAL]))
            data[operator]["labels"].append(parse_action_index(row["oracle_alpha"]))
            data[operator]["folds"].append(int(row["fold"]))
            data[operator]["image_ids"].append(name_to_id[name])
    for operator, values in data.items():
        values["scores"] = np.asarray(values["scores"], dtype=np.float64)
        values["labels"] = np.asarray(values["labels"], dtype=np.int8)
        values["folds"] = np.asarray(values["folds"], dtype=np.int8)
        values["image_ids"] = np.asarray(values["image_ids"], dtype=np.int32)
        del values["name_to_id"]
    return data


def per_image_metrics(scores, labels, predictions, image_ids):
    order = np.argsort(image_ids, kind="mergesort")
    sorted_ids = image_ids[order]
    metrics = []
    start = 0
    while start < sorted_ids.size:
        end = start + 1
        while end < sorted_ids.size and sorted_ids[end] == sorted_ids[start]:
            end += 1
        idx = order[start:end]
        group_scores = scores[idx]
        group_labels = labels[idx].astype(np.int64)
        group_predictions = predictions[idx].astype(np.int64)
        positive = (group_labels > FIXED_INDEX).astype(np.int64)
        predicted_positive = group_predictions > FIXED_INDEX
        true_positive = int(np.sum(predicted_positive & (positive == 1)))
        false_positive = int(np.sum(predicted_positive & (positive == 0)))
        false_negative = int(np.sum((~predicted_positive) & (positive == 1)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else float("nan")
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else float("nan")
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            f1 = float("nan")
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        auc = roc_auc(group_scores, positive)
        ap = average_precision(group_scores, positive)
        prevalence = float(np.mean(positive))
        metrics.append(
            {
                "image_id": int(sorted_ids[start]),
                "block_count": int(group_labels.size),
                "fixed_mae": float(np.mean(np.abs(group_labels - FIXED_INDEX))),
                "calibrated_mae": float(np.mean(np.abs(group_labels - group_predictions))),
                "ordinal_mae_improvement": float(
                    np.mean(np.abs(group_labels - FIXED_INDEX)) - np.mean(np.abs(group_labels - group_predictions))
                ),
                "exact_accuracy": float(np.mean(group_labels == group_predictions)),
                "oracle_positive_fraction": prevalence,
                "predicted_positive_fraction": float(np.mean(predicted_positive)),
                "escalation_precision": precision,
                "escalation_recall": recall,
                "escalation_f1": f1,
                "escalation_auroc": float("nan") if auc is None else float(auc),
                "escalation_ap_lift": float("nan") if ap is None else float(ap - prevalence),
            }
        )
        start = end
    return metrics


def summarize_values(metrics, key, draws, seed):
    values = [row[key] for row in metrics if not math.isnan(row[key])]
    if not values:
        return {
            f"{key}_mean": float("nan"),
            f"{key}_ci95_low": float("nan"),
            f"{key}_ci95_high": float("nan"),
            f"{key}_valid_image_count": 0,
        }
    low, high = bootstrap_interval(values, draws, seed)
    return {
        f"{key}_mean": float(np.mean(values)),
        f"{key}_ci95_low": low,
        f"{key}_ci95_high": high,
        f"{key}_valid_image_count": len(values),
    }


def calibrate_operator(operator, values, args):
    scores = values["scores"]
    labels = values["labels"]
    folds = values["folds"]
    image_ids = values["image_ids"]
    predictions = np.full(labels.shape, -1, dtype=np.int8)
    fold_rows = []
    bin_rows = []

    unique_folds = sorted(int(fold) for fold in np.unique(folds))
    if unique_folds != list(range(args.fold_count)):
        raise ValueError(f"{operator} fold set mismatch: expected 0..{args.fold_count - 1}, got {unique_folds}")

    for holdout_fold in unique_folds:
        train_mask = folds != holdout_fold
        eval_mask = folds == holdout_fold
        train_scores = scores[train_mask]
        train_labels = labels[train_mask]
        boundaries = quantile_boundaries(train_scores, args.bin_count)
        train_bins = np.searchsorted(boundaries, train_scores, side="right").astype(np.int16)
        eval_scores = scores[eval_mask]
        eval_labels = labels[eval_mask]
        eval_bins = np.searchsorted(boundaries, eval_scores, side="right").astype(np.int16)
        raw_map = []
        monotone_map = []
        previous = 0
        bin_count_actual = int(boundaries.size + 1)
        for bin_index in range(bin_count_actual):
            bin_labels = train_labels[train_bins == bin_index]
            if bin_labels.size == 0:
                median = previous
            else:
                median = lower_median_action(bin_labels)
            raw_map.append(median)
            previous = max(previous, median)
            monotone_map.append(previous)
        monotone_map_array = np.asarray(monotone_map, dtype=np.int8)
        eval_predictions = monotone_map_array[eval_bins]
        predictions[eval_mask] = eval_predictions

        heldout_mean_action_by_bin = []
        heldout_positive_rate_by_bin = []
        for bin_index in range(bin_count_actual):
            train_bin_mask = train_bins == bin_index
            eval_bin_mask = eval_bins == bin_index
            train_bin_labels = train_labels[train_bin_mask]
            eval_bin_labels = eval_labels[eval_bin_mask]
            train_counts = np.bincount(train_bin_labels.astype(np.int64), minlength=len(ACTION_LADDER))
            eval_counts = np.bincount(eval_bin_labels.astype(np.int64), minlength=len(ACTION_LADDER))
            heldout_mean_action = float(np.mean(eval_bin_labels)) if eval_bin_labels.size else float("nan")
            heldout_positive_rate = float(np.mean(eval_bin_labels > FIXED_INDEX)) if eval_bin_labels.size else float("nan")
            heldout_mean_action_by_bin.append(heldout_mean_action)
            heldout_positive_rate_by_bin.append(heldout_positive_rate)
            lower = float("-inf") if bin_index == 0 else float(boundaries[bin_index - 1])
            upper = float("inf") if bin_index == bin_count_actual - 1 else float(boundaries[bin_index])
            bin_rows.append(
                {
                    "operator_label": operator,
                    "holdout_fold": holdout_fold,
                    "bin_index": bin_index,
                    "score_lower_exclusive": lower,
                    "score_upper_inclusive": upper,
                    "train_block_count": int(train_bin_labels.size),
                    "heldout_block_count": int(eval_bin_labels.size),
                    "train_mean_action_index": float(np.mean(train_bin_labels)) if train_bin_labels.size else float("nan"),
                    "heldout_mean_action_index": heldout_mean_action,
                    "train_positive_rate": float(np.mean(train_bin_labels > FIXED_INDEX)) if train_bin_labels.size else float("nan"),
                    "heldout_positive_rate": heldout_positive_rate,
                    "raw_median_action_index": raw_map[bin_index],
                    "monotone_action_index": monotone_map[bin_index],
                    "monotone_alpha": ACTION_LADDER[monotone_map[bin_index]],
                    "train_count_alpha_0": int(train_counts[0]),
                    "train_count_alpha_0p125": int(train_counts[1]),
                    "train_count_alpha_0p25": int(train_counts[2]),
                    "train_count_alpha_0p5": int(train_counts[3]),
                    "train_count_alpha_1": int(train_counts[4]),
                    "heldout_count_alpha_0": int(eval_counts[0]),
                    "heldout_count_alpha_0p125": int(eval_counts[1]),
                    "heldout_count_alpha_0p25": int(eval_counts[2]),
                    "heldout_count_alpha_0p5": int(eval_counts[3]),
                    "heldout_count_alpha_1": int(eval_counts[4]),
                }
            )

        fold_image_metrics = per_image_metrics(
            eval_scores,
            eval_labels,
            eval_predictions,
            image_ids[eval_mask],
        )
        global_positive = eval_labels > FIXED_INDEX
        global_predicted_positive = eval_predictions > FIXED_INDEX
        tp = int(np.sum(global_positive & global_predicted_positive))
        fp = int(np.sum((~global_positive) & global_predicted_positive))
        fn = int(np.sum(global_positive & (~global_predicted_positive)))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if not math.isnan(precision) and not math.isnan(recall) and precision + recall else float("nan")
        fold_rows.append(
            {
                "operator_label": operator,
                "holdout_fold": holdout_fold,
                "train_block_count": int(np.sum(train_mask)),
                "heldout_block_count": int(np.sum(eval_mask)),
                "heldout_image_count": len(fold_image_metrics),
                "bin_count": bin_count_actual,
                "raw_map": " ".join(str(item) for item in raw_map),
                "monotone_map": " ".join(str(item) for item in monotone_map),
                "train_map_non_decreasing": all(
                    monotone_map[index] <= monotone_map[index + 1] for index in range(len(monotone_map) - 1)
                ),
                "heldout_mean_action_spearman": spearman_with_bin_index(heldout_mean_action_by_bin),
                "heldout_positive_rate_spearman": spearman_with_bin_index(heldout_positive_rate_by_bin),
                "heldout_mean_action_monotonic_violations": monotonic_violations(heldout_mean_action_by_bin),
                "heldout_positive_rate_monotonic_violations": monotonic_violations(heldout_positive_rate_by_bin),
                "block_fixed_ordinal_mae": float(np.mean(np.abs(eval_labels - FIXED_INDEX))),
                "block_calibrated_ordinal_mae": float(np.mean(np.abs(eval_labels - eval_predictions))),
                "block_ordinal_mae_improvement": float(
                    np.mean(np.abs(eval_labels - FIXED_INDEX)) - np.mean(np.abs(eval_labels - eval_predictions))
                ),
                "block_exact_action_accuracy": float(np.mean(eval_labels == eval_predictions)),
                "block_oracle_positive_fraction": float(np.mean(global_positive)),
                "block_predicted_positive_fraction": float(np.mean(global_predicted_positive)),
                "block_escalation_precision": precision,
                "block_escalation_recall": recall,
                "block_escalation_f1": f1,
                "image_ordinal_mae_improvement_mean": float(
                    np.mean([row["ordinal_mae_improvement"] for row in fold_image_metrics])
                ),
                "image_escalation_auroc_mean": float(
                    np.mean([row["escalation_auroc"] for row in fold_image_metrics if not math.isnan(row["escalation_auroc"])])
                ),
                "image_escalation_ap_lift_mean": float(
                    np.mean([row["escalation_ap_lift"] for row in fold_image_metrics if not math.isnan(row["escalation_ap_lift"])])
                ),
            }
        )

    if np.any(predictions < 0):
        raise RuntimeError(f"{operator} has unfilled held-out predictions")
    image_metrics = per_image_metrics(scores, labels, predictions, image_ids)
    action_counts = Counter(int(item) for item in predictions.tolist())
    oracle_counts = Counter(int(item) for item in labels.tolist())
    summary = {
        "operator_label": operator,
        "row_count": int(labels.size),
        "image_count": len(values["names"]),
        "fold_count": len(unique_folds),
        "oracle_action_counts": {str(index): int(oracle_counts[index]) for index in range(len(ACTION_LADDER))},
        "predicted_action_counts": {str(index): int(action_counts[index]) for index in range(len(ACTION_LADDER))},
    }
    for key in (
        "fixed_mae",
        "calibrated_mae",
        "ordinal_mae_improvement",
        "exact_accuracy",
        "oracle_positive_fraction",
        "predicted_positive_fraction",
        "escalation_precision",
        "escalation_recall",
        "escalation_f1",
        "escalation_auroc",
        "escalation_ap_lift",
    ):
        summary.update(summarize_values(image_metrics, key, args.bootstrap_draws, args.bootstrap_seed))
    summary["minimum_fold_heldout_mean_action_spearman"] = float(
        np.nanmin([row["heldout_mean_action_spearman"] for row in fold_rows])
    )
    summary["maximum_fold_heldout_mean_action_monotonic_violations"] = int(
        max(row["heldout_mean_action_monotonic_violations"] for row in fold_rows)
    )
    summary["all_train_maps_non_decreasing"] = all(bool(row["train_map_non_decreasing"]) for row in fold_rows)
    summary["gate_ordinal_mae_improvement_ci95_low_gt_0p03"] = (
        summary["ordinal_mae_improvement_ci95_low"] > args.minimum_ordinal_mae_improvement_ci95_low
    )
    summary["gate_escalation_auroc_ci95_low_ge_0p80"] = (
        summary["escalation_auroc_ci95_low"] >= args.minimum_escalation_auroc_ci95_low
    )
    summary["gate_escalation_ap_lift_ci95_low_ge_0p15"] = (
        summary["escalation_ap_lift_ci95_low"] >= args.minimum_escalation_ap_lift_ci95_low
    )
    summary["gate_min_fold_spearman_ge_0p85"] = (
        summary["minimum_fold_heldout_mean_action_spearman"] >= args.minimum_fold_spearman
    )
    summary["operator_gate_pass"] = all(
        bool(summary[key])
        for key in (
            "gate_ordinal_mae_improvement_ci95_low_gt_0p03",
            "gate_escalation_auroc_ci95_low_ge_0p80",
            "gate_escalation_ap_lift_ci95_low_ge_0p15",
            "gate_min_fold_spearman_ge_0p85",
            "all_train_maps_non_decreasing",
        )
    )
    return summary, fold_rows, bin_rows


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "fold_summary": output_dir / f"{args.run_tag}_fold_summary.csv",
        "calibration_bins": output_dir / f"{args.run_tag}_calibration_bins.csv",
        "source_manifest": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite A2 outputs: " + ", ".join(existing))

    input_hashes = validate_a1_contract(args)
    data = load_block_rows(args.a1_block_rows)
    operator_summaries = []
    all_fold_rows = []
    all_bin_rows = []
    for operator in OPERATORS:
        summary, fold_rows, bin_rows = calibrate_operator(operator, data[operator], args)
        if summary["image_count"] != args.expected_image_count:
            raise ValueError(f"{operator} image count mismatch: {summary['image_count']}")
        operator_summaries.append(summary)
        all_fold_rows.extend(fold_rows)
        all_bin_rows.extend(bin_rows)

    decision = (
        "V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY"
        if all(row["operator_gate_pass"] for row in operator_summaries)
        else "V3M_A2_OOF_CALIBRATION_WEAK_STOP_NO_POLICY_REPLAY"
    )
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "phase": "v3m-A2 fold-separated OOF calibration audit",
        "primary_signal": PRIMARY_SIGNAL,
        "v3m_a2_script_sha256": sha256_file(Path(__file__)),
        "calibration": {
            "fold_count": args.fold_count,
            "bin_count": args.bin_count,
            "action_ladder": list(ACTION_LADDER),
            "median_rule": "lower_median_action_index_per_train_bin",
            "monotone_enforcement": "cumulative_max_over_train_bin_medians",
        },
        "input_sha256": input_hashes,
        "training_authorized": False,
        "canary_authorized": False,
        "locked_test_touched": False,
        "route_confirm_used_for_strategy_selection": False,
    }
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3m-A2 fold-separated OOF calibration audit",
        "run_tag": args.run_tag,
        "decision": decision,
        "next_stage_authorized": (
            "v3m-A3 frozen-policy replay only" if decision.endswith("A3_FROZEN_POLICY_REPLAY_ONLY") else None
        ),
        "metric_contract": {
            "primary_signal": PRIMARY_SIGNAL,
            "fold_count": args.fold_count,
            "bin_count": args.bin_count,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
            "minimum_ordinal_mae_improvement_ci95_low": args.minimum_ordinal_mae_improvement_ci95_low,
            "minimum_escalation_auroc_ci95_low": args.minimum_escalation_auroc_ci95_low,
            "minimum_escalation_ap_lift_ci95_low": args.minimum_escalation_ap_lift_ci95_low,
            "minimum_fold_spearman": args.minimum_fold_spearman,
        },
        "operator_summaries": operator_summaries,
        "source_manifest": source_manifest,
        "training_authorized": False,
        "canary_authorized": False,
        "locked_test_touched": False,
        "route_confirm_used_for_strategy_selection": False,
    }
    write_rows(outputs["fold_summary"], all_fold_rows)
    write_rows(outputs["calibration_bins"], all_bin_rows)
    write_json(outputs["source_manifest"], source_manifest)
    write_json(outputs["summary"], summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-block-rows", required=True)
    parser.add_argument("--a1-summary", required=True)
    parser.add_argument("--a1-signal-summary", required=True)
    parser.add_argument("--a1-source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="v3m_a2")
    parser.add_argument("--expected-a1-block-rows-sha256", required=True)
    parser.add_argument("--expected-a1-block-rows-line-count", type=int, required=True)
    parser.add_argument("--expected-a1-summary-sha256", required=True)
    parser.add_argument("--expected-a1-signal-summary-sha256", required=True)
    parser.add_argument("--expected-a1-source-manifest-sha256", required=True)
    parser.add_argument("--expected-image-count", type=int, default=1200)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--bin-count", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--minimum-ordinal-mae-improvement-ci95-low", type=float, default=0.03)
    parser.add_argument("--minimum-escalation-auroc-ci95-low", type=float, default=0.80)
    parser.add_argument("--minimum-escalation-ap-lift-ci95-low", type=float, default=0.15)
    parser.add_argument("--minimum-fold-spearman", type=float, default=0.85)
    return parser


def main():
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps({"decision": summary["decision"], "run_tag": args.run_tag}, sort_keys=True))


if __name__ == "__main__":
    main()
