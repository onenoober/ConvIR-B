#!/usr/bin/env python3
"""v3m-A3 diagnostic-only failure decomposition.

This script reads the already completed v3m-A3 frozen policy replay rows and
the v3m-A2 calibration-bin table.  It does not train, tune thresholds, rerun
inference, replay a new policy, or authorize any next stage.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


DECISION = "V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION"
OPERATORS = ("D_ref", "D_rep")
ACTION_LABELS = ("0", "0p125", "0p25", "0p5", "1")
ACTION_VALUES = (0.0, 0.125, 0.25, 0.5, 1.0)


def parse_float(value):
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    if text.lower() == "inf":
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    return float(text)


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
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def quantile(values, q):
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for _, y in pairs))
    if x_den == 0.0 or y_den == 0.0:
        return float("nan")
    return float(num / (x_den * y_den))


def summarize_rows(rows):
    lifts = [row["policy_lift_vs_fixed"] for row in rows]
    oracle_lifts = [row["oracle_block16_lift_vs_fixed"] for row in rows]
    fixed = [row["fixed_psnr_delta"] for row in rows]
    return {
        "count": len(rows),
        "mean_policy_lift_vs_fixed": mean(lifts),
        "p10_policy_lift_vs_fixed": quantile(lifts, 0.10),
        "p05_policy_lift_vs_fixed": quantile(lifts, 0.05),
        "worst_policy_lift_vs_fixed": min(lifts) if rows else float("nan"),
        "severe_le_minus_0p2_count": sum(1 for row in rows if row["policy_lift_vs_fixed"] <= -0.2),
        "hard_le_minus_0p5_count": sum(1 for row in rows if row["policy_lift_vs_fixed"] <= -0.5),
        "negative_count": sum(1 for row in rows if row["policy_lift_vs_fixed"] < 0.0),
        "mean_fixed_psnr_delta": mean(fixed),
        "mean_oracle_block16_lift_vs_fixed": mean(oracle_lifts),
        "mean_retention_vs_oracle": (
            mean(lifts) / mean(oracle_lifts) if math.isfinite(mean(oracle_lifts)) and mean(oracle_lifts) > 0 else float("nan")
        ),
        "mean_selected_alpha": mean(row["selected_alpha_mean"] for row in rows),
        "mean_frac_alpha_1": mean(row["frac_alpha_1"] for row in rows),
        "mean_frac_alpha_ge_0p5": mean(row["frac_alpha_ge_0p5"] for row in rows),
    }


def load_a3_rows(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts = [int(row[f"selected_action_{idx}_count"]) for idx in range(5)]
            total = sum(counts)
            if total <= 0:
                raise ValueError(f"non-positive block count for {row.get('operator_label')} {row.get('name')}")
            parsed = {
                "operator_label": row["operator_label"],
                "seed": int(row["seed"]),
                "split": row["split"],
                "fold": int(row["fold"]),
                "index": int(row["index"]),
                "name": row["name"],
                "policy": row["policy"],
                "base_psnr": parse_float(row["base_psnr"]),
                "fixed_psnr_delta": parse_float(row["fixed_psnr_delta"]),
                "reference_fixed_psnr_delta": parse_float(row["reference_fixed_psnr_delta"]),
                "oracle_block16_psnr_delta": parse_float(row["oracle_block16_psnr_delta"]),
                "policy_psnr_delta": parse_float(row["policy_psnr_delta"]),
                "policy_lift_vs_fixed": parse_float(row["policy_lift_vs_fixed"]),
                "oracle_block16_lift_vs_fixed": parse_float(row["oracle_block16_lift_vs_fixed"]),
                "fixed_replay_abs_diff_db": parse_float(row["fixed_replay_abs_diff_db"]),
                "selected_alpha_mean": parse_float(row["selected_alpha_mean"]),
                "selected_alpha_p10": parse_float(row["selected_alpha_p10"]),
                "selected_alpha_p50": parse_float(row["selected_alpha_p50"]),
                "selected_alpha_p90": parse_float(row["selected_alpha_p90"]),
                "total_blocks": total,
                "frac_alpha_0": counts[0] / total,
                "frac_alpha_0p125": counts[1] / total,
                "frac_alpha_0p25": counts[2] / total,
                "frac_alpha_0p5": counts[3] / total,
                "frac_alpha_1": counts[4] / total,
                "frac_alpha_ge_0p5": (counts[3] + counts[4]) / total,
                "frac_alpha_ge_0p25": (counts[2] + counts[3] + counts[4]) / total,
            }
            rows.append(parsed)
    if not rows:
        raise ValueError(f"no A3 rows found: {path}")
    return rows


def build_decile_rows(rows, metric_name):
    out = []
    for operator in OPERATORS:
        subset = [row for row in rows if row["operator_label"] == operator]
        ordered = sorted(subset, key=lambda row: row[metric_name])
        for decile in range(10):
            start = decile * len(ordered) // 10
            end = (decile + 1) * len(ordered) // 10
            bucket = ordered[start:end]
            summary = summarize_rows(bucket)
            summary.update(
                {
                    "operator_label": operator,
                    "sort_metric": metric_name,
                    "decile": decile,
                    "metric_min": min(row[metric_name] for row in bucket),
                    "metric_max": max(row[metric_name] for row in bucket),
                }
            )
            out.append(summary)
    return out


def build_operator_summary(rows):
    out = []
    for operator in OPERATORS:
        subset = [row for row in rows if row["operator_label"] == operator]
        summary = summarize_rows(subset)
        summary.update(
            {
                "operator_label": operator,
                "corr_lift_selected_alpha_mean": pearson(
                    [row["selected_alpha_mean"] for row in subset],
                    [row["policy_lift_vs_fixed"] for row in subset],
                ),
                "corr_lift_frac_alpha_1": pearson(
                    [row["frac_alpha_1"] for row in subset],
                    [row["policy_lift_vs_fixed"] for row in subset],
                ),
                "corr_lift_frac_alpha_ge_0p5": pearson(
                    [row["frac_alpha_ge_0p5"] for row in subset],
                    [row["policy_lift_vs_fixed"] for row in subset],
                ),
                "corr_lift_oracle_lift": pearson(
                    [row["oracle_block16_lift_vs_fixed"] for row in subset],
                    [row["policy_lift_vs_fixed"] for row in subset],
                ),
                "corr_severe_score_selected_alpha_mean": pearson(
                    [row["selected_alpha_mean"] for row in subset],
                    [1.0 if row["policy_lift_vs_fixed"] <= -0.2 else 0.0 for row in subset],
                ),
                "severe_subset_mean_oracle_lift": mean(
                    row["oracle_block16_lift_vs_fixed"] for row in subset if row["policy_lift_vs_fixed"] <= -0.2
                ),
                "severe_subset_mean_selected_alpha": mean(
                    row["selected_alpha_mean"] for row in subset if row["policy_lift_vs_fixed"] <= -0.2
                ),
                "severe_subset_mean_frac_alpha_1": mean(
                    row["frac_alpha_1"] for row in subset if row["policy_lift_vs_fixed"] <= -0.2
                ),
                "severe_subset_mean_frac_alpha_ge_0p5": mean(
                    row["frac_alpha_ge_0p5"] for row in subset if row["policy_lift_vs_fixed"] <= -0.2
                ),
            }
        )
        out.append(summary)
    return out


def build_cross_operator_summary(rows):
    by_operator = {operator: {row["name"]: row for row in rows if row["operator_label"] == operator} for operator in OPERATORS}
    shared_names = sorted(set(by_operator["D_ref"]).intersection(by_operator["D_rep"]))
    severe_ref = {name for name, row in by_operator["D_ref"].items() if row["policy_lift_vs_fixed"] <= -0.2}
    severe_rep = {name for name, row in by_operator["D_rep"].items() if row["policy_lift_vs_fixed"] <= -0.2}
    hard_ref = {name for name, row in by_operator["D_ref"].items() if row["policy_lift_vs_fixed"] <= -0.5}
    hard_rep = {name for name, row in by_operator["D_rep"].items() if row["policy_lift_vs_fixed"] <= -0.5}

    def jaccard(a_set, b_set):
        union = len(a_set.union(b_set))
        return len(a_set.intersection(b_set)) / union if union else float("nan")

    return {
        "shared_name_count": len(shared_names),
        "severe_ref_count": len(severe_ref),
        "severe_rep_count": len(severe_rep),
        "severe_overlap_count": len(severe_ref.intersection(severe_rep)),
        "severe_union_count": len(severe_ref.union(severe_rep)),
        "severe_jaccard": jaccard(severe_ref, severe_rep),
        "hard_ref_count": len(hard_ref),
        "hard_rep_count": len(hard_rep),
        "hard_overlap_count": len(hard_ref.intersection(hard_rep)),
        "hard_union_count": len(hard_ref.union(hard_rep)),
        "hard_jaccard": jaccard(hard_ref, hard_rep),
        "corr_ref_rep_policy_lift": pearson(
            [by_operator["D_ref"][name]["policy_lift_vs_fixed"] for name in shared_names],
            [by_operator["D_rep"][name]["policy_lift_vs_fixed"] for name in shared_names],
        ),
        "corr_ref_rep_selected_alpha_mean": pearson(
            [by_operator["D_ref"][name]["selected_alpha_mean"] for name in shared_names],
            [by_operator["D_rep"][name]["selected_alpha_mean"] for name in shared_names],
        ),
        "corr_ref_rep_oracle_lift": pearson(
            [by_operator["D_ref"][name]["oracle_block16_lift_vs_fixed"] for name in shared_names],
            [by_operator["D_rep"][name]["oracle_block16_lift_vs_fixed"] for name in shared_names],
        ),
    }


def load_calibration_confusion(path):
    by_operator_action = defaultdict(lambda: defaultdict(int))
    by_operator_fold_action = defaultdict(lambda: defaultdict(int))
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator = row["operator_label"]
            if operator not in OPERATORS:
                continue
            fold = int(row["holdout_fold"])
            selected_action = int(row["monotone_action_index"])
            for oracle_action, label in enumerate(ACTION_LABELS):
                heldout_count = int(row[f"heldout_count_alpha_{label}"])
                train_count = int(row[f"train_count_alpha_{label}"])
                key = (operator, selected_action)
                by_operator_action[key][f"heldout_oracle_action_{oracle_action}_count"] += heldout_count
                by_operator_action[key][f"train_oracle_action_{oracle_action}_count"] += train_count
                by_operator_action[key]["heldout_total"] += heldout_count
                by_operator_action[key]["train_total"] += train_count
                fold_key = (operator, fold, selected_action)
                by_operator_fold_action[fold_key][f"heldout_oracle_action_{oracle_action}_count"] += heldout_count
                by_operator_fold_action[fold_key]["heldout_total"] += heldout_count

    rows = []
    for (operator, selected_action), counts in sorted(by_operator_action.items()):
        total = counts["heldout_total"]
        train_total = counts["train_total"]
        row = {
            "operator_label": operator,
            "selected_action_index": selected_action,
            "selected_alpha": ACTION_VALUES[selected_action],
            "heldout_total_blocks": total,
            "train_total_blocks": train_total,
        }
        heldout_match = 0
        heldout_over = 0
        heldout_under = 0
        train_match = 0
        train_over = 0
        train_under = 0
        for oracle_action in range(5):
            heldout_count = counts[f"heldout_oracle_action_{oracle_action}_count"]
            train_count = counts[f"train_oracle_action_{oracle_action}_count"]
            row[f"heldout_oracle_action_{oracle_action}_count"] = heldout_count
            row[f"train_oracle_action_{oracle_action}_count"] = train_count
            if selected_action == oracle_action:
                heldout_match += heldout_count
                train_match += train_count
            elif selected_action > oracle_action:
                heldout_over += heldout_count
                train_over += train_count
            else:
                heldout_under += heldout_count
                train_under += train_count
        row.update(
            {
                "heldout_match_rate": heldout_match / total if total else float("nan"),
                "heldout_over_escalate_rate": heldout_over / total if total else float("nan"),
                "heldout_under_escalate_rate": heldout_under / total if total else float("nan"),
                "train_match_rate": train_match / train_total if train_total else float("nan"),
                "train_over_escalate_rate": train_over / train_total if train_total else float("nan"),
                "train_under_escalate_rate": train_under / train_total if train_total else float("nan"),
            }
        )
        rows.append(row)
    return rows


def write_closeout(path, summary):
    operator_lines = []
    for row in summary["operator_summaries"]:
        operator_lines.append(
            "| `{operator_label}` | `{mean_policy_lift_vs_fixed:+.7f}` | `{p10_policy_lift_vs_fixed:+.7f}` | "
            "`{severe_le_minus_0p2_count}` | `{hard_le_minus_0p5_count}` | `{mean_selected_alpha:.7f}` | "
            "`{mean_frac_alpha_1:.7f}` | `{severe_subset_mean_oracle_lift:+.7f}` |".format(**row)
        )

    cross = summary["cross_operator_summary"]
    text = f"""# v3m A3 Failure Decomposition

Decision: `{DECISION}`.

This is a diagnostic-only post-fail audit over already completed cloud A3/A2
rows. It does not train, tune thresholds, rerun inference, replay a new policy,
use route-confirm, touch canary, or touch locked test.

## Operator Summary

| Operator | Mean lift | p10 lift | Severe | Hard | Mean selected alpha | Mean alpha=1 fraction | Severe-subset oracle lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(operator_lines)}

## Cross-Operator Tail Stability

- shared names: `{cross['shared_name_count']}`;
- severe overlap: `{cross['severe_overlap_count']}` of union `{cross['severe_union_count']}` (Jaccard `{cross['severe_jaccard']:.7f}`);
- hard overlap: `{cross['hard_overlap_count']}` of union `{cross['hard_union_count']}` (Jaccard `{cross['hard_jaccard']:.7f}`);
- policy-lift correlation: `{cross['corr_ref_rep_policy_lift']:.7f}`;
- selected-alpha correlation: `{cross['corr_ref_rep_selected_alpha_mean']:.7f}`;
- oracle-lift correlation: `{cross['corr_ref_rep_oracle_lift']:.7f}`.

## Interpretation

The failure remains a safe-utility calibration problem. A2 label calibration is
strong enough to create positive mean image PSNR, but A3's selected action mix
is too aggressive at image level and produces stable tail regressions across
the two frozen operators. Severe images still have positive block16-oracle
headroom on average, so the issue is not absence of oracle value; it is
incorrect allocation of aggressive local escalation under the current
deployable signal.

No route-confirm audit, canary, locked-test access, controller training,
learned ranker, physics/proxy continuation, or policy deployment is authorized
by this diagnostic.
"""
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evid", required=True, type=Path)
    parser.add_argument("--a3-rows", type=Path)
    parser.add_argument("--a2-bins", type=Path)
    args = parser.parse_args()

    evid = args.evid
    a3_rows_path = args.a3_rows or evid / "v3m_a3_policy_replay_rows_cloud_only.csv"
    a2_bins_path = args.a2_bins or evid / "v3m_a2_calibration_bins.csv"

    a3_rows = load_a3_rows(a3_rows_path)
    if sorted({row["operator_label"] for row in a3_rows}) != list(OPERATORS):
        raise ValueError("A3 rows do not contain exactly the expected operators")
    for operator in OPERATORS:
        count = sum(1 for row in a3_rows if row["operator_label"] == operator)
        if count != 1200:
            raise ValueError(f"expected 1200 A3 rows for {operator}, found {count}")

    operator_summaries = build_operator_summary(a3_rows)
    decile_rows = []
    for metric in ("selected_alpha_mean", "frac_alpha_1", "frac_alpha_ge_0p5", "oracle_block16_lift_vs_fixed"):
        decile_rows.extend(build_decile_rows(a3_rows, metric))
    confusion_rows = load_calibration_confusion(a2_bins_path)
    cross_operator_summary = build_cross_operator_summary(a3_rows)

    summary = {
        "decision": DECISION,
        "diagnostic_only": True,
        "training_used": False,
        "threshold_search_used": False,
        "policy_replay_rerun": False,
        "route_confirm_used": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "next_stage_authorized": "NONE",
        "inputs": {
            "a3_rows": str(a3_rows_path),
            "a3_rows_sha256": sha256_file(a3_rows_path),
            "a2_bins": str(a2_bins_path),
            "a2_bins_sha256": sha256_file(a2_bins_path),
        },
        "operator_summaries": operator_summaries,
        "cross_operator_summary": cross_operator_summary,
    }

    write_rows(evid / "v3m_a3_failure_operator_summary.csv", operator_summaries)
    write_rows(evid / "v3m_a3_failure_alpha_deciles.csv", decile_rows)
    write_rows(evid / "v3m_a3_failure_calibration_action_confusion.csv", confusion_rows)
    write_json(evid / "v3m_a3_failure_decomposition_summary.json", summary)
    write_closeout(evid / "v3m_a3_failure_decomposition_closeout.md", summary)
    print("V3M_A3_FAILURE_DECOMPOSITION_OK")


if __name__ == "__main__":
    main()
