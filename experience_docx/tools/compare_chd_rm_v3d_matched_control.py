#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


DECISION_PAUSE = "V3D_PAUSE_D7C_SAFER_BUT_NOT_MATCHED_CONTROL_UTILITY_NO_20EPOCH_NO_V4"
DECISION_CONTINUE = "V3D_MATCHED_CONTROL_PASS_WRITE_NEXT_DECISION"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values, pct):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 10),
        "worst": min(values),
        "best": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d7c_summary", required=True)
    parser.add_argument("--d7c_per_image", required=True)
    parser.add_argument("--control_summary", required=True)
    parser.add_argument("--control_per_image", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    d7c_summary = read_json(args.d7c_summary)
    control_summary = read_json(args.control_summary)
    d7c_rows = {row["name"]: row for row in read_csv(args.d7c_per_image)}
    control_rows = {row["name"]: row for row in read_csv(args.control_per_image)}
    names = sorted(set(d7c_rows) & set(control_rows))
    if len(names) != int(d7c_summary["sample_count"]) or len(names) != int(control_summary["sample_count"]):
        raise ValueError(
            f"Sample mismatch: common={len(names)} "
            f"d7c={d7c_summary['sample_count']} control={control_summary['sample_count']}"
        )

    paired = []
    d7c_minus_control = []
    for name in names:
        d7c_delta = float(d7c_rows[name]["psnr_delta"])
        control_delta = float(control_rows[name]["psnr_delta"])
        diff = d7c_delta - control_delta
        d7c_minus_control.append(diff)
        paired.append(
            {
                "name": name,
                "d7c_psnr_delta": d7c_delta,
                "control_psnr_delta": control_delta,
                "d7c_minus_control_psnr_delta": diff,
                "d7c_better": diff > 0,
                "d7c_safer_at_le_0p2": d7c_delta > -0.2 and control_delta <= -0.2,
                "control_safer_at_le_0p2": control_delta > -0.2 and d7c_delta <= -0.2,
            }
        )

    d7c_mean = float(d7c_summary["mean_psnr_delta"])
    control_mean = float(control_summary["mean_psnr_delta"])
    d7c_regress_0p2 = int(d7c_summary["regression_le_0p2_count"])
    control_regress_0p2 = int(control_summary["regression_le_0p2_count"])
    d7c_utility_beats_control = d7c_mean > control_mean + 0.01
    d7c_tail_safer = d7c_regress_0p2 < control_regress_0p2
    continue_allowed = d7c_utility_beats_control
    decision = DECISION_CONTINUE if continue_allowed else DECISION_PAUSE

    comparison = {
        "decision": decision,
        "continue_allowed": continue_allowed,
        "locked_test_authorized": False,
        "sample_count": len(names),
        "d7c": {
            "mean_psnr_delta": d7c_mean,
            "median_psnr_delta": float(d7c_summary["median_psnr_delta"]),
            "p10_psnr_delta": float(d7c_summary["p10_psnr_delta"]),
            "worst_psnr_delta": float(d7c_summary["worst_psnr_delta"]),
            "positive_psnr_ratio": float(d7c_summary["positive_psnr_ratio"]),
            "regression_le_0p2_count": d7c_regress_0p2,
            "regression_le_1p0_count": int(d7c_summary["regression_le_1p0_count"]),
            "max_output_max_abs_diff": float(d7c_summary["max_output_max_abs_diff"]),
        },
        "control": {
            "mean_psnr_delta": control_mean,
            "median_psnr_delta": float(control_summary["median_psnr_delta"]),
            "p10_psnr_delta": float(control_summary["p10_psnr_delta"]),
            "worst_psnr_delta": float(control_summary["worst_psnr_delta"]),
            "positive_psnr_ratio": float(control_summary["positive_psnr_ratio"]),
            "regression_le_0p2_count": control_regress_0p2,
            "regression_le_1p0_count": int(control_summary["regression_le_1p0_count"]),
            "max_output_max_abs_diff": float(control_summary["max_output_max_abs_diff"]),
        },
        "paired_d7c_minus_control": summarize(d7c_minus_control),
        "d7c_better_image_ratio": sum(1 for value in d7c_minus_control if value > 0) / len(d7c_minus_control),
        "d7c_utility_beats_control": d7c_utility_beats_control,
        "d7c_tail_safer": d7c_tail_safer,
        "regression_le_0p2_reduction_vs_control": control_regress_0p2 - d7c_regress_0p2,
        "rationale": (
            "D7c is safer in mild-tail regression count, but it does not beat "
            "the matched FAM2 modres control on mean PSNR delta. The route "
            "therefore cannot claim matched-budget utility and must pause "
            "before 20-epoch, neighbor, v4, or locked-test steps."
        ),
        "next_action": "Archive v3d evidence to GitHub main; no further RARM training is authorized from v3d.",
    }
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "v3d_matched_control_comparison.csv", paired)
    write_json(output_dir / "v3d_matched_control_comparison.json", comparison)
    write_json(
        output_dir / "v3d_final_closeout.json",
        {
            "decision": decision,
            "continue_allowed": continue_allowed,
            "locked_test_authorized": False,
            "next_action": comparison["next_action"],
        },
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if not continue_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
