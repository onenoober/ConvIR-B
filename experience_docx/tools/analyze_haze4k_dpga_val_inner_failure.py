#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return None
    return statistics.mean(values)


def percentile(values, pct):
    values = sorted(value for value in values if value is not None and not math.isnan(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def parse_haze_params(name):
    stem = Path(name).stem
    values = []
    for item in stem.split("_")[1:]:
        try:
            values.append(float(item))
        except ValueError:
            pass
    return {
        "haze_param_1": values[0] if len(values) > 0 else None,
        "haze_param_2": values[1] if len(values) > 1 else None,
    }


def resolve_candidate_keys(fieldnames, candidate_name):
    fieldnames = set(fieldnames or [])
    if candidate_name:
        candidate_psnr_key = f"{candidate_name}_psnr"
        candidate_ssim_key = f"{candidate_name}_ssim"
        if candidate_psnr_key in fieldnames and candidate_ssim_key in fieldnames:
            return candidate_psnr_key, candidate_ssim_key

    candidates = []
    for key in fieldnames:
        if key in ("original_psnr", "delta_psnr") or not key.endswith("_psnr"):
            continue
        prefix = key[: -len("_psnr")]
        if f"{prefix}_ssim" in fieldnames:
            candidates.append(prefix)
    if len(candidates) == 1:
        prefix = candidates[0]
        return f"{prefix}_psnr", f"{prefix}_ssim"
    if not candidates:
        raise KeyError("No candidate *_psnr/*_ssim columns found in per-image CSV")
    raise KeyError(f"Multiple candidate prefixes found; pass an explicit candidate name: {candidates}")


def load_per_image(path, candidate_name):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        candidate_psnr_key, candidate_ssim_key = resolve_candidate_keys(
            reader.fieldnames,
            candidate_name,
        )
        for row in reader:
            parsed = {
                "name": row["name"],
                "original_psnr": parse_float(row["original_psnr"]),
                "candidate_psnr": parse_float(row[candidate_psnr_key]),
                "delta_psnr": parse_float(row["delta_psnr"]),
                "original_ssim": parse_float(row["original_ssim"]),
                "candidate_ssim": parse_float(row[candidate_ssim_key]),
                "delta_ssim": parse_float(row["delta_ssim"]),
            }
            parsed.update(parse_haze_params(row["name"]))
            rows.append(parsed)
    return rows


def group_summary(rows):
    if not rows:
        return {"count": 0}
    deltas = [row["delta_psnr"] for row in rows]
    return {
        "count": len(rows),
        "original_psnr_mean": mean(row["original_psnr"] for row in rows),
        "delta_psnr_mean": mean(deltas),
        "delta_psnr_median": statistics.median(deltas),
        "delta_psnr_p5": percentile(deltas, 5),
        "delta_psnr_p95": percentile(deltas, 95),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "strong_regression_count_delta_le_-0.05": sum(delta <= -0.05 for delta in deltas),
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "haze_param_1_mean": mean(row["haze_param_1"] for row in rows),
        "haze_param_2_mean": mean(row["haze_param_2"] for row in rows),
    }


def summarize(rows):
    ordered = sorted(rows, key=lambda row: row["original_psnr"])
    count = len(ordered)
    bucket = max(1, count // 4)
    strong_cut = percentile([row["original_psnr"] for row in ordered], 75)
    groups = {
        "all": ordered,
        "hard_bottom25": ordered[:bucket],
        "middle50": ordered[bucket : count - bucket],
        "easy_top25": ordered[-bucket:],
        "strong_reference_top25": [row for row in ordered if row["original_psnr"] >= strong_cut],
        "regressions_delta_lt_0": [row for row in ordered if row["delta_psnr"] < 0],
        "worst_regressions_delta_le_-0.20": [row for row in ordered if row["delta_psnr"] <= -0.20],
    }
    return {
        "strong_reference_cut_psnr": strong_cut,
        "groups": {name: group_summary(group_rows) for name, group_rows in groups.items()},
        "worst20": sorted(ordered, key=lambda row: row["delta_psnr"])[:20],
        "hard_bottom25_worst20": sorted(groups["hard_bottom25"], key=lambda row: row["delta_psnr"])[:20],
    }


def write_group_csv(path, rows):
    fieldnames = [
        "group",
        "name",
        "original_psnr",
        "candidate_psnr",
        "delta_psnr",
        "original_ssim",
        "candidate_ssim",
        "delta_ssim",
        "haze_param_1",
        "haze_param_2",
    ]
    ordered = sorted(rows, key=lambda row: row["original_psnr"])
    bucket = max(1, len(ordered) // 4)
    groups = {
        "hard_bottom25": ordered[:bucket],
        "easy_top25": ordered[-bucket:],
        "worst20": sorted(ordered, key=lambda row: row["delta_psnr"])[:20],
    }
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group_rows in groups.items():
            for row in group_rows:
                writer.writerow({"group": group_name, **row})


def write_markdown(path, payload):
    best = payload["best"]["summary"]
    final = payload["final"]["summary"]
    gate = payload["gate"]
    lines = [
        "# DPGA val_inner Failure Analysis",
        "",
        f"Gate pass: `{str(gate.get('pass')).lower()}`",
        f"Locked test allowed: `{str(gate.get('locked_test_allowed')).lower()}`",
        "",
        "## Best Checkpoint",
        "",
        "| group | count | mean delta | positive ratio | worst <= -0.20 | haze p1 | haze p2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in (
        "all",
        "hard_bottom25",
        "middle50",
        "easy_top25",
        "strong_reference_top25",
        "worst_regressions_delta_le_-0.20",
    ):
        item = best["groups"][group]
        lines.append(
            "| {group} | {count} | {delta:.6f} | {pos:.3f} | {worst} | {p1:.4f} | {p2:.4f} |".format(
                group=group,
                count=item["count"],
                delta=item.get("delta_psnr_mean") or 0.0,
                pos=item.get("positive_ratio") or 0.0,
                worst=item.get("worst_regression_count_delta_le_-0.20") or 0,
                p1=item.get("haze_param_1_mean") or 0.0,
                p2=item.get("haze_param_2_mean") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Final Checkpoint",
            "",
            f"- Mean delta: `{final['groups']['all']['delta_psnr_mean']:.6f}`",
            f"- Hard bottom25 delta: `{final['groups']['hard_bottom25']['delta_psnr_mean']:.6f}`",
            f"- Worst `<= -0.20 dB`: `{final['groups']['all']['worst_regression_count_delta_le_-0.20']}`",
            "",
            "## Decision Hint",
            "",
            payload["decision_hint"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_per_image_csv", required=True)
    parser.add_argument("--final_per_image_csv", required=True)
    parser.add_argument("--gate_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_group_csv", required=True)
    args = parser.parse_args()

    gate = json.loads(Path(args.gate_json).read_text(encoding="utf-8"))
    best_rows = load_per_image(args.best_per_image_csv, "dpga_v1_1_best")
    final_rows = load_per_image(args.final_per_image_csv, "dpga_v1_1_final")
    best_summary = summarize(best_rows)
    final_summary = summarize(final_rows)

    hard_delta = best_summary["groups"]["hard_bottom25"]["delta_psnr_mean"]
    mean_delta = best_summary["groups"]["all"]["delta_psnr_mean"]
    worst_count = best_summary["groups"]["all"]["worst_regression_count_delta_le_-0.20"]
    if gate.get("pass"):
        hint = "Gate passed; this analysis is archival only."
    elif mean_delta >= 0.03 and hard_delta < 0.03 and worst_count <= 12:
        hint = (
            "Mean gain and tail safety are acceptable, but hard-bottom gain is short. "
            "Prefer a small hard-gain follow-up: keep shallow-only, raise scale modestly, "
            "and reduce global anchor pressure before adding architecture."
        )
    else:
        hint = "Gate failed on multiple axes; do not launch a higher-scale follow-up without a new diagnostic."

    payload = {
        "gate": gate,
        "best": {"source": args.best_per_image_csv, "summary": best_summary},
        "final": {"source": args.final_per_image_csv, "summary": final_summary},
        "decision_hint": hint,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_group_csv(Path(args.output_group_csv), best_rows)
    write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
