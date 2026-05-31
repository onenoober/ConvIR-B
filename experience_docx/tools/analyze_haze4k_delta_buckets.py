import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def correlation(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def summarize(rows):
    deltas = [row["delta_psnr"] for row in rows]
    ssim_deltas = [row["delta_ssim"] for row in rows]
    return {
        "count": len(rows),
        "mean_delta_psnr": statistics.mean(deltas),
        "median_delta_psnr": statistics.median(deltas),
        "p5_delta_psnr": percentile(deltas, 5),
        "p95_delta_psnr": percentile(deltas, 95),
        "positive_delta_count": sum(delta > 0 for delta in deltas),
        "positive_delta_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "regression_delta_le_-0.05": sum(delta <= -0.05 for delta in deltas),
        "regression_delta_le_-0.10": sum(delta <= -0.10 for delta in deltas),
        "regression_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "gain_delta_ge_0.05": sum(delta >= 0.05 for delta in deltas),
        "gain_delta_ge_0.10": sum(delta >= 0.10 for delta in deltas),
        "mean_delta_ssim": statistics.mean(ssim_deltas),
        "median_delta_ssim": statistics.median(ssim_deltas),
        "worst_10_mean_delta_psnr": statistics.mean(sorted(deltas)[:10]),
        "best_10_mean_delta_psnr": statistics.mean(sorted(deltas)[-10:]),
    }


def parse_rows(csv_path, candidate_name):
    candidate_psnr_key = f"{candidate_name}_psnr"
    candidate_ssim_key = f"{candidate_name}_ssim"
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            stem = Path(raw["name"]).stem
            filename_params = []
            for part in stem.split("_")[1:]:
                try:
                    filename_params.append(float(part))
                except ValueError:
                    pass
            rows.append(
                {
                    "name": raw["name"],
                    "original_psnr": float(raw["original_psnr"]),
                    "candidate_psnr": float(raw[candidate_psnr_key]),
                    "delta_psnr": float(raw["delta_psnr"]),
                    "original_ssim": float(raw["original_ssim"]),
                    "candidate_ssim": float(raw[candidate_ssim_key]),
                    "delta_ssim": float(raw["delta_ssim"]),
                    "filename_param_1": filename_params[0] if len(filename_params) > 0 else None,
                    "filename_param_2": filename_params[1] if len(filename_params) > 1 else None,
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Per-image checkpoint comparison CSV.")
    parser.add_argument("--candidate_name", default="fam2_modres")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = parse_rows(Path(args.csv), args.candidate_name)
    by_original = sorted(rows, key=lambda row: row["original_psnr"])
    count = len(by_original)
    hard = by_original[: count // 4]
    medium = by_original[count // 4 : 3 * count // 4]
    easy = by_original[3 * count // 4 :]

    summary = {
        "source_csv": args.csv,
        "candidate_name": args.candidate_name,
        "overall": summarize(rows),
        "difficulty_buckets_by_original_psnr": {
            "hard_bottom_25pct": {
                "original_psnr_range": [hard[0]["original_psnr"], hard[-1]["original_psnr"]],
                **summarize(hard),
            },
            "medium_middle_50pct": {
                "original_psnr_range": [medium[0]["original_psnr"], medium[-1]["original_psnr"]],
                **summarize(medium),
            },
            "easy_top_25pct": {
                "original_psnr_range": [easy[0]["original_psnr"], easy[-1]["original_psnr"]],
                **summarize(easy),
            },
        },
        "filename_parameter_correlations_inferred": {
            "note": "Haze4K filenames look like id_param1_param2.png; this file records correlation only and does not assume the parameter meaning.",
            "param_1_vs_original_psnr": correlation(
                [row["filename_param_1"] for row in rows], [row["original_psnr"] for row in rows]
            ),
            "param_1_vs_delta_psnr": correlation(
                [row["filename_param_1"] for row in rows], [row["delta_psnr"] for row in rows]
            ),
            "param_2_vs_original_psnr": correlation(
                [row["filename_param_2"] for row in rows], [row["original_psnr"] for row in rows]
            ),
            "param_2_vs_delta_psnr": correlation(
                [row["filename_param_2"] for row in rows], [row["delta_psnr"] for row in rows]
            ),
        },
        "worst_regressions": sorted(rows, key=lambda row: row["delta_psnr"])[:20],
        "best_gains": sorted(rows, key=lambda row: row["delta_psnr"], reverse=True)[:20],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
