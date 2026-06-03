import argparse
import csv
import json
import math
from pathlib import Path


def to_float(value, default=None):
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(rows, keep_mask):
    adjusted = []
    for row, keep in zip(rows, keep_mask):
        output_gain = row["output_gain"] if keep else 0.0
        final_l1_num = row["final_l1_num"] if keep else row["initial_l1_num"]
        adjusted.append({**row, "output_gain": output_gain, "final_l1_num": final_l1_num, "kept": keep})

    den = sum(row["weighted_den"] for row in adjusted)
    initial = sum(row["initial_l1_num"] for row in adjusted) / max(den, 1e-12)
    final = sum(row["final_l1_num"] for row in adjusted) / max(den, 1e-12)
    ordered = sorted(adjusted, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    easy = ordered[3 * len(ordered) // 4 :]
    positive = [row for row in adjusted if row["oracle_gain"] > 1e-6]
    oracle_sum = sum(row["oracle_gain"] for row in positive)
    strong_cut = percentile([row["anchor_psnr"] for row in adjusted], 75)
    strong = [row for row in adjusted if row["anchor_psnr"] >= strong_cut]
    return {
        "count": len(adjusted),
        "open_count": sum(row["P_benefit"] >= 0.5 for row in adjusted),
        "keep_count": sum(1 for keep in keep_mask if keep),
        "open_keep_count": sum(1 for row, keep in zip(adjusted, keep_mask) if keep and row["P_benefit"] >= 0.5),
        "initial_weighted_delta_l1": initial,
        "projection_weighted_delta_l1": final,
        "weighted_delta_l1_drop": (initial - final) / max(initial, 1e-12),
        "mean_output_gain": mean([row["output_gain"] for row in adjusted]),
        "mean_oracle_gain": mean([row["oracle_gain"] for row in adjusted]),
        "oracle_recovery": sum(row["output_gain"] for row in positive) / max(oracle_sum, 1e-12),
        "hard_bottom25_output_gain": mean([row["output_gain"] for row in hard]),
        "hard_bottom25_oracle_gain": mean([row["oracle_gain"] for row in hard]),
        "easy_top25_output_gain": mean([row["output_gain"] for row in easy]),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row["output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["output_gain"] <= -0.20 for row in adjusted),
    }


def threshold_values(values):
    values = sorted(set(values))
    if not values:
        return []
    if len(values) == 1:
        value = values[0]
        return [value - 1e-9, value, value + 1e-9]
    picks = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    thresholds = {percentile(values, pct) for pct in picks}
    span = values[-1] - values[0]
    eps = max(abs(span), 1.0) * 1e-9
    thresholds.add(values[0] - eps)
    thresholds.add(values[-1] + eps)
    return sorted(thresholds)


def load_rows(path, split):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            parsed = {
                "low_size": int(float(row["low_size"])),
                "K": int(float(row["K"])),
                "mapper": row["mapper"],
                "family": row["family"],
                "split": row["split"],
                "index": int(float(row["index"])),
                "name": row["name"],
            }
            numeric_keys = [
                "anchor_psnr",
                "output_gain",
                "oracle_gain",
                "P_benefit",
                "weighted_den",
                "initial_l1_num",
                "final_l1_num",
                "nn_distance",
                "confidence_proxy",
                "kernel_confidence",
                "proxy_score",
                "M_safe_mean",
                "pred_abs_mean",
            ]
            ok = True
            for key in numeric_keys:
                parsed[key] = to_float(row.get(key))
                if key in {"anchor_psnr", "output_gain", "oracle_gain", "P_benefit", "weighted_den", "initial_l1_num", "final_l1_num"} and parsed[key] is None:
                    ok = False
                    break
            if ok:
                rows.append(parsed)
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows):
    safe = [
        row
        for row in rows
        if row["severe_regressions"] == 0
        and row["strong_reference_regressions"] <= 1
        and row["easy_top25_output_gain"] >= -0.02
        and row["keep_count"] > 0
    ]
    positive_safe = [row for row in safe if row["weighted_delta_l1_drop"] > 1e-9 or row["mean_output_gain"] > 1e-9]
    return {
        "best_safe_l1": max(positive_safe, key=lambda row: row["weighted_delta_l1_drop"], default=None),
        "best_safe_gain": max(positive_safe, key=lambda row: row["mean_output_gain"], default=None),
        "safe_count": len(safe),
        "positive_safe_count": len(positive_safe),
        "candidate_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="sigma3")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument(
        "--confidence_keys",
        default="confidence_proxy,kernel_confidence,proxy_score,M_safe_mean,pred_abs_mean,nn_distance",
    )
    parser.add_argument("--min_keep", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.per_image_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path, args.split)
    grouped = {}
    for row in rows:
        if row["family"].endswith("_zero"):
            continue
        grouped.setdefault((row["low_size"], row["K"], row["mapper"], row["family"]), []).append(row)

    confidence_keys = [key.strip() for key in args.confidence_keys.split(",") if key.strip()]
    sweep_rows = []
    for (low_size, k_dim, mapper, family), members in sorted(grouped.items()):
        for key in confidence_keys:
            values = [row[key] for row in members if row.get(key) is not None]
            if len(values) < max(args.min_keep, 2):
                continue
            direction = "low" if key == "nn_distance" else "high"
            for threshold in threshold_values(values):
                if direction == "high":
                    keep_mask = [row.get(key) is not None and row[key] >= threshold for row in members]
                else:
                    keep_mask = [row.get(key) is not None and row[key] <= threshold for row in members]
                if sum(keep_mask) < args.min_keep:
                    continue
                summary = summarize(members, keep_mask)
                gate_pass = (
                    summary["severe_regressions"] == 0
                    and summary["strong_reference_regressions"] <= 1
                    and summary["easy_top25_output_gain"] >= -0.02
                    and (summary["weighted_delta_l1_drop"] > 1e-9 or summary["mean_output_gain"] > 1e-9)
                )
                sweep_rows.append(
                    {
                        "low_size": low_size,
                        "K": k_dim,
                        "mapper": mapper,
                        "family": family,
                        "split": args.split,
                        "confidence_key": key,
                        "direction": direction,
                        "threshold": threshold,
                        "gate_pass": gate_pass,
                        **summary,
                    }
                )

    csv_path = output_dir / f"spatial_coeff_probe_confidence_sweep_{args.label}.csv"
    json_path = output_dir / f"spatial_coeff_probe_confidence_sweep_summary_{args.label}.json"
    write_csv(csv_path, sweep_rows)
    summary = {
        "stage": "APDR-v0.4D spatial probe confidence/no-op fallback sweep",
        "source": str(input_path),
        "split": args.split,
        "confidence_keys": confidence_keys,
        "fallback": "rows below threshold are treated as no-op anchor output",
        "gate": "severe=0, strong<=1, easy>=-0.02, and nonzero l1/gain",
        "best_rows": best_rows(sweep_rows),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
