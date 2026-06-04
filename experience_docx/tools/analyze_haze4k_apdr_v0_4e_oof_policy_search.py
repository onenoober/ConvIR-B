import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def to_float(value, default=np.nan):
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values, pct):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, pct))


def read_rows(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def thresholds(values, percentiles):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    out = []
    for pct in percentiles:
        out.append(float(np.percentile(values, pct)))
    return sorted(set(out))


def load_group(rows, keys):
    data = {}
    for key in keys:
        data[key] = np.array([to_float(row.get(key)) for row in rows], dtype=np.float64)
    data["index"] = np.array([int(float(row["index"])) for row in rows], dtype=np.int64)
    data["anchor_psnr"] = np.array([to_float(row["anchor_psnr"]) for row in rows], dtype=np.float64)
    data["output_gain"] = np.array([to_float(row["output_gain"]) for row in rows], dtype=np.float64)
    data["oracle_gain"] = np.array([to_float(row["oracle_gain"]) for row in rows], dtype=np.float64)
    data["P_benefit"] = np.array([to_float(row["P_benefit"]) for row in rows], dtype=np.float64)
    data["weighted_den"] = np.array([to_float(row["weighted_den"]) for row in rows], dtype=np.float64)
    data["initial_l1_num"] = np.array([to_float(row["initial_l1_num"]) for row in rows], dtype=np.float64)
    data["final_l1_num"] = np.array([to_float(row["final_l1_num"]) for row in rows], dtype=np.float64)
    return data


def atom_mask(values, direction, threshold):
    if direction == "high":
        return np.isfinite(values) & (values >= threshold)
    if direction == "low":
        return np.isfinite(values) & (values <= threshold)
    raise ValueError(f"unknown direction {direction}")


def summarize(data, keep):
    count = int(keep.size)
    policy_gain = np.where(keep, data["output_gain"], 0.0)
    policy_final = np.where(keep, data["final_l1_num"], data["initial_l1_num"])
    den = float(np.nansum(data["weighted_den"]))
    initial = float(np.nansum(data["initial_l1_num"]) / max(den, 1e-12))
    final = float(np.nansum(policy_final) / max(den, 1e-12))
    order = np.argsort(data["anchor_psnr"])
    hard = order[: max(1, count // 4)]
    easy = order[3 * count // 4 :]
    strong_cut = percentile(data["anchor_psnr"], 75)
    strong = data["anchor_psnr"] >= strong_cut
    positive = data["oracle_gain"] > 1e-6
    oracle_sum = float(np.nansum(data["oracle_gain"][positive]))
    strong_regressions = int(np.nansum((policy_gain <= -0.05) & strong))
    severe_regressions = int(np.nansum(policy_gain <= -0.20))
    keep_count = int(np.nansum(keep))
    return {
        "count": count,
        "open_count": int(np.nansum(data["P_benefit"] >= 0.5)),
        "keep_count": keep_count,
        "open_keep_count": int(np.nansum(keep & (data["P_benefit"] >= 0.5))),
        "coverage": keep_count / max(count, 1),
        "initial_weighted_delta_l1": initial,
        "projection_weighted_delta_l1": final,
        "weighted_delta_l1_drop": (initial - final) / max(initial, 1e-12),
        "mean_gain": float(np.nanmean(policy_gain)),
        "mean_oracle_gain": float(np.nanmean(data["oracle_gain"])),
        "oracle_recovery": float(np.nansum(policy_gain[positive]) / max(oracle_sum, 1e-12)),
        "hard_bottom25_gain": float(np.nanmean(policy_gain[hard])),
        "hard_bottom25_oracle_gain": float(np.nanmean(data["oracle_gain"][hard])),
        "easy_top25_gain": float(np.nanmean(policy_gain[easy])),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regressions": strong_regressions,
        "severe_regressions": severe_regressions,
        "strong_rate": strong_regressions / max(count, 1),
    }


def gate_pass(summary):
    return (
        summary["severe_regressions"] == 0
        and summary["strong_rate"] <= 0.01
        and summary["easy_top25_gain"] >= -0.02
        and summary["hard_bottom25_gain"] >= 0.25
        and summary["mean_gain"] > 0
        and summary["coverage"] >= 0.10
        and summary["oracle_recovery"] >= 0.15
    )


def policy_row(meta, summary, primary, secondary):
    row = {
        **meta,
        "gate_pass": gate_pass(summary),
        "primary_feature": primary[0],
        "primary_direction": primary[1],
        "primary_threshold": primary[2],
        "secondary_feature": secondary[0] if secondary else "",
        "secondary_direction": secondary[1] if secondary else "",
        "secondary_threshold": secondary[2] if secondary else "",
        **summary,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="sigma3")
    parser.add_argument(
        "--primary_features",
        default="pred_abs_mean,pred_low_energy,weighted_residual_norm,kenel_confidence,kernel_confidence,nn_distance",
    )
    parser.add_argument(
        "--secondary_features",
        default="M_safe_mean,proxy_score,M_safe_nonzero_frac,nn_distance,kenel_confidence,kernel_confidence,confidence_proxy,pred_abs_mean,pred_low_energy,weighted_residual_norm",
    )
    parser.add_argument("--percentiles", default="1,2,5,10,15,20,30,40,50,60,70,80,90,95,98,99")
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.per_image_csv)
    primary_features = [item.strip() for item in args.primary_features.split(",") if item.strip()]
    secondary_features = [item.strip() for item in args.secondary_features.split(",") if item.strip()]
    percentiles = [float(item) for item in args.percentiles.split(",") if item.strip()]
    all_features = sorted(set(primary_features + secondary_features))
    groups = {}
    for row in rows:
        key = (row["mapper"], int(float(row["K"])), float(row["candidate_scale"]))
        groups.setdefault(key, []).append(row)

    policy_rows = []
    best_by_group = []
    for (mapper, k_dim, scale), members in sorted(groups.items()):
        data = load_group(members, all_features)
        feature_thresholds = {key: thresholds(data[key], percentiles) for key in all_features}
        meta = {"mapper": mapper, "K": k_dim, "scale": scale, "split": "oof"}
        local_rows = []
        primary_atoms = []
        for feature in primary_features:
            directions = ["low"] if feature == "nn_distance" else ["high"]
            for direction in directions:
                for threshold in feature_thresholds.get(feature, []):
                    primary_atoms.append((feature, direction, threshold))
        secondary_atoms = []
        for feature in secondary_features:
            for direction in ("high", "low"):
                for threshold in feature_thresholds.get(feature, []):
                    secondary_atoms.append((feature, direction, threshold))

        for primary in primary_atoms:
            keep = atom_mask(data[primary[0]], primary[1], primary[2])
            if keep.sum() == 0:
                continue
            local_rows.append(policy_row(meta, summarize(data, keep), primary, None))
            for secondary in secondary_atoms:
                if secondary[0] == primary[0] and secondary[1] == primary[1]:
                    continue
                keep2 = keep & atom_mask(data[secondary[0]], secondary[1], secondary[2])
                if keep2.sum() == 0:
                    continue
                local_rows.append(policy_row(meta, summarize(data, keep2), primary, secondary))

        local_rows = sorted(
            local_rows,
            key=lambda row: (
                not row["gate_pass"],
                row["severe_regressions"],
                -row["hard_bottom25_gain"],
                -row["mean_gain"],
                -row["coverage"],
            ),
        )
        policy_rows.extend(local_rows[: args.top_k])
        if local_rows:
            best_by_group.append(local_rows[0])

    policy_rows = sorted(
        policy_rows,
        key=lambda row: (
            not row["gate_pass"],
            row["severe_regressions"],
            -row["hard_bottom25_gain"],
            -row["mean_gain"],
            -row["coverage"],
        ),
    )
    best_by_group = sorted(
        best_by_group,
        key=lambda row: (
            not row["gate_pass"],
            row["severe_regressions"],
            -row["hard_bottom25_gain"],
            -row["mean_gain"],
            -row["coverage"],
        ),
    )

    write_csv(output_dir / f"v04e_oof_policy_search_{args.tag}.csv", policy_rows)
    write_csv(output_dir / f"v04e_oof_policy_search_best_by_group_{args.tag}.csv", best_by_group)
    passed = [row for row in policy_rows if row["gate_pass"]]
    summary = {
        "stage": "APDR-v0.4E OOF low-capacity threshold policy search",
        "source": str(args.per_image_csv),
        "tag": args.tag,
        "note": "Post-hoc OOF calibration search only; any passing policy still needs locked held-out E2.",
        "group_count": len(groups),
        "searched_policy_rows_retained": len(policy_rows),
        "gate_pass_count_retained": len(passed),
        "best_policy": policy_rows[0] if policy_rows else None,
        "best_passing_policy": passed[0] if passed else None,
    }
    write_json(output_dir / f"v04e_oof_policy_search_summary_{args.tag}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
