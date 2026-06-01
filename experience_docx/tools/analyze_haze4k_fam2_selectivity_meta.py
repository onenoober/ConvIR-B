import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path


DEFAULT_ROOT = Path("experience_docx/experiment_logs")

DEFAULT_VARIANTS = {
    "fam2_only_best": DEFAULT_ROOT
    / "haze4k_fam2_modres_stop20_20260531/scout_eval_per_image_seed3407_stop20.csv",
    "bounded_gamma_best": DEFAULT_ROOT
    / "haze4k_fam2_bounded_gamma_stop20_20260601/scout_eval_per_image_seed3407_stop20_best.csv",
    "bounded_gamma_last": DEFAULT_ROOT
    / "haze4k_fam2_bounded_gamma_stop20_20260601/scout_eval_per_image_seed3407_stop20_last.csv",
    "conf_gate_best": DEFAULT_ROOT
    / "haze4k_fam2_conf_gate_stop20_20260601/scout_eval_per_image_seed3407_stop20_best.csv",
    "conf_gate_last": DEFAULT_ROOT
    / "haze4k_fam2_conf_gate_stop20_20260601/scout_eval_per_image_seed3407_stop20_last.csv",
}

DEFAULT_MODULATION = {
    "bounded_gamma_best": DEFAULT_ROOT
    / "haze4k_fam2_bounded_gamma_stop20_20260601/modulation_per_image_seed3407_stop20_best.csv",
    "bounded_gamma_last": DEFAULT_ROOT
    / "haze4k_fam2_bounded_gamma_stop20_20260601/modulation_per_image_seed3407_stop20_last.csv",
    "conf_gate_best": DEFAULT_ROOT
    / "haze4k_fam2_conf_gate_stop20_20260601/modulation_per_image_seed3407_stop20_best.csv",
    "conf_gate_last": DEFAULT_ROOT
    / "haze4k_fam2_conf_gate_stop20_20260601/modulation_per_image_seed3407_stop20_last.csv",
}

DEFAULT_PROXY_CSV = (
    DEFAULT_ROOT / "haze4k_fam2_conf_gate_stop20_20260601/proxy_separability_seed3407.csv"
)
DEFAULT_SEED_NOISE_CSV = (
    DEFAULT_ROOT / "haze4k_stop20_noise_floor_20260601/original_seed_noise_per_image.csv"
)

BUCKETS = ("hard_bottom_25pct", "medium_middle_50pct", "easy_top_25pct")
GATED_REGRESSION_VARIANTS = (
    "bounded_gamma_best",
    "bounded_gamma_last",
    "conf_gate_best",
    "conf_gate_last",
)


def as_float(value):
    if value is None or value == "":
        return None
    return float(value)


def load_csv_by_name(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            name = raw["name"]
            row = {"name": name}
            for key, value in raw.items():
                if key == "name":
                    continue
                try:
                    row[key] = as_float(value)
                except ValueError:
                    row[key] = value
            rows[name] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_std": None,
            "min": None,
            "max": None,
            "range": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_std": sample_std(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def auc_score(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_count = sum(1 for _score, label in pairs if label)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    rank_sum_positive = 0.0
    idx = 0
    while idx < len(pairs):
        jdx = idx + 1
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        # Ranks are 1-based. Ties receive the average rank.
        avg_rank = (idx + 1 + jdx) / 2.0
        rank_sum_positive += avg_rank * sum(1 for _score, label in pairs[idx:jdx] if label)
        idx = jdx

    return (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def inverse_auc(auc):
    return None if auc is None else 1.0 - auc


def best_auc(auc):
    return None if auc is None else max(auc, 1.0 - auc)


def orient_auc(raw_auc):
    if raw_auc is None or raw_auc >= 0.5:
        return 1.0
    return -1.0


def z_values(values):
    avg = mean(values)
    var = mean([(value - avg) ** 2 for value in values])
    std = math.sqrt(var) if var and var > 0.0 else 1.0
    return [(value - avg) / std for value in values]


def prefix_keys(row, prefix, skip):
    return {
        f"{prefix}_{key}": value
        for key, value in row.items()
        if key not in skip and isinstance(value, (int, float)) and value is not None
    }


def variant_delta_key(variant):
    return f"{variant}_delta_psnr"


def variant_ssim_delta_key(variant):
    return f"{variant}_delta_ssim"


def load_variants(variant_paths):
    variant_rows = {}
    for variant, path in variant_paths.items():
        rows = load_csv_by_name(path)
        cleaned = {}
        for name, row in rows.items():
            cleaned[name] = {
                f"{variant}_original_psnr": row.get("original_psnr"),
                f"{variant}_delta_psnr": row.get("delta_psnr"),
                f"{variant}_original_ssim": row.get("original_ssim"),
                f"{variant}_delta_ssim": row.get("delta_ssim"),
            }
        variant_rows[variant] = cleaned
    return variant_rows


def load_modulation(modulation_paths):
    modulation_rows = {}
    skip = {
        "name",
        "bucket",
        "original_psnr",
        "candidate_psnr",
        "delta_psnr",
        "original_ssim",
        "candidate_ssim",
        "delta_ssim",
    }
    for variant, path in modulation_paths.items():
        rows = load_csv_by_name(path)
        modulation_rows[variant] = {
            name: prefix_keys(row, f"{variant}_mod", skip)
            for name, row in rows.items()
        }
    return modulation_rows


def merge_rows(seed_noise, proxies, variants, modulation):
    common_names = set(seed_noise) & set(proxies)
    for rows in variants.values():
        common_names &= set(rows)
    for rows in modulation.values():
        common_names &= set(rows)
    if not common_names:
        raise ValueError("No common image names across evidence files")

    rows = []
    for name in sorted(common_names):
        base = seed_noise[name]
        proxy = proxies[name]
        row = {
            "name": name,
            "bucket": base["bucket"],
            "original_seed_psnr_mean": base["psnr_mean"],
            "original_seed_psnr_sample_std": base["psnr_sample_std"],
            "original_seed_psnr_range": base["psnr_range"],
            "original_seed_ssim_mean": base["ssim_mean"],
            "original_seed_ssim_sample_std": base["ssim_sample_std"],
            "original_seed3407_psnr": base["seed_3407_psnr"],
            "original_seed2027_psnr": base["seed_2027_psnr"],
            "original_seed8675_psnr": base["seed_8675_psnr"],
        }
        for key, value in proxy.items():
            if key in ("name", "bucket", "delta_psnr"):
                continue
            if key == "original_psnr":
                row["reference_original_psnr_control"] = value
            else:
                row[f"proxy_{key}"] = value
        for variant, variant_by_name in variants.items():
            row.update(variant_by_name[name])
        for variant, modulation_by_name in modulation.items():
            row.update(modulation_by_name[name])

        fam2_delta = row["fam2_only_best_delta_psnr"]
        noise_std = row["original_seed_psnr_sample_std"]
        row["fam2_positive_gain"] = fam2_delta > 0.0
        row["fam2_noise_margin_gain"] = fam2_delta > max(0.20, noise_std)
        row["fam2_strong_gain_ge_050"] = fam2_delta >= 0.50
        row["fam2_severe_regression_le_m020"] = fam2_delta <= -0.20
        row["fam2_strong_reference_regression"] = (
            row["bucket"] == "easy_top_25pct" and fam2_delta <= -0.05
        )

        regression_hits = 0
        for variant in GATED_REGRESSION_VARIANTS:
            delta = row[variant_delta_key(variant)]
            row[f"{variant}_severe_regression_le_m020"] = delta <= -0.20
            row[f"{variant}_positive_gain"] = delta > 0.0
            row[f"{variant}_strong_reference_regression"] = (
                row["bucket"] == "easy_top_25pct" and delta <= -0.05
            )
            regression_hits += int(delta <= -0.20)
        row["bounded_gated_severe_regression_hits"] = regression_hits
        row["bounded_gated_stable_regression_ge3"] = regression_hits >= 3
        row["fam2_gain_bounded_gated_regression_overlap_ge2"] = (
            fam2_delta > 0.0 and regression_hits >= 2
        )
        rows.append(row)
    return rows


def proxy_columns(rows):
    excluded = {"proxy_original_psnr", "reference_original_psnr_control"}
    keys = []
    for key in rows[0]:
        if key.startswith("proxy_") and key not in excluded:
            values = [row[key] for row in rows]
            if all(isinstance(value, (int, float)) for value in values):
                keys.append(key)
    return sorted(keys)


def build_selectors(rows, proxy_keys):
    fam2_positive = [row["fam2_positive_gain"] for row in rows]
    z_by_key = {
        key: z_values([row[key] for row in rows])
        for key in proxy_keys
    }
    selectors = []

    for key in proxy_keys:
        raw_auc = auc_score(z_by_key[key], fam2_positive)
        direction = orient_auc(raw_auc)
        selectors.append(
            {
                "id": key,
                "kind": "proxy",
                "features": [key],
                "signs": [direction],
                "deployable": True,
                "open_scores": [direction * value for value in z_by_key[key]],
                "orientation_source": "fam2_only_positive_gain",
            }
        )

    for size in (2, 3):
        for keys in itertools.combinations(proxy_keys, size):
            signs = []
            parts = []
            for key in keys:
                raw_auc = auc_score(z_by_key[key], fam2_positive)
                sign = orient_auc(raw_auc)
                signs.append(sign)
                parts.append([sign * value for value in z_by_key[key]])
            scores = [sum(values) / size for values in zip(*parts)]
            selectors.append(
                {
                    "id": "combo_" + "__".join(keys),
                    "kind": f"combo_{size}",
                    "features": list(keys),
                    "signs": signs,
                    "deployable": True,
                    "open_scores": scores,
                    "orientation_source": "fam2_only_positive_gain",
                }
            )

    control_values = [row["reference_original_psnr_control"] for row in rows]
    control_z = z_values(control_values)
    raw_auc = auc_score(control_z, fam2_positive)
    direction = orient_auc(raw_auc)
    selectors.append(
        {
            "id": "control_reference_original_psnr_not_deployable",
            "kind": "control",
            "features": ["reference_original_psnr_control"],
            "signs": [direction],
            "deployable": False,
            "open_scores": [direction * value for value in control_z],
            "orientation_source": "fam2_only_positive_gain",
        }
    )
    return selectors


def threshold_metrics(rows, selected, delta_key):
    total = len(rows)
    selected_set = set(selected)
    all_delta = []
    selected_delta = []
    bucket_delta = {bucket: [] for bucket in BUCKETS}
    selected_by_bucket = {bucket: 0 for bucket in BUCKETS}
    strong_ref_regressions = 0
    severe_regressions = 0
    positive_selected = 0
    for idx, row in enumerate(rows):
        is_selected = idx in selected_set
        delta = row[delta_key] if is_selected else 0.0
        all_delta.append(delta)
        bucket_delta[row["bucket"]].append(delta)
        if is_selected:
            selected_delta.append(row[delta_key])
            selected_by_bucket[row["bucket"]] += 1
            positive_selected += int(row[delta_key] > 0.0)
            severe_regressions += int(row[delta_key] <= -0.20)
            strong_ref_regressions += int(
                row["bucket"] == "easy_top_25pct" and row[delta_key] <= -0.05
            )
    return {
        "selected_count": len(selected),
        "selected_ratio": len(selected) / total,
        "selected_by_bucket": selected_by_bucket,
        "positive_selected_count": positive_selected,
        "selected_delta_mean": mean(selected_delta) if selected_delta else 0.0,
        "mean_delta": mean(all_delta),
        "hard_mean_delta": mean(bucket_delta["hard_bottom_25pct"]),
        "medium_mean_delta": mean(bucket_delta["medium_middle_50pct"]),
        "easy_mean_delta": mean(bucket_delta["easy_top_25pct"]),
        "severe_regressions_le_m020": severe_regressions,
        "strong_reference_regressions_le_m005": strong_ref_regressions,
    }


def best_threshold_upper_bound(rows, scores, delta_key, strong_cap=25, easy_floor=-0.05):
    order = sorted(range(len(rows)), key=lambda idx: scores[idx], reverse=True)
    total_count = len(rows)
    bucket_counts = {
        bucket: sum(row["bucket"] == bucket for row in rows)
        for bucket in BUCKETS
    }
    selected_count = 0
    selected_delta_sum = 0.0
    selected_by_bucket = {bucket: 0 for bucket in BUCKETS}
    bucket_delta_sums = {bucket: 0.0 for bucket in BUCKETS}
    severe_regressions = 0
    strong_ref_regressions = 0
    positive_selected = 0

    def current_metrics(k):
        return {
            "selected_count": selected_count,
            "selected_ratio": selected_count / total_count,
            "selected_by_bucket": dict(selected_by_bucket),
            "positive_selected_count": positive_selected,
            "selected_delta_mean": selected_delta_sum / selected_count if selected_count else 0.0,
            "mean_delta": selected_delta_sum / total_count,
            "hard_mean_delta": bucket_delta_sums["hard_bottom_25pct"]
            / bucket_counts["hard_bottom_25pct"],
            "medium_mean_delta": bucket_delta_sums["medium_middle_50pct"]
            / bucket_counts["medium_middle_50pct"],
            "easy_mean_delta": bucket_delta_sums["easy_top_25pct"]
            / bucket_counts["easy_top_25pct"],
            "severe_regressions_le_m020": severe_regressions,
            "strong_reference_regressions_le_m005": strong_ref_regressions,
            "top_k": k,
        }

    best_any = current_metrics(0)
    best_feasible = best_any if best_any["easy_mean_delta"] >= easy_floor else None
    for k in range(0, len(rows) + 1):
        if k > 0:
            row = rows[order[k - 1]]
            delta = row[delta_key]
            bucket = row["bucket"]
            selected_count += 1
            selected_delta_sum += delta
            selected_by_bucket[bucket] += 1
            bucket_delta_sums[bucket] += delta
            positive_selected += int(delta > 0.0)
            severe_regressions += int(delta <= -0.20)
            strong_ref_regressions += int(bucket == "easy_top_25pct" and delta <= -0.05)

        metrics = current_metrics(k)
        if metrics["mean_delta"] > best_any["mean_delta"]:
            best_any = metrics
        feasible = (
            metrics["easy_mean_delta"] >= easy_floor
            and metrics["strong_reference_regressions_le_m005"] <= strong_cap
        )
        if feasible and (
            best_feasible is None or metrics["mean_delta"] > best_feasible["mean_delta"]
        ):
            best_feasible = metrics
    return {
        "best_any": best_any,
        "best_feasible": best_feasible,
    }


def selector_variant_metrics(rows, selector, variant):
    delta_key = variant_delta_key(variant)
    scores = selector["open_scores"]
    positive = [row[delta_key] > 0.0 for row in rows]
    severe = [row[delta_key] <= -0.20 for row in rows]
    easy_items = [
        (score, row)
        for score, row in zip(scores, rows)
        if row["bucket"] == "easy_top_25pct"
    ]
    easy_scores = [score for score, _row in easy_items]
    strong_ref = [row[delta_key] <= -0.05 for _score, row in easy_items]

    positive_auc_open = auc_score(scores, positive)
    severe_auc_open_event = auc_score(scores, severe)
    strong_auc_open_event = auc_score(easy_scores, strong_ref)
    ub = best_threshold_upper_bound(rows, scores, delta_key)
    return {
        "selector_id": selector["id"],
        "selector_kind": selector["kind"],
        "selector_features": "+".join(selector["features"]),
        "selector_signs": "+".join("+" if sign > 0 else "-" for sign in selector["signs"]),
        "deployable": selector["deployable"],
        "variant": variant,
        "positive_gain_auc_open_direction": positive_auc_open,
        "positive_gain_auc_best_direction": best_auc(positive_auc_open),
        "severe_regression_auc_open_predicts_event": severe_auc_open_event,
        "severe_regression_auc_open_avoids_event": inverse_auc(severe_auc_open_event),
        "severe_regression_auc_best_direction": best_auc(severe_auc_open_event),
        "strong_reference_auc_open_predicts_event_easy_only": strong_auc_open_event,
        "strong_reference_auc_open_avoids_event_easy_only": inverse_auc(strong_auc_open_event),
        "strong_reference_auc_best_direction_easy_only": best_auc(strong_auc_open_event),
        "ub_any_mean_delta": ub["best_any"]["mean_delta"],
        "ub_any_hard_mean_delta": ub["best_any"]["hard_mean_delta"],
        "ub_any_medium_mean_delta": ub["best_any"]["medium_mean_delta"],
        "ub_any_easy_mean_delta": ub["best_any"]["easy_mean_delta"],
        "ub_any_strong_reference_regressions": ub["best_any"][
            "strong_reference_regressions_le_m005"
        ],
        "ub_any_selected_count": ub["best_any"]["selected_count"],
        "ub_feasible_mean_delta": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["mean_delta"],
        "ub_feasible_hard_mean_delta": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["hard_mean_delta"],
        "ub_feasible_medium_mean_delta": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["medium_mean_delta"],
        "ub_feasible_easy_mean_delta": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["easy_mean_delta"],
        "ub_feasible_strong_reference_regressions": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["strong_reference_regressions_le_m005"],
        "ub_feasible_selected_count": None
        if ub["best_feasible"] is None
        else ub["best_feasible"]["selected_count"],
    }


def summarize_variants(rows, variants):
    output = {}
    for variant in variants:
        delta_key = variant_delta_key(variant)
        output[variant] = {
            "overall": {
                **summarize([row[delta_key] for row in rows]),
                "positive_gain_count": sum(row[delta_key] > 0.0 for row in rows),
                "severe_regression_le_m020_count": sum(row[delta_key] <= -0.20 for row in rows),
                "regression_le_m005_count": sum(row[delta_key] <= -0.05 for row in rows),
            },
            "buckets": {},
        }
        for bucket in BUCKETS:
            bucket_rows = [row for row in rows if row["bucket"] == bucket]
            output[variant]["buckets"][bucket] = {
                **summarize([row[delta_key] for row in bucket_rows]),
                "positive_gain_count": sum(row[delta_key] > 0.0 for row in bucket_rows),
                "severe_regression_le_m020_count": sum(
                    row[delta_key] <= -0.20 for row in bucket_rows
                ),
                "regression_le_m005_count": sum(row[delta_key] <= -0.05 for row in bucket_rows),
            }
    return output


def overlap_summary(rows):
    output = {}
    for bucket in (*BUCKETS, "all"):
        bucket_rows = rows if bucket == "all" else [row for row in rows if row["bucket"] == bucket]
        output[bucket] = {
            "count": len(bucket_rows),
            "fam2_positive_gain_count": sum(row["fam2_positive_gain"] for row in bucket_rows),
            "fam2_noise_margin_gain_count": sum(
                row["fam2_noise_margin_gain"] for row in bucket_rows
            ),
            "bounded_gated_stable_regression_ge3_count": sum(
                row["bounded_gated_stable_regression_ge3"] for row in bucket_rows
            ),
            "fam2_gain_and_bounded_gated_regression_overlap_ge2_count": sum(
                row["fam2_gain_bounded_gated_regression_overlap_ge2"] for row in bucket_rows
            ),
        }
        for variant in GATED_REGRESSION_VARIANTS:
            output[bucket][f"{variant}_severe_regression_le_m020_count"] = sum(
                row[f"{variant}_severe_regression_le_m020"] for row in bucket_rows
            )
            output[bucket][f"fam2_gain_and_{variant}_severe_overlap_count"] = sum(
                row["fam2_positive_gain"] and row[f"{variant}_severe_regression_le_m020"]
                for row in bucket_rows
            )
    return output


def top_images(rows, key, reverse=True, limit=30):
    ranked = sorted(rows, key=lambda row: row[key], reverse=reverse)
    return [
        {
            "name": row["name"],
            "bucket": row["bucket"],
            key: row[key],
            "original_seed_psnr_sample_std": row["original_seed_psnr_sample_std"],
            "bounded_gated_severe_regression_hits": row["bounded_gated_severe_regression_hits"],
        }
        for row in ranked[:limit]
    ]


def stable_regression_images(rows, limit=60):
    selected = [
        row for row in rows
        if row["bounded_gated_stable_regression_ge3"]
    ]
    selected.sort(
        key=lambda row: (
            row["bounded_gated_severe_regression_hits"],
            -row["fam2_only_best_delta_psnr"],
        ),
        reverse=True,
    )
    output = []
    for row in selected[:limit]:
        item = {
            "name": row["name"],
            "bucket": row["bucket"],
            "fam2_only_best_delta_psnr": row["fam2_only_best_delta_psnr"],
            "bounded_gated_severe_regression_hits": row[
                "bounded_gated_severe_regression_hits"
            ],
        }
        for variant in GATED_REGRESSION_VARIANTS:
            item[f"{variant}_delta_psnr"] = row[variant_delta_key(variant)]
        output.append(item)
    return output


def oracle_metrics(rows):
    delta_key = "fam2_only_best_delta_psnr"
    positive_selected = [
        idx for idx, row in enumerate(rows)
        if row[delta_key] > 0.0
    ]
    noise_margin_selected = [
        idx for idx, row in enumerate(rows)
        if row["fam2_noise_margin_gain"]
    ]
    true_delta_scores = [row[delta_key] for row in rows]
    return {
        "true_positive_only": threshold_metrics(rows, positive_selected, delta_key),
        "noise_margin_gain_only": threshold_metrics(rows, noise_margin_selected, delta_key),
        "true_delta_threshold_upper_bound": best_threshold_upper_bound(
            rows, true_delta_scores, delta_key
        ),
    }


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-csv", default=str(DEFAULT_PROXY_CSV))
    parser.add_argument("--seed-noise-csv", default=str(DEFAULT_SEED_NOISE_CSV))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ROOT / "haze4k_fam2_selectivity_or_kill_20260601"),
    )
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--meta-csv", default="")
    parser.add_argument("--per-image-csv", default="")
    parser.add_argument("--top-selectors", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    meta_json = Path(args.meta_json) if args.meta_json else output_dir / "selectivity_meta.json"
    meta_csv = Path(args.meta_csv) if args.meta_csv else output_dir / "selectivity_meta.csv"
    per_image_csv = (
        Path(args.per_image_csv) if args.per_image_csv else output_dir / "selectivity_per_image.csv"
    )

    seed_noise = load_csv_by_name(args.seed_noise_csv)
    proxies = load_csv_by_name(args.proxy_csv)
    variants = load_variants(DEFAULT_VARIANTS)
    modulation = load_modulation(DEFAULT_MODULATION)
    rows = merge_rows(seed_noise, proxies, variants, modulation)
    proxy_keys = proxy_columns(rows)
    selectors = build_selectors(rows, proxy_keys)

    metric_rows = []
    for selector in selectors:
        for variant in DEFAULT_VARIANTS:
            metric_rows.append(selector_variant_metrics(rows, selector, variant))

    deployable_fam2_metrics = [
        row for row in metric_rows
        if row["variant"] == "fam2_only_best" and row["deployable"]
    ]
    deployable_fam2_metrics.sort(
        key=lambda row: (
            row["positive_gain_auc_open_direction"] or 0.0,
            row["severe_regression_auc_open_avoids_event"] or 0.0,
            row["ub_feasible_mean_delta"] or -999.0,
        ),
        reverse=True,
    )
    best_positive = deployable_fam2_metrics[0]
    best_avoid_regression = max(
        deployable_fam2_metrics,
        key=lambda row: row["severe_regression_auc_open_avoids_event"] or 0.0,
    )
    best_threshold = max(
        deployable_fam2_metrics,
        key=lambda row: row["ub_feasible_mean_delta"] if row["ub_feasible_mean_delta"] is not None else -999.0,
    )
    passing_selectors = [
        row for row in deployable_fam2_metrics
        if (row["positive_gain_auc_open_direction"] or 0.0) >= 0.65
        and (row["severe_regression_auc_open_avoids_event"] or 0.0) >= 0.70
        and (row["ub_feasible_mean_delta"] or -999.0) >= 0.20
        and (row["ub_feasible_easy_mean_delta"] or -999.0) >= -0.05
        and (row["ub_feasible_strong_reference_regressions"] or 999) <= 25
    ]

    decision = {
        "label": "PASS_TARGET_BUDGET_GATE_ALLOWED" if passing_selectors else "FAIL_STOP_FAM_ROUTE",
        "pass_thresholds": {
            "same_deployable_selector_positive_gain_auc_open_direction_min": 0.65,
            "same_deployable_selector_severe_regression_auc_open_avoids_event_min": 0.70,
            "threshold_gate_global_mean_delta_min_db": 0.20,
            "threshold_gate_easy_top25_mean_delta_floor_db": -0.05,
            "threshold_gate_strong_reference_regression_cap": "25/250",
        },
        "passing_selector_count": len(passing_selectors),
        "best_passing_selectors": passing_selectors[: args.top_selectors],
        "recommendation": (
            "Run the target-budget gamma-only gate pilot."
            if passing_selectors
            else "Do not launch another FAM gate from this evidence; switch to hard-aware loss/FFL route."
        ),
    }

    summary = {
        "date": "2026-06-01",
        "branch": "codex/haze4k-fam2-selectivity-or-kill",
        "analysis_type": "no-training per-image selectivity meta-analysis",
        "inputs": {
            "proxy_csv": args.proxy_csv,
            "seed_noise_csv": args.seed_noise_csv,
            "variants": {key: str(value) for key, value in DEFAULT_VARIANTS.items()},
            "modulation": {key: str(value) for key, value in DEFAULT_MODULATION.items()},
        },
        "definitions": {
            "deployable_proxy": "input or baseline-output statistic from proxy_separability_seed3407.csv; GT PSNR control is excluded from pass/fail.",
            "positive_gain": "delta_psnr > 0.0",
            "severe_regression": "delta_psnr <= -0.20",
            "strong_reference_regression": "easy_top_25pct image with delta_psnr <= -0.05; AUC is computed inside easy_top_25pct only.",
            "fam2_noise_margin_gain": "fam2_only_best delta_psnr > max(0.20, original per-image seed PSNR sample std)",
            "bounded_gated_stable_regression": "at least 3 of bounded_gamma_best, bounded_gamma_last, conf_gate_best, conf_gate_last have delta_psnr <= -0.20",
            "open_direction": "selector scores are oriented so higher score predicts FAM2-only positive gain; severe-regression avoidance uses AUC(-open_score -> severe regression).",
        },
        "counts": {
            "images": len(rows),
            "bucket_counts": {
                bucket: sum(row["bucket"] == bucket for row in rows)
                for bucket in BUCKETS
            },
            "deployable_proxy_count": len(proxy_keys),
            "selector_count_including_combos_and_control": len(selectors),
        },
        "variant_delta_summary": summarize_variants(rows, DEFAULT_VARIANTS),
        "gain_regression_overlap": overlap_summary(rows),
        "oracle_upper_bounds": oracle_metrics(rows),
        "selector_summary": {
            "best_positive_gain_selector": best_positive,
            "best_severe_regression_avoidance_selector": best_avoid_regression,
            "best_threshold_gate_selector": best_threshold,
            "top_deployable_fam2_selectors": deployable_fam2_metrics[: args.top_selectors],
        },
        "decision": decision,
        "image_sets": {
            "fam2_top_positive_gains": top_images(rows, "fam2_only_best_delta_psnr", True),
            "fam2_worst_regressions": top_images(rows, "fam2_only_best_delta_psnr", False),
            "bounded_gated_stable_regressions": stable_regression_images(rows),
        },
        "outputs": {
            "meta_json": str(meta_json),
            "meta_csv": str(meta_csv),
            "per_image_csv": str(per_image_csv),
        },
    }

    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metric_fieldnames = list(metric_rows[0].keys())
    write_csv(meta_csv, metric_rows, metric_fieldnames)
    per_image_fieldnames = list(rows[0].keys())
    write_csv(per_image_csv, rows, per_image_fieldnames)

    print(json.dumps({
        "decision": decision["label"],
        "passing_selector_count": len(passing_selectors),
        "best_positive_gain_selector": {
            key: best_positive[key]
            for key in (
                "selector_id",
                "positive_gain_auc_open_direction",
                "severe_regression_auc_open_avoids_event",
                "ub_feasible_mean_delta",
                "ub_feasible_easy_mean_delta",
                "ub_feasible_strong_reference_regressions",
            )
        },
        "best_threshold_gate_selector": {
            key: best_threshold[key]
            for key in (
                "selector_id",
                "positive_gain_auc_open_direction",
                "severe_regression_auc_open_avoids_event",
                "ub_feasible_mean_delta",
                "ub_feasible_easy_mean_delta",
                "ub_feasible_strong_reference_regressions",
            )
        },
        "meta_json": str(meta_json),
        "meta_csv": str(meta_csv),
        "per_image_csv": str(per_image_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
