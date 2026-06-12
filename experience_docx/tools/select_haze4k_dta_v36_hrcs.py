#!/usr/bin/env python3
"""High-coverage risk-calibrated selector diagnostics for DTA-v3.6 HRCS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from select_haze4k_dta_v35_nested_calibration import build_rows as build_v35_rows
except Exception:  # pragma: no cover
    build_v35_rows = None

STRICT_GATES = {
    "mean_true_a0_min": 0.055,
    "hard_true_a0_min": 0.040,
    "dssim_min": -0.000005,
    "positive_ratio_min": 0.630,
    "true_vs_zero_min": 0.040,
    "true_vs_shuffle_min": 0.035,
    "true_vs_normal_min": 0.030,
    "worst_per_600_max": 48.0,
    "max_outer_worst_per_600_max": 60.0,
    "coverage_min": 0.93,
}

# Loose exploratory gates keep the user-requested queue moving. They never
# replace the strict gates, which are emitted in every summary.
RELAXED_EXPLORATORY_GATES = {
    "mean_true_a0_min": -0.020,
    "hard_true_a0_min": -0.050,
    "dssim_min": -0.000100,
    "positive_ratio_min": 0.450,
    "true_vs_zero_min": -0.020,
    "true_vs_shuffle_min": -0.030,
    "true_vs_normal_min": -0.030,
    "worst_per_600_max": 220.0,
    "max_outer_worst_per_600_max": 260.0,
    "coverage_min": 0.88,
}

TARGETS = {
    "severe_regression": lambda row: row["dPSNR"] <= -0.20,
    "strong_regression": lambda row: row["dPSNR"] <= -0.05,
    "ssim_regression": lambda row: row["dSSIM"] < -0.000005,
    "good_action": lambda row: row["dPSNR"] > 0.02 and row["dSSIM"] >= -0.000005,
}

NON_FEATURE_COLUMNS = {
    "image_id",
    "fold",
    "seed",
    "variant",
    "run_id",
    "failure_group",
    "A0_PSNR",
    "cand_PSNR",
    "dPSNR",
    "dSSIM",
    "zero_delta_psnr",
    "shuffle_delta_psnr",
    "normal_delta_psnr",
}

LEAKY_DIAGNOSTIC_COLUMNS = {
    "A0_PSNR",
    "cand_PSNR",
    "zero_delta_psnr",
    "shuffle_delta_psnr",
    "normal_delta_psnr",
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else default


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"image_id", "fold", "seed", "variant", "run_id", "failure_group"}:
                    out[key] = value
                else:
                    out[key] = finite_float(value)
            rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "")
                for key in keys
            })


def metric_summary(rows: list[dict[str, Any]], accept: list[bool]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [finite_float(row.get("dPSNR"), 0.0) if keep else 0.0 for row, keep in zip(rows, accept)]
    ssim_deltas = [finite_float(row.get("dSSIM"), 0.0) if keep else 0.0 for row, keep in zip(rows, accept)]
    original_psnr = [finite_float(row.get("A0_PSNR"), 0.0) for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(original_psnr, 75.0)
    strong_idx = [idx for idx, psnr in enumerate(original_psnr) if psnr >= strong_cut]
    selected = [row for row, keep in zip(rows, accept) if keep]
    selected_deltas = [finite_float(row.get("dPSNR"), 0.0) for row in selected]
    worst_count = sum(delta <= -0.20 for delta in deltas)
    strong_count = sum(delta <= -0.05 for delta in deltas)
    n = len(rows)

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    def surplus(control_key: str) -> float:
        return statistics.mean(
            (finite_float(row.get("dPSNR"), 0.0) - finite_float(row.get(control_key), 0.0)) if keep else 0.0
            for row, keep in zip(rows, accept)
        )

    return {
        "count": n,
        "selected_count": sum(accept),
        "coverage": sum(accept) / n,
        "mean_dPSNR": statistics.mean(deltas),
        "hard_bottom25_dPSNR": mean_at(hard_idx),
        "easy_top25_dPSNR": mean_at(easy_idx),
        "dSSIM": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / n,
        "selected_positive_ratio": (sum(delta > 0 for delta in selected_deltas) / len(selected_deltas)) if selected_deltas else float("nan"),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "strong_count_le_-0.05": strong_count,
        "strong_per_600": strong_count * 600.0 / n,
        "worst_count_le_-0.20": worst_count,
        "worst_per_600": worst_count * 600.0 / n,
        "true_vs_zero": surplus("zero_delta_psnr"),
        "true_vs_shuffle": surplus("shuffle_delta_psnr"),
        "true_vs_normal": surplus("normal_delta_psnr"),
        "false_reject_good_count": sum((not keep) and finite_float(row.get("dPSNR"), 0.0) > 0.02 for row, keep in zip(rows, accept)),
        "false_accept_severe_count": sum(keep and finite_float(row.get("dPSNR"), 0.0) <= -0.20 for row, keep in zip(rows, accept)),
        "selected_group_counts": dict(Counter(str(row.get("failure_group", "")) for row in selected)),
    }


def gate_checks(metrics: dict[str, Any], gates: dict[str, float], max_outer_worst: float | None = None) -> dict[str, bool]:
    max_outer = metrics.get("max_outer_worst_per_600", metrics.get("worst_per_600", 10**9)) if max_outer_worst is None else max_outer_worst
    return {
        "coverage": finite_float(metrics.get("coverage"), 0.0) >= gates["coverage_min"],
        "mean": finite_float(metrics.get("mean_dPSNR"), -10.0) >= gates["mean_true_a0_min"],
        "hard": finite_float(metrics.get("hard_bottom25_dPSNR"), -10.0) >= gates["hard_true_a0_min"],
        "dssim": finite_float(metrics.get("dSSIM"), -10.0) >= gates["dssim_min"],
        "positive_ratio": finite_float(metrics.get("positive_ratio"), 0.0) >= gates["positive_ratio_min"],
        "true_vs_zero": finite_float(metrics.get("true_vs_zero"), -10.0) >= gates["true_vs_zero_min"],
        "true_vs_shuffle": finite_float(metrics.get("true_vs_shuffle"), -10.0) >= gates["true_vs_shuffle_min"],
        "true_vs_normal": finite_float(metrics.get("true_vs_normal"), -10.0) >= gates["true_vs_normal_min"],
        "worst": finite_float(metrics.get("worst_per_600"), 10**9) <= gates["worst_per_600_max"],
        "max_outer_worst": finite_float(max_outer, 10**9) <= gates["max_outer_worst_per_600_max"],
    }


def selector_score(metrics: dict[str, Any]) -> float:
    return (
        finite_float(metrics.get("mean_dPSNR"), 0.0)
        + 0.25 * finite_float(metrics.get("hard_bottom25_dPSNR"), 0.0)
        + 0.15 * finite_float(metrics.get("true_vs_zero"), 0.0)
        + 0.020 * finite_float(metrics.get("coverage"), 0.0)
        + 12.0 * min(finite_float(metrics.get("dSSIM"), 0.0), 0.0)
        - 0.0008 * finite_float(metrics.get("worst_per_600"), 0.0)
        - 0.0002 * finite_float(metrics.get("strong_per_600"), 0.0)
    )


def numeric_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    candidates = sorted({key for row in rows for key in row if key not in NON_FEATURE_COLUMNS})
    out = []
    for key in candidates:
        vals = [finite_float(row.get(key)) for row in rows]
        finite_vals = [v for v in vals if math.isfinite(v)]
        if len(finite_vals) >= max(5, len(rows) // 20) and len(set(round(v, 12) for v in finite_vals)) >= 2:
            out.append(key)
    return out


def group_features(all_features: list[str]) -> dict[str, list[str]]:
    def starts(prefixes: tuple[str, ...]) -> list[str]:
        return [key for key in all_features if key.startswith(prefixes)]

    input_only = starts(("input_",))
    depth = starts(("depth_",))
    action = starts(("dta_",))
    airlight = [key for key in all_features if key.startswith("airlight_fallback")]
    trans_gt = starts(("trans_gt_",))
    deployable = [key for key in all_features if key not in LEAKY_DIAGNOSTIC_COLUMNS and not key.startswith("trans_gt_")]
    groups = {
        "input_only": input_only,
        "input_depth": sorted(set(input_only + depth + airlight)),
        "input_depth_action": sorted(set(input_only + depth + airlight + action)),
        "deployable_all": sorted(set(deployable)),
        "diagnostic_with_trans_gt": sorted(set(deployable + trans_gt)),
        "diagnostic_with_cf_delta": sorted(set(deployable + ["zero_delta_psnr", "shuffle_delta_psnr", "normal_delta_psnr"])),
    }
    return {key: [f for f in vals if f in all_features] for key, vals in groups.items()}


def is_deployable_group(group_name: str) -> bool:
    return not group_name.startswith("diagnostic_")


def labels_for(rows: list[dict[str, Any]], target: str) -> list[int]:
    func = TARGETS[target]
    return [1 if func(row) else 0 for row in rows]


def rank_auc(y_true: list[int], scores: list[float]) -> float:
    pairs = [(s, y) for s, y in zip(scores, y_true) if math.isfinite(s)]
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if ordered[k][1][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(y_true: list[int], scores: list[float]) -> float:
    pairs = sorted(((s, y) for s, y in zip(scores, y_true) if math.isfinite(s)), reverse=True)
    pos = sum(y for _, y in pairs)
    if pos == 0:
        return float("nan")
    seen_pos = 0
    total = 0.0
    for idx, (_, y) in enumerate(pairs, start=1):
        if y:
            seen_pos += 1
            total += seen_pos / idx
    return total / pos


def brier_score(y_true: list[int], probs: list[float]) -> float:
    vals = [(min(max(p, 0.0), 1.0) - y) ** 2 for y, p in zip(y_true, probs) if math.isfinite(p)]
    return statistics.mean(vals) if vals else float("nan")


def ece_score(y_true: list[int], probs: list[float], bins: int = 10) -> float:
    pairs = [(min(max(p, 0.0), 1.0), y) for y, p in zip(y_true, probs) if math.isfinite(p)]
    if not pairs:
        return float("nan")
    total = len(pairs)
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        chunk = [(p, y) for p, y in pairs if p >= lo and (p < hi or b == bins - 1)]
        if not chunk:
            continue
        ece += len(chunk) / total * abs(statistics.mean(p for p, _ in chunk) - statistics.mean(y for _, y in chunk))
    return ece


def sigmoid(x: float) -> float:
    if x >= 35:
        return 1.0
    if x <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class ConstantModel:
    def __init__(self, prob: float) -> None:
        self.prob = min(max(prob, 1e-6), 1 - 1e-6)
        self.top_features: list[str] = []
        self.backend = "constant"

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.prob] * len(rows)


class FeatureScaler:
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for feature in self.features:
            vals = [finite_float(row.get(feature)) for row in rows]
            finite_vals = [v for v in vals if math.isfinite(v)]
            med = percentile(finite_vals, 50.0) if finite_vals else 0.0
            filled = [v if math.isfinite(v) else med for v in vals]
            mean = statistics.mean(filled) if filled else 0.0
            std = statistics.pstdev(filled) if len(filled) > 1 else 1.0
            self.medians[feature] = med
            self.means[feature] = mean
            self.stds[feature] = std if std >= 1e-9 else 1.0

    def transform(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        out = []
        for row in rows:
            vals = []
            for feature in self.features:
                value = finite_float(row.get(feature))
                if not math.isfinite(value):
                    value = self.medians.get(feature, 0.0)
                vals.append((value - self.means.get(feature, 0.0)) / self.stds.get(feature, 1.0))
            out.append(vals)
        return out


class PureLogisticModel:
    def __init__(self, features: list[str], l2: float = 0.05, iterations: int = 180, lr: float = 0.08) -> None:
        self.features = features
        self.l2 = l2
        self.iterations = iterations
        self.lr = lr
        self.scaler = FeatureScaler(features)
        self.weights = [0.0] * len(features)
        self.bias = 0.0
        self.top_features: list[str] = []
        self.backend = "pure_python_logistic"

    def fit(self, rows: list[dict[str, Any]], y: list[int]) -> None:
        self.scaler.fit(rows)
        x = self.scaler.transform(rows)
        n = len(y)
        pos = sum(y)
        neg = n - pos
        if n == 0 or pos == 0 or neg == 0:
            return
        weights = [0.5 / pos if label else 0.5 / neg for label in y]
        prior = min(max(pos / n, 1e-5), 1 - 1e-5)
        self.bias = math.log(prior / (1 - prior))
        for _ in range(self.iterations):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            for xi, yi, wi in zip(x, y, weights):
                p = sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, xi)))
                err = (p - yi) * wi
                grad_b += err
                for j, value in enumerate(xi):
                    grad_w[j] += err * value
            for j in range(len(self.weights)):
                grad_w[j] += self.l2 * self.weights[j] / max(1, len(self.weights))
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * grad_b
        ranked = sorted(zip(self.features, self.weights), key=lambda item: abs(item[1]), reverse=True)
        self.top_features = [name for name, _ in ranked[:8]]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        return [sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, xi))) for xi in self.scaler.transform(rows)]


class PureBoostedStumpsModel:
    def __init__(self, features: list[str], rounds: int = 35, lr: float = 0.12) -> None:
        self.features = features
        self.rounds = rounds
        self.lr = lr
        self.scaler = FeatureScaler(features)
        self.bias = 0.0
        self.stumps: list[tuple[int, float, int, float]] = []
        self.top_features: list[str] = []
        self.backend = "pure_python_boosted_stumps"

    def fit(self, rows: list[dict[str, Any]], y: list[int]) -> None:
        self.scaler.fit(rows)
        x = self.scaler.transform(rows)
        n = len(y)
        pos = sum(y)
        neg = n - pos
        if n == 0 or pos == 0 or neg == 0 or not self.features:
            return
        prior = min(max(pos / n, 1e-5), 1 - 1e-5)
        self.bias = math.log(prior / (1 - prior))
        scores = [self.bias] * n
        feature_thresholds: dict[int, list[float]] = {}
        for j in range(len(self.features)):
            vals = [row[j] for row in x]
            feature_thresholds[j] = sorted({percentile(vals, pct) for pct in (10, 25, 40, 50, 60, 75, 90)})
        for _ in range(self.rounds):
            probs = [sigmoid(s) for s in scores]
            residuals = [yi - pi for yi, pi in zip(y, probs)]
            best: tuple[float, int, float, int, float] | None = None
            for j, thresholds in feature_thresholds.items():
                for threshold in thresholds:
                    for direction in (-1, 1):
                        group = [idx for idx, xi in enumerate(x) if (xi[j] <= threshold if direction < 0 else xi[j] >= threshold)]
                        if len(group) < 5 or len(group) > n - 5:
                            continue
                        step = statistics.mean(residuals[idx] for idx in group)
                        gain = abs(step) * math.sqrt(len(group))
                        if best is None or gain > best[0]:
                            best = (gain, j, threshold, direction, step)
            if best is None or best[0] <= 1e-9:
                break
            _, j, threshold, direction, step = best
            clipped_step = max(min(step, 1.5), -1.5)
            self.stumps.append((j, threshold, direction, clipped_step))
            for idx, xi in enumerate(x):
                if xi[j] <= threshold if direction < 0 else xi[j] >= threshold:
                    scores[idx] += self.lr * clipped_step
        counts = Counter(self.features[j] for j, _, _, _ in self.stumps)
        self.top_features = [name for name, _ in counts.most_common(8)]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        x = self.scaler.transform(rows)
        out = []
        for xi in x:
            score = self.bias
            for j, threshold, direction, step in self.stumps:
                if xi[j] <= threshold if direction < 0 else xi[j] >= threshold:
                    score += self.lr * step
            out.append(sigmoid(score))
        return out


def univariate_top_features(rows: list[dict[str, Any]], y: list[int], features: list[str]) -> list[str]:
    scored = []
    for feature in features:
        scores = [finite_float(row.get(feature)) for row in rows]
        auc = rank_auc(y, scores)
        if math.isfinite(auc):
            scored.append((abs(auc - 0.5), feature))
    return [feature for _, feature in sorted(scored, reverse=True)]


def try_sklearn_model(model_type: str, rows: list[dict[str, Any]], y: list[int], features: list[str]) -> Any | None:
    try:
        import numpy as np  # type: ignore
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception:
        return None
    if not features or len(set(y)) < 2:
        return None
    arr = np.array([[finite_float(row.get(feature)) for feature in features] for row in rows], dtype=float)
    labels = np.array(y, dtype=int)
    if model_type == "logistic":
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=0.5))
    elif model_type == "gbdt":
        clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=0.1, random_state=3407))
    else:
        return None
    clf.fit(arr, labels)

    class SkModel:
        backend = "sklearn"

        def __init__(self, pipeline: Any, feats: list[str]) -> None:
            self.pipeline = pipeline
            self.features = feats
            self.top_features = univariate_top_features(rows, y, feats)[:8]

        def predict_proba(self, pred_rows: list[dict[str, Any]]) -> list[float]:
            x_pred = np.array([[finite_float(row.get(feature)) for feature in self.features] for row in pred_rows], dtype=float)
            return [float(v) for v in self.pipeline.predict_proba(x_pred)[:, 1]]

    return SkModel(clf, features)


def fit_binary_model(model_type: str, rows: list[dict[str, Any]], y: list[int], features: list[str]) -> Any:
    if not rows or not features or len(set(y)) < 2:
        return ConstantModel(sum(y) / len(y) if y else 0.0)
    sk_model = try_sklearn_model(model_type, rows, y, features)
    if sk_model is not None:
        return sk_model
    model = PureBoostedStumpsModel(features) if model_type == "gbdt" else PureLogisticModel(features)
    model.fit(rows, y)
    if not getattr(model, "top_features", None):
        model.top_features = univariate_top_features(rows, y, features)[:8]
    return model


def fit_risk_model(model_type: str, train_rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    models = {}
    top_features = []
    backends = set()
    for target in TARGETS:
        y = labels_for(train_rows, target)
        model = fit_binary_model(model_type, train_rows, y, features)
        models[target] = model
        top_features.extend(getattr(model, "top_features", []))
        backends.add(getattr(model, "backend", "unknown"))
    return {
        "model_type": model_type,
        "features": features,
        "models": models,
        "top_features": [name for name, _ in Counter(top_features).most_common(10)],
        "backend": "+".join(sorted(backends)),
    }


def predict_components(model_bundle: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {target: model.predict_proba(rows) for target, model in model_bundle["models"].items()}


def combined_risk(components: dict[str, list[float]]) -> list[float]:
    n = len(next(iter(components.values()))) if components else 0
    out = []
    for idx in range(n):
        out.append(
            components["severe_regression"][idx]
            + 0.35 * components["strong_regression"][idx]
            + 0.20 * components["ssim_regression"][idx]
            - 0.10 * components["good_action"][idx]
        )
    return out


def combined_utility(components: dict[str, list[float]]) -> list[float]:
    n = len(next(iter(components.values()))) if components else 0
    out = []
    for idx in range(n):
        out.append(
            components["good_action"][idx]
            - components["severe_regression"][idx]
            - 0.35 * components["strong_regression"][idx]
            - 0.20 * components["ssim_regression"][idx]
        )
    return out


def threshold_for_coverage(scores: list[float], coverage: float) -> float:
    vals = [v for v in scores if math.isfinite(v)]
    return percentile(vals, min(max(coverage, 0.0), 1.0) * 100.0) if vals else float("inf")


def apply_threshold(scores: list[float], tau: float) -> list[bool]:
    return [math.isfinite(score) and score <= tau for score in scores]


def oracle_scores(rows: list[dict[str, Any]]) -> list[float]:
    return [-(finite_float(row.get("dPSNR"), 0.0) + 0.02 * finite_float(row.get("dSSIM"), 0.0)) for row in rows]


def split_train_calib(train_pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    folds = sorted({str(row.get("fold", "")) for row in train_pool})
    if len(folds) >= 2:
        calib_fold = folds[-1]
        return [row for row in train_pool if str(row.get("fold", "")) != calib_fold], [row for row in train_pool if str(row.get("fold", "")) == calib_fold], calib_fold
    return train_pool, train_pool, "same_as_model_train_due_to_two_fold_source"


def auc_report_rows(
    variant: str,
    outer_fold: str,
    model_type: str,
    feature_group: str,
    deployable: bool,
    components: dict[str, list[float]],
    eval_rows: list[dict[str, Any]],
    top_features: list[str],
) -> list[dict[str, Any]]:
    out = []
    for target, probs in components.items():
        y = labels_for(eval_rows, target)
        out.append({
            "candidate": variant,
            "selector_type": model_type,
            "feature_group": feature_group,
            "deployable": deployable,
            "outer_fold": outer_fold,
            "target": target,
            "roc_auc": rank_auc(y, probs),
            "pr_auc": average_precision(y, probs),
            "brier_score": brier_score(y, probs),
            "ece": ece_score(y, probs),
            "positive_count": sum(y),
            "eval_count": len(y),
            "top_features": ";".join(top_features[:8]),
        })
    return out


def reliability_rows(
    variant: str,
    outer_fold: str,
    model_type: str,
    feature_group: str,
    eval_rows: list[dict[str, Any]],
    risk_scores: list[float],
    bins: int = 10,
) -> list[dict[str, Any]]:
    pairs = [(score, row) for score, row in zip(risk_scores, eval_rows) if math.isfinite(score)]
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda item: item[0])
    out = []
    for b in range(bins):
        start = round(len(ordered) * b / bins)
        end = round(len(ordered) * (b + 1) / bins)
        chunk = ordered[start:end]
        if not chunk:
            continue
        rows = [row for _, row in chunk]
        out.append({
            "candidate": variant,
            "outer_fold": outer_fold,
            "selector_type": model_type,
            "feature_group": feature_group,
            "risk_bin": b,
            "count": len(rows),
            "risk_score_min": chunk[0][0],
            "risk_score_max": chunk[-1][0],
            "predicted_severe_rate_proxy": statistics.mean(score for score, _ in chunk),
            "observed_severe_rate": statistics.mean(1 if row["dPSNR"] <= -0.20 else 0 for row in rows),
            "observed_strong_rate": statistics.mean(1 if row["dPSNR"] <= -0.05 else 0 for row in rows),
            "observed_positive_rate": statistics.mean(1 if row["dPSNR"] > 0 else 0 for row in rows),
            "mean_dPSNR": statistics.mean(row["dPSNR"] for row in rows),
        })
    return out


def error_type(row: dict[str, Any], accept: bool) -> str:
    dpsnr = finite_float(row.get("dPSNR"), 0.0)
    dssim = finite_float(row.get("dSSIM"), 0.0)
    if accept and dpsnr <= -0.20:
        return "false_accept_severe"
    if accept and dpsnr <= -0.05:
        return "false_accept_strong"
    if (not accept) and dpsnr > 0.02 and dssim >= -0.000005:
        return "false_reject_good"
    if (not accept) and dpsnr <= -0.05:
        return "true_reject_bad"
    if accept and dpsnr > 0.02:
        return "true_accept_good"
    return "neutral"


def compact_row_features(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "input_brightness_mean",
        "input_texture_mean",
        "depth_mean",
        "depth_std",
        "dta_final_feature_action_abs_mean",
        "dta_final_feature_gate_mean",
        "dta_stage2_feature_action_abs_mean",
        "dta_stage3_feature_action_abs_mean",
        "zero_delta_psnr",
        "shuffle_delta_psnr",
        "normal_delta_psnr",
    ]
    return {key: row.get(key, "") for key in keys if key in row}


def aggregate_curve_rows(curve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(row["variant"], row["selector_type"], row["feature_group"], float(row["coverage_target"]))].append(row)
    out = []
    metric_keys = [
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "worst_per_600",
        "strong_per_600",
        "true_vs_zero",
        "true_vs_shuffle",
        "true_vs_normal",
        "false_reject_good_count",
        "false_accept_severe_count",
    ]
    for (variant, selector_type, feature_group, coverage_target), rows in sorted(grouped.items()):
        avg_metrics = {key: safe_mean([finite_float(row.get(key)) for row in rows], 0.0) for key in metric_keys}
        max_outer_worst = max(finite_float(row.get("worst_per_600"), 0.0) for row in rows)
        avg_metrics["max_outer_worst_per_600"] = max_outer_worst
        strict = gate_checks(avg_metrics, STRICT_GATES, max_outer_worst)
        relaxed = gate_checks(avg_metrics, RELAXED_EXPLORATORY_GATES, max_outer_worst)
        avg_metrics.update({
            "variant": variant,
            "selector_type": selector_type,
            "feature_group": feature_group,
            "deployable": is_deployable_group(feature_group) and selector_type != "oracle",
            "coverage_target": coverage_target,
            "outer_reports": len(rows),
            "selector_score": selector_score(avg_metrics),
            "strict_gate_pass": all(strict.values()),
            "relaxed_exploratory_gate_pass": all(relaxed.values()),
            "strict_gate_checks": strict,
            "relaxed_gate_checks": relaxed,
        })
        out.append(avg_metrics)
    return out


def choose_best_configs(aggregate_rows: list[dict[str, Any]], prefer_deployable: bool = True) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant in sorted({row["variant"] for row in aggregate_rows}):
        rows = [row for row in aggregate_rows if row["variant"] == variant]
        if prefer_deployable:
            deployable_rows = [row for row in rows if row.get("deployable")]
            if deployable_rows:
                rows = deployable_rows
        viable = [row for row in rows if finite_float(row.get("coverage"), 0.0) >= 0.88]
        if viable:
            rows = viable
        rows = sorted(rows, key=lambda row: (bool(row.get("relaxed_exploratory_gate_pass")), finite_float(row.get("selector_score"), -999.0)), reverse=True)
        if rows:
            out[variant] = rows[0]
    return out


def make_oracle_curve(variant_rows: list[dict[str, Any]], variant: str, coverage_grid: list[float]) -> list[dict[str, Any]]:
    scores = oracle_scores(variant_rows)
    out = []
    for coverage in coverage_grid:
        tau = threshold_for_coverage(scores, coverage)
        accept = apply_threshold(scores, tau)
        out.append({
            "variant": variant,
            "outer_fold": "all_oof_oracle",
            "selector_type": "oracle",
            "feature_group": "oracle_dpsnr_leak",
            "deployable": False,
            "coverage_target": coverage,
            "threshold": tau,
            **metric_summary(variant_rows, accept),
        })
    return out


def action_bank_summary(
    best_configs: dict[str, dict[str, Any]],
    eval_cache: dict[tuple[str, str, str, str, float], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    bank_variants = [variant for variant in ("l2_fdf_lite_s002_g025_bm2", "l3_fdf_lite_s004_g015_bm2", "l1_fdf_lite_s004_g025_bm2") if variant in best_configs]
    if not bank_variants:
        return []
    per_variant_items: dict[str, list[dict[str, Any]]] = {}
    for variant in bank_variants:
        cfg = best_configs[variant]
        selector_type = str(cfg["selector_type"])
        feature_group = str(cfg["feature_group"])
        coverage = float(cfg["coverage_target"])
        items = []
        for key, cached_rows in eval_cache.items():
            v, _, mt, fg, cov = key
            if v == variant and mt == selector_type and fg == feature_group and abs(float(cov) - coverage) <= 1e-9:
                items.extend(cached_rows)
        per_variant_items[variant] = items

    keyed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for variant, items in per_variant_items.items():
        for item in items:
            row = item["row"]
            key = (str(row.get("image_id")), str(row.get("fold")), str(row.get("seed")))
            keyed[key][variant] = item

    selector_rows = []
    oracle_rows = []
    only_rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for by_variant in keyed.values():
        if not by_variant:
            continue
        sample = next(iter(by_variant.values()))["row"]
        best_oracle_delta = 0.0
        best_oracle_ssim = 0.0
        best_oracle_variant = "A0"
        chosen_item: dict[str, Any] | None = None
        for variant, item in by_variant.items():
            row = item["row"]
            only_rows_by_variant[variant].append(row)
            dpsnr = finite_float(row.get("dPSNR"), 0.0)
            dssim = finite_float(row.get("dSSIM"), 0.0)
            if dpsnr > best_oracle_delta:
                best_oracle_delta = dpsnr
                best_oracle_ssim = dssim
                best_oracle_variant = variant
            if item["accept"] and (chosen_item is None or finite_float(item.get("utility_score"), -999.0) > finite_float(chosen_item.get("utility_score"), -999.0)):
                chosen_item = item
        if chosen_item is None:
            selector_rows.append({**sample, "dPSNR": 0.0, "dSSIM": 0.0, "zero_delta_psnr": 0.0, "shuffle_delta_psnr": 0.0, "normal_delta_psnr": 0.0, "failure_group": "fallback_a0"})
        else:
            selector_rows.append(chosen_item["row"])
        oracle_rows.append({**sample, "dPSNR": best_oracle_delta, "dSSIM": best_oracle_ssim, "failure_group": f"oracle_{best_oracle_variant}"})

    out = []
    a0_rows = [{**row, "dPSNR": 0.0, "dSSIM": 0.0, "zero_delta_psnr": 0.0, "shuffle_delta_psnr": 0.0, "normal_delta_psnr": 0.0} for row in selector_rows]
    for label, label_rows in [("A0_only", a0_rows), ("oracle_choose_A0_L2_L3_L1", oracle_rows), ("selector_choose_A0_L2_L3_L1", selector_rows)]:
        out.append({"policy": label, **metric_summary(label_rows, [True] * len(label_rows))})
    for variant, label_rows in only_rows_by_variant.items():
        out.append({"policy": f"{variant}_only", **metric_summary(label_rows, [True] * len(label_rows))})
    return out


def nested_selector_analysis(rows: list[dict[str, Any]], variants: list[str], feature_group_names: list[str], model_types: list[str], coverage_grid: list[float]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("variant") in variants]
    all_features = numeric_feature_columns(rows)
    groups = group_features(all_features)
    requested_groups = {name: groups[name] for name in feature_group_names if name in groups and groups[name]}
    curve_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    reliability_all: list[dict[str, Any]] = []
    eval_cache: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    prediction_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        if not variant_rows:
            continue
        curve_rows.extend(make_oracle_curve(variant_rows, variant, coverage_grid))
        folds = sorted({str(row.get("fold", "")) for row in variant_rows})
        for outer_fold in folds:
            eval_rows = [row for row in variant_rows if str(row.get("fold", "")) == outer_fold]
            train_pool = [row for row in variant_rows if str(row.get("fold", "")) != outer_fold]
            if not eval_rows or not train_pool:
                continue
            model_train, calib_rows, calib_fold = split_train_calib(train_pool)
            for feature_group, features in requested_groups.items():
                for model_type in model_types:
                    bundle = fit_risk_model(model_type, model_train, features)
                    calib_scores = combined_risk(predict_components(bundle, calib_rows))
                    eval_components = predict_components(bundle, eval_rows)
                    eval_scores = combined_risk(eval_components)
                    eval_utility = combined_utility(eval_components)
                    prediction_cache[(variant, outer_fold, model_type, feature_group)] = {
                        "eval_rows": eval_rows,
                        "risk_scores": eval_scores,
                        "utility_scores": eval_utility,
                        "components": eval_components,
                        "backend": bundle["backend"],
                        "top_features": bundle["top_features"],
                    }
                    auc_rows.extend(auc_report_rows(variant, outer_fold, model_type, feature_group, is_deployable_group(feature_group), eval_components, eval_rows, bundle["top_features"]))
                    reliability_all.extend(reliability_rows(variant, outer_fold, model_type, feature_group, eval_rows, eval_scores))
                    for coverage in coverage_grid:
                        tau = threshold_for_coverage(calib_scores, coverage)
                        accept = apply_threshold(eval_scores, tau)
                        metrics = metric_summary(eval_rows, accept)
                        row = {
                            "variant": variant,
                            "outer_fold": outer_fold,
                            "selector_type": model_type,
                            "feature_group": feature_group,
                            "deployable": is_deployable_group(feature_group),
                            "backend": bundle["backend"],
                            "calibration_fold": calib_fold,
                            "model_train_count": len(model_train),
                            "calibration_count": len(calib_rows),
                            "eval_count": len(eval_rows),
                            "coverage_target": coverage,
                            "threshold": tau,
                            "top_features": ";".join(bundle["top_features"][:8]),
                        }
                        row.update(metrics)
                        curve_rows.append(row)
                        eval_cache[(variant, outer_fold, model_type, feature_group, coverage)] = [
                            {"row": eval_row, "accept": keep, "risk_score": score, "utility_score": utility, "threshold": tau}
                            for eval_row, keep, score, utility in zip(eval_rows, accept, eval_scores, eval_utility)
                        ]

    aggregate_rows = aggregate_curve_rows([row for row in curve_rows if row["selector_type"] != "oracle"])
    oracle_aggregate = aggregate_curve_rows([row for row in curve_rows if row["selector_type"] == "oracle"])
    best_configs = choose_best_configs(aggregate_rows, prefer_deployable=True)
    reliability_rows_best: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for variant, config in best_configs.items():
        selector_type = str(config["selector_type"])
        feature_group = str(config["feature_group"])
        coverage = float(config["coverage_target"])
        for key, cached_rows in eval_cache.items():
            v, outer_fold, mt, fg, cov = key
            if v != variant or mt != selector_type or fg != feature_group or abs(float(cov) - coverage) > 1e-9:
                continue
            pred = prediction_cache.get((variant, outer_fold, selector_type, feature_group))
            if pred:
                reliability_rows_best.extend(reliability_rows(variant, outer_fold, selector_type, feature_group, pred["eval_rows"], pred["risk_scores"]))
            for item in cached_rows:
                row = item["row"]
                accept = bool(item["accept"])
                error_rows.append({
                    "image_id": row.get("image_id"),
                    "fold": row.get("fold"),
                    "seed": row.get("seed"),
                    "candidate": variant,
                    "selector_type": selector_type,
                    "feature_group": feature_group,
                    "coverage_target": coverage,
                    "risk_score": item["risk_score"],
                    "utility_score": item["utility_score"],
                    "threshold": item["threshold"],
                    "accept": accept,
                    "dPSNR": row.get("dPSNR"),
                    "dSSIM": row.get("dSSIM"),
                    "failure_group": row.get("failure_group"),
                    "error_type": error_type(row, accept),
                    **compact_row_features(row),
                })

    return {
        "curve_rows": curve_rows,
        "aggregate_rows": aggregate_rows,
        "oracle_aggregate_rows": oracle_aggregate,
        "auc_rows": auc_rows,
        "reliability_rows_all": reliability_all,
        "reliability_rows_best": reliability_rows_best,
        "error_rows": error_rows,
        "bank_rows": action_bank_summary(best_configs, eval_cache),
        "best_configs": best_configs,
        "feature_groups": requested_groups,
    }


def make_summary_payload(args: argparse.Namespace, analysis: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_configs_payload = {}
    for variant, config in analysis["best_configs"].items():
        best_configs_payload[variant] = {key: value for key, value in config.items() if key not in {"strict_gate_checks", "relaxed_gate_checks"}}
        best_configs_payload[variant]["strict_gate_checks"] = config.get("strict_gate_checks")
        best_configs_payload[variant]["relaxed_gate_checks"] = config.get("relaxed_gate_checks")
    return {
        "route": "DTA-v3.6 HRCS",
        "protocol": "nested_high_coverage_reject_only_risk_calibrated_selector",
        "input_action_table": str(args.input_action_table) if args.input_action_table else None,
        "input_evidence_dir": str(args.input_evidence_dir) if args.input_evidence_dir else None,
        "rows": len(rows),
        "variants": parse_str_list(args.variants),
        "coverage_grid": parse_float_list(args.coverage_grid),
        "selector_models": parse_str_list(args.selector_models),
        "feature_groups": {key: len(vals) for key, vals in analysis["feature_groups"].items()},
        "strict_gates": STRICT_GATES,
        "relaxed_exploratory_gates": RELAXED_EXPLORATORY_GATES,
        "locked_test_policy": "blocked by default; user-requested relaxed one-shot override must be documented separately before touching locked test",
        "leakage_note": "deployable groups exclude GT PSNR deltas and trans_gt columns; diagnostic groups are not deployable.",
        "best_configs": best_configs_payload,
        "oracle_best_by_variant": choose_best_configs(analysis["oracle_aggregate_rows"], prefer_deployable=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_action_table", type=Path, default=None)
    parser.add_argument("--input_evidence_dir", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--variants", default="l3_fdf_lite_s004_g015_bm2,l1_fdf_lite_s004_g025_bm2,l2_fdf_lite_s002_g025_bm2")
    parser.add_argument("--selector_models", default="logistic,gbdt")
    parser.add_argument("--feature_groups", default="input_only,input_depth,input_depth_action,deployable_all,diagnostic_with_trans_gt,diagnostic_with_cf_delta")
    parser.add_argument("--coverage_grid", default="1.00,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.90")
    parser.add_argument("--write_all_reliability", action="store_true")
    args = parser.parse_args()

    if args.input_action_table:
        rows = read_csv_rows(args.input_action_table)
    elif args.input_evidence_dir and build_v35_rows is not None:
        rows = build_v35_rows(args.input_evidence_dir)
    else:
        raise SystemExit("Provide --input_action_table or --input_evidence_dir with the v3.5 builder available")

    analysis = nested_selector_analysis(
        rows,
        parse_str_list(args.variants),
        parse_str_list(args.feature_groups),
        parse_str_list(args.selector_models),
        parse_float_list(args.coverage_grid),
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "v36_high_coverage_rejection_curve.csv", analysis["curve_rows"])
    write_csv(out / "v36_high_coverage_rejection_curve_aggregate.csv", analysis["aggregate_rows"] + analysis["oracle_aggregate_rows"])
    write_csv(out / "v36_risk_feature_auc_report.csv", analysis["auc_rows"])
    write_csv(out / "v36_selector_reliability_bins.csv", analysis["reliability_rows_all"] if args.write_all_reliability else analysis["reliability_rows_best"])
    write_csv(out / "v36_selector_error_table.csv", analysis["error_rows"])
    write_csv(out / "v36_action_bank_oracle_vs_selector.csv", analysis["bank_rows"])
    write_csv(out / "v36_selector_best_configs.csv", [{"variant": variant, **config} for variant, config in analysis["best_configs"].items()])
    (out / "v36_selector_summary.json").write_text(json.dumps(make_summary_payload(args, analysis, rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_6_HRCS_SELECTOR_OK "
        f"rows={len(rows)} curve_rows={len(analysis['curve_rows'])} best_configs={len(analysis['best_configs'])}"
    )


if __name__ == "__main__":
    main()
