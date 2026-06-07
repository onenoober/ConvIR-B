#!/usr/bin/env python3
"""Train/evaluate a lightweight v1.9 patch mask head from tile-oracle tables.

This is table-only. It consumes `v19_patch_alpha_oracle_tiles.csv`, fits a
small logistic mask from deployable tile features, and evaluates fixed-alpha
mask policies with true OOF plus train-derived heldout confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ALPHAS = [0.25, 0.50, 0.75, 1.00]
THRESHOLDS = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]
LEAKY_PREFIXES = ("alpha_", "best_", "teacher_", "v19_", "patch_", "image_")
LEAKY_COLUMNS = {
    "split",
    "name",
    "tile_index",
    "tile_top",
    "tile_left",
    "tile_bottom",
    "tile_right",
    "tile_height",
    "tile_width",
    "a0_tile_mse",
    "best_alpha",
    "best_tile_mse",
    "best_tile_psnr",
    "a0_tile_psnr",
    "best_alpha_delta_psnr",
    "fold",
}


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
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


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            fvalue = to_float(value)
            row[key] = fvalue if fvalue is not None else value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fold_for_tile(row: dict[str, Any], folds: int = 5) -> int:
    key = f"{row['name']}:{row['tile_index']}"
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def add_folds(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["fold"] = fold_for_tile(row)


def feature_names(rows: list[dict[str, Any]], max_features: int) -> list[str]:
    candidates: list[str] = []
    for key in rows[0]:
        if key in LEAKY_COLUMNS or key.startswith(LEAKY_PREFIXES):
            continue
        if not key.startswith(("tile_", "filename_param")):
            continue
        finite = [to_float(row.get(key)) for row in rows]
        values = [value for value in finite if value is not None]
        if len(values) < max(10, len(rows) // 20):
            continue
        if max(values) - min(values) <= 1e-12:
            continue
        candidates.append(key)

    scored = []
    labels = np.array([1.0 if float(row.get("best_alpha", 0.0)) > 0 else 0.0 for row in rows])
    for key in candidates:
        values = np.array([to_float(row.get(key), 0.0) or 0.0 for row in rows], dtype=np.float64)
        if labels.std() <= 1e-12 or values.std() <= 1e-12:
            corr = 0.0
        else:
            corr = float(abs(np.corrcoef(values, labels)[0, 1]))
            if not math.isfinite(corr):
                corr = 0.0
        scored.append((corr, key))
    scored.sort(reverse=True)
    return [key for _score, key in scored[:max_features]]


def matrix(rows: list[dict[str, Any]], features: list[str], stats: dict[str, tuple[float, float]] | None = None) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    if stats is None:
        stats = {}
        for feature in features:
            values = np.array([to_float(row.get(feature), 0.0) or 0.0 for row in rows], dtype=np.float64)
            mu = float(values.mean())
            sigma = float(values.std())
            if sigma <= 1e-12:
                sigma = 1.0
            stats[feature] = (mu, sigma)
    cols = []
    for feature in features:
        mu, sigma = stats[feature]
        values = np.array([to_float(row.get(feature), 0.0) or 0.0 for row in rows], dtype=np.float64)
        cols.append((values - mu) / sigma)
    x = np.stack(cols, axis=1) if cols else np.zeros((len(rows), 0), dtype=np.float64)
    x = np.concatenate([np.ones((len(rows), 1), dtype=np.float64), x], axis=1)
    return x, stats


def fit_logistic(rows: list[dict[str, Any]], features: list[str], epochs: int, lr: float, l2: float) -> dict[str, Any]:
    x, stats = matrix(rows, features)
    y = np.array([1.0 if float(row.get("best_alpha", 0.0)) > 0 else 0.0 for row in rows], dtype=np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights_per_row = np.where(y > 0, 0.5 / pos, 0.5 / neg) * len(y)
    w = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(epochs):
        logits = x @ w
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        grad = (x.T @ ((probs - y) * weights_per_row)) / max(1, len(y))
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return {
        "features": features,
        "stats": stats,
        "weights": w.tolist(),
        "epochs": epochs,
        "lr": lr,
        "l2": l2,
    }


def predict(rows: list[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    features = list(model["features"])
    stats = {key: tuple(value) for key, value in model["stats"].items()}
    x, _ = matrix(rows, features, stats=stats)
    w = np.array(model["weights"], dtype=np.float64)
    logits = x @ w
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def tile_mse(row: dict[str, Any], alpha: float) -> float:
    if alpha <= 0:
        return float(row["a0_tile_mse"])
    return float(row[f"alpha_{alpha_tag(alpha)}_tile_mse"])


def aggregate_images(rows: list[dict[str, Any]], decisions: list[float], label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = {}
    for row, decision in zip(rows, decisions):
        grouped.setdefault((str(row["split"]), str(row["name"])), []).append((row, decision))
    image_rows = []
    for (split, name), items in grouped.items():
        total_pixels = sum(float(row["tile_height"]) * float(row["tile_width"]) for row, _decision in items)
        a0_mse = sum(float(row["a0_tile_mse"]) * float(row["tile_height"]) * float(row["tile_width"]) for row, _decision in items) / total_pixels
        selected_mse = sum(tile_mse(row, decision) * float(row["tile_height"]) * float(row["tile_width"]) for row, decision in items) / total_pixels
        selected = [decision for _row, decision in items if decision > 0]
        image_rows.append(
            {
                "split": split,
                "name": name,
                "label": label,
                "a0_mse_from_tiles": a0_mse,
                "selected_mse_from_tiles": selected_mse,
                "a0_psnr_from_tiles": psnr_from_mse(a0_mse),
                "selected_psnr_from_tiles": psnr_from_mse(selected_mse),
                "delta_psnr_from_tiles": psnr_from_mse(selected_mse) - psnr_from_mse(a0_mse),
                "tile_count": len(items),
                "selected_tile_count": len(selected),
                "selected_tile_ratio": len(selected) / max(1, len(items)),
                "selected_alpha_mean": mean(selected) if selected else 0.0,
            }
        )
    deltas = [float(row["delta_psnr_from_tiles"]) for row in image_rows]
    ordered_images = sorted(image_rows, key=lambda row: float(row["a0_psnr_from_tiles"]))
    n = max(1, len(ordered_images) // 4)
    hard = ordered_images[:n]
    easy = ordered_images[-n:]
    worst = [delta for delta in deltas if delta <= -0.20]
    strong_reg = [row for row in easy if float(row["delta_psnr_from_tiles"]) <= -0.05]
    tail_n = max(1, len(deltas) // 10)
    ordered = sorted(deltas)
    summary = {
        "label": label,
        "image_count": len(image_rows),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "hard_bottom25_delta": mean([float(row["delta_psnr_from_tiles"]) for row in hard]),
        "easy_top25_delta": mean([float(row["delta_psnr_from_tiles"]) for row in easy]),
        "worst10pct_delta": mean(ordered[:tail_n]),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "worst_regression_ratio": len(worst) / max(1, len(image_rows)),
        "strong_regression_ratio": len(strong_reg) / max(1, len(easy)),
        "mean_selected_tile_ratio": mean([float(row["selected_tile_ratio"]) for row in image_rows]),
    }
    return image_rows, summary


def score_summary(summary: dict[str, Any]) -> float:
    return (
        float(summary.get("mean_delta") or -999)
        + 0.9 * float(summary.get("hard_bottom25_delta") or -999)
        + 0.3 * float(summary.get("easy_top25_delta") or -999)
        - 1.5 * float(summary.get("worst_regression_ratio") or 0.0)
        - 0.8 * float(summary.get("strong_regression_ratio") or 0.0)
    )


def add_gate(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    checks = {
        f"{prefix}_mean_delta_ge_0p12": float(summary.get("mean_delta") or -999) >= 0.12,
        f"{prefix}_hard_bottom25_ge_0p20": float(summary.get("hard_bottom25_delta") or -999) >= 0.20,
        f"{prefix}_easy_top25_ge_neg0p02": float(summary.get("easy_top25_delta") or -999) >= -0.02,
        f"{prefix}_worst_ratio_le_0p06": float(summary.get("worst_regression_ratio") or 999) <= 0.06,
        f"{prefix}_strong_ratio_le_0p12": float(summary.get("strong_regression_ratio") or 999) <= 0.12,
    }
    summary[f"{prefix}_gate_checks"] = checks
    summary[f"{prefix}_gate_pass"] = all(checks.values())
    return summary


def select_policy(train_rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alpha in ALPHAS:
        for threshold in THRESHOLDS:
            decisions = [alpha if prob >= threshold else 0.0 for prob in probs]
            _image_rows, summary = aggregate_images(train_rows, decisions, f"train_alpha_{alpha_tag(alpha)}_thr_{threshold:.2f}")
            row = {
                "alpha": alpha,
                "threshold": threshold,
                **summary,
                "score": score_summary(summary),
            }
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
    assert best is not None
    return best


def run_oof(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    oof_tile_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        valid_rows = [row for row in rows if int(row["fold"]) == fold]
        features = feature_names(train_rows, args.max_features)
        model = fit_logistic(train_rows, features, args.epochs, args.learning_rate, args.l2)
        train_probs = predict(train_rows, model)
        policy = select_policy(train_rows, train_probs)
        valid_probs = predict(valid_rows, model)
        decisions = [float(policy["alpha"]) if prob >= float(policy["threshold"]) else 0.0 for prob in valid_probs]
        valid_image_rows, valid_summary = aggregate_images(valid_rows, decisions, f"fold_{fold}")
        fold_rows.append(
            {
                "fold": fold,
                "features": ",".join(features),
                "alpha": policy["alpha"],
                "threshold": policy["threshold"],
                **valid_summary,
                "score": score_summary(valid_summary),
            }
        )
        for row, prob, decision in zip(valid_rows, valid_probs, decisions):
            oof_tile_rows.append(
                {
                    "split": row["split"],
                    "name": row["name"],
                    "tile_index": row["tile_index"],
                    "fold": fold,
                    "prob_teacher_positive": float(prob),
                    "selected_alpha": decision,
                    "selected_tile_mse": tile_mse(row, decision),
                    "a0_tile_mse": row["a0_tile_mse"],
                    "oracle_best_alpha": row["best_alpha"],
                    "oracle_best_delta_psnr": row["best_alpha_delta_psnr"],
                }
            )
        _ = valid_image_rows
    decisions = [float(row["selected_alpha"]) for row in oof_tile_rows]
    # Preserve the same order as rows for aggregation.
    by_key = {
        (str(row["name"]), int(float(row["tile_index"]))): float(row["selected_alpha"])
        for row in oof_tile_rows
    }
    ordered_decisions = [by_key[(str(row["name"]), int(float(row["tile_index"])))] for row in rows]
    image_rows, summary = aggregate_images(rows, ordered_decisions, "oof_mask_head")
    return oof_tile_rows, fold_rows, add_gate(summary, "oof")


def run_heldout(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = [row for row in rows if str(row["split"]) == "train_inner"]
    heldout_rows = [row for row in rows if str(row["split"]) in {"val_regular", "val_hard"}]
    features = feature_names(train_rows, args.max_features)
    model = fit_logistic(train_rows, features, args.epochs, args.learning_rate, args.l2)
    policy = select_policy(train_rows, predict(train_rows, model))
    probs = predict(heldout_rows, model)
    decisions = [float(policy["alpha"]) if prob >= float(policy["threshold"]) else 0.0 for prob in probs]
    image_rows, summary = aggregate_images(heldout_rows, decisions, "heldout_mask_head")
    tile_rows = []
    for row, prob, decision in zip(heldout_rows, probs, decisions):
        tile_rows.append(
            {
                "split": row["split"],
                "name": row["name"],
                "tile_index": row["tile_index"],
                "prob_teacher_positive": float(prob),
                "selected_alpha": decision,
                "selected_tile_mse": tile_mse(row, decision),
                "a0_tile_mse": row["a0_tile_mse"],
                "oracle_best_alpha": row["best_alpha"],
                "oracle_best_delta_psnr": row["best_alpha_delta_psnr"],
            }
        )
    payload_summary = {
        **policy,
        **add_gate(summary, "heldout"),
        "features": features,
        "model": model,
    }
    return tile_rows + image_rows, payload_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_features", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()

    rows = read_rows(Path(args.tile_csv))
    add_folds(rows)
    oof_tiles, fold_rows, oof_summary = run_oof(rows, args)
    heldout_rows, heldout_summary = run_heldout(rows, args)
    decision = "MASK_HEAD_GATE_PASS"
    if not oof_summary["oof_gate_pass"] or not heldout_summary["heldout_gate_pass"]:
        decision = "MASK_HEAD_GATE_FAIL_CONTINUE_STUDENT_EXPERIMENTS"

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "v19_patch_mask_head_oof_tiles.csv", oof_tiles)
    write_csv(output_dir / "v19_patch_mask_head_oof_folds.csv", fold_rows)
    write_csv(output_dir / "v19_patch_mask_head_heldout_rows.csv", heldout_rows)
    payload = {
        "route": "ConvIR-Dehaze-v1.9-ConditionalTeacherGuided",
        "stage": "patch mask head table training",
        "locked_test_touched": False,
        "tile_csv": args.tile_csv,
        "tile_count": len(rows),
        "oof_summary": oof_summary,
        "heldout_summary": heldout_summary,
        "decision": decision,
    }
    write_json(output_dir / "v19_patch_mask_head_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V19_PATCH_MASK_HEAD_OK", flush=True)


if __name__ == "__main__":
    main()
