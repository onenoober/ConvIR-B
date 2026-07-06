#!/usr/bin/env python3
"""Build the v2.40 teacher residual alignment atlas.

This is a diagnostic-only audit. It reads existing train-derived Haze4K
per-image evidence and tensor caches, computes teacher residual geometry against
GT, and optionally measures whether runtime-visible features can predict the
alignment labels. It does not train a restoration model or touch locked test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


EPS = 1.0e-12
RUNTIME_FEATURE_EXCLUDE_TOKENS = (
    "label",
    "score",
    "delta",
    "rule",
    "eligible",
    "noop",
    "unsafe",
    "severe",
    "strong_reference",
    "bucket",
    "fold_id",
    "image_id",
)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def parse_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_int(v: Any, default: int = -1) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def alpha_matches(v: Any, target: float, tol: float = 1.0e-9) -> bool:
    return abs(parse_float(v) - target) <= tol


def is_unsafe_alpha_row(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    return (
        parse_bool(row.get("negative"))
        or parse_bool(row.get("severe"))
        or parse_bool(row.get("strong_reference_regression"))
    )


def tensor_load(path: str) -> torch.Tensor:
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        tensor_values = [v for v in x.values() if torch.is_tensor(v)]
        if not tensor_values:
            raise ValueError(f"no tensor payload in {path}")
        x = tensor_values[0]
    if not torch.is_tensor(x):
        raise TypeError(f"unsupported tensor payload {type(x)!r} in {path}")
    x = x.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [N,3,H,W] tensor in {path}, got {tuple(x.shape)}")
    return x.clamp(0.0, 1.0)


def image_load(path: str, like: torch.Tensor) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if tuple(x.shape[-2:]) != tuple(like.shape[-2:]):
        x = F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)
    return x.clamp(0.0, 1.0)


def luma(x: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def mean_sq(x: torch.Tensor) -> float:
    return float((x * x).mean().item())


def low_high_energy(x: torch.Tensor) -> tuple[float, float]:
    low = F.avg_pool2d(x, kernel_size=15, stride=1, padding=7)
    high = x - low
    return mean_sq(low), mean_sq(high)


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def teacher_geometry(a0: torch.Tensor, gt: torch.Tensor, teacher: torch.Tensor) -> dict[str, float | bool]:
    e0 = a0 - gt
    d = teacher - a0
    et = teacher - gt
    mse_a0 = mean_sq(e0)
    mse_teacher = mean_sq(et)
    d_energy = mean_sq(d)
    dot = float((e0 * d).mean().item())
    e_energy = mse_a0
    denom = math.sqrt(max(e_energy, 0.0) * max(d_energy, 0.0)) + EPS
    alpha_mse_opt = -dot / (d_energy + EPS)
    alpha_safe_upper = -2.0 * dot / (d_energy + EPS)
    lum = luma(d)
    chroma = d - lum.repeat(1, 3, 1, 1)
    low_energy, high_energy = low_high_energy(d)
    return {
        "mse_A0": mse_a0,
        "mse_teacher": mse_teacher,
        "psnr_A0_recomputed": psnr_from_mse(mse_a0),
        "psnr_teacher_recomputed": psnr_from_mse(mse_teacher),
        "headroom_mse": mse_a0 - mse_teacher,
        "teacher_delta_energy": d_energy,
        "teacher_delta_luma_energy": mean_sq(lum),
        "teacher_delta_chroma_energy": mean_sq(chroma),
        "teacher_delta_lowfreq_energy": low_energy,
        "teacher_delta_highfreq_energy": high_energy,
        "alignment_dot": dot,
        "alignment_cos": dot / denom,
        "alpha_mse_opt": alpha_mse_opt,
        "alpha_safe_upper": alpha_safe_upper,
        "alpha_safe_upper_pos": max(0.0, alpha_safe_upper),
        "anti_aligned": dot >= 0.0,
        "overshoot_risk": math.sqrt(d_energy) / (math.sqrt(mse_a0) + EPS),
    }


def q(values: list[float], pct: float) -> float | None:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.quantile(vals, pct))


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return float(np.mean(np.asarray(flags, dtype=np.float64)))


def corr(xs: list[float], ys: list[float | bool]) -> float | None:
    pairs = []
    for x, y in zip(xs, ys):
        xf = float(x)
        yf = float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if np.std(arr[:, 0]) <= EPS or np.std(arr[:, 1]) <= EPS:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def describe_values(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = [parse_float(r.get(key)) for r in rows]
    return {
        "mean": mean(vals),
        "p01": q(vals, 0.01),
        "p05": q(vals, 0.05),
        "p10": q(vals, 0.10),
        "p50": q(vals, 0.50),
        "p90": q(vals, 0.90),
        "p95": q(vals, 0.95),
    }


def rows_where(rows: list[dict[str, Any]], pred) -> list[dict[str, Any]]:
    return [r for r in rows if pred(r)]


def build_alpha_map(rows: list[dict[str, str]], alpha: float) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if alpha_matches(row.get("alpha"), alpha):
            out[row["image_id"]] = row
    return out


def choose_first_by_image(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out.setdefault(row["image_id"], row)
    return out


def build_atlas(args: argparse.Namespace) -> list[dict[str, Any]]:
    v237_rows = read_rows(args.v237_p0_csv)
    v238_rows = read_rows(args.v238_p0_csv)
    v239_rows = read_rows(args.v239_p0_csv)
    base = choose_first_by_image(v237_rows)
    conv_base = choose_first_by_image(v239_rows)
    w_a003125 = build_alpha_map(v238_rows, 0.03125)
    w_a0125 = build_alpha_map(v238_rows, 0.125)
    c_a0015625 = build_alpha_map(v239_rows, 0.015625)
    c_a025 = build_alpha_map(v239_rows, 0.25)

    missing = []
    atlas: list[dict[str, Any]] = []
    image_ids = sorted(base)
    if len(image_ids) != 600:
        raise ValueError(f"expected 600 base images, found {len(image_ids)}")

    for idx, image_id in enumerate(image_ids, 1):
        w_row = base[image_id]
        c_row = conv_base.get(image_id)
        if c_row is None:
            missing.append((image_id, "missing_convirl_row"))
            continue
        paths = {
            "gt": w_row["gt_path"],
            "a0": w_row["A0_full_output_path"],
            "wdmamba": w_row["WDMamba_full_output_path"],
            "convirl": c_row["ConvIRL_full_output_path"],
        }
        absent = [k for k, p in paths.items() if not p or not os.path.exists(p)]
        if absent:
            missing.append((image_id, "missing_paths:" + ",".join(absent)))
            continue

        a0 = tensor_load(paths["a0"])
        gt = image_load(paths["gt"], a0)
        wdmamba = tensor_load(paths["wdmamba"])
        convirl = tensor_load(paths["convirl"])

        w_geom = teacher_geometry(a0, gt, wdmamba)
        c_geom = teacher_geometry(a0, gt, convirl)
        w_003125_unsafe = is_unsafe_alpha_row(w_a003125.get(image_id))
        w_0125_unsafe = is_unsafe_alpha_row(w_a0125.get(image_id))
        c_0015625_unsafe = is_unsafe_alpha_row(c_a0015625.get(image_id))
        c_025_unsafe = is_unsafe_alpha_row(c_a025.get(image_id))
        shared_useful = w_0125_unsafe and c_025_unsafe
        union_useful = w_0125_unsafe or c_025_unsafe
        if shared_useful:
            overlap_class = "shared_unsafe_wdmamba0125_convirl025"
        elif w_0125_unsafe:
            overlap_class = "wdmamba0125_only"
        elif c_025_unsafe:
            overlap_class = "convirl025_only"
        else:
            overlap_class = "neither_useful_alpha_unsafe"

        row: dict[str, Any] = {
            "image_id": image_id,
            "image_name": w_row.get("image_name", ""),
            "fold_id": parse_int(w_row.get("fold_id")),
            "hardness_bucket": w_row.get("hardness_bucket", ""),
            "easy_bucket": w_row.get("easy_bucket", ""),
            "strong_reference_bucket": w_row.get("strong_reference_bucket", ""),
            "A0_psnr": parse_float(w_row.get("A0_same_context_psnr")),
            "WDMamba_psnr": parse_float(w_row.get("WDMamba_full_psnr")),
            "ConvIRL_psnr": parse_float(c_row.get("ConvIRL_full_psnr")),
            "delta_wdmamba_vs_A0": parse_float(w_row.get("WDMamba_full_psnr"))
            - parse_float(w_row.get("A0_same_context_psnr")),
            "delta_convirl_vs_A0": parse_float(c_row.get("ConvIRL_full_psnr"))
            - parse_float(c_row.get("A0_same_context_psnr")),
            "unsafe_wdmamba_alpha003125": w_003125_unsafe,
            "unsafe_wdmamba_alpha0125": w_0125_unsafe,
            "unsafe_convirl_alpha0015625": c_0015625_unsafe,
            "unsafe_convirl_alpha025": c_025_unsafe,
            "unsafe_overlap_class": overlap_class,
            "shared_useful_alpha_unsafe": shared_useful,
            "union_useful_alpha_unsafe": union_useful,
            "locked_test_touched": False,
        }
        for prefix, geom in (("wdmamba", w_geom), ("convirl", c_geom)):
            for k, v in geom.items():
                row[f"{prefix}_{k}"] = v
        atlas.append(row)
        if idx % 100 == 0:
            print(f"v240_atlas_progress {idx}/{len(image_ids)}", flush=True)

    if missing:
        raise RuntimeError(f"missing required rows/paths for {len(missing)} images: {missing[:5]}")
    return atlas


def summarize_atlas(atlas: list[dict[str, Any]], predictability: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": "COMPLETED_DIAGNOSTIC",
        "image_count": len(atlas),
        "locked_test_touched": False,
        "teachers": {},
        "unsafe_overlap": {},
        "correlations": {},
        "predictability": predictability or {},
        "interpretation_flags": {},
    }
    for teacher in ("wdmamba", "convirl"):
        prefix = f"{teacher}_"
        all_rows = atlas
        hard_rows = rows_where(atlas, lambda r: r.get("hardness_bucket") == "hard")
        easy_rows = rows_where(atlas, lambda r: r.get("easy_bucket") == "easy_top25")
        strong_rows = rows_where(atlas, lambda r: r.get("strong_reference_bucket") == "strong_reference")
        strong_or_easy = rows_where(
            atlas,
            lambda r: r.get("easy_bucket") == "easy_top25"
            or r.get("strong_reference_bucket") == "strong_reference",
        )
        summary["teachers"][teacher] = {
            "anti_aligned_rate_all": rate([bool(r[f"{prefix}anti_aligned"]) for r in all_rows]),
            "anti_aligned_rate_hard": rate([bool(r[f"{prefix}anti_aligned"]) for r in hard_rows]),
            "anti_aligned_rate_easy": rate([bool(r[f"{prefix}anti_aligned"]) for r in easy_rows]),
            "anti_aligned_rate_strong_reference": rate(
                [bool(r[f"{prefix}anti_aligned"]) for r in strong_rows]
            ),
            "alpha_safe_upper": describe_values(all_rows, f"{prefix}alpha_safe_upper"),
            "alpha_safe_upper_hard": describe_values(hard_rows, f"{prefix}alpha_safe_upper"),
            "alpha_safe_upper_easy": describe_values(easy_rows, f"{prefix}alpha_safe_upper"),
            "alpha_safe_upper_strong_reference": describe_values(
                strong_rows, f"{prefix}alpha_safe_upper"
            ),
            "alpha_safe_upper_strong_or_easy": describe_values(
                strong_or_easy, f"{prefix}alpha_safe_upper"
            ),
            "alpha_safe_upper_p05_by_fold": {
                str(fold): q(
                    [
                        parse_float(r[f"{prefix}alpha_safe_upper"])
                        for r in atlas
                        if parse_int(r.get("fold_id")) == fold
                    ],
                    0.05,
                )
                for fold in sorted({parse_int(r.get("fold_id")) for r in atlas})
            },
            "headroom_mse": describe_values(all_rows, f"{prefix}headroom_mse"),
            "overshoot_risk": describe_values(all_rows, f"{prefix}overshoot_risk"),
            "alignment_cos": describe_values(all_rows, f"{prefix}alignment_cos"),
            "teacher_delta_energy": describe_values(all_rows, f"{prefix}teacher_delta_energy"),
            "teacher_delta_lowfreq_energy": describe_values(
                all_rows, f"{prefix}teacher_delta_lowfreq_energy"
            ),
            "teacher_delta_highfreq_energy": describe_values(
                all_rows, f"{prefix}teacher_delta_highfreq_energy"
            ),
        }

    w_unsafe = [bool(r["unsafe_wdmamba_alpha0125"]) for r in atlas]
    c_unsafe = [bool(r["unsafe_convirl_alpha025"]) for r in atlas]
    w_low_unsafe = [bool(r["unsafe_wdmamba_alpha003125"]) for r in atlas]
    c_low_unsafe = [bool(r["unsafe_convirl_alpha0015625"]) for r in atlas]
    shared = [w and c for w, c in zip(w_unsafe, c_unsafe)]
    union = [w or c for w, c in zip(w_unsafe, c_unsafe)]
    low_shared = [w and c for w, c in zip(w_low_unsafe, c_low_unsafe)]
    low_union = [w or c for w, c in zip(w_low_unsafe, c_low_unsafe)]
    summary["unsafe_overlap"] = {
        "useful_alpha_pair": {
            "wdmamba_alpha": 0.125,
            "convirl_alpha": 0.25,
            "wdmamba_unsafe_rate": rate(w_unsafe),
            "convirl_unsafe_rate": rate(c_unsafe),
            "shared_unsafe_rate": rate(shared),
            "union_unsafe_rate": rate(union),
            "unsafe_jaccard_wdmamba_convirl": (
                float(sum(shared) / sum(union)) if sum(union) else 0.0
            ),
            "wdmamba_only_rate": rate([w and not c for w, c in zip(w_unsafe, c_unsafe)]),
            "convirl_only_rate": rate([c and not w for w, c in zip(w_unsafe, c_unsafe)]),
            "unsafe_overlap_easy_strong_reference_rate": rate(
                [
                    u
                    for u, r in zip(union, atlas)
                    if r.get("easy_bucket") == "easy_top25"
                    or r.get("strong_reference_bucket") == "strong_reference"
                ]
            ),
        },
        "low_alpha_pair": {
            "wdmamba_alpha": 0.03125,
            "convirl_alpha": 0.015625,
            "wdmamba_unsafe_rate": rate(w_low_unsafe),
            "convirl_unsafe_rate": rate(c_low_unsafe),
            "shared_unsafe_rate": rate(low_shared),
            "union_unsafe_rate": rate(low_union),
            "unsafe_jaccard_wdmamba_convirl": (
                float(sum(low_shared) / sum(low_union)) if sum(low_union) else 0.0
            ),
        },
    }
    for teacher, unsafe_key in (
        ("wdmamba", "unsafe_wdmamba_alpha0125"),
        ("convirl", "unsafe_convirl_alpha025"),
    ):
        unsafe = [bool(r[unsafe_key]) for r in atlas]
        summary["correlations"][teacher] = {
            "corr_A0_psnr_unsafe": corr([parse_float(r["A0_psnr"]) for r in atlas], unsafe),
            "corr_overshoot_risk_unsafe": corr(
                [parse_float(r[f"{teacher}_overshoot_risk"]) for r in atlas], unsafe
            ),
            "corr_alignment_cos_unsafe": corr(
                [parse_float(r[f"{teacher}_alignment_cos"]) for r in atlas], unsafe
            ),
            "corr_alpha_safe_upper_unsafe": corr(
                [parse_float(r[f"{teacher}_alpha_safe_upper"]) for r in atlas], unsafe
            ),
        }

    w_se_alpha_p05 = summary["teachers"]["wdmamba"]["alpha_safe_upper_strong_or_easy"]["p05"]
    c_se_alpha_p05 = summary["teachers"]["convirl"]["alpha_safe_upper_strong_or_easy"]["p05"]
    summary["interpretation_flags"] = {
        "wdmamba_no_selector_alpha_theory_closed_by_strong_or_easy_p05_lt_0p02": (
            w_se_alpha_p05 is not None and w_se_alpha_p05 < 0.02
        ),
        "convirl_no_selector_alpha_theory_closed_by_strong_or_easy_p05_lt_0p02": (
            c_se_alpha_p05 is not None and c_se_alpha_p05 < 0.02
        ),
        "useful_alpha_unsafe_overlap_nonzero": sum(shared) > 0,
        "diagnostic_only_no_v241_authorization_without_manual_review": True,
    }
    return summary


def feature_columns(feature_rows: list[dict[str, str]]) -> list[str]:
    cols = list(feature_rows[0].keys())
    chosen = []
    for col in cols:
        low = col.lower()
        if any(tok in low for tok in RUNTIME_FEATURE_EXCLUDE_TOKENS):
            continue
        numeric = [parse_float(r.get(col)) for r in feature_rows[:20]]
        if any(math.isfinite(v) for v in numeric):
            chosen.append(col)
    return chosen


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return None
    best = 0.0
    for threshold in np.unique(scores)[::-1]:
        pred = scores >= threshold
        fpr = float((pred & negatives).sum() / negatives.sum())
        if fpr <= max_fpr + 1.0e-12:
            rec = float((pred & positives).sum() / positives.sum())
            best = max(best, rec)
    return best


def oof_predictability(
    atlas: list[dict[str, Any]], feature_manifest: str | None, out_dir: Path
) -> dict[str, Any] | None:
    if not feature_manifest:
        return None
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - cloud env dependent
        return {"skipped": True, "reason": f"sklearn_import_failed:{exc}"}

    feature_rows = read_rows(feature_manifest)
    by_id = {r["image_id"]: r for r in feature_rows}
    atlas_by_id = {r["image_id"]: r for r in atlas}
    ids = [image_id for image_id in sorted(atlas_by_id) if image_id in by_id]
    cols = feature_columns([by_id[i] for i in ids])
    if not cols:
        return {"skipped": True, "reason": "no_numeric_runtime_feature_columns"}
    x = np.asarray([[parse_float(by_id[i].get(c)) for c in cols] for i in ids], dtype=np.float64)
    folds = np.asarray([parse_int(atlas_by_id[i].get("fold_id")) for i in ids], dtype=np.int64)

    labels = {
        "wdmamba_anti_aligned": np.asarray(
            [bool(atlas_by_id[i]["wdmamba_anti_aligned"]) for i in ids], dtype=np.int64
        ),
        "convirl_anti_aligned": np.asarray(
            [bool(atlas_by_id[i]["convirl_anti_aligned"]) for i in ids], dtype=np.int64
        ),
        "wdmamba_alpha_safe_upper_lt_0p02": np.asarray(
            [parse_float(atlas_by_id[i]["wdmamba_alpha_safe_upper"]) < 0.02 for i in ids],
            dtype=np.int64,
        ),
        "convirl_alpha_safe_upper_lt_0p02": np.asarray(
            [parse_float(atlas_by_id[i]["convirl_alpha_safe_upper"]) < 0.02 for i in ids],
            dtype=np.int64,
        ),
    }
    rng = np.random.default_rng(240)
    per_fold_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "skipped": False,
        "image_count": len(ids),
        "feature_count": len(cols),
        "feature_manifest": feature_manifest,
        "labels": {},
    }
    for name, y in labels.items():
        scores = np.full(len(ids), np.nan, dtype=np.float64)
        for fold in sorted(set(folds.tolist())):
            train = folds != fold
            test = folds == fold
            if len(np.unique(y[train])) < 2 or test.sum() == 0:
                continue
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=240,
                ),
            )
            clf.fit(x[train], y[train])
            fold_scores = clf.predict_proba(x[test])[:, 1]
            scores[test] = fold_scores
            fold_row = {
                "label": name,
                "fold_id": int(fold),
                "test_count": int(test.sum()),
                "test_positive_rate": float(y[test].mean()),
            }
            if len(np.unique(y[test])) == 2:
                fold_row["auroc"] = float(roc_auc_score(y[test], fold_scores))
                fold_row["auprc"] = float(average_precision_score(y[test], fold_scores))
            else:
                fold_row["auroc"] = math.nan
                fold_row["auprc"] = math.nan
            per_fold_rows.append(fold_row)

        valid = np.isfinite(scores)
        label_summary: dict[str, Any] = {
            "base_rate": float(y.mean()),
            "valid_oof_count": int(valid.sum()),
        }
        if valid.sum() and len(np.unique(y[valid])) == 2:
            label_summary["auroc"] = float(roc_auc_score(y[valid], scores[valid]))
            label_summary["auprc"] = float(average_precision_score(y[valid], scores[valid]))
            label_summary["recall_at_fpr0p05"] = recall_at_fpr(scores[valid], y[valid], 0.05)
            shuffled_ap = []
            for _ in range(20):
                shuffled = rng.permutation(y[valid])
                shuffled_ap.append(float(average_precision_score(shuffled, scores[valid])))
            label_summary["shuffle_auprc_mean"] = float(np.mean(shuffled_ap))
            label_summary["shuffle_auprc_p95"] = float(np.quantile(shuffled_ap, 0.95))
        summary["labels"][name] = label_summary

    if per_fold_rows:
        write_rows(
            out_dir / "v240_alignment_predictability_per_fold.csv",
            per_fold_rows,
            ["label", "fold_id", "test_count", "test_positive_rate", "auroc", "auprc"],
        )
    return summary


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas(args)
    atlas_fields = list(atlas[0].keys())
    write_rows(out_dir / "v240_teacher_residual_alignment_atlas_per_image.csv", atlas, atlas_fields)
    predictability = oof_predictability(atlas, args.feature_manifest, out_dir)
    summary = summarize_atlas(atlas, predictability)
    write_json(out_dir / "v240_teacher_residual_alignment_atlas_summary.json", summary)
    closeout = {
        "status": "COMPLETED_DIAGNOSTIC",
        "decision": "V240_TEACHER_RESIDUAL_ALIGNMENT_ATLAS_COMPLETE",
        "gate_pass": True,
        "locked_test_touched": False,
        "bridge_generator_authorized": False,
        "canary80_authorized": False,
        "v241_authorized": False,
        "manual_review_required_for_v241": True,
        "summary_path": str(out_dir / "v240_teacher_residual_alignment_atlas_summary.json"),
        "cloud_only_raw_outputs": [
            str(out_dir / "v240_teacher_residual_alignment_atlas_per_image.csv"),
        ],
    }
    if predictability and not predictability.get("skipped"):
        closeout["cloud_only_raw_outputs"].append(
            str(out_dir / "v240_alignment_predictability_per_fold.csv")
        )
    write_json(out_dir / "v240_closeout.json", closeout)
    print("V240_ATLAS_SUMMARY")
    print(json.dumps(to_jsonable(summary["interpretation_flags"]), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--v237-p0-csv", required=True)
    parser.add_argument("--v238-p0-csv", required=True)
    parser.add_argument("--v239-p0-csv", required=True)
    parser.add_argument("--feature-manifest")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
