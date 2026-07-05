#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from nopost_v227_ilfrb_acs_diagnostics import (  # noqa: E402
    SEVERE,
    STAGE_SETS,
    STRONG_REG,
    average_precision,
    build_models,
    cvar,
    finite,
    forward_cache,
    git,
    haar_dwt,
    image_tensor,
    load_samples,
    mean,
    pad_to,
    percentile,
    sha256_file,
    source_scan,
    std,
    tensor_psnr,
    write_csv,
    write_json,
    write_text,
)
from nopost_v228_action_bank_stratification_audit import aggregate_tensors  # noqa: E402
from nopost_v229_safe_oof_action_bank_calibration import (  # noqa: E402
    action_family,
    add_row,
    append_status,
    attach_sample_buckets,
    build_oracle_delta_bank,
    deployable_actions_for_variant,
    json_clean,
    run_variant_replay,
    stage_allowed,
)
from nopost_v230_compatibility_gated_oof_table_policy import (  # noqa: E402
    BUCKETS,
    LCB_POLICY,
    SAFETY_VARIANT,
    compat_bin,
    fast_roc_auc,
    row_for_policy,
    run_lcb_policy,
    safe_rate,
    select_restricted_oracle,
    summarize_selected,
)


ROUTE_ID = "haze4k_v2_31_nopost_action_value_identifiability_audit_20260705"
BRANCH = "codex/haze4k-v2-31-nopost-action-value-identifiability-audit"
PARENT_BRANCH = "codex/haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy"
ACTION_META_KEYS = [
    "action_strength",
    "bucket_distance",
    "action_family_code",
    "stage_set_code",
    "source_bucket_code",
    "target_bucket_code",
    "strength_bin_code",
    "compat_bin_code",
]
LEGACY_KEYS = [
    "a0_psnr",
    "input_luma_mean",
    "input_luma_std",
    "input_low_mean",
    "input_low_std",
    "delta_rms_to_target_ll_rms",
    "delta_absmax_to_target_ll_absmax",
    "stagewise_alignment_mean",
    "stagewise_alignment_min",
] + ACTION_META_KEYS
PHYSICS_KEYS = [
    "hazy_dark_mean",
    "hazy_dark_p05",
    "hazy_dark_p50",
    "hazy_bright_mean",
    "hazy_atmospheric_light_proxy",
    "hazy_saturation_mean",
    "hazy_saturation_std",
    "hazy_contrast_mean",
    "hazy_local_contrast_p10",
    "hazy_haze_density_proxy",
    "hazy_color_cast_rg",
    "hazy_color_cast_bg",
    "hazy_gradient_energy",
    "hazy_edge_density",
    "a0_dark_mean",
    "a0_contrast_mean",
    "a0_color_cast_rg",
    "a0_color_cast_bg",
    "a0_output_dark_channel_residual_proxy",
    "a0_output_contrast_deficit",
    "hazy_output_delta_rms",
    "hazy_output_lowfreq_mismatch",
    "bottleneck_ll_rms",
    "early_ll_rms",
    "mid_ll_rms",
    "final_ll_rms",
    "bottleneck_feature_spatial_var",
    "mid_feature_spatial_var",
    "final_feature_spatial_var",
    "feature_anisotropy_mean",
    "activation_sharpness_proxy",
]
FEATURE_SETS = {
    "legacy_bucket_stage_strength": LEGACY_KEYS,
    "target_physics_frequency": PHYSICS_KEYS,
    "action_metadata_only": ACTION_META_KEYS,
    "combined_physics_frequency_features": LEGACY_KEYS + PHYSICS_KEYS,
}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def quantile_tensor(x: torch.Tensor, q: float) -> float:
    flat = x.detach().float().flatten()
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, q).detach().cpu())


def rgb_feature_stats(prefix: str, img: torch.Tensor) -> dict[str, float]:
    x = img.detach().float().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    r, g, b = x[0], x[1], x[2]
    mx = x.max(dim=0).values
    mn = x.min(dim=0).values
    gray = x.mean(dim=0)
    dark = mn
    sat = (mx - mn) / mx.clamp_min(1e-6)
    local = F.avg_pool2d(gray[None, None], kernel_size=15, stride=1, padding=7)[0, 0]
    contrast = (gray - local).abs()
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    grad_energy = 0.5 * (gx.abs().mean() + gy.abs().mean())
    edge_thr = grad_energy + gray.std(unbiased=False)
    edge_density = 0.5 * ((gx.abs() > edge_thr).float().mean() + (gy.abs() > edge_thr).float().mean())
    low = F.avg_pool2d(gray[None, None], kernel_size=16, stride=16, ceil_mode=True)[0, 0]
    color_mean = x.flatten(1).mean(dim=1)
    rgb_mean = color_mean.mean().clamp_min(1e-6)
    return {
        f"{prefix}_dark_mean": float(dark.mean()),
        f"{prefix}_dark_p05": quantile_tensor(dark, 0.05),
        f"{prefix}_dark_p50": quantile_tensor(dark, 0.50),
        f"{prefix}_bright_mean": float(mx.mean()),
        f"{prefix}_atmospheric_light_proxy": quantile_tensor(mx, 0.99),
        f"{prefix}_saturation_mean": float(sat.mean()),
        f"{prefix}_saturation_std": float(sat.std(unbiased=False)),
        f"{prefix}_contrast_mean": float(contrast.mean()),
        f"{prefix}_local_contrast_p10": quantile_tensor(contrast, 0.10),
        f"{prefix}_haze_density_proxy": float(dark.mean() + 0.5 * (1.0 - gray.std(unbiased=False))),
        f"{prefix}_color_cast_rg": float(color_mean[0] / color_mean[1].clamp_min(1e-6)),
        f"{prefix}_color_cast_bg": float(color_mean[2] / color_mean[1].clamp_min(1e-6)),
        f"{prefix}_color_mean_ratio_r": float(color_mean[0] / rgb_mean),
        f"{prefix}_color_mean_ratio_g": float(color_mean[1] / rgb_mean),
        f"{prefix}_color_mean_ratio_b": float(color_mean[2] / rgb_mean),
        f"{prefix}_gradient_energy": float(grad_energy),
        f"{prefix}_edge_density": float(edge_density),
        f"{prefix}_low_mean": float(low.mean()),
        f"{prefix}_low_std": float(low.std(unbiased=False)),
    }


def internal_feature_stats(cache: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    anis = []
    sharp = []
    for stage in ("bottleneck", "early", "mid", "final"):
        feat = cache[stage].detach().float()
        ll, lh, hl, hh, _h, _w = haar_dwt(feat)
        ll_rms = torch.sqrt(torch.mean(ll.float() ** 2)).item()
        lh_rms = torch.sqrt(torch.mean(lh.float() ** 2)).item()
        hl_rms = torch.sqrt(torch.mean(hl.float() ** 2)).item()
        hh_rms = torch.sqrt(torch.mean(hh.float() ** 2)).item()
        spatial = feat.flatten(2).var(dim=2, unbiased=False).mean().item()
        channel_mean = feat.flatten(2).abs().mean(dim=2)
        channel_std = feat.flatten(2).std(dim=2, unbiased=False)
        sharp.append(float((channel_std / channel_mean.clamp_min(1e-6)).mean()))
        anis.append(abs(lh_rms - hl_rms) / max(lh_rms + hl_rms, 1e-8))
        out.update(
            {
                f"{stage}_feature_norm": float(torch.sqrt(torch.mean(feat ** 2))),
                f"{stage}_feature_spatial_var": spatial,
                f"{stage}_ll_rms": ll_rms,
                f"{stage}_lh_rms": lh_rms,
                f"{stage}_hl_rms": hl_rms,
                f"{stage}_hh_rms": hh_rms,
                f"{stage}_hf_to_ll_ratio": (lh_rms + hl_rms + hh_rms) / max(ll_rms, 1e-8),
            }
        )
    out["feature_anisotropy_mean"] = mean(anis)
    out["activation_sharpness_proxy"] = mean(sharp)
    return out


def build_target_features(args: argparse.Namespace, device: torch.device) -> dict[str, dict[str, float]]:
    a0, _route, _partial = build_models(args, device)
    features: dict[str, dict[str, float]] = {}
    for idx, sample in enumerate(load_samples(args), start=1):
        x0 = image_tensor(sample.input_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        pred = cache["out"][:, :, :h, :w].clamp(0, 1)
        hazy = x0[:, :, :h, :w]
        row: dict[str, float] = {"target_fold": float(sample.fold)}
        row.update(rgb_feature_stats("hazy", hazy.cpu()))
        row.update(rgb_feature_stats("a0", pred.cpu()))
        delta = (pred - hazy).detach().float()
        low_hazy = F.avg_pool2d(hazy.detach().float(), kernel_size=16, stride=16, ceil_mode=True)
        low_pred = F.avg_pool2d(pred.detach().float(), kernel_size=16, stride=16, ceil_mode=True)
        row.update(
            {
                "a0_output_dark_channel_residual_proxy": row["a0_dark_mean"],
                "a0_output_contrast_deficit": row["hazy_contrast_mean"] - row["a0_contrast_mean"],
                "hazy_output_delta_mean": float(delta.mean()),
                "hazy_output_delta_std": float(delta.std(unbiased=False)),
                "hazy_output_delta_rms": float(torch.sqrt(torch.mean(delta ** 2))),
                "hazy_output_lowfreq_mismatch": float(torch.sqrt(torch.mean((low_pred - low_hazy) ** 2))),
            }
        )
        row.update(internal_feature_stats(cache))
        features[sample.name] = row
        if idx % args.print_freq == 0:
            print(f"V231_FEATURE_PROGRESS {idx}", flush=True)
    return features


def phase_p0(args: argparse.Namespace) -> dict[str, Any]:
    scan = source_scan()
    report = {
        "phase": "p0_arch_contract_delta",
        "route": "v2.31",
        "parent_route": "v2.30",
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_commit": args.parent_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_delta": "none_from_v2_30_ilfrb_acs",
        "runtime_contract": "forward(self, x)",
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
        "forbidden_symbol_hits": scan["hit_count"],
        "decision": "P0_PASS_ARCH_CONTRACT_DELTA_AUDIT" if scan["hit_count"] == 0 else "P0_FAIL_SOURCE_CONTRACT",
        "pass": scan["hit_count"] == 0,
    }
    lines = [
        "# v2.31 P0 Architecture Contract Delta",
        "",
        f"branch: `{report['branch']}`",
        f"commit: `{report['commit']}`",
        f"parent_branch: `{PARENT_BRANCH}`",
        f"parent_commit: `{args.parent_commit}`",
        f"checkpoint: `{args.checkpoint}`",
        f"checkpoint_sha256: `{report['checkpoint_sha256']}`",
        "",
        "v2.31 does not add runtime model structure. It reuses the v2.30",
        "NoPost ILFRB-ACS audit stack and adds only train-derived target-only",
        "action-value identifiability diagnostics.",
        "",
        "runtime_forward_contract: `forward(self, x)`",
        "teacher_or_expert_forward_input: `false`",
        "rgb_output_output_residual: `false`",
        "learned_rgb_post_output_correction: `false`",
        "p2b_selector_probe_launched: `false`",
        "training_launched: `false`",
        "locked_test_touched: `false`",
        f"forbidden_symbol_hits: `{scan['hit_count']}`",
        f"decision: `{report['decision']}`",
    ]
    write_text(args.out_dir / "v231_p0_arch_contract_delta.md", "\n".join(lines))
    return report


def strength_bin(row: dict[str, Any]) -> str:
    fam = action_family(str(row.get("action_name", "")))
    if fam == "noop":
        return "noop"
    val = finite_float(row.get("action_strength"))
    if val <= 0.20:
        return "tiny"
    if val <= 0.40:
        return "mild"
    if val <= 0.70:
        return "medium"
    if val <= 1.05:
        return "strong_low"
    return "strong_high"


def code_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = {
        "action_family_code": [action_family(str(r.get("action_name", ""))) for r in rows],
        "stage_set_code": [str(r.get("stage_set", "")) for r in rows],
        "source_bucket_code": [str(r.get("prototype_source_bucket", "")) for r in rows],
        "target_bucket_code": [str(r.get("difficulty_bucket", "")) for r in rows],
        "strength_bin_code": [strength_bin(r) for r in rows],
        "compat_bin_code": [compat_bin(r) for r in rows],
    }
    return {key: {val: float(idx) for idx, val in enumerate(sorted(set(vals)))} for key, vals in fields.items()}


def enrich_rows(rows: list[dict[str, Any]], target_features: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    maps = code_maps(rows)
    out = []
    for row in rows:
        r = dict(row)
        r.update(target_features.get(str(row["sample_name"]), {}))
        r["action_family_code"] = maps["action_family_code"].get(action_family(str(row.get("action_name", ""))), 0.0)
        r["stage_set_code"] = maps["stage_set_code"].get(str(row.get("stage_set", "")), 0.0)
        r["source_bucket_code"] = maps["source_bucket_code"].get(str(row.get("prototype_source_bucket", "")), 0.0)
        r["target_bucket_code"] = maps["target_bucket_code"].get(str(row.get("difficulty_bucket", "")), 0.0)
        r["strength_bin_code"] = maps["strength_bin_code"].get(strength_bin(row), 0.0)
        r["compat_bin_code"] = maps["compat_bin_code"].get(compat_bin(row), 0.0)
        out.append(r)
    return out


def safe_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if int(r.get("deployable_candidate", 0)) == 1 and row_for_policy(r, LCB_POLICY)]


def group_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sample_name"])].append(row)
    return groups


def best_safe_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: max(group, key=lambda r: finite_float(r.get("dpsnr"))) for name, group in group_by_sample(rows).items()}


def feature_values(rows: list[dict[str, Any]], keys: list[str]) -> list[list[float]]:
    vals = []
    for row in rows:
        vals.append([finite_float(row.get(key)) for key in keys])
    return vals


def ridge_oof_scores(rows: list[dict[str, Any]], labels: list[float], keys: list[str]) -> list[float]:
    folds = [int(r["target_fold"]) for r in rows]
    xs = torch.tensor(feature_values(rows, keys), dtype=torch.float64)
    ys = torch.tensor(labels, dtype=torch.float64)
    out = [0.0 for _ in rows]
    for fold in sorted(set(folds)):
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        valid_idx = [i for i, f in enumerate(folds) if f == fold]
        if not train_idx or not valid_idx:
            continue
        xtr = xs[train_idx]
        xva = xs[valid_idx]
        ytr = ys[train_idx]
        mu = xtr.mean(dim=0, keepdim=True)
        sig = xtr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        xtr = (xtr - mu) / sig
        xva = (xva - mu) / sig
        xtr = torch.cat([torch.ones((xtr.shape[0], 1), dtype=xtr.dtype), xtr], dim=1)
        xva = torch.cat([torch.ones((xva.shape[0], 1), dtype=xva.dtype), xva], dim=1)
        reg = torch.eye(xtr.shape[1], dtype=xtr.dtype) * 1e-3
        reg[0, 0] = 0.0
        try:
            coef = torch.linalg.solve(xtr.T @ xtr + reg, xtr.T @ ytr)
        except RuntimeError:
            coef = torch.linalg.pinv(xtr.T @ xtr + reg) @ xtr.T @ ytr
        preds = xva @ coef
        for idx, pred in zip(valid_idx, preds.tolist()):
            out[idx] = float(pred)
    return out


def knn_oof_scores(rows: list[dict[str, Any]], labels: list[float], keys: list[str], k: int = 7) -> list[float]:
    folds = [int(r["target_fold"]) for r in rows]
    xs = torch.tensor(feature_values(rows, keys), dtype=torch.float64)
    ys = torch.tensor(labels, dtype=torch.float64)
    out = [0.0 for _ in rows]
    for fold in sorted(set(folds)):
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        valid_idx = [i for i, f in enumerate(folds) if f == fold]
        if not train_idx or not valid_idx:
            continue
        xtr = xs[train_idx]
        xva = xs[valid_idx]
        mu = xtr.mean(dim=0, keepdim=True)
        sig = xtr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        xtr = (xtr - mu) / sig
        xva = (xva - mu) / sig
        dists = torch.cdist(xva, xtr)
        kk = min(k, len(train_idx))
        near = dists.topk(kk, largest=False).indices
        pred = ys[train_idx][near].mean(dim=1)
        for idx, val in zip(valid_idx, pred.tolist()):
            out[idx] = float(val)
    return out


def stump_oof_scores(rows: list[dict[str, Any]], labels: list[float], keys: list[str]) -> list[float]:
    folds = [int(r["target_fold"]) for r in rows]
    xs = feature_values(rows, keys)
    out = [0.0 for _ in rows]
    global_mean = mean([float(v) for v in labels])
    for fold in sorted(set(folds)):
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        valid_idx = [i for i, f in enumerate(folds) if f == fold]
        best_j = 0
        best_gap = -1.0
        best_thr = 0.0
        best_low = global_mean
        best_high = global_mean
        for j in range(len(keys)):
            vals = [xs[i][j] for i in train_idx]
            thr = percentile(vals, 50)
            low = [labels[i] for i in train_idx if xs[i][j] <= thr]
            high = [labels[i] for i in train_idx if xs[i][j] > thr]
            gap = abs(mean(high) - mean(low)) if low and high else -1.0
            if gap > best_gap:
                best_j = j
                best_gap = gap
                best_thr = thr
                best_low = mean(low) if low else global_mean
                best_high = mean(high) if high else global_mean
        for i in valid_idx:
            out[i] = best_high if xs[i][best_j] > best_thr else best_low
    return out


def group_mean_oof_scores(rows: list[dict[str, Any]], labels: list[float]) -> list[float]:
    folds = [int(r["target_fold"]) for r in rows]
    out = [0.0 for _ in rows]
    for fold in sorted(set(folds)):
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        valid_idx = [i for i, f in enumerate(folds) if f == fold]
        table: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for i in train_idx:
            r = rows[i]
            key = (str(r["difficulty_bucket"]), action_family(str(r["action_name"])), strength_bin(r))
            table[key].append(labels[i])
        fallback = mean([labels[i] for i in train_idx])
        for i in valid_idx:
            r = rows[i]
            key = (str(r["difficulty_bucket"]), action_family(str(r["action_name"])), strength_bin(r))
            out[i] = mean(table[key]) if key in table else fallback
    return out


def binary_metrics(rows: list[dict[str, Any]], scores: list[float], labels: list[int], bucket: str) -> dict[str, Any]:
    idx = [i for i, r in enumerate(rows) if bucket == "all" or r["difficulty_bucket"] == bucket]
    sc = [scores[i] for i in idx]
    ys = [labels[i] for i in idx]
    folds = sorted({int(rows[i]["target_fold"]) for i in idx})
    fold_aucs = []
    for fold in folds:
        j = [i for i in idx if int(rows[i]["target_fold"]) == fold]
        fold_aucs.append(fast_roc_auc([scores[i] for i in j], [labels[i] for i in j]))
    return {
        "sample_count": len(idx),
        "positive_rate": sum(ys) / len(ys) if ys else float("nan"),
        "auroc": fast_roc_auc(sc, ys),
        "auprc": average_precision(sc, ys),
        "fold_std": std([v for v in fold_aucs if math.isfinite(v)]),
    }


def sample_label_rows(rows: list[dict[str, Any]], best: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, group in group_by_sample(rows).items():
        base = next((r for r in group if r["action_name"] == "noop"), group[0])
        b = best[name]
        fam = action_family(str(b["action_name"]))
        r = dict(base)
        r["sample_should_noop"] = int(fam == "noop" or finite_float(b["dpsnr"]) <= 0.05)
        r["hard_should_strong"] = int(r["difficulty_bucket"] == "hard_bottom25" and fam in ("medium", "strong"))
        r["easy_should_mild_not_noop"] = int(r["difficulty_bucket"] == "easy_top25" and fam == "mild" and finite_float(b["dpsnr"]) > 0.05)
        out.append(r)
    return out


def write_feature_separability(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safe_rows = safe_candidate_rows(rows)
    best = best_safe_by_sample(safe_rows)
    out = []
    action_labels: dict[str, Callable[[dict[str, Any]], int]] = {
        "is_useful_gt_0p10": lambda r: int(finite_float(r["dpsnr"]) >= 0.10),
        "is_useful_gt_0p30": lambda r: int(finite_float(r["dpsnr"]) >= 0.30),
        "is_best_safe_action_family": lambda r: int(action_family(str(r["action_name"])) == action_family(str(best[str(r["sample_name"])]["action_name"]))),
        "is_best_stage_set": lambda r: int(str(r["stage_set"]) == str(best[str(r["sample_name"])]["stage_set"])),
        "is_best_strength_bin": lambda r: int(strength_bin(r) == strength_bin(best[str(r["sample_name"])])),
    }
    for feature_set, keys in FEATURE_SETS.items():
        for label_name, fn in action_labels.items():
            labels = [fn(r) for r in safe_rows]
            scores = ridge_oof_scores(safe_rows, [float(y) for y in labels], keys)
            for bucket in ("all",) + BUCKETS:
                row = binary_metrics(safe_rows, scores, labels, bucket)
                row.update({"unit": "action", "feature_set": feature_set, "label": label_name, "target_bucket": bucket, "model": "fold_out_linear_ridge"})
                out.append(row)

    samples = sample_label_rows(safe_rows, best)
    sample_labels = {
        "should_noop": lambda r: int(r["sample_should_noop"]),
        "hard_should_strong": lambda r: int(r["hard_should_strong"]),
        "easy_should_mild_not_noop": lambda r: int(r["easy_should_mild_not_noop"]),
    }
    for feature_set, keys in FEATURE_SETS.items():
        sample_keys = [key for key in keys if key not in ACTION_META_KEYS]
        for label_name, fn in sample_labels.items():
            labels = [fn(r) for r in samples]
            scores = ridge_oof_scores(samples, [float(y) for y in labels], sample_keys)
            for bucket in ("all",) + BUCKETS:
                row = binary_metrics(samples, scores, labels, bucket)
                row.update({"unit": "sample", "feature_set": feature_set, "label": label_name, "target_bucket": bucket, "model": "fold_out_linear_ridge"})
                out.append(row)

    write_csv(args.out_dir / "v231_p2a_action_value_feature_separability.csv", out)
    gate = {
        "combined_useful_gt_0p30_all_auroc": metric_lookup(out, "combined_physics_frequency_features", "is_useful_gt_0p30", "all"),
        "combined_useful_gt_0p30_hard_auroc": metric_lookup(out, "combined_physics_frequency_features", "is_useful_gt_0p30", "hard_bottom25"),
        "combined_should_noop_easy_auroc": metric_lookup(out, "combined_physics_frequency_features", "should_noop", "easy_top25"),
        "combined_useful_gt_0p30_all_fold_std": metric_lookup(out, "combined_physics_frequency_features", "is_useful_gt_0p30", "all", "fold_std"),
    }
    gate["pass"] = bool(
        finite_float(gate["combined_useful_gt_0p30_all_auroc"]) >= 0.70
        and finite_float(gate["combined_useful_gt_0p30_hard_auroc"]) >= 0.70
        and finite_float(gate["combined_should_noop_easy_auroc"]) >= 0.80
        and finite_float(gate["combined_useful_gt_0p30_all_fold_std"], 99.0) <= 0.07
    )
    return out, gate


def metric_lookup(rows: list[dict[str, Any]], feature_set: str, label: str, bucket: str, key: str = "auroc") -> float:
    for row in rows:
        if row.get("feature_set") == feature_set and row.get("label") == label and row.get("target_bucket") == bucket:
            return finite_float(row.get(key), float("nan"))
    return float("nan")


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("action_name"), row.get("stage_set"), row.get("prototype_source_bucket"), finite_float(row.get("action_strength")))


def select_with_scores(rows: list[dict[str, Any]], scores: list[float], method: str) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[str(row["sample_name"])].append((row, score))
    selected = []
    for _name, group in groups.items():
        chosen, score = max(group, key=lambda item: item[1])
        selected.append({**chosen, "policy_name": method, "selection_reason": f"pred_score={score:.6f}", "pred_score": score})
    return selected


def ranking_metrics(rows: list[dict[str, Any]], scores: list[float], selected: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_sig = {str(r["sample_name"]): signature(r) for r in oracle}
    top1 = 0
    top3 = 0
    ndcgs = []
    for name, group in group_by_sample(rows).items():
        scored = sorted([(r, scores[i]) for i, r in enumerate(rows) if str(r["sample_name"]) == name], key=lambda item: item[1], reverse=True)
        actual_sorted = sorted(group, key=lambda r: finite_float(r["dpsnr"]), reverse=True)
        ideal = sum(max(0.0, finite_float(r["dpsnr"])) / math.log2(rank + 2) for rank, r in enumerate(actual_sorted[:3]))
        got = sum(max(0.0, finite_float(r["dpsnr"])) / math.log2(rank + 2) for rank, (r, _s) in enumerate(scored[:3]))
        ndcgs.append(got / ideal if ideal > 1e-12 else 1.0)
        if scored and signature(scored[0][0]) == oracle_sig.get(name):
            top1 += 1
        if any(signature(r) == oracle_sig.get(name) for r, _s in scored[:3]):
            top3 += 1
    s = summarize_selected(selected)
    return {
        "top1_action_hit_rate": top1 / len(oracle) if oracle else float("nan"),
        "top3_action_hit_rate": top3 / len(oracle) if oracle else float("nan"),
        "NDCG@3": mean(ndcgs),
        "predicted_policy_mean": s["mean"],
        "predicted_policy_hard": s["hard"],
        "predicted_policy_easy": s["easy"],
        "p05": s["p05"],
        "CVaR5": s["cvar5"],
        "severe": s["severe_rate"],
    }


def score_methods(rows: list[dict[str, Any]], seed: int) -> dict[str, list[float]]:
    labels = [finite_float(r["dpsnr"]) for r in rows]
    rng = random.Random(seed)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    return {
        "linear_ranker": ridge_oof_scores(rows, labels, FEATURE_SETS["combined_physics_frequency_features"]),
        "shallow_tree_median_bins": stump_oof_scores(rows, labels, FEATURE_SETS["combined_physics_frequency_features"]),
        "kNN_nonparametric": knn_oof_scores(rows, labels, FEATURE_SETS["combined_physics_frequency_features"]),
        "bucket_only_baseline": group_mean_oof_scores(rows, labels),
        "shuffled_label_control": ridge_oof_scores(rows, shuffled, FEATURE_SETS["combined_physics_frequency_features"]),
    }


def write_ranking_upper_bound(rows: list[dict[str, Any]], oracle: list[dict[str, Any]], table_selected: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    safe_rows = safe_candidate_rows(rows)
    method_scores = score_methods(safe_rows, args.seed)
    oracle_summary = summarize_selected(oracle)
    table_summary = summarize_selected(table_selected)
    reports = []
    selected_by_method: dict[str, list[dict[str, Any]]] = {}
    for method, scores in method_scores.items():
        selected = select_with_scores(safe_rows, scores, method)
        selected_by_method[method] = selected
        row = ranking_metrics(safe_rows, scores, selected, oracle)
        row["model"] = method
        row["safe_set_to_policy_gap"] = oracle_summary["mean"] - row["predicted_policy_mean"]
        row["real_vs_shuffled_gap"] = row["predicted_policy_mean"] - summarize_selected(selected_by_method.get("shuffled_label_control", selected))["mean"]
        reports.append(row)
    shuffled_mean = next((r["predicted_policy_mean"] for r in reports if r["model"] == "shuffled_label_control"), float("nan"))
    for row in reports:
        row["real_vs_shuffled_gap"] = row["predicted_policy_mean"] - shuffled_mean
    best = max([r for r in reports if r["model"] != "shuffled_label_control"], key=lambda r: finite_float(r["predicted_policy_mean"]))
    gate = {
        "best_model": best["model"],
        "best_mean": best["predicted_policy_mean"],
        "best_hard": best["predicted_policy_hard"],
        "best_easy": best["predicted_policy_easy"],
        "best_severe": best["severe"],
        "best_gap": best["safe_set_to_policy_gap"],
        "pass": bool(
            finite_float(best["predicted_policy_mean"]) >= 0.30
            and finite_float(best["predicted_policy_hard"]) >= 0.60
            and finite_float(best["severe"], 1.0) <= 0.035
            and finite_float(best["safe_set_to_policy_gap"], 99.0) <= 0.30
        ),
    }
    lines = [
        "# v2.31 Safe-Set Ranking Upper Bound",
        "",
        "All models are nested fold-out diagnostics using train-derived rows only.",
        "Features exclude target GT and locked-test data. Labels are safe-set action",
        "dPSNR for route diagnosis, not deployment input.",
        "",
        f"safe_set_oracle_mean: `{oracle_summary['mean']:.6f}`",
        f"safe_set_oracle_hard: `{oracle_summary['hard']:.6f}`",
        f"v2.30_table_mean: `{table_summary['mean']:.6f}`",
        "",
        "| model | top1 | top3 | NDCG@3 | mean | hard | easy | p05 | CVaR5 | severe | gap | real-vs-shuffled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in reports:
        lines.append(
            f"| {row['model']} | {row['top1_action_hit_rate']:.6f} | {row['top3_action_hit_rate']:.6f} | {row['NDCG@3']:.6f} | "
            f"{row['predicted_policy_mean']:.6f} | {row['predicted_policy_hard']:.6f} | {row['predicted_policy_easy']:.6f} | "
            f"{row['p05']:.6f} | {row['CVaR5']:.6f} | {row['severe']:.6f} | {row['safe_set_to_policy_gap']:.6f} | {row['real_vs_shuffled_gap']:.6f} |"
        )
    lines.extend(["", f"ranking_gate_pass: `{gate['pass']}`", f"best_model: `{gate['best_model']}`"])
    write_text(args.out_dir / "v231_p2a_safe_set_ranking_upper_bound.md", "\n".join(lines))
    return reports, gate, selected_by_method[str(gate["best_model"])]


def assign_physics_clusters(target_features: dict[str, dict[str, float]]) -> dict[str, int]:
    names = sorted(target_features)
    signals = []
    for name in names:
        f = target_features[name]
        signals.append(
            finite_float(f.get("hazy_dark_mean"))
            + finite_float(f.get("hazy_haze_density_proxy"))
            + finite_float(f.get("hazy_output_lowfreq_mismatch"))
            - 0.5 * finite_float(f.get("hazy_saturation_mean"))
            + 0.25 * finite_float(f.get("mid_ll_rms"))
        )
    q1 = percentile(signals, 33.333)
    q2 = percentile(signals, 66.667)
    clusters = {}
    for name, val in zip(names, signals):
        clusters[name] = 0 if val <= q1 else 1 if val <= q2 else 2
    return clusters


def make_filtered_prototype(
    sample_rows: list[dict[str, Any]],
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]],
    target_fold: int,
    stage_set: str,
    aggregate: str,
    pred: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, torch.Tensor] | None, int]:
    source_names = [
        str(row["sample_name"])
        for row in sample_rows
        if int(row["target_fold"]) != target_fold and pred(row) and stage_set in delta_bank[str(row["sample_name"])]
    ]
    if not source_names:
        return None, 0
    proto: dict[str, torch.Tensor] = {}
    for stage in STAGE_SETS[stage_set]:
        values = [delta_bank[name][stage_set][stage] for name in source_names if stage in delta_bank[name][stage_set]]
        if not values:
            return None, 0
        proto[stage] = aggregate_tensors(values, aggregate)
    return proto, len(source_names)


def cluster_replay(
    args: argparse.Namespace,
    sample_rows: list[dict[str, Any]],
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]],
    target_features: dict[str, dict[str, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    clusters = assign_physics_clusters(target_features)
    for row in sample_rows:
        row["physics_cluster"] = clusters[str(row["sample_name"])]
    a0, _route, _partial = build_models(args, device)
    selected = {str(row["sample_name"]): row for row in sample_rows}
    samples = [sample for sample in load_samples(args) if sample.name in selected]
    stage_sets = [item for item in args.stage_sets.split(",") if item]
    rows: list[dict[str, Any]] = []
    strategies = {
        "same_cluster_only": lambda meta, src: int(src["physics_cluster"]) == int(meta["physics_cluster"]),
        "adjacent_cluster_only": lambda meta, src: abs(int(src["physics_cluster"]) - int(meta["physics_cluster"])) <= 1,
        "same_bucket_same_cluster": lambda meta, src: src["difficulty_bucket"] == meta["difficulty_bucket"] and int(src["physics_cluster"]) == int(meta["physics_cluster"]),
        "same_bucket_cross_cluster": lambda meta, src: src["difficulty_bucket"] == meta["difficulty_bucket"] and int(src["physics_cluster"]) != int(meta["physics_cluster"]),
        "physics_distance_threshold": lambda meta, src: abs(int(src["physics_cluster"]) - int(meta["physics_cluster"])) <= 1,
    }
    for idx, sample in enumerate(samples, start=1):
        meta = selected[sample.name]
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        for strategy, pred_fn in strategies.items():
            noop_base = {
                "sample_name": sample.name,
                "target_fold": int(meta["target_fold"]),
                "source_fold": f"not_{int(meta['target_fold'])}",
                "a0_psnr": float(meta["a0_psnr"]),
                "difficulty_bucket": meta["difficulty_bucket"],
                "a0_psnr_quartile": meta["a0_psnr_quartile"],
                "lowfreq_std_quartile": meta["lowfreq_std_quartile"],
                "input_luma_mean": meta["input_luma_mean"],
                "input_luma_std": meta["input_luma_std"],
                "input_low_mean": meta["input_low_mean"],
                "input_low_std": meta["input_low_std"],
                "prototype_id": "noop",
                "prototype_source_bucket": "noop",
                "stage_set": "noop",
                "prototype_source_count": 0,
                "prototype_aggregate": args.prototype_aggregate,
                "bucket_distance": 0,
                "physics_cluster": int(meta["physics_cluster"]),
                "physics_cluster_strategy": strategy,
            }
            add_row(rows, a0, cache, gt, h, w, noop_base, {}, SAFETY_VARIANT, "noop", 0.0, True, False, device)
            for stage_set in stage_sets:
                proto, source_count = make_filtered_prototype(
                    sample_rows,
                    delta_bank,
                    int(meta["target_fold"]),
                    stage_set,
                    args.prototype_aggregate,
                    lambda src, m=meta, fn=pred_fn: fn(m, src),
                )
                if proto is None:
                    continue
                base = {
                    **noop_base,
                    "prototype_id": f"physics_{strategy}_fold{int(meta['target_fold'])}_{stage_set}",
                    "prototype_source_bucket": strategy,
                    "stage_set": stage_set,
                    "prototype_source_count": source_count,
                    "bucket_distance": 0 if strategy.startswith("same") else 1,
                }
                for action_name, strength in deployable_actions_for_variant(SAFETY_VARIANT, str(meta["difficulty_bucket"])):
                    if stage_allowed(SAFETY_VARIANT, stage_set, str(meta["difficulty_bucket"]), action_name):
                        add_row(rows, a0, cache, gt, h, w, base, proto, SAFETY_VARIANT, action_name, strength, True, False, device)
        if idx % args.print_freq == 0:
            print(f"V231_CLUSTER_REPLAY_PROGRESS {idx}/{len(samples)} rows={len(rows)}", flush=True)
    return enrich_rows(rows, target_features)


def write_cluster_report(cluster_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    for strategy in sorted({str(r["physics_cluster_strategy"]) for r in cluster_rows}):
        subset = [r for r in cluster_rows if r["physics_cluster_strategy"] == strategy]
        oracle = list(best_safe_by_sample(subset).values())
        scores = ridge_oof_scores(subset, [finite_float(r["dpsnr"]) for r in subset], FEATURE_SETS["combined_physics_frequency_features"])
        selected = select_with_scores(subset, scores, f"cluster_{strategy}_linear_ranker")
        o = summarize_selected(oracle)
        p = summarize_selected(selected)
        out.append(
            {
                "cluster_policy": strategy,
                "candidate_rows": len(subset),
                "mean_source_count": mean([finite_float(r.get("prototype_source_count")) for r in subset if r["action_name"] != "noop"]),
                "oracle_mean": o["mean"],
                "oracle_hard": o["hard"],
                "oracle_easy": o["easy"],
                "oracle_p05": o["p05"],
                "oracle_cvar5": o["cvar5"],
                "oracle_severe": o["severe_rate"],
                "ranker_mean": p["mean"],
                "ranker_hard": p["hard"],
                "ranker_easy": p["easy"],
                "ranker_p05": p["p05"],
                "ranker_cvar5": p["cvar5"],
                "ranker_severe": p["severe_rate"],
            }
        )
    best = max(out, key=lambda r: finite_float(r["ranker_mean"])) if out else {}
    gate = {"best_cluster_policy": best.get("cluster_policy"), "best_cluster_ranker_mean": best.get("ranker_mean"), "best_cluster_ranker_hard": best.get("ranker_hard")}
    write_csv(args.out_dir / "v231_p2a_physics_cluster_oof_prototype_bank.csv", out)
    return out, gate


def noop_rows_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for _name, group in group_by_sample(rows).items():
        base = next((r for r in group if r["action_name"] == "noop"), group[0])
        out.append({**base, "dpsnr": 0.0, "policy_name": "noop_only", "selection_reason": "reject_fallback"})
    return out


def write_risk_coverage(rows: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_name = {str(r["sample_name"]): r for r in selected}
    noops = {str(r["sample_name"]): r for r in noop_rows_for(rows)}
    ranked = sorted(selected, key=lambda r: finite_float(r.get("pred_score")), reverse=True)
    out = []
    for coverage in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        keep_n = max(1, int(round(len(ranked) * coverage)))
        keep = {str(r["sample_name"]) for r in ranked[:keep_n]}
        policy = [by_name[name] if name in keep else noops[name] for name in sorted(noops)]
        s = summarize_selected(policy)
        easy = [finite_float(r["dpsnr"]) for r in policy if r["difficulty_bucket"] == "easy_top25"]
        out.append(
            {
                "coverage": coverage,
                "mean_gain": s["mean"],
                "hard_gain": s["hard"],
                "easy_gain": s["easy"],
                "p05": s["p05"],
                "CVaR5": s["cvar5"],
                "severe": s["severe_rate"],
                "strong_reference_regression": safe_rate(easy, STRONG_REG) if easy else float("nan"),
                "noop_rate": sum(1 for r in policy if r["action_name"] == "noop") / len(policy),
            }
        )
    write_csv(args.out_dir / "v231_p2a_noop_risk_coverage_curve.csv", out)
    return out


def random_selected(rows: list[dict[str, Any]], seed: int, pred: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for _name, group in group_by_sample(rows).items():
        subset = [r for r in group if pred(r)]
        if not subset:
            subset = group
        out.append({**rng.choice(subset), "policy_name": "random_control", "selection_reason": "seeded_random"})
    return out


def write_controls(rows: list[dict[str, Any]], oracle: list[dict[str, Any]], selected: list[dict[str, Any]], target_features: dict[str, dict[str, float]], args: argparse.Namespace) -> list[dict[str, Any]]:
    safe_rows = safe_candidate_rows(rows)
    shuffled_features = dict(target_features)
    names = sorted(shuffled_features)
    rng = random.Random(args.seed + 17)
    shuffled_names = names[:]
    rng.shuffle(shuffled_names)
    feature_perm = {name: shuffled_features[other] for name, other in zip(names, shuffled_names)}
    shuffled_rows = enrich_rows([{k: v for k, v in r.items() if k not in PHYSICS_KEYS} for r in rows], feature_perm)
    shuffled_safe = safe_candidate_rows(shuffled_rows)
    shuffled_scores = ridge_oof_scores(shuffled_safe, [finite_float(r["dpsnr"]) for r in shuffled_safe], FEATURE_SETS["combined_physics_frequency_features"])
    shuffled_target_selected = select_with_scores(shuffled_safe, shuffled_scores, "shuffled_target_features")
    shuffled_label_scores = score_methods(safe_rows, args.seed)["shuffled_label_control"]
    shuffled_label_selected = select_with_scores(safe_rows, shuffled_label_scores, "shuffled_action_labels")
    policies = {
        "real_target_features": selected,
        "shuffled_target_features": shuffled_target_selected,
        "shuffled_action_labels": shuffled_label_selected,
        "same_bucket_random_action": random_selected(safe_rows, args.seed + 31, lambda r: r["prototype_source_bucket"] in ("all", r["difficulty_bucket"])),
        "same_cluster_random_action": random_selected(safe_rows, args.seed + 43, lambda r: True),
        "no_op_only": noop_rows_for(safe_rows),
        "GT_selected_oracle": oracle,
    }
    out = []
    oracle_mean = summarize_selected(oracle)["mean"]
    for name, policy in policies.items():
        s = summarize_selected(policy)
        out.append(
            {
                "control": name,
                "mean": s["mean"],
                "hard": s["hard"],
                "easy": s["easy"],
                "p05": s["p05"],
                "CVaR5": s["cvar5"],
                "severe": s["severe_rate"],
                "gap_to_oracle": oracle_mean - s["mean"],
            }
        )
    write_csv(args.out_dir / "v231_p2a_real_vs_shuffled_action_value_controls.csv", out)
    return out


def write_confusion(args: argparse.Namespace, oracle: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_name = {str(r["sample_name"]): r for r in selected}
    groups: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in oracle:
        pred = selected_by_name.get(str(row["sample_name"]))
        if pred is None:
            continue
        groups[(row["difficulty_bucket"], action_family(str(row["action_name"])), action_family(str(pred["action_name"])), row["stage_set"], pred["stage_set"], row["prototype_source_bucket"], pred["prototype_source_bucket"])] += 1
    out = []
    for key, count in sorted(groups.items()):
        target, oracle_action, policy_action, oracle_stage, policy_stage, oracle_source, policy_source = key
        out.append(
            {
                "target_bucket": target,
                "oracle_action_family": oracle_action,
                "policy_action_family": policy_action,
                "oracle_stage_set": oracle_stage,
                "policy_stage_set": policy_stage,
                "oracle_source_bucket": oracle_source,
                "policy_source_bucket": policy_source,
                "count": count,
            }
        )
    write_csv(args.out_dir / "v231_p2a_policy_vs_safe_set_confusion_matrix.csv", out)
    return out


def write_local_optimum_audit(args: argparse.Namespace, feature_gate: dict[str, Any], ranking_gate: dict[str, Any], controls: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> None:
    real = next((r for r in controls if r["control"] == "real_target_features"), {})
    shuffled = next((r for r in controls if r["control"] == "shuffled_action_labels"), {})
    best_cov = max(coverage, key=lambda r: finite_float(r["hard_gain"])) if coverage else {}
    lines = [
        "# v2.31 Local Optimum Audit",
        "",
        "Question: does adding target-only physics/frequency evidence rescue action-value",
        "selection, or does the v2.30 bank remain a no-op local optimum?",
        "",
        f"feature_gate_pass: `{feature_gate.get('pass')}`",
        f"ranking_gate_pass: `{ranking_gate.get('pass')}`",
        f"best_ranker: `{ranking_gate.get('best_model')}`",
        f"real_policy_mean: `{real.get('mean')}`",
        f"shuffled_label_mean: `{shuffled.get('mean')}`",
        f"best_coverage_hard_gain: `{best_cov.get('hard_gain')}` at coverage `{best_cov.get('coverage')}`",
        "",
        "Interpretation rule: if separability and nested ranking both fail, the",
        "current discrete action-bank selector route should close rather than receive",
        "more table/firewall micro-tuning.",
    ]
    write_text(args.out_dir / "v231_p2a_local_optimum_audit.md", "\n".join(lines))


def write_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    diag = closeout.get("p2a", {}).get("primary_diagnosis", {})
    lines = [
        "# Haze4K v2.31 NoPost Target-Only Action-Value Identifiability Audit Evidence",
        "",
        "Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-31-nopost-action-value-identifiability-audit.md`",
        "",
        f"Status: `{closeout.get('decision', 'UNKNOWN')}`",
        "",
        "Runtime server: `convir-4090`",
        "Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-31-nopost-action-value-identifiability-audit`",
        "Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`",
        "",
        "Hard blocks:",
        "",
        "- `training_launched: false`",
        "- `p2b_selector_probe_launched: false`",
        "- `locked_test_touched: false`",
        "",
        "## Key Results",
        "",
        f"- Decision: `{closeout.get('decision', 'UNKNOWN')}`",
        f"- Feature gate pass: `{diag.get('feature_gate_pass')}`",
        f"- Ranking gate pass: `{diag.get('ranking_gate_pass')}`",
        f"- Best ranker: `{diag.get('best_ranker')}`",
        f"- Best ranker mean/hard/severe/gap: `{diag.get('best_ranker_mean')} / {diag.get('best_ranker_hard')} / {diag.get('best_ranker_severe')} / {diag.get('best_ranker_gap')}`",
        "",
        "## Primary Files",
        "",
        "- `v231_p0_arch_contract_delta.md`",
        "- `v231_p2a_action_value_feature_separability.csv`",
        "- `v231_p2a_safe_set_ranking_upper_bound.md`",
        "- `v231_p2a_physics_cluster_oof_prototype_bank.csv`",
        "- `v231_p2a_noop_risk_coverage_curve.csv`",
        "- `v231_p2a_real_vs_shuffled_action_value_controls.csv`",
        "- `v231_p2a_policy_vs_safe_set_confusion_matrix.csv`",
        "- `v231_p2a_local_optimum_audit.md`",
        "- `v231_p2a_closeout.json`",
        "- `run_v231_p2a.sh`",
        "- `monitor_v231.sh`",
        "- `status.txt`",
        "",
        "This directory is compact text evidence only. It excludes checkpoints, weights, images, arrays, archives, and raw feature tables by default.",
    ]
    write_text(args.out_dir / "README.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", "--data-dir", dest="data_dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--delta-scale", type=float, default=0.25)
    ap.add_argument("--coverage-budget", type=float, default=0.35)
    ap.add_argument("--oracle-steps", type=int, default=10)
    ap.add_argument("--oracle-lr", type=float, default=0.06)
    ap.add_argument("--oracle-delta-scale", type=float, default=0.50)
    ap.add_argument("--oracle-reg", type=float, default=1e-4)
    ap.add_argument("--prototype-aggregate", default="median")
    ap.add_argument("--stage-sets", default="S6_early_mid_final,S5_bottleneck_mid,S4_final_decoder")
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--parent-commit", default="8971902")
    ap.add_argument("--conservative-penalty", type=float, default=0.20)
    ap.add_argument("--prototype-complexity-penalty", type=float, default=0.03)
    ap.add_argument("--bucket-distance-penalty", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=231)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    append_status(args, f"v231_start route_id={ROUTE_ID}")
    append_status(args, "training_launched=false")
    append_status(args, "p2b_selector_probe_launched=false")
    append_status(args, "locked_test_touched=false")
    closeout: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_branch": PARENT_BRANCH,
        "parent_commit": args.parent_commit,
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
    }
    p0 = phase_p0(args)
    closeout["p0"] = p0
    if not p0["pass"]:
        closeout["decision"] = p0["decision"]
        write_json(args.out_dir / "v231_p2a_closeout.json", json_clean(closeout))
        write_readme(args, closeout)
        append_status(args, f"v231_done decision={closeout['decision']}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_features = build_target_features(args, device)
    write_json(
        args.out_dir / "v231_p2a_feature_manifest.json",
        {
            "sample_count": len(target_features),
            "feature_count": len(PHYSICS_KEYS),
            "physics_feature_names": PHYSICS_KEYS,
            "raw_feature_table_synced": False,
            "locked_test_touched": False,
        },
    )
    sample_rows, _p1_rows, delta_bank = build_oracle_delta_bank(args, device)
    attach_sample_buckets(sample_rows)
    rows = enrich_rows(run_variant_replay(args, sample_rows, delta_bank, device), target_features)
    safe_rows = safe_candidate_rows(rows)
    oracle = select_restricted_oracle(rows, LCB_POLICY)
    table_selected, _cell_reports = run_lcb_policy(rows, LCB_POLICY)

    feature_sep, feature_gate = write_feature_separability(rows, args)
    ranking_rows, ranking_gate, best_selected = write_ranking_upper_bound(rows, oracle, table_selected, args)
    cluster_rows = cluster_replay(args, sample_rows, delta_bank, target_features, device)
    cluster_report, cluster_gate = write_cluster_report(cluster_rows, args)
    coverage = write_risk_coverage(safe_rows, best_selected, args)
    controls = write_controls(rows, oracle, best_selected, target_features, args)
    confusion = write_confusion(args, oracle, best_selected)
    write_local_optimum_audit(args, feature_gate, ranking_gate, controls, coverage)

    decision = "P2A_PASS_ACTION_VALUE_IDENTIFIABILITY" if feature_gate["pass"] and ranking_gate["pass"] else "P2A_FAIL_ACTION_VALUE_IDENTIFIABILITY_CLOSE_CURRENT_BANK"
    closeout["p2a"] = {
        "decision": decision,
        "pass": bool(feature_gate["pass"] and ranking_gate["pass"]),
        "primary_question": "Can target-only / physics-frequency features identify useful safe-set actions?",
        "primary_diagnosis": {
            "feature_gate_pass": feature_gate["pass"],
            "ranking_gate_pass": ranking_gate["pass"],
            "best_ranker": ranking_gate["best_model"],
            "best_ranker_mean": ranking_gate["best_mean"],
            "best_ranker_hard": ranking_gate["best_hard"],
            "best_ranker_severe": ranking_gate["best_severe"],
            "best_ranker_gap": ranking_gate["best_gap"],
            "best_cluster_policy": cluster_gate.get("best_cluster_policy"),
            "best_cluster_ranker_mean": cluster_gate.get("best_cluster_ranker_mean"),
        },
        "feature_gate": feature_gate,
        "ranking_gate": ranking_gate,
        "cluster_gate": cluster_gate,
        "artifact_counts": {
            "safe_candidate_rows": len(safe_rows),
            "feature_separability_rows": len(feature_sep),
            "ranking_rows": len(ranking_rows),
            "cluster_summary_rows": len(cluster_report),
            "coverage_rows": len(coverage),
            "control_rows": len(controls),
            "confusion_rows": len(confusion),
        },
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
    }
    closeout["decision"] = decision
    write_json(args.out_dir / "v231_p2a_closeout.json", json_clean(closeout))
    write_readme(args, closeout)
    append_status(args, f"v231_done decision={decision}")
    print("V231_ACTION_VALUE_IDENTIFIABILITY_AUDIT_OK " + decision, flush=True)


if __name__ == "__main__":
    main()
