#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nopost_v227_ilfrb_acs_diagnostics import (  # noqa: E402
    SEVERE,
    STAGE_SETS,
    STRONG_REG,
    average_precision,
    build_models,
    cvar,
    decode_from_features,
    finite,
    forward_cache,
    git,
    haar_dwt,
    image_tensor,
    input_stats,
    load_samples,
    mean,
    optimize_stage_set,
    pad_to,
    percentile,
    roc_auc,
    sha256_file,
    source_scan,
    std,
    tensor_psnr,
    write_csv,
    write_json,
    write_text,
)
from nopost_v228_action_bank_stratification_audit import (  # noqa: E402
    aggregate_tensors,
    attach_sample_buckets,
    make_oof_prototype,
    opposite_bucket,
    prototype_id,
    row_mean,
    target_same_channel_stage,
)


ROUTE_ID = "haze4k_v2_29_nopost_ilfrb_acs_safe_oof_action_bank_calibration_20260705"
BUCKETS = ("hard_bottom25", "mid_50", "easy_top25")
BASE_DEPLOYABLE_ACTIONS = (("mild_0.33", 0.33), ("medium_0.67", 0.67), ("strong_1.25", 1.25))
DIAGNOSTIC_ACTIONS = (
    ("overstrong_1.5", 1.50),
    ("overstrong_2.0", 2.00),
    ("overstrong_3.0", 3.00),
    ("sign_flip", 0.67),
    ("wrong_stage", 0.67),
    ("cross_bucket_mismatch", 0.67),
)
VARIANTS = (
    "raw_v228_baseline",
    "energy_norm",
    "rms_clip",
    "absmax_clip",
    "alignment_gate",
    "bucket_strength_grid",
    "s5_only",
    "s5_plus_s6_hard_only",
    "s5_plus_s4_mild_only",
    "energy_norm_plus_bucket_strength",
    "energy_norm_plus_bucket_strength_plus_alignment_gate",
)


def append_status(args: argparse.Namespace, line: str) -> None:
    path = args.out_dir / "status.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def phase_p0(args: argparse.Namespace) -> dict[str, Any]:
    scan = source_scan()
    report = {
        "phase": "p0_arch_contract_delta",
        "route": "v2.29",
        "parent_route": "v2.28",
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_commit": args.parent_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_delta": "none_from_v2_28_ilfrb_acs",
        "runtime_contract": "forward(self, x)",
        "training_launched": False,
        "locked_test_touched": False,
        "forbidden_symbol_hits": scan["hit_count"],
        "decision": "P0_PASS_ARCH_CONTRACT_DELTA_AUDIT" if scan["hit_count"] == 0 else "P0_FAIL_SOURCE_CONTRACT",
        "pass": scan["hit_count"] == 0,
    }
    lines = [
        "# v2.29 P0 Architecture Contract Delta",
        "",
        f"branch: `{report['branch']}`",
        f"commit: `{report['commit']}`",
        f"parent_branch: `codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`",
        f"parent_commit: `{args.parent_commit}`",
        f"checkpoint: `{args.checkpoint}`",
        f"checkpoint_sha256: `{report['checkpoint_sha256']}`",
        "",
        "v2.29 does not add a new runtime model structure. It reuses the v2.27/v2.28",
        "NoPost ILFRB-ACS snapshot and changes only the train-derived action-bank",
        "safety-envelope replay protocol.",
        "",
        "runtime_forward_contract: `forward(self, x)`",
        "teacher_or_expert_forward_input: `false`",
        "rgb_output_output_residual: `false`",
        "learned_rgb_post_output_correction: `false`",
        "training_launched: `false`",
        "locked_test_touched: `false`",
        f"forbidden_symbol_hits: `{scan['hit_count']}`",
        f"decision: `{report['decision']}`",
    ]
    write_text(args.out_dir / "v229_p0_arch_contract_delta.md", "\n".join(lines))
    return report


def build_oracle_delta_bank(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, torch.Tensor]]]]:
    a0, _route, _partial = build_models(args, device)
    samples = load_samples(args)
    stage_sets = [item for item in args.stage_sets.split(",") if item]
    p1_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    print(f"V229_SELECTED_SAMPLES count={len(samples)} stage_sets={','.join(stage_sets)}", flush=True)
    for idx, sample in enumerate(samples, start=1):
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        a0_pred = cache["out"][:, :, :h, :w]
        a0_psnr = tensor_psnr(a0_pred, gt)
        stats = input_stats(sample)
        sample_rows.append({"sample_name": sample.name, "target_fold": sample.fold, "a0_psnr": a0_psnr, **stats})
        delta_bank[sample.name] = {}
        for stage_set in stage_sets:
            deltas, opt_stats = optimize_stage_set(a0, cache, gt, STAGE_SETS[stage_set], args)
            pred = decode_from_features(a0, cache, {k: v.to(device) for k, v in deltas.items()})[:, :, :h, :w]
            psnr = tensor_psnr(pred, gt)
            delta_bank[sample.name][stage_set] = {stage: delta.detach().cpu().float() for stage, delta in deltas.items()}
            p1_rows.append(
                {
                    "sample_name": sample.name,
                    "target_fold": sample.fold,
                    "stage_set": stage_set,
                    "stages": "+".join(STAGE_SETS[stage_set]),
                    "a0_psnr": a0_psnr,
                    "same_sample_oracle_psnr": psnr,
                    "same_sample_oracle_dpsnr": psnr - a0_psnr,
                    "best_loss": opt_stats["loss"],
                    "delta_abs_mean": row_mean([float(delta.abs().mean()) for delta in deltas.values()]),
                    "delta_rms": row_mean([float(torch.sqrt(torch.mean(delta.float() ** 2))) for delta in deltas.values()]),
                    "delta_abs_max": max(float(delta.abs().max()) for delta in deltas.values()),
                }
            )
        if idx % args.print_freq == 0:
            print(f"V229_ORACLE_PROGRESS {idx}/{len(samples)}", flush=True)
    return sample_rows, p1_rows, delta_bank


def action_family(action_name: str) -> str:
    if action_name == "noop":
        return "noop"
    if action_name.startswith("mild"):
        return "mild"
    if action_name.startswith("medium"):
        return "medium"
    if action_name.startswith("strong"):
        return "strong"
    if action_name.startswith("overstrong"):
        return "overstrong"
    return action_name


def control_family(action_name: str) -> tuple[str, int, int, int, str]:
    if action_name in ("sign_flip", "overstrong_3.0"):
        return ("impossible_sanity", 1, 0, 0, "sensitivity_lower_bound")
    if action_name in ("overstrong_1.5", "overstrong_2.0"):
        return ("plausible_miscalibration", 0, 1, 0, "safety_upper_bound")
    if action_name in ("cross_bucket_mismatch", "wrong_stage"):
        return ("plausible_routing_error", 0, 0, 1, "safety_upper_bound")
    return ("deployable_candidate", 0, 0, 0, "deployable_tail")


def deployable_actions_for_variant(variant: str, target_bucket: str) -> tuple[tuple[str, float], ...]:
    if "bucket_strength" not in variant:
        return BASE_DEPLOYABLE_ACTIONS
    if target_bucket == "easy_top25":
        return (("mild_0.10", 0.10), ("mild_0.20", 0.20), ("mild_0.33", 0.33))
    if target_bucket == "mid_50":
        return (("mild_0.20", 0.20), ("mild_0.33", 0.33), ("medium_0.50", 0.50), ("medium_0.67", 0.67))
    return (
        ("mild_0.33", 0.33),
        ("medium_0.50", 0.50),
        ("medium_0.67", 0.67),
        ("strong_0.85", 0.85),
        ("strong_1.00", 1.00),
        ("strong_1.25", 1.25),
    )


def stage_allowed(variant: str, stage_set: str, target_bucket: str, action_name: str) -> bool:
    family = action_family(action_name)
    if variant == "s5_only":
        return stage_set == "S5_bottleneck_mid"
    if variant == "s5_plus_s6_hard_only":
        return stage_set == "S5_bottleneck_mid" or (stage_set == "S6_early_mid_final" and target_bucket == "hard_bottom25")
    if variant == "s5_plus_s4_mild_only":
        if stage_set == "S5_bottleneck_mid":
            return True
        return stage_set == "S4_final_decoder" and target_bucket != "easy_top25" and family == "mild"
    return True


def stage_ll_stats(cache: dict[str, torch.Tensor], stage: str) -> dict[str, float]:
    ll, _, _, _, _, _ = haar_dwt(cache[stage])
    flat = ll.detach().abs().flatten()
    p95 = torch.quantile(flat.float(), 0.95).clamp_min(1e-8)
    return {
        "ll_rms": float(torch.sqrt(torch.mean(ll.float() ** 2)).detach().cpu()),
        "ll_abs_p95": float(p95.detach().cpu()),
        "ll_abs_max": float(flat.max().detach().cpu()),
    }


def base_delta_stats(deltas: dict[str, torch.Tensor], cache: dict[str, torch.Tensor]) -> dict[str, float]:
    if not deltas:
        return {
            "delta_abs_mean": 0.0,
            "delta_rms": 0.0,
            "delta_abs_max": 0.0,
            "delta_alignment": 0.0,
            "stagewise_alignment_min": 0.0,
            "stagewise_alignment_mean": 0.0,
            "delta_rms_to_target_ll_rms": 0.0,
            "delta_absmax_to_target_ll_absmax": 0.0,
        }
    abs_means: list[float] = []
    rms_vals: list[float] = []
    max_vals: list[float] = []
    aligns: list[float] = []
    rms_ratios: list[float] = []
    max_ratios: list[float] = []
    for stage, delta_cpu in deltas.items():
        delta = delta_cpu.to(cache[stage].device, dtype=cache[stage].dtype)
        ll, _, _, _, _, _ = haar_dwt(cache[stage])
        ll_grid = F.adaptive_avg_pool2d(ll.detach(), delta.shape[-2:])
        flat_d = delta.flatten(1)
        flat_ll = ll_grid.flatten(1)
        align = ((flat_d * flat_ll).sum(dim=1) / (flat_d.norm(dim=1) * flat_ll.norm(dim=1).clamp_min(1e-8)).clamp_min(1e-8)).detach().cpu()
        ll_stats = stage_ll_stats(cache, stage)
        d_rms = float(torch.sqrt(torch.mean(delta.float() ** 2)).detach().cpu())
        d_absmax = float(delta.abs().max().detach().cpu())
        abs_means.append(float(delta.abs().mean().detach().cpu()))
        rms_vals.append(d_rms)
        max_vals.append(d_absmax)
        aligns.append(float(align.mean()))
        rms_ratios.append(d_rms / max(ll_stats["ll_rms"], 1e-8))
        max_ratios.append(d_absmax / max(ll_stats["ll_abs_p95"], 1e-8))
    return {
        "delta_abs_mean": mean(abs_means),
        "delta_rms": mean(rms_vals),
        "delta_abs_max": max(max_vals),
        "delta_alignment": mean(aligns),
        "stagewise_alignment_min": min(aligns),
        "stagewise_alignment_mean": mean(aligns),
        "delta_rms_to_target_ll_rms": mean(rms_ratios),
        "delta_absmax_to_target_ll_absmax": mean(max_ratios),
    }


def bucket_distance(source_bucket: str, target_bucket: str) -> int:
    if source_bucket == "all":
        return 1
    if source_bucket == target_bucket:
        return 0
    if "mid_50" in (source_bucket, target_bucket):
        return 1
    return 2


def apply_wrong_stage(deltas: dict[str, torch.Tensor], strength: float) -> dict[str, torch.Tensor]:
    shifted: dict[str, torch.Tensor] = {}
    for stage, delta in deltas.items():
        target = target_same_channel_stage(stage)
        if target is not None:
            shifted[target] = delta * strength
    return shifted


def scale_to_rms(delta: torch.Tensor, cap: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(delta.float() ** 2)).item()
    if rms <= cap or rms <= 1e-12:
        return delta
    return delta * (cap / rms)


def clip_absmax(delta: torch.Tensor, cap: float) -> torch.Tensor:
    return delta.clamp(min=-cap, max=cap)


def envelope_deltas(
    variant: str,
    proto: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    target_bucket: str,
    source_bucket: str,
    action_name: str,
    strength: float,
) -> tuple[dict[str, torch.Tensor], bool, str]:
    if action_name == "noop":
        return {}, True, "noop"
    if action_name == "wrong_stage":
        base = apply_wrong_stage(proto, strength)
        if not base:
            return {}, False, "wrong_stage_no_compatible_stage"
    elif action_name == "sign_flip":
        base = {stage: -delta * strength for stage, delta in proto.items()}
    else:
        base = {stage: delta * strength for stage, delta in proto.items()}

    if variant in ("alignment_gate", "energy_norm_plus_bucket_strength_plus_alignment_gate"):
        stats = base_delta_stats(base, cache)
        if stats["stagewise_alignment_mean"] < 0.10 and action_family(action_name) in ("medium", "strong", "overstrong"):
            return {}, False, "alignment_gate_blocked_non_mild"
        if source_bucket == "hard_bottom25" and target_bucket == "easy_top25" and action_family(action_name) != "mild":
            return {}, False, "hard_to_easy_gate_blocked_non_mild"

    out: dict[str, torch.Tensor] = {}
    for stage, delta in base.items():
        edited = delta.clone()
        ll_stats = stage_ll_stats(cache, stage)
        if variant in ("energy_norm", "energy_norm_plus_bucket_strength", "energy_norm_plus_bucket_strength_plus_alignment_gate"):
            ratio_cap = {"easy_top25": 0.010, "mid_50": 0.016, "hard_bottom25": 0.024}[target_bucket] * ll_stats["ll_rms"]
            edited = scale_to_rms(edited, max(ratio_cap, 1e-5))
        if variant == "rms_clip":
            cap = {"easy_top25": 0.0035, "mid_50": 0.0060, "hard_bottom25": 0.0100}[target_bucket]
            edited = scale_to_rms(edited, cap)
        if variant == "absmax_clip":
            cap = {"easy_top25": 0.040, "mid_50": 0.070, "hard_bottom25": 0.110}[target_bucket]
            edited = clip_absmax(edited, cap)
        out[stage] = edited
    return out, True, "allowed"


def add_row(
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    gt: torch.Tensor,
    h: int,
    w: int,
    base: dict[str, Any],
    proto: dict[str, torch.Tensor],
    variant: str,
    action_name: str,
    strength: float,
    deployable: bool,
    diagnostic: bool,
    device: torch.device,
) -> None:
    transformed, valid, reason = envelope_deltas(
        variant,
        proto,
        cache,
        str(base["difficulty_bucket"]),
        str(base["prototype_source_bucket"]),
        action_name,
        strength,
    )
    if not valid:
        return
    if transformed:
        pred = decode_from_features(model, cache, {k: v.to(device) for k, v in transformed.items()})[:, :, :h, :w]
        psnr = tensor_psnr(pred, gt)
    else:
        psnr = float(base["a0_psnr"])
    dpsnr = psnr - float(base["a0_psnr"])
    stats = base_delta_stats(transformed, cache)
    control, impossible, miscal, routing, role = control_family(action_name)
    rows.append(
        {
            **base,
            "variant": variant,
            "action_name": action_name,
            "action_family": action_family(action_name),
            "action_strength": strength,
            "deployable_candidate": int(deployable),
            "diagnostic_negative_control": int(diagnostic),
            "control_family": control,
            "is_impossible_sanity_control": impossible,
            "is_plausible_miscalibration": miscal,
            "is_plausible_routing_error": routing,
            "gate_role": role,
            "envelope_reason": reason,
            "action_psnr": psnr,
            "dpsnr": dpsnr,
            "is_unsafe": int(dpsnr <= STRONG_REG),
            "is_severe": int(dpsnr <= SEVERE),
            "is_strong_regression": int(dpsnr <= STRONG_REG and str(base["difficulty_bucket"]) == "easy_top25"),
            **stats,
        }
    )


def run_variant_replay(
    args: argparse.Namespace,
    sample_rows: list[dict[str, Any]],
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]],
    device: torch.device,
) -> list[dict[str, Any]]:
    a0, _route, _partial = build_models(args, device)
    selected = {str(row["sample_name"]): row for row in sample_rows}
    samples = [sample for sample in load_samples(args) if sample.name in selected]
    stage_sets = [item for item in args.stage_sets.split(",") if item]
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        meta = selected[sample.name]
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        target_bucket = str(meta["difficulty_bucket"])
        source_buckets = ["all", target_bucket, opposite_bucket(target_bucket)]
        if "mid_50" not in source_buckets:
            source_buckets.append("mid_50")
        for variant in VARIANTS:
            noop_base = {
                "sample_name": sample.name,
                "target_fold": int(meta["target_fold"]),
                "source_fold": f"not_{int(meta['target_fold'])}",
                "a0_psnr": float(meta["a0_psnr"]),
                "difficulty_bucket": target_bucket,
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
            }
            add_row(rows, a0, cache, gt, h, w, noop_base, {}, variant, "noop", 0.0, True, False, device)
            for stage_set in stage_sets:
                for source_bucket in source_buckets:
                    proto, source_count = make_oof_prototype(sample_rows, delta_bank, int(meta["target_fold"]), source_bucket, stage_set, args.prototype_aggregate)
                    if proto is None:
                        continue
                    base = {
                        **noop_base,
                        "prototype_id": prototype_id(int(meta["target_fold"]), source_bucket, stage_set, args.prototype_aggregate),
                        "prototype_source_bucket": source_bucket,
                        "stage_set": stage_set,
                        "prototype_source_count": source_count,
                        "bucket_distance": bucket_distance(source_bucket, target_bucket),
                    }
                    source_matches = source_bucket in ("all", target_bucket)
                    if source_matches:
                        for action_name, strength in deployable_actions_for_variant(variant, target_bucket):
                            if stage_allowed(variant, stage_set, target_bucket, action_name):
                                add_row(rows, a0, cache, gt, h, w, base, proto, variant, action_name, strength, True, False, device)
                        for action_name, strength in DIAGNOSTIC_ACTIONS:
                            if action_name != "cross_bucket_mismatch" and stage_allowed(variant, stage_set, target_bucket, action_name):
                                add_row(rows, a0, cache, gt, h, w, base, proto, variant, action_name, strength, False, True, device)
                    elif stage_allowed(variant, stage_set, target_bucket, "cross_bucket_mismatch"):
                        add_row(rows, a0, cache, gt, h, w, base, proto, variant, "cross_bucket_mismatch", 0.67, False, True, device)
        if idx % args.print_freq == 0:
            print(f"V229_REPLAY_PROGRESS {idx}/{len(samples)} rows={len(rows)}", flush=True)
    return rows


def conservative_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    easy = row["difficulty_bucket"] == "easy_top25"
    return (
        float(row["dpsnr"])
        - args.conservative_penalty * float(row["action_strength"]) * (1.0 + float(easy))
        - args.prototype_complexity_penalty * float(row["stage_set"] != "noop")
        - args.bucket_distance_penalty * float(row["bucket_distance"])
    )


def preference_rows(rows: list[dict[str, Any]], variant: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] != variant or int(row["deployable_candidate"]) != 1:
            continue
        if row["action_name"] != "noop" and row["prototype_source_bucket"] not in ("all", row["difficulty_bucket"]):
            continue
        by_name.setdefault(str(row["sample_name"]), []).append(row)
    prefs = []
    for name, group in by_name.items():
        best = max(group, key=lambda r: conservative_score(r, args))
        raw = max(group, key=lambda r: float(r["dpsnr"]))
        prefs.append(
            {
                "variant": variant,
                "sample_name": name,
                "target_fold": best["target_fold"],
                "difficulty_bucket": best["difficulty_bucket"],
                "stage_set": best["stage_set"],
                "prototype_source_bucket": best["prototype_source_bucket"],
                "action_name": best["action_name"],
                "action_family": best["action_family"],
                "action_strength": best["action_strength"],
                "dpsnr": best["dpsnr"],
                "conservative_score": conservative_score(best, args),
                "raw_best_action": raw["action_name"],
                "raw_best_dpsnr": raw["dpsnr"],
            }
        )
    return prefs


def summarize_policy_rows(rows: list[dict[str, Any]], value_key: str = "dpsnr") -> dict[str, Any]:
    vals = [float(row[value_key]) for row in rows]
    hard = [float(row[value_key]) for row in rows if row["difficulty_bucket"] == "hard_bottom25"]
    easy = [float(row[value_key]) for row in rows if row["difficulty_bucket"] == "easy_top25"]
    return {
        "count": len(vals),
        "mean": mean(vals),
        "hard": mean(hard),
        "easy": mean(easy),
        "p05": percentile(vals, 5),
        "cvar5": cvar(vals, 5),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_regression_rate": sum(v <= STRONG_REG for v in easy) / len(easy) if easy else float("nan"),
    }


def rate(rows: list[dict[str, Any]], pred) -> float:
    return sum(1 for row in rows if pred(row)) / len(rows) if rows else float("nan")


def subset_metric(rows: list[dict[str, Any]], pred, metric: str) -> float:
    subset = [row for row in rows if pred(row)]
    vals = [float(row["dpsnr"]) for row in subset]
    if metric == "unsafe":
        return sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan")
    if metric == "severe":
        return sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan")
    return float("nan")


def safety_envelope_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    best_pass: dict[str, Any] | None = None
    for variant in VARIANTS:
        prefs = preference_rows(rows, variant, args)
        selected = summarize_policy_rows(prefs)
        noop_rate = rate(prefs, lambda r: r["action_name"] == "noop")
        easy_noop_mild = rate([r for r in prefs if r["difficulty_bucket"] == "easy_top25"], lambda r: r["action_family"] in ("noop", "mild"))
        hard_medium_strong = rate([r for r in prefs if r["difficulty_bucket"] == "hard_bottom25"], lambda r: r["action_family"] in ("medium", "strong"))
        deploy = [r for r in rows if r["variant"] == variant and int(r["deployable_candidate"]) == 1]
        diag = [r for r in rows if r["variant"] == variant and int(r["diagnostic_negative_control"]) == 1]
        fold_tail = 0
        for fold in sorted({int(r["target_fold"]) for r in prefs}):
            frows = [r for r in prefs if int(r["target_fold"]) == fold]
            vals = [float(r["dpsnr"]) for r in frows]
            if vals and percentile(vals, 5) >= -0.15 and sum(v <= SEVERE for v in vals) / len(vals) <= 0.035:
                fold_tail += 1
        row = {
            "variant": variant,
            "selected_mean": selected["mean"],
            "selected_hard": selected["hard"],
            "selected_easy": selected["easy"],
            "p05": selected["p05"],
            "cvar5": selected["cvar5"],
            "severe_rate": selected["severe_rate"],
            "fold_tail_pass": fold_tail,
            "deployable_mild_unsafe": subset_metric(deploy, lambda r: r["action_family"] == "mild", "unsafe"),
            "deployable_medium_unsafe": subset_metric(deploy, lambda r: r["action_family"] == "medium", "unsafe"),
            "deployable_strong_unsafe": subset_metric(deploy, lambda r: r["action_family"] == "strong", "unsafe"),
            "deployable_strong_easy_unsafe": subset_metric(deploy, lambda r: r["action_family"] == "strong" and r["difficulty_bucket"] == "easy_top25", "unsafe"),
            "cross_bucket_unsafe": subset_metric(diag, lambda r: r["action_name"] == "cross_bucket_mismatch", "unsafe"),
            "hard_to_easy_cross_severe": subset_metric(diag, lambda r: r["action_name"] == "cross_bucket_mismatch" and r["prototype_source_bucket"] == "hard_bottom25" and r["difficulty_bucket"] == "easy_top25", "severe"),
            "overstrong_1p5_unsafe": subset_metric(diag, lambda r: r["action_name"] == "overstrong_1.5", "unsafe"),
            "overstrong_2p0_unsafe": subset_metric(diag, lambda r: r["action_name"] == "overstrong_2.0", "unsafe"),
            "overstrong_3p0_unsafe": subset_metric(diag, lambda r: r["action_name"] == "overstrong_3.0", "unsafe"),
            "sign_flip_unsafe": subset_metric(diag, lambda r: r["action_name"] == "sign_flip", "unsafe"),
            "wrong_stage_unsafe": subset_metric(diag, lambda r: r["action_name"] == "wrong_stage", "unsafe"),
            "noop_rate": noop_rate,
            "easy_noop_mild_rate": easy_noop_mild,
            "hard_medium_strong_rate": hard_medium_strong,
        }
        row["pass"] = int(
            0.10 <= row["noop_rate"] <= 0.50
            and row["easy_noop_mild_rate"] >= 0.40
            and row["hard_medium_strong_rate"] >= 0.30
            and row["selected_mean"] >= 0.40
            and row["selected_hard"] >= 0.80
            and row["selected_easy"] >= 0.0
            and row["p05"] >= -0.15
            and row["cvar5"] >= -0.35
            and row["severe_rate"] <= 0.035
            and row["fold_tail_pass"] >= 4
            and row["deployable_mild_unsafe"] <= 0.20
            and row["deployable_medium_unsafe"] <= 0.30
            and row["deployable_strong_unsafe"] <= 0.35
            and row["deployable_strong_easy_unsafe"] <= 0.20
            and row["cross_bucket_unsafe"] <= 0.35
            and row["hard_to_easy_cross_severe"] <= 0.35
            and row["overstrong_1p5_unsafe"] <= 0.35
            and row["wrong_stage_unsafe"] <= 0.30
            and row["sign_flip_unsafe"] >= 0.60
            and row["overstrong_3p0_unsafe"] >= 0.55
        )
        summaries.append(row)
        if row["pass"] and (best_pass is None or float(row["selected_mean"]) > float(best_pass["selected_mean"])):
            best_pass = row
    if best_pass is None:
        best_pass = max(summaries, key=lambda r: float(r["selected_mean"]) - 2.0 * max(0.0, float(r["deployable_strong_easy_unsafe"]) - 0.20))
    return summaries, best_pass


def group_summary(rows: list[dict[str, Any]], keys: list[str], extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for values, subset in sorted(groups.items()):
        vals = [float(row["dpsnr"]) for row in subset]
        payload = {k: v for k, v in zip(keys, values)}
        if extra:
            payload.update(extra)
        payload.update(
            {
                "count": len(subset),
                "mean_dpsnr": mean(vals),
                "p05_dpsnr": percentile(vals, 5),
                "cvar5_dpsnr": cvar(vals),
                "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
            }
        )
        out.append(payload)
    return out


def write_negative_taxonomy(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    diag = [r for r in rows if int(r["diagnostic_negative_control"]) == 1]
    out = group_summary(
        diag,
        ["variant", "control_family", "action_name", "stage_set", "prototype_source_bucket", "difficulty_bucket"],
    )
    for row in out:
        family, impossible, miscal, routing, role = control_family(str(row["action_name"]))
        row["source_bucket"] = row.pop("prototype_source_bucket")
        row["target_bucket"] = row.pop("difficulty_bucket")
        row["is_impossible_sanity_control"] = impossible
        row["is_plausible_miscalibration"] = miscal
        row["is_plausible_routing_error"] = routing
        row["gate_role"] = role
    write_csv(args.out_dir / "v229_p2a_negative_control_taxonomy.csv", out)


def write_deployable_tail(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    deploy = [r for r in rows if int(r["deployable_candidate"]) == 1 and r["action_name"] != "noop"]
    out = group_summary(deploy, ["variant", "difficulty_bucket", "prototype_source_bucket", "stage_set", "action_name", "action_strength"])
    for row in out:
        row["target_bucket"] = row.pop("difficulty_bucket")
        row["source_bucket"] = row.pop("prototype_source_bucket")
        subset = [
            r
            for r in deploy
            if r["variant"] == row["variant"]
            and r["difficulty_bucket"] == row["target_bucket"]
            and r["prototype_source_bucket"] == row["source_bucket"]
            and r["stage_set"] == row["stage_set"]
            and str(r["action_name"]) == str(row["action_name"])
            and float(r["action_strength"]) == float(row["action_strength"])
        ]
        easy_vals = [float(r["dpsnr"]) for r in subset if r["difficulty_bucket"] == "easy_top25"]
        row["strong_reference_regression_rate"] = sum(v <= STRONG_REG for v in easy_vals) / len(easy_vals) if easy_vals else 0.0
    write_csv(args.out_dir / "v229_p2a_deployable_candidate_tail_by_bucket_stage.csv", out)


def write_cross_and_stage_reports(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    cross = [r for r in rows if r["action_name"] == "cross_bucket_mismatch"]
    cross_rows = group_summary(cross, ["variant", "prototype_source_bucket", "difficulty_bucket", "stage_set"])
    for row in cross_rows:
        row["source_bucket"] = row.pop("prototype_source_bucket")
        row["target_bucket"] = row.pop("difficulty_bucket")
        row["direction"] = f"{row['source_bucket']}->{row['target_bucket']}"
    write_csv(args.out_dir / "v229_p2a_cross_bucket_directionality_report.csv", cross_rows)
    deploy = [r for r in rows if int(r["deployable_candidate"]) == 1 and r["action_name"] != "noop"]
    stage_rows = group_summary(deploy, ["variant", "stage_set", "action_family", "difficulty_bucket"])
    for row in stage_rows:
        row["target_bucket"] = row.pop("difficulty_bucket")
    write_csv(args.out_dir / "v229_p2a_stage_action_pruning_report.csv", stage_rows)


def fold_tail_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANTS:
        prefs = preference_rows(rows, variant, args)
        for fold in sorted({int(r["target_fold"]) for r in prefs}):
            subset = [r for r in prefs if int(r["target_fold"]) == fold]
            vals = [float(r["dpsnr"]) for r in subset]
            out.append(
                {
                    "variant": variant,
                    "target_fold": fold,
                    "sample_count": len(vals),
                    "mean_dpsnr": mean(vals),
                    "p05_dpsnr": percentile(vals, 5),
                    "cvar5_dpsnr": cvar(vals),
                    "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                    "tail_gate_pass": int(bool(vals) and percentile(vals, 5) >= -0.15 and sum(v <= SEVERE for v in vals) / len(vals) <= 0.035),
                }
            )
    return out


def policy_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["difficulty_bucket"]), str(row["stage_set"]), str(row["prototype_source_bucket"]), str(row["action_name"]))


def train_table_policy(rows: list[dict[str, Any]], variant: str, train_folds: set[int]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["variant"] == variant
        and int(row["deployable_candidate"]) == 1
        and row["action_name"] != "noop"
        and int(row["target_fold"]) in train_folds
        and row["prototype_source_bucket"] in ("all", row["difficulty_bucket"])
    ]
    table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(policy_key(row), []).append(row)
    for key, subset in groups.items():
        vals = [float(r["dpsnr"]) for r in subset]
        table[key] = {
            "key": key,
            "mean": mean(vals),
            "p05": percentile(vals, 5),
            "cvar5": cvar(vals),
            "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
            "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
            "lcb": percentile(vals, 20) - 0.15 * sum(v <= STRONG_REG for v in vals) / len(vals) if vals else -999.0,
        }
    return table


def run_table_policy(rows: list[dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = sorted({int(row["target_fold"]) for row in rows})
    selected_rows = []
    report_rows = []
    by_fold_name: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] == variant and int(row["deployable_candidate"]) == 1:
            by_fold_name.setdefault((int(row["target_fold"]), str(row["sample_name"])), []).append(row)
    for fold in folds:
        train_folds = {f for f in folds if f != fold}
        table = train_table_policy(rows, variant, train_folds)
        for (target_fold, sample_name), group in by_fold_name.items():
            if target_fold != fold:
                continue
            noop = next((r for r in group if r["action_name"] == "noop"), group[0])
            feasible = []
            for row in group:
                if row["action_name"] == "noop" or row["prototype_source_bucket"] not in ("all", row["difficulty_bucket"]):
                    continue
                key = policy_key(row)
                metric = table.get(key)
                if metric is None:
                    continue
                safe = metric["p05"] >= -0.15 and metric["cvar5"] >= -0.35 and metric["severe_rate"] <= 0.035
                if safe:
                    feasible.append((metric["lcb"], row, metric))
            if feasible:
                _score, chosen, metric = max(feasible, key=lambda item: item[0])
                reason = f"train_lcb={metric['lcb']:.6f};train_p05={metric['p05']:.6f}"
            else:
                chosen = noop
                reason = "no_safe_fold_table_action"
            selected = {
                "target_fold": fold,
                "policy_variant": variant,
                "sample_name": sample_name,
                "difficulty_bucket": chosen["difficulty_bucket"],
                "target_bucket": chosen["difficulty_bucket"],
                "stage_set": chosen["stage_set"],
                "chosen_source_bucket": chosen["prototype_source_bucket"],
                "chosen_action": chosen["action_name"],
                "chosen_strength": chosen["action_strength"],
                "selection_reason": reason,
                "dpsnr": chosen["dpsnr"],
                "is_unsafe": chosen["is_unsafe"],
                "is_severe": chosen["is_severe"],
            }
            selected_rows.append(selected)
    for fold in folds:
        for bucket in BUCKETS:
            subset = [r for r in selected_rows if int(r["target_fold"]) == fold and r["target_bucket"] == bucket]
            vals = [float(r["dpsnr"]) for r in subset]
            actions = sorted({str(r["chosen_action"]) for r in subset})
            report_rows.append(
                {
                    "target_fold": fold,
                    "policy_variant": variant,
                    "target_bucket": bucket,
                    "stage_set": "mixed",
                    "chosen_source_bucket": "table",
                    "chosen_action": "+".join(actions) if actions else "none",
                    "chosen_strength": "mixed",
                    "selection_reason": "fold_oof_table_policy",
                    "sample_count": len(subset),
                    "mean_dpsnr": mean(vals),
                    "p05_dpsnr": percentile(vals, 5),
                    "cvar5_dpsnr": cvar(vals),
                    "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
                    "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                }
            )
    return selected_rows, report_rows


def table_policy_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    all_reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected, reports = run_table_policy(rows, variant)
        all_selected.extend(selected)
        all_reports.extend(reports)
        s = summarize_policy_rows(selected)
        fold_tail = 0
        for fold in sorted({int(r["target_fold"]) for r in selected}):
            vals = [float(r["dpsnr"]) for r in selected if int(r["target_fold"]) == fold]
            if vals and percentile(vals, 5) >= -0.15 and sum(v <= SEVERE for v in vals) / len(vals) <= 0.035:
                fold_tail += 1
        row = {
            "policy_variant": variant,
            "table_policy_mean": s["mean"],
            "table_policy_hard": s["hard"],
            "table_policy_easy": s["easy"],
            "table_policy_p05": s["p05"],
            "table_policy_cvar5": s["cvar5"],
            "table_policy_severe_rate": s["severe_rate"],
            "table_policy_fold_tail_pass": fold_tail,
        }
        row["table_pass"] = int(
            row["table_policy_mean"] >= 0.30
            and row["table_policy_hard"] >= 0.60
            and row["table_policy_easy"] >= 0.0
            and row["table_policy_p05"] >= -0.15
            and row["table_policy_cvar5"] >= -0.35
            and row["table_policy_severe_rate"] <= 0.035
            and row["table_policy_fold_tail_pass"] >= 4
        )
        summary_rows.append(row)
    best = max(summary_rows, key=lambda r: (int(r["table_pass"]), float(r["table_policy_mean"])))
    return all_reports, best, all_selected


def write_oracle_selected_table_gap(
    args: argparse.Namespace,
    p1_rows: list[dict[str, Any]],
    safety_summary: list[dict[str, Any]],
    best_envelope: dict[str, Any],
    table_best: dict[str, Any],
) -> None:
    oracle_vals = [float(row["same_sample_oracle_dpsnr"]) for row in p1_rows]
    v228 = {
        "mean": 1.1377,
        "hard": 1.3769,
        "easy": 0.7035,
        "p05": 0.0,
        "cvar5": 0.0,
        "severe": 0.0,
        "unsafe": 0.0,
        "noop": 0.225,
    }
    table = [
        ("v2.27 same-sample oracle", mean(oracle_vals), float("nan"), float("nan"), percentile(oracle_vals, 5), cvar(oracle_vals), 0.0, 0.0, "n/a"),
        ("v2.28 OOF GT-preference selected", v228["mean"], v228["hard"], v228["easy"], v228["p05"], v228["cvar5"], v228["severe"], v228["unsafe"], v228["noop"]),
        (
            "v2.29 safe-envelope selected",
            best_envelope["selected_mean"],
            best_envelope["selected_hard"],
            best_envelope["selected_easy"],
            best_envelope["p05"],
            best_envelope["cvar5"],
            best_envelope["severe_rate"],
            0.0,
            best_envelope["noop_rate"],
        ),
        (
            "v2.29 GT-free table policy",
            table_best["table_policy_mean"],
            table_best["table_policy_hard"],
            table_best["table_policy_easy"],
            table_best["table_policy_p05"],
            table_best["table_policy_cvar5"],
            table_best["table_policy_severe_rate"],
            0.0,
            "n/a",
        ),
    ]
    lines = [
        "# v2.29 Oracle / Selected / Table-Policy Gap",
        "",
        "All v2.29 rows are train-derived OOF diagnostics. The table-policy row uses",
        "fold-out safety tables rather than per-sample target dPSNR preference.",
        "",
        "| replay/policy | mean | hard | easy | p05 | CVaR5 | severe | unsafe | no-op |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m, h, e, p05, cv, severe, unsafe, noop in table:
        lines.append(f"| {name} | {float(m):.6f} | {float(h):.6f} | {float(e):.6f} | {float(p05):.6f} | {float(cv):.6f} | {float(severe):.6f} | {float(unsafe):.6f} | {noop} |")
    lines.extend(
        [
            "",
            f"best_safe_envelope_variant: `{best_envelope['variant']}`",
            f"best_table_policy_variant: `{table_best['policy_variant']}`",
            "locked_test_touched: `false`",
            "training_launched: `false`",
        ]
    )
    write_text(args.out_dir / "v229_p2a_oracle_selected_table_policy_gap.md", "\n".join(lines))


def failonly_diagnostic_probe(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    work = [r for r in rows if r["variant"] == "raw_v228_baseline" and (int(r["deployable_candidate"]) == 1 or int(r["diagnostic_negative_control"]) == 1)]
    modes = ["state_action_proto", "compatibility_only"]
    targets = {
        "is_unsafe": lambda r: int(r["is_unsafe"]),
        "is_cross_bucket_unsafe": lambda r: int(r["action_name"] == "cross_bucket_mismatch" and int(r["is_unsafe"]) == 1),
        "is_hard_to_easy_mismatch_unsafe": lambda r: int(r["action_name"] == "cross_bucket_mismatch" and r["prototype_source_bucket"] == "hard_bottom25" and r["difficulty_bucket"] == "easy_top25" and int(r["is_unsafe"]) == 1),
        "is_overstrong_unsafe": lambda r: int(action_family(r["action_name"]) == "overstrong" and int(r["is_unsafe"]) == 1),
        "should_noop_under_safe_policy": lambda r: int(r["action_name"] == "noop" and r["difficulty_bucket"] == "easy_top25"),
    }
    out = []
    folds = sorted({int(r["target_fold"]) for r in work})
    for target, label_fn in targets.items():
        labels = [label_fn(r) for r in work]
        for mode in modes:
            scores: list[float] = []
            label_order: list[int] = []
            fold_aucs = []
            for fold in folds:
                valid = [r for r in work if int(r["target_fold"]) == fold]
                for row in valid:
                    if mode == "compatibility_only":
                        score = (
                            0.7 * float(row["delta_rms_to_target_ll_rms"])
                            + 0.5 * float(row["delta_absmax_to_target_ll_absmax"])
                            + 0.8 * float(row["bucket_distance"])
                            - 0.2 * float(row["stagewise_alignment_mean"])
                        )
                    else:
                        score = (
                            0.4 * float(row["action_strength"])
                            + 0.5 * float(row["delta_rms"])
                            + 0.2 * float(row["delta_abs_max"])
                            + 0.8 * float(row["bucket_distance"])
                            + 0.5 * float(row["diagnostic_negative_control"])
                            - 0.2 * float(row["delta_alignment"])
                        )
                    scores.append(score)
                    label_order.append(label_fn(row))
                fold_scores = scores[-len(valid) :]
                fold_labels = label_order[-len(valid) :]
                fold_aucs.append(roc_auc(fold_scores, fold_labels))
            out.append(
                {
                    "target": target,
                    "probe": mode,
                    "base_rate": mean([float(v) for v in labels]),
                    "auc": roc_auc(scores, label_order),
                    "ap": average_precision(scores, label_order),
                    "prob_std": std(scores),
                    "fold_mean_auc": mean([float(v) for v in fold_aucs]),
                    "fold_min_auc": min([float(v) for v in fold_aucs if math.isfinite(float(v))], default=float("nan")),
                    "decision": "FAILONLY_DIAGNOSTIC_ONLY_NO_SELECTOR_AUTHORIZATION",
                }
            )
    return out


def write_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    p2 = closeout.get("p2a", {})
    lines = [
        "# Haze4K v2.29 NoPost ILFRB-ACS Safe OOF Action-Bank Calibration Evidence",
        "",
        "Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration.md`",
        "",
        f"Status: `{closeout.get('decision', 'UNKNOWN')}`",
        "",
        "Runtime server: `convir-4090`",
        "",
        "Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`",
        "",
        "Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`",
        "",
        "Locked-test policy: blocked. Training is blocked.",
        "",
        "## Key Results",
        "",
        f"- P2A decision: `{p2.get('decision', 'UNKNOWN')}`",
        f"- Best safety-envelope variant: `{p2.get('best_safety_envelope_variant', 'n/a')}`",
        f"- Best table-policy variant: `{p2.get('best_table_policy_variant', 'n/a')}`",
        f"- Training launched: `{closeout.get('training_launched', False)}`",
        f"- Locked test touched: `{closeout.get('locked_test_touched', False)}`",
        "",
        "## Primary Files",
        "",
        "- `v229_p0_arch_contract_delta.md`",
        "- `v229_p2a_negative_control_taxonomy.csv`",
        "- `v229_p2a_deployable_candidate_tail_by_bucket_stage.csv`",
        "- `v229_p2a_safety_envelope_variant_summary.csv`",
        "- `v229_p2a_oof_table_policy_report.csv`",
        "- `v229_p2a_oracle_selected_table_policy_gap.md`",
        "- `v229_p2a_cross_bucket_directionality_report.csv`",
        "- `v229_p2a_stage_action_pruning_report.csv`",
        "- `v229_p2a_fold_tail_report.csv`",
        "- `v229_p2a_noop_useful_unsafe_base_rate_report.json`",
        "- `v229_p2a_failonly_diagnostic_probe.csv`",
        "- `v229_p2a_closeout.json`",
        "- `run_v229_p2a.sh`",
        "- `monitor_v229.sh`",
        "- `status.txt`",
        "",
        "This directory is compact text evidence only. It excludes checkpoints, weights, images, arrays, archives, and raw feature dumps.",
    ]
    write_text(args.out_dir / "README.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--delta-scale", type=float, default=0.25)
    ap.add_argument("--coverage-budget", type=float, default=0.35)
    ap.add_argument("--oracle-steps", type=int, default=10)
    ap.add_argument("--oracle-lr", type=float, default=0.08)
    ap.add_argument("--oracle-delta-scale", type=float, default=0.50)
    ap.add_argument("--oracle-reg", type=float, default=1e-4)
    ap.add_argument("--stage-sets", default="S6_early_mid_final,S5_bottleneck_mid,S4_final_decoder")
    ap.add_argument("--prototype-aggregate", choices=["median", "mean"], default="median")
    ap.add_argument("--conservative-penalty", type=float, default=0.08)
    ap.add_argument("--prototype-complexity-penalty", type=float, default=0.02)
    ap.add_argument("--bucket-distance-penalty", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=229)
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--parent-commit", default="82f4752")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    append_status(args, f"v229_start route_id={ROUTE_ID}")
    append_status(args, "locked_test_touched=false")
    append_status(args, "training_launched=false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V229_SAFE_OOF_ACTION_BANK_AUDIT_START device={device}", flush=True)
    closeout: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_branch": "codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit",
        "parent_commit": args.parent_commit,
        "locked_test_touched": False,
        "training_launched": False,
    }
    p0 = phase_p0(args)
    closeout["p0"] = p0
    if not p0["pass"]:
        closeout["decision"] = p0["decision"]
        write_json(args.out_dir / "v229_p2a_closeout.json", closeout)
        write_readme(args, closeout)
        append_status(args, f"v229_done decision={closeout['decision']}")
        return

    sample_rows, p1_rows, delta_bank = build_oracle_delta_bank(args, device)
    attach_sample_buckets(sample_rows)
    rows = run_variant_replay(args, sample_rows, delta_bank, device)
    print(f"V229_REPLAY_COMPLETE rows={len(rows)}", flush=True)

    safety_rows, best_envelope = safety_envelope_summary(rows, args)
    table_report, best_table, _selected = table_policy_summaries(rows)
    write_negative_taxonomy(rows, args)
    write_deployable_tail(rows, args)
    write_cross_and_stage_reports(rows, args)
    write_csv(args.out_dir / "v229_p2a_safety_envelope_variant_summary.csv", safety_rows)
    write_csv(args.out_dir / "v229_p2a_oof_table_policy_report.csv", table_report)
    write_csv(args.out_dir / "v229_p2a_fold_tail_report.csv", fold_tail_rows(rows, args))
    write_csv(args.out_dir / "v229_p2a_failonly_diagnostic_probe.csv", failonly_diagnostic_probe(rows, args))
    write_oracle_selected_table_gap(args, p1_rows, safety_rows, best_envelope, best_table)

    p2a_pass = bool(int(best_envelope["pass"]) and int(best_table["table_pass"]))
    p2a = {
        "decision": "P2A_PASS_SAFE_OOF_ACTION_BANK_CALIBRATION" if p2a_pass else "P2A_FAIL_SAFE_OOF_ACTION_BANK_CALIBRATION_PAUSE",
        "pass": p2a_pass,
        "best_safety_envelope_variant": best_envelope["variant"],
        "best_safety_envelope": best_envelope,
        "best_table_policy_variant": best_table["policy_variant"],
        "best_table_policy": best_table,
        "training_launched": False,
        "locked_test_touched": False,
    }
    closeout["p2a"] = p2a
    closeout["decision"] = p2a["decision"]
    write_json(args.out_dir / "v229_p2a_noop_useful_unsafe_base_rate_report.json", p2a)
    write_json(args.out_dir / "v229_p2a_closeout.json", closeout)
    write_readme(args, closeout)
    append_status(args, f"v229_done decision={closeout['decision']}")
    print("V229_SAFE_OOF_ACTION_BANK_AUDIT_OK " + closeout["decision"], flush=True)


if __name__ == "__main__":
    main()
