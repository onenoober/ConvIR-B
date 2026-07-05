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
    ACTION_STRENGTHS,
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


ROUTE_ID = "haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705"
DEPLOYABLE_ACTIONS = {
    "mild_0.33": 0.33,
    "medium_0.67": 0.67,
    "strong_1.25": 1.25,
}
DIAGNOSTIC_ACTIONS = {
    "overstrong_1.5": 1.50,
    "overstrong_2.0": 2.00,
    "overstrong_3.0": 3.00,
}
BUCKETS = ("hard_bottom25", "mid_50", "easy_top25")


def write_empty_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def row_mean(values: list[float]) -> float:
    vals = finite(values)
    return float(statistics.mean(vals)) if vals else float("nan")


def psnr_bucket(a0_psnr: float, hard_cut: float, easy_cut: float) -> str:
    if a0_psnr <= hard_cut:
        return "hard_bottom25"
    if a0_psnr >= easy_cut:
        return "easy_top25"
    return "mid_50"


def quartile_label(value: float, q1: float, q2: float, q3: float, prefix: str) -> str:
    if value <= q1:
        return f"{prefix}_q1"
    if value <= q2:
        return f"{prefix}_q2"
    if value <= q3:
        return f"{prefix}_q3"
    return f"{prefix}_q4"


def stage_channels(stage: str) -> int:
    return {"bottleneck": 128, "early": 128, "mid": 64, "final": 32}[stage]


def target_same_channel_stage(stage: str) -> str | None:
    if stage == "bottleneck":
        return "early"
    if stage == "early":
        return "bottleneck"
    return None


def tensor_median(items: list[torch.Tensor]) -> torch.Tensor:
    if len(items) == 1:
        return items[0].clone()
    return torch.stack([item.float() for item in items], dim=0).median(dim=0).values


def tensor_mean(items: list[torch.Tensor]) -> torch.Tensor:
    if len(items) == 1:
        return items[0].clone()
    return torch.stack([item.float() for item in items], dim=0).mean(dim=0)


def aggregate_tensors(items: list[torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "median":
        return tensor_median(items)
    if mode == "mean":
        return tensor_mean(items)
    raise ValueError(mode)


def transform_deltas(
    deltas: dict[str, torch.Tensor],
    action_name: str,
    strength: float,
) -> tuple[dict[str, torch.Tensor], bool]:
    if action_name == "noop":
        return {}, True
    if action_name == "sign_flip":
        return {stage: -delta * strength for stage, delta in deltas.items()}, True
    if action_name == "wrong_stage":
        shifted: dict[str, torch.Tensor] = {}
        for stage, delta in deltas.items():
            target = target_same_channel_stage(stage)
            if target is not None and stage_channels(target) == int(delta.shape[1]):
                shifted[target] = delta * strength
        return shifted, bool(shifted)
    return {stage: delta * strength for stage, delta in deltas.items()}, True


def delta_stats_for_target(deltas: dict[str, torch.Tensor], cache: dict[str, torch.Tensor]) -> dict[str, float]:
    if not deltas:
        return {
            "delta_abs_mean": 0.0,
            "delta_rms": 0.0,
            "delta_abs_max": 0.0,
            "delta_alignment": 0.0,
        }
    abs_means = []
    rms_vals = []
    max_vals = []
    aligns = []
    for stage, delta_cpu in deltas.items():
        delta = delta_cpu.to(cache[stage].device, dtype=cache[stage].dtype)
        ll, _, _, _, _, _ = haar_dwt(cache[stage])
        ll_grid = F.adaptive_avg_pool2d(ll.detach(), delta.shape[-2:])
        if delta.shape[-2:] != ll_grid.shape[-2:]:
            delta_for_align = F.interpolate(delta, size=ll_grid.shape[-2:], mode="bilinear", align_corners=False)
        else:
            delta_for_align = delta
        flat_d = delta_for_align.flatten(1)
        flat_ll = ll_grid.flatten(1)
        denom = flat_d.norm(dim=1) * flat_ll.norm(dim=1).clamp_min(1e-8)
        align = ((flat_d * flat_ll).sum(dim=1) / denom.clamp_min(1e-8)).detach().cpu()
        abs_means.append(float(delta.abs().mean().detach().cpu()))
        rms_vals.append(float(torch.sqrt(torch.mean(delta.float() ** 2)).detach().cpu()))
        max_vals.append(float(delta.abs().max().detach().cpu()))
        aligns.append(float(align.mean()))
    return {
        "delta_abs_mean": row_mean(abs_means),
        "delta_rms": row_mean(rms_vals),
        "delta_abs_max": max(max_vals),
        "delta_alignment": row_mean(aligns),
    }


def append_status(args: argparse.Namespace, line: str) -> None:
    status = args.out_dir / "status.txt"
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def phase_p0(args: argparse.Namespace) -> dict[str, Any]:
    scan = source_scan()
    report = {
        "phase": "p0_arch_contract_delta",
        "route": "v2.28",
        "parent_route": "v2.27",
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_commit": args.parent_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_delta": "none_from_v2_27_ilfrb_acs",
        "runtime_contract": "forward(self, x)",
        "new_runtime_training_launched": False,
        "locked_test_touched": False,
        "forbidden_symbol_hits": scan["hit_count"],
        "decision": "P0_PASS_ARCH_CONTRACT_DELTA_AUDIT" if scan["hit_count"] == 0 else "P0_FAIL_SOURCE_CONTRACT",
        "pass": scan["hit_count"] == 0,
    }
    lines = [
        "# v2.28 P0 Architecture Contract Delta",
        "",
        f"branch: `{report['branch']}`",
        f"commit: `{report['commit']}`",
        f"parent_branch: `codex/haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill`",
        f"parent_commit: `{args.parent_commit}`",
        f"checkpoint: `{args.checkpoint}`",
        f"checkpoint_sha256: `{report['checkpoint_sha256']}`",
        "",
        "v2.28 does not add a new model structure on top of v2.27. It reuses the",
        "NoPost ILFRB-ACS architecture and replaces only the action-bank replay",
        "diagnostic protocol with out-of-fold prototypes, cross-sample bucket swaps,",
        "and diagnostic negative controls.",
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
    write_text(args.out_dir / "v228_p0_arch_contract_delta.md", "\n".join(lines))
    return report


def build_oracle_delta_bank(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, torch.Tensor]]]]:
    a0, _route, _partial = build_models(args, device)
    samples = load_samples(args)
    stage_sets = [item for item in args.stage_sets.split(",") if item]
    if not stage_sets:
        raise RuntimeError("no stage sets selected")
    p1_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    print(f"V228_SELECTED_SAMPLES count={len(samples)} stage_sets={','.join(stage_sets)}", flush=True)
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
        sample_row = {
            "sample_name": sample.name,
            "target_fold": sample.fold,
            "a0_psnr": a0_psnr,
            **stats,
        }
        sample_rows.append(sample_row)
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
            print(f"V228_ORACLE_PROGRESS {idx}/{len(samples)}", flush=True)
    return sample_rows, p1_rows, delta_bank


def attach_sample_buckets(sample_rows: list[dict[str, Any]]) -> None:
    a0s = [float(row["a0_psnr"]) for row in sample_rows]
    lows = [float(row["input_low_std"]) for row in sample_rows]
    hard_cut = percentile(a0s, 25)
    easy_cut = percentile(a0s, 75)
    q1 = percentile(a0s, 25)
    q2 = percentile(a0s, 50)
    q3 = percentile(a0s, 75)
    lq1 = percentile(lows, 25)
    lq2 = percentile(lows, 50)
    lq3 = percentile(lows, 75)
    for row in sample_rows:
        row["difficulty_bucket"] = psnr_bucket(float(row["a0_psnr"]), hard_cut, easy_cut)
        row["a0_psnr_quartile"] = quartile_label(float(row["a0_psnr"]), q1, q2, q3, "a0")
        row["lowfreq_std_quartile"] = quartile_label(float(row["input_low_std"]), lq1, lq2, lq3, "lowstd")


def make_oof_prototype(
    sample_rows: list[dict[str, Any]],
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]],
    target_fold: int,
    source_bucket: str,
    stage_set: str,
    aggregate: str,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    source_names = [
        str(row["sample_name"])
        for row in sample_rows
        if int(row["target_fold"]) != target_fold
        and (source_bucket == "all" or str(row["difficulty_bucket"]) == source_bucket)
        and stage_set in delta_bank[str(row["sample_name"])]
    ]
    if not source_names:
        return None, 0
    stages = STAGE_SETS[stage_set]
    proto: dict[str, torch.Tensor] = {}
    for stage in stages:
        values = [delta_bank[name][stage_set][stage] for name in source_names if stage in delta_bank[name][stage_set]]
        if not values:
            return None, 0
        proto[stage] = aggregate_tensors(values, aggregate)
    return proto, len(source_names)


def opposite_bucket(bucket: str) -> str:
    if bucket == "hard_bottom25":
        return "easy_top25"
    if bucket == "easy_top25":
        return "hard_bottom25"
    return "hard_bottom25"


def prototype_id(target_fold: int, source_bucket: str, stage_set: str, aggregate: str) -> str:
    return f"oof_not_fold{target_fold}_{aggregate}_{source_bucket}_{stage_set}"


def add_replay_row(
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    gt: torch.Tensor,
    h: int,
    w: int,
    base: dict[str, Any],
    deltas: dict[str, torch.Tensor],
    action_name: str,
    strength: float,
    deployable: bool,
    diagnostic: bool,
    device: torch.device,
) -> None:
    transformed, valid = transform_deltas(deltas, action_name, strength)
    if not valid:
        return
    if transformed:
        pred = decode_from_features(model, cache, {k: v.to(device) for k, v in transformed.items()})[:, :, :h, :w]
        psnr = tensor_psnr(pred, gt)
    else:
        psnr = float(base["a0_psnr"])
    dpsnr = psnr - float(base["a0_psnr"])
    stats = delta_stats_for_target(transformed, cache)
    rows.append(
        {
            **base,
            "action_name": action_name,
            "action_strength": strength,
            "deployable_candidate": int(deployable),
            "diagnostic_negative_control": int(diagnostic),
            "action_psnr": psnr,
            "dpsnr": dpsnr,
            "is_noop_preferred": 0,
            "is_unsafe": int(dpsnr <= STRONG_REG),
            "is_severe": int(dpsnr <= SEVERE),
            "is_strong_regression": int(dpsnr <= STRONG_REG and str(base["difficulty_bucket"]) == "easy_top25"),
            **stats,
        }
    )


def run_oof_replay(
    args: argparse.Namespace,
    sample_rows: list[dict[str, Any]],
    p1_rows: list[dict[str, Any]],
    delta_bank: dict[str, dict[str, dict[str, torch.Tensor]]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a0, _route, _partial = build_models(args, device)
    selected = {str(row["sample_name"]): row for row in sample_rows}
    samples = [sample for sample in load_samples(args) if sample.name in selected]
    rows: list[dict[str, Any]] = []
    stage_sets = [item for item in args.stage_sets.split(",") if item]
    aggregate = args.prototype_aggregate
    for idx, sample in enumerate(samples, start=1):
        meta = selected[sample.name]
        x0 = image_tensor(sample.input_path, device)
        gt = image_tensor(sample.label_path, device)
        x, h, w = pad_to(x0)
        with torch.no_grad():
            cache = forward_cache(a0, x)
        cache = {key: value.detach() for key, value in cache.items()}
        target_bucket = str(meta["difficulty_bucket"])
        base = {
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
            "prototype_aggregate": aggregate,
        }
        add_replay_row(rows, a0, cache, gt, h, w, base, {}, "noop", 0.0, True, False, device)
        source_buckets = ["all", target_bucket, opposite_bucket(target_bucket)]
        if "mid_50" not in source_buckets:
            source_buckets.append("mid_50")
        for stage_set in stage_sets:
            for source_bucket in source_buckets:
                proto, source_count = make_oof_prototype(sample_rows, delta_bank, int(meta["target_fold"]), source_bucket, stage_set, aggregate)
                if proto is None:
                    continue
                proto_base = {
                    **base,
                    "prototype_id": prototype_id(int(meta["target_fold"]), source_bucket, stage_set, aggregate),
                    "prototype_source_bucket": source_bucket,
                    "stage_set": stage_set,
                    "prototype_source_count": source_count,
                }
                source_matches = source_bucket in ("all", target_bucket)
                if source_matches:
                    for action_name, strength in DEPLOYABLE_ACTIONS.items():
                        add_replay_row(rows, a0, cache, gt, h, w, proto_base, proto, action_name, strength, True, False, device)
                    for action_name, strength in DIAGNOSTIC_ACTIONS.items():
                        add_replay_row(rows, a0, cache, gt, h, w, proto_base, proto, action_name, strength, False, True, device)
                    add_replay_row(rows, a0, cache, gt, h, w, proto_base, proto, "sign_flip", 0.67, False, True, device)
                    add_replay_row(rows, a0, cache, gt, h, w, proto_base, proto, "wrong_stage", 0.67, False, True, device)
                else:
                    add_replay_row(rows, a0, cache, gt, h, w, proto_base, proto, "cross_bucket_mismatch", 0.67, False, True, device)
        if idx % args.print_freq == 0:
            print(f"V228_REPLAY_PROGRESS {idx}/{len(samples)} rows={len(rows)}", flush=True)
    for row in rows:
        if row["action_name"] == "noop":
            row["is_noop_preferred"] = 1
    return rows, p1_rows


def conservative_score(row: dict[str, Any], easy: bool, args: argparse.Namespace) -> float:
    return (
        float(row["dpsnr"])
        - args.conservative_penalty * float(row["action_strength"]) * (1.0 + float(easy))
        - args.prototype_complexity_penalty * float(row["stage_set"] != "noop")
    )


def deployable_preference_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["deployable_candidate"]) != 1:
            continue
        if row["action_name"] != "noop" and row["prototype_source_bucket"] not in ("all", row["difficulty_bucket"]):
            continue
        by_name.setdefault(str(row["sample_name"]), []).append(row)
    prefs: list[dict[str, Any]] = []
    for name, group in by_name.items():
        easy = str(group[0]["difficulty_bucket"]) == "easy_top25"
        best = max(group, key=lambda r: conservative_score(r, easy, args))
        raw = max(group, key=lambda r: float(r["dpsnr"]))
        prefs.append(
            {
                "sample_name": name,
                "target_fold": best["target_fold"],
                "a0_psnr": best["a0_psnr"],
                "difficulty_bucket": best["difficulty_bucket"],
                "a0_psnr_quartile": best["a0_psnr_quartile"],
                "lowfreq_std_quartile": best["lowfreq_std_quartile"],
                "prototype_source_bucket": best["prototype_source_bucket"],
                "stage_set": best["stage_set"],
                "raw_best_action": raw["action_name"],
                "raw_best_dpsnr": raw["dpsnr"],
                "conservative_best_action": best["action_name"],
                "conservative_best_dpsnr": best["dpsnr"],
                "conservative_score": conservative_score(best, easy, args),
            }
        )
    return prefs


def summarize_selected(prefs: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["conservative_best_dpsnr"]) for row in prefs]
    hard = [float(row["conservative_best_dpsnr"]) for row in prefs if row["difficulty_bucket"] == "hard_bottom25"]
    easy = [float(row["conservative_best_dpsnr"]) for row in prefs if row["difficulty_bucket"] == "easy_top25"]
    return {
        "count": len(vals),
        "mean_dpsnr": mean(vals),
        "hard_bottom25_dpsnr": mean(hard),
        "easy_top25_dpsnr": mean(easy),
        "p05_dpsnr": percentile(vals, 5),
        "cvar5_dpsnr": cvar(vals, 5),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_regression_rate": sum(v <= STRONG_REG for v in easy) / len(easy) if easy else float("nan"),
        "wrong_direction_rate": sum(v < 0 for v in vals) / len(vals) if vals else float("nan"),
    }


def summarize_preferences(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    group_specs = [
        ("difficulty_bucket", ["difficulty_bucket"]),
        ("a0_psnr_quartile", ["a0_psnr_quartile"]),
        ("lowfreq_std_quartile", ["lowfreq_std_quartile"]),
        ("prototype_source_bucket", ["prototype_source_bucket"]),
        ("stage_set", ["stage_set"]),
        ("bucket_source_stage", ["difficulty_bucket", "prototype_source_bucket", "stage_set"]),
    ]
    deployable = [
        row
        for row in rows
        if int(row["deployable_candidate"]) == 1
        and (row["action_name"] == "noop" or row["prototype_source_bucket"] in ("all", row["difficulty_bucket"]))
    ]
    out: list[dict[str, Any]] = []
    for group_type, keys in group_specs:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in deployable:
            if row["action_name"] == "noop" and any(k in ("prototype_source_bucket", "stage_set") for k in keys):
                continue
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
        for values, group in groups.items():
            by_name: dict[str, list[dict[str, Any]]] = {}
            for row in group:
                by_name.setdefault(str(row["sample_name"]), []).append(row)
            prefs = []
            for name, sample_group in by_name.items():
                easy = str(sample_group[0]["difficulty_bucket"]) == "easy_top25"
                prefs.append(max(sample_group, key=lambda r: conservative_score(r, easy, args)))
            vals = [float(row["dpsnr"]) for row in prefs]
            actions = [str(row["action_name"]) for row in prefs]
            out.append(
                {
                    "group_type": group_type,
                    "difficulty_bucket": values[keys.index("difficulty_bucket")] if "difficulty_bucket" in keys else "ALL",
                    "a0_psnr_quartile": values[keys.index("a0_psnr_quartile")] if "a0_psnr_quartile" in keys else "ALL",
                    "lowfreq_std_quartile": values[keys.index("lowfreq_std_quartile")] if "lowfreq_std_quartile" in keys else "ALL",
                    "prototype_source_bucket": values[keys.index("prototype_source_bucket")] if "prototype_source_bucket" in keys else "ALL",
                    "stage_set": values[keys.index("stage_set")] if "stage_set" in keys else "ALL",
                    "sample_count": len(prefs),
                    "noop_count": sum(a == "noop" for a in actions),
                    "mild_count": sum(a == "mild_0.33" for a in actions),
                    "medium_count": sum(a == "medium_0.67" for a in actions),
                    "strong_count": sum(a == "strong_1.25" for a in actions),
                    "overstrong_count": 0,
                    "unsafe_count": sum(v <= STRONG_REG for v in vals),
                    "mean_dpsnr": mean(vals),
                    "p05_dpsnr": percentile(vals, 5),
                    "cvar5_dpsnr": cvar(vals, 5),
                    "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                    "strong_regression_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
                }
            )
    return out


def strength_safety_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for action in [
        "noop",
        "mild_0.33",
        "medium_0.67",
        "strong_1.25",
        "overstrong_1.5",
        "overstrong_2.0",
        "overstrong_3.0",
        "sign_flip",
        "wrong_stage",
        "cross_bucket_mismatch",
    ]:
        subset = [row for row in rows if row["action_name"] == action]
        vals = [float(row["dpsnr"]) for row in subset]
        curve.append(
            {
                "action": action,
                "action_strength": subset[0]["action_strength"] if subset else float("nan"),
                "deployable_candidate": subset[0]["deployable_candidate"] if subset else 0,
                "diagnostic_negative_control": subset[0]["diagnostic_negative_control"] if subset else int(action not in ("noop", "mild_0.33", "medium_0.67", "strong_1.25")),
                "count": len(subset),
                "mean_dpsnr": mean(vals),
                "p05_dpsnr": percentile(vals, 5),
                "cvar5_dpsnr": cvar(vals, 5),
                "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
            }
        )
    return curve


def cross_sample_swap_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    keys = sorted(
        {
            (row["prototype_source_bucket"], row["difficulty_bucket"], row["stage_set"], row["action_name"])
            for row in rows
            if row["prototype_source_bucket"] in BUCKETS and row["stage_set"] != "noop"
        }
    )
    for source_bucket, target_bucket, stage_set, action in keys:
        subset = [
            row
            for row in rows
            if row["prototype_source_bucket"] == source_bucket
            and row["difficulty_bucket"] == target_bucket
            and row["stage_set"] == stage_set
            and row["action_name"] == action
        ]
        vals = [float(row["dpsnr"]) for row in subset]
        matrix.append(
            {
                "prototype_source_bucket": source_bucket,
                "target_bucket": target_bucket,
                "stage_set": stage_set,
                "action": action,
                "count": len(subset),
                "mean_dpsnr": mean(vals),
                "p05_dpsnr": percentile(vals, 5),
                "unsafe_rate": sum(v <= STRONG_REG for v in vals) / len(vals) if vals else float("nan"),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
            }
        )
    return matrix


def fold_tail_report(prefs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fold in sorted({int(row["target_fold"]) for row in prefs}):
        subset = [row for row in prefs if int(row["target_fold"]) == fold]
        vals = [float(row["conservative_best_dpsnr"]) for row in subset]
        out.append(
            {
                "target_fold": fold,
                "sample_count": len(vals),
                "mean_dpsnr": mean(vals),
                "p05_dpsnr": percentile(vals, 5),
                "cvar5_dpsnr": cvar(vals, 5),
                "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
                "tail_gate_pass": int(bool(vals) and percentile(vals, 5) >= -0.15 and sum(v <= SEVERE for v in vals) / len(vals) <= 0.035),
            }
        )
    return out


def write_oracle_vs_oof_gap(args: argparse.Namespace, p1_rows: list[dict[str, Any]], rows: list[dict[str, Any]], prefs: list[dict[str, Any]]) -> None:
    oracle_vals = [float(row["same_sample_oracle_dpsnr"]) for row in p1_rows]
    oracle_hard = [float(row["same_sample_oracle_dpsnr"]) for row in p1_rows if row["sample_name"] in {p["sample_name"] for p in prefs if p["difficulty_bucket"] == "hard_bottom25"}]
    oracle_easy = [float(row["same_sample_oracle_dpsnr"]) for row in p1_rows if row["sample_name"] in {p["sample_name"] for p in prefs if p["difficulty_bucket"] == "easy_top25"}]
    selected = summarize_selected(prefs)
    cross = [float(row["dpsnr"]) for row in rows if row["action_name"] == "cross_bucket_mismatch"]
    neg = [float(row["dpsnr"]) for row in rows if int(row["diagnostic_negative_control"]) == 1]
    table = [
        ("same-sample oracle", mean(oracle_vals), mean(oracle_hard), mean(oracle_easy), percentile(oracle_vals, 5), sum(v <= STRONG_REG for v in oracle_vals) / len(oracle_vals) if oracle_vals else 0.0, "n/a"),
        ("OOF prototype selected", selected["mean_dpsnr"], selected["hard_bottom25_dpsnr"], selected["easy_top25_dpsnr"], selected["p05_dpsnr"], selected["strong_reference_regression_rate"], str(args.last_noop_rate)),
        ("cross-bucket", mean(cross), float("nan"), float("nan"), percentile(cross, 5), sum(v <= STRONG_REG for v in cross) / len(cross) if cross else float("nan"), "n/a"),
        ("negative controls", mean(neg), float("nan"), float("nan"), percentile(neg, 5), sum(v <= STRONG_REG for v in neg) / len(neg) if neg else float("nan"), "n/a"),
    ]
    lines = [
        "# v2.28 Oracle vs OOF Gap",
        "",
        "This table compares v2.27-style same-sample GT oracle deltas with v2.28",
        "out-of-fold prototype replay. All rows are train-derived; locked test is untouched.",
        "",
        "| replay type | mean | hard | easy | p05 | unsafe | no-op pref |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m, h, e, p05, unsafe, noop in table:
        lines.append(f"| {name} | {m:.6f} | {h:.6f} | {e:.6f} | {p05:.6f} | {unsafe:.6f} | {noop} |")
    lines.extend(
        [
            "",
            f"sample_count: `{len(prefs)}`",
            f"stage_sets: `{args.stage_sets}`",
            f"prototype_aggregate: `{args.prototype_aggregate}`",
            "locked_test_touched: `false`",
        ]
    )
    write_text(args.out_dir / "v228_p2a_oracle_vs_oof_gap.md", "\n".join(lines))


def p2a_report_and_gate(args: argparse.Namespace, rows: list[dict[str, Any]], p1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefs = deployable_preference_rows(rows, args)
    for pref in prefs:
        for row in rows:
            if row["sample_name"] == pref["sample_name"] and row["action_name"] == pref["conservative_best_action"]:
                row["is_noop_preferred"] = int(pref["conservative_best_action"] == "noop")
    pref_group_rows = summarize_preferences(rows, args)
    curve = strength_safety_curve(rows)
    swap = cross_sample_swap_matrix(rows)
    folds = fold_tail_report(prefs)
    selected = summarize_selected(prefs)
    sample_count = len(prefs)
    noop_rate = sum(row["conservative_best_action"] == "noop" for row in prefs) / sample_count if sample_count else 0.0
    easy = [row for row in prefs if row["difficulty_bucket"] == "easy_top25"]
    hard = [row for row in prefs if row["difficulty_bucket"] == "hard_bottom25"]
    easy_noop_mild = sum(row["conservative_best_action"] in ("noop", "mild_0.33") for row in easy) / len(easy) if easy else 0.0
    hard_medium_strong = sum(row["conservative_best_action"] in ("medium_0.67", "strong_1.25") for row in hard) / len(hard) if hard else 0.0
    diagnostic = [row for row in rows if int(row["diagnostic_negative_control"]) == 1]
    unsafe_rate = sum(float(row["dpsnr"]) <= STRONG_REG for row in diagnostic) / len(diagnostic) if diagnostic else float("nan")
    fold_tail_pass = sum(int(row["tail_gate_pass"]) for row in folds)
    args.last_noop_rate = noop_rate
    report = {
        "sample_count": sample_count,
        "overall_noop_conservative_preference_rate": noop_rate,
        "easy_top25_noop_or_mild_preference_rate": easy_noop_mild,
        "hard_bottom25_medium_or_strong_preference_rate": hard_medium_strong,
        "diagnostic_negative_control_unsafe_rate": unsafe_rate,
        "deployable_selected": selected,
        "fold_tail_pass": fold_tail_pass,
    }
    p2_pass = (
        0.10 <= noop_rate <= 0.45
        and easy_noop_mild >= 0.40
        and hard_medium_strong >= 0.30
        and math.isfinite(unsafe_rate)
        and 0.05 <= unsafe_rate <= 0.40
        and selected["mean_dpsnr"] >= 0.20
        and selected["hard_bottom25_dpsnr"] >= 0.50
        and selected["easy_top25_dpsnr"] >= 0.0
        and selected["p05_dpsnr"] >= -0.15
        and selected["cvar5_dpsnr"] >= -0.35
        and selected["severe_rate"] <= 0.035
        and selected["strong_reference_regression_rate"] <= 0.075
        and fold_tail_pass >= 4
    )
    report["decision"] = "P2A_PASS_OOF_ACTION_BANK_STRATIFICATION" if p2_pass else "P2A_FAIL_OOF_ACTION_BANK_STRATIFICATION_PAUSE"
    report["pass"] = p2_pass
    write_csv(args.out_dir / "v228_p2a_oof_prototype_action_bank_replay.csv", rows)
    write_csv(args.out_dir / "v228_p2a_action_preference_by_bucket.csv", pref_group_rows)
    write_csv(args.out_dir / "v228_p2a_strength_safety_curve.csv", curve)
    write_json(args.out_dir / "v228_p2a_noop_unsafe_base_rate_report.json", report)
    write_csv(args.out_dir / "v228_p2a_cross_sample_swap_matrix.csv", swap)
    write_csv(args.out_dir / "v228_p2a_fold_tail_report.csv", folds)
    write_oracle_vs_oof_gap(args, p1_rows, rows, prefs)
    return report


def feature_maps(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "stage_set": sorted({str(row["stage_set"]) for row in rows}),
        "prototype_source_bucket": sorted({str(row["prototype_source_bucket"]) for row in rows}),
        "difficulty_bucket": sorted({str(row["difficulty_bucket"]) for row in rows}),
    }


def feature_vector(row: dict[str, Any], mode: str, maps: dict[str, list[str]]) -> list[float]:
    state = [
        float(row["a0_psnr"]),
        float(row["input_luma_mean"]),
        float(row["input_luma_std"]),
        float(row["input_low_mean"]),
        float(row["input_low_std"]),
    ]
    action = [
        float(row["action_strength"]),
        float(row["delta_abs_mean"]),
        float(row["delta_rms"]),
        float(row["delta_abs_max"]),
        float(row["delta_alignment"]),
        float(row["deployable_candidate"]),
        float(row["diagnostic_negative_control"]),
    ]
    one_hot = []
    for key in ("stage_set", "prototype_source_bucket", "difficulty_bucket"):
        one_hot.extend([float(row[key] == value) for value in maps[key]])
    if mode == "state_only":
        return state
    if mode == "old_action_stats":
        return state[:1] + action[:4]
    if mode == "rich_action_stats":
        return state + action
    if mode == "prototype_distance":
        return state + [float(row["delta_alignment"]), float(row["delta_rms"]), float(row["prototype_source_bucket"] == row["difficulty_bucket"])]
    if mode == "stagewise_features":
        return state + one_hot
    if mode == "state_plus_action":
        return state + action[:5]
    if mode == "state_plus_action_plus_bucket":
        return state + action + one_hot
    raise ValueError(mode)


def fit_probe(rows: list[dict[str, Any]], mode: str, target: str, args: argparse.Namespace, maps: dict[str, list[str]]) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_scores: dict[int, float] = {}
    labels_all = [int(row[target]) for row in rows]
    folds = sorted({int(row["target_fold"]) for row in rows})
    fold_rows = []
    for fold in folds:
        train_idx = [i for i, row in enumerate(rows) if int(row["target_fold"]) != fold]
        valid_idx = [i for i, row in enumerate(rows) if int(row["target_fold"]) == fold]
        y_train = torch.tensor([int(rows[i][target]) for i in train_idx], dtype=torch.float32).view(-1, 1)
        if len(set(int(v.item()) for v in y_train.flatten())) < 2:
            base = float(y_train.mean()) if len(y_train) else 0.0
            for i in valid_idx:
                out_scores[i] = base
            fold_rows.append({"fold": fold, "auc": float("nan"), "ap": float("nan"), "prob_std": 0.0})
            continue
        x_train = torch.tensor([feature_vector(rows[i], mode, maps) for i in train_idx], dtype=torch.float32)
        x_valid = torch.tensor([feature_vector(rows[i], mode, maps) for i in valid_idx], dtype=torch.float32)
        mu = x_train.mean(dim=0, keepdim=True)
        sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mu) / sigma
        x_valid = (x_valid - mu) / sigma
        model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 24), torch.nn.GELU(), torch.nn.Linear(24, 1))
        opt = torch.optim.Adam(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
        pos_weight = (len(y_train) - y_train.sum()).clamp_min(1.0) / y_train.sum().clamp_min(1.0)
        for _ in range(args.probe_epochs):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x_train), y_train, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        with torch.no_grad():
            probs = torch.sigmoid(model(x_valid)).flatten().tolist()
        labels = [int(rows[i][target]) for i in valid_idx]
        for i, p in zip(valid_idx, probs):
            out_scores[i] = float(p)
        fold_rows.append({"fold": fold, "auc": roc_auc(probs, labels), "ap": average_precision(probs, labels), "prob_std": std(probs)})
    scores = [out_scores.get(i, 0.0) for i in range(len(rows))]
    return {
        "target": target,
        "ablation": mode,
        "base_rate": mean([float(v) for v in labels_all]),
        "auc": roc_auc(scores, labels_all),
        "ap": average_precision(scores, labels_all),
        "prob_std": std(scores),
        "fold_mean_auc": mean([float(row["auc"]) for row in fold_rows]),
        "fold_min_auc": min([float(row["auc"]) for row in fold_rows if math.isfinite(float(row["auc"]))], default=float("nan")),
        "fold_mean_ap": mean([float(row["ap"]) for row in fold_rows]),
        "fold_min_ap": min([float(row["ap"]) for row in fold_rows if math.isfinite(float(row["ap"]))], default=float("nan")),
    }


def run_probe_if_allowed(args: argparse.Namespace, rows: list[dict[str, Any]], p2: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "target",
        "ablation",
        "base_rate",
        "auc",
        "ap",
        "prob_std",
        "fold_mean_auc",
        "fold_min_auc",
        "fold_mean_ap",
        "fold_min_ap",
        "decision",
    ]
    if not bool(p2["pass"]):
        write_csv(
            args.out_dir / "v228_p2b_probe_feature_ablation.csv",
            [{"target": "skipped", "ablation": "P2A_not_passed", "decision": "P2B_SKIPPED_P2A_GATE_FAIL"}],
        )
        return {"decision": "P2B_SKIPPED_P2A_GATE_FAIL", "pass": False}
    work = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["action_name"] == "noop" or int(row["deployable_candidate"]) == 1 or int(row["diagnostic_negative_control"]) == 1:
            by_name.setdefault(str(row["sample_name"]), []).append(row)
    for group in by_name.values():
        best = max(group, key=lambda r: float(r["dpsnr"]) - 0.03 * float(r["action_strength"]))
        for row in group:
            row = dict(row)
            row["should_noop"] = int(best["action_name"] == "noop")
            row["should_medium_or_strong"] = int(best["action_name"] in ("medium_0.67", "strong_1.25"))
            row["best_action_class"] = best["action_name"]
            work.append(row)
    maps = feature_maps(work)
    modes = [
        "state_only",
        "old_action_stats",
        "rich_action_stats",
        "prototype_distance",
        "stagewise_features",
        "state_plus_action",
        "state_plus_action_plus_bucket",
    ]
    targets = ["should_noop", "should_medium_or_strong", "is_unsafe"]
    out = []
    for target in targets:
        for mode in modes:
            summary = fit_probe(work, mode, target, args, maps)
            summary["decision"] = "P2B_PROBE_COMPLETED"
            out.append(summary)
    if not out:
        write_empty_csv(args.out_dir / "v228_p2b_probe_feature_ablation.csv", fields)
    else:
        write_csv(args.out_dir / "v228_p2b_probe_feature_ablation.csv", out)
    best = max(out, key=lambda row: float(row["auc"]) if math.isfinite(float(row["auc"])) else -1.0)
    passed = float(best["auc"]) >= 0.75 and float(best["ap"]) >= max(0.30, 2.0 * float(best["base_rate"])) and float(best["prob_std"]) >= 0.05
    return {
        "decision": "P2B_PASS_FEATURE_SEPARABILITY_DRY_RUN" if passed else "P2B_FAIL_FEATURE_SEPARABILITY_DRY_RUN_PAUSE",
        "pass": passed,
        "best": best,
    }


def write_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    p2 = closeout.get("p2a", {})
    lines = [
        "# Haze4K v2.28 NoPost ILFRB-ACS Action-Bank Stratification Audit Evidence",
        "",
        "Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit.md`",
        "",
        f"Status: `{closeout.get('decision', 'UNKNOWN')}`",
        "",
        "Runtime server: `convir-4090`",
        "",
        "Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`",
        "",
        "Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`",
        "",
        "Locked-test policy: blocked. This audit uses train-derived samples only.",
        "",
        "## Key Results",
        "",
        f"- P2A decision: `{p2.get('decision', 'UNKNOWN')}`",
        f"- Overall conservative no-op preference rate: `{p2.get('overall_noop_conservative_preference_rate', 'n/a')}`",
        f"- Easy top25 no-op/mild preference rate: `{p2.get('easy_top25_noop_or_mild_preference_rate', 'n/a')}`",
        f"- Hard bottom25 medium/strong preference rate: `{p2.get('hard_bottom25_medium_or_strong_preference_rate', 'n/a')}`",
        f"- Diagnostic unsafe rate: `{p2.get('diagnostic_negative_control_unsafe_rate', 'n/a')}`",
        f"- Training launched: `{closeout.get('training_launched', False)}`",
        f"- Locked test touched: `{closeout.get('locked_test_touched', False)}`",
        "",
        "## Primary Files",
        "",
        "- `v228_p0_arch_contract_delta.md`",
        "- `v228_p2a_oof_prototype_action_bank_replay.csv`",
        "- `v228_p2a_action_preference_by_bucket.csv`",
        "- `v228_p2a_strength_safety_curve.csv`",
        "- `v228_p2a_noop_unsafe_base_rate_report.json`",
        "- `v228_p2a_cross_sample_swap_matrix.csv`",
        "- `v228_p2a_oracle_vs_oof_gap.md`",
        "- `v228_p2a_fold_tail_report.csv`",
        "- `v228_p2b_probe_feature_ablation.csv`",
        "- `v228_closeout.json`",
        "- `run_v228_p2a.sh`",
        "- `monitor_v228.sh`",
        "- `status.txt`",
        "",
        "This directory is compact text evidence only. It intentionally excludes checkpoints, weights, images, arrays, archives, and raw feature dumps.",
    ]
    write_text(args.out_dir / "README.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--p0-images", type=int, default=8)
    ap.add_argument("--identity-tol", type=float, default=1e-6)
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
    ap.add_argument("--probe-epochs", type=int, default=220)
    ap.add_argument("--probe-lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=228)
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--parent-commit", default="de5a68b")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_status(args, f"v228_start device={device} route_id={ROUTE_ID}")
    append_status(args, "locked_test_touched=false")
    append_status(args, "training_launched=false")
    print(f"V228_ACTION_BANK_AUDIT_START device={device}", flush=True)
    closeout: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_branch": "codex/haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill",
        "parent_commit": args.parent_commit,
        "locked_test_touched": False,
        "training_launched": False,
    }
    p0 = phase_p0(args)
    closeout["p0"] = p0
    if not p0["pass"]:
        closeout["decision"] = p0["decision"]
        write_json(args.out_dir / "v228_closeout.json", closeout)
        write_readme(args, closeout)
        append_status(args, f"v228_done decision={closeout['decision']}")
        print("V228_P0_FAILED_STOP", flush=True)
        return
    sample_rows, p1_rows, delta_bank = build_oracle_delta_bank(args, device)
    attach_sample_buckets(sample_rows)
    write_csv(args.out_dir / "v228_p2a_same_sample_oracle_delta_summary.csv", p1_rows)
    rows, p1_rows = run_oof_replay(args, sample_rows, p1_rows, delta_bank, device)
    p2a = p2a_report_and_gate(args, rows, p1_rows)
    closeout["p2a"] = p2a
    p2b = run_probe_if_allowed(args, rows, p2a)
    closeout["p2b"] = p2b
    closeout["decision"] = p2b["decision"] if p2a["pass"] else p2a["decision"]
    write_json(args.out_dir / "v228_closeout.json", closeout)
    write_readme(args, closeout)
    append_status(args, f"v228_done decision={closeout['decision']}")
    print("V228_ACTION_BANK_AUDIT_OK " + str(closeout["decision"]), flush=True)


if __name__ == "__main__":
    main()
