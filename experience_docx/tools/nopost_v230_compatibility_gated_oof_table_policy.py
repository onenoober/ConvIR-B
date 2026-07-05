#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from nopost_v227_ilfrb_acs_diagnostics import (  # noqa: E402
    SEVERE,
    STRONG_REG,
    average_precision,
    cvar,
    git,
    mean,
    percentile,
    sha256_file,
    source_scan,
    std,
    write_csv,
    write_json,
    write_text,
)
from nopost_v229_safe_oof_action_bank_calibration import (  # noqa: E402
    VARIANTS,
    action_family,
    append_status,
    attach_sample_buckets,
    bucket_distance,
    build_oracle_delta_bank,
    control_family,
    deployable_actions_for_variant,
    json_clean,
    preference_rows,
    run_variant_replay,
    summarize_policy_rows,
)


ROUTE_ID = "haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705"
BUCKETS = ("hard_bottom25", "mid_50", "easy_top25")
SIGNAL_VARIANT = "bucket_strength_grid"
SAFETY_VARIANT = "energy_norm_plus_bucket_strength_plus_alignment_gate"
LCB_POLICY = "v230_hybrid_compat_firewall_lcb"
FIREWALL_VARIANTS = (
    "no_firewall",
    "forbid_hard_source_to_easy_target",
    "forbid_opposite_bucket",
    "allow_adjacent_bucket_only",
    "same_bucket_only",
    "same_bucket_plus_mid_bridge",
    "compatibility_distance_threshold",
    "compatibility_distance_threshold_plus_energy_norm",
)


def phase_p0(args: argparse.Namespace) -> dict[str, Any]:
    scan = source_scan()
    report = {
        "phase": "p0_arch_contract_delta",
        "route": "v2.30",
        "parent_route": "v2.29",
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_commit": args.parent_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_delta": "none_from_v2_29_ilfrb_acs",
        "runtime_contract": "forward(self, x)",
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
        "forbidden_symbol_hits": scan["hit_count"],
        "decision": "P0_PASS_ARCH_CONTRACT_DELTA_AUDIT" if scan["hit_count"] == 0 else "P0_FAIL_SOURCE_CONTRACT",
        "pass": scan["hit_count"] == 0,
    }
    lines = [
        "# v2.30 P0 Architecture Contract Delta",
        "",
        f"branch: `{report['branch']}`",
        f"commit: `{report['commit']}`",
        f"parent_branch: `codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`",
        f"parent_commit: `{args.parent_commit}`",
        f"checkpoint: `{args.checkpoint}`",
        f"checkpoint_sha256: `{report['checkpoint_sha256']}`",
        "",
        "v2.30 does not add a new runtime model structure. It reuses the v2.29",
        "NoPost ILFRB-ACS snapshot and changes only the train-derived P2A",
        "compatibility-gated table-policy audit.",
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
    write_text(args.out_dir / "v230_p0_arch_contract_delta.md", "\n".join(lines))
    return report


def safe_rate(vals: list[float], threshold: float) -> float:
    return sum(v <= threshold for v in vals) / len(vals) if vals else float("nan")


def accepted_risk_rate(vals: list[float], threshold: float) -> float:
    return safe_rate(vals, threshold) if vals else 0.0


def finite_or(value: float, default: float) -> float:
    return value if math.isfinite(float(value)) else default


def compat_bin(row: dict[str, Any]) -> str:
    source = str(row["prototype_source_bucket"])
    target = str(row["difficulty_bucket"])
    if source == "noop":
        return "noop"
    if source == "all":
        source_class = "all_source"
    elif source == target:
        source_class = "same_bucket"
    elif source == "hard_bottom25" and target == "easy_top25":
        return "hard_to_easy_forbidden"
    elif source == "easy_top25" and target == "hard_bottom25":
        return "easy_to_hard_opposite"
    elif "mid_50" in (source, target):
        source_class = "adjacent_bucket"
    else:
        source_class = "opposite_bucket"
    align = float(row.get("stagewise_alignment_mean", 0.0))
    rms_ratio = float(row.get("delta_rms_to_target_ll_rms", 0.0))
    abs_ratio = float(row.get("delta_absmax_to_target_ll_absmax", 0.0))
    if align >= 0.12 and rms_ratio <= 0.030 and abs_ratio <= 0.80:
        quality = "good"
    elif align >= 0.05 and rms_ratio <= 0.045 and abs_ratio <= 1.20:
        quality = "borderline"
    else:
        quality = "weak"
    return f"{source_class}_{quality}"


def row_risk_score(row: dict[str, Any], mode: str) -> float:
    strength = float(row.get("action_strength", 0.0))
    distance = float(row.get("bucket_distance", 0.0))
    align = float(row.get("stagewise_alignment_mean", 0.0))
    rms_ratio = float(row.get("delta_rms_to_target_ll_rms", 0.0))
    abs_ratio = float(row.get("delta_absmax_to_target_ll_absmax", 0.0))
    target = str(row.get("difficulty_bucket", ""))
    source = str(row.get("prototype_source_bucket", ""))
    hard_to_easy = float(source == "hard_bottom25" and target == "easy_top25")
    overstrong = float(action_family(str(row.get("action_name", ""))) == "overstrong")
    if mode == "bucket_stage_strength":
        return 0.55 * strength + 0.75 * distance + 1.25 * hard_to_easy + 0.65 * overstrong
    if mode == "compatibility_energy_alignment":
        return 0.80 * rms_ratio + 0.50 * abs_ratio - 0.45 * align + 0.70 * distance + 1.00 * hard_to_easy
    return (
        0.40 * strength
        + 0.45 * rms_ratio
        + 0.25 * abs_ratio
        - 0.30 * align
        + 0.65 * distance
        + 1.20 * hard_to_easy
        + 0.45 * overstrong
    )


def compatibility_pass(row: dict[str, Any], strict: bool = False) -> bool:
    action = str(row["action_name"])
    family = action_family(action)
    if action == "noop":
        return True
    source = str(row["prototype_source_bucket"])
    target = str(row["difficulty_bucket"])
    if source == "hard_bottom25" and target == "easy_top25":
        return False
    if bucket_distance(source, target) > (0 if strict else 1):
        return False
    if family in ("medium", "strong", "overstrong"):
        if float(row.get("stagewise_alignment_mean", 0.0)) < (0.10 if strict else 0.06):
            return False
        if float(row.get("delta_rms_to_target_ll_rms", 0.0)) > (0.035 if strict else 0.050):
            return False
    if target == "easy_top25" and family not in ("noop", "mild"):
        return False
    return True


def firewall_accept(row: dict[str, Any], firewall: str) -> bool:
    if row["action_name"] == "noop":
        return True
    source = str(row["prototype_source_bucket"])
    target = str(row["difficulty_bucket"])
    if firewall == "no_firewall":
        return True
    if firewall == "forbid_hard_source_to_easy_target":
        return not (source == "hard_bottom25" and target == "easy_top25")
    if firewall == "forbid_opposite_bucket":
        return bucket_distance(source, target) < 2
    if firewall == "allow_adjacent_bucket_only":
        return bucket_distance(source, target) <= 1
    if firewall == "same_bucket_only":
        return source in ("all", target)
    if firewall == "same_bucket_plus_mid_bridge":
        return source in ("all", target) or "mid_50" in (source, target)
    if firewall == "compatibility_distance_threshold":
        return compatibility_pass(row, strict=False)
    if firewall == "compatibility_distance_threshold_plus_energy_norm":
        return compatibility_pass(row, strict=True)
    raise ValueError(f"unknown firewall {firewall}")


def row_for_policy(row: dict[str, Any], policy: str) -> bool:
    if int(row.get("deployable_candidate", 0)) != 1:
        return False
    if row["action_name"] == "noop":
        return True
    if policy == "signal_bucket_strength_grid":
        return row["variant"] == SIGNAL_VARIANT and row["prototype_source_bucket"] in ("all", row["difficulty_bucket"])
    if policy == "safety_energy_bucket_alignment":
        return row["variant"] == SAFETY_VARIANT and row["prototype_source_bucket"] in ("all", row["difficulty_bucket"])
    if policy == LCB_POLICY:
        target = str(row["difficulty_bucket"])
        family = action_family(str(row["action_name"]))
        if target == "hard_bottom25":
            return row["variant"] in (SIGNAL_VARIANT, SAFETY_VARIANT) and compatibility_pass(row, strict=False)
        if target == "mid_50":
            return row["variant"] == SAFETY_VARIANT and compatibility_pass(row, strict=False) and family in ("mild", "medium")
        return row["variant"] == SAFETY_VARIANT and compatibility_pass(row, strict=True) and family in ("mild",)
    raise ValueError(policy)


def rows_for_firewall(rows: list[dict[str, Any]], firewall: str) -> list[dict[str, Any]]:
    base_variant = SAFETY_VARIANT if firewall.endswith("energy_norm") else SIGNAL_VARIANT
    return [
        r
        for r in rows
        if r["variant"] == base_variant
        and int(r.get("deployable_candidate", 0)) == 1
        and (r["action_name"] == "noop" or firewall_accept(r, firewall))
    ]


def diagnostic_rows_for_firewall(rows: list[dict[str, Any]], firewall: str) -> list[dict[str, Any]]:
    base_variant = SAFETY_VARIANT if firewall.endswith("energy_norm") else SIGNAL_VARIANT
    return [
        r
        for r in rows
        if r["variant"] == base_variant
        and int(r.get("diagnostic_negative_control", 0)) == 1
        and firewall_accept(r, firewall)
    ]


def select_restricted_oracle(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row_for_policy(row, policy):
            grouped[str(row["sample_name"])].append(row)
    selected = []
    for sample, subset in grouped.items():
        best = max(subset, key=lambda r: float(r["dpsnr"]))
        selected.append({**best, "policy_name": policy, "selection_reason": "safe_set_restricted_oracle"})
    return selected


def select_firewall_oracle(rows: list[dict[str, Any]], firewall: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_for_firewall(rows, firewall):
        grouped[str(row["sample_name"])].append(row)
    selected = []
    for _sample, subset in grouped.items():
        best = max(subset, key=lambda r: float(r["dpsnr"]) - 0.10 * float(r.get("bucket_distance", 0.0)))
        selected.append({**best, "firewall": firewall})
    return selected


def fold_tail_pass(rows: list[dict[str, Any]]) -> int:
    count = 0
    for fold in sorted({int(r["target_fold"]) for r in rows}):
        vals = [float(r["dpsnr"]) for r in rows if int(r["target_fold"]) == fold]
        if vals and percentile(vals, 5) >= -0.15 and safe_rate(vals, SEVERE) <= 0.035:
            count += 1
    return count


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = summarize_policy_rows(rows)
    return {
        "sample_count": s["count"],
        "mean": s["mean"],
        "hard": s["hard"],
        "easy": s["easy"],
        "p05": s["p05"],
        "cvar5": s["cvar5"],
        "unsafe_rate": s["unsafe_rate"],
        "severe_rate": s["severe_rate"],
        "fold_tail_pass": fold_tail_pass(rows),
        "easy_noop_mild_rate": rate_subset(rows, lambda r: r["difficulty_bucket"] == "easy_top25", lambda r: action_family(str(r["action_name"])) in ("noop", "mild")),
        "hard_medium_strong_rate": rate_subset(rows, lambda r: r["difficulty_bucket"] == "hard_bottom25", lambda r: action_family(str(r["action_name"])) in ("medium", "strong")),
    }


def rate_subset(rows: list[dict[str, Any]], where: Callable[[dict[str, Any]], bool], pred: Callable[[dict[str, Any]], bool]) -> float:
    subset = [r for r in rows if where(r)]
    return sum(1 for r in subset if pred(r)) / len(subset) if subset else float("nan")


def train_cell_table(rows: list[dict[str, Any]], policy: str, train_folds: set[int]) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["target_fold"]) not in train_folds or not row_for_policy(row, policy) or row["action_name"] == "noop":
            continue
        key = (
            str(row["difficulty_bucket"]),
            compat_bin(row),
            str(row["stage_set"]),
            str(row["prototype_source_bucket"]),
            str(row["action_name"]),
            float(row["action_strength"]),
        )
        groups[key].append(row)
    table: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, subset in groups.items():
        vals = [float(r["dpsnr"]) for r in subset]
        n = len(vals)
        margin = 0.10 + 0.20 / max(math.sqrt(n), 1.0)
        table[key] = {
            "key": key,
            "count": n,
            "mean": mean(vals),
            "mean_lcb": mean(vals) - margin,
            "p05": percentile(vals, 5),
            "p05_lcb": percentile(vals, 5) - margin,
            "cvar5": cvar(vals, 5),
            "cvar5_lcb": cvar(vals, 5) - margin,
            "unsafe_ucb": min(1.0, safe_rate(vals, STRONG_REG) + 1.0 / (n + 1)),
            "severe_ucb": min(1.0, safe_rate(vals, SEVERE) + 1.0 / (n + 1)),
        }
    return table


def run_lcb_policy(rows: list[dict[str, Any]], policy: str = LCB_POLICY) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = sorted({int(r["target_fold"]) for r in rows})
    by_fold_sample: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row_for_policy(row, policy):
            by_fold_sample[(int(row["target_fold"]), str(row["sample_name"]))].append(row)
    selected = []
    cell_reports = []
    seen_cells: set[tuple[int, tuple[Any, ...]]] = set()
    for fold in folds:
        table = train_cell_table(rows, policy, {f for f in folds if f != fold})
        for key, metrics in table.items():
            marker = (fold, key)
            if marker not in seen_cells:
                seen_cells.add(marker)
                target_bucket, cbin, stage_set, source_bucket, action_name, strength = key
                cell_reports.append(
                    {
                        "target_fold": fold,
                        "policy_variant": policy,
                        "target_bucket": target_bucket,
                        "compat_bin": cbin,
                        "stage_set": stage_set,
                        "source_bucket": source_bucket,
                        "action_name": action_name,
                        "action_strength": strength,
                        "coverage_count": metrics["count"],
                        "mean_lcb": metrics["mean_lcb"],
                        "p05_lcb": metrics["p05_lcb"],
                        "cvar5_lcb": metrics["cvar5_lcb"],
                        "unsafe_ucb": metrics["unsafe_ucb"],
                        "severe_ucb": metrics["severe_ucb"],
                    }
                )
        for (target_fold, sample_name), subset in by_fold_sample.items():
            if target_fold != fold:
                continue
            noop = next((r for r in subset if r["action_name"] == "noop"), subset[0])
            feasible = []
            for row in subset:
                if row["action_name"] == "noop":
                    continue
                key = (
                    str(row["difficulty_bucket"]),
                    compat_bin(row),
                    str(row["stage_set"]),
                    str(row["prototype_source_bucket"]),
                    str(row["action_name"]),
                    float(row["action_strength"]),
                )
                metrics = table.get(key)
                if not metrics:
                    continue
                safe = (
                    metrics["p05_lcb"] >= -0.15
                    and metrics["cvar5_lcb"] >= -0.35
                    and metrics["severe_ucb"] <= 0.10
                    and metrics["unsafe_ucb"] <= 0.35
                    and metrics["count"] >= 2
                )
                if safe:
                    feasible.append((metrics["mean_lcb"], row, metrics))
            if feasible:
                _score, chosen, metrics = max(feasible, key=lambda item: item[0])
                reason = f"mean_lcb={metrics['mean_lcb']:.6f};p05_lcb={metrics['p05_lcb']:.6f};severe_ucb={metrics['severe_ucb']:.6f}"
            else:
                chosen = noop
                reason = "no_lcb_safe_cell_noop_fallback"
            selected.append({**chosen, "policy_name": policy, "selection_reason": reason})
    return selected, cell_reports


def table_summary(selected: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    s = summarize_selected(selected)
    cross = [r for r in diagnostics if r["action_name"] == "cross_bucket_mismatch"]
    hard_to_easy = [r for r in cross if r["prototype_source_bucket"] == "hard_bottom25" and r["difficulty_bucket"] == "easy_top25"]
    out = {
        "table_policy_mean": s["mean"],
        "table_policy_hard": s["hard"],
        "table_policy_easy": s["easy"],
        "table_policy_p05": s["p05"],
        "table_policy_cvar5": s["cvar5"],
        "table_policy_unsafe_rate": s["unsafe_rate"],
        "table_policy_severe_rate": s["severe_rate"],
        "table_policy_fold_tail_pass": s["fold_tail_pass"],
        "cross_bucket_unsafe": accepted_risk_rate([float(r["dpsnr"]) for r in cross], STRONG_REG),
        "hard_to_easy_cross_severe": accepted_risk_rate([float(r["dpsnr"]) for r in hard_to_easy], SEVERE),
    }
    out["table_pass"] = int(
        out["table_policy_mean"] >= 0.30
        and out["table_policy_hard"] >= 0.60
        and out["table_policy_easy"] >= 0.0
        and out["table_policy_p05"] >= -0.15
        and out["table_policy_cvar5"] >= -0.35
        and out["table_policy_severe_rate"] <= 0.035
        and out["table_policy_fold_tail_pass"] >= 4
        and out["hard_to_easy_cross_severe"] <= 0.35
        and out["cross_bucket_unsafe"] <= 0.35
    )
    return out


def write_two_phase_negative_control_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = [r for r in rows if r["variant"] == "raw_v228_baseline" and int(r.get("diagnostic_negative_control", 0)) == 1]
    gated = [r for r in rows if r["variant"] == SAFETY_VARIANT and int(r.get("diagnostic_negative_control", 0)) == 1 and firewall_accept(r, "compatibility_distance_threshold_plus_energy_norm")]
    keys = sorted(
        {
            (
                r["control_family"],
                r["action_name"],
                r["stage_set"],
                r["prototype_source_bucket"],
                r["difficulty_bucket"],
            )
            for r in raw
        }
    )
    out = []
    for control, name, stage, source, target in keys:
        pre = [r for r in raw if (r["control_family"], r["action_name"], r["stage_set"], r["prototype_source_bucket"], r["difficulty_bucket"]) == (control, name, stage, source, target)]
        post = [r for r in gated if (r["control_family"], r["action_name"], r["stage_set"], r["prototype_source_bucket"], r["difficulty_bucket"]) == (control, name, stage, source, target)]
        pre_vals = [float(r["dpsnr"]) for r in pre]
        post_vals = [float(r["dpsnr"]) for r in post]
        raw_sanity = name in ("sign_flip", "overstrong_3.0")
        if raw_sanity:
            interp = "PASS_PRE_SENSITIVITY" if safe_rate(pre_vals, STRONG_REG) >= 0.60 else "FAIL_PRE_SENSITIVITY"
            if len(post) == 0 or safe_rate(post_vals, STRONG_REG) <= 0.35:
                interp += "_POST_ENVELOPE_BLOCK_OR_SAFE"
            else:
                interp += "_POST_ACCEPTED_UNSAFE"
        else:
            interp = "PASS_POST_PLAUSIBLE_CONTROL" if (len(post) == 0 or safe_rate(post_vals, STRONG_REG) <= 0.35) else "FAIL_POST_PLAUSIBLE_CONTROL"
        out.append(
            {
                "variant": SAFETY_VARIANT,
                "control_family": control,
                "control_name": name,
                "stage_set": stage,
                "source_bucket": source,
                "target_bucket": target,
                "pre_envelope_mean_dpsnr": mean(pre_vals),
                "pre_envelope_p05": percentile(pre_vals, 5),
                "pre_envelope_cvar5": cvar(pre_vals, 5),
                "pre_envelope_unsafe_rate": safe_rate(pre_vals, STRONG_REG),
                "pre_envelope_severe_rate": safe_rate(pre_vals, SEVERE),
                "post_envelope_accept_rate": len(post) / len(pre) if pre else float("nan"),
                "post_envelope_mean_dpsnr": mean(post_vals),
                "post_envelope_p05": percentile(post_vals, 5),
                "post_envelope_cvar5": cvar(post_vals, 5),
                "post_envelope_unsafe_rate": accepted_risk_rate(post_vals, STRONG_REG),
                "post_envelope_severe_rate": accepted_risk_rate(post_vals, SEVERE),
                "gate_interpretation": interp,
            }
        )
    write_csv(args.out_dir / "v230_p2a_two_phase_negative_control_report.csv", out)
    return out


def write_safe_set_restricted_oracle_gap(
    args: argparse.Namespace,
    p1_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    restricted: list[dict[str, Any]],
    table_selected: list[dict[str, Any]],
) -> dict[str, Any]:
    unrestricted = preference_rows(rows, SIGNAL_VARIANT, args)
    noop_rows = []
    for sample in {r["sample_name"] for r in rows if int(r.get("deployable_candidate", 0)) == 1}:
        base = next(r for r in rows if r["sample_name"] == sample and r["action_name"] == "noop")
        noop_rows.append({**base, "dpsnr": 0.0})
    names = [
        ("unrestricted selected oracle", unrestricted, "v2.29-style conservative selected policy"),
        ("safe-set restricted oracle", restricted, "best target-GT action inside GT-free compatibility gate"),
        ("GT-free table policy", table_selected, "LCB-risk constrained fold-out table"),
        ("no-op baseline", noop_rows, "fallback"),
    ]
    lines = [
        "# v2.30 Safe-Set Restricted Oracle Gap",
        "",
        "All rows are train-derived OOF diagnostics. The restricted oracle uses target",
        "dPSNR only after a GT-free compatibility gate has defined the safe action set.",
        "",
        "| policy | mean | hard | easy | p05 | CVaR5 | severe | explanation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    summary: dict[str, Any] = {}
    for name, subset, explanation in names:
        s = summarize_selected(subset)
        summary[name] = s
        lines.append(f"| {name} | {s['mean']:.6f} | {s['hard']:.6f} | {s['easy']:.6f} | {s['p05']:.6f} | {s['cvar5']:.6f} | {s['severe_rate']:.6f} | {explanation} |")
    gap = summary["safe-set restricted oracle"]["mean"] - summary["GT-free table policy"]["mean"]
    hard_gap = summary["safe-set restricted oracle"]["hard"] - summary["GT-free table policy"]["hard"]
    lines.extend(
        [
            "",
            f"safe_set_to_table_mean_gap: `{gap:.6f}`",
            f"safe_set_to_table_hard_gap: `{hard_gap:.6f}`",
            f"same_sample_oracle_mean_reference: `{mean([float(r['same_sample_oracle_dpsnr']) for r in p1_rows]):.6f}`",
        ]
    )
    write_text(args.out_dir / "v230_p2a_safe_set_restricted_oracle_gap.md", "\n".join(lines))
    return summary


def fast_roc_auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted((float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s)))
    pos = sum(y for _s, y in pairs)
    neg = len(pairs) - pos
    if pos <= 0 or neg <= 0:
        return float("nan")
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(y for _s, y in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def best_balanced_threshold(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    work = sorted([(float(s), int(y)) for s, y in pairs if math.isfinite(float(s))], reverse=True)
    if not work:
        return (float("nan"), float("nan"))
    pos = sum(y for _s, y in work)
    neg = len(work) - pos
    best_thr = work[0][0]
    best_bal = -1.0
    tp = 0
    fp = 0
    i = 0
    while i < len(work):
        score = work[i][0]
        group_pos = 0
        group_total = 0
        while i < len(work) and work[i][0] == score:
            group_pos += work[i][1]
            group_total += 1
            i += 1
        tp += group_pos
        fp += group_total - group_pos
        tn = neg - fp
        sensitivity = tp / pos if pos else 0.0
        specificity = tn / neg if neg else 0.0
        bal = 0.5 * (sensitivity + specificity)
        if bal > best_bal:
            best_bal = bal
            best_thr = score
    return best_thr, best_bal


def threshold_metrics(scores: list[float], labels: list[int], folds: list[int]) -> tuple[float, float, float, float]:
    preds: list[tuple[int, int]] = []
    train_bals: list[float] = []
    for fold in sorted(set(folds)):
        train = [(s, y) for s, y, f in zip(scores, labels, folds) if f != fold]
        valid = [(s, y) for s, y, f in zip(scores, labels, folds) if f == fold]
        if not train or not valid:
            continue
        best_thr, best_bal = best_balanced_threshold(train)
        train_bals.append(best_bal)
        preds.extend((int(s >= best_thr), y) for s, y in valid)
    if not preds:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    tp = sum(p == 1 and y == 1 for p, y in preds)
    tn = sum(p == 0 and y == 0 for p, y in preds)
    fp = sum(p == 1 and y == 0 for p, y in preds)
    fn = sum(p == 0 and y == 1 for p, y in preds)
    pos = tp + fn
    neg = tn + fp
    bal = 0.5 * ((tp / pos if pos else 0.0) + (tn / neg if neg else 0.0))
    false_safe = fn / pos if pos else float("nan")
    false_block = fp / neg if neg else float("nan")
    return bal, false_safe, false_block, mean(train_bals)


def write_feature_separability(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    work = [
        r
        for r in rows
        if r["variant"] in (SIGNAL_VARIANT, SAFETY_VARIANT)
        and (int(r.get("deployable_candidate", 0)) == 1 or int(r.get("diagnostic_negative_control", 0)) == 1)
        and r["action_name"] != "noop"
    ]
    labels: dict[str, Callable[[dict[str, Any]], int]] = {
        "is_unsafe": lambda r: int(float(r["dpsnr"]) <= STRONG_REG),
        "is_severe": lambda r: int(float(r["dpsnr"]) <= SEVERE),
        "is_useful_gt_0p30": lambda r: int(float(r["dpsnr"]) >= 0.30),
        "is_hard_to_easy_unsafe": lambda r: int(r["prototype_source_bucket"] == "hard_bottom25" and r["difficulty_bucket"] == "easy_top25" and float(r["dpsnr"]) <= STRONG_REG),
        "is_overstrong_unsafe": lambda r: int(action_family(str(r["action_name"])) == "overstrong" and float(r["dpsnr"]) <= STRONG_REG),
        "should_noop": lambda r: int(r["difficulty_bucket"] == "easy_top25" and (float(r["dpsnr"]) <= 0.05 or action_family(str(r["action_name"])) not in ("mild",))),
    }
    out = []
    for feature_set in ("bucket_stage_strength", "compatibility_energy_alignment", "combined"):
        raw_scores = [row_risk_score(r, feature_set) for r in work]
        for label, fn in labels.items():
            scores = [-s if label == "is_useful_gt_0p30" else s for s in raw_scores]
            for bucket in ("all",) + BUCKETS:
                subset = [(s, fn(r), int(r["target_fold"])) for s, r in zip(scores, work) if bucket == "all" or r["difficulty_bucket"] == bucket]
                if not subset:
                    continue
                sc = [s for s, _y, _f in subset]
                ys = [y for _s, y, _f in subset]
                fs = [f for _s, _y, f in subset]
                fold_aucs = [fast_roc_auc([s for s, y, f in subset if f == fold], [y for s, y, f in subset if f == fold]) for fold in sorted(set(fs))]
                bal, false_safe, false_block, _train_bal = threshold_metrics(sc, ys, fs)
                out.append(
                    {
                        "feature_set": feature_set,
                        "label": label,
                        "target_bucket": bucket,
                        "stage_set": "mixed",
                        "sample_count": len(subset),
                        "auroc": fast_roc_auc(sc, ys),
                        "auprc": average_precision(sc, ys),
                        "balanced_accuracy_at_lcb_threshold": bal,
                        "false_safe_rate": false_safe,
                        "false_block_rate": false_block,
                        "top_features": feature_set,
                        "fold_std": std([float(v) for v in fold_aucs if math.isfinite(float(v))]),
                    }
                )
    write_csv(args.out_dir / "v230_p2a_compatibility_feature_separability.csv", out)
    return out


def write_firewall_ablation(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for firewall in FIREWALL_VARIANTS:
        selected = select_firewall_oracle(rows, firewall)
        diagnostics = diagnostic_rows_for_firewall(rows, firewall)
        selected_summary = summarize_selected(selected)
        table_selected, _cell_reports = run_lcb_policy(rows, LCB_POLICY if firewall.startswith("compatibility") else "signal_bucket_strength_grid")
        table = summarize_selected(table_selected)
        cross = [r for r in diagnostics if r["action_name"] == "cross_bucket_mismatch"]
        hard_to_easy = [r for r in cross if r["prototype_source_bucket"] == "hard_bottom25" and r["difficulty_bucket"] == "easy_top25"]
        out.append(
            {
                "variant": firewall,
                "selected_mean": selected_summary["mean"],
                "selected_hard": selected_summary["hard"],
                "selected_easy": selected_summary["easy"],
                "table_policy_mean": table["mean"],
                "table_policy_hard": table["hard"],
                "table_policy_easy": table["easy"],
                "cross_bucket_unsafe": accepted_risk_rate([float(r["dpsnr"]) for r in cross], STRONG_REG),
                "hard_to_easy_cross_severe": accepted_risk_rate([float(r["dpsnr"]) for r in hard_to_easy], SEVERE),
                "easy_noop_mild_rate": selected_summary["easy_noop_mild_rate"],
                "hard_medium_strong_rate": selected_summary["hard_medium_strong_rate"],
                "p05": selected_summary["p05"],
                "cvar5": selected_summary["cvar5"],
                "severe_rate": selected_summary["severe_rate"],
                "fold_tail_pass": selected_summary["fold_tail_pass"],
            }
        )
    write_csv(args.out_dir / "v230_p2a_cross_bucket_firewall_ablation.csv", out)
    return out


def write_strength_dose_response(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    work = [
        r
        for r in rows
        if r["variant"] in (SIGNAL_VARIANT, SAFETY_VARIANT)
        and int(r.get("deployable_candidate", 0)) == 1
        and r["action_name"] != "noop"
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in work:
        key = (row["difficulty_bucket"], row["prototype_source_bucket"], row["stage_set"], compat_bin(row), float(row["action_strength"]))
        groups[key].append(row)
    base_groups: dict[tuple[Any, ...], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in work:
        base = (row["difficulty_bucket"], row["prototype_source_bucket"], row["stage_set"], compat_bin(row))
        base_groups[base][float(row["action_strength"])].append(float(row["dpsnr"]))
    safe_max: dict[tuple[Any, ...], float] = {}
    monotonicity: dict[tuple[Any, ...], int] = {}
    for base, by_strength in base_groups.items():
        prev = None
        violation = 0
        max_safe = 0.0
        for strength in sorted(by_strength):
            vals = by_strength[strength]
            m = mean(vals)
            if prev is not None and m + 1e-9 < prev:
                violation = 1
            prev = m
            if percentile(vals, 5) >= -0.15 and cvar(vals, 5) >= -0.35 and safe_rate(vals, SEVERE) <= 0.035:
                max_safe = max(max_safe, strength)
        safe_max[base] = max_safe
        monotonicity[base] = violation
    out = []
    for key, subset in sorted(groups.items()):
        target, source, stage, cbin, strength = key
        vals = [float(r["dpsnr"]) for r in subset]
        base = (target, source, stage, cbin)
        out.append(
            {
                "target_bucket": target,
                "source_bucket": source,
                "stage_set": stage,
                "compat_bin": cbin,
                "strength": strength,
                "count": len(vals),
                "mean_dpsnr": mean(vals),
                "p05_dpsnr": percentile(vals, 5),
                "cvar5_dpsnr": cvar(vals, 5),
                "unsafe_rate": safe_rate(vals, STRONG_REG),
                "severe_rate": safe_rate(vals, SEVERE),
                "monotonicity_violation": monotonicity[base],
                "safe_max_strength_lcb": safe_max[base],
            }
        )
    write_csv(args.out_dir / "v230_p2a_strength_dose_response_by_compat_bin.csv", out)
    return out


def write_lcb_table_report(args: argparse.Namespace, selected: list[dict[str, Any]], cell_reports: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted({int(r["target_fold"]) for r in selected}):
        for bucket in BUCKETS:
            subset = [r for r in selected if int(r["target_fold"]) == fold and r["difficulty_bucket"] == bucket]
            vals = [float(r["dpsnr"]) for r in subset]
            actions = sorted({str(r["action_name"]) for r in subset})
            rows.append(
                {
                    "target_fold": fold,
                    "policy_variant": LCB_POLICY,
                    "target_bucket": bucket,
                    "compat_bin": "mixed",
                    "stage_set": "mixed",
                    "chosen_source_bucket": "mixed",
                    "chosen_action": "+".join(actions) if actions else "none",
                    "chosen_strength": "mixed",
                    "selection_reason": "fold_out_lcb_risk_constrained_table",
                    "sample_count": len(vals),
                    "mean_dpsnr": mean(vals),
                    "p05_dpsnr": percentile(vals, 5),
                    "cvar5_dpsnr": cvar(vals, 5),
                    "unsafe_rate": safe_rate(vals, STRONG_REG),
                    "severe_rate": safe_rate(vals, SEVERE),
                }
            )
    summary = table_summary(selected, diagnostics)
    rows.append({"target_fold": "ALL", "policy_variant": LCB_POLICY, "target_bucket": "ALL", **summary})
    write_csv(args.out_dir / "v230_p2a_lcb_constrained_oof_table_policy_report.csv", rows + cell_reports)
    return rows


def write_action_confusion(args: argparse.Namespace, restricted: list[dict[str, Any]], table_selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_by_sample = {str(r["sample_name"]): r for r in table_selected}
    groups: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in restricted:
        table = table_by_sample.get(str(row["sample_name"]))
        if table is None:
            continue
        key = (
            row["difficulty_bucket"],
            row["action_family"],
            table["action_family"],
            row["stage_set"],
            table["stage_set"],
            row["prototype_source_bucket"],
            table["prototype_source_bucket"],
        )
        groups[key] += 1
    out = []
    for key, count in sorted(groups.items()):
        target, oracle_action, table_action, oracle_stage, table_stage, oracle_source, table_source = key
        out.append(
            {
                "target_bucket": target,
                "oracle_action_family": oracle_action,
                "table_action_family": table_action,
                "oracle_stage_set": oracle_stage,
                "table_stage_set": table_stage,
                "oracle_source_bucket": oracle_source,
                "table_source_bucket": table_source,
                "count": count,
            }
        )
    write_csv(args.out_dir / "v230_p2a_policy_action_confusion_matrix.csv", out)
    return out


def write_fold_tail(args: argparse.Namespace, selected_sets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for name, rows in selected_sets.items():
        for fold in sorted({int(r["target_fold"]) for r in rows}):
            subset = [r for r in rows if int(r["target_fold"]) == fold]
            vals = [float(r["dpsnr"]) for r in subset]
            out.append(
                {
                    "policy": name,
                    "target_fold": fold,
                    "sample_count": len(vals),
                    "mean_dpsnr": mean(vals),
                    "p05_dpsnr": percentile(vals, 5),
                    "cvar5_dpsnr": cvar(vals, 5),
                    "unsafe_rate": safe_rate(vals, STRONG_REG),
                    "severe_rate": safe_rate(vals, SEVERE),
                    "tail_gate_pass": int(bool(vals) and percentile(vals, 5) >= -0.15 and cvar(vals, 5) >= -0.35 and safe_rate(vals, SEVERE) <= 0.035),
                }
            )
    write_csv(args.out_dir / "v230_p2a_fold_tail_report.csv", out)
    return out


def write_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    p2 = closeout.get("p2a", {})
    diag = p2.get("primary_diagnosis", {})
    lines = [
        "# Haze4K v2.30 NoPost ILFRB-ACS Compatibility-Gated OOF Table Policy Evidence",
        "",
        "Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy.md`",
        "",
        f"Status: `{closeout.get('decision', 'UNKNOWN')}`",
        "",
        "Runtime server: `convir-4090`",
        "Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`",
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
        f"- Safe-set restricted oracle mean/hard/easy: `{diag.get('restricted_mean')} / {diag.get('restricted_hard')} / {diag.get('restricted_easy')}`",
        f"- GT-free table policy mean/hard/easy: `{diag.get('table_mean')} / {diag.get('table_hard')} / {diag.get('table_easy')}`",
        f"- hard-to-easy cross severe: `{diag.get('hard_to_easy_cross_severe')}`",
        f"- cross-bucket unsafe: `{diag.get('cross_bucket_unsafe')}`",
        f"- overstrong 1.5 unsafe: `{diag.get('overstrong_1p5_unsafe')}`",
        f"- selected-to-table mean gap: `{diag.get('selected_to_table_gap')}`",
        "",
        "## Primary Files",
        "",
        "- `v230_p0_arch_contract_delta.md`",
        "- `v230_p2a_two_phase_negative_control_report.csv`",
        "- `v230_p2a_safe_set_restricted_oracle_gap.md`",
        "- `v230_p2a_compatibility_feature_separability.csv`",
        "- `v230_p2a_cross_bucket_firewall_ablation.csv`",
        "- `v230_p2a_strength_dose_response_by_compat_bin.csv`",
        "- `v230_p2a_lcb_constrained_oof_table_policy_report.csv`",
        "- `v230_p2a_policy_action_confusion_matrix.csv`",
        "- `v230_p2a_fold_tail_report.csv`",
        "- `v230_p2a_closeout.json`",
        "- `run_v230_p2a.sh`",
        "- `monitor_v230.sh`",
        "- `status.txt`",
        "",
        "This directory is compact text evidence only. It excludes checkpoints, weights, images, arrays, archives, and raw feature dumps.",
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
    ap.add_argument("--parent-commit", default="936e3e0")
    ap.add_argument("--conservative-penalty", type=float, default=0.20)
    ap.add_argument("--prototype-complexity-penalty", type=float, default=0.03)
    ap.add_argument("--bucket-distance-penalty", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=230)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    append_status(args, f"v230_start route_id={ROUTE_ID}")
    append_status(args, "training_launched=false")
    append_status(args, "p2b_selector_probe_launched=false")
    append_status(args, "locked_test_touched=false")
    closeout: dict[str, Any] = {
        "route_id": ROUTE_ID,
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "--short", "HEAD"]),
        "parent_branch": "codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration",
        "parent_commit": args.parent_commit,
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
    }
    p0 = phase_p0(args)
    closeout["p0"] = p0
    if not p0["pass"]:
        closeout["decision"] = p0["decision"]
        write_json(args.out_dir / "v230_p2a_closeout.json", json_clean(closeout))
        write_readme(args, closeout)
        append_status(args, f"v230_done decision={closeout['decision']}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rows, p1_rows, delta_bank = build_oracle_delta_bank(args, device)
    attach_sample_buckets(sample_rows)
    rows = run_variant_replay(args, sample_rows, delta_bank, device)

    two_phase = write_two_phase_negative_control_report(rows, args)
    feature_sep = write_feature_separability(rows, args)
    firewall = write_firewall_ablation(rows, args)
    dose = write_strength_dose_response(rows, args)

    restricted = select_restricted_oracle(rows, LCB_POLICY)
    table_selected, cell_reports = run_lcb_policy(rows, LCB_POLICY)
    diagnostics = [r for r in rows if r["variant"] == SAFETY_VARIANT and int(r.get("diagnostic_negative_control", 0)) == 1 and compatibility_pass(r, strict=True)]
    table_rows = write_lcb_table_report(args, table_selected, cell_reports, diagnostics)
    gap_summary = write_safe_set_restricted_oracle_gap(args, p1_rows, rows, restricted, table_selected)
    confusion = write_action_confusion(args, restricted, table_selected)
    fold_tail = write_fold_tail(args, {"safe_set_restricted_oracle": restricted, "lcb_table_policy": table_selected})

    table_gate = table_summary(table_selected, diagnostics)
    overstrong_1p5 = [
        r
        for r in diagnostics
        if r["action_name"] == "overstrong_1.5"
    ]
    primary = {
        "restricted_mean": gap_summary["safe-set restricted oracle"]["mean"],
        "restricted_hard": gap_summary["safe-set restricted oracle"]["hard"],
        "restricted_easy": gap_summary["safe-set restricted oracle"]["easy"],
        "table_mean": table_gate["table_policy_mean"],
        "table_hard": table_gate["table_policy_hard"],
        "table_easy": table_gate["table_policy_easy"],
        "table_p05": table_gate["table_policy_p05"],
        "table_cvar5": table_gate["table_policy_cvar5"],
        "table_severe_rate": table_gate["table_policy_severe_rate"],
        "table_fold_tail_pass": table_gate["table_policy_fold_tail_pass"],
        "hard_to_easy_cross_severe": table_gate["hard_to_easy_cross_severe"],
        "cross_bucket_unsafe": table_gate["cross_bucket_unsafe"],
        "overstrong_1p5_unsafe": accepted_risk_rate([float(r["dpsnr"]) for r in overstrong_1p5], STRONG_REG),
        "selected_to_table_gap": gap_summary["safe-set restricted oracle"]["mean"] - table_gate["table_policy_mean"],
    }
    pass_gate = bool(table_gate["table_pass"])
    decision = "P2A_PASS_COMPATIBILITY_GATED_TABLE_POLICY" if pass_gate else "P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE"
    closeout["p2a"] = {
        "decision": decision,
        "pass": pass_gate,
        "primary_diagnosis": primary,
        "table_gate": table_gate,
        "artifact_counts": {
            "two_phase_negative_control_rows": len(two_phase),
            "feature_separability_rows": len(feature_sep),
            "firewall_ablation_rows": len(firewall),
            "dose_response_rows": len(dose),
            "lcb_table_report_rows": len(table_rows),
            "cell_report_rows": len(cell_reports),
            "confusion_rows": len(confusion),
            "fold_tail_rows": len(fold_tail),
        },
        "training_launched": False,
        "p2b_selector_probe_launched": False,
        "locked_test_touched": False,
    }
    closeout["decision"] = decision
    write_json(args.out_dir / "v230_p2a_closeout.json", json_clean(closeout))
    write_readme(args, closeout)
    append_status(args, f"v230_done decision={decision}")
    print("V230_COMPATIBILITY_GATED_OOF_TABLE_POLICY_OK " + decision, flush=True)


if __name__ == "__main__":
    main()
