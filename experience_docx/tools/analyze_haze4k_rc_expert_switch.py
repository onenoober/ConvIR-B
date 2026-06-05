#!/usr/bin/env python3
"""Offline v1.6 risk-calibrated expert-switch analysis.

This script reads existing text evidence under `experience_docx/` and writes
the intermediate artifacts needed before any new training:

- route utility leaderboards under split Mechanism / Utility / Promotion gates;
- A0 + official UDPNet oracle-switch upper bound;
- UDP accept/risk label predictability from deployable-ish proxy features;
- OOF-calibrated threshold switch summaries for the A0/UDP expert bank.

It is intentionally standard-library only so it can run as a cheap cloud audit
without depending on training packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROUTE_ID = "haze4k_rc_expert_switch_v16_20260605"


@dataclass(frozen=True)
class EvidenceSource:
    key: str
    display_name: str
    family: str
    priority_role: str
    path: str
    split_name: str
    format: str = "generic_per_image"
    known_status: str = ""
    notes: str = ""


DEFAULT_SOURCES = [
    EvidenceSource(
        "udpnet_phase0",
        "official UDPNet ConvIR Phase0",
        "FullUDP",
        "hard_expert",
        "experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/phase0_official_eval/udpnet_convir_bucket_compare.csv",
        "combined_val_regular_val_hard",
        "udp_bucket_compare",
        "PHASE0_REPRODUCTION_GATE_FAIL_GLOBAL_REPLACEMENT",
    ),
    EvidenceSource(
        "fam2_only",
        "FAM2-only stop20",
        "FAM2",
        "hard_expert_candidate",
        "experience_docx/experiment_logs/haze4k_fam2_modres_stop20_20260531/scout_eval_per_image_seed3407_stop20.csv",
        "stop20_test_like",
        notes="Legacy single-split stop20 evidence; compare cautiously with internal split evidence.",
    ),
    EvidenceSource(
        "fam2_confidence",
        "FAM2 confidence-gated gamma stop20",
        "FAM2",
        "hard_expert_candidate",
        "experience_docx/experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/scout_eval_per_image_seed3407_stop20_best.csv",
        "stop20_test_like",
        notes="Strong hard movement, but preservation/selectivity failed prior gate.",
    ),
    EvidenceSource(
        "hardfreq",
        "Hard-aware frequency loss",
        "frequency_prior",
        "mechanism_evidence",
        "experience_docx/experiment_logs/haze4k_hardfreq_loss_stop20_20260601/scout_eval_per_image_seed3407_stop20_best.csv",
        "stop20_test_like",
    ),
    EvidenceSource(
        "haze_prior_scm",
        "Haze-prior SCM + hard auxiliary",
        "frequency_prior",
        "mechanism_evidence",
        "experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/scout_eval_per_image_seed3407_stop20_best.csv",
        "stop20_test_like",
    ),
    EvidenceSource(
        "pfd_b1",
        "PFD/RHFD B1",
        "PFD_RHFD",
        "mechanism_evidence",
        "experience_docx/experiment_logs/haze4k_pfd_mainline_20260602/scout_eval_per_image_seed3407_B1_vs_A1_best.csv",
        "stop20_test_like",
        notes="Baseline column is A1, not A0; use only as within-route diagnostic.",
    ),
    EvidenceSource(
        "b1r_decoder_rhfd",
        "B1r decoder RHFD preservation",
        "PFD_RHFD",
        "low_risk_reference",
        "experience_docx/experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/scout_eval_per_image_seed3407_B1r_stop20_vs_A0_best.csv",
        "stop20_test_like",
    ),
    EvidenceSource(
        "apdr_v0_4e_rule_a_oof",
        "APDR-v0.4E OOF Rule A",
        "APDR",
        "safe_subset_expert",
        "experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/v04e_oof_locked_threshold_summary.json",
        "oof_full_train",
        "apdr_oof_rule",
        "FIXED_CODE_RERUN_REQUIRED_BEFORE_NUMERIC_SEAL",
        notes="Uses locked Rule A from v0.4E OOF summary.",
    ),
    EvidenceSource(
        "apdr_v0_4e_best_no_severe_oof",
        "APDR-v0.4E best no-severe OOF policy",
        "APDR",
        "safe_subset_expert",
        "experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/v04e_oof_policy_search_summary_sigma3.json",
        "oof_full_train",
        "apdr_policy_best",
        "FIXED_CODE_RERUN_REQUIRED_BEFORE_NUMERIC_SEAL",
        notes="Post-hoc OOF policy search; diagnostic until fixed-code rerun.",
    ),
    EvidenceSource(
        "dpga_lite_v1_0_best",
        "DPGA-Lite v1.0 Best",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_dpga_lite_20260604/eval_a0_compare/scout_eval_per_image_seed3407_dpga_lite_best_vs_A0.csv",
        "full_test_diagnostic",
    ),
    EvidenceSource(
        "dpga_v1_2_best",
        "DPGA tail-control v1.2 Best",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_val_inner_eval/scout_eval_per_image_v1_2_val_inner_best_vs_a0.csv",
        "val_inner",
    ),
    EvidenceSource(
        "dpga_v1_3b_best_regular",
        "DPGA-v1.3B HSDF Best val_regular",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/v13b_eval_regular_hard/scout_eval_per_image_v13b_best_val_regular_vs_a0.csv",
        "val_regular",
    ),
    EvidenceSource(
        "dpga_v1_3b_best_hard",
        "DPGA-v1.3B HSDF Best val_hard",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/v13b_eval_regular_hard/scout_eval_per_image_v13b_best_val_hard_vs_a0.csv",
        "val_hard",
    ),
    EvidenceSource(
        "udp_lite_v1_4b_best_regular",
        "UDP-Lite v1.4B BiDPFM1 Best val_regular",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/v14b_eval_regular_hard/scout_eval_per_image_v14b_best_val_regular_vs_a0.csv",
        "val_regular",
    ),
    EvidenceSource(
        "udp_lite_v1_4b_best_hard",
        "UDP-Lite v1.4B BiDPFM1 Best val_hard",
        "DPGA",
        "small_gain_reference",
        "experience_docx/experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/v14b_eval_regular_hard/scout_eval_per_image_v14b_best_val_hard_vs_a0.csv",
        "val_hard",
    ),
]


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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


def safe_ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    mx = statistics.mean(x_vals)
    my = statistics.mean(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    return pearson(rankdata([x for x, _ in pairs]), rankdata([y for _, y in pairs]))


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = sum(y == 1 for _, y in pairs)
    neg = sum(y == 0 for _, y in pairs)
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata([s for s, _ in pairs])
    pos_rank_sum = sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1)
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(
        [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)],
        key=lambda item: item[0],
        reverse=True,
    )
    pos = sum(y == 1 for _, y in pairs)
    if pos == 0:
        return None
    hits = 0
    total = 0.0
    for idx, (_score, label) in enumerate(pairs, 1):
        if label == 1:
            hits += 1
            total += hits / idx
    return total / pos


def brier_score(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    if not pairs:
        return None
    lo = min(s for s, _ in pairs)
    hi = max(s for s, _ in pairs)
    if hi - lo <= 1e-12:
        probs = [0.5 for _ in pairs]
    else:
        probs = [(s - lo) / (hi - lo) for s, _ in pairs]
    return statistics.mean((p - y) ** 2 for p, (_s, y) in zip(probs, pairs))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_baseline_keys(row: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    if "a0_psnr" in row:
        base_psnr = "a0_psnr"
        base_ssim = "a0_ssim" if "a0_ssim" in row else None
    elif "original_psnr" in row:
        base_psnr = "original_psnr"
        base_ssim = "original_ssim" if "original_ssim" in row else None
    elif "anchor_psnr" in row:
        base_psnr = "anchor_psnr"
        base_ssim = None
    else:
        raise KeyError(f"Could not find baseline PSNR key in columns: {sorted(row)}")
    cand_psnr = None
    cand_ssim = None
    for key in row:
        if key.endswith("_psnr") and key not in {base_psnr, "oracle_psnr"}:
            cand_psnr = key
            break
    for key in row:
        if key.endswith("_ssim") and key not in {base_ssim, "oracle_ssim"}:
            cand_ssim = key
            break
    return base_psnr, cand_psnr or "", base_ssim, cand_ssim


def normalize_generic_rows(raw_rows: list[dict[str, str]], source: EvidenceSource) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not raw:
            continue
        base_key, cand_key, base_ssim_key, cand_ssim_key = infer_baseline_keys(raw)
        base_psnr = to_float(raw.get(base_key))
        delta = to_float(raw.get("delta_psnr"))
        cand_psnr = to_float(raw.get(cand_key)) if cand_key else None
        if delta is None and base_psnr is not None and cand_psnr is not None:
            delta = cand_psnr - base_psnr
        if delta is None or base_psnr is None:
            continue
        base_ssim = to_float(raw.get(base_ssim_key)) if base_ssim_key else None
        cand_ssim = to_float(raw.get(cand_ssim_key)) if cand_ssim_key else None
        delta_ssim = to_float(raw.get("delta_ssim"))
        if delta_ssim is None and base_ssim is not None and cand_ssim is not None:
            delta_ssim = cand_ssim - base_ssim
        out.append(
            {
                "source_key": source.key,
                "source_name": source.display_name,
                "family": source.family,
                "split": raw.get("split") or source.split_name,
                "name": raw.get("name") or raw.get("image") or raw.get("image_name") or "",
                "baseline_psnr": base_psnr,
                "candidate_psnr": cand_psnr if cand_psnr is not None else base_psnr + delta,
                "delta_psnr": delta,
                "baseline_ssim": base_ssim,
                "candidate_ssim": cand_ssim,
                "delta_ssim": delta_ssim,
                "bucket": raw.get("bucket") or "",
            }
        )
    return out


def normalize_udp_rows(raw_rows: list[dict[str, str]], source: EvidenceSource) -> list[dict[str, Any]]:
    rows = normalize_generic_rows(raw_rows, source)
    for row in rows:
        row["candidate_name"] = "udpnet"
    return rows


def apdr_rule_summary(path: Path, source: EvidenceSource, rule_name: str) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data.get("locked_rules") or []
    if not rules:
        return None
    rule = None
    if rule_name:
        for candidate in rules:
            if rule_name in candidate.get("name", ""):
                rule = candidate
                break
    rule = rule or rules[0]
    return {
        "source_key": source.key,
        "source_name": source.display_name,
        "family": source.family,
        "priority_role": source.priority_role,
        "split_name": source.split_name,
        "source_path": source.path,
        "count": int(rule.get("count", data.get("data_count", 0)) or 0),
        "coverage": to_float(rule.get("coverage")),
        "mean_delta": to_float(rule.get("mean_gain")),
        "median_delta": None,
        "p5_delta": None,
        "p95_delta": None,
        "hard_bottom25_delta": to_float(rule.get("hard_bottom25_gain")),
        "easy_top25_delta": to_float(rule.get("easy_top25_gain")),
        "best10pct_delta": None,
        "worst10pct_delta": None,
        "mean_ssim_delta": None,
        "positive_ratio": None,
        "strong_regression_count": int(rule.get("strong_regressions", 0) or 0),
        "strong_regression_ratio": to_float(rule.get("strong_rate")),
        "worst_regression_count": int(rule.get("severe_regressions", 0) or 0),
        "worst_regression_ratio": (
            int(rule.get("severe_regressions", 0) or 0) / max(1, int(rule.get("count", 0) or 0))
        ),
        "mechanism_gate_pass": False,
        "utility_gate_pass": False,
        "promotion_gate_pass": False,
        "utility_score": None,
        "classification": "",
        "known_status": source.known_status,
        "notes": source.notes or "Summary row read from APDR OOF JSON.",
        "raw_rule_name": rule.get("name", ""),
    }


def apdr_best_policy_summary(path: Path, source: EvidenceSource) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rule = data.get("best_policy")
    if not rule:
        return None
    return {
        "source_key": source.key,
        "source_name": source.display_name,
        "family": source.family,
        "priority_role": source.priority_role,
        "split_name": source.split_name,
        "source_path": source.path,
        "count": int(rule.get("count", 0) or 0),
        "coverage": to_float(rule.get("coverage")),
        "mean_delta": to_float(rule.get("mean_gain")),
        "median_delta": None,
        "p5_delta": None,
        "p95_delta": None,
        "hard_bottom25_delta": to_float(rule.get("hard_bottom25_gain")),
        "easy_top25_delta": to_float(rule.get("easy_top25_gain")),
        "best10pct_delta": None,
        "worst10pct_delta": None,
        "mean_ssim_delta": None,
        "positive_ratio": None,
        "strong_regression_count": int(rule.get("strong_regressions", 0) or 0),
        "strong_regression_ratio": to_float(rule.get("strong_rate")),
        "worst_regression_count": int(rule.get("severe_regressions", 0) or 0),
        "worst_regression_ratio": (
            int(rule.get("severe_regressions", 0) or 0) / max(1, int(rule.get("count", 0) or 0))
        ),
        "mechanism_gate_pass": False,
        "utility_gate_pass": False,
        "promotion_gate_pass": False,
        "utility_score": None,
        "classification": "",
        "known_status": source.known_status,
        "notes": source.notes or "Summary row read from APDR policy-search JSON.",
        "raw_rule_name": "best_policy",
    }


def bucket_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    hard = [row for row in rows if str(row.get("bucket", "")).startswith("hard_bottom25")]
    easy = [row for row in rows if str(row.get("bucket", "")).startswith("easy_top25")]
    if hard and easy:
        return hard, easy
    ordered = sorted(rows, key=lambda row: row["baseline_psnr"])
    n = max(1, len(ordered) // 4)
    return ordered[:n], ordered[-n:]


def summarize_rows(rows: list[dict[str, Any]], source: EvidenceSource) -> dict[str, Any]:
    deltas = [float(row["delta_psnr"]) for row in rows]
    ssim_deltas = [float(row["delta_ssim"]) for row in rows if row.get("delta_ssim") is not None]
    hard, easy = bucket_rows(rows)
    strong_cut = percentile([float(row["baseline_psnr"]) for row in rows], 75)
    strong = [row for row in rows if strong_cut is not None and float(row["baseline_psnr"]) >= strong_cut]
    strong_reg = [row for row in strong if float(row["delta_psnr"]) <= -0.05]
    worst = [row for row in rows if float(row["delta_psnr"]) <= -0.20]
    tail_n = max(1, len(rows) // 10)
    ordered_deltas = sorted(deltas)
    return {
        "source_key": source.key,
        "source_name": source.display_name,
        "family": source.family,
        "priority_role": source.priority_role,
        "split_name": source.split_name,
        "source_path": source.path,
        "count": len(rows),
        "coverage": 1.0,
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "p5_delta": percentile(deltas, 5),
        "p95_delta": percentile(deltas, 95),
        "hard_bottom25_delta": mean([float(row["delta_psnr"]) for row in hard]),
        "easy_top25_delta": mean([float(row["delta_psnr"]) for row in easy]),
        "best10pct_delta": mean(ordered_deltas[-tail_n:]),
        "worst10pct_delta": mean(ordered_deltas[:tail_n]),
        "mean_ssim_delta": mean(ssim_deltas),
        "positive_ratio": safe_ratio(sum(delta > 0 for delta in deltas), len(deltas)),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count": len(strong_reg),
        "strong_regression_ratio": safe_ratio(len(strong_reg), len(strong)),
        "worst_regression_count": len(worst),
        "worst_regression_ratio": safe_ratio(len(worst), len(rows)),
        "mechanism_gate_pass": False,
        "utility_gate_pass": False,
        "promotion_gate_pass": False,
        "utility_score": None,
        "classification": "",
        "known_status": source.known_status,
        "notes": source.notes,
    }


def utility_score(summary: dict[str, Any]) -> float | None:
    required = [
        summary.get("mean_delta"),
        summary.get("hard_bottom25_delta"),
        summary.get("median_delta"),
        summary.get("easy_top25_delta"),
        summary.get("strong_regression_ratio"),
        summary.get("worst_regression_ratio"),
    ]
    if any(value is None for value in required):
        return None
    ssim_delta = summary.get("mean_ssim_delta")
    ssim_penalty = max(0.0, -(ssim_delta or 0.0) / 0.001)
    return (
        1.0 * summary["mean_delta"]
        + 0.6 * summary["hard_bottom25_delta"]
        + 0.2 * summary["median_delta"]
        - 0.5 * max(0.0, -summary["easy_top25_delta"])
        - 0.5 * summary["strong_regression_ratio"]
        - 0.6 * summary["worst_regression_ratio"]
        - 0.2 * ssim_penalty
    )


def apply_gates(summary: dict[str, Any]) -> dict[str, Any]:
    hard = summary.get("hard_bottom25_delta")
    mean_delta = summary.get("mean_delta")
    best10 = summary.get("best10pct_delta")
    easy = summary.get("easy_top25_delta")
    ssim_delta = summary.get("mean_ssim_delta")
    worst_ratio = summary.get("worst_regression_ratio")
    strong_ratio = summary.get("strong_regression_ratio")
    score = utility_score(summary)
    summary["utility_score"] = score
    summary["mechanism_gate_pass"] = bool(
        (hard is not None and hard >= 0.30)
        or (mean_delta is not None and mean_delta >= 0.20)
        or (best10 is not None and best10 >= 1.0)
    )
    summary["utility_gate_pass"] = bool(
        score is not None
        and score > 0
        and mean_delta is not None
        and mean_delta >= 0.10
        and hard is not None
        and hard >= 0.25
        and easy is not None
        and easy >= -0.10
        and (ssim_delta is None or ssim_delta >= -0.0005)
        and worst_ratio is not None
        and worst_ratio <= 0.10
        and strong_ratio is not None
        and strong_ratio <= 0.30
    )
    summary["promotion_gate_pass"] = bool(
        mean_delta is not None
        and mean_delta >= 0.15
        and hard is not None
        and hard >= 0.30
        and easy is not None
        and easy >= -0.03
        and (ssim_delta is None or ssim_delta >= 0)
        and worst_ratio is not None
        and worst_ratio <= 0.05
        and strong_ratio is not None
        and strong_ratio <= 0.16
    )
    if summary["promotion_gate_pass"]:
        summary["classification"] = "promotion_candidate"
    elif summary["utility_gate_pass"]:
        summary["classification"] = "utility_candidate"
    elif summary["mechanism_gate_pass"]:
        summary["classification"] = "mechanism_positive_expert_candidate"
    elif summary.get("priority_role") == "safe_subset_expert" and (summary.get("mean_delta") or 0) > 0:
        summary["classification"] = "safe_subset_diagnostic"
    else:
        summary["classification"] = "diagnostic_or_stopped"
    return summary


def load_route_summaries(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for source in DEFAULT_SOURCES:
        path = repo_root / source.path
        if not path.exists():
            missing.append(
                {
                    "source_key": source.key,
                    "source_name": source.display_name,
                    "source_path": source.path,
                    "status": "missing_source",
                    "notes": source.notes,
                }
            )
            continue
        if source.format == "apdr_oof_rule":
            row = apdr_rule_summary(path, source, "RuleA")
            if row:
                summaries.append(apply_gates(row))
            continue
        if source.format == "apdr_policy_best":
            row = apdr_best_policy_summary(path, source)
            if row:
                summaries.append(apply_gates(row))
            continue
        raw = read_csv_rows(path)
        if source.format == "udp_bucket_compare":
            rows = normalize_udp_rows(raw, source)
        else:
            rows = normalize_generic_rows(raw, source)
        if not rows:
            missing.append(
                {
                    "source_key": source.key,
                    "source_name": source.display_name,
                    "source_path": source.path,
                    "status": "empty_source",
                    "notes": source.notes,
                }
            )
            continue
        if source.key == "udpnet_phase0":
            for split in sorted({str(row["split"]) for row in rows}):
                split_source = EvidenceSource(
                    f"{source.key}_{split}",
                    f"{source.display_name} {split}",
                    source.family,
                    source.priority_role,
                    source.path,
                    split,
                    source.format,
                    source.known_status,
                    source.notes,
                )
                summaries.append(apply_gates(summarize_rows([r for r in rows if r["split"] == split], split_source)))
        summaries.append(apply_gates(summarize_rows(rows, source)))
    return summaries, missing


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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


def sort_summaries(summaries: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda row: (
            not bool(row.get(f"{key}_gate_pass")),
            -(row.get("utility_score") if row.get("utility_score") is not None else -999.0),
            -(row.get("hard_bottom25_delta") if row.get("hard_bottom25_delta") is not None else -999.0),
            -(row.get("mean_delta") if row.get("mean_delta") is not None else -999.0),
        ),
    )


def leaderboard_outputs(output_dir: Path, summaries: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "source_key",
        "source_name",
        "family",
        "priority_role",
        "split_name",
        "count",
        "coverage",
        "mean_delta",
        "median_delta",
        "hard_bottom25_delta",
        "easy_top25_delta",
        "best10pct_delta",
        "worst10pct_delta",
        "mean_ssim_delta",
        "positive_ratio",
        "strong_regression_count",
        "strong_regression_ratio",
        "worst_regression_count",
        "worst_regression_ratio",
        "utility_score",
        "mechanism_gate_pass",
        "utility_gate_pass",
        "promotion_gate_pass",
        "classification",
        "known_status",
        "source_path",
        "notes",
    ]
    write_csv(output_dir / "route_utility_leaderboard.csv", sort_summaries(summaries, "utility"), fields)
    write_csv(output_dir / "global_model_leaderboard.csv", sort_summaries(summaries, "promotion"), fields)
    write_csv(output_dir / "hard_expert_leaderboard.csv", sort_summaries(summaries, "mechanism"), fields)
    write_csv(output_dir / "utility_tradeoff_leaderboard.csv", sort_summaries(summaries, "utility"), fields)
    if missing:
        write_csv(output_dir / "route_utility_missing_sources.csv", missing)
    payload = {
        "route_id": ROUTE_ID,
        "stage": "retrospective route utility leaderboard",
        "source_count": len(DEFAULT_SOURCES),
        "summary_count": len(summaries),
        "missing_count": len(missing),
        "gate_definitions": {
            "mechanism_gate": "hard_bottom25_delta>=0.30 or mean_delta>=0.20 or best10pct_delta>=1.0",
            "utility_gate": "risk-adjusted score >0 plus mean/hard/easy/SSIM/tail/strong budgets",
            "promotion_gate": "strict combined replacement gate; old promotion-style safety constraints",
        },
        "top_hard_experts": sort_summaries(summaries, "mechanism")[:8],
        "top_utility": sort_summaries(summaries, "utility")[:8],
        "top_promotion": sort_summaries(summaries, "promotion")[:8],
        "missing_sources": missing,
    }
    write_json(output_dir / "route_utility_leaderboard_summary.json", payload)
    return payload


def load_udp_rows(repo_root: Path) -> list[dict[str, Any]]:
    source = DEFAULT_SOURCES[0]
    path = repo_root / source.path
    if not path.exists():
        return []
    return normalize_udp_rows(read_csv_rows(path), source)


def combine_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    source = EvidenceSource(label, label, "expert_switch", "combined_policy", "", "combined")
    return apply_gates(summarize_rows(rows, source))


def oracle_switch_outputs(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    udp_rows = load_udp_rows(repo_root)
    if not udp_rows:
        payload = {"status": "missing_udp_source", "rows": 0}
        write_json(output_dir / "expert_bank_oracle_switch_a0_udp.json", payload)
        return payload
    switch_rows = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in udp_rows:
        choose_udp = float(row["delta_psnr"]) > 0
        switched = dict(row)
        switched["choose_udp"] = choose_udp
        switched["switch_psnr"] = row["candidate_psnr"] if choose_udp else row["baseline_psnr"]
        switched["switch_ssim"] = row["candidate_ssim"] if choose_udp else row["baseline_ssim"]
        switched["delta_psnr"] = max(0.0, float(row["delta_psnr"]))
        if row.get("delta_ssim") is None:
            switched["delta_ssim"] = None
        else:
            switched["delta_ssim"] = float(row["delta_ssim"]) if choose_udp else 0.0
        switch_rows.append(switched)
        by_split.setdefault(str(row["split"]), []).append(switched)
    split_summaries = {split: combine_summary(rows, f"oracle_a0_udp_{split}") for split, rows in by_split.items()}
    combined = combine_summary(switch_rows, "oracle_a0_udp_combined")
    payload = {
        "route_id": ROUTE_ID,
        "stage": "A0 + official UDPNet oracle switch",
        "status": "completed_offline_gt_oracle",
        "locked_test_touched": False,
        "note": "Oracle uses GT PSNR to choose UDPNet if and only if UDPNet beats A0; this is an upper bound, not deployable.",
        "count": len(switch_rows),
        "udp_accept_count": sum(1 for row in switch_rows if row["choose_udp"]),
        "udp_accept_ratio": safe_ratio(sum(1 for row in switch_rows if row["choose_udp"]), len(switch_rows)),
        "combined_summary": combined,
        "split_summaries": split_summaries,
        "pass_lines": {
            "oracle_mean_delta_min": 0.30,
            "oracle_hard_bottom25_min": 0.50,
            "oracle_easy_top25_min": 0.0,
        },
        "oracle_gate_pass": bool(
            combined.get("mean_delta") is not None
            and combined["mean_delta"] >= 0.30
            and combined.get("hard_bottom25_delta") is not None
            and combined["hard_bottom25_delta"] >= 0.50
            and combined.get("easy_top25_delta") is not None
            and combined["easy_top25_delta"] >= 0.0
        ),
    }
    write_json(output_dir / "expert_bank_oracle_switch_a0_udp.json", payload)
    write_csv(
        output_dir / "expert_bank_oracle_switch_a0_udp_per_image.csv",
        [
            {
                "split": row["split"],
                "name": row["name"],
                "baseline_psnr": row["baseline_psnr"],
                "udpnet_psnr": row["candidate_psnr"],
                "udpnet_delta_psnr": row["candidate_psnr"] - row["baseline_psnr"],
                "choose_udp": row["choose_udp"],
                "switch_delta_psnr": row["delta_psnr"],
                "baseline_ssim": row.get("baseline_ssim"),
                "udpnet_ssim": row.get("candidate_ssim"),
                "switch_delta_ssim": row.get("delta_ssim"),
                "bucket": row.get("bucket", ""),
            }
            for row in switch_rows
        ],
    )
    return payload


def load_source_rows_by_key(repo_root: Path, source_key: str) -> list[dict[str, Any]]:
    matches = [source for source in DEFAULT_SOURCES if source.key == source_key]
    if not matches:
        return []
    source = matches[0]
    path = repo_root / source.path
    if not path.exists():
        return []
    if source.format == "udp_bucket_compare":
        return normalize_udp_rows(read_csv_rows(path), source)
    if source.format == "generic_per_image":
        return normalize_generic_rows(read_csv_rows(path), source)
    return []


def oracle_udp_fam2_outputs(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    udp_rows = load_source_rows_by_key(repo_root, "udpnet_phase0")
    fam2_rows = load_source_rows_by_key(repo_root, "fam2_confidence")
    fam2_by_name = {row["name"]: row for row in fam2_rows}
    rows = []
    for udp in udp_rows:
        fam2 = fam2_by_name.get(udp["name"])
        choices = [
            ("a0", 0.0, udp["baseline_psnr"], udp.get("baseline_ssim")),
            ("udpnet", float(udp["delta_psnr"]), udp["candidate_psnr"], udp.get("candidate_ssim")),
        ]
        if fam2 is not None:
            choices.append(
                (
                    "fam2_confidence",
                    float(fam2["delta_psnr"]),
                    fam2["candidate_psnr"],
                    fam2.get("candidate_ssim"),
                )
            )
        expert, delta, psnr, ssim = max(choices, key=lambda item: item[1])
        rows.append(
            {
                "split": udp["split"],
                "name": udp["name"],
                "chosen_expert": expert,
                "switch_delta_psnr": max(0.0, delta),
                "a0_psnr": udp["baseline_psnr"],
                "udpnet_psnr": udp["candidate_psnr"],
                "udpnet_delta_psnr": udp["delta_psnr"],
                "fam2_psnr": "" if fam2 is None else fam2["candidate_psnr"],
                "fam2_delta_psnr": "" if fam2 is None else fam2["delta_psnr"],
                "switch_psnr": psnr,
                "switch_ssim": "" if ssim is None else ssim,
                "bucket": udp.get("bucket", ""),
                "fam2_available": fam2 is not None,
            }
        )
    write_csv(output_dir / "expert_bank_oracle_switch_a0_udp_fam2.csv", rows)
    payload = {
        "route_id": ROUTE_ID,
        "stage": "optional A0 + UDPNet + FAM2-confidence oracle alignment",
        "status": "completed_optional_overlap_oracle" if rows else "missing_sources",
        "row_count": len(rows),
        "fam2_overlap_count": sum(1 for row in rows if row.get("fam2_available")),
        "warning": "FAM2-confidence evidence is from an older stop20 split/protocol. This CSV is an overlap diagnostic only, not a promotion gate.",
    }
    write_json(output_dir / "expert_bank_oracle_switch_a0_udp_fam2_summary.json", payload)
    return payload


def parse_filename_features(name: str) -> dict[str, float]:
    stem = Path(name).stem
    parts = stem.split("_")
    nums: list[float] = []
    for part in parts[1:]:
        value = to_float(part)
        if value is not None:
            nums.append(value)
    return {
        "filename_param_1": nums[0] if len(nums) > 0 else float("nan"),
        "filename_param_2": nums[1] if len(nums) > 1 else float("nan"),
    }


def basic_proxy_features(row: dict[str, Any]) -> dict[str, float]:
    feats = parse_filename_features(str(row.get("name", "")))
    base = float(row["baseline_psnr"])
    delta = float(row["candidate_psnr"] - row["baseline_psnr"])
    base_ssim = row.get("baseline_ssim")
    cand_ssim = row.get("candidate_ssim")
    ssim_delta = float(cand_ssim - base_ssim) if base_ssim is not None and cand_ssim is not None else float("nan")
    feats.update(
        {
            "a0_psnr": base,
            "a0_ssim": float(base_ssim) if base_ssim is not None else float("nan"),
            "udp_minus_a0_psnr_delta": delta,
            "udp_minus_a0_ssim_delta": ssim_delta,
            "a0_inverse_psnr": -base,
            "filename_param_1_abs": abs(feats["filename_param_1"]) if math.isfinite(feats["filename_param_1"]) else float("nan"),
            "filename_param_2_abs": abs(feats["filename_param_2"]) if math.isfinite(feats["filename_param_2"]) else float("nan"),
        }
    )
    return feats


LEAKY_OR_LABEL_COLUMNS = {
    "split",
    "name",
    "bucket",
    "a0_psnr",
    "udpnet_psnr",
    "delta_psnr",
    "a0_ssim",
    "udpnet_ssim",
    "delta_ssim",
    "gain_label_ge_0_05",
    "gain_label_ge_0_10",
    "gain_label_ge_0_20",
    "bad_risk_label_delta_le_-0_20_or_ssim_le_-0_001",
}


def load_feature_table(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = read_csv_rows(path)
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for key, value in raw.items():
            fvalue = to_float(value)
            if fvalue is not None:
                row[key] = fvalue
        rows.append(row)
    if raw_rows:
        for key in raw_rows[0]:
            if key in LEAKY_OR_LABEL_COLUMNS:
                continue
            if key.startswith("filename_param"):
                continue
            values = [to_float(row.get(key)) for row in raw_rows]
            if any(value is not None for value in values):
                feature_names.append(key)
    return rows, feature_names


def labels_from_udp_metric_table(repo_root: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    udp_rows = load_udp_rows(repo_root)
    labels_rows: list[dict[str, Any]] = []
    feature_names = [
        "a0_inverse_psnr",
        "filename_param_1",
        "filename_param_2",
        "filename_param_1_abs",
        "filename_param_2_abs",
    ]
    for row in udp_rows:
        delta = float(row["delta_psnr"])
        ssim_delta = row.get("delta_ssim")
        feats = basic_proxy_features(row)
        accept = int(delta >= 0.10)
        risk = int(delta <= -0.20 or (ssim_delta is not None and float(ssim_delta) <= -0.001))
        out = {
            "split": row["split"],
            "name": row["name"],
            "a0_psnr": row["baseline_psnr"],
            "udpnet_psnr": row["candidate_psnr"],
            "delta_psnr": delta,
            "delta_ssim": row.get("delta_ssim"),
            "gain_label_ge_0_05": int(delta >= 0.05),
            "gain_label_ge_0_10": accept,
            "gain_label_ge_0_20": int(delta >= 0.20),
            "bad_risk_label_delta_le_-0_20_or_ssim_le_-0_001": risk,
            "bucket": row.get("bucket", ""),
        }
        out.update(feats)
        labels_rows.append(out)
    return labels_rows, feature_names, "metric_proxy_no_delta_leak"


def label_predictability_outputs(
    repo_root: Path,
    output_dir: Path,
    udp_feature_csv: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], str]:
    if udp_feature_csv and udp_feature_csv.exists():
        labels_rows, feature_names = load_feature_table(udp_feature_csv)
        feature_source = "udp_switch_feature_table"
    else:
        labels_rows, feature_names, feature_source = labels_from_udp_metric_table(repo_root)
    if not labels_rows:
        payload = {"status": "missing_udp_source", "rows": 0}
        write_json(output_dir / "udp_accept_label_predictability_oof.json", payload)
        write_json(output_dir / "udp_bad_risk_predictability_oof.json", payload)
        return payload, labels_rows, feature_names, feature_source
    by_feature: dict[str, list[float]] = {name: [] for name in feature_names}
    accept_labels: list[int] = []
    risk_labels: list[int] = []
    for row in labels_rows:
        accept = int(to_float(row.get("gain_label_ge_0_10"), 0.0) or 0)
        risk = int(to_float(row.get("bad_risk_label_delta_le_-0_20_or_ssim_le_-0_001"), 0.0) or 0)
        accept_labels.append(accept)
        risk_labels.append(risk)
        for name in feature_names:
            value = to_float(row.get(name), float("nan"))
            by_feature[name].append(value if value is not None else float("nan"))
    pred_rows = []
    for feature in feature_names:
        scores = by_feature[feature]
        pred_rows.append(
            {
                "feature": feature,
                "accept_auc": roc_auc(scores, accept_labels),
                "accept_pr_auc": average_precision(scores, accept_labels),
                "accept_spearman": spearman(scores, [float(x) for x in accept_labels]),
                "accept_brier_minmax": brier_score(scores, accept_labels),
                "risk_auc": roc_auc(scores, risk_labels),
                "risk_pr_auc": average_precision(scores, risk_labels),
                "risk_spearman": spearman(scores, [float(x) for x in risk_labels]),
                "risk_brier_minmax": brier_score(scores, risk_labels),
                "note": "AUC orientation uses high feature score as positive; values below 0.5 may still be useful after inversion.",
            }
        )
    accept_payload = {
        "route_id": ROUTE_ID,
        "stage": "UDP accept label predictability",
        "status": "completed_proxy_feature_diagnostic",
        "label": "UDPNet delta_psnr >= +0.10 dB",
        "count": len(labels_rows),
        "positive_count": sum(accept_labels),
        "positive_ratio": safe_ratio(sum(accept_labels), len(accept_labels)),
        "feature_source": feature_source,
        "feature_count": len(feature_names),
        "features": pred_rows,
        "warning": "No PSNR/SSIM delta label columns are allowed as router features. Feature source controls whether this is metric-proxy or deployable-feature evidence.",
    }
    risk_payload = {
        "route_id": ROUTE_ID,
        "stage": "UDP bad-risk label predictability",
        "status": "completed_proxy_feature_diagnostic",
        "label": "UDPNet delta_psnr <= -0.20 dB or delta_ssim <= -0.001",
        "count": len(labels_rows),
        "positive_count": sum(risk_labels),
        "positive_ratio": safe_ratio(sum(risk_labels), len(risk_labels)),
        "feature_source": feature_source,
        "feature_count": len(feature_names),
        "features": pred_rows,
        "warning": "No PSNR/SSIM delta label columns are allowed as router features. Feature source controls whether this is metric-proxy or deployable-feature evidence.",
    }
    write_csv(output_dir / "udp_accept_label_training_table.csv", labels_rows)
    write_csv(output_dir / "udp_accept_label_predictability_oof.csv", pred_rows)
    write_csv(output_dir / "udp_bad_risk_predictability_oof.csv", pred_rows)
    write_json(output_dir / "udp_accept_label_predictability_oof.json", accept_payload)
    write_json(output_dir / "udp_bad_risk_predictability_oof.json", risk_payload)
    return accept_payload, labels_rows, feature_names, feature_source


def fold_id_for_row(row: dict[str, Any], fold_count: int = 5) -> int:
    name = str(row.get("name", ""))
    total = sum(ord(ch) for ch in name)
    split_offset = 0 if row.get("split") == "val_regular" else 1
    return (total + 17 * split_offset) % fold_count


def threshold_candidates(values: list[float], percentiles: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    return sorted({percentile(finite, pct) for pct in percentiles if percentile(finite, pct) is not None})


def threshold_keep(score: float, threshold: float, direction: str) -> bool:
    if not math.isfinite(score):
        return False
    if direction == "high":
        return score >= threshold
    if direction == "low":
        return score <= threshold
    raise ValueError(direction)


def summarize_switch_policy(rows: list[dict[str, Any]], keep_flags: list[bool], label: str) -> dict[str, Any]:
    switched = []
    for row, keep in zip(rows, keep_flags):
        out = dict(row)
        if keep:
            out["delta_psnr"] = float(row["delta_psnr"])
            out["delta_ssim"] = row.get("delta_ssim")
        else:
            out["candidate_psnr"] = row["baseline_psnr"]
            out["delta_psnr"] = 0.0
            out["delta_ssim"] = 0.0 if row.get("delta_ssim") is not None else None
        switched.append(out)
    summary = combine_summary(switched, label)
    summary["coverage"] = safe_ratio(sum(keep_flags), len(keep_flags))
    summary["udp_accept_count"] = sum(keep_flags)
    return summary


def policy_gate(summary: dict[str, Any], stage: str) -> bool:
    if stage == "utility":
        return bool(summary.get("utility_gate_pass"))
    if stage == "promotion":
        return bool(summary.get("promotion_gate_pass"))
    return bool(summary.get("mechanism_gate_pass"))


def oof_switch_outputs(
    label_rows: list[dict[str, Any]],
    output_dir: Path,
    feature_names: list[str],
    feature_source: str,
) -> dict[str, Any]:
    if not label_rows:
        payload = {"status": "missing_label_rows", "rows": 0}
        write_json(output_dir / "rc_expert_switch_oof_summary.json", payload)
        return payload
    for row in label_rows:
        row["fold"] = fold_id_for_row(row)
        row["baseline_psnr"] = row["a0_psnr"]
        row["candidate_psnr"] = row["udpnet_psnr"]
        row["delta_psnr"] = row["delta_psnr"]
    percentiles = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    candidate_rows: list[dict[str, Any]] = []

    def numeric_feature(row: dict[str, Any], feature: str) -> float:
        value = to_float(row.get(feature), float("nan"))
        return value if value is not None else float("nan")

    def policy_row(
        rows: list[dict[str, Any]],
        feature: str,
        direction: str,
        threshold: float,
        fold: int | str,
        phase: str,
    ) -> dict[str, Any]:
        keep_flags = [
            threshold_keep(numeric_feature(row, feature), threshold, direction)
            for row in rows
        ]
        summary = summarize_switch_policy(rows, keep_flags, f"{phase}_{fold}_{feature}_{direction}_{threshold:.6g}")
        return {
            "fold": fold,
            "phase": phase,
            "feature": feature,
            "direction": direction,
            "threshold": threshold,
            "coverage": summary.get("coverage"),
            "udp_accept_count": summary.get("udp_accept_count"),
            "mean_delta": summary.get("mean_delta"),
            "hard_bottom25_delta": summary.get("hard_bottom25_delta"),
            "easy_top25_delta": summary.get("easy_top25_delta"),
            "mean_ssim_delta": summary.get("mean_ssim_delta"),
            "strong_regression_ratio": summary.get("strong_regression_ratio"),
            "worst_regression_ratio": summary.get("worst_regression_ratio"),
            "utility_score": summary.get("utility_score"),
            "mechanism_gate_pass": summary.get("mechanism_gate_pass"),
            "utility_gate_pass": summary.get("utility_gate_pass"),
            "promotion_gate_pass": summary.get("promotion_gate_pass"),
            "feature_source": feature_source,
            "note": "Threshold selected on train folds only for OOF rows; PSNR/SSIM delta labels excluded as router features.",
        }

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            not bool(row.get("promotion_gate_pass")),
            not bool(row.get("utility_gate_pass")),
            -(row.get("utility_score") if row.get("utility_score") is not None else -999.0),
            -(row.get("mean_delta") if row.get("mean_delta") is not None else -999.0),
            -(row.get("hard_bottom25_delta") if row.get("hard_bottom25_delta") is not None else -999.0),
            row.get("worst_regression_ratio") if row.get("worst_regression_ratio") is not None else 999.0,
            row.get("strong_regression_ratio") if row.get("strong_regression_ratio") is not None else 999.0,
        )

    # Retain a post-hoc all-row search as an upper diagnostic, but do not use it
    # for the OOF result.
    posthoc_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        for direction in ("high", "low"):
            values = [numeric_feature(row, feature) for row in label_rows]
            for threshold in threshold_candidates(values, percentiles):
                posthoc_rows.append(policy_row(label_rows, feature, direction, threshold, "all", "posthoc"))
    posthoc_rows = sorted(posthoc_rows, key=sort_key)

    fold_policies: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    oof_eval_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in label_rows if row["fold"] != fold]
        val_rows = [row for row in label_rows if row["fold"] == fold]
        train_candidates: list[dict[str, Any]] = []
        for feature in feature_names:
            for direction in ("high", "low"):
                values = [numeric_feature(row, feature) for row in train_rows]
                for threshold in threshold_candidates(values, percentiles):
                    train_candidates.append(policy_row(train_rows, feature, direction, threshold, fold, "train_select"))
        train_candidates = sorted(train_candidates, key=sort_key)
        selected = train_candidates[0] if train_candidates else None
        if selected is None:
            fold_policies.append({"fold": fold, "phase": "train_select", "status": "no_candidate"})
            continue
        val_policy = policy_row(
            val_rows,
            str(selected["feature"]),
            str(selected["direction"]),
            float(selected["threshold"]),
            fold,
            "val_oof",
        )
        fold_policies.append({**selected, **{f"val_{k}": v for k, v in val_policy.items() if k not in {"fold", "phase", "feature", "direction", "threshold", "feature_source", "note"}}})
        for row in val_rows:
            feature_value = numeric_feature(row, str(selected["feature"]))
            keep = threshold_keep(feature_value, float(selected["threshold"]), str(selected["direction"]))
            selected_delta = float(row["delta_psnr"]) if keep else 0.0
            selected_ssim = row.get("delta_ssim") if keep else 0.0
            oof_row = dict(row)
            oof_row["delta_psnr"] = selected_delta
            oof_row["delta_ssim"] = selected_ssim
            if not keep:
                oof_row["candidate_psnr"] = row["baseline_psnr"]
            oof_eval_rows.append(oof_row)
            per_image.append(
                {
                    "fold": fold,
                    "split": row["split"],
                    "name": row["name"],
                    "choose_udp": keep,
                    "a0_psnr": row["a0_psnr"],
                    "udpnet_psnr": row["udpnet_psnr"],
                    "udpnet_delta_psnr": row["delta_psnr"],
                    "switch_delta_psnr": selected_delta,
                    "delta_ssim": selected_ssim,
                    "feature": selected["feature"],
                    "feature_value": feature_value,
                    "threshold": selected["threshold"],
                    "direction": selected["direction"],
                    "bucket": row.get("bucket", ""),
                }
            )
    oof_summary = combine_summary(oof_eval_rows, "rc_switch_true_oof") if oof_eval_rows else None
    if oof_summary is not None:
        oof_summary["coverage"] = safe_ratio(sum(1 for row in per_image if row["choose_udp"]), len(per_image))
        oof_summary["udp_accept_count"] = sum(1 for row in per_image if row["choose_udp"])
    fixed_policy_payload: dict[str, Any] | None = None
    fixed_policy_rows: list[dict[str, Any]] = []
    selected_pairs = [
        (str(row.get("feature")), str(row.get("direction")))
        for row in fold_policies
        if row.get("feature") and row.get("direction") and row.get("threshold") is not None
    ]
    if selected_pairs:
        pair_counts: dict[tuple[str, str], int] = {}
        for pair in selected_pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        feature, direction = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        thresholds = [
            float(row["threshold"])
            for row in fold_policies
            if row.get("feature") == feature and row.get("direction") == direction and row.get("threshold") is not None
        ]
        if thresholds:
            fixed_threshold = statistics.median(thresholds)
            fixed_keep = [
                threshold_keep(numeric_feature(row, feature), fixed_threshold, direction)
                for row in label_rows
            ]
            fixed_summary = summarize_switch_policy(
                label_rows,
                fixed_keep,
                f"fixed_{feature}_{direction}_{fixed_threshold:.6g}",
            )
            for row, keep in zip(label_rows, fixed_keep):
                fixed_policy_rows.append(
                    {
                        "split": row["split"],
                        "name": row["name"],
                        "choose_udp": keep,
                        "a0_psnr": row["a0_psnr"],
                        "udpnet_psnr": row["udpnet_psnr"],
                        "udpnet_delta_psnr": row["delta_psnr"],
                        "switch_delta_psnr": row["delta_psnr"] if keep else 0.0,
                        "delta_ssim": row.get("delta_ssim") if keep else 0.0,
                        "feature": feature,
                        "feature_value": numeric_feature(row, feature),
                        "threshold": fixed_threshold,
                        "direction": direction,
                        "bucket": row.get("bucket", ""),
                    }
                )
            fixed_policy_payload = {
                "route_id": ROUTE_ID,
                "stage": "fixed threshold candidate derived from fold-selected policies",
                "status": "completed_fixed_policy_candidate",
                "feature_source": feature_source,
                "feature": feature,
                "direction": direction,
                "threshold": fixed_threshold,
                "fold_thresholds": thresholds,
                "fold_pair_counts": {f"{k[0]}:{k[1]}": v for k, v in pair_counts.items()},
                "summary": fixed_summary,
                "locked_test_allowed_by_this_file": bool(fixed_summary.get("promotion_gate_pass")),
                "warning": "This is a fixed internal candidate. Locked test still requires route-card closeout and immutable one-shot command.",
            }
    passing_utility = [row for row in posthoc_rows if row.get("utility_gate_pass")]
    passing_promotion = [row for row in posthoc_rows if row.get("promotion_gate_pass")]
    payload = {
        "route_id": ROUTE_ID,
        "stage": "OOF risk-calibrated expert switch threshold diagnostic",
        "status": "completed_true_oof_threshold_search",
        "feature_source": feature_source,
        "warning": "Fold thresholds are selected on train folds and applied to held-out fold rows. This is still a threshold diagnostic, not a final learned router.",
        "posthoc_candidate_count": len(posthoc_rows),
        "posthoc_utility_gate_pass_count": len(passing_utility),
        "posthoc_promotion_gate_pass_count": len(passing_promotion),
        "best_posthoc_policy": posthoc_rows[0] if posthoc_rows else None,
        "best_posthoc_utility_policy": passing_utility[0] if passing_utility else None,
        "best_posthoc_promotion_policy": passing_promotion[0] if passing_promotion else None,
        "fold_policies": fold_policies,
        "true_oof_summary": oof_summary,
        "true_oof_utility_gate_pass": bool(oof_summary and oof_summary.get("utility_gate_pass")),
        "true_oof_promotion_gate_pass": bool(oof_summary and oof_summary.get("promotion_gate_pass")),
        "fixed_policy_candidate": fixed_policy_payload,
    }
    write_csv(output_dir / "rc_expert_switch_oof_policy_search.csv", fold_policies)
    write_csv(output_dir / "rc_expert_switch_posthoc_policy_search.csv", posthoc_rows)
    write_csv(output_dir / "rc_expert_switch_oof_per_image.csv", per_image)
    if fixed_policy_payload is not None:
        write_json(output_dir / "rc_expert_switch_fixed_policy_candidate.json", fixed_policy_payload)
        write_csv(output_dir / "rc_expert_switch_fixed_policy_candidate_per_image.csv", fixed_policy_rows)
    write_json(output_dir / "rc_expert_switch_oof_summary.json", payload)
    return payload


def write_readme(output_dir: Path, payloads: dict[str, Any]) -> None:
    lines = [
        "# Haze4K v1.6 Risk-Calibrated Expert Switch Evidence",
        "",
        "Status: `OFFLINE_INTERMEDIATE_ANALYSIS_COMPLETE` after the analysis script finishes.",
        "",
        "Primary files:",
        "",
        "- `route_utility_leaderboard.csv`: unified retrospective Mechanism/Utility/Promotion gate table.",
        "- `hard_expert_leaderboard.csv`: hard expert candidate ranking.",
        "- `global_model_leaderboard.csv`: strict promotion-style ranking.",
        "- `utility_tradeoff_leaderboard.csv`: risk-adjusted utility ranking.",
        "- `expert_bank_oracle_switch_a0_udp.json`: GT oracle upper bound for A0 + official UDPNet.",
        "- `expert_bank_oracle_switch_a0_udp_per_image.csv`: oracle per-image switch decisions.",
        "- `udp_accept_label_predictability_oof.csv`: first accept-label predictability diagnostic.",
        "- `udp_bad_risk_predictability_oof.csv`: first bad-risk predictability diagnostic.",
        "- `udp_accept_label_training_table.csv`: label table for later deployable router work.",
        "- `rc_expert_switch_oof_summary.json`: first threshold-switch OOF diagnostic.",
        "- `rc_expert_switch_oof_policy_search.csv`: searched threshold policies.",
        "",
        "Locked Haze4K test touched: no.",
        "",
        "Important interpretation: oracle and proxy-threshold switch outputs are",
        "intermediate diagnostics. Promotion still requires a deployable feature",
        "audit and OOF or held-out calibration that does not select thresholds on",
        "the locked test.",
        "",
    ]
    if payloads.get("oracle"):
        oracle = payloads["oracle"]
        lines.extend(
            [
                "## Oracle Snapshot",
                "",
                f"- Status: `{oracle.get('status')}`",
                f"- UDP accept ratio: `{oracle.get('udp_accept_ratio')}`",
                f"- Oracle gate pass: `{oracle.get('oracle_gate_pass')}`",
                "",
            ]
        )
    if payloads.get("switch"):
        switch = payloads["switch"]
        lines.extend(
            [
                "## Switch Diagnostic Snapshot",
                "",
                f"- Status: `{switch.get('status')}`",
                f"- Utility-gate policies: `{switch.get('utility_gate_pass_count')}`",
                f"- Promotion-gate policies: `{switch.get('promotion_gate_pass_count')}`",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument(
        "--output_dir",
        default=f"experience_docx/experiment_logs/{ROUTE_ID}/offline_intermediate_analysis",
    )
    parser.add_argument("--udp_feature_csv", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    udp_feature_csv = Path(args.udp_feature_csv).resolve() if args.udp_feature_csv else None
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, missing = load_route_summaries(repo_root)
    leaderboard_payload = leaderboard_outputs(output_dir, summaries, missing)
    oracle_payload = oracle_switch_outputs(repo_root, output_dir)
    udp_fam2_payload = oracle_udp_fam2_outputs(repo_root, output_dir)
    accept_payload, label_rows, feature_names, feature_source = label_predictability_outputs(
        repo_root,
        output_dir,
        udp_feature_csv,
    )
    switch_payload = oof_switch_outputs(label_rows, output_dir, feature_names, feature_source)
    status = {
        "route_id": ROUTE_ID,
        "status": "OFFLINE_INTERMEDIATE_ANALYSIS_COMPLETE",
        "leaderboard": leaderboard_payload,
        "oracle": oracle_payload,
        "udp_fam2_oracle": udp_fam2_payload,
        "accept_predictability": accept_payload,
        "switch": switch_payload,
        "feature_source": feature_source,
        "udp_feature_csv": "" if udp_feature_csv is None else str(udp_feature_csv),
    }
    write_json(output_dir / "offline_intermediate_analysis_status.json", status)
    write_readme(output_dir, status)
    print(
        "RC_EXPERT_SWITCH_OFFLINE_ANALYSIS_OK "
        f"output_dir={output_dir} "
        f"route_summaries={len(summaries)} "
        f"missing_sources={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
