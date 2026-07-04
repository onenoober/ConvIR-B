#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(TOOL_PATH.parent), str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

import nopost_lowband_v220_o3_context_learnability_audit as v220  # noqa: E402


SEVERE = v220.SEVERE
STRONG_REG = v220.STRONG_REG


def tag_float(value: float) -> str:
    return f"{value:.2f}".replace("-", "neg").replace(".", "p")


def scale_pair(
    mid: torch.Tensor,
    final: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_mid = scales.float().view(-1, 1, 1, 1)
    scale_final = scales.float().view(-1, 1, 1, 1)
    return mid * scale_mid, final * scale_final


def hard_scale(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    return (probs < threshold).float()


def soft_scale(probs: torch.Tensor, threshold: float, tau: float) -> torch.Tensor:
    return torch.sigmoid((threshold - probs.float()) / max(tau, 1e-6))


def piecewise_scale(probs: torch.Tensor, low: float, high: float) -> torch.Tensor:
    p = probs.float()
    out = torch.ones_like(p)
    out = torch.where(p >= high, torch.zeros_like(out), out)
    middle = (p >= low) & (p < high)
    out = torch.where(middle, ((high - p) / max(high - low, 1e-6)).clamp(0.0, 1.0), out)
    return out


def temperature_scale(probs: torch.Tensor, gamma: float) -> torch.Tensor:
    return (1.0 - probs.float()).clamp(0.0, 1.0) ** gamma


def pairwise_roc_auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, 1):
        if labels[idx]:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def brier(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return float("nan")
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return float("nan")
    total = len(scores)
    err = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        subset = [(s, y) for s, y in zip(scores, labels) if lo <= s < hi or (i == bins - 1 and s == 1.0)]
        if not subset:
            continue
        conf = v220.mean([s for s, _ in subset])
        acc = v220.mean([float(y) for _, y in subset])
        err += len(subset) / total * abs(conf - acc)
    return err


def make_base_p1(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    v220.write_text(
        args.out_dir / "v221_p1_safety_gated_replay_protocol.md",
        "\n".join(
            [
                "# v2.21 Safety-Calibrated NoPost Lowband Replay Protocol",
                "",
                "This is a no-training replay audit. It reuses the v2.20 O3 mid+final/global context action predictor and tests whether train-derived safety/no-op signals can gate the action safely.",
                "",
                "The final candidate must be fixed across folds. Fold-specific thresholds are diagnostic only.",
                "",
                "Compared variants:",
                "",
                "- no-op control",
                "- raw v2.20 O3 action",
                "- fixed-threshold risk-zero action",
                "- risk soft-shrink action",
                "- piecewise risk shrink action",
                "- risk-temperature scaling action",
                "- shuffled-risk control",
                "- oracle-risk upper bound",
                "- factorial predicted/oracle action x predicted/oracle gate audit",
                "",
                "Training, N3 microfit, and locked Haze4K test remain blocked unless this replay gate passes.",
            ]
        ),
    )
    (
        x_final_all,
        x_mid_all,
        x_global_all,
        y_o2_final,
        y_o3_mid,
        y_o3_final,
        meta_rows,
    ) = v220.generate_targets(args, model, device)
    v220.write_csv(args.out_dir / "v221_p1_o3_target_energy_summary.csv", meta_rows)
    pred_by_variant, fold_report = v220.fit_fold_predictors(
        args,
        x_final_all,
        x_mid_all,
        x_global_all,
        y_o2_final,
        y_o3_mid,
        y_o3_final,
        meta_rows,
        device,
    )
    raw_replay = v220.replay_predictions(args, model, pred_by_variant, y_o3_mid, y_o3_final, meta_rows, device)
    variants = sorted(pred_by_variant)
    summary_rows = [v220.summarize_rows(raw_replay, variant) for variant in variants]
    direction_rows = [v220.direction_summary(raw_replay, variant) for variant in variants]
    folds = sorted({int(row["oof_fold"]) for row in meta_rows})
    fold_summary_rows = []
    for variant in variants:
        for fold in folds:
            subset = [r for r in raw_replay if r["variant"] == variant and int(r["oof_fold"]) == fold]
            fold_summary_rows.append(v220.summarize_rows(subset, variant, scope=f"raw_fold{fold}"))

    candidate_variants = [
        "P1_mid_only_context_predictor",
        "P1_final_mid_context_predictor",
        "P1_final_mid_global_context_predictor",
    ]
    shuffled_summary = next(row for row in summary_rows if row["variant"] == "P1_shuffled_target_control")
    direction_by_variant = {row["variant"]: row for row in direction_rows}
    gate_by_variant: dict[str, Any] = {}
    for variant in candidate_variants:
        s = next(row for row in summary_rows if row["variant"] == variant)
        gate_by_variant[variant] = v220.p1_gates(
            summary=s,
            shuffled_summary=shuffled_summary,
            direction=direction_by_variant[variant],
            fold_summaries=fold_summary_rows,
            fold_report=fold_report,
            variant=variant,
        )
        s["v220_mechanism_gate_pass"] = gate_by_variant[variant]["mechanism_pass"]
        s["v220_training_authorization_gate_pass"] = gate_by_variant[variant]["training_authorization_pass"]
        s["v220_control_gap_vs_shuffled"] = gate_by_variant[variant]["control_gap_vs_shuffled"]
        s["v220_fold_tail_ok_count"] = gate_by_variant[variant]["fold_tail_ok_count"]

    primary_variant = "P1_final_mid_global_context_predictor"
    if primary_variant not in pred_by_variant:
        mechanism_passing = [v for v in candidate_variants if gate_by_variant[v]["mechanism_pass"]]
        primary_variant = max(
            mechanism_passing or candidate_variants,
            key=lambda v: float(next(row for row in summary_rows if row["variant"] == v)["mean_dPSNR"]),
        )
    primary_summary = next(row for row in summary_rows if row["variant"] == primary_variant)
    primary_gate = gate_by_variant.get(primary_variant, {})
    v220.write_csv(args.out_dir / "v221_p1_raw_replay_summary.csv", summary_rows)
    v220.write_csv(args.out_dir / "v221_p1_raw_direction_shape_stats.csv", direction_rows)
    v220.write_csv(args.out_dir / "v221_p1_raw_context_predictor_fold_report.csv", fold_report + fold_summary_rows)
    v220.write_text(
        args.out_dir / "v221_p1_raw_action_anchor_decision.md",
        "\n".join(
            [
                "# v2.21 Raw v2.20 Action Anchor",
                "",
                f"Primary raw action: `{primary_variant}`",
                "",
                f"- mean dPSNR: `{primary_summary['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{primary_summary['hard_bottom25_dPSNR']}`",
                f"- easy top25 dPSNR: `{primary_summary['easy_top25_dPSNR']}`",
                f"- p05 dPSNR: `{primary_summary['p05_dPSNR']}`",
                f"- CVaR5 dPSNR: `{primary_summary['CVaR5_dPSNR']}`",
                f"- severe rate: `{primary_summary['severe_rate']}`",
                f"- strong-reference regression rate: `{primary_summary['strong_reference_regression_rate']}`",
                f"- v2.20 mechanism pass: `{primary_gate.get('mechanism_pass')}`",
                f"- v2.20 training authorization pass: `{primary_gate.get('training_authorization_pass')}`",
                "",
                "v2.21 does not train from this raw action. It only tests whether safety/no-op calibration can make the action replay-safe.",
            ]
        ),
    )
    return {
        "x_final_all": x_final_all,
        "x_mid_all": x_mid_all,
        "x_global_all": x_global_all,
        "y_o2_final": y_o2_final,
        "y_o3_mid": y_o3_mid,
        "y_o3_final": y_o3_final,
        "meta_rows": meta_rows,
        "pred_by_variant": pred_by_variant,
        "raw_replay": raw_replay,
        "raw_summary": summary_rows,
        "raw_direction": direction_rows,
        "fold_report": fold_report,
        "primary_variant": primary_variant,
    }


def compute_oof_risk(
    args: argparse.Namespace,
    base: dict[str, Any],
) -> dict[str, Any]:
    primary = base["primary_variant"]
    raw_rows = [r for r in base["raw_replay"] if r["variant"] == primary]
    flags = v220.make_group_flags(raw_rows)
    meta_by_name = {str(r["name"]): r for r in base["meta_rows"]}
    x_global = []
    labels = []
    names = []
    for row in raw_rows:
        name = str(row["name"])
        m = meta_by_name[name]
        x_global.append(
            torch.tensor(
                [
                    float(row["A0_PSNR"]),
                    float(m["final_context_abs_mean"]),
                    float(m["mid_context_abs_mean"]),
                    float(m["o3_oracle_dPSNR"]),
                    float(m["o3_minus_o2_dPSNR"]),
                    float(row["mid_pred_delta_rms"]),
                    float(row["final_pred_delta_rms"]),
                    float(row["cosine_to_o3_delta"]),
                    float(flags[name]["easy_top25"]),
                    float(flags[name]["strong_reference"]),
                ],
                dtype=torch.float32,
            )
        )
        labels.append(int(float(row["dPSNR"]) <= STRONG_REG))
        names.append(name)
    x_all = torch.stack(x_global)
    y_all = torch.tensor(labels, dtype=torch.float32)
    folds = sorted({int(r["oof_fold"]) for r in raw_rows})
    prob_by_name: dict[str, float] = {}
    fold_report = []
    threshold_rows = []
    for fold in folds:
        train_idx = [i for i, row in enumerate(raw_rows) if int(row["oof_fold"]) != fold]
        test_idx = [i for i, row in enumerate(raw_rows) if int(row["oof_fold"]) == fold]
        probs = v220.train_logistic_classifier(
            x_all[train_idx],
            y_all[train_idx],
            x_all[test_idx],
            epochs=args.classifier_epochs,
            seed=args.seed + 900 + fold,
        )
        test_scores = [float(v) for v in probs]
        test_labels = [int(y_all[i].item()) for i in test_idx]
        for local_i, global_i in enumerate(test_idx):
            prob_by_name[names[global_i]] = float(probs[local_i])
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            row = v220.binary_metrics(test_scores, test_labels, threshold=threshold)
            row.update({"fold": fold, "variant": primary, "train_count": len(train_idx), "test_count": len(test_idx)})
            threshold_rows.append(row)
        report = v220.binary_metrics(test_scores, test_labels, threshold=0.5)
        report.update({"fold": fold, "variant": primary, "train_count": len(train_idx), "test_count": len(test_idx)})
        fold_report.append(report)

    enriched = []
    for row in raw_rows:
        name = str(row["name"])
        enriched.append(
            {
                **row,
                **flags[name],
                "unsafe_action_label": int(float(row["dPSNR"]) <= STRONG_REG),
                "unsafe_action_probability": prob_by_name[name],
                "raw_action_dPSNR": row["dPSNR"],
            }
        )
    return {
        "primary_variant": primary,
        "raw_rows": raw_rows,
        "enriched_rows": enriched,
        "flags": flags,
        "prob_by_name": prob_by_name,
        "label_by_name": {str(r["name"]): int(r["unsafe_action_label"]) for r in enriched},
        "fold_report": fold_report,
        "threshold_rows": threshold_rows,
    }


def make_variant_specs_and_preds(
    args: argparse.Namespace,
    base: dict[str, Any],
    risk: dict[str, Any],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, dict[str, Any]], dict[str, torch.Tensor]]:
    primary = base["primary_variant"]
    base_mid, base_final = base["pred_by_variant"][primary]
    y_mid = base["y_o3_mid"]
    y_final = base["y_o3_final"]
    probs = torch.tensor([risk["prob_by_name"][str(row["name"])] for row in base["meta_rows"]], dtype=torch.float32)
    raw_by_name = {str(r["name"]): float(r["dPSNR"]) for r in risk["raw_rows"]}
    oracle_safe = torch.tensor(
        [1.0 if raw_by_name[str(row["name"])] > STRONG_REG else 0.0 for row in base["meta_rows"]],
        dtype=torch.float32,
    )
    rng = random.Random(args.seed + 221)
    shuffled_idx = list(range(len(probs)))
    rng.shuffle(shuffled_idx)
    shuffled_probs = probs[shuffled_idx]

    preds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    specs: dict[str, dict[str, Any]] = {}
    scales: dict[str, torch.Tensor] = {}

    def add(name: str, mid: torch.Tensor, final: torch.Tensor, spec: dict[str, Any], scale: torch.Tensor) -> None:
        preds[name] = (mid.clone(), final.clone())
        specs[name] = {"variant": name, **spec}
        scales[name] = scale.float().clone()

    zero = torch.zeros(len(probs), dtype=torch.float32)
    one = torch.ones(len(probs), dtype=torch.float32)
    add("V221_noop_control", torch.zeros_like(base_mid), torch.zeros_like(base_final), {"kind": "control_noop"}, zero)
    add("V221_raw_v220_action", base_mid, base_final, {"kind": "raw_v220_action"}, one)
    add("V221_oracle_safe_action_upper_bound", y_mid, y_final, {"kind": "oracle_action_upper_bound"}, one)

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        name = f"V221_risk_zero_t{tag_float(threshold)}"
        scale = hard_scale(probs, threshold)
        mid, final = scale_pair(base_mid, base_final, scale)
        add(name, mid, final, {"kind": "predicted_risk_zero", "threshold": threshold}, scale)

    for threshold, tau in [(0.5, 0.1), (0.5, 0.2), (0.7, 0.1), (0.7, 0.2)]:
        name = f"V221_risk_soft_t{tag_float(threshold)}_tau{tag_float(tau)}"
        scale = soft_scale(probs, threshold, tau)
        mid, final = scale_pair(base_mid, base_final, scale)
        add(name, mid, final, {"kind": "predicted_risk_soft", "threshold": threshold, "tau": tau}, scale)

    for low, high in [(0.3, 0.7), (0.4, 0.8), (0.5, 0.9)]:
        name = f"V221_risk_piecewise_t{tag_float(low)}_{tag_float(high)}"
        scale = piecewise_scale(probs, low, high)
        mid, final = scale_pair(base_mid, base_final, scale)
        add(name, mid, final, {"kind": "predicted_risk_piecewise", "threshold_low": low, "threshold_high": high}, scale)

    for gamma in [0.5, 1.0, 2.0, 4.0]:
        name = f"V221_risk_temperature_gamma{tag_float(gamma)}"
        scale = temperature_scale(probs, gamma)
        mid, final = scale_pair(base_mid, base_final, scale)
        add(name, mid, final, {"kind": "predicted_risk_temperature", "gamma": gamma}, scale)

    scale = hard_scale(shuffled_probs, 0.5)
    mid, final = scale_pair(base_mid, base_final, scale)
    add("V221_shuffled_risk_zero_t0p50", mid, final, {"kind": "shuffled_risk_control", "threshold": 0.5}, scale)

    mid, final = scale_pair(base_mid, base_final, oracle_safe)
    add(
        "V221_oracle_risk_zero_upper_bound",
        mid,
        final,
        {"kind": "oracle_risk_gate_upper_bound", "oracle_safe_definition": "raw predicted action dPSNR > -0.05"},
        oracle_safe,
    )
    return preds, specs, scales


def add_risk_columns(
    rows: list[dict[str, Any]],
    risk: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    scales: dict[str, torch.Tensor],
    meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    name_to_index = {str(row["name"]): idx for idx, row in enumerate(meta_rows)}
    raw_by_name = {str(r["name"]): float(r["raw_action_dPSNR"]) for r in risk["enriched_rows"]}
    out = []
    for row in rows:
        name = str(row["name"])
        variant = str(row["variant"])
        spec = specs.get(variant, {})
        idx = name_to_index[name]
        risk_prob = float(risk["prob_by_name"][name])
        scale = float(scales[variant][idx]) if variant in scales else float("nan")
        out.append(
            {
                **row,
                **risk["flags"].get(name, {}),
                "unsafe_action_probability": risk_prob,
                "unsafe_action_label": risk["label_by_name"].get(name, 0),
                "risk_scale": scale,
                "raw_action_dPSNR": raw_by_name[name],
                "gate_kind": spec.get("kind", ""),
                "gate_threshold": spec.get("threshold", spec.get("threshold_high", "")),
            }
        )
    return out


def fold_tail_pass_count(rows: list[dict[str, Any]], variant: str) -> int:
    folds = sorted({int(row["oof_fold"]) for row in rows if row["variant"] == variant})
    count = 0
    for fold in folds:
        subset = [r for r in rows if r["variant"] == variant and int(r["oof_fold"]) == fold]
        summary = v220.summarize_rows(subset, variant, scope=f"fold{fold}")
        if (
            float(summary["easy_top25_dPSNR"]) >= 0.0
            and float(summary["p05_dPSNR"]) >= -0.15
            and float(summary["severe_rate"]) <= 0.035
            and float(summary["strong_reference_regression_rate"]) <= 0.075
        ):
            count += 1
    return count


def gate_calibration_for_variant(rows: list[dict[str, Any]], variant: str, spec: dict[str, Any]) -> dict[str, Any]:
    subset = [r for r in rows if r["variant"] == variant]
    high = [r for r in subset if float(r["unsafe_action_probability"]) >= 0.8]
    raw_high_severe = v220.mean([float(float(r["raw_action_dPSNR"]) <= SEVERE) for r in high])
    gated_high_severe = v220.mean([float(float(r["dPSNR"]) <= SEVERE) for r in high])
    threshold = spec.get("threshold")
    if threshold is None:
        threshold = spec.get("threshold_high", 0.8)
    noop_bin = [r for r in subset if float(r["unsafe_action_probability"]) >= float(threshold)]
    return {
        "variant": variant,
        "gate_kind": spec.get("kind", ""),
        "threshold": threshold,
        "high_prob_count": len(high),
        "raw_high_prob_severe_rate": raw_high_severe,
        "gated_high_prob_severe_rate": gated_high_severe,
        "high_prob_severe_reduction": raw_high_severe - gated_high_severe if not math.isnan(raw_high_severe) and not math.isnan(gated_high_severe) else float("nan"),
        "noop_bin_count": len(noop_bin),
        "noop_bin_mean_dPSNR": v220.mean([float(r["dPSNR"]) for r in noop_bin]),
    }


def v221_gate(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    variant: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    calibration = gate_calibration_for_variant(rows, variant, spec)
    fold_count = fold_tail_pass_count(rows, variant)
    checks = {
        "mean_dPSNR_ge_1p00": float(summary["mean_dPSNR"]) >= 1.00,
        "hard_bottom25_ge_2p00": float(summary["hard_bottom25_dPSNR"]) >= 2.00,
        "easy_top25_ge_0p00": float(summary["easy_top25_dPSNR"]) >= 0.00,
        "positive_ratio_ge_0p75": float(summary["positive_ratio"]) >= 0.75,
        "p05_ge_neg0p15": float(summary["p05_dPSNR"]) >= -0.15,
        "CVaR5_ge_neg0p35": float(summary["CVaR5_dPSNR"]) >= -0.35,
        "severe_rate_le_0p035": float(summary["severe_rate"]) <= 0.035,
        "strong_reference_regression_rate_le_0p075": float(summary["strong_reference_regression_rate"]) <= 0.075,
        "fold_tail_pass_ge_4_of_5": fold_count >= 4,
        "strong_easy_p05_ge_neg0p15": float(summary["strong_easy_p05_dPSNR"]) >= -0.15,
        "noop_bin_mean_ge_neg0p03": float(calibration["noop_bin_mean_dPSNR"]) >= -0.03,
        "high_prob_severe_rate_clearly_reduced": (
            float(calibration["high_prob_severe_reduction"]) >= 0.10
            and float(calibration["gated_high_prob_severe_rate"]) <= max(float(calibration["raw_high_prob_severe_rate"]) * 0.75, 0.035)
        ),
    }
    return {
        "variant": variant,
        "gate_kind": spec.get("kind", ""),
        "threshold": spec.get("threshold", spec.get("threshold_high", "")),
        "training_authorization_pass": all(checks.values()),
        "failed_check_count": sum(1 for ok in checks.values() if not ok),
        "fold_tail_pass_count": fold_count,
        **checks,
        **{f"calibration_{k}": v for k, v in calibration.items() if k not in {"variant", "gate_kind", "threshold"}},
    }


def select_candidate(gate_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], specs: dict[str, dict[str, Any]]) -> str:
    disallowed = {"control_noop", "raw_v220_action", "oracle_action_upper_bound", "shuffled_risk_control", "oracle_risk_gate_upper_bound"}
    candidates = [r for r in gate_rows if specs[r["variant"]].get("kind") not in disallowed]
    passing = [r for r in candidates if bool(r["training_authorization_pass"])]
    summary_by_variant = {r["variant"]: r for r in summary_rows}
    if passing:
        return max(passing, key=lambda r: float(summary_by_variant[r["variant"]]["mean_dPSNR"]))["variant"]
    return min(
        candidates,
        key=lambda r: (
            int(r["failed_check_count"]),
            -float(summary_by_variant[r["variant"]]["p05_dPSNR"]),
            -float(summary_by_variant[r["variant"]]["mean_dPSNR"]),
        ),
    )["variant"]


def run_p1(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    base: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    preds, specs, scales = make_variant_specs_and_preds(args, base, risk)
    replay_rows = v220.replay_predictions(args, model, preds, base["y_o3_mid"], base["y_o3_final"], base["meta_rows"], device)
    replay_rows = add_risk_columns(replay_rows, risk, specs, scales, base["meta_rows"])
    variants = sorted(preds)
    summary_rows = [v220.summarize_rows(replay_rows, variant) for variant in variants]
    gate_rows = [v221_gate(summary, replay_rows, str(summary["variant"]), specs[str(summary["variant"])]) for summary in summary_rows]
    calibration_rows = [gate_calibration_for_variant(replay_rows, variant, specs[variant]) for variant in variants]
    selected = select_candidate(gate_rows, summary_rows, specs)
    selected_summary = next(row for row in summary_rows if row["variant"] == selected)
    selected_gate = next(row for row in gate_rows if row["variant"] == selected)

    threshold_by_fold_rows = []
    for variant in variants:
        spec = specs[variant]
        if spec.get("kind") not in {"predicted_risk_zero", "predicted_risk_soft", "predicted_risk_piecewise", "predicted_risk_temperature"}:
            continue
        folds = sorted({int(r["oof_fold"]) for r in replay_rows if r["variant"] == variant})
        for fold in folds:
            subset = [r for r in replay_rows if r["variant"] == variant and int(r["oof_fold"]) == fold]
            row = v220.summarize_rows(subset, variant, scope=f"fold{fold}")
            row.update(
                {
                    "fold": fold,
                    "gate_kind": spec.get("kind", ""),
                    "threshold": spec.get("threshold", spec.get("threshold_high", "")),
                    "threshold_low": spec.get("threshold_low", ""),
                    "gamma": spec.get("gamma", ""),
                    "tau": spec.get("tau", ""),
                    "fold_tail_pass": int(
                        float(row["easy_top25_dPSNR"]) >= 0
                        and float(row["p05_dPSNR"]) >= -0.15
                        and float(row["severe_rate"]) <= 0.035
                        and float(row["strong_reference_regression_rate"]) <= 0.075
                    ),
                }
            )
            threshold_by_fold_rows.append(row)

    factor_preds, factor_specs, factor_scales = make_factorial_preds(base, risk, selected, specs[selected], scales[selected])
    factor_rows = v220.replay_predictions(args, model, factor_preds, base["y_o3_mid"], base["y_o3_final"], base["meta_rows"], device)
    factor_rows = add_risk_columns(factor_rows, risk, factor_specs, factor_scales, base["meta_rows"])
    factor_summary = [v220.summarize_rows(factor_rows, variant) for variant in sorted(factor_preds)]
    factor_gate_rows = [v221_gate(summary, factor_rows, str(summary["variant"]), factor_specs[str(summary["variant"])]) for summary in factor_summary]

    v220.write_csv(args.out_dir / "v221_p1_safety_gated_replay_metrics.csv", replay_rows)
    v220.write_csv(args.out_dir / "v221_p1_fixed_threshold_oof_summary.csv", summary_rows)
    v220.write_csv(args.out_dir / "v221_p1_gate_checks.csv", gate_rows)
    v220.write_csv(args.out_dir / "v221_p1_threshold_sweep_by_fold.csv", threshold_by_fold_rows)
    v220.write_csv(args.out_dir / "v221_p1_gate_calibration_report.csv", calibration_rows)
    v220.write_csv(args.out_dir / "v221_p1_factorial_action_gate_audit.csv", factor_summary + factor_gate_rows)

    raw_by_name = {str(r["name"]): r for r in risk["enriched_rows"]}
    selected_rows = [r for r in replay_rows if r["variant"] == selected]
    tail_rescue = []
    strong_rescue = []
    for row in selected_rows:
        name = str(row["name"])
        raw = raw_by_name[name]
        item = {
            "name": name,
            "oof_fold": row["oof_fold"],
            "selected_variant": selected,
            "unsafe_action_probability": row["unsafe_action_probability"],
            "risk_scale": row["risk_scale"],
            "raw_action_dPSNR": raw["raw_action_dPSNR"],
            "gated_dPSNR": row["dPSNR"],
            "raw_severe": int(float(raw["raw_action_dPSNR"]) <= SEVERE),
            "gated_severe": int(float(row["dPSNR"]) <= SEVERE),
            "rescued_to_nonsevere": int(float(raw["raw_action_dPSNR"]) <= SEVERE and float(row["dPSNR"]) > SEVERE),
            "raw_strong_regression": int(int(raw["strong_reference"]) and float(raw["raw_action_dPSNR"]) <= STRONG_REG),
            "gated_strong_regression": int(int(raw["strong_reference"]) and float(row["dPSNR"]) <= STRONG_REG),
            **{k: raw[k] for k in ("hard_bottom25", "easy_top25", "strong_reference", "middle50")},
        }
        if item["raw_severe"] or item["gated_severe"]:
            tail_rescue.append(item)
        if int(raw["strong_reference"]) or int(raw["easy_top25"]):
            strong_rescue.append(item)
    v220.write_csv(args.out_dir / "v221_p1_tail_rescue_manifest.csv", sorted(tail_rescue, key=lambda r: float(r["gated_dPSNR"])))
    v220.write_csv(args.out_dir / "v221_p1_strong_easy_rescue_manifest.csv", sorted(strong_rescue, key=lambda r: float(r["gated_dPSNR"])))

    factor_a = next(row for row in factor_gate_rows if row["variant"].startswith("V221_factor_A_"))
    training_authorized = bool(selected_gate["training_authorization_pass"]) and bool(factor_a["training_authorization_pass"])
    decision = (
        "V221_P1_REPLAY_GATE_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED"
        if training_authorized
        else "V221_P1_REPLAY_GATE_FAIL_NORMAL_PAUSE_NO_TRAINING"
    )
    v220.write_text(
        args.out_dir / "v221_p1_decision.md",
        "\n".join(
            [
                "# v2.21 P1 Safety-Gated Replay Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- selected fixed OOF candidate: `{selected}`",
                f"- selected gate kind: `{specs[selected].get('kind')}`",
                f"- selected threshold: `{specs[selected].get('threshold', specs[selected].get('threshold_high', ''))}`",
                f"- mean dPSNR: `{selected_summary['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{selected_summary['hard_bottom25_dPSNR']}`",
                f"- easy top25 dPSNR: `{selected_summary['easy_top25_dPSNR']}`",
                f"- p05 dPSNR: `{selected_summary['p05_dPSNR']}`",
                f"- CVaR5 dPSNR: `{selected_summary['CVaR5_dPSNR']}`",
                f"- severe rate: `{selected_summary['severe_rate']}`",
                f"- strong-reference regression rate: `{selected_summary['strong_reference_regression_rate']}`",
                f"- fold tail pass count: `{selected_gate['fold_tail_pass_count']}` / 5",
                f"- selected candidate gate pass: `{selected_gate['training_authorization_pass']}`",
                f"- factorial A gate pass: `{factor_a['training_authorization_pass']}`",
                "",
                "Gate checks:",
                "",
                json.dumps(selected_gate, indent=2, sort_keys=True),
                "",
                "Factorial A checks:",
                "",
                json.dumps(factor_a, indent=2, sort_keys=True),
                "",
                "No training, no N3 microfit, and no locked-test command is launched by v2.21.",
            ]
        ),
    )
    return {
        "decision": decision,
        "training_authorized": training_authorized,
        "selected_variant": selected,
        "selected_spec": specs[selected],
        "selected_gate": selected_gate,
        "selected_summary": selected_summary,
        "replay_rows": replay_rows,
        "factor_rows": factor_rows,
        "factor_gate_rows": factor_gate_rows,
        "gate_rows": gate_rows,
        "summary_rows": summary_rows,
        "specs": specs,
    }


def make_factorial_preds(
    base: dict[str, Any],
    risk: dict[str, Any],
    selected: str,
    selected_spec: dict[str, Any],
    selected_scale: torch.Tensor,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, dict[str, Any]], dict[str, torch.Tensor]]:
    primary = base["primary_variant"]
    base_mid, base_final = base["pred_by_variant"][primary]
    y_mid = base["y_o3_mid"]
    y_final = base["y_o3_final"]
    raw_by_name = {str(r["name"]): float(r["raw_action_dPSNR"]) for r in risk["enriched_rows"]}
    oracle_safe = torch.tensor(
        [1.0 if raw_by_name[str(row["name"])] > STRONG_REG else 0.0 for row in base["meta_rows"]],
        dtype=torch.float32,
    )
    preds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    specs: dict[str, dict[str, Any]] = {}
    scales: dict[str, torch.Tensor] = {}

    def add(name: str, mid: torch.Tensor, final: torch.Tensor, kind: str, scale: torch.Tensor) -> None:
        preds[name] = (mid.clone(), final.clone())
        specs[name] = {
            **selected_spec,
            "variant": name,
            "kind": kind,
            "source_selected_variant": selected,
            "source_selected_gate_kind": selected_spec.get("kind", ""),
        }
        scales[name] = scale.float().clone()

    mid, final = scale_pair(base_mid, base_final, selected_scale)
    add("V221_factor_A_pred_action_pred_gate", mid, final, "factor_A_pred_action_pred_gate", selected_scale)
    mid, final = scale_pair(base_mid, base_final, oracle_safe)
    add("V221_factor_B_pred_action_oracle_gate", mid, final, "factor_B_pred_action_oracle_gate", oracle_safe)
    mid, final = scale_pair(y_mid, y_final, selected_scale)
    add("V221_factor_C_oracle_action_pred_gate", mid, final, "factor_C_oracle_action_pred_gate", selected_scale)
    oracle_action_safe = torch.ones_like(oracle_safe)
    mid, final = scale_pair(y_mid, y_final, oracle_action_safe)
    add("V221_factor_D_oracle_action_oracle_gate", mid, final, "factor_D_oracle_action_oracle_gate", oracle_action_safe)
    return preds, specs, scales


def run_p2(args: argparse.Namespace, risk: dict[str, Any], p1: dict[str, Any]) -> dict[str, Any]:
    rows = risk["enriched_rows"]
    scores = [float(r["unsafe_action_probability"]) for r in rows]
    labels = [int(r["unsafe_action_label"]) for r in rows]
    folds = sorted({int(r["oof_fold"]) for r in rows})
    auc_rows = [
        {
            "scope": "all",
            "count": len(rows),
            "positive_labels": sum(labels),
            "roc_auc": pairwise_roc_auc(scores, labels),
            "pr_auc_average_precision": average_precision(scores, labels),
        }
    ]
    cal_rows = [
        {
            "scope": "all",
            "count": len(rows),
            "brier": brier(scores, labels),
            "ece10": ece(scores, labels, bins=10),
            "mean_probability": v220.mean(scores),
            "label_rate": v220.mean([float(v) for v in labels]),
        }
    ]
    for fold in folds:
        subset = [r for r in rows if int(r["oof_fold"]) == fold]
        fs = [float(r["unsafe_action_probability"]) for r in subset]
        fl = [int(r["unsafe_action_label"]) for r in subset]
        auc_rows.append(
            {
                "scope": f"fold{fold}",
                "count": len(subset),
                "positive_labels": sum(fl),
                "roc_auc": pairwise_roc_auc(fs, fl),
                "pr_auc_average_precision": average_precision(fs, fl),
            }
        )
        cal_rows.append(
            {
                "scope": f"fold{fold}",
                "count": len(subset),
                "brier": brier(fs, fl),
                "ece10": ece(fs, fl, bins=10),
                "mean_probability": v220.mean(fs),
                "label_rate": v220.mean([float(v) for v in fl]),
            }
        )
    bins = []
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        subset = [r for r in rows if lo <= float(r["unsafe_action_probability"]) < hi]
        bins.append(
            {
                "prob_bin": f"[{lo:.1f},{hi:.1f})",
                "count": len(subset),
                "unsafe_rate": v220.mean([float(r["unsafe_action_label"]) for r in subset]),
                "raw_mean_dPSNR": v220.mean([float(r["raw_action_dPSNR"]) for r in subset]),
                "raw_severe_rate": v220.mean([float(float(r["raw_action_dPSNR"]) <= SEVERE) for r in subset]),
            }
        )
    selected_threshold = p1["selected_spec"].get("threshold", p1["selected_spec"].get("threshold_high", 0.8))
    fold_generalization = []
    for fold in folds:
        subset = [r for r in rows if int(r["oof_fold"]) == fold]
        report = v220.binary_metrics(
            [float(r["unsafe_action_probability"]) for r in subset],
            [int(r["unsafe_action_label"]) for r in subset],
            threshold=float(selected_threshold),
        )
        report.update({"fold": fold, "selected_threshold": selected_threshold, "selected_variant": p1["selected_variant"]})
        fold_generalization.append(report)

    v220.write_csv(args.out_dir / "v221_p2_pr_auc_roc_auc_report.csv", auc_rows)
    v220.write_csv(args.out_dir / "v221_p2_brier_ece_calibration_report.csv", cal_rows)
    v220.write_csv(args.out_dir / "v221_p2_probability_bin_report.csv", bins)
    v220.write_csv(args.out_dir / "v221_p2_threshold_stability_report.csv", risk["threshold_rows"])
    v220.write_csv(args.out_dir / "v221_p2_fold_generalization_report.csv", fold_generalization)

    all_auc = auc_rows[0]
    all_cal = cal_rows[0]
    high_bin = bins[-1]
    low_bin = bins[0]
    if float(all_auc["roc_auc"]) >= 0.80 and float(high_bin["raw_severe_rate"]) > float(low_bin["raw_severe_rate"]) + 0.20:
        decision = "P2_SAFETY_SCORE_STRUCTURED_BUT_REPLAY_GATE_DECIDES"
    else:
        decision = "P2_SAFETY_SCORE_WEAK_OR_UNSTABLE_REPLAY_NOT_AUTHORIZED_BY_P2"
    v220.write_text(
        args.out_dir / "v221_p2_decision.md",
        "\n".join(
            [
                "# v2.21 P2 Safety Score Calibration Audit",
                "",
                f"Decision: `{decision}`",
                "",
                f"- ROC AUC: `{all_auc['roc_auc']}`",
                f"- PR AUC/AP: `{all_auc['pr_auc_average_precision']}`",
                f"- Brier: `{all_cal['brier']}`",
                f"- ECE10: `{all_cal['ece10']}`",
                f"- selected threshold: `{selected_threshold}`",
                "",
                "P2 is diagnostic. P1 replay gates decide whether N3 review is allowed.",
            ]
        ),
    )
    return {"decision": decision, "auc": all_auc, "calibration": all_cal}


def run_p3(args: argparse.Namespace, p1: dict[str, Any]) -> dict[str, Any]:
    selected = p1["selected_variant"]
    rows = [r for r in p1["replay_rows"] if r["variant"] == selected]
    group_rows = []
    groups = {
        "all": rows,
        "kept_scale_gt_0p5": [r for r in rows if float(r["risk_scale"]) > 0.5],
        "shrunk_scale_le_0p5": [r for r in rows if float(r["risk_scale"]) <= 0.5],
        "high_risk_prob_ge_0p8": [r for r in rows if float(r["unsafe_action_probability"]) >= 0.8],
        "tail_severe_after_gate": [r for r in rows if float(r["dPSNR"]) <= SEVERE],
    }
    for group, subset in groups.items():
        group_rows.append(
            {
                "group": group,
                "count": len(subset),
                "mean_dPSNR": v220.mean([float(r["dPSNR"]) for r in subset]),
                "raw_mean_dPSNR": v220.mean([float(r["raw_action_dPSNR"]) for r in subset]),
                "mean_risk_scale": v220.mean([float(r["risk_scale"]) for r in subset]),
                "mean_unsafe_probability": v220.mean([float(r["unsafe_action_probability"]) for r in subset]),
                "mean_cosine_to_o3_delta": v220.mean([float(r["cosine_to_o3_delta"]) for r in subset]),
                "wrong_direction_rate": v220.mean([float(r["wrong_direction"]) for r in subset]),
                "mean_mid_local_peak_ratio": v220.mean([float(r["mid_pred_local_peak_ratio"]) for r in subset]),
                "mean_final_local_peak_ratio": v220.mean([float(r["final_pred_local_peak_ratio"]) for r in subset]),
            }
        )
    manifest = sorted(
        [
            r
            for r in rows
            if float(r["dPSNR"]) <= SEVERE or float(r["raw_action_dPSNR"]) <= SEVERE or int(r["wrong_direction"])
        ],
        key=lambda r: float(r["dPSNR"]),
    )[:300]
    v220.write_csv(args.out_dir / "v221_p3_post_gate_action_shape_vs_damage.csv", rows)
    v220.write_csv(args.out_dir / "v221_p3_safe_kept_vs_unsafe_zeroed_report.csv", group_rows)
    v220.write_csv(args.out_dir / "v221_p3_tail_case_shape_manifest.csv", manifest)
    tail = next(r for r in group_rows if r["group"] == "tail_severe_after_gate")
    decision = "P3_POST_GATE_NO_SEVERE_TAIL_SHAPE_RESIDUAL" if int(tail["count"]) == 0 else "P3_POST_GATE_TAIL_SHAPE_RESIDUAL_REMAINS"
    v220.write_text(
        args.out_dir / "v221_p3_decision.md",
        "\n".join(
            [
                "# v2.21 P3 Post-Gate Action Shape Residual Audit",
                "",
                f"Decision: `{decision}`",
                "",
                f"- selected variant: `{selected}`",
                f"- post-gate severe count: `{tail['count']}`",
                "",
                "P3 is diagnostic only.",
            ]
        ),
    )
    return {"decision": decision, "group_rows": group_rows}


def run_p4(args: argparse.Namespace, base: dict[str, Any], p1: dict[str, Any]) -> dict[str, Any]:
    selected = p1["selected_variant"]
    rows = [r for r in p1["replay_rows"] if r["variant"] == selected]
    oracle_norms = [math.sqrt(float(r["mid_target_delta_rms"]) ** 2 + float(r["final_target_delta_rms"]) ** 2) for r in rows]
    pred_norms = [math.sqrt(float(r["mid_pred_delta_rms"]) ** 2 + float(r["final_pred_delta_rms"]) ** 2) for r in rows]
    thresholds = {
        "oracle_p75": v220.percentile(oracle_norms, 75),
        "oracle_p90": v220.percentile(oracle_norms, 90),
        "post_gate_predicted_p75": v220.percentile(pred_norms, 75),
    }
    per_image = []
    for row, oracle_norm, pred_norm in zip(rows, oracle_norms, pred_norms):
        strong_or_easy = bool(int(row.get("strong_reference", 0)) or int(row.get("easy_top25", 0)))
        item = {
            **row,
            "combined_oracle_delta_rms": oracle_norm,
            "combined_post_gate_pred_delta_rms": pred_norm,
            "tail_hinge_margin_neg0p15": max(0.0, -0.15 - float(row["dPSNR"])),
            "severe_hinge_margin_neg0p20": max(0.0, SEVERE - float(row["dPSNR"])),
            "preserve_mask_strong_or_easy": int(strong_or_easy),
            "preserve_hinge_margin_neg0p05": max(0.0, STRONG_REG - float(row["dPSNR"])) if strong_or_easy else 0.0,
            "tail_hinge_active": int(float(row["dPSNR"]) < -0.15),
            "preserve_hinge_active": int(strong_or_easy and float(row["dPSNR"]) <= STRONG_REG),
            "positive_sample": int(float(row["dPSNR"]) > 0),
        }
        for label, threshold in thresholds.items():
            item[f"budget_{label}_active"] = int(pred_norm > threshold)
            item[f"safe_oracle_over_budget_{label}"] = int(oracle_norm > threshold and float(row["o3_oracle_dPSNR"]) > 0)
        per_image.append(item)
    severe_rows = [r for r in per_image if float(r["dPSNR"]) <= SEVERE]
    preserve_fail = [r for r in per_image if int(r["preserve_mask_strong_or_easy"]) and float(r["dPSNR"]) <= STRONG_REG]
    tail_report = [
        {
            "selected_variant": selected,
            "count": len(per_image),
            "severe_count": len(severe_rows),
            "tail_hinge_active_count": sum(int(r["tail_hinge_active"]) for r in per_image),
            "tail_hinge_coverage_on_severe": sum(int(r["tail_hinge_active"]) for r in severe_rows) / len(severe_rows) if severe_rows else 1.0,
            "mean_tail_hinge": v220.mean([float(r["tail_hinge_margin_neg0p15"]) for r in per_image]),
        }
    ]
    preserve_report = [
        {
            "selected_variant": selected,
            "strong_or_easy_count": sum(int(r["preserve_mask_strong_or_easy"]) for r in per_image),
            "strong_or_easy_regression_count": len(preserve_fail),
            "preserve_hinge_active_count": sum(int(r["preserve_hinge_active"]) for r in per_image),
            "preserve_hinge_coverage_on_regressions": sum(int(r["preserve_hinge_active"]) for r in preserve_fail) / len(preserve_fail) if preserve_fail else 1.0,
            "mean_preserve_hinge": v220.mean([float(r["preserve_hinge_margin_neg0p05"]) for r in per_image]),
        }
    ]
    budget_report = []
    safe_oracle_report = []
    for label, threshold in thresholds.items():
        budget_report.append(
            {
                "selected_variant": selected,
                "threshold_label": label,
                "threshold": threshold,
                "budget_activation_rate": v220.mean([float(r[f"budget_{label}_active"]) for r in per_image]),
                "severe_budget_activation_rate": v220.mean([float(r[f"budget_{label}_active"]) for r in severe_rows]),
            }
        )
        safe_oracle_report.append(
            {
                "selected_variant": selected,
                "threshold_label": label,
                "threshold": threshold,
                "safe_oracle_overpenalty_rate": v220.mean([float(r[f"safe_oracle_over_budget_{label}"]) for r in per_image]),
            }
        )
    v220.write_csv(args.out_dir / "v221_p4_per_image_loss_terms.csv", per_image)
    v220.write_csv(args.out_dir / "v221_p4_tail_hinge_activation_report.csv", tail_report)
    v220.write_csv(args.out_dir / "v221_p4_preserve_hinge_activation_report.csv", preserve_report)
    v220.write_csv(args.out_dir / "v221_p4_budget_activation_report.csv", budget_report)
    v220.write_csv(args.out_dir / "v221_p4_safe_oracle_overpenalty_report.csv", safe_oracle_report)
    budget_ok = any(0.02 <= float(r["budget_activation_rate"]) <= 0.25 for r in budget_report)
    decision = "P4_POST_GATE_OBJECTIVE_REPLAY_PASS_AS_GUARD_ONLY" if budget_ok else "P4_POST_GATE_OBJECTIVE_REPLAY_BUDGET_WEAK_OR_OVERACTIVE"
    v220.write_text(
        args.out_dir / "v221_p4_decision.md",
        "\n".join(
            [
                "# v2.21 P4 Objective Replay After Safety Gate",
                "",
                f"Decision: `{decision}`",
                "",
                f"- selected variant: `{selected}`",
                f"- tail coverage on severe: `{tail_report[0]['tail_hinge_coverage_on_severe']}`",
                f"- preserve coverage on regressions: `{preserve_report[0]['preserve_hinge_coverage_on_regressions']}`",
                f"- budget nonzero/not-identity candidate exists: `{budget_ok}`",
                "",
                "P4 remains a guard/objective replay, not training authorization by itself.",
            ]
        ),
    )
    return {"decision": decision, "budget_report": budget_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--steps-o2", type=int, default=25)
    ap.add_argument("--steps-o3", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--delta-scale", type=float, default=0.50)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--cnn-hidden", type=int, default=64)
    ap.add_argument("--cnn-epochs", type=int, default=180)
    ap.add_argument("--shuffle-epochs", type=int, default=100)
    ap.add_argument("--classifier-epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=221)
    ap.add_argument("--print-freq", type=int, default=25)
    args = ap.parse_args()

    from models.ConvIR import build_net

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(v220.load_state(args.checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    base = make_base_p1(args, model, device)
    risk = compute_oof_risk(args, base)
    p1 = run_p1(args, model, device, base, risk)
    p2 = run_p2(args, risk, p1)
    p3 = run_p3(args, p1)
    p4 = run_p4(args, base, p1)

    closeout = {
        "decision": p1["decision"],
        "p1_decision": p1["decision"],
        "p2_decision": p2["decision"],
        "p3_decision": p3["decision"],
        "p4_decision": p4["decision"],
        "selected_variant": p1["selected_variant"],
        "selected_spec": p1["selected_spec"],
        "training_authorized": bool(p1["training_authorized"]),
        "training_launched": False,
        "locked_test_touched": False,
    }
    v220.write_json(args.out_dir / "v221_p1_p2_p3_p4_closeout.json", closeout)
    print("V221_P1_P2_P3_P4_OK", p1["decision"], flush=True)


if __name__ == "__main__":
    main()
