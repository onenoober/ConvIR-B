#!/usr/bin/env python3
"""C1b leakage-safe deployable-proxy audit for A0/FullUDP selection.

This audit intentionally excludes split labels and filename-derived haze
parameters. It answers a narrow question before C2: can a deployable proxy that
is available at inference time (currently only A0 PSNR/quality) select FullUDP
without relying on validation split membership or synthetic filename metadata?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable


STRICT_GATE = {
    "mean_dPSNR": 0.12,
    "hard_bottom25_dPSNR": 0.20,
    "easy_top25_dPSNR": -0.02,
    "dSSIM": 0.0,
    "positive_ratio": 0.65,
    "severe_loss_per_600": 48.0,
}

ABSTENTION_GATE = {
    "mean_dPSNR": 0.12,
    "hard_bottom25_dPSNR": 0.20,
    "easy_top25_dPSNR": -0.02,
    "dSSIM": 0.0,
    "selected_precision": 0.65,
    "nonnegative_ratio": 0.90,
    "severe_loss_per_600": 48.0,
    "coverage": 0.10,
}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def enrich(row: dict[str, str]) -> dict[str, Any]:
    delta = fnum(row.get("delta_psnr"))
    return {
        "image_id": row.get("name", ""),
        "A0_PSNR": fnum(row.get("a0_psnr")),
        "FullUDP_PSNR": fnum(row.get("udpnet_psnr")),
        "dPSNR": delta,
        "dSSIM": fnum(row.get("delta_ssim")),
        "target_positive": int(delta > 0.0),
        "target_nonnegative": int(delta >= 0.0),
        "target_severe_loss": int(delta <= -0.20),
    }


def fold_id(image_id: str, folds: int = 5) -> int:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def summarize_policy(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {}
    selected_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    ssims: list[float] = []
    selected_count = 0
    selected_positive = 0
    selected_nonnegative = 0
    selected_severe = 0
    for row in rows:
        choose = predicate(row)
        if choose:
            selected_count += 1
            selected_rows.append(row)
            dpsnr = float(row["dPSNR"])
            dssim = float(row["dSSIM"])
            selected_positive += int(dpsnr > 0.0)
            selected_nonnegative += int(dpsnr >= 0.0)
            selected_severe += int(dpsnr <= -0.20)
        else:
            dpsnr = 0.0
            dssim = 0.0
        deltas.append(dpsnr)
        ssims.append(dssim)
    order = sorted(range(count), key=lambda i: float(rows[i]["A0_PSNR"]))
    k = max(1, count // 4)
    severe = sum(1 for d in deltas if d <= -0.20)
    strong = sum(1 for d in deltas if d <= -0.05)
    coverage = selected_count / count
    selected_precision = selected_positive / selected_count if selected_count else 0.0
    selected_nonnegative_ratio = selected_nonnegative / selected_count if selected_count else 1.0
    return {
        "count": count,
        "selected_count": selected_count,
        "coverage": coverage,
        "mean_dPSNR": mean(deltas),
        "hard_bottom25_dPSNR": mean([deltas[i] for i in order[:k]]),
        "easy_top25_dPSNR": mean([deltas[i] for i in order[-k:]]),
        "dSSIM": mean(ssims),
        "positive_ratio": sum(1 for d in deltas if d > 0.0) / count,
        "nonnegative_ratio": sum(1 for d in deltas if d >= 0.0) / count,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / count * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / count * 600.0,
        "selected_precision": selected_precision,
        "selected_nonnegative_ratio": selected_nonnegative_ratio,
        "selected_severe_count": selected_severe,
    }


def strict_gate_pass(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= STRICT_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= STRICT_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= STRICT_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= STRICT_GATE["dSSIM"]
        and fnum(row.get("positive_ratio")) >= STRICT_GATE["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= STRICT_GATE["severe_loss_per_600"]
    )


def abstention_gate_pass(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= ABSTENTION_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= ABSTENTION_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= ABSTENTION_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= ABSTENTION_GATE["dSSIM"]
        and fnum(row.get("selected_precision")) >= ABSTENTION_GATE["selected_precision"]
        and fnum(row.get("nonnegative_ratio")) >= ABSTENTION_GATE["nonnegative_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= ABSTENTION_GATE["severe_loss_per_600"]
        and fnum(row.get("coverage")) >= ABSTENTION_GATE["coverage"]
    )


def score(row: dict[str, Any]) -> float:
    return (
        fnum(row.get("mean_dPSNR"))
        + 0.25 * fnum(row.get("hard_bottom25_dPSNR"))
        + 0.05 * fnum(row.get("selected_precision"))
        - 0.002 * fnum(row.get("severe_loss_per_600"))
    )


def threshold_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vals = [float(r["A0_PSNR"]) for r in rows if math.isfinite(float(r["A0_PSNR"]))]
    quantiles = [i / 100 for i in range(5, 96, 5)] + [0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90]
    thresholds = sorted({quantile(vals, q) for q in quantiles if math.isfinite(quantile(vals, q))})
    policies: list[dict[str, Any]] = []

    def add(name: str, predicate: Callable[[dict[str, Any]], bool]) -> None:
        rec = {"policy_id": name}
        rec.update(summarize_policy(rows, predicate))
        rec["strict_gate_pass"] = strict_gate_pass(rec)
        rec["abstention_gate_pass"] = abstention_gate_pass(rec)
        rec["score"] = score(rec)
        policies.append(rec)

    add("a0_anchor", lambda _r: False)
    add("all_fulludp", lambda _r: True)
    for threshold in thresholds:
        add(f"A0_PSNR_le_{threshold:.6g}", lambda r, threshold=threshold: float(r["A0_PSNR"]) <= threshold)
        add(f"A0_PSNR_ge_{threshold:.6g}", lambda r, threshold=threshold: float(r["A0_PSNR"]) >= threshold)
    policies.sort(key=lambda r: (bool(r["strict_gate_pass"]), bool(r["abstention_gate_pass"]), fnum(r["score"])), reverse=True)
    return policies


def parse_threshold(policy_id: str) -> tuple[str, float]:
    if policy_id.startswith("A0_PSNR_le_"):
        return "le", float(policy_id.removeprefix("A0_PSNR_le_"))
    if policy_id.startswith("A0_PSNR_ge_"):
        return "ge", float(policy_id.removeprefix("A0_PSNR_ge_"))
    if policy_id == "all_fulludp":
        return "all", 0.0
    return "none", 0.0


def make_predicate(kind: str, threshold: float) -> Callable[[dict[str, Any]], bool]:
    if kind == "le":
        return lambda r: float(r["A0_PSNR"]) <= threshold
    if kind == "ge":
        return lambda r: float(r["A0_PSNR"]) >= threshold
    if kind == "all":
        return lambda _r: True
    return lambda _r: False


def choose_policy(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = threshold_grid(train_rows)
    strict = [r for r in grid if r["strict_gate_pass"]]
    if strict:
        return strict[0]
    abstention = [r for r in grid if r["abstention_gate_pass"]]
    if abstention:
        return abstention[0]
    return grid[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulludp-eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [enrich(r) for r in read_csv(args.fulludp_eval_dir / "udpnet_convir_bucket_compare.csv")]
    if not rows:
        raise SystemExit("no rows loaded")

    policy_rows = threshold_grid(rows)
    fields = [
        "policy_id",
        "count",
        "selected_count",
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "severe_loss_count",
        "severe_loss_per_600",
        "strong_loss_count",
        "strong_loss_per_600",
        "selected_precision",
        "selected_nonnegative_ratio",
        "selected_severe_count",
        "strict_gate_pass",
        "abstention_gate_pass",
        "score",
    ]
    write_csv(out_dir / "v20_c1b_deployable_policy_grid.csv", policy_rows, fields)

    fold_eval_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train = [r for r in rows if fold_id(str(r["image_id"])) != fold]
        heldout = [r for r in rows if fold_id(str(r["image_id"])) == fold]
        chosen = choose_policy(train)
        kind, threshold = parse_threshold(str(chosen["policy_id"]))
        predicate = make_predicate(kind, threshold)
        eval_rec = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_strict_gate_pass": chosen["strict_gate_pass"],
            "train_abstention_gate_pass": chosen["abstention_gate_pass"],
        }
        eval_rec.update(summarize_policy(heldout, predicate))
        eval_rec["strict_gate_pass"] = strict_gate_pass(eval_rec)
        eval_rec["abstention_gate_pass"] = abstention_gate_pass(eval_rec)
        eval_rec["score"] = score(eval_rec)
        fold_eval_rows.append(eval_rec)
        for row in heldout:
            choose = predicate(row)
            clone = dict(row)
            clone["fold"] = fold
            clone["chosen_fulludp"] = choose
            clone["policy_id"] = chosen["policy_id"]
            oof_rows.append(clone)
    write_csv(out_dir / "v20_c1b_oof_fold_metrics.csv", fold_eval_rows, ["fold", "train_policy_id", "train_strict_gate_pass", "train_abstention_gate_pass"] + fields[1:])

    # Reconstruct the OOF policy using each row's held-out decision.
    oof_count = len(oof_rows)
    oof_selected = {str(r["image_id"]) for r in oof_rows if r["chosen_fulludp"]}
    oof_summary = summarize_policy(rows, lambda r: str(r["image_id"]) in oof_selected)
    oof_summary["strict_gate_pass"] = strict_gate_pass(oof_summary)
    oof_summary["abstention_gate_pass"] = abstention_gate_pass(oof_summary)
    oof_summary["score"] = score(oof_summary)
    oof_summary["count"] = oof_count

    strict_pass = [r for r in policy_rows if r["strict_gate_pass"]]
    abstention_pass = [r for r in policy_rows if r["abstention_gate_pass"]]
    best_policy = policy_rows[0]
    if oof_summary["strict_gate_pass"]:
        decision = "C1B_DEPLOYABLE_STRICT_PASS_START_C2_ROUTER"
    elif oof_summary["abstention_gate_pass"]:
        decision = "C1B_ABSTENTION_PROXY_PASS_REACQUIRE_OUTPUTS_FOR_C2"
    elif abstention_pass:
        decision = "C1B_IN_SAMPLE_ABSTENTION_ONLY_NEEDS_OOF_OR_OUTPUTDIFF"
    else:
        decision = "C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C1b Deployable-Feature Audit",
        "locked_test_touched": False,
        "excluded_features": ["split", "filename/name_param_1", "filename/name_param_2"],
        "allowed_features": ["A0_PSNR"],
        "rows": len(rows),
        "strict_gate": STRICT_GATE,
        "abstention_gate": ABSTENTION_GATE,
        "best_policy": best_policy,
        "best_strict_policy": strict_pass[0] if strict_pass else None,
        "best_abstention_policy": abstention_pass[0] if abstention_pass else None,
        "oof_summary": oof_summary,
        "decision": decision,
    }
    with (out_dir / "v20_c1b_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# Haze4K v2.0 C1b Deployable-Feature Audit",
        "",
        f"Decision: `{decision}`",
        "",
        "This corrected C1 audit excludes validation split membership and filename-derived parameters.",
        "It uses existing internal-validation A0/FullUDP endpoint metrics only; locked test data was not touched.",
        "",
        "## Gates",
        "",
        f"- Strict gate includes all-sample positive ratio `>= {STRICT_GATE['positive_ratio']}`.",
        f"- Abstention-aware gate requires selected precision `>= {ABSTENTION_GATE['selected_precision']}`, all-sample nonnegative ratio `>= {ABSTENTION_GATE['nonnegative_ratio']}`, and coverage `>= {ABSTENTION_GATE['coverage']}`.",
        "",
        "## Best In-Sample Deployable Proxy",
        "",
    ]
    for key in fields:
        if key in best_policy:
            lines.append(f"- `{key}`: `{best_policy[key]}`")
    lines.extend(["", "## OOF Threshold Replay", ""])
    for key, value in oof_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C1b is a leakage-safe audit; it is not a trained image/patch router.",
            "- A C2 router should not claim deployability from split/name-param policies.",
            "- If C1b does not pass, the efficient next step is to reacquire/render FullUDP outputs and compute real output-difference, depth, texture, and artifact features before router training.",
        ]
    )
    (out_dir / "v20_c1b_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C1B_DEPLOYABLE_PROXY_OK decision={decision} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
