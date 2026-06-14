#!/usr/bin/env python3
"""Build the Haze4K v2.0 strong-expert candidate-zoo capacity audit.

This is an evidence-only audit: it reads existing per-image/internal validation
tables and writes compact CSV/JSON/Markdown artifacts. It does not touch locked
test data and does not train or tune a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    return vals[lo] * (1 - frac) + vals[hi] * frac


def summarize_records(records: list[dict[str, Any]], *, label: str, scope: str) -> dict[str, Any]:
    count = len(records)
    deltas = [fnum(r.get("dPSNR")) for r in records]
    ssims = [fnum(r.get("dSSIM")) for r in records]
    a0s = [fnum(r.get("A0_PSNR")) for r in records]
    if count == 0:
        return {
            "candidate": label,
            "scope": scope,
            "count": 0,
            "mean_dPSNR": "",
            "hard_bottom25_dPSNR": "",
            "easy_top25_dPSNR": "",
            "dSSIM": "",
            "positive_ratio": "",
            "nonnegative_ratio": "",
            "strong_reference_cut_psnr": "",
            "strong_reference_regression_count": "",
            "strong_per_600": "",
            "worst_count_le_-0.20": "",
            "worst_per_600": "",
            "intervention_rate": "",
        }

    order = sorted(range(count), key=lambda i: a0s[i])
    k = max(1, count // 4)
    hard = [deltas[i] for i in order[:k]]
    easy = [deltas[i] for i in order[-k:]]
    strong_cut = quantile(a0s, 0.75)
    strong_count = sum(
        1 for d, a in zip(deltas, a0s, strict=False) if a >= strong_cut and d <= -0.05
    )
    worst_count = sum(1 for d in deltas if d <= -0.20)
    intervention = sum(1 for r in records if str(r.get("chosen_expert", "")).lower() not in {"a0", "a0_anchor", ""})
    return {
        "candidate": label,
        "scope": scope,
        "count": count,
        "mean_dPSNR": mean(deltas),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "dSSIM": mean(ssims),
        "positive_ratio": sum(1 for d in deltas if d > 0.0) / count,
        "nonnegative_ratio": sum(1 for d in deltas if d >= 0.0) / count,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": strong_count,
        "strong_per_600": strong_count / count * 600.0,
        "worst_count_le_-0.20": worst_count,
        "worst_per_600": worst_count / count * 600.0,
        "intervention_rate": intervention / count,
    }


def fulludp_rows(bucket_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for row in bucket_rows:
        dpsnr = fnum(row.get("delta_psnr"))
        dssim = fnum(row.get("delta_ssim"))
        base = fnum(row.get("a0_psnr"))
        split = row.get("split", "")
        name = row.get("name", "")
        endpoint.append(
            {
                "image_id": name,
                "split": split,
                "candidate": "fulludp_official_endpoint",
                "A0_PSNR": base,
                "candidate_PSNR": fnum(row.get("udpnet_psnr")),
                "dPSNR": dpsnr,
                "dSSIM": dssim,
                "bucket": row.get("bucket", ""),
                "chosen_expert": "FullUDP",
                "source": "v15_phase0_official_eval",
            }
        )
        choose_udp = dpsnr > 0.0
        oracle.append(
            {
                "image_id": name,
                "split": split,
                "candidate": "oracle_a0_fulludp_endpoint",
                "A0_PSNR": base,
                "candidate_PSNR": fnum(row.get("udpnet_psnr")) if choose_udp else base,
                "dPSNR": max(0.0, dpsnr),
                "dSSIM": dssim if choose_udp else 0.0,
                "bucket": row.get("bucket", ""),
                "chosen_expert": "FullUDP" if choose_udp else "A0",
                "source": "v15_phase0_official_eval",
            }
        )
    return endpoint, oracle


def append_group_summaries(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    out = [summarize_records(rows, label=label, scope="all")]
    splits = sorted({str(r.get("split", "")) for r in rows if r.get("split", "")})
    for split in splits:
        out.append(summarize_records([r for r in rows if r.get("split") == split], label=label, scope=split))
    return out


def parse_jsonish_counts(value: str) -> Counter[str]:
    if not value:
        return Counter()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return Counter()
    return Counter({str(k): int(v) for k, v in parsed.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dta-evidence-dir", type=Path, required=True)
    parser.add_argument("--fulludp-eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C0 Strong Candidate Zoo Oracle",
        "locked_test_touched": False,
        "inputs": {
            "dta_evidence_dir": str(args.dta_evidence_dir),
            "fulludp_eval_dir": str(args.fulludp_eval_dir),
        },
        "candidate_availability": {
            "A0_anchor": {"available": True, "role": "fallback anchor"},
            "DTA_v37_TAU_oracle": {
                "available": (args.dta_evidence_dir / "v37_tau_real_blend_oracle_grid.csv").exists(),
                "role": "safe-small train-derived oracle evidence",
            },
            "DTA_v37_FDF_oracle": {
                "available": (args.dta_evidence_dir / "v37_real_blend_oracle_grid.csv").exists(),
                "role": "safe-small train-derived oracle evidence",
            },
            "FullUDP_official": {
                "available": (args.fulludp_eval_dir / "udpnet_convir_bucket_compare.csv").exists(),
                "role": "strong-risk expert endpoint evidence",
                "note": "Current convir-4090 checkpoint/repo was not present during this audit; C0a uses existing exact per-image evidence.",
            },
            "ConvIR_L_or_larger": {"available": False, "role": "future candidate if checkpoint/protocol becomes available"},
            "DehazeFormer": {"available": False, "role": "future candidate if checkpoint/protocol becomes available"},
            "PromptIR": {"available": False, "role": "future candidate if checkpoint/protocol becomes available"},
        },
    }

    bucket_rows = read_csv(args.fulludp_eval_dir / "udpnet_convir_bucket_compare.csv")
    endpoint, oracle = fulludp_rows(bucket_rows)

    per_image_rows = endpoint + oracle
    per_image_fields = [
        "image_id",
        "split",
        "candidate",
        "A0_PSNR",
        "candidate_PSNR",
        "dPSNR",
        "dSSIM",
        "bucket",
        "chosen_expert",
        "source",
    ]
    write_csv(out_dir / "v20_candidate_zoo_per_image_metrics.csv", per_image_rows, per_image_fields)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(append_group_summaries(endpoint, "fulludp_official_endpoint"))
    summary_rows.extend(append_group_summaries(oracle, "oracle_a0_fulludp_endpoint"))

    # Include already-computed DTA fixed-policy summaries as safe-small policy rows.
    for path, label in [
        (args.dta_evidence_dir / "v37_d8_fixed_formal_policy_aggregate.csv", "dta_v37_d8_fixed_train_derived_policy"),
        (args.dta_evidence_dir / "v37_d9_locked_fixed_policy_aggregate.csv", "dta_v37_d9_locked_policy"),
    ]:
        rows = read_csv(path)
        for row in rows:
            summary_rows.append(
                {
                    "candidate": label,
                    "scope": "source_aggregate",
                    "count": row.get("count", ""),
                    "mean_dPSNR": row.get("mean_dPSNR", ""),
                    "hard_bottom25_dPSNR": row.get("hard_bottom25_dPSNR", ""),
                    "easy_top25_dPSNR": row.get("easy_top25_dPSNR", ""),
                    "dSSIM": row.get("dSSIM", ""),
                    "positive_ratio": row.get("positive_ratio", ""),
                    "nonnegative_ratio": "",
                    "strong_reference_cut_psnr": row.get("strong_reference_cut_psnr", ""),
                    "strong_reference_regression_count": row.get("strong_reference_regression_count", ""),
                    "strong_per_600": row.get("strong_per_600", ""),
                    "worst_count_le_-0.20": row.get("worst_count_le_-0.20", row.get("worst_count_le_-0.20", "")),
                    "worst_per_600": row.get("worst_per_600", ""),
                    "intervention_rate": row.get("intervention_rate", ""),
                }
            )

    summary_fields = [
        "candidate",
        "scope",
        "count",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "strong_reference_cut_psnr",
        "strong_reference_regression_count",
        "strong_per_600",
        "worst_count_le_-0.20",
        "worst_per_600",
        "intervention_rate",
    ]
    write_csv(out_dir / "v20_candidate_zoo_single_expert_summary.csv", summary_rows, summary_fields)

    alpha_rows: list[dict[str, Any]] = []
    for split in ["all"] + sorted({r["split"] for r in endpoint}):
        scoped_endpoint = endpoint if split == "all" else [r for r in endpoint if r["split"] == split]
        scoped_oracle = oracle if split == "all" else [r for r in oracle if r["split"] == split]
        alpha_rows.append({"candidate_pair": "A0_FullUDP", "scope": split, "alpha": 0.0, **summarize_records([], label="A0", scope=split)})
        alpha_rows[-1].update({"count": len(scoped_endpoint), "mean_dPSNR": 0.0, "hard_bottom25_dPSNR": 0.0, "easy_top25_dPSNR": 0.0, "dSSIM": 0.0, "positive_ratio": 0.0, "nonnegative_ratio": 1.0, "worst_per_600": 0.0})
        alpha_rows.append({"candidate_pair": "A0_FullUDP", "scope": split, "alpha": 1.0, **summarize_records(scoped_endpoint, label="FullUDP", scope=split)})
        alpha_rows.append({"candidate_pair": "A0_FullUDP", "scope": split, "alpha": "oracle_endpoint", **summarize_records(scoped_oracle, label="A0_or_FullUDP", scope=split)})
    alpha_fields = ["candidate_pair", "alpha"] + summary_fields
    write_csv(out_dir / "v20_candidate_zoo_alpha_grid.csv", alpha_rows, alpha_fields)

    oracle_rows: list[dict[str, Any]] = []
    for source_name in ["v37_real_blend_oracle_grid.csv", "v37_tau_real_blend_oracle_grid.csv"]:
        for row in read_csv(args.dta_evidence_dir / source_name):
            oracle_rows.append(
                {
                    "source": source_name,
                    "scope": "train_derived_dta",
                    "bank_name": row.get("bank_name", ""),
                    "utility_mode": row.get("utility_mode", ""),
                    "count": row.get("count", ""),
                    "mean_dPSNR": row.get("mean_dPSNR", ""),
                    "hard_bottom25_dPSNR": row.get("hard_bottom25_dPSNR", ""),
                    "easy_top25_dPSNR": row.get("easy_top25_dPSNR", ""),
                    "dSSIM": row.get("dSSIM", ""),
                    "positive_ratio": row.get("positive_ratio", ""),
                    "worst_per_600": row.get("worst_per_600", ""),
                    "max_outer_worst_per_600": row.get("max_outer_worst_per_600", ""),
                    "intervention_rate": row.get("intervention_rate", ""),
                    "mean_chosen_alpha": row.get("mean_chosen_alpha", ""),
                    "strict_gate_pass": row.get("strict_gate_pass", ""),
                }
            )

    fulludp_oracle_all = summarize_records(oracle, label="oracle_a0_fulludp_endpoint", scope="all")
    oracle_rows.append(
        {
            "source": "v15_phase0_official_eval",
            "scope": "val_regular_plus_val_hard",
            "bank_name": "A0_FullUDP_endpoint",
            "utility_mode": "per_image_max_psnr",
            **{k: fulludp_oracle_all.get(k, "") for k in ["count", "mean_dPSNR", "hard_bottom25_dPSNR", "easy_top25_dPSNR", "dSSIM", "positive_ratio", "worst_per_600", "intervention_rate"]},
            "max_outer_worst_per_600": "",
            "mean_chosen_alpha": "",
            "strict_gate_pass": "",
        }
    )
    oracle_fields = [
        "source",
        "scope",
        "bank_name",
        "utility_mode",
        "count",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "worst_per_600",
        "max_outer_worst_per_600",
        "intervention_rate",
        "mean_chosen_alpha",
        "strict_gate_pass",
    ]
    write_csv(out_dir / "v20_candidate_zoo_oracle_grid.csv", oracle_rows, oracle_fields)

    composition_rows: list[dict[str, Any]] = []
    fulludp_counts = Counter(r["chosen_expert"] for r in oracle)
    for expert, count in sorted(fulludp_counts.items()):
        composition_rows.append({"source": "v15_phase0_official_eval", "bank_name": "A0_FullUDP_endpoint", "choice": expert, "count": count, "ratio": count / max(1, len(oracle))})
    for source_name in ["v37_real_blend_oracle_grid.csv", "v37_tau_real_blend_oracle_grid.csv"]:
        for row in read_csv(args.dta_evidence_dir / source_name):
            counts = parse_jsonish_counts(row.get("chosen_variant_counts", ""))
            total = sum(counts.values()) or 1
            for choice, count in sorted(counts.items()):
                composition_rows.append({"source": source_name, "bank_name": row.get("bank_name", ""), "choice": choice, "count": count, "ratio": count / total})
    write_csv(out_dir / "v20_candidate_zoo_oracle_composition.csv", composition_rows, ["source", "bank_name", "choice", "count", "ratio"])

    failure_rows: list[dict[str, Any]] = []
    for candidate, records in [("fulludp_official_endpoint", endpoint), ("oracle_a0_fulludp_endpoint", oracle)]:
        for split in sorted({r["split"] for r in records}):
            split_rows = [r for r in records if r["split"] == split]
            for bucket in sorted({r["bucket"] for r in split_rows}):
                bucket_rows_scoped = [r for r in split_rows if r["bucket"] == bucket]
                row = summarize_records(bucket_rows_scoped, label=candidate, scope=f"{split}:{bucket}")
                row["bucket"] = bucket
                row["split"] = split
                failure_rows.append(row)
    write_csv(out_dir / "v20_candidate_zoo_failure_bins.csv", failure_rows, ["candidate", "split", "bucket"] + summary_fields[1:])

    gate = {
        "mean_ge_0_30": fnum(fulludp_oracle_all.get("mean_dPSNR")) >= 0.30,
        "hard_ge_0_30": fnum(fulludp_oracle_all.get("hard_bottom25_dPSNR")) >= 0.30,
        "positive_ge_0_75": fnum(fulludp_oracle_all.get("positive_ratio")) >= 0.75,
        "easy_ge_neg_0_02": fnum(fulludp_oracle_all.get("easy_top25_dPSNR")) >= -0.02,
        "dssim_ge_0": fnum(fulludp_oracle_all.get("dSSIM")) >= 0.0,
        "worst_per_600_le_5": fnum(fulludp_oracle_all.get("worst_per_600")) <= 5.0,
    }
    if all(gate.values()):
        decision = "C0_CAPACITY_PASS_START_STRONGEXPERT_ROUTER"
    elif gate["mean_ge_0_30"] and gate["hard_ge_0_30"] and gate["worst_per_600_le_5"]:
        decision = "C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED"
    else:
        decision = "C0_CAPACITY_INSUFFICIENT_FIND_OR_TRAIN_STRONGER_EXPERTS"

    manifest["gate"] = gate
    manifest["decision"] = decision
    manifest["fulludp_oracle_all"] = fulludp_oracle_all
    with (out_dir / "v20_candidate_zoo_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    md = [
        "# Haze4K v2.0 C0 Strong Candidate Zoo Oracle",
        "",
        f"Decision: `{decision}`",
        "",
        "This audit uses existing train-derived/internal validation evidence only. Locked test data was not touched.",
        "",
        "## A0 + FullUDP Endpoint Oracle",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ["count", "mean_dPSNR", "hard_bottom25_dPSNR", "easy_top25_dPSNR", "dSSIM", "positive_ratio", "nonnegative_ratio", "worst_per_600", "intervention_rate"]:
        md.append(f"| `{key}` | `{fulludp_oracle_all.get(key)}` |")
    md.extend(
        [
            "",
            "## Gate",
            "",
            *[f"- `{key}`: `{value}`" for key, value in gate.items()],
            "",
            "## Interpretation",
            "",
            "- The FullUDP endpoint remains unsafe as a global replacement; use `v20_candidate_zoo_single_expert_summary.csv` and `v20_candidate_zoo_failure_bins.csv` for the damage profile.",
            "- The A0-preserving endpoint oracle is the capacity signal for the next phase; any deployable route must learn abstention and preservation rather than transplanting FullUDP globally.",
            "- ConvIR-L/DehazeFormer/PromptIR were not available on `convir-4090` during C0a, so they are logged as future candidate slots rather than silently skipped.",
        ]
    )
    (out_dir / "v20_candidate_zoo_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"V20_CANDIDATE_ZOO_ORACLE_OK decision={decision} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
