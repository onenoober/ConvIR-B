#!/usr/bin/env python3
"""C9 profile-level shifted strong validation for C7c risk profiles.

This is a fast train-derived stress validation: for each stress dimension/bin, it
chooses the C7c risk profile on all other bins using true C7c OOF per-image
results, then evaluates that selected profile on the held-out bin. It does not
retrain patch policies and does not touch locked data; C10 formal handles the
more expensive seeded replay.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_haze4k_v20_c2b_multirule_router import fnum, write_csv
from audit_haze4k_v21_c7b_local_alpha_prototype import C7B_STRONG_GATE, gate_pass, score

BIN_MEAN_FLOOR = -0.02
BIN_POSITIVE_FLOOR = 0.60
BIN_SEVERE_CAP = 96.0


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_name_params(name: str) -> tuple[float, float]:
    parts = Path(name).stem.split("_")
    return (float(parts[1]) if len(parts) > 1 else 1.0, float(parts[2]) if len(parts) > 2 else 1.0)


def rank_bins(values: list[float], bins: int = 4) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [""] * len(values)
    for rank, idx in enumerate(order):
        labels[idx] = f"q{min(bins - 1, int(rank * bins / max(1, len(values)))) + 1}"
    return labels


def build_group_labels(image_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    feat = {str(row["name"]): row for row in feature_rows}
    air, beta = [], []
    for row in image_rows:
        a, b = parse_name_params(str(row["name"]))
        air.append(a); beta.append(b)
    def vals(field: str) -> list[float]:
        return [fnum(feat.get(str(row["name"]), {}).get(field, 0.0)) for row in image_rows]
    return {
        "split": [str(row["split"]) for row in image_rows],
        "airlight_q4": rank_bins(air),
        "beta_haze_q4": rank_bins(beta),
        "depth_mean_q4": rank_bins(vals("depth_mean")),
        "low_texture_input_grad_q4": rank_bins(vals("input_grad_mean")),
        "input_dark_q4": rank_bins(vals("input_dark_mean")),
        "diff_abs_q4": rank_bins(vals("diff_abs_mean")),
        "a0_psnr_q4": rank_bins([fnum(row["A0_PSNR"]) for row in image_rows]),
        "diff_signed_q4": rank_bins(vals("diff_signed_mean")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mean_dPSNR": 0.0, "hard_bottom25_dPSNR": 0.0, "easy_top25_dPSNR": 0.0, "dSSIM": 0.0, "positive_ratio": 0.0, "severe_loss_per_600": 0.0}
    d = np.array([fnum(r["dPSNR"]) for r in rows], dtype=np.float64)
    s = np.array([fnum(r["dSSIM"]) for r in rows], dtype=np.float64)
    a0 = np.array([fnum(r["A0_PSNR"]) for r in rows], dtype=np.float64)
    order = np.argsort(a0); bucket = max(1, len(rows)//4)
    severe = int(np.sum(d <= -0.20)); strong = int(np.sum(d <= -0.05))
    selected = np.array([fnum(r.get("patch_action_fraction_a0")) < 1.0 for r in rows], dtype=bool)
    selected_d = d[selected]
    rec = {
        "count": len(rows),
        "selected_count": int(selected.sum()),
        "coverage": float(np.mean(selected)),
        "mean_dPSNR": float(np.mean(d)),
        "hard_bottom25_dPSNR": float(np.mean(d[order[:bucket]])),
        "easy_top25_dPSNR": float(np.mean(d[order[-bucket:]])),
        "dSSIM": float(np.mean(s)),
        "positive_ratio": float(np.mean(d > 0.0)),
        "nonnegative_ratio": float(np.mean(d >= 0.0)),
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / len(rows) * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / len(rows) * 600.0,
        "selected_precision": float(np.mean(selected_d > 0.0)) if selected_d.size else 0.0,
        "selected_nonnegative_ratio": float(np.mean(selected_d >= 0.0)) if selected_d.size else 1.0,
        "selected_severe_count": int(np.sum(selected_d <= -0.20)) if selected_d.size else 0,
    }
    rec["strong_gate_pass"] = gate_pass(rec, C7B_STRONG_GATE)
    rec["score"] = score(rec)
    return rec


def choose_profile(train_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    profiles = sorted({str(r["profile"]) for r in train_rows})
    summaries = []
    for profile in profiles:
        rec = {"profile": profile, **summarize([r for r in train_rows if str(r["profile"]) == profile])}
        summaries.append(rec)
    summaries.sort(key=lambda r: (bool(r["strong_gate_pass"]), fnum(r["score"])), reverse=True)
    return str(summaries[0]["profile"]), summaries[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_per_image", type=Path, required=True)
    parser.add_argument("--image_rows", type=Path, required=True)
    parser.add_argument("--image_feature_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_rows = read_csv(args.profile_per_image)
    image_rows = read_csv(args.image_rows)
    feature_rows = read_csv(args.image_feature_rows)
    group_labels = build_group_labels(image_rows, feature_rows)
    image_order = [str(r["name"]) for r in image_rows]
    labels_by_dim = {dim: {name: labels[i] for i, name in enumerate(image_order)} for dim, labels in group_labels.items()}

    selected_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for dim, name_to_label in labels_by_dim.items():
        dim_selected: list[dict[str, Any]] = []
        for heldout in sorted(set(name_to_label.values())):
            train = [r for r in profile_rows if name_to_label[str(r["name"])] != heldout]
            chosen_profile, train_summary = choose_profile(train)
            held = [r for r in profile_rows if name_to_label[str(r["name"])] == heldout and str(r["profile"]) == chosen_profile]
            for row in held:
                clone = dict(row); clone["dimension"] = dim; clone["heldout_bin"] = heldout; clone["chosen_profile"] = chosen_profile
                selected_rows.append(clone); dim_selected.append(clone)
            brec = {"dimension": dim, "heldout_bin": heldout, "chosen_profile": chosen_profile, **summarize(held)}
            brec["bin_safety_pass"] = fnum(brec["mean_dPSNR"]) >= BIN_MEAN_FLOOR and fnum(brec["positive_ratio"]) >= BIN_POSITIVE_FLOOR and fnum(brec["severe_loss_per_600"]) <= BIN_SEVERE_CAP
            bin_rows.append(brec)
            selection_rows.append({"dimension": dim, "heldout_bin": heldout, "chosen_profile": chosen_profile, **{f"train_{k}": v for k, v in train_summary.items() if k != "profile"}})
        drec = {"dimension": dim, **summarize(dim_selected)}
        bins = [r for r in bin_rows if r["dimension"] == dim]
        drec["min_bin_mean_dPSNR"] = min(fnum(r["mean_dPSNR"]) for r in bins)
        drec["min_bin_positive_ratio"] = min(fnum(r["positive_ratio"]) for r in bins)
        drec["max_bin_severe_loss_per_600"] = max(fnum(r["severe_loss_per_600"]) for r in bins)
        drec["bin_safety_pass_count"] = sum(bool(r["bin_safety_pass"]) for r in bins)
        drec["dimension_shift_strong_pass"] = bool(drec["strong_gate_pass"]) and drec["bin_safety_pass_count"] == len(bins)
        dim_rows.append(drec)
    write_csv(args.out_dir / "v21_c9_shifted_selected_per_image.csv", selected_rows, sorted({k for r in selected_rows for k in r}))
    write_csv(args.out_dir / "v21_c9_shifted_selection_rows.csv", selection_rows, sorted({k for r in selection_rows for k in r}))
    write_csv(args.out_dir / "v21_c9_shifted_bin_metrics.csv", bin_rows, sorted({k for r in bin_rows for k in r}))
    write_csv(args.out_dir / "v21_c9_shifted_dimension_summary.csv", dim_rows, sorted({k for r in dim_rows for k in r}))
    all_pass = all(bool(r["dimension_shift_strong_pass"]) for r in dim_rows)
    decision = "C9_SHIFTED_STRONG_PASS_START_C10_FORMAL_5X3" if all_pass else "C9_SHIFTED_STRONG_FAIL_REASSESS_LOCAL_ALPHA_OR_C8"
    payload = {"route": "Haze4K-v2.1 SEG-Mix", "phase": "C9 Profile-Level Shifted Strong Validation", "locked_test_touched": False, "strong_gate": C7B_STRONG_GATE, "dimension_rows": dim_rows, "bin_rows": bin_rows, "decision": decision}
    (args.out_dir / "v21_c9_shifted_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Haze4K v2.1 C9 Profile-Level Shifted Strong Validation", "", f"Decision: `{decision}`", "", "| Dimension | Pass | Mean | Hard | Easy | dSSIM | Positive | Severe/600 | Min Bin Mean | Min Bin Pos | Max Bin Severe |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in dim_rows:
        lines.append(f"| `{r['dimension']}` | `{r['dimension_shift_strong_pass']}` | `{fnum(r['mean_dPSNR']):.6f}` | `{fnum(r['hard_bottom25_dPSNR']):.6f}` | `{fnum(r['easy_top25_dPSNR']):.6f}` | `{fnum(r['dSSIM']):.8f}` | `{fnum(r['positive_ratio']):.6f}` | `{fnum(r['severe_loss_per_600']):.1f}` | `{fnum(r['min_bin_mean_dPSNR']):.6f}` | `{fnum(r['min_bin_positive_ratio']):.6f}` | `{fnum(r['max_bin_severe_loss_per_600']):.1f}` |")
    lines += ["", "C10 formal 5x3 is authorized only if every dimension passes. Locked test remains blocked."]
    (args.out_dir / "v21_c9_shifted_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V21_C9_SHIFTED_OK decision={decision} dimensions={len(dim_rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
