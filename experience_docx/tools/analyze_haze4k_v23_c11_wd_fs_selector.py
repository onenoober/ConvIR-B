#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

SEVERE = -0.20
BOOT_N = 400
SEEDS = [3407, 3411, 2026]
OUTER_FOLDS = 5
INNER_FOLDS = 4

AGG_REQUIREMENTS = {
    "mean_improve_vs_wd": 0.20,
    "hard_floor_vs_wd": -0.10,
    "easy_improve_vs_wd": 0.25,
    "positive_ratio": 0.975,
    "severe_loss_per_600_max": 11.0,
}

GROUP_REQUIREMENTS = {
    "min_mean_dPSNR": 1.10,
    "min_hard_bottom25_dPSNR": 1.50,
    "min_positive_ratio": 0.90,
    "max_severe_loss_per_600": 40.0,
}

ACTION_SET = ["WD0375", "FS050", "A0"]

GROUP_SPECS: list[tuple[str, str | None]] = [
    ("split", None),
    ("A0_PSNR_q4", "A0_PSNR"),
    ("WDMamba_A0_diff_abs_q4", "wdmamba_residual_abs_mean"),
    ("WDMamba_A0_diff_signed_q4", "wdmamba_residual_signed_mean"),
    ("FSNet_A0_diff_abs_q4", "fsudp_residual_abs_mean"),
    ("FSNet_A0_diff_signed_q4", "fsudp_residual_signed_mean"),
    ("WD0375_FS050_disagreement_q4", "wd0375_fs050_disagreement_proxy"),
    ("haze_density_q4", "haze_density_mean"),
    ("transmission_q4", "transmission_mean"),
    ("airlight_q4", "airlight_proxy_p99"),
    ("depth_q4", "feature_depth_mean"),
    ("dark_channel_q4", "dark_channel_mean"),
    ("low_texture_q4", "input_low_texture_proxy"),
    ("edge_density_q4", "input_edge_density"),
    ("sky_highlight_proxy_q4", "sky_highlight_proxy"),
]

FULL_FEATURES = [
    "feature_input_mean",
    "feature_input_std",
    "feature_input_grad_mean",
    "feature_input_dark_mean",
    "feature_depth_mean",
    "feature_depth_std",
    "feature_depth_grad_mean",
    "feature_a0_mean",
    "feature_a0_std",
    "feature_a0_grad_mean",
    "feature_a0_saturation_high",
    "feature_a0_saturation_low",
    "feature_diff_signed_mean",
    "feature_diff_abs_mean",
    "feature_diff_abs_std",
    "feature_diff_abs_p50",
    "feature_diff_abs_p90",
    "feature_diff_abs_p95",
    "feature_diff_abs_max",
    "feature_diff_grad_mean",
    "feature_diff_to_a0_ratio",
    "feature_a0_udp_psnr",
    "input_edge_density",
    "input_low_texture_proxy",
    "dark_channel_mean",
    "sky_highlight_proxy",
    "airlight_proxy_p99",
    "transmission_mean",
    "transmission_std",
    "haze_density_mean",
    "haze_density_p90",
    "wdmamba_residual_signed_mean",
    "wdmamba_residual_abs_mean",
    "wdmamba_residual_grad_mean",
    "wdmamba_fulludp_mae",
    "wdmamba_residual_cosine",
    "fsudp_residual_signed_mean",
    "fsudp_residual_abs_mean",
    "fsudp_residual_grad_mean",
    "fsudp_fulludp_mae",
    "fsudp_residual_cosine",
    "wd0375_fs050_signed_gap_proxy",
    "wd0375_fs050_abs_gap_proxy",
    "wd0375_fs050_disagreement_proxy",
]

FEATURE_SETS = {
    "full": FULL_FEATURES,
    "residual_consensus": [
        "wdmamba_residual_signed_mean",
        "wdmamba_residual_abs_mean",
        "wdmamba_residual_grad_mean",
        "wdmamba_fulludp_mae",
        "wdmamba_residual_cosine",
        "fsudp_residual_signed_mean",
        "fsudp_residual_abs_mean",
        "fsudp_residual_grad_mean",
        "fsudp_fulludp_mae",
        "fsudp_residual_cosine",
        "wd0375_fs050_signed_gap_proxy",
        "wd0375_fs050_abs_gap_proxy",
        "wd0375_fs050_disagreement_proxy",
    ],
    "image_physics": [
        "feature_input_mean",
        "feature_input_std",
        "feature_input_grad_mean",
        "feature_input_dark_mean",
        "feature_depth_mean",
        "feature_depth_std",
        "feature_depth_grad_mean",
        "input_edge_density",
        "input_low_texture_proxy",
        "dark_channel_mean",
        "sky_highlight_proxy",
        "airlight_proxy_p99",
        "transmission_mean",
        "transmission_std",
        "haze_density_mean",
        "haze_density_p90",
    ],
    "a0_output": [
        "feature_a0_mean",
        "feature_a0_std",
        "feature_a0_grad_mean",
        "feature_a0_saturation_high",
        "feature_a0_saturation_low",
        "feature_diff_signed_mean",
        "feature_diff_abs_mean",
        "feature_diff_abs_std",
        "feature_diff_abs_p90",
        "feature_diff_abs_p95",
        "feature_diff_grad_mean",
        "feature_diff_to_a0_ratio",
        "feature_a0_udp_psnr",
    ],
}


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x == "":
            return default
        return float(x)
    except Exception:
        return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wilson_lcb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - rad) / denom)


def wilson_ucb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (center + rad) / denom)


def bootstrap_lcb(vals: list[float], seed: int, q: float = 0.05) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(BOOT_N, len(arr)))
    return float(np.quantile(arr[idx].mean(axis=1), q))


def summarize(rows: list[dict[str, Any]], dkey: str = "dPSNR", sskey: str = "dSSIM") -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "count": 0,
            "mean_dPSNR": 0.0,
            "hard_bottom25_dPSNR": 0.0,
            "easy_top25_dPSNR": 0.0,
            "dSSIM": 0.0,
            "positive_ratio": 0.0,
            "nonnegative_ratio": 0.0,
            "severe_loss_count": 0,
            "severe_loss_per_600": 0.0,
        }
    ds = [fnum(row[dkey]) for row in rows]
    ss = [fnum(row.get(sskey, 0.0)) for row in rows]
    a0 = [fnum(row["A0_PSNR"]) for row in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    k = max(1, n // 4)
    hard = [ds[i] for i in order[:k]]
    easy = [ds[i] for i in order[-k:]]
    pos = sum(d > 0 for d in ds)
    nonneg = sum(d >= 0 for d in ds)
    severe = sum(d <= SEVERE for d in ds)
    seed = 23011 + n + int(abs(sum(ds)) * 100)
    return {
        "count": n,
        "mean_dPSNR": statistics.mean(ds),
        "mean_bootstrap_LCB": bootstrap_lcb(ds, seed),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "hard_bootstrap_LCB": bootstrap_lcb(hard, seed + 11),
        "easy_top25_dPSNR": statistics.mean(easy),
        "dSSIM": statistics.mean(ss),
        "positive_ratio": pos / n,
        "positive_Wilson_LCB": wilson_lcb(pos, n),
        "nonnegative_ratio": nonneg / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
        "severe_Wilson_UCB": wilson_ucb(severe, n),
    }


def qbins(vals: list[float]) -> list[str]:
    arr = np.asarray(vals, dtype=np.float64)
    qs = np.quantile(arr, [0.25, 0.5, 0.75])
    out = []
    for val in arr:
        out.append("q1" if val <= qs[0] else "q2" if val <= qs[1] else "q3" if val <= qs[2] else "q4")
    return out


def alpha_key(alpha: float) -> str:
    return (("a%.6f" % alpha).rstrip("0").rstrip(".")).replace(".", "p")


def action_value(row: dict[str, Any], action: str) -> tuple[float, float]:
    if action == "A0":
        return 0.0, 0.0
    if action == "WD0375":
        return fnum(row["WD0375_dPSNR"]), fnum(row["WD0375_dSSIM"])
    if action == "FS050":
        return fnum(row["FS050_dPSNR"]), fnum(row["FS050_dSSIM"])
    raise KeyError(action)


def profile_rows(rows: list[dict[str, Any]], profile: str, dkey: str, sskey: str, action: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rec = dict(row)
        rec["profile"] = profile
        rec["selector"] = profile
        rec["selected_action"] = action or profile
        rec["dPSNR"] = fnum(row[dkey])
        rec["dSSIM"] = fnum(row[sskey])
        out.append(rec)
    return out


def fold_id(name: str, seed: int, k: int = OUTER_FOLDS) -> int:
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) % k


def inner_fold_id(name: str, seed: int, k: int = INNER_FOLDS) -> int:
    h = hashlib.sha256(f"inner:{seed}:{name}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) % k


def load_merged(c8_root: Path) -> list[dict[str, Any]]:
    files = {
        "wdmamba": c8_root / "v22_c8_1_wdmamba_per_image.csv",
        "fsudp": c8_root / "v22_c8_2_fsudp_per_image.csv",
        "mbtaylor": c8_root / "v22_c8_3_mbtaylor_per_image.csv",
    }
    expert_rows = {k: read_csv(v) for k, v in files.items()}
    names = None
    for rows in expert_rows.values():
        cur = [(r["split"], r["name"]) for r in rows]
        if names is None:
            names = cur
        elif cur != names:
            raise RuntimeError("C8 per-image row order mismatch")

    merged: list[dict[str, Any]] = []
    for i, wr in enumerate(expert_rows["wdmamba"]):
        row: dict[str, Any] = {}
        for k, v in wr.items():
            row[k] = fnum(v) if looks_numeric(v) else v
        for expert in ["wdmamba", "fsudp", "mbtaylor"]:
            er = expert_rows[expert][i]
            row[f"{expert}_endpoint_dPSNR"] = fnum(er["dPSNR_endpoint"])
            row[f"{expert}_endpoint_dSSIM"] = fnum(er["dSSIM_endpoint"])
            row[f"{expert}_residual_signed_mean"] = fnum(er.get("expert_residual_signed_mean"))
            row[f"{expert}_residual_abs_mean"] = fnum(er.get("expert_residual_abs_mean"))
            row[f"{expert}_residual_grad_mean"] = fnum(er.get("expert_residual_grad_mean"))
            row[f"{expert}_fulludp_mae"] = fnum(er.get("expert_fulludp_mae"))
            row[f"{expert}_residual_cosine"] = fnum(er.get("residual_cosine_fulludp_expert"))
            for alpha in [0.0625, 0.125, 0.25, 0.375, 0.50]:
                key = alpha_key(alpha)
                row[f"{expert}_{key}_dPSNR"] = fnum(er[f"expert_{key}_dPSNR"])
                row[f"{expert}_{key}_dSSIM"] = fnum(er[f"expert_{key}_dSSIM"])

        row["WD0375_dPSNR"] = row["wdmamba_a0p375_dPSNR"]
        row["WD0375_dSSIM"] = row["wdmamba_a0p375_dSSIM"]
        row["FS050_dPSNR"] = row["fsudp_a0p5_dPSNR"]
        row["FS050_dSSIM"] = row["fsudp_a0p5_dSSIM"]
        row["A0_dPSNR"] = 0.0
        row["A0_dSSIM"] = 0.0

        wd_signed = 0.375 * fnum(row["wdmamba_residual_signed_mean"])
        fs_signed = 0.5 * fnum(row["fsudp_residual_signed_mean"])
        wd_abs = 0.375 * fnum(row["wdmamba_residual_abs_mean"])
        fs_abs = 0.5 * fnum(row["fsudp_residual_abs_mean"])
        row["wd0375_fs050_signed_gap_proxy"] = wd_signed - fs_signed
        row["wd0375_fs050_abs_gap_proxy"] = wd_abs - fs_abs
        row["wd0375_fs050_disagreement_proxy"] = abs(wd_signed - fs_signed) + abs(wd_abs - fs_abs)

        candidates = []
        for action in ACTION_SET:
            d, s = action_value(row, action)
            candidates.append((d, s, action))
        best = max(candidates, key=lambda x: (x[0], x[1]))
        row["WD0375_or_FS050_or_A0_oracle_dPSNR"] = best[0]
        row["WD0375_or_FS050_or_A0_oracle_dSSIM"] = best[1]
        row["WD0375_or_FS050_or_A0_oracle_action"] = best[2]
        merged.append(row)
    return merged


def looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def group_bins(rows: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    out = []
    for group, key in GROUP_SPECS:
        if key is None:
            bins = [str(r["split"]) for r in rows]
        else:
            vals = [fnum(r.get(key), float("nan")) for r in rows]
            finite = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
            labels = qbins([v for _, v in finite]) if finite else []
            idx_to_bin = {i: b for (i, _), b in zip(finite, labels)}
            bins = [idx_to_bin.get(i, "nan") for i in range(len(rows))]
        for b in sorted(set(bins)):
            if b == "nan":
                continue
            sub = [r for r, bb in zip(rows, bins) if bb == b]
            if not sub:
                continue
            rec = {"profile": profile, "group": group, "bin": b, **summarize(sub)}
            actions: dict[str, int] = {}
            for r in sub:
                a = str(r.get("selected_action", profile))
                actions[a] = actions.get(a, 0) + 1
            for action in ACTION_SET:
                rec[f"usage_{action}"] = actions.get(action, 0) / len(sub)
            rec["group_gate_pass"] = group_bin_pass(rec)
            out.append(rec)
    return out


def group_bin_pass(rec: dict[str, Any]) -> bool:
    return (
        fnum(rec.get("mean_dPSNR")) >= GROUP_REQUIREMENTS["min_mean_dPSNR"]
        and fnum(rec.get("hard_bottom25_dPSNR")) >= GROUP_REQUIREMENTS["min_hard_bottom25_dPSNR"]
        and fnum(rec.get("positive_ratio")) >= GROUP_REQUIREMENTS["min_positive_ratio"]
        and fnum(rec.get("severe_loss_per_600")) <= GROUP_REQUIREMENTS["max_severe_loss_per_600"]
    )


def aggregate_pass(summary: dict[str, Any], wd_baseline: dict[str, Any]) -> bool:
    return (
        fnum(summary["mean_dPSNR"]) >= fnum(wd_baseline["mean_dPSNR"]) + AGG_REQUIREMENTS["mean_improve_vs_wd"]
        and fnum(summary["hard_bottom25_dPSNR"]) >= fnum(wd_baseline["hard_bottom25_dPSNR"]) + AGG_REQUIREMENTS["hard_floor_vs_wd"]
        and fnum(summary["easy_top25_dPSNR"]) >= fnum(wd_baseline["easy_top25_dPSNR"]) + AGG_REQUIREMENTS["easy_improve_vs_wd"]
        and fnum(summary["positive_ratio"]) >= AGG_REQUIREMENTS["positive_ratio"]
        and fnum(summary["severe_loss_per_600"]) <= AGG_REQUIREMENTS["severe_loss_per_600_max"]
        and fnum(summary["dSSIM"]) >= 0.0
    )


def selector_score(summary: dict[str, Any], wd_baseline: dict[str, Any]) -> float:
    mean_gain = fnum(summary["mean_dPSNR"]) - fnum(wd_baseline["mean_dPSNR"])
    easy_gain = fnum(summary["easy_top25_dPSNR"]) - fnum(wd_baseline["easy_top25_dPSNR"])
    hard_drop = max(0.0, fnum(wd_baseline["hard_bottom25_dPSNR"]) - fnum(summary["hard_bottom25_dPSNR"]) - 0.10)
    pos_gap = fnum(summary["positive_ratio"]) - fnum(wd_baseline["positive_ratio"])
    severe_extra = max(0.0, fnum(summary["severe_loss_per_600"]) - fnum(wd_baseline["severe_loss_per_600"]))
    return mean_gain + 0.50 * easy_gain + 2.0 * pos_gap - 1.5 * hard_drop - 0.02 * severe_extra


def c11_0(out_dir: Path, c8_root: Path, c9_root: Path, script_path: Path, rows: list[dict[str, Any]]) -> None:
    artifacts = {
        "route": "Haze4K v2.3 C11 WD0375-FS050 Two-Profile Selector",
        "locked_policy": "locked evidence is evidence-only and not read by this script",
        "allowed_sources": {
            "c8_wdmamba_per_image": str(c8_root / "v22_c8_1_wdmamba_per_image.csv"),
            "c8_fsudp_per_image": str(c8_root / "v22_c8_2_fsudp_per_image.csv"),
            "c8_mbtaylor_per_image": str(c8_root / "v22_c8_3_mbtaylor_per_image.csv"),
            "c9_fixed_summary": str(c9_root / "v22_c9a_fixed_profiles_summary.csv"),
        },
        "sha256": {
            "script": sha256(script_path),
            "c8_wdmamba_per_image": sha256(c8_root / "v22_c8_1_wdmamba_per_image.csv"),
            "c8_fsudp_per_image": sha256(c8_root / "v22_c8_2_fsudp_per_image.csv"),
            "c8_mbtaylor_per_image": sha256(c8_root / "v22_c8_3_mbtaylor_per_image.csv"),
            "c9_fixed_summary": sha256(c9_root / "v22_c9a_fixed_profiles_summary.csv"),
        },
        "row_count": len(rows),
        "splits": sorted(set(str(r["split"]) for r in rows)),
        "actions": ACTION_SET,
        "forbidden": [
            "locked per-image output",
            "locked-informed alpha/profile/action/expert tuning",
            "new expert acquisition",
            "distillation",
            "MB-Taylor action in C11 selector",
            "WD050 action in C11 selector",
            "patch-level alpha",
            "deep MoE or MLP router",
        ],
    }
    write_json(out_dir / "v23_c11_0_source_artifact_manifest.json", artifacts)
    (out_dir / "v23_c11_0_no_locked_status.txt").write_text(
        "locked_test_touched=false\n"
        "locked_per_image_read=false\n"
        "locked_informed_tuning=false\n"
        "distillation=false\n"
        "allowed_scope=train-derived val_regular + val_hard C8/C9 tables only\n",
        encoding="utf-8",
    )
    parity_rows = []
    for profile, dkey, skey in [
        ("WD0375", "WD0375_dPSNR", "WD0375_dSSIM"),
        ("FS050", "FS050_dPSNR", "FS050_dSSIM"),
        ("WD0375_or_FS050_or_A0_oracle", "WD0375_or_FS050_or_A0_oracle_dPSNR", "WD0375_or_FS050_or_A0_oracle_dSSIM"),
    ]:
        rec = summarize(profile_rows(rows, profile, dkey, skey))
        rec["profile"] = profile
        parity_rows.append(rec)
    write_csv(out_dir / "v23_c11_0_metric_parity_report.csv", parity_rows)
    (out_dir / "v23_c11_0_route_card.md").write_text(
        "# C11-0 Route Freeze\n\n"
        "Route: `Haze4K v2.3 C11 WD0375 + FS050 Two-Profile Selector Feasibility`.\n\n"
        "Allowed actions: `WD0375`, `FS050`, `A0`.\n\n"
        "Allowed data: C8/C9 train-derived `val_regular + val_hard` text tables only.\n\n"
        "Forbidden: locked per-image output, locked-informed tuning, new experts, `WD050`, MB-Taylor action, "
        "patch alpha, deep MoE, and distillation.\n",
        encoding="utf-8",
    )


def c11_a(rows: list[dict[str, Any]], out_dir: Path, wd_baseline: dict[str, Any]) -> dict[str, Any]:
    profiles = [
        ("A0", "A0_dPSNR", "A0_dSSIM"),
        ("WD0375", "WD0375_dPSNR", "WD0375_dSSIM"),
        ("FS050", "FS050_dPSNR", "FS050_dSSIM"),
        ("WD0375_or_FS050_or_A0_oracle", "WD0375_or_FS050_or_A0_oracle_dPSNR", "WD0375_or_FS050_or_A0_oracle_dSSIM"),
    ]
    summary_rows = []
    for profile, dkey, skey in profiles:
        rec = summarize(profile_rows(rows, profile, dkey, skey, action=profile if profile in ACTION_SET else None))
        rec["profile"] = profile
        rec["delta_mean_vs_WD0375"] = fnum(rec["mean_dPSNR"]) - fnum(wd_baseline["mean_dPSNR"])
        rec["delta_hard_vs_WD0375"] = fnum(rec["hard_bottom25_dPSNR"]) - fnum(wd_baseline["hard_bottom25_dPSNR"])
        rec["delta_easy_vs_WD0375"] = fnum(rec["easy_top25_dPSNR"]) - fnum(wd_baseline["easy_top25_dPSNR"])
        rec["delta_positive_vs_WD0375"] = fnum(rec["positive_ratio"]) - fnum(wd_baseline["positive_ratio"])
        rec["delta_severe_vs_WD0375"] = fnum(rec["severe_loss_per_600"]) - fnum(wd_baseline["severe_loss_per_600"])
        summary_rows.append(rec)
    write_csv(out_dir / "v23_c11a_wd_fs_oracle_summary.csv", summary_rows)

    categories = build_categories(rows)
    unique_rows = []
    for cat, mask in categories.items():
        sub = [r for r, keep in zip(rows, mask) if keep]
        if not sub:
            continue
        for action in ACTION_SET:
            cnt = sum(str(r["WD0375_or_FS050_or_A0_oracle_action"]) == action for r in sub)
            unique_rows.append({
                "category": cat,
                "action": action,
                "count": cnt,
                "rate": cnt / len(sub),
                "category_count": len(sub),
                "mean_action_gain_when_selected": statistics.mean([action_value(r, action)[0] for r in sub if str(r["WD0375_or_FS050_or_A0_oracle_action"]) == action]) if cnt else "",
            })
    write_csv(out_dir / "v23_c11a_wd_fs_unique_wins.csv", unique_rows)

    oracle_rows = []
    for r in rows:
        rec = dict(r)
        rec["profile"] = "WD0375_or_FS050_or_A0_oracle"
        rec["selector"] = "WD0375_or_FS050_or_A0_oracle"
        rec["selected_action"] = str(r["WD0375_or_FS050_or_A0_oracle_action"])
        rec["dPSNR"] = fnum(r["WD0375_or_FS050_or_A0_oracle_dPSNR"])
        rec["dSSIM"] = fnum(r["WD0375_or_FS050_or_A0_oracle_dSSIM"])
        oracle_rows.append(rec)
    comp = group_bins(oracle_rows, "WD0375_or_FS050_or_A0_oracle")
    write_csv(out_dir / "v23_c11a_wd_fs_group_composition.csv", comp)

    wd_neg = [r for r in rows if fnum(r["WD0375_dPSNR"]) <= 0.0]
    wd_severe = [r for r in rows if fnum(r["WD0375_dPSNR"]) <= SEVERE]
    fs_rescues = [r for r in wd_neg if fnum(r["FS050_dPSNR"]) > fnum(r["WD0375_dPSNR"]) and fnum(r["FS050_dPSNR"]) > 0]
    a0_rescues = [r for r in wd_neg if fnum(r["WD0375_or_FS050_or_A0_oracle_action"]) == "A0"]
    top_fs = sorted(fs_rescues, key=lambda r: fnum(r["FS050_dPSNR"]) - fnum(r["WD0375_dPSNR"]), reverse=True)[:20]
    lines = [
        "# C11-A WD0375 Selected-Negative Report",
        "",
        f"WD0375 negative count: `{len(wd_neg)}` / `{len(rows)}`.",
        f"WD0375 severe count: `{len(wd_severe)}` / `{len(rows)}`.",
        f"FS050 positive rescues among WD0375 negatives: `{len(fs_rescues)}`.",
        f"A0 oracle rescues among WD0375 negatives: `{len(a0_rescues)}`.",
        "",
        "## Top FS050 Rescues",
        "",
        "| split | name | WD0375 | FS050 | FS-WD | A0_PSNR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in top_fs:
        lines.append(
            f"| {r['split']} | {r['name']} | {fnum(r['WD0375_dPSNR']):.6f} | "
            f"{fnum(r['FS050_dPSNR']):.6f} | {fnum(r['FS050_dPSNR']) - fnum(r['WD0375_dPSNR']):.6f} | "
            f"{fnum(r['A0_PSNR']):.6f} |"
        )
    (out_dir / "v23_c11a_wd_fs_selected_negative_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    oracle_summary = next(r for r in summary_rows if r["profile"] == "WD0375_or_FS050_or_A0_oracle")
    fs_all = next((r for r in unique_rows if r["category"] == "all" and r["action"] == "FS050"), None)
    fs_unique_rate = fnum(fs_all["rate"]) if fs_all else 0.0
    decision = (
        "C11A_ORACLE_HEADROOM_PASS_RUN_C11B_SELECTOR"
        if aggregate_pass(oracle_summary, wd_baseline) and fs_unique_rate >= 0.05
        else "C11A_ORACLE_HEADROOM_FAIL_STOP_SELECTOR"
    )
    (out_dir / "v23_c11a_wd_fs_decision.md").write_text(
        "# C11-A Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Oracle mean/hard/easy/positive/severe: `{oracle_summary['mean_dPSNR']:.6f}` / "
        f"`{oracle_summary['hard_bottom25_dPSNR']:.6f}` / `{oracle_summary['easy_top25_dPSNR']:.6f}` / "
        f"`{oracle_summary['positive_ratio']:.6f}` / `{oracle_summary['severe_loss_per_600']:.2f}/600`.\n\n"
        f"FS050 unique oracle win rate: `{fs_unique_rate:.6f}`.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "oracle_summary": oracle_summary, "fs_unique_rate": fs_unique_rate}


def build_categories(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    a0_vals = [fnum(r["A0_PSNR"]) for r in rows]
    a0_q1 = float(np.quantile(a0_vals, 0.25))
    dis_vals = [fnum(r["wd0375_fs050_disagreement_proxy"]) for r in rows]
    dis_q4 = float(np.quantile(dis_vals, 0.75))
    return {
        "all": [True for _ in rows],
        "val_regular": [str(r["split"]) == "val_regular" for r in rows],
        "val_hard": [str(r["split"]) == "val_hard" for r in rows],
        "a0_hard_bottom25": [fnum(r["A0_PSNR"]) <= a0_q1 for r in rows],
        "wd0375_negative": [fnum(r["WD0375_dPSNR"]) <= 0.0 for r in rows],
        "wd0375_severe": [fnum(r["WD0375_dPSNR"]) <= SEVERE for r in rows],
        "wd0375_low_positive": [0.0 < fnum(r["WD0375_dPSNR"]) < 0.20 for r in rows],
        "wd_fs_disagreement_q4": [fnum(r["wd0375_fs050_disagreement_proxy"]) >= dis_q4 for r in rows],
    }


def make_action_rows(rows: list[dict[str, Any]], selector: str, actions: list[str], seed: int | None = None, fold: int | None = None, config: str = "") -> list[dict[str, Any]]:
    out = []
    for r, action in zip(rows, actions):
        d, s = action_value(r, action)
        rec = dict(r)
        rec.update({
            "selector": selector,
            "profile": selector,
            "selected_action": action,
            "dPSNR": d,
            "dSSIM": s,
            "seed": "" if seed is None else seed,
            "outer_fold": "" if fold is None else fold,
            "selected_config": config,
        })
        out.append(rec)
    return out


def fit_ridge(train_rows: list[dict[str, Any]], features: list[str], y: np.ndarray, lam: float) -> dict[str, np.ndarray]:
    x = np.asarray([[fnum(r.get(f)) for f in features] for r in train_rows], dtype=np.float64)
    mean = x.mean(axis=0) if len(x) else np.zeros(len(features), dtype=np.float64)
    std = x.std(axis=0) if len(x) else np.ones(len(features), dtype=np.float64)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    eye = np.eye(xb.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(xb.T @ xb + lam * eye, xb.T @ y)
    return {"mean": mean, "std": std, "coef": coef}


def predict_ridge(rows: list[dict[str, Any]], model: dict[str, np.ndarray], features: list[str]) -> np.ndarray:
    x = np.asarray([[fnum(r.get(f)) for f in features] for r in rows], dtype=np.float64)
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ model["coef"]


def train_pairwise(train_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    features = FEATURE_SETS[cfg["feature_set"]]
    y = np.asarray([fnum(r["FS050_dPSNR"]) - fnum(r["WD0375_dPSNR"]) for r in train_rows], dtype=np.float64)
    y = y - cfg["severe_penalty"] * np.asarray([
        (fnum(r["FS050_dPSNR"]) <= SEVERE) - (fnum(r["WD0375_dPSNR"]) <= SEVERE) for r in train_rows
    ], dtype=np.float64)
    return {"kind": "pairwise", "cfg": cfg, "features": features, "delta_model": fit_ridge(train_rows, features, y, cfg["lambda"])}


def predict_pairwise(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    pred = predict_ridge(rows, model["delta_model"], model["features"])
    threshold = model["cfg"]["threshold"]
    return ["FS050" if p >= threshold else "WD0375" for p in pred]


def train_utility(train_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    features = FEATURE_SETS[cfg["feature_set"]]
    models = {}
    for action in ACTION_SET:
        y = np.asarray([action_value(r, action)[0] for r in train_rows], dtype=np.float64)
        y = y - cfg["severe_penalty"] * np.asarray([action_value(r, action)[0] <= SEVERE for r in train_rows], dtype=np.float64)
        models[action] = fit_ridge(train_rows, features, y, cfg["lambda"])
    return {"kind": "utility", "cfg": cfg, "features": features, "models": models}


def predict_utility(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    scores = []
    for action in ACTION_SET:
        scores.append(predict_ridge(rows, model["models"][action], model["features"]))
    arr = np.vstack(scores).T
    return [ACTION_SET[int(i)] for i in np.argmax(arr, axis=1)]


def train_fixed(train_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "fixed", "cfg": cfg}


def predict_fixed(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    return [model["cfg"]["action"] for _ in rows]


def train_stump(train_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "stump", "cfg": cfg}


def predict_stump(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    cfg = model["cfg"]
    feature = cfg["feature"]
    threshold = cfg["threshold"]
    direction = cfg["direction"]
    out = []
    for r in rows:
        val = fnum(r.get(feature))
        cond = val >= threshold if direction == "ge" else val <= threshold
        out.append("FS050" if cond else "WD0375")
    return out


def train_model(train_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["kind"] == "fixed":
        return train_fixed(train_rows, cfg)
    if cfg["kind"] == "pairwise":
        return train_pairwise(train_rows, cfg)
    if cfg["kind"] == "utility":
        return train_utility(train_rows, cfg)
    if cfg["kind"] == "stump":
        return train_stump(train_rows, cfg)
    raise RuntimeError(cfg["kind"])


def predict_model(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    if model["kind"] == "fixed":
        return predict_fixed(model, rows)
    if model["kind"] == "pairwise":
        return predict_pairwise(model, rows)
    if model["kind"] == "utility":
        return predict_utility(model, rows)
    if model["kind"] == "stump":
        return predict_stump(model, rows)
    raise RuntimeError(model["kind"])


def config_name(cfg: dict[str, Any]) -> str:
    items = [f"{k}={cfg[k]}" for k in sorted(cfg)]
    return ";".join(items)


def candidate_configs(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {"kind": "fixed", "action": "WD0375"},
        {"kind": "fixed", "action": "FS050"},
    ]
    for feature_set in ["full", "residual_consensus", "image_physics", "a0_output"]:
        for lam in [0.5, 2.0, 8.0, 32.0]:
            for severe_penalty in [0.5, 2.0, 6.0]:
                configs.append({"kind": "utility", "feature_set": feature_set, "lambda": lam, "severe_penalty": severe_penalty})
                for threshold in [-0.15, 0.0, 0.10, 0.25, 0.50]:
                    configs.append({
                        "kind": "pairwise",
                        "feature_set": feature_set,
                        "lambda": lam,
                        "severe_penalty": severe_penalty,
                        "threshold": threshold,
                    })
    stump_features = [
        "wd0375_fs050_disagreement_proxy",
        "wdmamba_residual_abs_mean",
        "fsudp_residual_abs_mean",
        "feature_a0_saturation_high",
        "haze_density_mean",
        "transmission_mean",
        "dark_channel_mean",
        "input_low_texture_proxy",
        "input_edge_density",
    ]
    for feature in stump_features:
        vals = [fnum(r.get(feature)) for r in train_rows]
        if not vals:
            continue
        for q in [0.20, 0.35, 0.50, 0.65, 0.80]:
            threshold = float(np.quantile(vals, q))
            for direction in ["ge", "le"]:
                configs.append({"kind": "stump", "feature": feature, "threshold": threshold, "direction": direction})
    return configs


def eval_config_oof(train_rows: list[dict[str, Any]], cfg: dict[str, Any], seed: int, wd_train_baseline: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    pred_rows = []
    for fold in range(INNER_FOLDS):
        inner_train = [r for r in train_rows if inner_fold_id(str(r["name"]), seed) != fold]
        inner_held = [r for r in train_rows if inner_fold_id(str(r["name"]), seed) == fold]
        if not inner_held:
            continue
        model = train_model(inner_train, cfg)
        actions = predict_model(model, inner_held)
        pred_rows.extend(make_action_rows(inner_held, cfg["kind"], actions, seed=seed, fold=fold, config=config_name(cfg)))
    summary = summarize(pred_rows)
    return selector_score(summary, wd_train_baseline), summary


def select_config(train_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    wd_train = summarize(profile_rows(train_rows, "WD0375", "WD0375_dPSNR", "WD0375_dSSIM", action="WD0375"))
    scored = []
    for cfg in candidate_configs(train_rows):
        score, summary = eval_config_oof(train_rows, cfg, seed, wd_train)
        scored.append((score, summary["mean_dPSNR"], config_name(cfg), cfg))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][3]


def c11_b(rows: list[dict[str, Any]], out_dir: Path, wd_baseline: dict[str, Any]) -> dict[str, Any]:
    fixed_rows = []
    for action in ["WD0375", "FS050"]:
        for seed in SEEDS:
            fixed_rows.extend(make_action_rows(rows, f"fixed_{action.lower()}", [action] * len(rows), seed=seed, fold=-1, config=f"fixed:{action}"))

    nested_rows = []
    selected_configs = []
    for seed in SEEDS:
        for fold in range(OUTER_FOLDS):
            train = [r for r in rows if fold_id(str(r["name"]), seed) != fold]
            held = [r for r in rows if fold_id(str(r["name"]), seed) == fold]
            cfg = select_config(train, seed)
            model = train_model(train, cfg)
            actions = predict_model(model, held)
            selector = f"nested_{cfg['kind']}"
            cfg_name = config_name(cfg)
            selected_configs.append({"seed": seed, "outer_fold": fold, "selector": selector, "selected_config": cfg_name, "held_count": len(held)})
            nested_rows.extend(make_action_rows(held, selector, actions, seed=seed, fold=fold, config=cfg_name))

    all_rows = fixed_rows + nested_rows
    summary_rows = []
    for selector in sorted(set(r["selector"] for r in all_rows)):
        sub = [r for r in all_rows if r["selector"] == selector]
        rec = summarize(sub)
        rec["selector"] = selector
        rec["mean_improve_vs_WD0375"] = fnum(rec["mean_dPSNR"]) - fnum(wd_baseline["mean_dPSNR"])
        rec["hard_delta_vs_WD0375"] = fnum(rec["hard_bottom25_dPSNR"]) - fnum(wd_baseline["hard_bottom25_dPSNR"])
        rec["easy_improve_vs_WD0375"] = fnum(rec["easy_top25_dPSNR"]) - fnum(wd_baseline["easy_top25_dPSNR"])
        rec["positive_delta_vs_WD0375"] = fnum(rec["positive_ratio"]) - fnum(wd_baseline["positive_ratio"])
        rec["severe_delta_vs_WD0375"] = fnum(rec["severe_loss_per_600"]) - fnum(wd_baseline["severe_loss_per_600"])
        rec["aggregate_gate_pass"] = aggregate_pass(rec, wd_baseline)
        rec["selector_score"] = selector_score(rec, wd_baseline)
        summary_rows.append(rec)
    write_csv(out_dir / "v23_c11b_selector_oof_summary.csv", summary_rows)

    dist_rows = []
    for selector in sorted(set(r["selector"] for r in all_rows)):
        sub = [r for r in all_rows if r["selector"] == selector]
        for action in ACTION_SET:
            count = sum(r["selected_action"] == action for r in sub)
            dist_rows.append({"selector": selector, "action": action, "count": count, "usage": count / max(1, len(sub))})
    write_csv(out_dir / "v23_c11b_selector_action_distribution.csv", dist_rows)

    group_all = []
    for selector in sorted(set(r["selector"] for r in all_rows)):
        sub = [r for r in all_rows if r["selector"] == selector]
        group_all.extend(group_bins(sub, selector))
    write_csv(out_dir / "v23_c11b_selector_groupmin_bins.csv", group_all)

    best = choose_best_selector(summary_rows)
    best_rows = [r for r in all_rows if r["selector"] == best["selector"]]
    feature_ablation = feature_ablation_rows(rows, wd_baseline)
    write_csv(out_dir / "v23_c11b_selector_feature_ablation.csv", feature_ablation)
    removal = profile_removal_ablation(best_rows)
    write_csv(out_dir / "v23_c11b_selector_profile_removal_ablation.csv", removal)
    write_csv(out_dir / "v23_c11b_selector_selected_configs.csv", selected_configs)

    best_groups = [g for g in group_all if g["profile"] == best["selector"]]
    group_pass = all(bool(g["group_gate_pass"]) for g in best_groups)
    decision = (
        "C11B_SELECTOR_AGGREGATE_PASS_RUN_C11C_GROUPMIN"
        if bool(best["aggregate_gate_pass"])
        else "C11B_SELECTOR_AGGREGATE_FAIL_RUN_C11C_FOR_FAILURE_LOCALIZATION"
    )
    (out_dir / "v23_c11b_selector_decision.md").write_text(
        "# C11-B Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Best deployable selector: `{best['selector']}`.\n\n"
        f"OOF mean/hard/easy/positive/severe: `{best['mean_dPSNR']:.6f}` / "
        f"`{best['hard_bottom25_dPSNR']:.6f}` / `{best['easy_top25_dPSNR']:.6f}` / "
        f"`{best['positive_ratio']:.6f}` / `{best['severe_loss_per_600']:.2f}/600`.\n\n"
        f"Aggregate gate pass: `{best['aggregate_gate_pass']}`. Group gate pass at C11-B scan: `{group_pass}`.\n",
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "best_selector": best["selector"],
        "best_summary": best,
        "best_rows": best_rows,
        "all_rows": all_rows,
        "selected_configs": selected_configs,
    }


def choose_best_selector(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in summary_rows if not str(r["selector"]).startswith("fixed_")]
    if not candidates:
        candidates = summary_rows
    candidates.sort(key=lambda r: (bool(r["aggregate_gate_pass"]), fnum(r["selector_score"]), fnum(r["mean_dPSNR"])), reverse=True)
    return candidates[0]


def feature_ablation_rows(rows: list[dict[str, Any]], wd_baseline: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base_cfgs = [
        {"kind": "utility", "lambda": 8.0, "severe_penalty": 2.0},
        {"kind": "pairwise", "lambda": 8.0, "severe_penalty": 2.0, "threshold": 0.10},
    ]
    for feature_set in FEATURE_SETS:
        for cfg_base in base_cfgs:
            recs = []
            for fold in range(OUTER_FOLDS):
                train = [r for r in rows if fold_id(str(r["name"]), 3407) != fold]
                held = [r for r in rows if fold_id(str(r["name"]), 3407) == fold]
                cfg = dict(cfg_base)
                cfg["feature_set"] = feature_set
                model = train_model(train, cfg)
                actions = predict_model(model, held)
                recs.extend(make_action_rows(held, f"{cfg['kind']}_{feature_set}", actions, seed=3407, fold=fold, config=config_name(cfg)))
            summ = summarize(recs)
            summ["selector"] = cfg_base["kind"]
            summ["feature_set"] = feature_set
            summ["mean_improve_vs_WD0375"] = fnum(summ["mean_dPSNR"]) - fnum(wd_baseline["mean_dPSNR"])
            summ["easy_improve_vs_WD0375"] = fnum(summ["easy_top25_dPSNR"]) - fnum(wd_baseline["easy_top25_dPSNR"])
            summ["aggregate_gate_pass"] = aggregate_pass(summ, wd_baseline)
            out.append(summ)
    return out


def profile_removal_ablation(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [{"ablation": "full", **summarize(best_rows)}]
    for action in ACTION_SET:
        repl = []
        for r in best_rows:
            rec = dict(r)
            if rec["selected_action"] == action:
                fallback = "WD0375" if action != "WD0375" else "FS050"
                rec["selected_action"] = fallback
                rec["dPSNR"], rec["dSSIM"] = action_value(rec, fallback)
            repl.append(rec)
        s = summarize(repl)
        base = out[0]
        s["ablation"] = f"replace_{action}"
        s["mean_drop_vs_full"] = fnum(base["mean_dPSNR"]) - fnum(s["mean_dPSNR"])
        s["hard_drop_vs_full"] = fnum(base["hard_bottom25_dPSNR"]) - fnum(s["hard_bottom25_dPSNR"])
        s["easy_drop_vs_full"] = fnum(base["easy_top25_dPSNR"]) - fnum(s["easy_top25_dPSNR"])
        out.append(s)
    return out


def c11_c(best_rows: list[dict[str, Any]], out_dir: Path, selector: str, wd_baseline: dict[str, Any]) -> dict[str, Any]:
    bins = group_bins(best_rows, selector)
    write_csv(out_dir / "v23_c11c_shifted_bin_metrics.csv", bins)
    dim_rows = []
    for group in sorted(set(r["group"] for r in bins)):
        sub = [r for r in bins if r["group"] == group]
        rec = {
            "selector": selector,
            "group": group,
            "bins": len(sub),
            "min_mean_dPSNR": min(fnum(r["mean_dPSNR"]) for r in sub),
            "min_hard_bottom25_dPSNR": min(fnum(r["hard_bottom25_dPSNR"]) for r in sub),
            "min_positive_ratio": min(fnum(r["positive_ratio"]) for r in sub),
            "max_severe_loss_per_600": max(fnum(r["severe_loss_per_600"]) for r in sub),
            "group_pass": all(bool(r["group_gate_pass"]) for r in sub),
        }
        dim_rows.append(rec)
    write_csv(out_dir / "v23_c11c_shifted_dimension_summary.csv", dim_rows)
    bounds = []
    for r in bins:
        bounds.append({
            "selector": selector,
            "group": r["group"],
            "bin": r["bin"],
            "count": r["count"],
            "mean_bootstrap_LCB": r["mean_bootstrap_LCB"],
            "hard_bootstrap_LCB": r["hard_bootstrap_LCB"],
            "positive_Wilson_LCB": r["positive_Wilson_LCB"],
            "severe_Wilson_UCB": r["severe_Wilson_UCB"],
        })
    write_csv(out_dir / "v23_c11c_bootstrap_wilson_bounds.csv", bounds)

    summary = summarize(best_rows)
    agg_pass = aggregate_pass(summary, wd_baseline)
    group_pass = all(bool(r["group_gate_pass"]) for r in bins)
    decision = "C11C_GROUPMIN_PASS_AUTHORIZE_C11D_FORMAL" if agg_pass and group_pass else "C11C_GROUPMIN_OR_AGGREGATE_FAIL_NO_FORMAL_LOCKED"
    (out_dir / "v23_c11c_groupmin_decision.md").write_text(
        "# C11-C Group-Min Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Selector: `{selector}`.\n\n"
        f"Aggregate pass: `{agg_pass}`. Group-min pass: `{group_pass}`.\n\n"
        f"Overall mean/hard/easy/positive/severe: `{summary['mean_dPSNR']:.6f}` / "
        f"`{summary['hard_bottom25_dPSNR']:.6f}` / `{summary['easy_top25_dPSNR']:.6f}` / "
        f"`{summary['positive_ratio']:.6f}` / `{summary['severe_loss_per_600']:.2f}/600`.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "summary": summary, "aggregate_pass": agg_pass, "group_pass": group_pass, "dimension_summary": dim_rows}


def c11_d(best_rows: list[dict[str, Any]], out_dir: Path, selector: str, wd_baseline: dict[str, Any], c11c_pass: bool) -> dict[str, Any]:
    if not c11c_pass:
        for name in [
            "v23_c11d_formal_5x3_summary.csv",
            "v23_c11d_formal_groupmin_summary.csv",
            "v23_c11d_formal_selector_stability.csv",
        ]:
            write_csv(out_dir / name, [{"status": "SKIPPED_C11C_NOT_PASS"}])
        decision = "C11D_FORMAL_SKIPPED_C11C_NOT_PASS_LOCKED_BLOCKED"
        (out_dir / "v23_c11d_formal_decision.md").write_text(
            "# C11-D Formal Decision\n\n"
            f"Decision: `{decision}`\n\nC11-C did not pass; formal promotion replay is blocked.\n",
            encoding="utf-8",
        )
        return {"decision": decision}

    summary_rows = []
    for seed in SEEDS:
        sub = [r for r in best_rows if int(r["seed"]) == seed]
        rec = summarize(sub)
        rec["selector"] = selector
        rec["seed"] = seed
        rec["aggregate_gate_pass"] = aggregate_pass(rec, wd_baseline)
        summary_rows.append(rec)
    overall = summarize(best_rows)
    overall["selector"] = selector
    overall["seed"] = "overall"
    overall["aggregate_gate_pass"] = aggregate_pass(overall, wd_baseline)
    summary_rows.append(overall)
    write_csv(out_dir / "v23_c11d_formal_5x3_summary.csv", summary_rows)

    group_rows = []
    for seed in SEEDS:
        sub = [r for r in best_rows if int(r["seed"]) == seed]
        bins = group_bins(sub, selector)
        for group in sorted(set(r["group"] for r in bins)):
            gsub = [r for r in bins if r["group"] == group]
            group_rows.append({
                "selector": selector,
                "seed": seed,
                "group": group,
                "bins": len(gsub),
                "min_mean_dPSNR": min(fnum(r["mean_dPSNR"]) for r in gsub),
                "min_hard_bottom25_dPSNR": min(fnum(r["hard_bottom25_dPSNR"]) for r in gsub),
                "min_positive_ratio": min(fnum(r["positive_ratio"]) for r in gsub),
                "max_severe_loss_per_600": max(fnum(r["severe_loss_per_600"]) for r in gsub),
                "group_pass": all(bool(r["group_gate_pass"]) for r in gsub),
            })
    write_csv(out_dir / "v23_c11d_formal_groupmin_summary.csv", group_rows)

    stability = []
    for seed in SEEDS:
        for fold in range(OUTER_FOLDS):
            sub = [r for r in best_rows if int(r["seed"]) == seed and int(r["outer_fold"]) == fold]
            if not sub:
                continue
            rec = {"selector": selector, "seed": seed, "outer_fold": fold, "count": len(sub), "selected_config": sub[0].get("selected_config", "")}
            for action in ACTION_SET:
                rec[f"usage_{action}"] = sum(r["selected_action"] == action for r in sub) / len(sub)
            stability.append(rec)
    write_csv(out_dir / "v23_c11d_formal_selector_stability.csv", stability)

    all_seed_agg = all(bool(r["aggregate_gate_pass"]) for r in summary_rows if r["seed"] != "overall")
    all_seed_group = all(bool(r["group_pass"]) for r in group_rows)
    decision = "C11D_FORMAL_5X3_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW" if all_seed_agg and all_seed_group else "C11D_FORMAL_5X3_FAIL_LOCKED_BLOCKED"
    (out_dir / "v23_c11d_formal_decision.md").write_text(
        "# C11-D Formal Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"All seed aggregate pass: `{all_seed_agg}`. All seed group-min pass: `{all_seed_group}`.\n\n"
        "No locked command is run by C11-D; a separate review would be required even after pass.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "summary": overall, "all_seed_aggregate_pass": all_seed_agg, "all_seed_group_pass": all_seed_group}


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Haze4K v2.3 C11 WD0375-FS050 Selector Evidence",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This route uses only train-derived C8/C9 per-image tables. Locked Haze4K output is not read, and no distillation or new expert acquisition is performed.",
        "",
        "## Key Metrics",
        "",
    ]
    best = summary.get("c11b", {}).get("best_summary", {})
    if best:
        lines.extend([
            f"- Best deployable selector: `{summary['c11b']['best_selector']}`",
            f"- OOF mean/hard/easy: `{best['mean_dPSNR']:.6f}` / `{best['hard_bottom25_dPSNR']:.6f}` / `{best['easy_top25_dPSNR']:.6f}`",
            f"- OOF positive/severe: `{best['positive_ratio']:.6f}` / `{best['severe_loss_per_600']:.2f}/600`",
        ])
    lines.extend([
        "",
        "## Output Map",
        "",
        "- C11-0: provenance, no-locked, source manifest, metric parity.",
        "- C11-A: WD0375/FS050/A0 oracle decomposition and selected-negative report.",
        "- C11-B: nested OOF low-capacity selector screen and ablations.",
        "- C11-C: group-min shifted validation.",
        "- C11-D: formal replay only if C11-C passes; otherwise explicit skipped decision.",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--c8-root", type=Path, default=Path("experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"))
    ap.add_argument("--c9-root", type=Path, default=Path("experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615"))
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    c8_root = args.c8_root if args.c8_root.is_absolute() else repo / args.c8_root
    c9_root = args.c9_root if args.c9_root.is_absolute() else repo / args.c9_root
    script_path = Path(__file__).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_merged(c8_root)
    wd_rows = profile_rows(rows, "WD0375", "WD0375_dPSNR", "WD0375_dSSIM", action="WD0375")
    wd_baseline = summarize(wd_rows)

    c11_0(out_dir, c8_root, c9_root, script_path, rows)
    c11a = c11_a(rows, out_dir, wd_baseline)
    c11b = c11_b(rows, out_dir, wd_baseline)
    c11c = c11_c(c11b["best_rows"], out_dir, c11b["best_selector"], wd_baseline)
    c11d = c11_d(
        c11b["best_rows"],
        out_dir,
        c11b["best_selector"],
        wd_baseline,
        c11c_pass=bool(c11c["aggregate_pass"]) and bool(c11c["group_pass"]),
    )

    final_decision = (
        "C11_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW"
        if str(c11d["decision"]).startswith("C11D_FORMAL_5X3_PASS")
        else "C11_FAIL_SELECTOR_NOT_READY_LOCKED_BLOCKED"
    )
    summary = {
        "route": "Haze4K v2.3 C11 WD0375-FS050 Two-Profile Selector Feasibility",
        "decision": final_decision,
        "wd0375_baseline": wd_baseline,
        "c11a": c11a,
        "c11b": {k: v for k, v in c11b.items() if k not in {"best_rows", "all_rows"}},
        "c11c": c11c,
        "c11d": c11d,
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "distillation": False,
        "actions": ACTION_SET,
    }
    write_json(out_dir / "v23_c11_summary.json", summary)
    (out_dir / "v23_c11_decision.md").write_text(
        "# C11 Decision\n\n"
        f"Decision: `{final_decision}`\n\n"
        f"C11-A: `{c11a['decision']}`.\n\n"
        f"C11-B: `{c11b['decision']}` with best selector `{c11b['best_selector']}`.\n\n"
        f"C11-C: `{c11c['decision']}`.\n\n"
        f"C11-D: `{c11d['decision']}`.\n\n"
        "Locked Haze4K remains blocked for this route unless a separate review authorizes a new sealed one-shot.\n",
        encoding="utf-8",
    )
    write_readme(out_dir, summary)
    print("C11_ANALYSIS_OK")
    print(json.dumps({"decision": final_decision, "best_selector": c11b["best_selector"]}, sort_keys=True))


if __name__ == "__main__":
    main()
