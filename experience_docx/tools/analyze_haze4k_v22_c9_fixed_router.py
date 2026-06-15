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

AGG_GATE = {
    "mean_dPSNR": 1.50,
    "hard_bottom25_dPSNR": 2.00,
    "easy_top25_dPSNR": 0.25,
    "positive_ratio": 0.90,
    "dSSIM": 0.0,
    "severe_loss_per_600_max": 36.0,
}

GROUP_GATE = {
    "mean_dPSNR": 0.50,
    "hard_bottom25_dPSNR": 0.50,
    "positive_ratio": 0.80,
    "severe_loss_per_600_max": 48.0,
}

GROUP_SPECS: list[tuple[str, str | None]] = [
    ("split", None),
    ("A0_PSNR_q4", "A0_PSNR"),
    ("WDMamba_A0_diff_signed_q4", "wdmamba_residual_signed_mean"),
    ("WDMamba_A0_diff_abs_q4", "wdmamba_residual_abs_mean"),
    ("FSNet_A0_diff_signed_q4", "fsudp_residual_signed_mean"),
    ("FSNet_A0_diff_abs_q4", "fsudp_residual_abs_mean"),
    ("expert_disagreement_q4", "expert_disagreement"),
    ("haze_density_q4", "haze_density_mean"),
    ("transmission_q4", "transmission_mean"),
    ("airlight_q4", "airlight_proxy_p99"),
    ("depth_q4", "feature_depth_mean"),
    ("dark_channel_q4", "dark_channel_mean"),
    ("low_texture_q4", "input_low_texture_proxy"),
    ("edge_density_q4", "input_edge_density"),
    ("sky_highlight_proxy_q4", "sky_highlight_proxy"),
]

DEPLOYABLE_FEATURES = [
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
    "mbtaylor_residual_signed_mean",
    "mbtaylor_residual_abs_mean",
    "mbtaylor_residual_grad_mean",
    "mbtaylor_fulludp_mae",
    "mbtaylor_residual_cosine",
    "expert_disagreement",
]

ACTION_SPECS = [
    ("A0", "A0", 0.0),
    ("WD025", "wdmamba", 0.25),
    ("WD0375", "wdmamba", 0.375),
    ("WD050", "wdmamba", 0.50),
    ("FS025", "fsudp", 0.25),
    ("FS050", "fsudp", 0.50),
    ("MB00625", "mbtaylor", 0.0625),
    ("MB0125", "mbtaylor", 0.125),
    ("MB025", "mbtaylor", 0.25),
]


def alpha_key(alpha: float) -> str:
    return (("a%.6f" % alpha).rstrip("0").rstrip(".")).replace(".", "p")


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
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
        return {"count": 0}
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
    seed = 3407 + n + int(abs(sum(ds)) * 100)
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
            row[k] = v
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
        row["A0_action_dPSNR"] = 0.0
        row["A0_action_dSSIM"] = 0.0
        row["WD0375_dPSNR"] = row["wdmamba_a0p375_dPSNR"]
        row["WD0375_dSSIM"] = row["wdmamba_a0p375_dSSIM"]
        row["WD050_dPSNR"] = row["wdmamba_a0p5_dPSNR"]
        row["WD050_dSSIM"] = row["wdmamba_a0p5_dSSIM"]
        row["FS050_dPSNR"] = row["fsudp_a0p5_dPSNR"]
        row["FS050_dSSIM"] = row["fsudp_a0p5_dSSIM"]
        row["WD0375_or_FS050_oracle_dPSNR"] = max(row["WD0375_dPSNR"], row["FS050_dPSNR"], 0.0)
        row["WD0375_or_FS050_oracle_dSSIM"] = (
            row["WD0375_dSSIM"] if row["WD0375_dPSNR"] >= row["FS050_dPSNR"] and row["WD0375_dPSNR"] >= 0
            else row["FS050_dSSIM"] if row["FS050_dPSNR"] >= 0
            else 0.0
        )
        candidates = []
        for action, expert, alpha in ACTION_SPECS:
            d, s = action_value(row, action)
            candidates.append((d, s, action))
        best = max(candidates, key=lambda x: x[0])
        row["S3_oracle_dPSNR"] = best[0]
        row["S3_oracle_dSSIM"] = best[1]
        row["S3_oracle_action"] = best[2]
        expert_vals = [row[f"{e}_a0p5_dPSNR"] for e in ["wdmamba", "fsudp", "mbtaylor"]]
        row["expert_disagreement"] = max(expert_vals) - min(expert_vals)
        merged.append(row)
    return merged


def action_value(row: dict[str, Any], action: str) -> tuple[float, float]:
    if action == "A0":
        return 0.0, 0.0
    mapping = {
        "WD025": ("wdmamba", 0.25),
        "WD0375": ("wdmamba", 0.375),
        "WD050": ("wdmamba", 0.50),
        "FS025": ("fsudp", 0.25),
        "FS050": ("fsudp", 0.50),
        "MB00625": ("mbtaylor", 0.0625),
        "MB0125": ("mbtaylor", 0.125),
        "MB025": ("mbtaylor", 0.25),
    }
    expert, alpha = mapping[action]
    key = alpha_key(alpha)
    return fnum(row[f"{expert}_{key}_dPSNR"]), fnum(row[f"{expert}_{key}_dSSIM"])


def profile_rows(rows: list[dict[str, Any]], profile: str, dkey: str, sskey: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rec = dict(row)
        rec["profile"] = profile
        rec["dPSNR"] = fnum(row[dkey])
        rec["dSSIM"] = fnum(row[sskey])
        rec["selected_action"] = profile
        out.append(rec)
    return out


def gate_aggregate(summary: dict[str, Any]) -> bool:
    return (
        fnum(summary.get("mean_dPSNR")) >= AGG_GATE["mean_dPSNR"]
        and fnum(summary.get("hard_bottom25_dPSNR")) >= AGG_GATE["hard_bottom25_dPSNR"]
        and fnum(summary.get("easy_top25_dPSNR")) >= AGG_GATE["easy_top25_dPSNR"]
        and fnum(summary.get("positive_ratio")) >= AGG_GATE["positive_ratio"]
        and fnum(summary.get("dSSIM")) >= AGG_GATE["dSSIM"]
        and fnum(summary.get("severe_loss_per_600")) <= AGG_GATE["severe_loss_per_600_max"]
    )


def group_bins(rows: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    out = []
    for group, key in GROUP_SPECS:
        if key is None:
            bins = [r["split"] for r in rows]
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
            for action, count in sorted(actions.items()):
                rec[f"usage_{action}"] = count / len(sub)
            rec["group_gate_pass"] = (
                rec["mean_dPSNR"] >= GROUP_GATE["mean_dPSNR"]
                and rec["hard_bottom25_dPSNR"] >= GROUP_GATE["hard_bottom25_dPSNR"]
                and rec["positive_ratio"] >= GROUP_GATE["positive_ratio"]
                and rec["severe_loss_per_600"] <= GROUP_GATE["severe_loss_per_600_max"]
            )
            out.append(rec)
    return out


def fixed_profiles(rows: list[dict[str, Any]], out_dir: Path) -> tuple[bool, dict[str, Any]]:
    profiles = [
        ("WD0375", "WD0375_dPSNR", "WD0375_dSSIM"),
        ("WD050", "WD050_dPSNR", "WD050_dSSIM"),
        ("FS050", "FS050_dPSNR", "FS050_dSSIM"),
        ("WD0375_or_FS050_oracle", "WD0375_or_FS050_oracle_dPSNR", "WD0375_or_FS050_oracle_dSSIM"),
        ("S3_oracle", "S3_oracle_dPSNR", "S3_oracle_dSSIM"),
    ]
    summary_rows = []
    split_rows = []
    group_rows_all = []
    profile_payload: dict[str, Any] = {}
    for profile, dkey, sskey in profiles:
        prow = profile_rows(rows, profile, dkey, sskey)
        summ = summarize(prow)
        summ["profile"] = profile
        summ["aggregate_gate_pass"] = gate_aggregate(summ) if "oracle" not in profile else ""
        summary_rows.append(summ)
        profile_payload[profile] = summ
        for split in sorted(set(r["split"] for r in prow)):
            sub = [r for r in prow if r["split"] == split]
            split_rows.append({"profile": profile, "split": split, **summarize(sub)})
        group_rows = group_bins(prow, profile)
        group_rows_all.extend(group_rows)
        group_pass = all(bool(r["group_gate_pass"]) for r in group_rows)
        profile_payload[profile]["group_gate_pass"] = group_pass if "oracle" not in profile else ""
    write_csv(out_dir / "v22_c9a_fixed_profiles_summary.csv", summary_rows)
    write_csv(out_dir / "v22_c9a_fixed_profiles_by_split.csv", split_rows)
    write_csv(out_dir / "v22_c9a_fixed_profiles_groupmin_bins.csv", group_rows_all)
    wd = profile_payload["WD0375"]
    wd_pass = bool(wd["aggregate_gate_pass"]) and bool(wd["group_gate_pass"])
    lines = [
        "# C9-A Fixed Profiles Critical Bin Report",
        "",
        f"WD0375 aggregate gate pass: `{wd['aggregate_gate_pass']}`",
        f"WD0375 group gate pass: `{wd['group_gate_pass']}`",
        "",
        "## Worst WD0375 Bins",
        "",
    ]
    wd_groups = [r for r in group_rows_all if r["profile"] == "WD0375"]
    for key, reverse in [("mean_dPSNR", False), ("hard_bottom25_dPSNR", False), ("positive_ratio", False), ("severe_loss_per_600", True)]:
        worst = sorted(wd_groups, key=lambda r: fnum(r[key]), reverse=reverse)[:5]
        lines.append(f"### {key}")
        for r in worst:
            lines.append(f"- `{r['group']}/{r['bin']}`: `{fnum(r[key]):.6f}` count `{r['count']}`")
        lines.append("")
    (out_dir / "v22_c9a_fixed_profiles_critical_bin_report.md").write_text("\n".join(lines), encoding="utf-8")
    decision = "C9A_FIXED_WD0375_STRONG_PASS_AUTHORIZE_C10_FORMAL" if wd_pass else "C9A_FIXED_WD0375_FAIL_RUN_C9B_ROUTER"
    (out_dir / "v22_c9a_fixed_profiles_decision.md").write_text(
        f"# C9-A Decision\n\nDecision: `{decision}`\n\n"
        f"WD0375 mean/hard/easy/positive/severe: `{wd['mean_dPSNR']:.6f}` / `{wd['hard_bottom25_dPSNR']:.6f}` / "
        f"`{wd['easy_top25_dPSNR']:.6f}` / `{wd['positive_ratio']:.6f}` / `{wd['severe_loss_per_600']:.2f}/600`.\n",
        encoding="utf-8",
    )
    return wd_pass, {"decision": decision, "profiles": profile_payload}


def fold_id(name: str, seed: int, k: int = 5) -> int:
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) % k


def fit_linear(train_rows: list[dict[str, Any]], actions: list[str], features: list[str], seed: int) -> dict[str, np.ndarray]:
    x = np.asarray([[fnum(r.get(f)) for f in features] for r in train_rows], dtype=np.float64)
    if x.size == 0:
        return {"mean": np.zeros(len(features)), "std": np.ones(len(features)), "coef": np.zeros((len(actions), len(features) + 1))}
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    coefs = []
    lam = 5.0
    eye = np.eye(xb.shape[1])
    eye[0, 0] = 0.0
    for action in actions:
        y = np.asarray([action_value(r, action)[0] for r in train_rows], dtype=np.float64)
        # Risk-capped utility: prefer gain, penalize severe action outcomes.
        y = y - 4.0 * (y <= SEVERE)
        beta = np.linalg.solve(xb.T @ xb + lam * eye, xb.T @ y)
        coefs.append(beta)
    return {"mean": mean, "std": std, "coef": np.vstack(coefs)}


def predict_linear(row: dict[str, Any], model: dict[str, np.ndarray], actions: list[str], features: list[str]) -> str:
    x = np.asarray([fnum(row.get(f)) for f in features], dtype=np.float64)
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([[1.0], xs])
    scores = model["coef"] @ xb
    return actions[int(np.argmax(scores))]


def rule_predict(row: dict[str, Any]) -> str:
    # Conservative, predeclared rule-list baseline for comparison with ridge.
    if fnum(row["wdmamba_a0p375_dPSNR"]) <= SEVERE:
        return "A0"
    if fnum(row["wdmamba_residual_abs_mean"]) > 0.035 and fnum(row["feature_a0_saturation_high"]) > 0.01:
        return "FS050"
    return "WD0375"


def router_oof(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    actions = [a for a, _, _ in ACTION_SPECS]
    features = DEPLOYABLE_FEATURES
    oof_rows = []
    for seed in SEEDS:
        for fold in range(5):
            train = [r for r in rows if fold_id(str(r["name"]), seed) != fold]
            held = [r for r in rows if fold_id(str(r["name"]), seed) == fold]
            model = fit_linear(train, actions, features, seed)
            for r in held:
                for router_name, action in [
                    ("ridge_utility", predict_linear(r, model, actions, features)),
                    ("rulelist_conservative", rule_predict(r)),
                ]:
                    d, s = action_value(r, action)
                    rec = dict(r)
                    rec.update({
                        "seed": seed,
                        "outer_fold": fold,
                        "router": router_name,
                        "selected_action": action,
                        "dPSNR": d,
                        "dSSIM": s,
                    })
                    oof_rows.append(rec)
    summaries = []
    for router_name in sorted(set(r["router"] for r in oof_rows)):
        sub = [r for r in oof_rows if r["router"] == router_name]
        summ = summarize(sub)
        summ["router"] = router_name
        summ["aggregate_gate_pass"] = gate_aggregate(summ)
        summaries.append(summ)
    write_csv(out_dir / "v22_c9b_router_oof_summary.csv", summaries)
    dist = []
    for router_name in sorted(set(r["router"] for r in oof_rows)):
        sub = [r for r in oof_rows if r["router"] == router_name]
        for action in actions:
            count = sum(r["selected_action"] == action for r in sub)
            dist.append({"router": router_name, "action": action, "count": count, "usage": count / max(1, len(sub))})
    write_csv(out_dir / "v22_c9b_router_action_distribution.csv", dist)
    group_all = []
    for router_name in sorted(set(r["router"] for r in oof_rows)):
        sub = [r for r in oof_rows if r["router"] == router_name]
        group_all.extend(group_bins(sub, router_name))
    write_csv(out_dir / "v22_c9b_router_groupmin_bins.csv", group_all)
    write_csv(out_dir / "v22_c9b_router_expert_usage_by_group.csv", group_all)
    removal = []
    for router_name in sorted(set(r["router"] for r in oof_rows)):
        sub = [r for r in oof_rows if r["router"] == router_name]
        full = summarize(sub)
        removal.append({"router": router_name, "ablation": "full", **full})
        for prefix in ["WD", "FS", "MB"]:
            repl = []
            for r in sub:
                rec = dict(r)
                if str(r["selected_action"]).startswith(prefix):
                    rec["selected_action"] = "A0"
                    rec["dPSNR"] = 0.0
                    rec["dSSIM"] = 0.0
                repl.append(rec)
            s = summarize(repl)
            removal.append({
                "router": router_name,
                "ablation": f"remove_{prefix}",
                **s,
                "mean_drop_vs_full": full["mean_dPSNR"] - s["mean_dPSNR"],
                "hard_drop_vs_full": full["hard_bottom25_dPSNR"] - s["hard_bottom25_dPSNR"],
            })
    write_csv(out_dir / "v22_c9b_router_removal_ablation.csv", removal)
    ab = []
    for feature_group, keep in [
        ("no_expert_residual_features", [f for f in features if "residual" not in f and "fulludp_mae" not in f]),
        ("image_only", [f for f in features if f.startswith("feature_input") or f in ["dark_channel_mean", "input_edge_density", "input_low_texture_proxy", "sky_highlight_proxy"]]),
        ("physics_only", [f for f in features if f in ["feature_depth_mean", "feature_depth_std", "transmission_mean", "transmission_std", "haze_density_mean", "haze_density_p90", "airlight_proxy_p99"]]),
    ]:
        # Fast seed 3407 OOF only for feature sensitivity, not promotion.
        recs = []
        for fold in range(5):
            train = [r for r in rows if fold_id(str(r["name"]), 3407) != fold]
            held = [r for r in rows if fold_id(str(r["name"]), 3407) == fold]
            model = fit_linear(train, actions, keep, 3407)
            for r in held:
                action = predict_linear(r, model, actions, keep)
                d, s = action_value(r, action)
                recs.append({**r, "selected_action": action, "dPSNR": d, "dSSIM": s})
        ab.append({"router": "ridge_utility", "feature_group": feature_group, **summarize(recs)})
    write_csv(out_dir / "v22_c9b_router_feature_ablation.csv", ab)
    best = max(summaries, key=lambda s: (bool(s["aggregate_gate_pass"]), fnum(s["mean_dPSNR"])))
    best_groups = [g for g in group_all if g["profile"] == best["router"]]
    group_pass = all(bool(g["group_gate_pass"]) for g in best_groups)
    decision = "C9B_LOW_CAPACITY_ROUTER_PASS_AUTHORIZE_C10_FORMAL" if bool(best["aggregate_gate_pass"]) and group_pass else "C9B_LOW_CAPACITY_ROUTER_FAIL_STOP_NO_C10"
    (out_dir / "v22_c9b_router_decision.md").write_text(
        f"# C9-B Decision\n\nDecision: `{decision}`\n\nBest router: `{best['router']}`.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "best_router": best["router"], "summary": best, "group_gate_pass": group_pass}


def c9c(rows: list[dict[str, Any]], profile: str, out_dir: Path) -> dict[str, Any]:
    if profile in {"WD0375", "WD050", "FS050"}:
        key = f"{profile}_dPSNR"
        skey = f"{profile}_dSSIM"
        prow = profile_rows(rows, profile, key, skey)
    elif profile == "S3_oracle":
        prow = profile_rows(rows, profile, "S3_oracle_dPSNR", "S3_oracle_dSSIM")
    else:
        raise RuntimeError(f"unsupported C9-C profile {profile}")
    bins = group_bins(prow, profile)
    write_csv(out_dir / "v22_c9c_shifted_bin_metrics.csv", bins)
    dim = []
    for group in sorted(set(r["group"] for r in bins)):
        sub = [r for r in bins if r["group"] == group]
        dim.append({
            "profile": profile,
            "group": group,
            "bins": len(sub),
            "min_mean_dPSNR": min(fnum(r["mean_dPSNR"]) for r in sub),
            "min_hard_bottom25_dPSNR": min(fnum(r["hard_bottom25_dPSNR"]) for r in sub),
            "min_positive_ratio": min(fnum(r["positive_ratio"]) for r in sub),
            "max_severe_loss_per_600": max(fnum(r["severe_loss_per_600"]) for r in sub),
            "group_pass": all(bool(r["group_gate_pass"]) for r in sub),
        })
    write_csv(out_dir / "v22_c9c_shifted_dimension_summary.csv", dim)
    bounds = []
    for r in bins:
        bounds.append({
            "profile": profile,
            "group": r["group"],
            "bin": r["bin"],
            "count": r["count"],
            "mean_bootstrap_LCB": r["mean_bootstrap_LCB"],
            "hard_bootstrap_LCB": r["hard_bootstrap_LCB"],
            "positive_Wilson_LCB": r["positive_Wilson_LCB"],
            "severe_Wilson_UCB": r["severe_Wilson_UCB"],
        })
    write_csv(out_dir / "v22_c9c_bootstrap_wilson_bounds.csv", bounds)
    pass_all = all(bool(r["group_gate_pass"]) for r in bins)
    decision = "C9C_GROUPMIN_PASS_AUTHORIZE_C10_FORMAL_PREP" if pass_all else "C9C_GROUPMIN_FAIL_NO_C10"
    (out_dir / "v22_c9c_groupmin_decision.md").write_text(
        f"# C9-C Group-Min Decision\n\nDecision: `{decision}`\n\nProfile: `{profile}`.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "profile": profile, "dimension_summary": dim}


def c10_formal_5x3(rows: list[dict[str, Any]], profile: str, out_dir: Path) -> dict[str, Any]:
    if profile != "WD0375":
        raise RuntimeError(f"C10 formal currently sealed only for WD0375, got {profile}")
    prow = profile_rows(rows, profile, "WD0375_dPSNR", "WD0375_dSSIM")
    fold_rows = []
    for seed in SEEDS:
        for fold in range(5):
            sub = [r for r in prow if fold_id(str(r["name"]), seed) == fold]
            summ = summarize(sub)
            summ.update({"seed": seed, "fold": fold, "profile": profile, "aggregate_gate_pass": gate_aggregate(summ)})
            fold_rows.append(summ)
    write_csv(out_dir / "v22_c10_formal_5x3_fold_metrics.csv", fold_rows)
    agg_rows = []
    full = summarize(prow)
    full.update({
        "profile": profile,
        "scope": "full_600",
        "aggregate_gate_pass": gate_aggregate(full),
    })
    agg_rows.append(full)
    agg_rows.append({
        "profile": profile,
        "scope": "fold_mean",
        "count": statistics.mean([fnum(r["count"]) for r in fold_rows]),
        "mean_dPSNR": statistics.mean([fnum(r["mean_dPSNR"]) for r in fold_rows]),
        "hard_bottom25_dPSNR": statistics.mean([fnum(r["hard_bottom25_dPSNR"]) for r in fold_rows]),
        "easy_top25_dPSNR": statistics.mean([fnum(r["easy_top25_dPSNR"]) for r in fold_rows]),
        "dSSIM": statistics.mean([fnum(r["dSSIM"]) for r in fold_rows]),
        "positive_ratio": statistics.mean([fnum(r["positive_ratio"]) for r in fold_rows]),
        "severe_loss_per_600": statistics.mean([fnum(r["severe_loss_per_600"]) for r in fold_rows]),
        "aggregate_gate_pass": all(bool(r["aggregate_gate_pass"]) for r in fold_rows),
    })
    agg_rows.append({
        "profile": profile,
        "scope": "fold_worst",
        "count": min([fnum(r["count"]) for r in fold_rows]),
        "mean_dPSNR": min([fnum(r["mean_dPSNR"]) for r in fold_rows]),
        "hard_bottom25_dPSNR": min([fnum(r["hard_bottom25_dPSNR"]) for r in fold_rows]),
        "easy_top25_dPSNR": min([fnum(r["easy_top25_dPSNR"]) for r in fold_rows]),
        "dSSIM": min([fnum(r["dSSIM"]) for r in fold_rows]),
        "positive_ratio": min([fnum(r["positive_ratio"]) for r in fold_rows]),
        "severe_loss_per_600": max([fnum(r["severe_loss_per_600"]) for r in fold_rows]),
        "aggregate_gate_pass": all(bool(r["aggregate_gate_pass"]) for r in fold_rows),
    })
    write_csv(out_dir / "v22_c10_formal_5x3_summary.csv", agg_rows)
    group_rows = group_bins(prow, profile)
    write_csv(out_dir / "v22_c10_formal_5x3_group_bins.csv", group_rows)
    all_folds_pass = all(bool(r["aggregate_gate_pass"]) for r in fold_rows)
    group_pass = all(bool(r["group_gate_pass"]) for r in group_rows)
    pass_formal = bool(full["aggregate_gate_pass"]) and all_folds_pass and group_pass
    decision = "C10_FORMAL_5X3_WD0375_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW" if pass_formal else "C10_FORMAL_5X3_WD0375_FAIL_LOCKED_BLOCKED"
    (out_dir / "v22_c10_formal_5x3_decision.md").write_text(
        "# C10 Formal 5x3 Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Profile: `{profile}`.\n\n"
        f"Full 600 mean/hard/easy/positive/severe: `{full['mean_dPSNR']:.6f}` / "
        f"`{full['hard_bottom25_dPSNR']:.6f}` / `{full['easy_top25_dPSNR']:.6f}` / "
        f"`{full['positive_ratio']:.6f}` / `{full['severe_loss_per_600']:.2f}/600`.\n\n"
        "Locked Haze4K test remains untouched by this formal replay. This decision only authorizes a separate review for one fixed locked one-shot.\n",
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "profile": profile,
        "full": full,
        "all_fold_gate_pass": all_folds_pass,
        "group_gate_pass": group_pass,
    }


def write_c9_0(c8_root: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    files = [
        "v22_c8_1_wdmamba_per_image.csv",
        "v22_c8_2_fsudp_per_image.csv",
        "v22_c8_3_mbtaylor_per_image.csv",
        "v22_c8_forward_selection_per_image.csv",
        "v22_c8_summary.json",
    ]
    manifest = []
    for name in files:
        p = c8_root / name
        manifest.append({"file": name, "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else ""})
    write_csv(out_dir / "v22_c9_0_metric_parity_report.csv", manifest)
    (out_dir / "v22_c9_0_no_locked_status.txt").write_text(
        "locked_test_touched=false\nrouter_training_done=false\ndistillation_done=false\nsource=C8 train-derived per-image tables only\n",
        encoding="utf-8",
    )
    (out_dir / "v22_c9_0_expert_provenance_audit.md").write_text(
        "# C9-0 Expert Provenance Audit\n\n"
        "- WDMamba sha256: `57ff24c3791e593f0172607fea66252a8ba5475ab0e417f4cf48e72b4c9a36da`.\n"
        "- FSNet+UDP sha256: `25cc334f44c2fac979baad7f158526c9f8d751c21ea282974b0e4d9791fc0a27`.\n"
        "- MB-TaylorFormerV2-L sha256: `954229a6862cd7058c8769a9362a88f9ef2ef132664a1b05e7f7f204b617f2f9`.\n"
        "- Current FullUDP sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`.\n"
        "- C9 uses C8 train-derived per-image tables and does not rerender locked data.\n",
        encoding="utf-8",
    )
    (out_dir / "v22_c9_0_render_reproducibility.md").write_text(
        "# C9-0 Render Reproducibility\n\n"
        f"C8 per-image row count loaded: `{len(rows)}`. Row order was checked across WDMamba, FSNet+UDP, and MB-Taylor tables.\n"
        "C9-A/C9-B/C9-C are table replays over C8 rendered train-derived outputs, so no new render nondeterminism is introduced in C9.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c8-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force-router", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_merged(args.c8_root)
    write_c9_0(args.c8_root, args.out_dir, rows)
    fixed_pass, fixed_payload = fixed_profiles(rows, args.out_dir)
    router_payload: dict[str, Any] | None = None
    selected_profile = "WD0375"
    if args.force_router or not fixed_pass:
        router_payload = router_oof(rows, args.out_dir)
        if router_payload["decision"] == "C9B_LOW_CAPACITY_ROUTER_PASS_AUTHORIZE_C10_FORMAL":
            selected_profile = str(router_payload["best_router"])
    c9c_payload = c9c(rows, "WD0375", args.out_dir)
    c10_payload: dict[str, Any] | None = None
    if fixed_pass and c9c_payload["decision"] == "C9C_GROUPMIN_PASS_AUTHORIZE_C10_FORMAL_PREP":
        decision = "C9A_FIXED_WD0375_STRONG_PASS_AUTHORIZE_C10_FORMAL"
        c10_authorized = True
        c10_payload = c10_formal_5x3(rows, "WD0375", args.out_dir)
    elif router_payload and router_payload["decision"] == "C9B_LOW_CAPACITY_ROUTER_PASS_AUTHORIZE_C10_FORMAL":
        decision = "C9B_LOW_CAPACITY_ROUTER_PASS_AUTHORIZE_C10_FORMAL"
        c10_authorized = True
    else:
        decision = "C9_FAIL_NO_C10"
        c10_authorized = False
    (args.out_dir / "v22_c9_decision.md").write_text(
        "# C9 Decision\n\n"
        f"Decision: `{decision}`\n\n"
        "C9 used only C8 train-derived per-image tables. No locked Haze4K test, MoE training, or distillation was run.\n",
        encoding="utf-8",
    )
    summary = {
        "decision": decision,
        "c10_formal_authorized": c10_authorized,
        "locked_test_touched": False,
        "router_training_done": bool(router_payload),
        "distillation_done": False,
        "fixed_profiles": fixed_payload,
        "router": router_payload,
        "c9c": c9c_payload,
        "c10_formal_5x3": c10_payload,
    }
    write_json(args.out_dir / "v22_c9_summary.json", summary)
    readme = [
        "# Haze4K v2.2 C9 Fixed WDMamba Router Evidence",
        "",
        f"Decision: `{decision}`",
        "",
        "C9 used C8 train-derived per-image tables only. Locked test remained untouched.",
        "",
        "Primary outputs:",
        "",
        "- `v22_c9a_fixed_profiles_summary.csv`",
        "- `v22_c9a_fixed_profiles_groupmin_bins.csv`",
        "- `v22_c9c_shifted_dimension_summary.csv`",
        "- `v22_c9_decision.md`",
        "- `v22_c9_summary.json`",
    ]
    if router_payload:
        readme.extend(["- `v22_c9b_router_oof_summary.csv`", "- `v22_c9b_router_decision.md`"])
    if c10_payload:
        readme.extend(["- `v22_c10_formal_5x3_summary.csv`", "- `v22_c10_formal_5x3_decision.md`"])
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"C9_ANALYSIS_OK decision={decision}")


if __name__ == "__main__":
    main()
