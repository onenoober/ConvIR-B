#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for p in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from experience_docx.tools import nopost_lowband_v222_n3_microfit as v222  # noqa: E402
from experience_docx.tools import nopost_lowband_v223_oof_train as v223  # noqa: E402
from models.ConvIR import build_net as build_a0_net  # noqa: E402
from models.NoPostGatedLowbandConvIR import NoPostGatedLowbandConvIR  # noqa: E402
from models.NoPostGatedLowbandConvIR import build_net as build_gated_net  # noqa: E402
from models.NoPostGatedLowbandConvIR import load_haze4k_partial  # noqa: E402

SEVERE = -0.20
STRONG_REG = -0.05


def wtxt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def wjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rcsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def wcsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def fnum(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def vals(xs: list[float]) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


def mean(xs: list[float]) -> float:
    ys = vals(xs)
    return sum(ys) / len(ys) if ys else float("nan")


def std(xs: list[float]) -> float:
    ys = vals(xs)
    return statistics.pstdev(ys) if ys else float("nan")


def pct(xs: list[float], q: float) -> float:
    ys = sorted(vals(xs))
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def cvar(xs: list[float], q: float = 5.0) -> float:
    cut = pct(xs, q)
    return mean([x for x in xs if math.isfinite(x) and x <= cut])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=str(cwd or REPO_ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xv, yv = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = mean(xv), mean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda x: x[1])
    out = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = r
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def roc_auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if math.isfinite(s) and y == 1]
    neg = [s for s, y in zip(scores, labels) if math.isfinite(s) and y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def ap_score(scores: list[float], labels: list[int]) -> float:
    pairs = sorted([(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)], reverse=True)
    total = sum(y for _, y in pairs)
    if total <= 0:
        return float("nan")
    tp, precisions = 0, []
    for i, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / total


def ece(scores: list[float], labels: list[int], nbin: int = 10) -> tuple[float, float, list[dict[str, Any]]]:
    pairs = [(max(0.0, min(1.0, s)), int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    if not pairs:
        return float("nan"), float("nan"), []
    rows, total, out_ece, out_mce = [], len(pairs), 0.0, 0.0
    for i in range(nbin):
        lo, hi = i / nbin, (i + 1) / nbin
        sub = [(s, y) for s, y in pairs if lo <= s < hi or (i == nbin - 1 and lo <= s <= hi)]
        if not sub:
            rows.append({"bin": i, "lo": lo, "hi": hi, "count": 0})
            continue
        conf, acc = mean([s for s, _ in sub]), mean([float(y) for _, y in sub])
        gap = abs(conf - acc)
        out_ece += len(sub) / total * gap
        out_mce = max(out_mce, gap)
        rows.append({"bin": i, "lo": lo, "hi": hi, "count": len(sub), "mean_prob": conf, "unsafe_rate": acc, "abs_gap": gap})
    return out_ece, out_mce, rows


def split_rows(v223_evidence: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(v223_evidence.glob("fold*/v223_fold*_per_image_eval.csv")):
        rows.extend(rcsv(path))
    return rows


def metric_summary(rows: list[dict[str, Any]], variant: str, source: str, folds: int = 3) -> dict[str, Any]:
    d = [fnum(r.get("dPSNR")) for r in rows]
    ordered = sorted(rows, key=lambda r: fnum(r.get("A0_PSNR")))
    k = max(1, len(ordered) // 4) if ordered else 1
    strong = [r for r in rows if int(fnum(r.get("strong_reference"), 0)) == 1]
    fold_tail = 0
    for fold in sorted({int(fnum(r.get("fold", r.get("oof_fold", -1)))) for r in rows}):
        sub = [r for r in rows if int(fnum(r.get("fold", r.get("oof_fold", -999)))) == fold]
        sd = [fnum(r.get("dPSNR")) for r in sub]
        if pct(sd, 5) >= -0.50 and mean([1.0 if x <= SEVERE else 0.0 for x in sd]) <= 0.08:
            fold_tail += 1
    return {
        "source": source,
        "variant": variant,
        "count": len(rows),
        "mean_dPSNR": mean(d),
        "median_dPSNR": pct(d, 50),
        "p05_dPSNR": pct(d, 5),
        "CVaR5_dPSNR": cvar(d),
        "hard_bottom25_dPSNR": mean([fnum(r.get("dPSNR")) for r in ordered[:k]]),
        "easy_top25_dPSNR": mean([fnum(r.get("dPSNR")) for r in ordered[-k:]]),
        "positive_ratio": mean([1.0 if x > 0 else 0.0 for x in d]),
        "severe_count": sum(1 for x in d if x <= SEVERE),
        "severe_rate": mean([1.0 if x <= SEVERE else 0.0 for x in d]),
        "strong_reference_count": len(strong),
        "strong_reference_regressions": sum(1 for r in strong if fnum(r.get("dPSNR")) <= STRONG_REG),
        "strong_reference_regression_rate": mean([1.0 if fnum(r.get("dPSNR")) <= STRONG_REG else 0.0 for r in strong]) if strong else 0.0,
        "fold_count": folds,
        "fold_tail_pass": fold_tail,
    }


def loose_gate(summary: dict[str, Any]) -> bool:
    return (
        fnum(summary.get("mean_dPSNR")) >= 0.05
        and fnum(summary.get("hard_bottom25_dPSNR")) >= 0.05
        and fnum(summary.get("easy_top25_dPSNR")) >= -0.05
        and fnum(summary.get("positive_ratio")) >= 0.55
        and fnum(summary.get("p05_dPSNR")) >= -0.50
        and fnum(summary.get("CVaR5_dPSNR")) >= -0.80
        and fnum(summary.get("severe_rate")) <= 0.08
        and fnum(summary.get("strong_reference_regression_rate")) <= 0.20
        and int(summary.get("fold_tail_pass", 0)) >= max(1, math.ceil(int(summary.get("fold_count", 3)) * 2 / 3))
    )


def load_full_noop(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    for row in rcsv(path):
        if row.get("variant") == "V221_noop_control" and row.get("name") not in out:
            out[row["name"]] = row
    return out


def base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        split_csv=args.split_csv,
        v221_metrics_csv=args.v221_metrics_csv,
        v221_variant=args.v221_variant,
        hidden_channels=args.hidden_channels,
        mid_grid=args.mid_grid,
        final_grid=args.final_grid,
        crop_size=args.crop_size,
        risk_gamma=args.risk_gamma,
        risk_bias=args.risk_bias,
        identity_tol=args.identity_tol,
        train_scope="adapter_only",
        learning_rate=8e-5,
        weight_decay=1e-4,
        grad_clip_norm=0.001,
        risk_loss_weight=args.risk_loss_weight,
        gate_mean_weight=args.gate_mean_weight,
        action_l1_weight=args.action_l1_weight,
        strong_reference_psnr=args.strong_reference_psnr,
        seed=args.seed,
    )


def p0(args: argparse.Namespace, samples: list[v223.FoldSample], device: torch.device) -> dict[str, str]:
    files = [
        REPO_ROOT / "Dehazing/ITS/models/NoPostGatedLowbandConvIR.py",
        REPO_ROOT / "experience_docx/tools/nopost_lowband_v222_n3_microfit.py",
        REPO_ROOT / "experience_docx/tools/nopost_lowband_v223_oof_train.py",
        TOOL_PATH,
    ]
    payload = {
        "branch": cmd(["git", "branch", "--show-current"]),
        "commit": cmd(["git", "rev-parse", "--short", "HEAD"]),
        "anchor_commit": cmd(["git", "rev-parse", "--short", "github/codex/haze4k-official-arch-anchor"]),
        "v223_runtime_repo": str(args.v223_repo),
        "v223_runtime_commit": cmd(["git", "rev-parse", "--short", "HEAD"], args.v223_repo),
        "v223_runtime_status_short": cmd(["git", "status", "--short"], args.v223_repo),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "key_file_sha256": {str(p.relative_to(REPO_ROOT)): sha(p) for p in files if p.is_file()},
        "locked_test_touched": False,
    }
    wjson(args.out_dir / "v224_p0_runtime_source_hash.json", payload)
    wtxt(args.out_dir / "v224_p0_route_code_diffstat.txt", cmd(["git", "diff", "--stat", "github/codex/haze4k-official-arch-anchor", "--", "Dehazing/ITS/models/NoPostGatedLowbandConvIR.py", "experience_docx/tools"]))
    sig = str(inspect.signature(NoPostGatedLowbandConvIR.forward))
    wjson(args.out_dir / "v224_p0_forward_signature.json", {"forward_signature": sig, "pass": sig == "(self, x)"})
    hits = []
    for pth in files:
        if not pth.is_file():
            continue
        for i, line in enumerate(pth.read_text(errors="ignore").splitlines(), start=1):
            low = line.lower()
            if any(tok in low for tok in ["output-output", "output_level", "output-level", "learned correction", "expert output", "cand_pred - a0_pred"]):
                hits.append(f"{pth.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    wtxt(args.out_dir / "v224_p0_forbidden_symbol_scan.txt", "\n".join(hits) if hits else "NO_FORBIDDEN_SYMBOL_CANDIDATES_FOUND")
    state = v222.load_state(args.checkpoint, "cpu")
    a0 = build_a0_net("base", "Haze4K", "original").to(device)
    a0.load_state_dict(state)
    a0.eval()
    model = build_gated_net("base", "Haze4K", "original", hidden_channels=args.hidden_channels, mid_grid=args.mid_grid, final_grid=args.final_grid, risk_gamma=args.risk_gamma, risk_bias=args.risk_bias).to(device)
    partial = load_haze4k_partial(model, state)
    scope = v222.set_train_scope(model, "adapter_only")
    model.eval()
    x = v222.image_tensor(samples[0].input_path, device)
    x, h, w = v222.pad_to(x, 32)
    with torch.no_grad():
        a0p = a0(x)[2][:, :, :h, :w]
        rp = model(x)[2][:, :, :h, :w]
    identity = {
        "zero_init_max_abs_vs_A0": float((a0p - rp).abs().max().cpu()),
        "forward_finite": bool(torch.isfinite(rp).all().item()),
        "partial_load": partial,
        "scope": scope,
        "official_checkpoint_hash_matches": payload["checkpoint_sha256"] == "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088",
        "only_nopost_policy_trainable": all(n.startswith("nopost_gated_lowband_policy.") for n in scope["trainable_names"]),
        "locked_test_touched": False,
    }
    identity["pass"] = sig == "(self, x)" and identity["zero_init_max_abs_vs_A0"] <= args.identity_tol and identity["forward_finite"] and identity["official_checkpoint_hash_matches"] and identity["only_nopost_policy_trainable"]
    wjson(args.out_dir / "v224_p0_identity_summary.json", identity)
    decision = "P0_PASS" if identity["pass"] else "P0_FAIL_ENGINEERING"
    wtxt(args.out_dir / "v224_p0_contract_audit.md", f"# v2.24 P0 Contract Audit\n\nDecision: `{decision}`\n\n- forward: `{sig}`\n- zero-init max abs vs A0: `{identity['zero_init_max_abs_vs_A0']}`\n- locked test touched: `false`")
    wtxt(args.out_dir / "v224_p0_decision.md", f"# v2.24 P0 Decision\n\nDecision: `{decision}`\n")
    return {"decision": decision}


def p1(args: argparse.Namespace, v223_rows: list[dict[str, str]], full_noop: dict[str, dict[str, str]]) -> dict[str, str]:
    rows = []
    for row in v223_rows:
        full = full_noop.get(row["name"], {})
        crop, fullpsnr, d = fnum(row.get("A0_PSNR")), fnum(full.get("A0_PSNR")), fnum(row.get("dPSNR"))
        rows.append({
            "fold": row.get("fold"),
            "name": row["name"],
            "crop_A0_PSNR": crop,
            "full_A0_PSNR": fullpsnr,
            "crop_minus_full_A0_PSNR": crop - fullpsnr if math.isfinite(crop) and math.isfinite(fullpsnr) else float("nan"),
            "dPSNR": d,
            "crop_strong_reference": int(crop >= args.strong_reference_psnr) if math.isfinite(crop) else 0,
            "full_strong_reference": int(fullpsnr >= args.strong_reference_psnr) if math.isfinite(fullpsnr) else 0,
            "crop_strong_regression": int(crop >= args.strong_reference_psnr and d <= STRONG_REG) if math.isfinite(crop) else 0,
            "full_strong_regression": int(fullpsnr >= args.strong_reference_psnr and d <= STRONG_REG) if math.isfinite(fullpsnr) else 0,
            "v221_full_row_present": int(bool(full)),
        })
    wcsv(args.out_dir / "v224_p1_crop_vs_full_metric_report.csv", rows)
    comp = []
    for mask in ["crop", "full"]:
        sub = [r for r in rows if r[f"{mask}_strong_reference"]]
        comp.append({"mask_source": mask, "count": len(sub), "coverage": len(sub) / len(rows), "regression_rate": mean([float(r[f"{mask}_strong_regression"]) for r in sub]) if sub else 0.0, "mean_dPSNR": mean([fnum(r["dPSNR"]) for r in sub])})
    comp += [
        {"mask_source": "overlap_crop_and_full", "count": sum(1 for r in rows if r["crop_strong_reference"] and r["full_strong_reference"])},
        {"mask_source": "crop_only", "count": sum(1 for r in rows if r["crop_strong_reference"] and not r["full_strong_reference"])},
        {"mask_source": "full_only", "count": sum(1 for r in rows if r["full_strong_reference"] and not r["crop_strong_reference"])},
    ]
    wcsv(args.out_dir / "v224_p1_strong_mask_source_comparison.csv", comp)
    bucket_rows = []
    for scope in ["all"] + sorted({f"fold{r['fold']}" for r in rows}):
        sub = rows if scope == "all" else [r for r in rows if f"fold{r['fold']}" == scope]
        if not sub:
            continue
        k = max(1, len(sub) // 4)
        by = {r["name"]: r for r in sub}
        sets = {
            "crop_hard": {r["name"] for r in sorted(sub, key=lambda r: fnum(r["crop_A0_PSNR"]))[:k]},
            "full_hard": {r["name"] for r in sorted(sub, key=lambda r: fnum(r["full_A0_PSNR"]))[:k]},
            "crop_easy": {r["name"] for r in sorted(sub, key=lambda r: fnum(r["crop_A0_PSNR"]))[-k:]},
            "full_easy": {r["name"] for r in sorted(sub, key=lambda r: fnum(r["full_A0_PSNR"]))[-k:]},
        }
        for a, b in [("crop_hard", "full_hard"), ("crop_easy", "full_easy")]:
            inter, union = sets[a] & sets[b], sets[a] | sets[b]
            bucket_rows.append({"scope": scope, "bucket_pair": f"{a}_vs_{b}", "overlap": len(inter), "jaccard": len(inter) / len(union), "mean_dPSNR_a": mean([fnum(by[n]["dPSNR"]) for n in sets[a]]), "mean_dPSNR_b": mean([fnum(by[n]["dPSNR"]) for n in sets[b]])})
    wcsv(args.out_dir / "v224_p1_hard_easy_bucket_comparison.csv", bucket_rows)
    crop_count, full_count = comp[0]["count"], comp[1]["count"]
    decision = "P1_CROP_STRONG_MASK_OVERBROAD_STRONG_GATE_SHOULD_USE_FULL_IMAGE_OR_PRECOMPUTED_MASK" if crop_count == len(rows) and full_count < crop_count else "P1_METRIC_COMPARABILITY_RECORDED"
    wtxt(args.out_dir / "v224_p1_metric_decision.md", f"# v2.24 P1 Metric Decision\n\nDecision: `{decision}`\n\n- crop strong count: `{crop_count}/{len(rows)}`\n- full-image strong count: `{full_count}/{len(rows)}`\n")
    return {"decision": decision}


def p2(args: argparse.Namespace, v223_rows: list[dict[str, str]]) -> dict[str, str]:
    joined = []
    for r in v223_rows:
        mid, fin = fnum(r.get("mid_unsafe_prob")), fnum(r.get("final_unsafe_prob"))
        joined.append({
            "fold": r.get("fold"),
            "name": r["name"],
            "dPSNR": fnum(r.get("dPSNR")),
            "severe": int(fnum(r.get("dPSNR")) <= SEVERE),
            "unsafe_action_label": int(fnum(r.get("unsafe_action_label"), 0)),
            "v221_unsafe_action_probability": fnum(r.get("unsafe_action_probability")),
            "trained_mid_unsafe_prob": mid,
            "trained_final_unsafe_prob": fin,
            "trained_mean_unsafe_prob": mean([mid, fin]),
        })
    wcsv(args.out_dir / "v224_p2_trained_gate_vs_v221_probability.csv", joined)
    labels = [int(r["unsafe_action_label"]) for r in joined]
    v221_probs = [fnum(r["v221_unsafe_action_probability"]) for r in joined]
    hist, metrics, bin_rows = [], [], []
    for source, field in [("trained_mid", "trained_mid_unsafe_prob"), ("trained_final", "trained_final_unsafe_prob"), ("trained_mean", "trained_mean_unsafe_prob"), ("v221_reference", "v221_unsafe_action_probability")]:
        scores = [fnum(r[field]) for r in joined]
        clean = vals(scores)
        hist.append({"source": source, "bin": "stats", "count": len(clean), "mean": mean(scores), "std": std(scores), "min": min(clean) if clean else float("nan"), "max": max(clean) if clean else float("nan")})
        ee, mm, bins = ece(scores, labels, 10)
        metrics.append({"source": source, "count": len(scores), "positive_labels": sum(labels), "label_base_rate": sum(labels) / len(labels), "prob_mean": mean(scores), "prob_std": std(scores), "prob_min": min(clean) if clean else float("nan"), "prob_max": max(clean) if clean else float("nan"), "roc_auc": roc_auc(scores, labels), "pr_auc_average_precision": ap_score(scores, labels), "brier": mean([(max(0, min(1, s)) - y) ** 2 for s, y in zip(scores, labels) if math.isfinite(s)]), "ece10": ee, "mce10": mm, "pearson_vs_v221_probability": pearson(scores, v221_probs), "spearman_vs_v221_probability": spearman(scores, v221_probs)})
        for b in bins:
            lo, hi = b.get("lo", 9), b.get("hi", -1)
            sub = [r for r in joined if lo <= fnum(r[field]) < hi or (b.get("bin") == 9 and lo <= fnum(r[field]) <= hi)]
            bin_rows.append({"source": source, **b, "mean_dPSNR": mean([fnum(r["dPSNR"]) for r in sub]), "severe_rate": mean([float(r["severe"]) for r in sub])})
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            sub = [r for r in joined if lo <= fnum(r[field]) < hi or (i == 9 and lo <= fnum(r[field]) <= hi)]
            hist.append({"source": source, "bin": f"[{lo:.1f},{hi:.1f}]", "count": len(sub), "unsafe_label_rate": mean([float(r["unsafe_action_label"]) for r in sub]), "severe_rate": mean([float(r["severe"]) for r in sub]), "mean_dPSNR": mean([fnum(r["dPSNR"]) for r in sub])})
    wcsv(args.out_dir / "v224_p2_trained_gate_probability_histogram.csv", hist)
    wcsv(args.out_dir / "v224_p2_trained_gate_roc_pr_ece.csv", metrics)
    wcsv(args.out_dir / "v224_p2_trained_gate_bin_report.csv", bin_rows)
    fold_rows = []
    for fold in sorted({str(r["fold"]) for r in joined}):
        sub = [r for r in joined if str(r["fold"]) == fold]
        labs = [int(r["unsafe_action_label"]) for r in sub]
        for source, field in [("trained_mean", "trained_mean_unsafe_prob"), ("v221_reference", "v221_unsafe_action_probability")]:
            scores = [fnum(r[field]) for r in sub]
            fold_rows.append({"fold": fold, "source": source, "count": len(sub), "positive_labels": sum(labs), "prob_mean": mean(scores), "prob_std": std(scores), "roc_auc": roc_auc(scores, labs), "pr_auc_average_precision": ap_score(scores, labs), "median_threshold": pct(scores, 50), "top20_threshold": pct(scores, 80)})
    wcsv(args.out_dir / "v224_p2_fold_stability_report.csv", fold_rows)
    tm = next(r for r in metrics if r["source"] == "trained_mean")
    vr = next(r for r in metrics if r["source"] == "v221_reference")
    decision = "P2_RISK_HEAD_COLLAPSE_OR_BASE_RATE_LEARNING_CONFIRMED" if fnum(tm["prob_std"]) < 0.01 or fnum(tm["roc_auc"]) <= 0.60 else "P2_TRAINED_GATE_HAS_SOME_INFORMATION_REVIEW_ACTION_LOSS_INTERACTION"
    wtxt(args.out_dir / "v224_p2_decision.md", f"# v2.24 P2 Decision\n\nDecision: `{decision}`\n\n- trained prob mean/std/min/max: `{tm['prob_mean']}` / `{tm['prob_std']}` / `{tm['prob_min']}` / `{tm['prob_max']}`\n- trained ROC-AUC/AP/ECE10/Brier: `{tm['roc_auc']}` / `{tm['pr_auc_average_precision']}` / `{tm['ece10']}` / `{tm['brier']}`\n- v2.21 reference ROC-AUC/AP/ECE10/Brier: `{vr['roc_auc']}` / `{vr['pr_auc_average_precision']}` / `{vr['ece10']}` / `{vr['brier']}`\n")
    return {"decision": decision}


def fwd_override(model: torch.nn.Module, x: torch.Tensor, mid_override: float | None, final_override: float | None) -> tuple[torch.Tensor, dict[str, float]]:
    pol = model.nopost_gated_lowband_policy
    pol.last_tensors = {}
    x2 = F.interpolate(x, scale_factor=0.5)
    x4 = F.interpolate(x2, scale_factor=0.5)
    z2, z4 = model.SCM2(x2), model.SCM1(x4)
    x0 = model.feat_extract[0](x)
    res1 = model.Encoder[0](x0)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    z = model.Decoder[0](z)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    ll, _, _, _, _, _ = pol.dwt(z)
    raw_mid, mid_logit = pol.mid_policy(ll)
    mid_prob, mid_scale = pol._scale_from_logit(mid_logit)
    if mid_override is not None:
        mid_scale = torch.ones_like(mid_scale) * float(mid_override)
    z, _, mid_delta = pol._apply_lowband(z, raw_mid, mid_scale)
    mid_ctx = z
    z = model.feat_extract[4](z)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    ll, _, _, _, _, _ = pol.dwt(z)
    raw_final, final_logit = pol.final_policy(ll, mid_ctx)
    final_prob, final_scale = pol._scale_from_logit(final_logit)
    if final_override is not None:
        final_scale = torch.ones_like(final_scale) * float(final_override)
    z, _, final_delta = pol._apply_lowband(z, raw_final, final_scale)
    out = model.feat_extract[5](z) + x
    return out, {"mid_unsafe_prob": float(mid_prob.mean().cpu()), "final_unsafe_prob": float(final_prob.mean().cpu()), "mid_scale": float(mid_scale.mean().cpu()), "final_scale": float(final_scale.mean().cpu()), "mid_delta_rms": float(torch.sqrt(torch.mean(mid_delta.float() ** 2)).cpu()), "final_delta_rms": float(torch.sqrt(torch.mean(final_delta.float() ** 2)).cpu())}


def p3(args: argparse.Namespace, samples: list[v223.FoldSample], device: torch.device) -> dict[str, str]:
    per: dict[str, list[dict[str, Any]]] = {}
    for fold in args.folds_list:
        _, eval_samples = v223.select_samples(samples, fold, args)
        a0, model, _ = v222.load_a0_and_route(base_args(args), device)
        model.load_state_dict(v222.load_state(args.v223_evidence / f"fold{fold}" / f"v223_fold{fold}_adapter_only_Final.pkl", device))
        a0.eval()
        model.eval()
        with torch.no_grad():
            for s in v223.to_v222_samples(eval_samples):
                x = v222.image_tensor(s.input_path, device)
                y = v222.image_tensor(s.label_path, device)
                x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + v222.idx_hash(s.name))
                x, h, w = v222.pad_to(x, 32)
                y = y[:, :, :h, :w]
                a0p = a0(x)[2][:, :, :h, :w]
                a0ps = v222.tensor_psnr(a0p, y)
                strong = int(a0ps >= args.strong_reference_psnr)
                rawp, rawstats = fwd_override(model, x, 1.0, 1.0)
                rawd = v222.tensor_psnr(rawp[:, :, :h, :w], y) - a0ps
                variants = [("H_noop_control", a0p, {"mid_scale": 0.0, "final_scale": 0.0}, "direct")]
                tp = model(x)[2][:, :, :h, :w]
                ten = model.nopost_gated_lowband_policy.last_tensors
                variants.append(("A_trained_action_trained_gate", tp, {"mid_unsafe_prob": float(ten["mid_unsafe_prob"].mean().cpu()), "final_unsafe_prob": float(ten["final_unsafe_prob"].mean().cpu()), "mid_scale": float(ten["mid_scale"].mean().cpu()), "final_scale": float(ten["final_scale"].mean().cpu()), "mid_delta_rms": float(torch.sqrt(torch.mean(ten["mid_delta"].float() ** 2)).cpu()), "final_delta_rms": float(torch.sqrt(torch.mean(ten["final_delta"].float() ** 2)).cpu())}, "direct"))
                variants.append(("trained_action_all_open_gate", rawp[:, :, :h, :w], rawstats, "direct"))
                scale = float(s.risk.get("v221_risk_scale", 1.0))
                pred, st = fwd_override(model, x, scale, scale)
                variants.append(("B_trained_action_v221_replay_gate", pred[:, :, :h, :w], st, "direct"))
                hard = 1.0 if float(s.risk.get("unsafe_action_label", 0.0)) < 0.5 else 0.0
                pred, st = fwd_override(model, x, hard, hard)
                variants.append(("trained_action_v221_label_hard_gate", pred[:, :, :h, :w], st, "direct"))
                oracle = 1.0 if rawd > STRONG_REG else 0.0
                pred, st = fwd_override(model, x, oracle, oracle)
                variants.append(("C_trained_action_oracle_unsafe_gate", pred[:, :, :h, :w], st, "diagnostic_gt_oracle"))
                for name, img, st, source in variants:
                    ps = v222.tensor_psnr(img, y)
                    d = ps - a0ps
                    per.setdefault(name, []).append({"source": source, "variant": name, "fold": fold, "name": s.name, "A0_PSNR": a0ps, "candidate_PSNR": ps, "dPSNR": d, "strong_reference": strong, "strong_reference_regression": int(strong and d <= STRONG_REG), "severe": int(d <= SEVERE), "unsafe_action_probability": s.risk.get("unsafe_action_probability", float("nan")), "unsafe_action_label": s.risk.get("unsafe_action_label", float("nan")), "v221_risk_scale": s.risk.get("v221_risk_scale", float("nan")), "raw_trained_action_dPSNR": rawd, **st})
    sums = []
    for name, rows in per.items():
        m = metric_summary(rows, name, rows[0].get("source", "direct"), len(args.folds_list))
        m["loose_v223_gate_pass"] = loose_gate(m)
        sums.append(m)
    if args.v221_factorial_csv.is_file():
        for r in rcsv(args.v221_factorial_csv):
            if r.get("variant", "").startswith("V221_factor_"):
                sums.append({"source": "v221_replay_reference_full_image", "variant": r["variant"], "count": r.get("count"), "mean_dPSNR": r.get("mean_dPSNR"), "hard_bottom25_dPSNR": r.get("hard_bottom25_dPSNR"), "p05_dPSNR": r.get("p05_dPSNR"), "CVaR5_dPSNR": r.get("CVaR5_dPSNR"), "severe_rate": r.get("severe_rate"), "strong_reference_regression_rate": r.get("strong_reference_regression_rate"), "protocol_note": "full-image v2.21 replay reference"})
    wcsv(args.out_dir / "v224_p3_posttrain_factorial_action_gate_audit.csv", sums)
    rescue = [r for r in sums if r["variant"] in {"A_trained_action_trained_gate", "B_trained_action_v221_replay_gate", "C_trained_action_oracle_unsafe_gate", "trained_action_v221_label_hard_gate", "trained_action_all_open_gate", "H_noop_control"}]
    wcsv(args.out_dir / "v224_p3_trained_action_v221_gate_rescue.csv", rescue)
    wcsv(args.out_dir / "v224_p3_trained_gate_v221_action_rescue.csv", [{"status": "LIMITED_BY_NO_CACHED_V221_ACTION_TENSORS_IN_V223_CROP_PROTOCOL", "locked_test_touched": False}])
    by = {r["variant"]: r for r in rescue}
    a, b, c = by.get("A_trained_action_trained_gate", {}), by.get("B_trained_action_v221_replay_gate", {}), by.get("C_trained_action_oracle_unsafe_gate", {})
    if b.get("loose_v223_gate_pass") or c.get("loose_v223_gate_pass"):
        decision = "P3_TRAINED_ACTION_CAN_BE_RESCUED_BY_BETTER_GATE"
    elif fnum(b.get("mean_dPSNR")) > fnum(a.get("mean_dPSNR")) + 0.02 or fnum(c.get("severe_rate")) < fnum(a.get("severe_rate")):
        decision = "P3_GATE_RESCUE_IMPROVES_SAFETY_BUT_DOES_NOT_PASS"
    else:
        decision = "P3_TRAINED_ACTION_NOT_RESCUED_BY_AVAILABLE_GATES_OR_GATE_ACTION_INTERACTION_FAILS"
    wtxt(args.out_dir / "v224_p3_decision.md", f"# v2.24 P3 Decision\n\nDecision: `{decision}`\n\nDirect crop-protocol variants were run for trained action under trained, v2.21 replay, v2.21 hard-label, oracle unsafe, all-open, and no-op gates. v2.21 full-image factorial rows are included as references. Locked test remained untouched.\n")
    return {"decision": decision}


def group(name: str) -> str:
    if "nopost_gated_lowband_policy" not in name:
        return "base_or_other"
    if ".risk." in name:
        return "risk_head"
    if ".action." in name:
        return "action_head"
    return "context_head"


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = group(name)
        val = float(torch.linalg.vector_norm(p.grad.detach()).cpu())
        out[g] = out.get(g, 0.0) + val
        out["policy_all"] = out.get("policy_all", 0.0) + val
    return out


def p4(args: argparse.Namespace, samples: list[v223.FoldSample], device: torch.device) -> dict[str, str]:
    balance, audit = [], []
    per_fold = max(1, args.p4_max_samples // len(args.folds_list))
    for fold in args.folds_list:
        train, ev = v223.select_samples(samples, fold, args)
        for scope, sub in [("train_selected", train), ("eval_selected", ev)]:
            labs = [float(s.risk.get("unsafe_action_label", 0.0)) for s in sub]
            probs = [float(s.risk.get("unsafe_action_probability", float("nan"))) for s in sub]
            balance.append({"fold": fold, "scope": scope, "count": len(sub), "unsafe_positive_count": sum(1 for x in labs if x >= 0.5), "unsafe_positive_rate": mean(labs), "v221_probability_mean": mean(probs), "v221_probability_std": std(probs)})
        audit += [(fold, s, i) for i, s in enumerate(train[:per_fold])]
    wcsv(args.out_dir / "v224_p4_risk_label_balance_report.csv", balance)
    _, model, _ = v222.load_a0_and_route(base_args(args), device)
    ckpt = args.v223_evidence / "fold0/v223_fold0_adapter_only_Final.pkl"
    model.load_state_dict(v222.load_state(ckpt, device))
    scope = v222.set_train_scope(model, "adapter_only")
    model.eval()
    model.nopost_gated_lowband_policy.train()
    comp, gvals = [], {}
    for fold, sample, idx in audit:
        x = v222.image_tensor(sample.input_path, device)
        y = v222.image_tensor(sample.label_path, device)
        x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + idx)
        x, h, w = v222.pad_to(x, 32)
        y = y[:, :, :h, :w]
        pred = model(x)
        full = pred[2][:, :, :h, :w]
        y2 = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
        y4 = F.interpolate(y, scale_factor=0.25, mode="bilinear", align_corners=False)
        content = F.l1_loss(pred[0][:, :, : y4.shape[-2], : y4.shape[-1]], y4) + F.l1_loss(pred[1][:, :, : y2.shape[-2], : y2.shape[-1]], y2) + F.l1_loss(full, y)
        ten = model.nopost_gated_lowband_policy.last_tensors
        gate = ten["mid_unsafe_prob"].mean() + ten["final_unsafe_prob"].mean()
        action = ten["mid_scaled_delta_grid"].abs().mean() + ten["final_scaled_delta_grid"].abs().mean()
        target = full.new_tensor([[float(sample.risk.get("unsafe_action_label", 0.0))]])
        risk = F.binary_cross_entropy_with_logits(ten["mid_unsafe_logit"], target) + F.binary_cross_entropy_with_logits(ten["final_unsafe_logit"], target)
        for cname, loss, wt in [("content_l1", content, 1.0), ("risk_bce", risk, args.risk_loss_weight), ("gate_mean_regularizer", gate, args.gate_mean_weight), ("action_l1", action, args.action_l1_weight)]:
            model.zero_grad(set_to_none=True)
            (loss * wt).backward(retain_graph=True)
            for gg, val in grad_norms(model).items():
                gvals.setdefault((cname, gg), []).append(val)
            comp.append({"fold": fold, "name": sample.name, "component": cname, "unweighted_value": float(loss.detach().cpu()), "weight": wt, "weighted_value": float((loss * wt).detach().cpu()), "unsafe_label": sample.risk.get("unsafe_action_label", float("nan")), "mid_prob": float(ten["mid_unsafe_prob"].mean().detach().cpu()), "final_prob": float(ten["final_unsafe_prob"].mean().detach().cpu())})
    wcsv(args.out_dir / "v224_p4_loss_component_magnitude.csv", [{"component": c, "count": len([r for r in comp if r["component"] == c]), "mean_unweighted_value": mean([fnum(r["unweighted_value"]) for r in comp if r["component"] == c]), "mean_weighted_value": mean([fnum(r["weighted_value"]) for r in comp if r["component"] == c])} for c in sorted({r["component"] for r in comp})])
    grows = [{"component": c, "module_group": g, "count": len(v), "mean_grad_norm": mean(v), "std_grad_norm": std(v)} for (c, g), v in sorted(gvals.items())]
    wcsv(args.out_dir / "v224_p4_gradient_norm_by_module.csv", grows)
    wcsv(args.out_dir / "v224_p4_gate_mean_regularizer_effect.csv", [{"risk_bias": args.risk_bias, "initial_sigmoid_bias": 1 / (1 + math.exp(-args.risk_bias)), "trained_mid_prob_mean_in_audit": mean([fnum(r["mid_prob"]) for r in comp]), "trained_final_prob_mean_in_audit": mean([fnum(r["final_prob"]) for r in comp]), "gate_mean_weight": args.gate_mean_weight, "optimizer_effect": "positive gate_mean_weight minimizes unsafe probability and opens scale=(1-p)^gamma"}])
    def gn(c: str, g: str) -> float:
        return next((fnum(r["mean_grad_norm"]) for r in grows if r["component"] == c and r["module_group"] == g), float("nan"))
    report = [{"metric": "content_l1_action_grad_to_action_l1_action_grad_ratio", "value": gn("content_l1", "action_head") / max(gn("action_l1", "action_head"), 1e-12)}, {"metric": "risk_bce_risk_grad_to_content_l1_risk_grad_ratio", "value": gn("risk_bce", "risk_head") / max(gn("content_l1", "risk_head"), 1e-12)}]
    wcsv(args.out_dir / "v224_p4_action_l1_vs_content_gradient_report.csv", report)
    decision = "P4_SUPERVISION_GRADIENT_IMBALANCE_RISK_CONFIRMED" if fnum(report[1]["value"]) < 0.25 or fnum(report[0]["value"]) > 10 else "P4_OBJECTIVE_AUDIT_RECORDED"
    wtxt(args.out_dir / "v224_p4_decision.md", f"# v2.24 P4 Decision\n\nDecision: `{decision}`\n\n- audited samples: `{len(audit)}`\n- trainable parameters: `{scope['trainable_param_count']}`\n- no optimizer step, no training, no locked test.\n")
    return {"decision": decision}


def p5(args: argparse.Namespace) -> dict[str, str]:
    rows, manifest, ckpts = [], [], []
    for fold in args.folds_list:
        fold_dir = args.v223_evidence / f"fold{fold}"
        hist_path = fold_dir / f"v223_fold{fold}_train_history.csv"
        if not hist_path.is_file():
            continue
        hist = rcsv(hist_path)
        rows.extend(hist)
        by = {int(fnum(r.get("epoch"))): r for r in hist}
        if 1 in by and 2 in by:
            e1, e2 = by[1], by[2]
            manifest.append({"fold": fold, "epoch1_mean_dPSNR": e1.get("train_mean_dPSNR_vs_A0"), "epoch2_mean_dPSNR": e2.get("train_mean_dPSNR_vs_A0"), "mean_delta_epoch2_minus_epoch1": fnum(e2.get("train_mean_dPSNR_vs_A0")) - fnum(e1.get("train_mean_dPSNR_vs_A0")), "epoch1_p05": e1.get("train_p05_dPSNR_vs_A0"), "epoch2_p05": e2.get("train_p05_dPSNR_vs_A0"), "p05_delta_epoch2_minus_epoch1": fnum(e2.get("train_p05_dPSNR_vs_A0")) - fnum(e1.get("train_p05_dPSNR_vs_A0")), "epoch1_CVaR5": e1.get("train_CVaR5_dPSNR_vs_A0"), "epoch2_CVaR5": e2.get("train_CVaR5_dPSNR_vs_A0"), "CVaR5_delta_epoch2_minus_epoch1": fnum(e2.get("train_CVaR5_dPSNR_vs_A0")) - fnum(e1.get("train_CVaR5_dPSNR_vs_A0")), "epoch1_severe_rate": e1.get("train_severe_rate"), "epoch2_severe_rate": e2.get("train_severe_rate"), "severe_delta_epoch2_minus_epoch1": fnum(e2.get("train_severe_rate")) - fnum(e1.get("train_severe_rate"))})
        ckpts.append({"fold": fold, "epoch_checkpoint_count": len(list(fold_dir.glob("*epoch*.pkl"))), "final_checkpoint_exists": (fold_dir / f"v223_fold{fold}_adapter_only_Final.pkl").is_file()})
    wcsv(args.out_dir / "v224_p5_epoch1_epoch2_eval_summary.csv", rows)
    wcsv(args.out_dir / "v224_p5_tail_growth_manifest.csv", manifest)
    wcsv(args.out_dir / "v224_p5_epoch_checkpoint_manifest.csv", ckpts)
    bad = [r for r in manifest if fnum(r["severe_delta_epoch2_minus_epoch1"]) > 0 and fnum(r["p05_delta_epoch2_minus_epoch1"]) < 0]
    decision = "P5_EPOCH2_MEAN_CAN_RISE_WHILE_TAIL_WORSENS_EXPANDING_EPOCHS_FORBIDDEN" if len(bad) >= max(1, len(manifest) // 2) else "P5_TRAJECTORY_AUDIT_RECORDED_NO_EPOCH_PROMOTION"
    wtxt(args.out_dir / "v224_p5_decision.md", f"# v2.24 P5 Decision\n\nDecision: `{decision}`\n\nPer-epoch checkpoints were not present in compact evidence; no new training was launched to recreate them.\n")
    return {"decision": decision}


def update_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    lines = ["# Haze4K v2.24 NoPost Train-Time Controller Failure Audit Evidence", "", f"Status: {closeout['decision']}", "", "Diagnostic-only route. No new training, no locked Haze4K test, and no checkpoint or threshold selection from locked test.", "", "## Key Decisions", ""]
    for k in ["p0", "p1", "p2", "p3", "p4", "p5"]:
        lines.append(f"- {k.upper()}: `{closeout.get(k, {}).get('decision', 'missing')}`")
    lines += ["", "## Locked-Test Policy", "", "Locked test remained untouched and blocked throughout v2.24."]
    wtxt(args.out_dir / "README.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--v221-metrics-csv", type=Path, required=True)
    ap.add_argument("--v221-factorial-csv", type=Path, required=True)
    ap.add_argument("--v223-repo", type=Path, required=True)
    ap.add_argument("--v223-evidence", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--train-samples-per-fold", type=int, default=384)
    ap.add_argument("--eval-samples-per-fold", type=int, default=160)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--risk-gamma", type=float, default=0.5)
    ap.add_argument("--risk-bias", type=float, default=-1.5)
    ap.add_argument("--v221-variant", default="V221_risk_temperature_gamma0p50")
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--risk-loss-weight", type=float, default=0.05)
    ap.add_argument("--gate-mean-weight", type=float, default=0.0005)
    ap.add_argument("--action-l1-weight", type=float, default=0.0001)
    ap.add_argument("--strong-reference-psnr", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=223)
    ap.add_argument("--p4-max-samples", type=int, default=36)
    args = ap.parse_args()
    args.folds_list = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "status.txt").open("a") as f:
        f.write(f"v224_audit_start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = v223.load_split_samples(args)
    v223_rows = split_rows(args.v223_evidence)
    full_noop = load_full_noop(args.v221_metrics_csv)
    out = {
        "p0": p0(args, samples, device),
        "p1": p1(args, v223_rows, full_noop),
        "p2": p2(args, v223_rows),
        "p3": p3(args, samples, device),
        "p4": p4(args, samples, device),
        "p5": p5(args),
    }
    if out["p2"]["decision"].startswith("P2_RISK_HEAD_COLLAPSE"):
        decision = "V224_DIAGNOSTIC_COMPLETE_CASE_A_RISK_HEAD_COLLAPSE_LOCKED_TEST_BLOCKED"
        next_action = "Authorize v2.25A risk soft-label / scale distillation; do not expand v2.23."
    elif out["p3"]["decision"].startswith("P3_TRAINED_ACTION_CAN_BE_RESCUED"):
        decision = "V224_DIAGNOSTIC_COMPLETE_GATE_FAILURE_ACTION_RESCUABLE_LOCKED_TEST_BLOCKED"
        next_action = "Prioritize calibrated gate transfer before action expansion."
    elif "NOT_RESCUED" in out["p3"]["decision"]:
        decision = "V224_DIAGNOSTIC_COMPLETE_CASE_B_OR_C_ACTION_GATE_INTERACTION_FAIL_LOCKED_TEST_BLOCKED"
        next_action = "Consider safe-action distillation or close the current small-adapter form depending on P4."
    else:
        decision = "V224_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED_LOCKED_TEST_BLOCKED"
        next_action = "Manual review of P2-P4 required before v2.25."
    closeout = {"decision": decision, "next_action": next_action, **out, "locked_test_touched": False, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    wjson(args.out_dir / "v224_closeout.json", closeout)
    update_readme(args, closeout)
    with (args.out_dir / "status.txt").open("a") as f:
        f.write(f"v224_audit_done {decision} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print("V224_AUDIT_OK", decision, flush=True)


if __name__ == "__main__":
    main()
