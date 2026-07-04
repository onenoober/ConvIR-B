#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
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
from models.NoPostGatedLowbandConvIR import build_net as build_gated_net  # noqa: E402
from models.NoPostGatedLowbandConvIR import load_haze4k_partial  # noqa: E402

SEVERE = -0.20


def wjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def wtxt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def fnum(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def finite(xs: list[float]) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


def mean(xs: list[float]) -> float:
    ys = finite(xs)
    return sum(ys) / len(ys) if ys else float("nan")


def std(xs: list[float]) -> float:
    ys = finite(xs)
    if not ys:
        return float("nan")
    m = mean(ys)
    return math.sqrt(sum((x - m) ** 2 for x in ys) / len(ys))


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(finite(xs))
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


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


def average_precision(scores: list[float], labels: list[int]) -> float:
    pairs = sorted([(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)], reverse=True)
    total = sum(y for _, y in pairs)
    if total <= 0:
        return float("nan")
    tp, prec = 0, []
    for i, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            prec.append(tp / i)
    return sum(prec) / total


def ece(scores: list[float], labels: list[int], bins: int = 10) -> tuple[float, float]:
    pairs = [(max(0.0, min(1.0, s)), int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    if not pairs:
        return float("nan"), float("nan")
    total, out, mce = len(pairs), 0.0, 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sub = [(s, y) for s, y in pairs if lo <= s < hi or (i == bins - 1 and lo <= s <= hi)]
        if not sub:
            continue
        conf = mean([s for s, _ in sub])
        acc = mean([float(y) for _, y in sub])
        gap = abs(conf - acc)
        out += len(sub) / total * gap
        mce = max(mce, gap)
    return out, mce


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
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        risk_loss_weight=args.risk_loss_weight,
        gate_mean_weight=0.0,
        action_l1_weight=0.0,
        strong_reference_psnr=args.strong_reference_psnr,
        seed=args.seed,
    )


def set_risk_scope(model: torch.nn.Module) -> dict[str, Any]:
    prefixes = ("nopost_gated_lowband_policy.",)
    trainable_names = []
    for name, param in model.named_parameters():
        is_policy = name.startswith(prefixes)
        is_action = ".action." in name
        trainable = is_policy and not is_action
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    model.eval()
    model.nopost_gated_lowband_policy.train()
    return {
        "scope": "risk_context_only_no_action",
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": trainable_names,
        "frozen_action_names": [n for n, _ in model.named_parameters() if n.startswith(prefixes) and ".action." in n],
    }


def preflight(args: argparse.Namespace, samples: list[v223.FoldSample], device: torch.device) -> dict[str, Any]:
    state = v222.load_state(args.checkpoint, "cpu")
    a0 = build_a0_net("base", "Haze4K", "original").to(device)
    a0.load_state_dict(state)
    a0.eval()
    model = build_gated_net("base", "Haze4K", "original", hidden_channels=args.hidden_channels, mid_grid=args.mid_grid, final_grid=args.final_grid, risk_gamma=args.risk_gamma, risk_bias=args.risk_bias).to(device)
    partial = load_haze4k_partial(model, state)
    scope = set_risk_scope(model)
    model.eval()
    sample = samples[0]
    x = v222.image_tensor(sample.input_path, device)
    x, h, w = v222.pad_to(x, 32)
    with torch.no_grad():
        a0p = a0(x)[2][:, :, :h, :w]
        routep = model(x)[2][:, :, :h, :w]
    max_abs = float((a0p - routep).abs().max().cpu())
    payload = {
        "branch": cmd(["git", "branch", "--show-current"]),
        "commit": cmd(["git", "rev-parse", "--short", "HEAD"]),
        "anchor_commit": cmd(["git", "rev-parse", "--short", "github/codex/haze4k-official-arch-anchor"]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "partial_load": partial,
        "scope": scope,
        "zero_init_max_abs_vs_A0": max_abs,
        "forward_finite": bool(torch.isfinite(routep).all().item()),
        "locked_test_touched": False,
    }
    payload["pass"] = max_abs <= args.identity_tol and payload["forward_finite"] and payload["checkpoint_sha256"] == "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
    wjson(args.out_dir / "v225a_preflight.json", payload)
    return payload


def scale_from_prob(prob: torch.Tensor, gamma: float) -> torch.Tensor:
    return (1.0 - prob).clamp(0.0, 1.0) ** gamma


def train_fold(args: argparse.Namespace, fold: int, samples: list[v223.FoldSample], device: torch.device) -> dict[str, Any]:
    train, _ = v223.select_samples(samples, fold, args)
    a0, model, partial = v222.load_a0_and_route(base_args(args), device)
    del a0
    scope = set_risk_scope(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + fold * 1009)
    rows = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        losses: list[float] = []
        soft_losses: list[float] = []
        scale_losses: list[float] = []
        probs: list[float] = []
        for pos, idx in enumerate(order):
            sample = train[idx]
            x = v222.image_tensor(sample.input_path, device)
            y = v222.image_tensor(sample.label_path, device)
            x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + epoch * 10000 + pos)
            x, _, _ = v222.pad_to(x, 32)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            del pred, y
            t = model.nopost_gated_lowband_policy.last_tensors
            target_prob = torch.tensor([[float(sample.risk.get("unsafe_action_probability", 0.0))]], dtype=t["mid_unsafe_logit"].dtype, device=device)
            target_scale = torch.tensor([[float(sample.risk.get("v221_risk_scale", 1.0))]], dtype=t["mid_unsafe_logit"].dtype, device=device)
            mid_prob = t["mid_unsafe_prob"]
            final_prob = t["final_unsafe_prob"]
            mid_scale = t["mid_scale"]
            final_scale = t["final_scale"]
            soft = F.binary_cross_entropy_with_logits(t["mid_unsafe_logit"], target_prob)
            soft = soft + F.binary_cross_entropy_with_logits(t["final_unsafe_logit"], target_prob)
            scale_loss = F.mse_loss(mid_scale, target_scale) + F.mse_loss(final_scale, target_scale)
            consistency = F.mse_loss(mid_prob, final_prob.detach()) + F.mse_loss(final_prob, mid_prob.detach())
            loss = args.soft_prob_weight * soft + args.scale_distill_weight * scale_loss + args.mid_final_consistency_weight * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip_norm)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            soft_losses.append(float(soft.detach().cpu()))
            scale_losses.append(float(scale_loss.detach().cpu()))
            probs.append(float(((mid_prob + final_prob) * 0.5).mean().detach().cpu()))
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_count": len(train),
            "loss": mean(losses),
            "soft_bce": mean(soft_losses),
            "scale_mse": mean(scale_losses),
            "mean_trained_probability": mean(probs),
        }
        rows.append(row)
        print(f"V225A_TRAIN fold={fold} epoch={epoch}/{args.epochs} loss={row['loss']:.6f} prob={row['mean_trained_probability']:.4f}", flush=True)
    fold_dir = args.out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = fold_dir / f"v225a_fold{fold}_risk_context_Final.pkl"
    torch.save({"model": model.state_dict(), "scope": scope, "partial_load": partial}, ckpt)
    wcsv(fold_dir / f"v225a_fold{fold}_train_history.csv", rows)
    return {"fold": fold, "checkpoint": str(ckpt), "scope": scope, "partial_load": partial, "history_last": rows[-1], "train_seconds": time.time() - start}


def eval_fold(args: argparse.Namespace, fold: int, checkpoint: Path, samples: list[v223.FoldSample], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, eval_samples = v223.select_samples(samples, fold, args)
    _, model, partial = v222.load_a0_and_route(base_args(args), device)
    model.load_state_dict(v222.load_state(checkpoint, device))
    model.eval()
    rows = []
    with torch.no_grad():
        for sample in v223.to_v222_samples(eval_samples):
            x = v222.image_tensor(sample.input_path, device)
            y = v222.image_tensor(sample.label_path, device)
            x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + v222.idx_hash(sample.name))
            x, h, w = v222.pad_to(x, 32)
            _ = model(x)
            t = model.nopost_gated_lowband_policy.last_tensors
            mid = float(t["mid_unsafe_prob"].mean().cpu())
            final = float(t["final_unsafe_prob"].mean().cpu())
            prob = (mid + final) * 0.5
            target_prob = float(sample.risk.get("unsafe_action_probability", float("nan")))
            label = int(float(sample.risk.get("unsafe_action_label", 0.0)))
            rows.append({
                "fold": fold,
                "name": sample.name,
                "unsafe_action_label": label,
                "v221_unsafe_action_probability": target_prob,
                "v221_risk_scale": sample.risk.get("v221_risk_scale", float("nan")),
                "trained_mid_unsafe_prob": mid,
                "trained_final_unsafe_prob": final,
                "trained_mean_unsafe_prob": prob,
                "trained_mid_scale": float(t["mid_scale"].mean().cpu()),
                "trained_final_scale": float(t["final_scale"].mean().cpu()),
            })
    fold_dir = args.out_dir / f"fold{fold}"
    wcsv(fold_dir / f"v225a_fold{fold}_risk_eval.csv", rows)
    labels = [int(r["unsafe_action_label"]) for r in rows]
    scores = [float(r["trained_mean_unsafe_prob"]) for r in rows]
    targets = [float(r["v221_unsafe_action_probability"]) for r in rows]
    ee, mm = ece(scores, labels, 10)
    summary = {
        "fold": fold,
        "count": len(rows),
        "partial_load_loaded_count": partial["loaded_count"],
        "label_base_rate": mean([float(x) for x in labels]),
        "trained_prob_mean": mean(scores),
        "trained_prob_std": std(scores),
        "trained_prob_min": min(finite(scores)) if finite(scores) else float("nan"),
        "trained_prob_max": max(finite(scores)) if finite(scores) else float("nan"),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "ece10": ee,
        "mce10": mm,
        "target_prob_mae": mean([abs(s - t) for s, t in zip(scores, targets) if math.isfinite(t)]),
    }
    wjson(fold_dir / f"v225a_fold{fold}_risk_summary.json", summary)
    return summary, rows


def aggregate(args: argparse.Namespace, fold_summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(r["unsafe_action_label"]) for r in rows]
    scores = [float(r["trained_mean_unsafe_prob"]) for r in rows]
    targets = [float(r["v221_unsafe_action_probability"]) for r in rows]
    ee, mm = ece(scores, labels, 10)
    summary = {
        "fold_count": len(fold_summaries),
        "count": len(rows),
        "label_base_rate": mean([float(x) for x in labels]),
        "trained_prob_mean": mean(scores),
        "trained_prob_std": std(scores),
        "trained_prob_min": min(finite(scores)) if finite(scores) else float("nan"),
        "trained_prob_max": max(finite(scores)) if finite(scores) else float("nan"),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "ece10": ee,
        "mce10": mm,
        "target_prob_mae": mean([abs(s - t) for s, t in zip(scores, targets) if math.isfinite(t)]),
        "fold_summaries": fold_summaries,
        "locked_test_touched": False,
    }
    gate = {
        "prob_std": summary["trained_prob_std"] >= args.min_prob_std,
        "roc_auc": summary["roc_auc"] >= args.min_roc_auc,
        "average_precision": summary["average_precision"] >= args.min_average_precision,
        "ece10": summary["ece10"] <= args.max_ece10,
        "target_prob_mae": summary["target_prob_mae"] <= args.max_target_prob_mae,
        "locked_test_untouched": True,
    }
    return {"summary": summary, "gate": {"pass": all(gate.values()), "checks": gate, "locked_test_allowed": False}}


def update_readme(args: argparse.Namespace, closeout: dict[str, Any]) -> None:
    s = closeout.get("summary", {})
    lines = [
        "# Haze4K v2.25A NoPost Risk Soft-Label / Scale Distillation Evidence",
        "",
        f"Status: {closeout['decision']}",
        "",
        "Risk-head calibration screen only. Action heads are frozen; locked Haze4K test is untouched.",
        "",
        "## Summary",
        "",
        f"- ROC-AUC: `{s.get('roc_auc')}`",
        f"- AP: `{s.get('average_precision')}`",
        f"- ECE10: `{s.get('ece10')}`",
        f"- probability std: `{s.get('trained_prob_std')}`",
        f"- target probability MAE: `{s.get('target_prob_mae')}`",
        f"- locked test touched: `{closeout.get('locked_test_touched')}`",
    ]
    wtxt(args.out_dir / "README.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--v221-metrics-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--train-samples-per-fold", type=int, default=384)
    ap.add_argument("--eval-samples-per-fold", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--risk-gamma", type=float, default=0.5)
    ap.add_argument("--risk-bias", type=float, default=-1.5)
    ap.add_argument("--v221-variant", default="V221_risk_temperature_gamma0p50")
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=0.01)
    ap.add_argument("--risk-loss-weight", type=float, default=1.0)
    ap.add_argument("--soft-prob-weight", type=float, default=1.0)
    ap.add_argument("--scale-distill-weight", type=float, default=2.0)
    ap.add_argument("--mid-final-consistency-weight", type=float, default=0.1)
    ap.add_argument("--strong-reference-psnr", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=225)
    ap.add_argument("--min-prob-std", type=float, default=0.05)
    ap.add_argument("--min-roc-auc", type=float, default=0.85)
    ap.add_argument("--min-average-precision", type=float, default=0.45)
    ap.add_argument("--max-ece10", type=float, default=0.12)
    ap.add_argument("--max-target-prob-mae", type=float, default=0.20)
    args = ap.parse_args()
    args.folds_list = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = args.out_dir / "status.txt"
    with status.open("a") as f:
        f.write(f"v225a_start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = v223.load_split_samples(args)
    pre = preflight(args, samples, device)
    if not pre["pass"]:
        closeout = {"decision": "V225A_PREFLIGHT_FAILED_ENGINEERING", "preflight": pre, "locked_test_touched": False}
        wjson(args.out_dir / "v225a_closeout.json", closeout)
        update_readme(args, closeout)
        raise SystemExit(2)
    fold_summaries, all_rows, train_results = [], [], []
    for fold in args.folds_list:
        train_results.append(train_fold(args, fold, samples, device))
        summary, rows = eval_fold(args, fold, Path(train_results[-1]["checkpoint"]), samples, device)
        fold_summaries.append(summary)
        all_rows.extend(rows)
    agg = aggregate(args, fold_summaries, all_rows)
    wcsv(args.out_dir / "v225a_oof_risk_eval.csv", all_rows)
    wcsv(args.out_dir / "v225a_fold_summary.csv", fold_summaries)
    wjson(args.out_dir / "v225a_oof_summary.json", agg["summary"])
    wjson(args.out_dir / "v225a_gate.json", agg["gate"])
    decision = "V225A_RISK_CALIBRATION_GATE_PASS_ACTION_STAGE_REVIEW_ONLY" if agg["gate"]["pass"] else "V225A_RISK_CALIBRATION_GATE_FAIL_NORMAL_PAUSE"
    closeout = {
        "decision": decision,
        "preflight_pass": pre["pass"],
        "train_results": train_results,
        "summary": agg["summary"],
        "gate": agg["gate"],
        "locked_test_touched": False,
        "next_action": "If pass, run post-train factorial rescue before any action joint train; if fail, stop v2.25A and redesign calibration.",
    }
    wjson(args.out_dir / "v225a_closeout.json", closeout)
    update_readme(args, closeout)
    with status.open("a") as f:
        f.write(f"v225a_done {decision} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print("V225A_OK", decision, flush=True)


if __name__ == "__main__":
    main()
