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
import torch.nn as nn
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for p in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from experience_docx.tools import nopost_lowband_v222_n3_microfit as v222  # noqa: E402
from experience_docx.tools import nopost_lowband_v223_oof_train as v223  # noqa: E402


def wjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def wtxt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


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


def cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def average_precision_buggy(scores: list[float], labels: list[int]) -> float:
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


def average_precision_tie_aware(scores: list[float], labels: list[int]) -> float:
    pairs = sorted([(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)], key=lambda x: x[0], reverse=True)
    total_pos = sum(y for _, y in pairs)
    if total_pos <= 0:
        return float("nan")
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        group_pos = 0
        group_total = 0
        while i < len(pairs) and pairs[i][0] == score:
            group_pos += pairs[i][1]
            group_total += 1
            i += 1
        tp += group_pos
        fp += group_total - group_pos
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


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


def logit(p: float) -> float:
    p = max(1e-4, min(1.0 - 1e-4, p))
    return math.log(p / (1.0 - p))


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
        risk_loss_weight=1.0,
        gate_mean_weight=0.0,
        action_l1_weight=0.0,
        strong_reference_psnr=args.strong_reference_psnr,
        seed=args.seed,
    )


def set_risk_scope(model: torch.nn.Module) -> dict[str, Any]:
    trainable_names = []
    frozen_action_names = []
    for name, param in model.named_parameters():
        is_policy = name.startswith("nopost_gated_lowband_policy.")
        is_action = ".action." in name
        trainable = is_policy and not is_action
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
        elif is_policy and is_action:
            frozen_action_names.append(name)
    model.eval()
    model.nopost_gated_lowband_policy.train()
    return {
        "scope": "risk_context_only_no_action",
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": trainable_names,
        "frozen_action_names": frozen_action_names,
    }


def load_route_model(args: argparse.Namespace, device: torch.device, checkpoint: Path | None = None) -> torch.nn.Module:
    a0, model, _partial = v222.load_a0_and_route(base_args(args), device)
    del a0
    if checkpoint is not None:
        model.load_state_dict(v222.load_state(checkpoint, device))
    model.eval()
    return model


def load_samples(args: argparse.Namespace) -> list[v223.FoldSample]:
    return v223.load_split_samples(args)


def active_samples(args: argparse.Namespace) -> list[v223.FoldSample]:
    allowed = set(args.folds_list)
    return [s for s in load_samples(args) if int(s.fold) in allowed]


def subset_per_fold(samples: list[v223.FoldSample], limit: int) -> list[v223.FoldSample]:
    if limit <= 0:
        return sorted(samples, key=lambda s: (s.fold, s.name))
    out: list[v223.FoldSample] = []
    for fold in sorted({s.fold for s in samples}):
        rows = sorted([s for s in samples if s.fold == fold], key=lambda s: s.name)
        out.extend(rows[:limit])
    return out


def phase_p0_metric_fix(args: argparse.Namespace) -> dict[str, Any]:
    rows = rcsv(args.v225a_evid / "v225a_oof_risk_eval.csv")
    labels = [int(fnum(r["unsafe_action_label"], 0.0)) for r in rows]
    scores = [fnum(r["trained_mean_unsafe_prob"]) for r in rows]
    targets = [fnum(r["v221_unsafe_action_probability"]) for r in rows]
    ee, mm = ece(scores, labels, 10)
    summary = {
        "count": len(rows),
        "label_base_rate": mean([float(x) for x in labels]),
        "trained_prob_mean": mean(scores),
        "trained_prob_std": std(scores),
        "trained_prob_min": min(finite(scores)) if finite(scores) else float("nan"),
        "trained_prob_max": max(finite(scores)) if finite(scores) else float("nan"),
        "roc_auc": roc_auc(scores, labels),
        "average_precision_buggy": average_precision_buggy(scores, labels),
        "average_precision_tie_aware": average_precision_tie_aware(scores, labels),
        "ece10": ee,
        "mce10": mm,
        "target_prob_mae": mean([abs(s - t) for s, t in zip(scores, targets) if math.isfinite(t)]),
    }
    fold_rows = []
    for fold in sorted({int(fnum(r["fold"], -1)) for r in rows}):
        sub = [r for r in rows if int(fnum(r["fold"], -999)) == fold]
        flabels = [int(fnum(r["unsafe_action_label"], 0.0)) for r in sub]
        fscores = [fnum(r["trained_mean_unsafe_prob"]) for r in sub]
        fold_rows.append(
            {
                "fold": fold,
                "count": len(sub),
                "label_base_rate": mean([float(x) for x in flabels]),
                "trained_prob_std": std(fscores),
                "roc_auc": roc_auc(fscores, flabels),
                "average_precision_buggy": average_precision_buggy(fscores, flabels),
                "average_precision_tie_aware": average_precision_tie_aware(fscores, flabels),
            }
        )
    sanity_labels = [1] * 15 + [0] * 145
    sanity_scores = [0.2] * 160
    sanity = {
        "constant_score": 0.2,
        "count": 160,
        "positive_count": 15,
        "base_rate": 15 / 160,
        "average_precision_buggy": average_precision_buggy(sanity_scores, sanity_labels),
        "average_precision_tie_aware": average_precision_tie_aware(sanity_scores, sanity_labels),
        "expected_tie_aware_ap": 15 / 160,
    }
    invalid = summary["average_precision_tie_aware"] < summary["average_precision_buggy"] - 0.10
    payload = {
        "phase": "P0_metric_fix",
        "branch": cmd(["git", "branch", "--show-current"]),
        "commit": cmd(["git", "rev-parse", "--short", "HEAD"]),
        "v225a_evid": str(args.v225a_evid),
        "summary": summary,
        "folds": fold_rows,
        "decision": "V226_P0_V225A_AP_PASS_INVALID_DIAGNOSTIC_ARTIFACT" if invalid else "V226_P0_AP_RECOMPUTED_NO_INVALIDATION",
        "locked_test_touched": False,
    }
    wjson(args.out_dir / "v226_p0_v225a_recomputed_metrics.json", payload)
    wjson(args.out_dir / "v226_p0_constant_score_sanity.json", sanity)
    lines = [
        "# v2.26 P0 metric fix report",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        f"- v2.25A OOF count: `{summary['count']}`",
        f"- probability std: `{summary['trained_prob_std']}`",
        f"- ROC-AUC: `{summary['roc_auc']}`",
        f"- AP old tuple-sort implementation: `{summary['average_precision_buggy']}`",
        f"- AP tie-aware implementation: `{summary['average_precision_tie_aware']}`",
        f"- label base rate: `{summary['label_base_rate']}`",
        "",
        "The gate failure remains valid because probability spread, ROC-AUC, and target-probability MAE still fail.",
        "The old AP pass is not positive evidence when scores are tied or constant.",
    ]
    wtxt(args.out_dir / "v226_p0_metric_fix_report.md", "\n".join(lines))
    return payload


def phase_p1_join_replay(args: argparse.Namespace) -> dict[str, Any]:
    split_rows = rcsv(args.split_csv)
    v221_rows = [r for r in rcsv(args.v221_metrics_csv) if r.get("variant") == args.v221_variant]
    v221_by_name = {r["name"]: r for r in v221_rows}
    v225_rows = rcsv(args.v225a_evid / "v225a_oof_risk_eval.csv")
    missing_split = [r["name"] for r in split_rows if r["name"] not in v221_by_name]
    missing_v225 = [r["name"] for r in v225_rows if r["name"] not in v221_by_name]
    labels = [int(fnum(r["unsafe_action_label"], 0.0)) for r in v225_rows]
    probs = [fnum(r["v221_unsafe_action_probability"]) for r in v225_rows]
    ee, mm = ece(probs, labels, 10)
    hist_rows = []
    fold_rows = []
    for fold in sorted({int(fnum(r["fold"], -1)) for r in v225_rows}):
        sub = [r for r in v225_rows if int(fnum(r["fold"], -999)) == fold]
        flabels = [int(fnum(r["unsafe_action_label"], 0.0)) for r in sub]
        fprobs = [fnum(r["v221_unsafe_action_probability"]) for r in sub]
        fee, fmm = ece(fprobs, flabels, 10)
        fold_rows.append(
            {
                "fold": fold,
                "count": len(sub),
                "v221_target_prob_mean": mean(fprobs),
                "v221_target_prob_std": std(fprobs),
                "v221_target_prob_min": min(finite(fprobs)) if finite(fprobs) else float("nan"),
                "v221_target_prob_max": max(finite(fprobs)) if finite(fprobs) else float("nan"),
                "unsafe_action_label_base_rate": mean([float(x) for x in flabels]),
                "v221_replay_probability_auc": roc_auc(fprobs, flabels),
                "v221_replay_probability_ap": average_precision_tie_aware(fprobs, flabels),
                "v221_replay_probability_ece10": fee,
                "v221_replay_probability_mce10": fmm,
            }
        )
        for b in range(10):
            lo, hi = b / 10.0, (b + 1) / 10.0
            vals = [p for p in fprobs if lo <= p < hi or (b == 9 and lo <= p <= hi)]
            hist_rows.append({"fold": fold, "bin": b, "lo": lo, "hi": hi, "count": len(vals)})
    summary = {
        "phase": "P1_join_replay_audit",
        "split_csv": str(args.split_csv),
        "v221_metrics_csv": str(args.v221_metrics_csv),
        "v221_variant": args.v221_variant,
        "split_row_count": len(split_rows),
        "v221_variant_row_count": len(v221_rows),
        "v225a_exact_eval_count": len(v225_rows),
        "split_join_exact_match_count": len(split_rows) - len(missing_split),
        "split_join_missing_count": len(missing_split),
        "v225a_eval_join_exact_match_count": len(v225_rows) - len(missing_v225),
        "v225a_eval_join_missing_count": len(missing_v225),
        "v221_target_prob_mean": mean(probs),
        "v221_target_prob_std": std(probs),
        "v221_target_prob_min": min(finite(probs)) if finite(probs) else float("nan"),
        "v221_target_prob_max": max(finite(probs)) if finite(probs) else float("nan"),
        "unsafe_action_label_base_rate": mean([float(x) for x in labels]),
        "v221_replay_probability_auc": roc_auc(probs, labels),
        "v221_replay_probability_ap": average_precision_tie_aware(probs, labels),
        "v221_replay_probability_ece10": ee,
        "v221_replay_probability_mce10": mm,
        "folds": fold_rows,
        "decision": "V226_P1_JOIN_REPLAY_PASS" if not missing_split and not missing_v225 and roc_auc(probs, labels) >= 0.85 else "V226_P1_JOIN_REPLAY_REVIEW_REQUIRED",
        "locked_test_touched": False,
    }
    wjson(args.out_dir / "v226_p1_join_replay_audit.json", summary)
    wcsv(args.out_dir / "v226_p1_target_probability_histogram_per_fold.csv", hist_rows)
    mismatch_rows = [{"source": "split", "name": n} for n in missing_split[:200]] + [{"source": "v225a_eval", "name": n} for n in missing_v225[:200]]
    wcsv(args.out_dir / "v226_p1_join_mismatches.csv", mismatch_rows)
    return summary


def tensor_channel_stats(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().float()
    flat = x.flatten(2)
    return torch.cat([flat.mean(dim=2), flat.std(dim=2, unbiased=False)], dim=1).flatten(1)


class FeatureCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.captures: dict[str, torch.Tensor] = {}
        p = model.nopost_gated_lowband_policy
        self.handles = [
            p.mid_policy.context[0].register_forward_pre_hook(self._pre("mid_ll_grid")),
            p.final_policy.context[0].register_forward_pre_hook(self._pre("final_context_input")),
            p.mid_policy.risk[-1].register_forward_pre_hook(self._pre("mid_risk_hidden")),
            p.final_policy.risk[-1].register_forward_pre_hook(self._pre("final_risk_hidden")),
        ]

    def _pre(self, name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self.captures[name] = inputs[0].detach()

        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    def vectors(self) -> dict[str, torch.Tensor]:
        final_context = self.captures["final_context_input"]
        final_ll_grid = final_context[:, :32]
        return {
            "A_mid_ll_pooled": tensor_channel_stats(self.captures["mid_ll_grid"]),
            "B_final_ll_pooled": tensor_channel_stats(final_ll_grid),
            "C_final_ll_mid_global_context": tensor_channel_stats(final_context),
            "D_risk_hidden_before_final_1x1": torch.cat(
                [
                    self.captures["mid_risk_hidden"].detach().float().flatten(1),
                    self.captures["final_risk_hidden"].detach().float().flatten(1),
                ],
                dim=1,
            ),
        }


def numeric_v221_features(args: argparse.Namespace) -> tuple[dict[str, list[float]], list[str]]:
    rows = [r for r in rcsv(args.v221_metrics_csv) if r.get("variant") == args.v221_variant]
    excluded = {"name", "variant", "gate_kind", "gate_threshold", "unsafe_action_label"}
    cols: list[str] = []
    for row in rows[:50]:
        for key, value in row.items():
            if key in excluded:
                continue
            try:
                float(value)
            except Exception:
                continue
            if key not in cols:
                cols.append(key)
    out: dict[str, list[float]] = {}
    for row in rows:
        vals = [fnum(row.get(c), 0.0) for c in cols]
        out[row["name"]] = vals
    return out, cols


def v225a_checkpoint_for_fold(args: argparse.Namespace, fold: int) -> Path | None:
    path = args.v225a_evid / f"fold{fold}" / f"v225a_fold{fold}_risk_context_Final.pkl"
    return path if path.is_file() else None


def extract_feature_table(args: argparse.Namespace, samples: list[v223.FoldSample], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    v221_features, v221_cols = numeric_v221_features(args)
    feature_lists: dict[str, list[torch.Tensor]] = {
        "A_mid_ll_pooled": [],
        "B_final_ll_pooled": [],
        "C_final_ll_mid_global_context": [],
        "D_risk_hidden_before_final_1x1": [],
        "E_v221_cached_scalar_positive_control": [],
    }
    labels: list[int] = []
    target_probs: list[float] = []
    folds: list[int] = []
    names: list[str] = []
    model_by_fold: dict[int, torch.nn.Module] = {}
    for i, sample in enumerate(samples):
        fold = int(sample.fold)
        if fold not in model_by_fold:
            model_by_fold[fold] = load_route_model(args, device, v225a_checkpoint_for_fold(args, fold))
        model = model_by_fold[fold]
        capture = FeatureCapture(model)
        try:
            with torch.no_grad():
                x = v222.image_tensor(sample.input_path, device)
                x, _ = v222.crop_pair(x, x, args.crop_size, args.seed + fold * 1000000 + v222.idx_hash(sample.name))
                x, _, _ = v222.pad_to(x, 32)
                _ = model(x)
                vecs = capture.vectors()
        finally:
            capture.close()
        for key in ("A_mid_ll_pooled", "B_final_ll_pooled", "C_final_ll_mid_global_context", "D_risk_hidden_before_final_1x1"):
            feature_lists[key].append(vecs[key].cpu().squeeze(0))
        feature_lists["E_v221_cached_scalar_positive_control"].append(torch.tensor(v221_features.get(sample.name, [0.0] * len(v221_cols)), dtype=torch.float32))
        labels.append(int(fnum(sample.risk.get("unsafe_action_label"), 0.0)))
        target_probs.append(fnum(sample.risk.get("unsafe_action_probability"), 0.0))
        folds.append(fold)
        names.append(sample.name)
        if (i + 1) % 100 == 0:
            print(f"V226_P2_EXTRACT progress={i + 1}/{len(samples)}", flush=True)
    features = {k: torch.stack(v, dim=0).float() for k, v in feature_lists.items()}
    features["F_current_features_plus_v221_scalar"] = torch.cat(
        [
            features["A_mid_ll_pooled"],
            features["B_final_ll_pooled"],
            features["C_final_ll_mid_global_context"],
            features["D_risk_hidden_before_final_1x1"],
            features["E_v221_cached_scalar_positive_control"],
        ],
        dim=1,
    )
    meta = {
        "labels": labels,
        "target_probs": target_probs,
        "folds": folds,
        "names": names,
        "v221_feature_columns": v221_cols,
    }
    return features, meta


class ProbeNet(nn.Module):
    def __init__(self, in_dim: int, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, 1)
        elif kind == "mlp":
            hidden = min(128, max(16, in_dim // 2))
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()


def train_probe_oof(X: torch.Tensor, labels: list[int], target_probs: list[float], folds: list[int], kind: str, epochs: int, seed: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    fold_tensor = torch.tensor(folds, dtype=torch.long, device=device)
    scores = torch.full_like(y, float("nan"))
    for fold in sorted(set(folds)):
        train_idx = (fold_tensor != fold).nonzero(as_tuple=False).flatten()
        val_idx = (fold_tensor == fold).nonzero(as_tuple=False).flatten()
        Xtr = X[train_idx.cpu()].to(device)
        Xva = X[val_idx.cpu()].to(device)
        mu = Xtr.mean(dim=0, keepdim=True)
        sig = Xtr.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sig
        Xva = (Xva - mu) / sig
        torch.manual_seed(seed + fold * 17 + (0 if kind == "linear" else 10000))
        model = ProbeNet(X.shape[1], kind).to(device)
        pos = y[train_idx].sum().clamp_min(1.0)
        neg = (len(train_idx) - pos).clamp_min(1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).detach())
        opt = torch.optim.AdamW(model.parameters(), lr=0.01 if kind == "linear" else 0.003, weight_decay=1e-3)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr), y[train_idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores[val_idx] = torch.sigmoid(model(Xva))
    score_list = [float(x) for x in scores.detach().cpu()]
    ee, mm = ece(score_list, labels, 10)
    return {
        "probe_type": kind,
        "count": len(labels),
        "feature_dim": int(X.shape[1]),
        "label_base_rate": mean([float(x) for x in labels]),
        "prob_std": std(score_list),
        "roc_auc": roc_auc(score_list, labels),
        "average_precision": average_precision_tie_aware(score_list, labels),
        "ece10": ee,
        "mce10": mm,
        "target_prob_mae": mean([abs(s - t) for s, t in zip(score_list, target_probs)]),
    }


def phase_p2_probe(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    samples = subset_per_fold(active_samples(args), args.probe_samples_per_fold)
    features, meta = extract_feature_table(args, samples, device)
    rows: list[dict[str, Any]] = []
    for feature_name, X in features.items():
        for kind in ("linear", "mlp"):
            result = train_probe_oof(X, meta["labels"], meta["target_probs"], meta["folds"], kind, args.probe_epochs, args.seed)
            result["feature_set"] = feature_name
            rows.append(result)
            print(
                "V226_P2_PROBE "
                f"feature={feature_name} kind={kind} auc={result['roc_auc']:.4f} "
                f"ap={result['average_precision']:.4f} std={result['prob_std']:.4f}",
                flush=True,
            )
    wcsv(args.out_dir / "v226_probe_feature_auc_table.csv", rows)
    current = [r for r in rows if str(r["feature_set"]).startswith(("A_", "B_", "C_", "D_"))]
    positive = [r for r in rows if str(r["feature_set"]).startswith("E_")]
    fused = [r for r in rows if str(r["feature_set"]).startswith("F_")]
    best_current = max(current, key=lambda r: fnum(r["roc_auc"])) if current else {}
    best_positive = max(positive, key=lambda r: fnum(r["roc_auc"])) if positive else {}
    best_fused = max(fused, key=lambda r: fnum(r["roc_auc"])) if fused else {}
    pos_ok = fnum(best_positive.get("roc_auc")) >= 0.85
    cur_auc = fnum(best_current.get("roc_auc"))
    if not pos_ok:
        decision = "V226_P2_POSITIVE_CONTROL_FAIL_REVIEW_PIPELINE"
    elif cur_auc <= 0.60:
        decision = "V226_P2_CURRENT_RISK_FEATURES_NO_SIGNAL_PAUSE_CURRENT_INPUT"
    elif cur_auc >= 0.75:
        decision = "V226_P2_CURRENT_RISK_FEATURES_HAVE_SIGNAL_CONTINUE_TRAINABILITY"
    else:
        decision = "V226_P2_CURRENT_RISK_FEATURES_INCONCLUSIVE_CONTINUE_CANARY"
    summary = {
        "phase": "P2_frozen_feature_probe",
        "sample_count": len(samples),
        "feature_rows": rows,
        "best_current_feature": best_current,
        "best_positive_control": best_positive,
        "best_fused": best_fused,
        "v221_feature_columns": meta["v221_feature_columns"],
        "decision": decision,
        "locked_test_touched": False,
    }
    wjson(args.out_dir / "v226_p2_probe_summary.json", summary)
    return summary


def select_canary(samples: list[v223.FoldSample], count: int) -> list[v223.FoldSample]:
    valid = [s for s in samples if "unsafe_action_label" in s.risk]
    pos = sorted([s for s in valid if int(fnum(s.risk.get("unsafe_action_label"), 0.0)) == 1], key=lambda s: (-fnum(s.risk.get("unsafe_action_probability"), 0.0), s.name))
    neg = sorted([s for s in valid if int(fnum(s.risk.get("unsafe_action_label"), 0.0)) == 0], key=lambda s: (fnum(s.risk.get("unsafe_action_probability"), 0.0), s.name))
    half = count // 2
    chosen = pos[:half] + neg[: count - half]
    if len(chosen) < count:
        chosen.extend([s for s in valid if s not in chosen][: count - len(chosen)])
    return sorted(chosen, key=lambda s: s.name)


def trainable_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def update_norm_rows(model: torch.nn.Module, before: dict[str, torch.Tensor], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for name, p in model.named_parameters():
        if name not in before:
            continue
        delta = p.detach().cpu() - before[name]
        rows.append(
            {
                "prefix": prefix,
                "param": name,
                "update_l2": float(torch.linalg.vector_norm(delta.float()).item()),
                "update_abs_mean": float(delta.float().abs().mean().item()),
                "final_l2": float(torch.linalg.vector_norm(p.detach().cpu().float()).item()),
            }
        )
    return rows


def risk_final_stats(model: torch.nn.Module) -> dict[str, float]:
    sd = dict(model.named_parameters())
    vals: dict[str, float] = {}
    for key in (
        "nopost_gated_lowband_policy.mid_policy.risk.3.weight",
        "nopost_gated_lowband_policy.mid_policy.risk.3.bias",
        "nopost_gated_lowband_policy.final_policy.risk.3.weight",
        "nopost_gated_lowband_policy.final_policy.risk.3.bias",
    ):
        p = sd[key].detach().float().cpu()
        vals[key.replace("nopost_gated_lowband_policy.", "").replace(".", "_") + "_mean"] = float(p.mean().item())
        vals[key.replace("nopost_gated_lowband_policy.", "").replace(".", "_") + "_std"] = float(p.std(unbiased=False).item())
    return vals


def loss_from_tensors(t: dict[str, torch.Tensor], sample: v223.FoldSample, mode: str, gamma: float) -> torch.Tensor:
    device = t["mid_unsafe_logit"].device
    target_prob = torch.tensor([[fnum(sample.risk.get("unsafe_action_probability"), 0.0)]], dtype=t["mid_unsafe_logit"].dtype, device=device)
    target_label = torch.tensor([[fnum(sample.risk.get("unsafe_action_label"), 0.0)]], dtype=t["mid_unsafe_logit"].dtype, device=device)
    target_scale = torch.tensor([[fnum(sample.risk.get("v221_risk_scale"), (1.0 - float(target_prob.item())) ** gamma)]], dtype=t["mid_unsafe_logit"].dtype, device=device)
    logits = (t["mid_unsafe_logit"], t["final_unsafe_logit"])
    if mode == "soft_bce_scale":
        soft = sum(F.binary_cross_entropy_with_logits(x, target_prob) for x in logits)
        scale = F.mse_loss(t["mid_scale"], target_scale) + F.mse_loss(t["final_scale"], target_scale)
        return soft + 2.0 * scale
    if mode == "bce_only":
        return sum(F.binary_cross_entropy_with_logits(x, target_label) for x in logits)
    if mode == "balanced_bce":
        weight = torch.where(target_label > 0.5, torch.tensor(3.0, device=device), torch.tensor(1.0, device=device))
        return sum(F.binary_cross_entropy_with_logits(x, target_label, weight=weight) for x in logits)
    if mode == "focal_bce":
        out = t["mid_unsafe_logit"].new_tensor(0.0)
        for x in logits:
            bce = F.binary_cross_entropy_with_logits(x, target_label, reduction="none")
            p = torch.sigmoid(x)
            pt = torch.where(target_label > 0.5, p, 1.0 - p)
            out = out + ((1.0 - pt) ** 2.0 * bce).mean()
        return out
    if mode == "logit_mse":
        target = torch.tensor([[logit(float(target_prob.item()))]], dtype=t["mid_unsafe_logit"].dtype, device=device)
        return sum(F.mse_loss(x, target) for x in logits)
    raise ValueError(mode)


def eval_risk_model(model: torch.nn.Module, samples: list[v223.FoldSample], args: argparse.Namespace, device: torch.device, tag: str) -> dict[str, Any]:
    labels, probs, targets, hidden_stds = [], [], [], []
    recorder = FeatureCapture(model)
    try:
        with torch.no_grad():
            for sample in samples:
                x = v222.image_tensor(sample.input_path, device)
                x, _y = v222.crop_pair(x, x, args.crop_size, args.seed + v222.idx_hash(sample.name))
                x, _, _ = v222.pad_to(x, 32)
                _ = model(x)
                t = model.nopost_gated_lowband_policy.last_tensors
                prob = float(((t["mid_unsafe_prob"] + t["final_unsafe_prob"]) * 0.5).mean().cpu())
                labels.append(int(fnum(sample.risk.get("unsafe_action_label"), 0.0)))
                targets.append(fnum(sample.risk.get("unsafe_action_probability"), 0.0))
                probs.append(prob)
                vec = recorder.vectors()["D_risk_hidden_before_final_1x1"]
                hidden_stds.append(float(vec.std(unbiased=False).cpu()))
    finally:
        recorder.close()
    return {
        f"{tag}_count": len(samples),
        f"{tag}_auc": roc_auc(probs, labels),
        f"{tag}_ap": average_precision_tie_aware(probs, labels),
        f"{tag}_prob_mean": mean(probs),
        f"{tag}_prob_std": std(probs),
        f"{tag}_target_mae": mean([abs(p - t) for p, t in zip(probs, targets)]),
        f"{tag}_hidden_std_mean": mean(hidden_stds),
    }


def apply_tiny_nonzero_init(model: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for module in (
        model.nopost_gated_lowband_policy.mid_policy.risk[-1],
        model.nopost_gated_lowband_policy.final_policy.risk[-1],
    ):
        nn.init.normal_(module.weight, mean=0.0, std=1e-3)


def train_risk_only(
    args: argparse.Namespace,
    train_samples: list[v223.FoldSample],
    eval_samples: list[v223.FoldSample],
    device: torch.device,
    variant: dict[str, Any],
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model = load_route_model(args, device, None)
    scope = set_risk_scope(model)
    if variant.get("tiny_nonzero_init"):
        apply_tiny_nonzero_init(model, args.seed)
    before = trainable_snapshot(model)
    before_stats = risk_final_stats(model)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(variant["lr"]),
        weight_decay=float(variant["weight_decay"]),
    )
    curves = []
    for epoch in range(1, int(variant["epochs"]) + 1):
        losses = []
        random.Random(args.seed + epoch).shuffle(train_samples)
        for sample in train_samples:
            x = v222.image_tensor(sample.input_path, device)
            x, _y = v222.crop_pair(x, x, args.crop_size, args.seed + v222.idx_hash(sample.name))
            x, _, _ = v222.pad_to(x, 32)
            opt.zero_grad(set_to_none=True)
            _ = model(x)
            loss = loss_from_tensors(model.nopost_gated_lowband_policy.last_tensors, sample, str(variant["loss_mode"]), args.risk_gamma)
            loss.backward()
            if variant["grad_clip_norm"] is not None and math.isfinite(float(variant["grad_clip_norm"])):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(variant["grad_clip_norm"]))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = eval_risk_model(model, train_samples, args, device, "train")
        val_metrics = eval_risk_model(model, eval_samples, args, device, "val") if eval_samples else {}
        row = {"prefix": prefix, "epoch": epoch, "loss": mean(losses)}
        row.update(train_metrics)
        row.update(val_metrics)
        curves.append(row)
        print(
            "V226_RISK_TRAIN "
            f"{prefix} epoch={epoch}/{variant['epochs']} loss={row['loss']:.5f} "
            f"train_auc={row.get('train_auc')} val_auc={row.get('val_auc')}",
            flush=True,
        )
    final_stats = risk_final_stats(model)
    summary = {"prefix": prefix, "scope": scope, **variant, **curves[-1], **{f"before_{k}": v for k, v in before_stats.items()}, **{f"after_{k}": v for k, v in final_stats.items()}}
    update_rows = update_norm_rows(model, before, prefix)
    return summary, curves, update_rows


def phase_p3_canary(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    samples = active_samples(args)
    summaries, curves, updates = [], [], []
    for count in [int(x) for x in args.canary_sizes.split(",") if x.strip()]:
        chosen = select_canary(samples, count)
        variant = {
            "loss_mode": "soft_bce_scale",
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "epochs": args.canary_epochs,
            "tiny_nonzero_init": False,
        }
        summary, curve, update = train_risk_only(args, chosen, [], device, variant, f"canary{count}")
        summary["canary_count"] = count
        summary["pass"] = (
            fnum(summary.get("train_prob_std")) >= 0.10
            and fnum(summary.get("train_auc")) >= 0.95
            and fnum(summary.get("train_target_mae")) <= 0.08
        )
        summaries.append(summary)
        curves.extend(curve)
        updates.extend(update)
    decision = "V226_P3_CANARY_PASS" if any(s.get("pass") for s in summaries) else "V226_P3_CANARY_FAIL_TRAINABILITY_REVIEW"
    wcsv(args.out_dir / "v226_canary_overfit_curve.csv", curves)
    wcsv(args.out_dir / "v226_p3_param_update_norm.csv", updates)
    payload = {"phase": "P3_tiny_canary_overfit", "summaries": summaries, "decision": decision, "locked_test_touched": False}
    wjson(args.out_dir / "v226_p3_canary_summary.json", payload)
    return payload


def phase_p4_ablation(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    samples = active_samples(args)
    train_pool = [s for s in samples if s.fold != args.p4_eval_fold]
    val_pool = [s for s in samples if s.fold == args.p4_eval_fold]
    rng = random.Random(args.seed + 2264)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    train_samples = train_pool[: args.p4_train_samples]
    eval_samples = val_pool[: args.p4_eval_samples]
    variants = [
        {"name": "baseline_soft_bce_scale_clip_wd_zero", "loss_mode": "soft_bce_scale", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "bce_only", "loss_mode": "bce_only", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "no_grad_clip", "loss_mode": "soft_bce_scale", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": None, "tiny_nonzero_init": False},
        {"name": "lr_3e_4", "loss_mode": "soft_bce_scale", "lr": 3e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "lr_1e_3", "loss_mode": "soft_bce_scale", "lr": 1e-3, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "weight_decay_0", "loss_mode": "soft_bce_scale", "lr": 1e-4, "weight_decay": 0.0, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "tiny_nonzero_init", "loss_mode": "soft_bce_scale", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": True},
        {"name": "logit_target_mse", "loss_mode": "logit_mse", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "class_balanced_bce", "loss_mode": "balanced_bce", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
        {"name": "focal_bce", "loss_mode": "focal_bce", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
    ]
    summaries, curves, updates = [], [], []
    for variant in variants:
        variant = {**variant, "epochs": args.p4_epochs}
        summary, curve, update = train_risk_only(args, list(train_samples), list(eval_samples), device, variant, str(variant["name"]))
        summaries.append(summary)
        curves.extend(curve)
        updates.extend(update)
    wcsv(args.out_dir / "v226_optimizer_ablation_summary.csv", summaries)
    wcsv(args.out_dir / "v226_p4_optimizer_ablation_curve.csv", curves)
    wcsv(args.out_dir / "v226_p4_param_update_norm_by_variant.csv", updates)
    best = max(summaries, key=lambda r: (fnum(r.get("val_auc")), fnum(r.get("val_prob_std")))) if summaries else {}
    decision = "V226_P4_OPTIMIZATION_CAN_MOVE_RISK_HEAD" if fnum(best.get("val_auc")) >= 0.75 and fnum(best.get("val_prob_std")) >= 0.05 else "V226_P4_NO_MINIMAL_OPTIMIZATION_RESCUE"
    payload = {
        "phase": "P4_minimal_optimizer_ablation",
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "best_variant": best,
        "decision": decision,
        "locked_test_touched": False,
    }
    wjson(args.out_dir / "v226_p4_ablation_summary.json", payload)
    return payload


def update_readme(args: argparse.Namespace, decisions: dict[str, Any]) -> None:
    lines = [
        "# Haze4K v2.26 NoPost Risk Signal Separability Audit Evidence",
        "",
        f"Status: `{decisions.get('overall_decision', 'RUNNING_AUDIT')}`",
        "",
        "Diagnostic route only. No action joint training, post-train rescue, or locked Haze4K test command is authorized here.",
        "",
        "## Phase Decisions",
        "",
    ]
    for key in ("p0", "p1", "p2", "p3", "p4"):
        if key in decisions:
            lines.append(f"- {key.upper()}: `{decisions[key].get('decision')}`")
    lines.extend(
        [
            "",
            "## Primary Artifacts",
            "",
            "- `v226_p0_metric_fix_report.md`",
            "- `v226_p0_v225a_recomputed_metrics.json`",
            "- `v226_p1_join_replay_audit.json`",
            "- `v226_probe_feature_auc_table.csv`",
            "- `v226_p2_probe_summary.json`",
            "- `v226_canary_overfit_curve.csv`",
            "- `v226_optimizer_ablation_summary.csv`",
            "",
            "Large raw feature tensors, checkpoints, datasets, and images are not synced to GitHub by default.",
        ]
    )
    wtxt(args.out_dir / "README.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="p0,p1,p2")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--v221-metrics-csv", type=Path, required=True)
    ap.add_argument("--v225a-evid", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--v221-variant", default="V221_risk_temperature_gamma0p50")
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--risk-gamma", type=float, default=0.5)
    ap.add_argument("--risk-bias", type=float, default=-1.5)
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=0.01)
    ap.add_argument("--strong-reference-psnr", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=226)
    ap.add_argument("--probe-samples-per-fold", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=160)
    ap.add_argument("--canary-sizes", default="32,64")
    ap.add_argument("--canary-epochs", type=int, default=40)
    ap.add_argument("--p4-eval-fold", type=int, default=0)
    ap.add_argument("--p4-train-samples", type=int, default=192)
    ap.add_argument("--p4-eval-samples", type=int, default=96)
    ap.add_argument("--p4-epochs", type=int, default=3)
    args = ap.parse_args()
    args.folds_list = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    args.train_samples_per_fold = 384
    args.eval_samples_per_fold = 160
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = args.out_dir / "status.txt"
    with status.open("a") as f:
        f.write(f"v226_start phases={args.phases} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decisions: dict[str, Any] = {
        "branch": cmd(["git", "branch", "--show-current"]),
        "commit": cmd(["git", "rev-parse", "--short", "HEAD"]),
        "checkpoint_sha256": sha(args.checkpoint) if args.checkpoint.is_file() else "missing",
        "locked_test_touched": False,
    }
    for phase in [p.strip().lower() for p in args.phases.split(",") if p.strip()]:
        if phase == "p0":
            decisions["p0"] = phase_p0_metric_fix(args)
        elif phase == "p1":
            decisions["p1"] = phase_p1_join_replay(args)
        elif phase == "p2":
            decisions["p2"] = phase_p2_probe(args, device)
        elif phase == "p3":
            decisions["p3"] = phase_p3_canary(args, device)
        elif phase == "p4":
            decisions["p4"] = phase_p4_ablation(args, device)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        wjson(args.out_dir / "v226_phase_decisions.json", decisions)
        update_readme(args, decisions)
    final_bits = [decisions[k].get("decision", "") for k in ("p0", "p1", "p2", "p3", "p4") if k in decisions]
    if any("FAIL" in x or "NO_SIGNAL" in x or "NO_MINIMAL" in x for x in final_bits):
        decisions["overall_decision"] = "V226_DIAGNOSTIC_PAUSE_REVIEW_REQUIRED"
    else:
        decisions["overall_decision"] = "V226_PHASES_COMPLETED_REVIEW_NEXT_GATE"
    decisions["locked_test_touched"] = False
    wjson(args.out_dir / "v226_closeout.json", decisions)
    update_readme(args, decisions)
    with status.open("a") as f:
        f.write(f"v226_done phases={args.phases} decision={decisions['overall_decision']} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print("V226_OK", decisions["overall_decision"], flush=True)


if __name__ == "__main__":
    main()
