#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-26-nopost-risk-signal-separability-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
V225A=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-25a-nopost-risk-softlabel-scale-distill/experience_docx/experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704
STATUS=$EVID/status.txt
LOG=$EVID/v226_correctness_supplement.log

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi

echo "v226_correctness_supplement_start $(date --iso-8601=seconds) gpu=$GPU_ID" | tee -a "$STATUS"

set +e
"$PY" - "$REMOTE_ROOT" "$EVID" "$DATA" "$CKPT" "$SPLIT" "$V221" "$V225A" <<'PY' 2>&1 | tee "$LOG"
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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

remote_root = Path(sys.argv[1])
evid = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
checkpoint = Path(sys.argv[4])
split_csv = Path(sys.argv[5])
v221_metrics_csv = Path(sys.argv[6])
v225a_evid = Path(sys.argv[7])

repo_root = remote_root
its_root = repo_root / "Dehazing" / "ITS"
for item in (str(its_root), str(repo_root)):
    if item not in sys.path:
        sys.path.insert(0, item)

from experience_docx.tools import nopost_lowband_v226_risk_signal_audit as v226  # noqa: E402
from experience_docx.tools import nopost_lowband_v222_n3_microfit as v222  # noqa: E402
from experience_docx.tools import nopost_lowband_v223_oof_train as v223  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def run(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def finite(values: list[float]) -> list[float]:
    return [x for x in values if math.isfinite(x)]


def mean(values: list[float]) -> float:
    xs = finite(values)
    return sum(xs) / len(xs) if xs else float("nan")


def std(values: list[float]) -> float:
    xs = finite(values)
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def percentile(values: list[float], q: float) -> float:
    xs = sorted(finite(values))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def metric_bundle(scores: list[float], labels: list[int]) -> dict[str, float]:
    ee, mm = v226.ece(scores, labels, 10)
    return {
        "auc": v226.roc_auc(scores, labels),
        "ap": v226.average_precision_tie_aware(scores, labels),
        "prob_mean": mean(scores),
        "prob_std": std(scores),
        "ece10": ee,
        "mce10": mm,
        "label_base_rate": mean([float(x) for x in labels]),
    }


def rounded_metric_rows(dataset: str, scores: list[float], labels: list[int]) -> list[dict[str, Any]]:
    rows = []
    for digits in (None, 6, 5, 4, 3):
        rounded = scores if digits is None else [round(x, digits) for x in scores]
        rows.append(
            {
                "dataset": dataset,
                "round_digits": digits,
                "auc": v226.roc_auc(rounded, labels),
                "ap": v226.average_precision_tie_aware(rounded, labels),
                "prob_std": std(scores),
                "unique_scores": len(set(rounded)),
            }
        )
    return rows


def crop_box_for(path: Path, crop_size: int, seed: int) -> dict[str, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        return {"crop_top": -1, "crop_left": -1, "crop_h": crop_size, "crop_w": crop_size}
    if crop_size <= 0:
        return {"crop_top": 0, "crop_left": 0, "crop_h": h, "crop_w": w}
    if h <= crop_size or w <= crop_size:
        return {"crop_top": 0, "crop_left": 0, "crop_h": min(h, crop_size), "crop_w": min(w, crop_size)}
    rng = random.Random(seed)
    top = rng.randint(0, h - crop_size)
    left = rng.randint(0, w - crop_size)
    return {"crop_top": top, "crop_left": left, "crop_h": crop_size, "crop_w": crop_size}


def make_args() -> argparse.Namespace:
    args = argparse.Namespace(
        phases="supplement",
        data_dir=data_dir,
        checkpoint=checkpoint,
        split_csv=split_csv,
        v221_metrics_csv=v221_metrics_csv,
        v225a_evid=v225a_evid,
        out_dir=evid,
        folds="0,1,2",
        v221_variant="V221_risk_temperature_gamma0p50",
        hidden_channels=32,
        mid_grid=8,
        final_grid=16,
        crop_size=256,
        risk_gamma=0.5,
        risk_bias=-1.5,
        identity_tol=1e-6,
        learning_rate=1e-4,
        weight_decay=1e-4,
        grad_clip_norm=0.01,
        strong_reference_psnr=27.0,
        seed=226,
        probe_samples_per_fold=0,
        probe_epochs=160,
        canary_sizes="32,64",
        canary_epochs=40,
        p4_eval_fold=0,
        p4_train_samples=192,
        p4_eval_samples=96,
        p4_epochs=3,
        train_samples_per_fold=384,
        eval_samples_per_fold=160,
    )
    args.folds_list = [0, 1, 2]
    return args


def risk_biases(model: torch.nn.Module) -> dict[str, Any]:
    params = dict(model.named_parameters())
    out = {}
    for key in (
        "nopost_gated_lowband_policy.mid_policy.risk.3.bias",
        "nopost_gated_lowband_policy.final_policy.risk.3.bias",
    ):
        if key in params:
            out[key.replace("nopost_gated_lowband_policy.", "").replace(".", "_")] = [
                float(x) for x in params[key].detach().cpu().flatten().tolist()
            ]
    return out


def checkpoint_load_manifest(args: argparse.Namespace, device: torch.device) -> None:
    rows = []
    base_model = v226.load_route_model(args, device, None)
    base_state = base_model.state_dict()
    before_biases = risk_biases(base_model)
    for fold in args.folds_list:
        path = args.v225a_evid / f"fold{fold}" / f"v225a_fold{fold}_risk_context_Final.pkl"
        row: dict[str, Any] = {
            "fold": fold,
            "v225a_checkpoint_expected": str(path),
            "path_tail": "/".join(path.parts[-4:]),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path) if path.is_file() else "missing",
            "loaded_from_checkpoint": False,
            "load_state_missing": [],
            "load_state_unexpected": [],
            "shape_mismatch": [],
            "risk_final_bias_before": before_biases,
            "risk_final_bias_after_load": {},
            "strict_load_error": "",
        }
        if path.is_file():
            state = v222.load_state(path, device)
            missing = sorted([k for k in base_state if k not in state])
            unexpected = sorted([k for k in state if k not in base_state])
            mismatch = sorted(
                [
                    {"key": k, "checkpoint_shape": list(state[k].shape), "model_shape": list(base_state[k].shape)}
                    for k in state
                    if k in base_state and tuple(state[k].shape) != tuple(base_state[k].shape)
                ],
                key=lambda item: item["key"],
            )
            row["load_state_missing"] = missing
            row["load_state_unexpected"] = unexpected
            row["shape_mismatch"] = mismatch
            model = v226.load_route_model(args, device, None)
            try:
                model.load_state_dict(state)
                row["loaded_from_checkpoint"] = True
                row["risk_final_bias_after_load"] = risk_biases(model)
            except Exception as exc:
                row["strict_load_error"] = str(exc)
                compatible = {
                    k: v for k, v in state.items() if k in base_state and tuple(v.shape) == tuple(base_state[k].shape)
                }
                model.load_state_dict(compatible, strict=False)
                row["risk_final_bias_after_load"] = risk_biases(model)
        rows.append(row)
    write_json(evid / "v226_fold_checkpoint_load_manifest.json", {"folds": rows, "locked_test_touched": False})


def feature_variance_rows(features: dict[str, torch.Tensor], folds: list[int]) -> list[dict[str, Any]]:
    rows = []
    fold_tensor = torch.tensor(folds)
    for name, x in features.items():
        for fold in sorted(set(folds)):
            xf = x[(fold_tensor == fold).nonzero(as_tuple=False).flatten()]
            flat = xf.float()
            finite_mask = torch.isfinite(flat)
            channel_std = flat.std(dim=0, unbiased=False)
            row_l2 = torch.linalg.vector_norm(flat, dim=1)
            rows.append(
                {
                    "feature_set": name,
                    "fold": fold,
                    "feature_dim": int(flat.shape[1]),
                    "row_count": int(flat.shape[0]),
                    "finite_ratio": float(finite_mask.float().mean().item()),
                    "channel_zero_variance_count": int((channel_std <= 1e-12).sum().item()),
                    "feature_l2_mean": float(row_l2.mean().item()),
                    "feature_l2_std": float(row_l2.std(unbiased=False).item()),
                    "feature_abs_mean": float(flat.abs().mean().item()),
                    "feature_std_mean": float(channel_std.mean().item()),
                    "feature_std_p01": percentile([float(v) for v in channel_std.tolist()], 1),
                    "feature_std_p50": percentile([float(v) for v in channel_std.tolist()], 50),
                    "feature_std_p99": percentile([float(v) for v in channel_std.tolist()], 99),
                    "nan_count": int(torch.isnan(flat).sum().item()),
                    "inf_count": int(torch.isinf(flat).sum().item()),
                }
            )
    return rows


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


def train_probe_detail(
    X: torch.Tensor,
    labels: list[int],
    target_probs: list[float],
    folds: list[int],
    kind: str,
    epochs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    fold_tensor = torch.tensor(folds, dtype=torch.long, device=device)
    scores = torch.full_like(y, float("nan"))
    detail_rows: list[dict[str, Any]] = []
    coeffs: list[list[float]] = []
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
        lr = 0.01 if kind == "linear" else 0.003
        weight_decay = 1e-3
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr), y[train_idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            tr_scores = torch.sigmoid(model(Xtr)).detach().cpu().tolist()
            va_scores = torch.sigmoid(model(Xva)).detach().cpu().tolist()
            scores[val_idx] = torch.tensor(va_scores, dtype=scores.dtype, device=device)
        if kind == "linear":
            coeffs.append([float(v) for v in model.net.weight.detach().cpu().flatten().tolist()])
        train_labels = [labels[int(i)] for i in train_idx.detach().cpu().tolist()]
        val_labels = [labels[int(i)] for i in val_idx.detach().cpu().tolist()]
        train_targets = [target_probs[int(i)] for i in train_idx.detach().cpu().tolist()]
        val_targets = [target_probs[int(i)] for i in val_idx.detach().cpu().tolist()]
        ee, _ = v226.ece(va_scores, val_labels, 10)
        detail_rows.append(
            {
                "probe_type": kind,
                "fold_eval": fold,
                "train_folds": ",".join(str(x) for x in sorted(set(folds) - {fold})),
                "train_count": len(train_labels),
                "val_count": len(val_labels),
                "label_base_rate_train": mean([float(x) for x in train_labels]),
                "label_base_rate_val": mean([float(x) for x in val_labels]),
                "target_prob_mean_train": mean(train_targets),
                "target_prob_mean_val": mean(val_targets),
                "standardization": "train_fold_mean_std_clamp_min_1e-6",
                "seed": seed,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "train_auc": v226.roc_auc(tr_scores, train_labels),
                "val_auc": v226.roc_auc(va_scores, val_labels),
                "val_ap": v226.average_precision_tie_aware(va_scores, val_labels),
                "val_ap_epsilon_1e-4": v226.average_precision_tie_aware([round(x, 4) for x in va_scores], val_labels),
                "val_auc_epsilon_1e-4": v226.roc_auc([round(x, 4) for x in va_scores], val_labels),
                "val_prob_std": std(va_scores),
                "val_target_mae": mean([abs(s - t) for s, t in zip(va_scores, val_targets)]),
                "val_ece10": ee,
            }
        )
    return detail_rows, [float(x) for x in scores.detach().cpu().tolist()], coeffs


def p2_probe_outputs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    samples = v226.subset_per_fold(v226.active_samples(args), args.probe_samples_per_fold)
    features, meta = v226.extract_feature_table(args, samples, device)
    write_csv(evid / "v226_p2_feature_variance_summary.csv", feature_variance_rows(features, meta["folds"]))

    detail: list[dict[str, Any]] = []
    prediction_columns: dict[str, list[float]] = {}
    linear_coeffs: dict[str, list[list[float]]] = {}
    for feature_name, X in features.items():
        for kind in ("linear", "mlp"):
            rows, preds, coeffs = train_probe_detail(
                X, meta["labels"], meta["target_probs"], meta["folds"], kind, args.probe_epochs, args.seed
            )
            for row in rows:
                row["feature_set"] = feature_name
                detail.append(row)
            prediction_columns[f"{feature_name}|{kind}"] = preds
            if coeffs:
                linear_coeffs[feature_name] = coeffs
            print(f"SUPPLEMENT_P2_DETAIL feature={feature_name} kind={kind}", flush=True)

    fields = [
        "feature_set",
        "probe_type",
        "fold_eval",
        "train_folds",
        "train_count",
        "val_count",
        "label_base_rate_train",
        "label_base_rate_val",
        "target_prob_mean_train",
        "target_prob_mean_val",
        "standardization",
        "seed",
        "epochs",
        "lr",
        "weight_decay",
        "train_auc",
        "val_auc",
        "val_ap",
        "val_ap_epsilon_1e-4",
        "val_auc_epsilon_1e-4",
        "val_prob_std",
        "val_target_mae",
        "val_ece10",
    ]
    write_csv(evid / "v226_p2_probe_oof_detail.csv", detail, fields)
    return {"samples": samples, "features": features, "meta": meta, "prediction_columns": prediction_columns, "linear_coeffs": linear_coeffs}


def group_for_sample(sample: v223.FoldSample) -> str:
    label = int(fnum(sample.risk.get("unsafe_action_label"), 0.0))
    if label == 1:
        return "positive_highest_probability"
    return "negative_lowest_probability"


def target_bin(p: float) -> str:
    for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0000001)]:
        if lo <= p < hi:
            return f"[{lo},{hi})"
    return "out_of_range"


def canary_manifest(args: argparse.Namespace) -> dict[str, list[v223.FoldSample]]:
    samples = v226.active_samples(args)
    out: dict[str, list[v223.FoldSample]] = {}
    rows: list[dict[str, Any]] = []
    for count in [int(x) for x in args.canary_sizes.split(",") if x.strip()]:
        name = f"canary{count}"
        chosen = v226.select_canary(samples, count)
        out[name] = chosen
        for sample in chosen:
            seed = args.seed + v222.idx_hash(sample.name)
            p = fnum(sample.risk.get("unsafe_action_probability"), 0.0)
            rows.append(
                {
                    "canary_name": name,
                    "sample_name": sample.name,
                    "fold": sample.fold,
                    "selected_group": group_for_sample(sample),
                    "unsafe_action_label": int(fnum(sample.risk.get("unsafe_action_label"), 0.0)),
                    "unsafe_action_probability": p,
                    "v221_risk_scale": fnum(sample.risk.get("v221_risk_scale"), float("nan")),
                    "crop_seed": seed,
                    "crop_size": args.crop_size,
                    "target_prob_bin": target_bin(p),
                    "v221_replay_probability": p,
                    "raw_action_dpsnr": fnum(sample.risk.get("v221_raw_action_dPSNR"), float("nan")),
                    "hard_bottom25": v221_by_name.get(sample.name, {}).get("hard_bottom25", ""),
                    "easy_top25": v221_by_name.get(sample.name, {}).get("easy_top25", ""),
                    "strong_reference": v221_by_name.get(sample.name, {}).get("strong_reference", ""),
                }
            )
    write_csv(evid / "v226_p3_canary_sample_manifest.csv", rows)
    return out


def trainable_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "mid_context": [],
        "mid_risk": [],
        "final_context": [],
        "final_risk": [],
        "other_trainable": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".mid_policy.risk." in name:
            groups["mid_risk"].append((name, param))
        elif ".mid_policy.context." in name:
            groups["mid_context"].append((name, param))
        elif ".final_policy.risk." in name:
            groups["final_risk"].append((name, param))
        elif any(part in name for part in (".final_policy.context.", ".final_policy.mid_project.", ".final_policy.global_project.")):
            groups["final_context"].append((name, param))
        else:
            groups["other_trainable"].append((name, param))
    return groups


def group_l2(items: list[tuple[str, torch.nn.Parameter]], attr: str = "grad") -> float:
    vals = []
    for _, p in items:
        tensor = getattr(p, attr) if attr == "grad" else p
        if tensor is None:
            continue
        vals.append(torch.linalg.vector_norm(tensor.detach().float()).item() ** 2)
    return math.sqrt(sum(vals)) if vals else 0.0


def group_weight_stats(items: list[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    if not items:
        return {"weight_l2": 0.0, "weight_std": 0.0, "bias_mean": float("nan")}
    flats = [p.detach().float().cpu().flatten() for _, p in items]
    all_vals = torch.cat(flats) if flats else torch.empty(0)
    bias_vals = [p.detach().float().cpu().flatten() for n, p in items if n.endswith(".bias")]
    all_bias = torch.cat(bias_vals) if bias_vals else torch.empty(0)
    return {
        "weight_l2": float(torch.linalg.vector_norm(all_vals).item()) if all_vals.numel() else 0.0,
        "weight_std": float(all_vals.std(unbiased=False).item()) if all_vals.numel() else 0.0,
        "bias_mean": float(all_bias.mean().item()) if all_bias.numel() else float("nan"),
    }


def snapshot_groups(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict[str, dict[str, torch.Tensor]]:
    return {g: {n: p.detach().cpu().clone() for n, p in items} for g, items in groups.items()}


def group_delta_l2(items: list[tuple[str, torch.nn.Parameter]], before: dict[str, torch.Tensor]) -> float:
    vals = []
    for name, p in items:
        if name not in before:
            continue
        delta = p.detach().cpu().float() - before[name].float()
        vals.append(torch.linalg.vector_norm(delta).item() ** 2)
    return math.sqrt(sum(vals)) if vals else 0.0


def evaluate_samples(model: torch.nn.Module, samples: list[v223.FoldSample], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    recorder = v226.FeatureCapture(model)
    try:
        with torch.no_grad():
            for sample in samples:
                x = v222.image_tensor(sample.input_path, device)
                x, _ = v222.crop_pair(x, x, args.crop_size, args.seed + v222.idx_hash(sample.name))
                x, _, _ = v222.pad_to(x, 32)
                _ = model(x)
                tensors = model.nopost_gated_lowband_policy.last_tensors
                label = int(fnum(sample.risk.get("unsafe_action_label"), 0.0))
                target_prob = fnum(sample.risk.get("unsafe_action_probability"), 0.0)
                target = torch.tensor([[target_prob]], dtype=tensors["mid_unsafe_logit"].dtype, device=device)
                target_scale_value = fnum(sample.risk.get("v221_risk_scale"), (1.0 - target_prob) ** args.risk_gamma)
                target_scale = torch.tensor([[target_scale_value]], dtype=tensors["mid_scale"].dtype, device=device)
                soft = F.binary_cross_entropy_with_logits(tensors["mid_unsafe_logit"], target) + F.binary_cross_entropy_with_logits(
                    tensors["final_unsafe_logit"], target
                )
                scale_loss = F.mse_loss(tensors["mid_scale"], target_scale) + F.mse_loss(tensors["final_scale"], target_scale)
                mid_prob = float(tensors["mid_unsafe_prob"].mean().cpu())
                final_prob = float(tensors["final_unsafe_prob"].mean().cpu())
                vec = recorder.vectors()["D_risk_hidden_before_final_1x1"]
                rows.append(
                    {
                        "sample_name": sample.name,
                        "fold": sample.fold,
                        "label": label,
                        "target_prob": target_prob,
                        "mid_logit": float(tensors["mid_unsafe_logit"].mean().cpu()),
                        "final_logit": float(tensors["final_unsafe_logit"].mean().cpu()),
                        "mean_unsafe_prob": (mid_prob + final_prob) * 0.5,
                        "mid_prob": mid_prob,
                        "final_prob": final_prob,
                        "mid_scale": float(tensors["mid_scale"].mean().cpu()),
                        "final_scale": float(tensors["final_scale"].mean().cpu()),
                        "risk_hidden_std": float(vec.std(unbiased=False).cpu()),
                        "loss_soft_bce": float(soft.cpu()),
                        "loss_scale": float(scale_loss.cpu()),
                    }
                )
    finally:
        recorder.close()
    ranked = sorted(rows, key=lambda r: (-fnum(r["mean_unsafe_prob"], 0.0), r["sample_name"]))
    ranks = {row["sample_name"]: i + 1 for i, row in enumerate(ranked)}
    for row in rows:
        row["rank_by_prob"] = ranks[row["sample_name"]]
    return rows


def train_canary_with_details(
    args: argparse.Namespace,
    canary_name: str,
    train_samples: list[v223.FoldSample],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    model = v226.load_route_model(args, device, None)
    v226.set_risk_scope(model)
    groups = trainable_groups(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    prediction_snapshots: dict[int, list[dict[str, Any]]] = {}
    grad_rows: list[dict[str, Any]] = []
    curve_scores: dict[str, list[float]] = {}
    for epoch in range(1, args.canary_epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_samples)
        before = snapshot_groups(groups)
        losses = []
        grad_before: dict[str, list[float]] = {g: [] for g in groups}
        grad_after: dict[str, list[float]] = {g: [] for g in groups}
        clip_coefs = []
        for sample in train_samples:
            x = v222.image_tensor(sample.input_path, device)
            x, _ = v222.crop_pair(x, x, args.crop_size, args.seed + v222.idx_hash(sample.name))
            x, _, _ = v222.pad_to(x, 32)
            opt.zero_grad(set_to_none=True)
            _ = model(x)
            loss = v226.loss_from_tensors(model.nopost_gated_lowband_policy.last_tensors, sample, "soft_bce_scale", args.risk_gamma)
            loss.backward()
            all_trainable = [p for p in model.parameters() if p.requires_grad]
            total_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for p in all_trainable if p.grad is not None])).item())
            for group_name, items in groups.items():
                grad_before[group_name].append(group_l2(items, "grad"))
            clip_coef = min(1.0, args.grad_clip_norm / (total_norm + 1e-6)) if math.isfinite(args.grad_clip_norm) else 1.0
            clip_coefs.append(clip_coef)
            torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip_norm)
            for group_name, items in groups.items():
                grad_after[group_name].append(group_l2(items, "grad"))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        preds = evaluate_samples(model, train_samples, args, device)
        prediction_snapshots[epoch] = preds
        curve_scores[f"{canary_name}_epoch{epoch}"] = [fnum(r["mean_unsafe_prob"]) for r in preds]
        for group_name, items in groups.items():
            stats = group_weight_stats(items)
            grad_rows.append(
                {
                    "canary_name": canary_name,
                    "epoch": epoch,
                    "param_group": group_name,
                    "grad_l2_before_clip": mean(grad_before[group_name]),
                    "grad_l2_after_clip": mean(grad_after[group_name]),
                    "clip_coef": mean(clip_coefs),
                    "update_l2": group_delta_l2(items, before[group_name]),
                    "weight_l2": stats["weight_l2"],
                    "weight_std": stats["weight_std"],
                    "bias_mean": stats["bias_mean"],
                    "bias_delta": group_delta_l2([item for item in items if item[0].endswith(".bias")], before[group_name]),
                    "loss": mean(losses),
                }
            )
        print(f"SUPPLEMENT_P3 {canary_name} epoch={epoch}/{args.canary_epochs}", flush=True)
    best_epoch = max(prediction_snapshots, key=lambda ep: std([fnum(r["mean_unsafe_prob"]) for r in prediction_snapshots[ep]]))
    final_epoch = args.canary_epochs
    pred_rows = []
    for tag, epoch in (("best_prob_std_epoch", best_epoch), ("final_epoch", final_epoch)):
        for row in prediction_snapshots[epoch]:
            pred_rows.append({"canary_name": canary_name, "snapshot": tag, "epoch": epoch, **row})
    return pred_rows, grad_rows, curve_scores


def p4_variants() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_soft_bce_scale_clip_wd1e-4", "loss_mode": "soft_bce_scale", "lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 0.01, "tiny_nonzero_init": False},
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


def risk_weight_stds(model: torch.nn.Module) -> dict[str, float]:
    params = dict(model.named_parameters())
    return {
        "mid_risk_weight_std": float(params["nopost_gated_lowband_policy.mid_policy.risk.3.weight"].detach().float().std(unbiased=False).cpu()),
        "final_risk_weight_std": float(params["nopost_gated_lowband_policy.final_policy.risk.3.weight"].detach().float().std(unbiased=False).cpu()),
    }


def p4_replay(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    samples = v226.active_samples(args)
    train_pool = [s for s in samples if s.fold != args.p4_eval_fold]
    val_pool = [s for s in samples if s.fold == args.p4_eval_fold]
    rng = random.Random(args.seed + 2264)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    train_samples = train_pool[: args.p4_train_samples]
    val_samples = val_pool[: args.p4_eval_samples]
    summary_rows = []
    curve_rows = []
    best_scores: list[float] = []
    best_labels: list[int] = []
    best_key: tuple[float, float] = (-1.0, -1.0)
    for variant in p4_variants():
        model = v226.load_route_model(args, device, None)
        v226.set_risk_scope(model)
        if variant.get("tiny_nonzero_init"):
            v226.apply_tiny_nonzero_init(model, args.seed)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(variant["lr"]),
            weight_decay=float(variant["weight_decay"]),
        )
        final_train_preds: list[dict[str, Any]] = []
        final_val_preds: list[dict[str, Any]] = []
        for epoch in range(1, args.p4_epochs + 1):
            random.Random(args.seed + epoch).shuffle(train_samples)
            losses = []
            for sample in train_samples:
                x = v222.image_tensor(sample.input_path, device)
                x, _ = v222.crop_pair(x, x, args.crop_size, args.seed + v222.idx_hash(sample.name))
                x, _, _ = v222.pad_to(x, 32)
                opt.zero_grad(set_to_none=True)
                _ = model(x)
                loss = v226.loss_from_tensors(model.nopost_gated_lowband_policy.last_tensors, sample, str(variant["loss_mode"]), args.risk_gamma)
                loss.backward()
                if variant["grad_clip_norm"] is not None and math.isfinite(float(variant["grad_clip_norm"])):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(variant["grad_clip_norm"]))
                opt.step()
                losses.append(float(loss.detach().cpu()))
            final_train_preds = evaluate_samples(model, train_samples, args, device)
            final_val_preds = evaluate_samples(model, val_samples, args, device)
            train_scores = [fnum(r["mean_unsafe_prob"]) for r in final_train_preds]
            train_labels = [int(r["label"]) for r in final_train_preds]
            val_scores = [fnum(r["mean_unsafe_prob"]) for r in final_val_preds]
            val_labels = [int(r["label"]) for r in final_val_preds]
            weight_stats = risk_weight_stds(model)
            curve_rows.append(
                {
                    "variant": variant["name"],
                    "epoch": epoch,
                    "loss": mean(losses),
                    "train_auc": v226.roc_auc(train_scores, train_labels),
                    "train_prob_std": std(train_scores),
                    "train_target_mae": mean([abs(fnum(r["mean_unsafe_prob"]) - fnum(r["target_prob"])) for r in final_train_preds]),
                    "val_auc": v226.roc_auc(val_scores, val_labels),
                    "val_prob_std": std(val_scores),
                    "val_target_mae": mean([abs(fnum(r["mean_unsafe_prob"]) - fnum(r["target_prob"])) for r in final_val_preds]),
                    "mid_risk_weight_std": weight_stats["mid_risk_weight_std"],
                    "final_risk_weight_std": weight_stats["final_risk_weight_std"],
                    "hidden_std_mean": mean([fnum(r["risk_hidden_std"]) for r in final_val_preds]),
                }
            )
        train_scores = [fnum(r["mean_unsafe_prob"]) for r in final_train_preds]
        train_labels = [int(r["label"]) for r in final_train_preds]
        val_scores = [fnum(r["mean_unsafe_prob"]) for r in final_val_preds]
        val_labels = [int(r["label"]) for r in final_val_preds]
        train_metrics = metric_bundle(train_scores, train_labels)
        val_metrics = metric_bundle(val_scores, val_labels)
        weight_stats = risk_weight_stds(model)
        pass_auc = val_metrics["auc"] >= 0.75
        pass_std = val_metrics["prob_std"] >= 0.05
        summary_rows.append(
            {
                "variant": variant["name"],
                "loss_mode": variant["loss_mode"],
                "lr": variant["lr"],
                "weight_decay": variant["weight_decay"],
                "grad_clip_norm": variant["grad_clip_norm"],
                "tiny_nonzero_init": variant["tiny_nonzero_init"],
                "epochs": args.p4_epochs,
                "train_count": len(train_samples),
                "val_count": len(val_samples),
                "train_auc": train_metrics["auc"],
                "train_ap": train_metrics["ap"],
                "train_prob_mean": train_metrics["prob_mean"],
                "train_prob_std": train_metrics["prob_std"],
                "train_target_mae": mean([abs(fnum(r["mean_unsafe_prob"]) - fnum(r["target_prob"])) for r in final_train_preds]),
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "val_prob_mean": val_metrics["prob_mean"],
                "val_prob_std": val_metrics["prob_std"],
                "val_target_mae": mean([abs(fnum(r["mean_unsafe_prob"]) - fnum(r["target_prob"])) for r in final_val_preds]),
                "val_ece10": val_metrics["ece10"],
                "risk_mid_final_weight_std": weight_stats["mid_risk_weight_std"],
                "risk_final_final_weight_std": weight_stats["final_risk_weight_std"],
                "pass_line_auc": pass_auc,
                "pass_line_prob_std": pass_std,
                "decision": "pass" if pass_auc and pass_std else "fail",
            }
        )
        key = (val_metrics["auc"], val_metrics["prob_std"])
        if key > best_key:
            best_key = key
            best_scores = val_scores
            best_labels = val_labels
        print(f"SUPPLEMENT_P4 variant={variant['name']}", flush=True)
    write_csv(evid / "v226_p4_all_variants_summary.csv", summary_rows)
    write_csv(evid / "v226_p4_optimizer_ablation_curve.csv", curve_rows)
    return {"p4_best_scores": best_scores, "p4_best_labels": best_labels, "train_count": len(train_samples), "val_count": len(val_samples)}


def target_key_audit(args: argparse.Namespace, canaries: dict[str, list[v223.FoldSample]]) -> None:
    rows = [r for r in read_csv(args.v221_metrics_csv) if r.get("variant") == args.v221_variant]
    samples = v226.active_samples(args)
    p4_samples = [s for s in samples if s.fold != args.p4_eval_fold][: args.p4_train_samples] + [s for s in samples if s.fold == args.p4_eval_fold][: args.p4_eval_samples]
    values = []
    for row in rows:
        for key in ("unsafe_action_label", "unsafe_action_probability", "risk_scale", "raw_action_dPSNR"):
            values.append(fnum(row.get(key)))
    target_probs = [fnum(r.get("unsafe_action_probability")) for r in rows]
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0000001]
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        hist.append({"bin": f"[{lo},{hi})", "count": sum(1 for p in target_probs if lo <= p < hi)})
    payload = {
        "sample_count": len(rows),
        "unsafe_action_label_missing": sum(1 for r in rows if r.get("unsafe_action_label", "") == ""),
        "unsafe_action_probability_missing": sum(1 for r in rows if r.get("unsafe_action_probability", "") == ""),
        "v221_risk_scale_missing": sum(1 for s in samples if "v221_risk_scale" not in s.risk),
        "risk_scale_missing": sum(1 for r in rows if r.get("risk_scale", "") == ""),
        "raw_action_dPSNR_missing": sum(1 for r in rows if r.get("raw_action_dPSNR", "") == ""),
        "scale_fallback_used_count_p3": sum(1 for ss in canaries.values() for s in ss if "v221_risk_scale" not in s.risk),
        "scale_fallback_used_count_p4": sum(1 for s in p4_samples if "v221_risk_scale" not in s.risk),
        "nan_count": sum(1 for x in values if math.isnan(x)),
        "inf_count": sum(1 for x in values if math.isinf(x)),
        "target_prob_histogram": hist,
        "locked_test_touched": False,
    }
    write_json(evid / "v226_target_key_presence_audit.json", payload)


def positive_control_importance(args: argparse.Namespace, p2: dict[str, Any]) -> None:
    feature_name = "E_v221_cached_scalar_positive_control"
    X = p2["features"][feature_name]
    labels = p2["meta"]["labels"]
    targets = p2["meta"]["target_probs"]
    folds = p2["meta"]["folds"]
    cols = p2["meta"]["v221_feature_columns"]
    _, full_preds, coeffs = train_probe_detail(X, labels, targets, folds, "linear", args.probe_epochs, args.seed)
    full_auc = v226.roc_auc(full_preds, labels)
    full_ap = v226.average_precision_tie_aware(full_preds, labels)
    coeff_by_idx = []
    for idx in range(len(cols)):
        vals = [c[idx] for c in coeffs if idx < len(c)]
        coeff_by_idx.append((mean([abs(v) for v in vals]), mean(vals)))
    rows = []
    for idx, col in enumerate(cols):
        keep = [i for i in range(X.shape[1]) if i != idx]
        _, preds, _ = train_probe_detail(X[:, keep], labels, targets, folds, "linear", args.probe_epochs, args.seed)
        auc = v226.roc_auc(preds, labels)
        ap = v226.average_precision_tie_aware(preds, labels)
        abscoef, signed = coeff_by_idx[idx]
        rows.append(
            {
                "feature_name": col,
                "importance_type": "linear_probe_leave_one_out_and_standardized_coeff",
                "importance_value": abscoef,
                "auc_drop_when_removed": full_auc - auc,
                "ap_drop_when_removed": full_ap - ap,
                "sign": "positive" if signed > 0 else "negative" if signed < 0 else "zero",
                "notes": f"full_linear_auc={full_auc}; full_linear_ap={full_ap}",
            }
        )
        print(f"SUPPLEMENT_B2 feature={col}", flush=True)
    write_csv(evid / "v221_positive_control_feature_importance.csv", rows)


def probe_sample_predictions(args: argparse.Namespace, p2: dict[str, Any]) -> None:
    meta = p2["meta"]
    preds = p2["prediction_columns"]
    rows = []
    for i, name in enumerate(meta["names"]):
        b_prob = fnum(preds["B_final_ll_pooled|linear"][i])
        label = int(meta["labels"][i])
        pred_label = int(b_prob >= 0.5)
        if pred_label == label:
            bucket = "correct_positive" if label else "correct_negative"
        else:
            bucket = "false_positive" if pred_label else "false_negative"
        src = v221_by_name.get(name, {})
        rows.append(
            {
                "sample_name": name,
                "fold": meta["folds"][i],
                "label": label,
                "target_prob": meta["target_probs"][i],
                "v221_prob": src.get("unsafe_action_probability", meta["target_probs"][i]),
                "A_linear_prob": preds["A_mid_ll_pooled|linear"][i],
                "B_linear_prob": b_prob,
                "C_linear_prob": preds["C_final_ll_mid_global_context|linear"][i],
                "D_linear_prob": preds["D_risk_hidden_before_final_1x1|linear"][i],
                "E_mlp_prob": preds["E_v221_cached_scalar_positive_control|mlp"][i],
                "hard_bottom25": src.get("hard_bottom25", ""),
                "easy_top25": src.get("easy_top25", ""),
                "strong_reference": src.get("strong_reference", ""),
                "raw_action_dpsnr": src.get("raw_action_dPSNR", ""),
                "error_bucket": bucket,
            }
        )
    write_csv(evid / "v226_probe_sample_predictions_compact.csv", rows)


def crop_consistency_manifests(args: argparse.Namespace) -> None:
    rows = []
    repeated = []
    base_rows = [r for r in read_csv(args.v221_metrics_csv) if r.get("variant") == args.v221_variant]
    for row in base_rows:
        p = fnum(row.get("unsafe_action_probability"))
        rows.append(
            {
                "image_name": row["name"],
                "num_crops": 1,
                "crop_target_prob_mean": p,
                "crop_target_prob_std": 0.0,
                "crop_target_prob_min": p,
                "crop_target_prob_max": p,
                "crop_label_positive_rate": fnum(row.get("unsafe_action_label")),
                "full_image_a0_psnr": row.get("A0_PSNR", ""),
                "full_image_candidate_psnr": row.get("candidate_PSNR", ""),
                "full_image_action_dpsnr": row.get("raw_action_dPSNR", ""),
                "hard_bottom25": row.get("hard_bottom25", ""),
                "easy_top25": row.get("easy_top25", ""),
                "strong_reference": row.get("strong_reference", ""),
                "target_source": "cached_v221_image_level_metric_single_row_per_name",
            }
        )
    write_csv(evid / "full_image_vs_crop_risk_consistency_compact.csv", rows)

    input_dir, _ = v222.train_dirs(args.data_dir)
    for row in base_rows[:64]:
        name = row["name"]
        src = input_dir / name
        for crop_id, offset in enumerate([0, 1, 2, 3, 4]):
            seed = args.seed + offset * 1009 + v222.idx_hash(name)
            box = crop_box_for(src, args.crop_size, seed)
            repeated.append(
                {
                    "image_name": name,
                    "seed": seed,
                    "crop_id": crop_id,
                    "unsafe_action_label": row.get("unsafe_action_label", ""),
                    "unsafe_action_probability": row.get("unsafe_action_probability", ""),
                    "raw_action_dpsnr": row.get("raw_action_dPSNR", ""),
                    "v221_prob": row.get("unsafe_action_probability", ""),
                    "risk_scale": row.get("risk_scale", ""),
                    **box,
                    "target_source": "cached_v221_image_level_metric_not_recomputed_per_crop",
                }
            )
    write_csv(evid / "crop_target_noise_repeated_seed_audit.csv", repeated)


def cross_route_matrices() -> None:
    gate_rows = [
        {"route": "v2.16", "mechanism": "WLDB-A trained lowband decoder", "trained_or_oracle": "trained", "mean_dpsnr": 0.081889, "hard_dpsnr": 0.105887, "easy_dpsnr": 0.020994, "p05": "", "cvar5": "", "severe_count": 67, "strong_reference_regressions": 48, "positive_ratio": 0.6625, "wrong_direction_rate": "", "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING"},
        {"route": "v2.17-O1", "mechanism": "global final-feature LL oracle", "trained_or_oracle": "oracle", "mean_dpsnr": 0.842954, "hard_dpsnr": 1.591207, "easy_dpsnr": 0.359026, "p05": 0.001803, "cvar5": "", "severe_count": 0, "strong_reference_regressions": 0, "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "oracle_headroom_positive_not_train_authorized"},
        {"route": "v2.17-O2", "mechanism": "spatial final-feature LL oracle", "trained_or_oracle": "oracle", "mean_dpsnr": 6.16049, "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "oracle_headroom_positive_not_train_authorized"},
        {"route": "v2.17-O3", "mechanism": "mid+final feature LL oracle", "trained_or_oracle": "oracle", "mean_dpsnr": 6.832469, "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "oracle_headroom_positive_not_train_authorized"},
        {"route": "v2.18", "mechanism": "O1 deployable pooled-LL MLP", "trained_or_oracle": "learnability", "mean_dpsnr": 0.263178, "hard_dpsnr": 0.859418, "easy_dpsnr": -0.183929, "p05": -1.164642, "cvar5": -2.050251, "severe_count": 568, "strong_reference_regressions": 303, "positive_ratio": 0.592083, "wrong_direction_rate": 0.19375, "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "V218_PAUSE_P1_GLOBAL_POLICY_LEARNABILITY_FAIL"},
        {"route": "v2.19", "mechanism": "O2 final-only spatial predictor", "trained_or_oracle": "learnability", "mean_dpsnr": 0.9921, "hard_dpsnr": 2.6504, "easy_dpsnr": -0.1346, "p05": -1.1486, "cvar5": -2.1058, "severe_count": "0.2025_rate", "strong_reference_regressions": 302, "positive_ratio": 0.7192, "wrong_direction_rate": "", "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "V219_LEARNABILITY_FAIL_OR_GUARD_FAIL_PAUSE_BEFORE_TRAINING"},
        {"route": "v2.20", "mechanism": "O3 mid+final/global-context predictor", "trained_or_oracle": "learnability", "mean_dpsnr": 2.0684, "hard_dpsnr": 4.1450, "easy_dpsnr": 0.5199, "p05": -0.7255, "cvar5": -1.6967, "severe_count": "0.11125_rate", "strong_reference_regressions": "0.2667_rate", "positive_ratio": 0.8508, "wrong_direction_rate": 0.00417, "risk_auc": "", "risk_ap": "", "risk_prob_std": "", "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "V220_P1A_PASS_P1B_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING"},
        {"route": "v2.24", "mechanism": "trained risk-head collapse audit", "trained_or_oracle": "diagnostic", "mean_dpsnr": "", "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": 0.5175, "risk_ap": "", "risk_prob_std": 0.0005259, "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "V224_DIAGNOSTIC_COMPLETE_CASE_A_RISK_HEAD_COLLAPSE_LOCKED_TEST_BLOCKED"},
        {"route": "v2.25A", "mechanism": "direct risk soft-label / scale distillation", "trained_or_oracle": "trained", "mean_dpsnr": "", "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": 0.5501, "risk_ap": 0.1397, "risk_prob_std": 0.001669, "canary_pass": "", "optimizer_rescue_pass": "", "locked_test_touched": False, "decision": "V225A_RISK_CALIBRATION_GATE_FAIL_NORMAL_PAUSE"},
        {"route": "v2.26", "mechanism": "risk signal separability / trainability audit", "trained_or_oracle": "diagnostic", "mean_dpsnr": "", "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "positive_ratio": "", "wrong_direction_rate": "", "risk_auc": 0.6436, "risk_ap": 0.2061, "risk_prob_std": 0.0224, "canary_pass": False, "optimizer_rescue_pass": False, "locked_test_touched": False, "decision": "V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED"},
    ]
    write_csv(evid / "nopost_cross_route_gate_matrix_v216_v226.csv", gate_rows)
    oracle_rows = [
        {"oracle_or_policy": "v2.17 O1", "insertion_stage": "global final-feature LL", "uses_mid_context": False, "uses_final_context": True, "uses_global_context": True, "mean_dpsnr": 0.842954, "hard_dpsnr": 1.591207, "easy_dpsnr": 0.359026, "p05": 0.001803, "cvar5": "", "severe_count": 0, "strong_reference_regressions": 0, "wrong_direction_rate": "", "notes": "oracle; headroom exists but not deployable policy"},
        {"oracle_or_policy": "v2.17 O2", "insertion_stage": "spatial final-feature LL", "uses_mid_context": False, "uses_final_context": True, "uses_global_context": False, "mean_dpsnr": 6.16049, "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "wrong_direction_rate": "", "notes": "oracle; strong capacity"},
        {"oracle_or_policy": "v2.17 O3", "insertion_stage": "mid+final feature LL", "uses_mid_context": True, "uses_final_context": True, "uses_global_context": False, "mean_dpsnr": 6.832469, "hard_dpsnr": "", "easy_dpsnr": "", "p05": "", "cvar5": "", "severe_count": "", "strong_reference_regressions": "", "wrong_direction_rate": "", "notes": "oracle; strongest v2.17 feature-lowband headroom"},
        {"oracle_or_policy": "v2.20 P1_final_mid_global_context_predictor", "insertion_stage": "mid+final/global-context predictor", "uses_mid_context": True, "uses_final_context": True, "uses_global_context": True, "mean_dpsnr": 2.0684, "hard_dpsnr": 4.1450, "easy_dpsnr": 0.5199, "p05": -0.7255, "cvar5": -1.6967, "severe_count": "0.11125_rate", "strong_reference_regressions": "0.2667_rate", "wrong_direction_rate": 0.00417, "notes": "mechanism pass, safety fail; not training authorized"},
    ]
    write_csv(evid / "oracle_capacity_by_insertion_stage_compact.csv", oracle_rows)


def run_manifest(args: argparse.Namespace) -> None:
    diff_check = git(["diff", "--check"])
    input_files = []
    for role, path in (
        ("official_checkpoint", args.checkpoint),
        ("split_csv", args.split_csv),
        ("v221_metrics_csv", args.v221_metrics_csv),
        ("v225a_oof_risk_eval_csv", args.v225a_evid / "v225a_oof_risk_eval.csv"),
    ):
        item = {
            "role": role,
            "path_tail": "/".join(path.parts[-6:]),
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.is_file() else "missing",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        if path.suffix.lower() == ".csv" and path.is_file():
            item["row_count"] = len(read_csv(path))
        input_files.append(item)
    payload = {
        "branch": git(["branch", "--show-current"]),
        "route_commit": git(["rev-parse", "--short", "HEAD"]),
        "main_sync_commit": "fc5b68b",
        "git_status_porcelain": git(["status", "--porcelain"]),
        "git_diff_check": "pass" if not diff_check else diff_check,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "exact_commands": {
            "p0_p2": "bash experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/run_v226_p0_p2_audit.sh",
            "p3": "bash experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/run_v226_p3_canary.sh",
            "p4": "bash experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/run_v226_p4_ablation.sh",
            "supplement": "bash experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/run_v226_correctness_supplement.sh",
        },
        "args": {
            "seed": args.seed,
            "crop_size": args.crop_size,
            "folds": args.folds,
            "probe_epochs": args.probe_epochs,
            "canary_epochs": args.canary_epochs,
            "p4_epochs": args.p4_epochs,
            "p4_train_samples": args.p4_train_samples,
            "p4_eval_samples": args.p4_eval_samples,
        },
        "input_files": input_files,
        "locked_test_touched": False,
    }
    write_json(evid / "v226_run_manifest.json", payload)


def closeout_text(args: argparse.Namespace) -> None:
    forbidden = run(["bash", "-lc", f"find {str(evid)!r} -maxdepth 1 -type f | grep -Ei '\\.(pkl|pth|pt|ckpt|onnx|png|jpg|jpeg|bmp|gif|webp|npy|npz|mat|zip|tar|gz|7z|rar)$' || true"])
    locked_grep = run(["bash", "-lc", "grep -RInE 'locked|test' experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/*.sh experience_docx/tools/nopost_lowband_v226_risk_signal_audit.py | sed -n '1,80p'"])
    text = "\n".join(
        [
            "git branch --show-current",
            git(["branch", "--show-current"]),
            "",
            "git rev-parse HEAD",
            git(["rev-parse", "HEAD"]),
            "",
            "git status --porcelain",
            git(["status", "--porcelain"]),
            "",
            "git diff --check",
            git(["diff", "--check"]) or "DIFF_CHECK_OK",
            "",
            "forbidden artifact scan",
            forbidden or "FORBIDDEN_ARTIFACT_SCAN_OK",
            "",
            "tmux ls",
            run(["bash", "-lc", "tmux ls 2>/dev/null || true"]) or "NO_TMUX_SESSIONS_REPORTED",
            "",
            "pgrep/python process scan",
            run(["bash", "-lc", "ps -eo pid,etime,cmd | grep -E 'v226|nopost_lowband_v226|haze4k-v2-26' | grep -v grep || true"]) or "NO_ACTIVE_V226_PROCESS_REPORTED",
            "",
            "locked Haze4K command/path grep result",
            locked_grep,
            "",
            "V226_CLOUD_CLOSEOUT_MANIFEST_OK",
        ]
    )
    (evid / "v226_cloud_closeout_manifest.txt").write_text(text + "\n", encoding="utf-8")

    source_diff = "\n".join(
        [
            "git show --stat 30ca5aa",
            git(["show", "--stat", "--oneline", "30ca5aa"]),
            "",
            "git diff --name-only github/codex/haze4k-official-arch-anchor...HEAD",
            git(["diff", "--name-only", "github/codex/haze4k-official-arch-anchor...HEAD"]),
            "",
            "V226_SOURCE_DIFF_STAT_OK",
        ]
    )
    (evid / "v226_source_diff_stat.txt").write_text(source_diff + "\n", encoding="utf-8")


def epsilon_sanity(v225_scores: list[float], v225_labels: list[int], canary_scores: dict[str, list[float]], p4_scores: list[float], p4_labels: list[int]) -> None:
    datasets = []
    datasets.extend(rounded_metric_rows("v225a_oof", v225_scores, v225_labels))
    if "canary32_epoch1" in canary_scores:
        datasets.extend(rounded_metric_rows("p3_canary32_epoch1", canary_scores["canary32_epoch1"], [int(fnum(s.risk.get("unsafe_action_label"), 0.0)) for s in canary_sets["canary32"]]))
    if f"canary32_epoch{args.canary_epochs}" in canary_scores:
        datasets.extend(rounded_metric_rows("p3_canary32_final", canary_scores[f"canary32_epoch{args.canary_epochs}"], [int(fnum(s.risk.get("unsafe_action_label"), 0.0)) for s in canary_sets["canary32"]]))
    datasets.extend(rounded_metric_rows("p4_best_final", p4_scores, p4_labels))
    base = mean([float(x) for x in v225_labels])
    write_json(
        evid / "v226_metric_epsilon_tie_sanity.json",
        {
            "datasets": ["v225a_oof", "p3_canary32_epoch1", "p3_canary32_final", "p4_best_final"],
            "metrics": datasets,
            "constant_score_control": {"base_rate": base, "ap": base, "auc": 0.5},
            "locked_test_touched": False,
        },
    )


def final_forbidden_scan() -> str:
    forbidden_exts = {".pkl", ".pth", ".pt", ".ckpt", ".onnx", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".npy", ".npz", ".mat", ".zip", ".tar", ".gz", ".7z", ".rar"}
    hits = [str(p) for p in evid.iterdir() if p.is_file() and p.suffix.lower() in forbidden_exts]
    return "pass" if not hits else "fail:" + ",".join(hits)


args = make_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evid.mkdir(parents=True, exist_ok=True)
v221_rows = [r for r in read_csv(v221_metrics_csv) if r.get("variant") == args.v221_variant]
v221_by_name = {r["name"]: r for r in v221_rows}

run_manifest(args)
checkpoint_load_manifest(args, device)
p2 = p2_probe_outputs(args, device)
canary_sets = canary_manifest(args)
canary_pred_rows: list[dict[str, Any]] = []
grad_rows: list[dict[str, Any]] = []
canary_metric_scores: dict[str, list[float]] = {}
for name, selected in canary_sets.items():
    preds, grads, scores = train_canary_with_details(args, name, list(selected), device)
    canary_pred_rows.extend(preds)
    grad_rows.extend(grads)
    canary_metric_scores.update(scores)
write_csv(evid / "v226_p3_canary_final_predictions.csv", canary_pred_rows)
write_csv(evid / "v226_p3_gradient_flow_summary.csv", grad_rows)

p4 = p4_replay(args, device)
target_key_audit(args, canary_sets)
positive_control_importance(args, p2)
probe_sample_predictions(args, p2)
crop_consistency_manifests(args)
cross_route_matrices()

v225_rows = read_csv(v225a_evid / "v225a_oof_risk_eval.csv")
v225_scores = [fnum(r.get("trained_mean_unsafe_prob")) for r in v225_rows]
v225_labels = [int(fnum(r.get("unsafe_action_label"), 0.0)) for r in v225_rows]
epsilon_sanity(v225_scores, v225_labels, canary_metric_scores, p4["p4_best_scores"], p4["p4_best_labels"])
closeout_text(args)

manifest_path = evid / "v226_run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["forbidden_artifact_scan"] = final_forbidden_scan()
write_json(manifest_path, manifest)

print("V226_CORRECTNESS_SUPPLEMENT_OK", flush=True)
PY
rc=${PIPESTATUS[0]}
set -e

echo "v226_correctness_supplement_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo V226_CORRECTNESS_SUPPLEMENT_SCRIPT_OK
else
  echo V226_CORRECTNESS_SUPPLEMENT_SCRIPT_FAILED
fi
exit "$rc"
