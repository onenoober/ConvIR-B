#!/usr/bin/env python3
"""Haze4K v2.38B richer target-only no-op/unsafe separability audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ConvIR import build_net  # noqa: E402

ROUTE_ID = "haze4k_v2_38b_rich_target_only_noop_unsafe_separability_audit_20260706"
FORBIDDEN_DEPLOY_TOKENS = ("gt", "teacher", "wdmamba", "delta", "psnr", "image_id", "sample", "fold", "label", "crop")


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_checkpoint_model(path: str | Path, map_location: Any) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_official_base(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_checkpoint_model(checkpoint, device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_tensor(path: str | Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict) and "tensor" in tensor:
        tensor = tensor["tensor"]
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor at {path}")
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected CHW/NCHW tensor at {path}, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(tensor, 0.0, 1.0)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().clamp(0.0, 1.0)


def pad_to_factor(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), "reflect"), h, w


def lowpass(tensor: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def tensor_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    t = tensor.detach().float()
    flat = t.flatten()
    ch = t.flatten(2)
    channel_mean = ch.mean(dim=2).flatten()
    spatial_std = ch.std(dim=2, unbiased=False).flatten()
    low = lowpass(t) if t.ndim == 4 and min(t.shape[-2:]) >= 9 else t
    rms = torch.sqrt((flat * flat).mean().clamp_min(1e-12))
    low_rms = torch.sqrt((low.flatten() * low.flatten()).mean().clamp_min(1e-12))
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_abs_mean": float(flat.abs().mean().item()),
        f"{prefix}_rms": float(rms.item()),
        f"{prefix}_p95_abs": float(torch.quantile(flat.abs(), 0.95).item()),
        f"{prefix}_channel_mean_std": float(channel_mean.std(unbiased=False).item()),
        f"{prefix}_spatial_std_mean": float(spatial_std.mean().item()),
        f"{prefix}_low_rms": float(low_rms.item()),
        f"{prefix}_low_ratio": float((low_rms / rms.clamp_min(1e-12)).item()),
    }


def image_extra_features(input_tensor: torch.Tensor, a0: torch.Tensor) -> dict[str, float]:
    resid = a0 - input_tensor
    low = lowpass(resid)
    high = resid - low
    low_rms = float(torch.sqrt((low * low).mean().clamp_min(1e-12)).item())
    high_rms = float(torch.sqrt((high * high).mean().clamp_min(1e-12)).item())
    resid_rms = float(torch.sqrt((resid * resid).mean().clamp_min(1e-12)).item())
    out = {
        "input_clip_low_rate": float((input_tensor <= 1e-4).float().mean().item()),
        "input_clip_high_rate": float((input_tensor >= 1.0 - 1e-4).float().mean().item()),
        "A0_clip_low_rate": float((a0 <= 1e-4).float().mean().item()),
        "A0_clip_high_rate": float((a0 >= 1.0 - 1e-4).float().mean().item()),
        "input_A0_resid_mean": float(resid.mean().item()),
        "input_A0_resid_std": float(resid.std(unbiased=False).item()),
        "input_A0_resid_rms": resid_rms,
        "input_A0_resid_low_rms": low_rms,
        "input_A0_resid_high_rms": high_rms,
        "input_A0_resid_low_ratio": low_rms / max(resid_rms, 1e-12),
    }
    for idx, name in enumerate(("r", "g", "b")):
        out[f"input_A0_resid_{name}_mean"] = float(resid[:, idx].mean().item())
        out[f"A0_{name}_mean"] = float(a0[:, idx].mean().item())
        out[f"input_{name}_mean"] = float(input_tensor[:, idx].mean().item())
    return out


def capture_convir_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    feats: dict[str, torch.Tensor] = {}
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x_ = model.feat_extract[0](x)
    feats["S1_feat_extract0"] = x_
    res1 = model.Encoder[0](x_)
    feats["S2_encoder0"] = res1
    z = model.feat_extract[1](res1)
    feats["S3_feat_extract1"] = z
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    feats["S4_encoder_late"] = res2
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)
    feats["S5_bottleneck_mid"] = z
    z = model.Decoder[0](z)
    feats["S6_decoder_early"] = z
    return feats


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [score for label, score in zip(labels, scores) if label == 1]
    neg = [score for label, score in zip(labels, scores) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def threshold_at_fpr(labels: list[int], scores: list[float], max_fpr: float) -> tuple[float, float, float]:
    negatives = sum(1 for label in labels if label == 0)
    positives = sum(1 for label in labels if label == 1)
    best = (math.inf, 0.0, 0.0)
    for threshold in sorted(set(scores), reverse=True):
        pred = [score >= threshold for score in scores]
        fp = sum(1 for p, y in zip(pred, labels) if p and y == 0)
        tp = sum(1 for p, y in zip(pred, labels) if p and y == 1)
        fpr = fp / negatives if negatives else 0.0
        recall = tp / positives if positives else 0.0
        if fpr <= max_fpr and recall >= best[1]:
            best = (threshold, recall, fpr)
    if not math.isfinite(best[0]):
        return max(scores) if scores else 0.0, 0.0, 0.0
    return best


def train_logistic(train_x: torch.Tensor, train_y: torch.Tensor, epochs: int, lr: float) -> torch.nn.Linear:
    model = torch.nn.Linear(train_x.shape[1], 1)
    positives = float(train_y.sum().item())
    negatives = float(train_y.numel() - positives)
    pos_weight = torch.tensor([negatives / positives]) if positives > 0 else torch.tensor([1.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
    return model


def oof_scores(rows: list[dict[str, Any]], feature_cols: list[str], labels: list[int], epochs: int, lr: float) -> tuple[list[float], list[dict[str, Any]]]:
    if not feature_cols:
        scores = [sum(labels) / len(labels) if labels else 0.0 for _ in rows]
        per_fold = []
        for fold_id in range(5):
            fold_labels = [int(labels[i]) for i, row in enumerate(rows) if int(row["fold_id"]) == fold_id]
            per_fold.append({
                "fold_id": fold_id,
                "sample_count": len(fold_labels),
                "unsafe_count": sum(fold_labels),
                "unsafe_AUROC": None,
                "unsafe_AUPRC": None,
                "fold_pass": False,
            })
        return scores, per_fold
    matrix = torch.tensor([[fnum(row.get(col), 0.0) or 0.0 for col in feature_cols] for row in rows], dtype=torch.float32)
    y = torch.tensor([float(v) for v in labels], dtype=torch.float32)
    scores = [0.0 for _ in rows]
    per_fold = []
    for fold_id in range(5):
        train_idx = [i for i, row in enumerate(rows) if int(row["fold_id"]) != fold_id]
        test_idx = [i for i, row in enumerate(rows) if int(row["fold_id"]) == fold_id]
        train_x = matrix[train_idx]
        test_x = matrix[test_idx]
        mu = train_x.mean(dim=0, keepdim=True)
        sigma = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = (train_x - mu) / sigma
        test_x = (test_x - mu) / sigma
        train_y = y[train_idx]
        if float(train_y.sum().item()) == 0 or float(train_y.sum().item()) == float(train_y.numel()):
            fold_scores = [float(train_y.mean().item()) for _ in test_idx]
        else:
            torch.manual_seed(238)
            model = train_logistic(train_x, train_y, epochs, lr)
            with torch.no_grad():
                fold_scores = torch.sigmoid(model(test_x).squeeze(1)).tolist()
        for idx, score in zip(test_idx, fold_scores):
            scores[idx] = float(score)
        fold_labels = [int(labels[idx]) for idx in test_idx]
        fold_auc = auroc(fold_labels, [float(score) for score in fold_scores])
        fold_pr = auprc(fold_labels, [float(score) for score in fold_scores])
        per_fold.append({
            "fold_id": fold_id,
            "sample_count": len(test_idx),
            "unsafe_count": sum(fold_labels),
            "unsafe_AUROC": fold_auc,
            "unsafe_AUPRC": fold_pr,
            "fold_pass": bool(fold_auc is not None and fold_auc >= 0.90 and fold_pr is not None and fold_pr >= 0.50),
        })
    return scores, per_fold


def score_summary(rows: list[dict[str, Any]], labels: list[int], scores: list[float], per_fold: list[dict[str, Any]], feature_cols: list[str], *, variant: str) -> dict[str, Any]:
    threshold, unsafe_recall, unsafe_fpr = threshold_at_fpr(labels, scores, 0.05)
    predicted = [score >= threshold for score in scores]
    severe_rows = [i for i, row in enumerate(rows) if int(row["severe_label"]) == 1]
    strong_rows = [i for i, row in enumerate(rows) if int(row["strong_reference_unsafe_label"]) == 1]
    severe_recall = sum(1 for i in severe_rows if predicted[i]) / len(severe_rows) if severe_rows else None
    strong_recall = sum(1 for i in strong_rows if predicted[i]) / len(strong_rows) if strong_rows else None
    easy_pred = [i for i, row in enumerate(rows) if row["hardness_bucket"] == "easy" and predicted[i]]
    easy_noop_precision = sum(1 for i in easy_pred if int(rows[i]["noop_label"]) == 1) / len(easy_pred) if easy_pred else None
    fold_pass_count = sum(1 for row in per_fold if row["fold_pass"])
    auc = auroc(labels, scores)
    pr = auprc(labels, scores)
    return {
        "variant": variant,
        "feature_count": len(feature_cols),
        "unsafe_count": sum(labels),
        "unsafe_base_rate": sum(labels) / len(labels),
        "unsafe_AUROC": auc,
        "unsafe_AUPRC": pr,
        "threshold_at_FPR_0p05": threshold,
        "unsafe_recall_at_FPR_0p05": unsafe_recall,
        "unsafe_FPR_at_threshold": unsafe_fpr,
        "severe_recall_at_FPR_0p05": severe_recall,
        "strong_reference_unsafe_recall_at_FPR_0p05": strong_recall,
        "easy_noop_precision": easy_noop_precision,
        "fold_pass": f"{fold_pass_count}/5",
        "fold_pass_count": fold_pass_count,
        "gate_pass": bool(
            auc is not None and auc >= 0.90
            and pr is not None and pr >= 0.50
            and severe_recall is not None and severe_recall >= 0.80
            and strong_recall is not None and strong_recall >= 0.80
            and easy_noop_precision is not None and easy_noop_precision >= 0.90
            and fold_pass_count >= 4
        ),
        "fold_summary": per_fold,
    }


def feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    proxy = [
        "input_luma_mean",
        "input_luma_std",
        "input_saturation_mean",
        "input_dark_channel_mean",
        "input_edge_energy",
        "input_lowfreq_energy",
        "A0_luma_mean",
        "A0_luma_std",
        "A0_saturation_mean",
        "A0_dark_channel_mean",
        "A0_edge_energy",
        "A0_lowfreq_energy",
        "input_A0_abs_mean",
        "input_A0_lowfreq_abs_mean",
        "input_A0_hf_abs_mean",
    ]
    proxy = [col for col in proxy if col in feature_cols]
    residual = [col for col in feature_cols if col.startswith(("input_clip", "A0_clip", "input_A0_resid", "A0_r_mean", "A0_g_mean", "A0_b_mean", "input_r_mean", "input_g_mean", "input_b_mean"))]
    internal = [col for col in feature_cols if col.startswith("ConvIR_")]
    return {
        "proxy15_replay": proxy,
        "proxy_plus_residual": proxy + residual,
        "proxy_plus_internal": proxy + internal,
        "all_rich_target_only": proxy + residual + internal,
    }


def forbidden_feature_hits(cols: list[str]) -> list[str]:
    hits = []
    for col in cols:
        low = col.lower()
        if any(token in low for token in FORBIDDEN_DEPLOY_TOKENS):
            hits.append(col)
    return hits


def coerce_numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            number = fnum(value)
            out[key] = number if number is not None else value
        out_rows.append(out)
    return out_rows


def run_audit(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(238)
    random.seed(238)
    p4_rows = read_csv(args.p4_features)
    p0_unique = {row["image_id"]: row for row in read_csv(args.p0_csv) if abs((fnum(row.get("alpha")) or -1.0) - 0.125) < 1e-12}
    manifest_path = args.out_dir / "v238b_p0_rich_target_feature_manifest.csv"
    if args.reuse_feature_manifest and manifest_path.exists():
        rows = coerce_numeric_rows(read_csv(manifest_path))
        print(f"v238b_reuse_feature_manifest rows={len(rows)}", flush=True)
    else:
        device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        model = build_official_base(args.checkpoint, device)
        rows = []
        for index, row in enumerate(p4_rows, start=1):
            image_id = row["image_id"]
            rec = p0_unique[image_id]
            inp = load_image_tensor(rec["input_path"])
            a0 = load_tensor(rec["A0_full_output_path"])
            out: dict[str, Any] = dict(row)
            out.update(image_extra_features(inp, a0))
            with torch.no_grad():
                x, _h, _w = pad_to_factor(inp.to(device), 32)
                feats = capture_convir_features(model, x)
                for name, feat in feats.items():
                    stat_prefix = "ConvIR_" + name
                    out.update(tensor_stats(feat.detach().cpu(), stat_prefix))
            rows.append(out)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index % 25 == 0 or index == len(p4_rows):
                print(f"v238b_feature_progress {index}/{len(p4_rows)} {image_id}", flush=True)
        rows = coerce_numeric_rows(rows)
        write_csv(manifest_path, rows)
    feature_cols = [
        col for col in rows[0]
        if fnum(rows[0].get(col)) is not None
        and col not in {"fold_id", "unsafe_label", "severe_label", "strong_reference_unsafe_label", "eligible_label", "noop_label"}
    ]
    deploy_groups = feature_groups(feature_cols)
    labels = [int(row["unsafe_label"]) for row in rows]
    variant_summaries = []
    per_fold_rows = []
    score_cols = []
    for variant, cols in deploy_groups.items():
        hits = forbidden_feature_hits(cols)
        scores, per_fold = oof_scores(rows, cols, labels, args.epochs, args.lr)
        for row, score in zip(rows, scores):
            row[f"{variant}_oof_score"] = score
        score_cols.append(f"{variant}_oof_score")
        summary = score_summary(rows, labels, scores, per_fold, cols, variant=variant)
        summary["forbidden_feature_hits"] = hits
        summary["forbidden_runtime_features_used"] = bool(hits)
        summary["gate_pass"] = bool(summary["gate_pass"] and not hits)
        variant_summaries.append(summary)
        for item in per_fold:
            per_fold_rows.append({"variant": variant, **item})
    # Controls.
    rich_cols = deploy_groups["all_rich_target_only"]
    shuffled = labels[:]
    random.Random(238).shuffle(shuffled)
    shuffle_scores, shuffle_per_fold = oof_scores(rows, rich_cols, shuffled, args.epochs, args.lr)
    shuffle_summary = score_summary(rows, shuffled, shuffle_scores, shuffle_per_fold, rich_cols, variant="label_shuffle_control")
    base_rate = sum(labels) / len(labels)
    shuffle_summary["near_base_rate_gate_pass"] = bool(shuffle_summary["unsafe_AUPRC"] is not None and shuffle_summary["unsafe_AUPRC"] <= base_rate + 0.10)
    leak_scores = [-(fnum(row.get("base_alpha0p5_delta"), 0.0) or 0.0) for row in rows]
    leak_summary = score_summary(rows, labels, leak_scores, [], ["teacher_delta_positive_control"], variant="teacher_delta_leak_positive_control")
    leak_summary["fold_summary"] = []
    a0_scores = [fnum(p0_unique[row["image_id"]].get("A0_same_context_psnr"), 0.0) or 0.0 for row in rows]
    a0_summary = score_summary(rows, labels, a0_scores, [], ["A0_PSNR_illegal_upper_bound"], variant="A0_PSNR_illegal_upper_bound_control")
    a0_summary["fold_summary"] = []
    write_csv(args.out_dir / "v238b_p0_rich_target_oof_per_fold.csv", per_fold_rows)
    main = next(row for row in variant_summaries if row["variant"] == "all_rich_target_only")
    controls_gate = bool(shuffle_summary["near_base_rate_gate_pass"])
    gate_pass = bool(main["gate_pass"] and controls_gate)
    payload = {
        "route_id": ROUTE_ID,
        "phase": "P0 richer target-only no-op/unsafe separability audit",
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "feature_manifest": str(args.out_dir / "v238b_p0_rich_target_feature_manifest.csv"),
        "feature_variants": variant_summaries,
        "selected_variant": "all_rich_target_only",
        "selected_variant_summary": main,
        "controls": {
            "label_shuffle": shuffle_summary,
            "teacher_delta_leak_positive_control": leak_summary,
            "A0_PSNR_illegal_upper_bound_control": a0_summary,
        },
        "gate": {
            "unsafe_AUROC_min": 0.90,
            "unsafe_AUPRC_min": 0.50,
            "severe_recall_at_FPR_0p05_min": 0.80,
            "strong_reference_unsafe_recall_at_FPR_0p05_min": 0.80,
            "easy_noop_precision_min": 0.90,
            "fold_pass_min": "4/5",
            "shuffle_AUPRC_max": base_rate + 0.10,
            "no_forbidden_feature_hits": True,
        },
        "gate_pass": gate_pass,
        "decision": "P0_PASS_RICH_TARGET_ONLY_SEPARABILITY_DIAGNOSTIC" if gate_pass else "P0_FAIL_RICH_TARGET_ONLY_SEPARABILITY_DIAGNOSTIC",
    }
    write_json(args.out_dir / "v238b_p0_rich_target_separability_summary.json", payload)
    write_json(args.out_dir / "v238b_closeout.json", {
        "route_id": ROUTE_ID,
        "inherited_reference": "v2.37 P4_FAIL_STOP_TARGET_ONLY_NOOP_UNSAFE_NOT_SEPARABLE and v2.38 P0_FAIL_NO_MICROALPHA_SAFE_SUBSTRATE",
        "locked_test_touched": False,
        "bridge_training_authorized": False,
        "generator_training_authorized": False,
        "canary80_authorized": False,
        "rich_target_only_separability_pass": gate_pass,
        "decision": payload["decision"],
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("V238B_RICH_TARGET_ONLY_SEPARABILITY_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--p0-csv", type=Path, required=True)
    parser.add_argument("--p4-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reuse-feature-manifest", action="store_true")
    return parser


def main() -> None:
    run_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
