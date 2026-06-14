#!/usr/bin/env python3
"""C2 output-difference feature audit and abstaining router screen.

This script renders A0 and official FullUDP in memory on the train-derived
internal-validation splits, extracts deployable image/depth/output-difference
features, and runs a 5-fold threshold-router screen. It does not write raw
images or tensors and it does not touch the locked Haze4K test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_udpnet_v15_phase0_repro import (  # noqa: E402
    infer_one,
    load_a0_model,
    load_convir_builders,
    load_udpnet_builder,
    load_udpnet_model,
    metric_pair,
    sha256_file,
)


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


POLICY_FEATURES = [
    "input_mean",
    "input_std",
    "input_grad_mean",
    "input_dark_mean",
    "depth_mean",
    "depth_std",
    "depth_grad_mean",
    "a0_mean",
    "a0_std",
    "a0_grad_mean",
    "a0_saturation_high",
    "a0_saturation_low",
    "udp_mean",
    "udp_std",
    "udp_grad_mean",
    "diff_signed_mean",
    "diff_abs_mean",
    "diff_abs_std",
    "diff_abs_p50",
    "diff_abs_p90",
    "diff_abs_p95",
    "diff_abs_max",
    "diff_grad_mean",
    "diff_to_a0_ratio",
    "a0_udp_psnr",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
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


def fold_id(image_id: str, folds: int = 5) -> int:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def grad_mean(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    return 0.5 * (dx + dy)


def tensor_std(x: torch.Tensor) -> torch.Tensor:
    return x.float().std(unbiased=False)


def tensor_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(x.flatten().float(), q)


def psnr_between(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def load_split_names(split_json: Path, split: str) -> list[str]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    if split not in splits:
        raise KeyError(f"split {split} not found in {split_json}")
    names = splits[split]
    if names and isinstance(names[0], dict):
        names = [row.get("name") or row.get("image") for row in names]
    return [os.path.basename(str(name)) for name in names if name]


def label_path(label_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        candidates.extend([f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"])
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"no label for {image_name} in {label_dir}; tried {candidates}")


def depth_path(depth_cache: Path, depth_split: str, image_name: str) -> Path:
    candidates = [
        depth_cache / depth_split / f"{image_name.replace('/', '__')}.npy",
        depth_cache / depth_split / f"{image_name}.npy",
        depth_cache / f"{image_name.replace('/', '__')}.npy",
        depth_cache / f"{image_name}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing depth cache for {image_name}; tried {candidates}")


def normalize_depth_minmax(depth: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(depth))
    hi = float(np.nanmax(depth))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo + 1e-6)).astype(np.float32)


def load_sample(data_dir: Path, depth_cache: Path, image_name: str, depth_split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_root = data_dir / "train"
    input_dir = first_existing_dir(split_root, ("IN", "haze", "hazy"))
    gt_dir = first_existing_dir(split_root, ("GT", "gt"))
    image = Image.open(input_dir / image_name).convert("RGB")
    label = Image.open(label_path(gt_dir, image_name)).convert("RGB")
    depth_arr = np.load(depth_path(depth_cache, depth_split, image_name)).astype(np.float32)
    depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if depth_arr.ndim == 3:
        depth_arr = np.squeeze(depth_arr)
    # DepthAnything cache is metric-like float; UDPNet's depth2l contract is [0, 1].
    depth_arr = normalize_depth_minmax(depth_arr)
    depth_img = Image.fromarray(depth_arr, mode="F")
    if depth_img.size != image.size:
        depth_img = depth_img.resize(image.size, resample=Image.BICUBIC)
    return TVF.to_tensor(image), TVF.to_tensor(label), TVF.to_tensor(depth_img).float()


def feature_dict(input_img: torch.Tensor, depth: torch.Tensor, a0_pred: torch.Tensor, udp_pred: torch.Tensor) -> dict[str, float]:
    diff = udp_pred - a0_pred
    abs_diff = diff.abs()
    a0_abs_mean = a0_pred.abs().mean().clamp_min(1e-6)
    dark = input_img.min(dim=1, keepdim=True).values
    return {
        "input_mean": float(input_img.mean().item()),
        "input_std": float(tensor_std(input_img).item()),
        "input_grad_mean": float(grad_mean(input_img).item()),
        "input_dark_mean": float(dark.mean().item()),
        "depth_mean": float(depth.mean().item()),
        "depth_std": float(tensor_std(depth).item()),
        "depth_grad_mean": float(grad_mean(depth).item()),
        "a0_mean": float(a0_pred.mean().item()),
        "a0_std": float(tensor_std(a0_pred).item()),
        "a0_grad_mean": float(grad_mean(a0_pred).item()),
        "a0_saturation_high": float((a0_pred >= 0.98).float().mean().item()),
        "a0_saturation_low": float((a0_pred <= 0.02).float().mean().item()),
        "udp_mean": float(udp_pred.mean().item()),
        "udp_std": float(tensor_std(udp_pred).item()),
        "udp_grad_mean": float(grad_mean(udp_pred).item()),
        "diff_signed_mean": float(diff.mean().item()),
        "diff_abs_mean": float(abs_diff.mean().item()),
        "diff_abs_std": float(tensor_std(abs_diff).item()),
        "diff_abs_p50": float(tensor_quantile(abs_diff, 0.50).item()),
        "diff_abs_p90": float(tensor_quantile(abs_diff, 0.90).item()),
        "diff_abs_p95": float(tensor_quantile(abs_diff, 0.95).item()),
        "diff_abs_max": float(abs_diff.max().item()),
        "diff_grad_mean": float(grad_mean(diff).abs().item()),
        "diff_to_a0_ratio": float((abs_diff.mean() / a0_abs_mean).item()),
        "a0_udp_psnr": psnr_between(a0_pred, udp_pred),
    }


def summarize_policy(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    count = len(rows)
    selected_count = 0
    selected_positive = 0
    selected_nonnegative = 0
    selected_severe = 0
    deltas: list[float] = []
    ssims: list[float] = []
    for row in rows:
        choose = predicate(row)
        if choose:
            selected_count += 1
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
    coverage = selected_count / count if count else 0.0
    return {
        "count": count,
        "selected_count": selected_count,
        "coverage": coverage,
        "mean_dPSNR": statistics.mean(deltas) if deltas else 0.0,
        "hard_bottom25_dPSNR": statistics.mean([deltas[i] for i in order[:k]]) if deltas else 0.0,
        "easy_top25_dPSNR": statistics.mean([deltas[i] for i in order[-k:]]) if deltas else 0.0,
        "dSSIM": statistics.mean(ssims) if ssims else 0.0,
        "positive_ratio": sum(1 for d in deltas if d > 0.0) / count if count else 0.0,
        "nonnegative_ratio": sum(1 for d in deltas if d >= 0.0) / count if count else 0.0,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / count * 600.0 if count else 0.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / count * 600.0 if count else 0.0,
        "selected_precision": selected_positive / selected_count if selected_count else 0.0,
        "selected_nonnegative_ratio": selected_nonnegative / selected_count if selected_count else 1.0,
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


def policy_grid(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
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
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]
    for feature in features:
        vals = [float(r[feature]) for r in rows if math.isfinite(float(r[feature]))]
        if not vals:
            continue
        thresholds = sorted({quantile(vals, q) for q in quantiles})
        for threshold in thresholds:
            add(f"{feature}_le_{threshold:.6g}", lambda r, feature=feature, threshold=threshold: float(r[feature]) <= threshold)
            add(f"{feature}_ge_{threshold:.6g}", lambda r, feature=feature, threshold=threshold: float(r[feature]) >= threshold)
    policies.sort(key=lambda r: (bool(r["strict_gate_pass"]), bool(r["abstention_gate_pass"]), fnum(r["score"])), reverse=True)
    return policies


def parse_policy(policy_id: str) -> Callable[[dict[str, Any]], bool]:
    if policy_id == "all_fulludp":
        return lambda _r: True
    if policy_id == "a0_anchor":
        return lambda _r: False
    if "_le_" in policy_id:
        feature, threshold = policy_id.rsplit("_le_", 1)
        return lambda r, feature=feature, threshold=float(threshold): float(r[feature]) <= threshold
    if "_ge_" in policy_id:
        feature, threshold = policy_id.rsplit("_ge_", 1)
        return lambda r, feature=feature, threshold=float(threshold): float(r[feature]) >= threshold
    return lambda _r: False


def choose_policy(train_rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    grid = policy_grid(train_rows, features)
    strict = [r for r in grid if r["strict_gate_pass"]]
    if strict:
        return strict[0]
    abstention = [r for r in grid if r["abstention_gate_pass"]]
    if abstention:
        return abstention[0]
    return grid[0]


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = [(v, y) for v, y in zip(values, labels, strict=False) if math.isfinite(v)]
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if not pairs or pos == 0 or neg == 0:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def render_feature_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    convir_its_dir = Path(args.convir_its_dir)
    _test_dataloader, build_convir_net = load_convir_builders(convir_its_dir)
    build_udpnet = load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udpnet_model, ckpt_meta = load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    start_time = time.time()
    for split in args.splits:
        names = load_split_names(Path(args.split_json), split)
        depth_split = "train" if args.split_json else args.depth_split
        with torch.no_grad():
            for idx, image_name in enumerate(names):
                input_img, label_img, depth = load_sample(
                    Path(args.data_dir),
                    Path(args.depth_cache_dir),
                    image_name,
                    depth_split,
                )
                input_img = input_img.unsqueeze(0).to(device)
                label_img = label_img.unsqueeze(0).to(device)
                depth = depth.unsqueeze(0).to(device)
                h, w = input_img.shape[2], input_img.shape[3]
                h_pad = ((h + factor) // factor) * factor
                w_pad = ((w + factor) // factor) * factor
                padh = h_pad - h if h % factor != 0 else 0
                padw = w_pad - w if w % factor != 0 else 0
                rgb_padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
                depth_padded = F.pad(depth, (0, padw, 0, padh), "reflect")
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)
                a0_pred = infer_one(a0_model, rgb_padded, h, w)
                udp_pred = infer_one(udpnet_model, udp_input, h, w)
                a0_psnr, a0_ssim = metric_pair(a0_pred, label_img, (h_pad, w_pad))
                udp_psnr, udp_ssim = metric_pair(udp_pred, label_img, (h_pad, w_pad))
                rec: dict[str, Any] = {
                    "name": image_name,
                    "split": split,
                    "A0_PSNR": a0_psnr,
                    "FullUDP_PSNR": udp_psnr,
                    "dPSNR": udp_psnr - a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "FullUDP_SSIM": udp_ssim,
                    "dSSIM": udp_ssim - a0_ssim,
                }
                rec.update(feature_dict(input_img, depth, a0_pred, udp_pred))
                rows.append(rec)
                if (idx + 1) % args.print_freq == 0:
                    mean_delta = statistics.mean(float(r["dPSNR"]) for r in rows)
                    print(f"{split} {idx + 1}/{len(names)} rows={len(rows)} mean_delta={mean_delta:.4f}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    meta = {
        "elapsed_sec": time.time() - start_time,
        "device": str(device),
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint": str(args.official_checkpoint),
        "official_sha256": sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "udp_repo": str(args.udp_repo),
        "depth_normalization": "per_image_minmax_to_udpnet_depth2l_contract",
        "splits": args.splits,
        "split_json": args.split_json,
        "locked_test_touched": False,
    }
    return rows, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convir_its_dir", required=True)
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--splits", nargs="+", default=["val_regular", "val_hard"])
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, meta = render_feature_rows(args)
    row_fields = [
        "name",
        "split",
        "A0_PSNR",
        "FullUDP_PSNR",
        "dPSNR",
        "A0_SSIM",
        "FullUDP_SSIM",
        "dSSIM",
    ] + POLICY_FEATURES
    write_csv(out_dir / "v20_c2_outputdiff_feature_rows.csv", rows, row_fields)

    feature_auc_rows: list[dict[str, Any]] = []
    targets = {
        "target_positive": [int(float(r["dPSNR"]) > 0.0) for r in rows],
        "target_high_gain_ge_0.20": [int(float(r["dPSNR"]) >= 0.20) for r in rows],
        "target_severe_loss_le_-0.20": [int(float(r["dPSNR"]) <= -0.20) for r in rows],
        "target_loss_lt_0": [int(float(r["dPSNR"]) < 0.0) for r in rows],
    }
    for feature in POLICY_FEATURES:
        vals = [float(r[feature]) for r in rows]
        for target, labels in targets.items():
            auc = auc_score(vals, labels)
            feature_auc_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "auc_raw": auc,
                    "auc_best_orientation": max(auc, 1.0 - auc) if math.isfinite(auc) else "",
                    "direction": "higher" if math.isfinite(auc) and auc >= 0.5 else "lower",
                }
            )
    write_csv(out_dir / "v20_c2_feature_auc.csv", feature_auc_rows, ["feature", "target", "auc_raw", "auc_best_orientation", "direction"])

    policies = policy_grid(rows, POLICY_FEATURES)
    policy_fields = [
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
    write_csv(out_dir / "v20_c2_policy_grid.csv", policies, policy_fields)

    fold_rows: list[dict[str, Any]] = []
    selected_oof: set[str] = set()
    for fold in range(5):
        train = [r for r in rows if fold_id(str(r["name"])) != fold]
        heldout = [r for r in rows if fold_id(str(r["name"])) == fold]
        chosen = choose_policy(train, POLICY_FEATURES)
        predicate = parse_policy(str(chosen["policy_id"]))
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
        fold_rows.append(eval_rec)
        for row in heldout:
            if predicate(row):
                selected_oof.add(str(row["name"]))
    write_csv(out_dir / "v20_c2_oof_fold_metrics.csv", fold_rows, ["fold", "train_policy_id", "train_strict_gate_pass", "train_abstention_gate_pass"] + policy_fields[1:])

    oof_summary = summarize_policy(rows, lambda r: str(r["name"]) in selected_oof)
    oof_summary["strict_gate_pass"] = strict_gate_pass(oof_summary)
    oof_summary["abstention_gate_pass"] = abstention_gate_pass(oof_summary)
    oof_summary["score"] = score(oof_summary)
    strict_pass = [r for r in policies if r["strict_gate_pass"]]
    abstention_pass = [r for r in policies if r["abstention_gate_pass"]]
    if oof_summary["strict_gate_pass"]:
        decision = "C2_OUTPUTDIFF_STRICT_SCREEN_PASS_START_C3_SHIFTED"
    elif oof_summary["abstention_gate_pass"]:
        decision = "C2_OUTPUTDIFF_ABSTENTION_SCREEN_PASS_START_C3_SHIFTED"
    elif abstention_pass:
        decision = "C2_OUTPUTDIFF_IN_SAMPLE_ONLY_FAIL_OOF"
    else:
        decision = "C2_OUTPUTDIFF_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C2 OutputDiff Router Screen",
        "locked_test_touched": False,
        "meta": meta,
        "rows": len(rows),
        "policy_features": POLICY_FEATURES,
        "strict_gate": STRICT_GATE,
        "abstention_gate": ABSTENTION_GATE,
        "best_policy": policies[0] if policies else None,
        "best_strict_policy": strict_pass[0] if strict_pass else None,
        "best_abstention_policy": abstention_pass[0] if abstention_pass else None,
        "oof_summary": oof_summary,
        "decision": decision,
    }
    (out_dir / "v20_c2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Haze4K v2.0 C2 OutputDiff Router Screen",
        "",
        f"Decision: `{decision}`",
        "",
        "This phase rendered A0 and official FullUDP in memory on internal-validation splits only.",
        "No raw images/tensors were written, and locked test data was not touched.",
        "",
        "## Best In-Sample Policy",
        "",
    ]
    for key in policy_fields:
        if policies and key in policies[0]:
            lines.append(f"- `{key}`: `{policies[0][key]}`")
    lines.extend(["", "## OOF Replay", ""])
    for key, value in oof_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C3 shifted validation is authorized only if the OOF screen passes.",
            "- If OOF fails, do not touch locked test; improve features or expert compatibility first.",
        ]
    )
    (out_dir / "v20_c2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C2_OUTPUTDIFF_ROUTER_OK decision={decision} rows={len(rows)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
