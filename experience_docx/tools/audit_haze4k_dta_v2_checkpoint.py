#!/usr/bin/env python3
"""Audit DTA-v2 checkpoint internals against Haze4K transmission maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader
from models.ConvIR import build_net


def is_name_field(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str)


def unpack_batch(data):
    name = data[-1] if is_name_field(data[-1]) else None
    if name is not None:
        data = data[:-1]
    input_img, label_img = data[0], data[1]
    depth = data[2] if len(data) >= 3 else None
    trans = data[3] if len(data) >= 4 else None
    airlight = data[4] if len(data) >= 5 else None
    return input_img, label_img, depth, trans, airlight, name


def load_model_state(path: str, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_dta_v2(args):
    return build_net(
        "base",
        "Haze4K",
        "original",
        arch="dta_v2",
        dta_variant="v2",
        dta_prior_channels=args.dta_prior_channels,
        dta_gate_bias=args.dta_gate_bias,
        dta_gate_limit=args.dta_gate_limit,
        dta_gamma_limit=args.dta_gamma_limit,
        dta_beta_limit=args.dta_beta_limit,
        dta_alpha_init=args.dta_alpha_init,
        dta_depth_mode=args.dta_depth_mode,
        dta_confidence_floor=args.dta_confidence_floor,
        dta_confidence_local_scale=args.dta_confidence_local_scale,
        dta_output_residual_scale=args.dta_output_residual_scale,
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    n = len(values)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def sampled_corr(a: torch.Tensor, b: torch.Tensor, max_pixels: int, seed: int) -> float:
    a_np = a.detach().float().cpu().reshape(-1).numpy()
    b_np = b.detach().float().cpu().reshape(-1).numpy()
    keep = np.isfinite(a_np) & np.isfinite(b_np)
    a_np = a_np[keep]
    b_np = b_np[keep]
    if a_np.size < 2:
        return float("nan")
    if max_pixels > 0 and a_np.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(a_np.size, size=max_pixels, replace=False)
        a_np = a_np[idx]
        b_np = b_np[idx]
    return spearman(a_np, b_np)


def finite_mean(values: list[float]) -> float | None:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return None
    return float(statistics.mean(values))


def finite_median(values: list[float]) -> float | None:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return None
    return float(statistics.median(values))


def audit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dta_v2(args).to(device)
    model.load_state_dict(load_model_state(args.checkpoint, device))
    model.eval()
    depth_split = args.depth_split
    if args.split_json and args.split_name and depth_split == "test":
        depth_split = "train"
    loader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=depth_split,
        root_split=args.eval_root_split,
        return_trans=True,
        return_meta=True,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    rows = []
    factor = 32
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, _, depth, trans, airlight, name = unpack_batch(data)
            if args.dta_depth_mode == "shuffle":
                shuffle_idx = (idx + args.depth_shuffle_offset) % len(loader.dataset)
                _, _, shuffled_depth, _, _, _ = unpack_batch(loader.dataset[shuffle_idx])
                depth = shuffled_depth.unsqueeze(0)
            input_img = input_img.to(device)
            depth = depth.to(device)
            trans = trans.to(device).float().clamp(1e-4, 1.0)
            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
            depth = F.pad(depth, (0, padw, 0, padh), "reflect")
            trans_pad = F.pad(trans, (0, padw, 0, padh), "reflect")
            outputs = model(padded, depth)
            aux = model.dta_auxiliary_losses(rank_pairs=args.rank_pairs, min_depth_gap=args.rank_min_depth_gap)
            last_aux = model.DTA.last_aux
            t_pred = last_aux["t_pred"].detach().clamp(1e-4, 1.0)
            t_gt = F.interpolate(trans_pad, size=t_pred.shape[-2:], mode="bilinear", align_corners=False)
            depth_stage = last_aux["depth"].detach()
            confidence = last_aux.get("confidence")
            diff = t_pred - t_gt
            stats = model.DTA.stats()
            row = {
                "name": name[0] if name else str(idx),
                "t_l1": float(diff.abs().mean().cpu()),
                "t_rmse": float(torch.sqrt((diff * diff).mean()).cpu()),
                "t_pred_mean": float(t_pred.mean().cpu()),
                "t_gt_mean": float(t_gt.mean().cpu()),
                "t_pred_std": float(t_pred.std(unbiased=False).cpu()),
                "t_gt_std": float(t_gt.std(unbiased=False).cpu()),
                "spearman_tpred_tgt": sampled_corr(t_pred, t_gt, args.max_corr_pixels, args.seed + idx),
                "spearman_depth_neglogt": sampled_corr(depth_stage, -torch.log(t_gt), args.max_corr_pixels, args.seed + idx + 100000),
                "rank_loss": float(aux["rank"].detach().cpu()),
                "tv_loss": float(aux["tv"].detach().cpu()),
                "proxy_loss": float(aux["proxy"].detach().cpu()),
                "confidence_mean": float(confidence.mean().cpu()) if confidence is not None else float("nan"),
                "airlight": float(airlight.item()) if hasattr(airlight, "item") else float("nan"),
            }
            row.update(stats)
            rows.append(row)
            if (idx + 1) % 100 == 0:
                print(f"tpred_audit {idx + 1}/{len(loader)} t_l1={finite_mean([r['t_l1'] for r in rows]):.6f}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--eval_root_split", default="test", choices=["train", "test"])
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="dta_v2")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_corr_pixels", type=int, default=65536)
    parser.add_argument("--depth_shuffle_offset", type=int, default=137)
    parser.add_argument("--dta_depth_mode", default="normal", choices=["normal", "invert", "zero", "shuffle"])
    parser.add_argument("--dta_prior_channels", type=int, default=32)
    parser.add_argument("--dta_gate_bias", type=float, default=-6.0)
    parser.add_argument("--dta_gate_limit", type=float, default=0.06)
    parser.add_argument("--dta_gamma_limit", type=float, default=0.12)
    parser.add_argument("--dta_beta_limit", type=float, default=0.06)
    parser.add_argument("--dta_alpha_init", type=float, default=1.0)
    parser.add_argument("--dta_confidence_floor", type=float, default=0.25)
    parser.add_argument("--dta_confidence_local_scale", type=float, default=6.0)
    parser.add_argument("--dta_output_residual_scale", type=float, default=0.03)
    parser.add_argument("--rank_pairs", type=int, default=512)
    parser.add_argument("--rank_min_depth_gap", type=float, default=0.03)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit(args)
    csv_path = output_dir / f"dta_v2_tpred_quality_{args.tag}.csv"
    json_path = output_dir / f"dta_v2_tpred_quality_{args.tag}.json"
    write_csv(csv_path, rows)
    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "count": len(rows),
        "config": vars(args),
        "metrics": {},
        "outputs": {"csv": csv_path.name, "json": json_path.name},
    }
    for key in rows[0].keys() if rows else []:
        if key == "name":
            continue
        values = [row[key] for row in rows]
        summary["metrics"][f"{key}_mean"] = finite_mean(values)
        summary["metrics"][f"{key}_median"] = finite_median(values)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("DTA_V2_TPRED_QUALITY_AUDIT_OK")


if __name__ == "__main__":
    main()
