#!/usr/bin/env python3
"""C7b train-derived patch/local-alpha prototype.

This is the first deployable local-alpha prototype after C7 patch oracle. It
renders A0/FullUDP on internal validation splits only, builds patch-level
features and alpha SSE targets, chooses a transparent patch policy using
image-fold OOF, then re-renders held-out images to measure true PSNR/SSIM. It
writes text evidence only and never touches locked Haze4K test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import QUANTILES, fnum, fold_id, write_csv

ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.75]
LOW_ALPHAS = [0.125, 0.25, 0.375]
HIGH_ALPHAS = [0.375, 0.50, 0.75]
FEATURES = [
    "input_mean", "input_std", "input_grad_mean", "input_dark_mean",
    "depth_mean", "depth_std", "depth_grad_mean",
    "a0_mean", "a0_std", "a0_grad_mean",
    "udp_mean", "udp_std", "udp_grad_mean",
    "diff_signed_mean", "diff_abs_mean", "diff_abs_std", "diff_abs_p90",
    "diff_abs_max", "diff_grad_mean", "diff_to_a0_ratio", "a0_udp_psnr",
]

C7B_SCREEN_GATE = {
    "mean_dPSNR": 0.20,
    "hard_bottom25_dPSNR": 0.28,
    "easy_top25_dPSNR": 0.0,
    "dSSIM": 0.0,
    "positive_ratio": 0.69,
    "severe_loss_per_600": 48.0,
}

C7B_STRONG_GATE = {
    "mean_dPSNR": 0.20,
    "hard_bottom25_dPSNR": 0.30,
    "easy_top25_dPSNR": 0.0,
    "dSSIM": 0.0,
    "positive_ratio": 0.70,
    "severe_loss_per_600": 48.0,
}


def alpha_key(alpha: float) -> str:
    text = ("%.6f" % alpha).rstrip("0").rstrip(".")
    return f"a{text.replace('.', 'p')}"


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def tensor_std(x: torch.Tensor) -> float:
    return float(x.float().std(unbiased=False).item())


def grad_mean(x: torch.Tensor) -> float:
    return float(c2.grad_mean(x).item())


def tensor_q(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten().float(), q).item())


def patch_features(input_p: torch.Tensor, depth_p: torch.Tensor, a0_p: torch.Tensor, udp_p: torch.Tensor) -> dict[str, float]:
    diff = udp_p - a0_p
    abs_diff = diff.abs()
    dark = input_p.min(dim=1, keepdim=True).values
    return {
        "input_mean": float(input_p.mean().item()),
        "input_std": tensor_std(input_p),
        "input_grad_mean": grad_mean(input_p),
        "input_dark_mean": float(dark.mean().item()),
        "depth_mean": float(depth_p.mean().item()),
        "depth_std": tensor_std(depth_p),
        "depth_grad_mean": grad_mean(depth_p),
        "a0_mean": float(a0_p.mean().item()),
        "a0_std": tensor_std(a0_p),
        "a0_grad_mean": grad_mean(a0_p),
        "udp_mean": float(udp_p.mean().item()),
        "udp_std": tensor_std(udp_p),
        "udp_grad_mean": grad_mean(udp_p),
        "diff_signed_mean": float(diff.mean().item()),
        "diff_abs_mean": float(abs_diff.mean().item()),
        "diff_abs_std": tensor_std(abs_diff),
        "diff_abs_p90": tensor_q(abs_diff, 0.90),
        "diff_abs_max": float(abs_diff.max().item()),
        "diff_grad_mean": grad_mean(diff),
        "diff_to_a0_ratio": float((abs_diff.mean() / a0_p.abs().mean().clamp_min(1e-6)).item()),
        "a0_udp_psnr": float(c2.psnr_between(a0_p, udp_p)),
    }


def gate_pass(row: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= gate["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= gate["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= gate["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= gate["dSSIM"]
        and fnum(row.get("positive_ratio")) >= gate["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= gate["severe_loss_per_600"]
    )


def proxy_gate_pass(row: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= gate["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= gate["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= gate["easy_top25_dPSNR"]
        and fnum(row.get("positive_ratio")) >= gate["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= gate["severe_loss_per_600"]
    )


def score(row: dict[str, Any]) -> float:
    hard = fnum(row.get("hard_bottom25_dPSNR"))
    pos = fnum(row.get("positive_ratio"))
    easy = fnum(row.get("easy_top25_dPSNR"))
    severe = fnum(row.get("severe_loss_per_600"))
    penalty = 0.0
    penalty += 2.0 * max(0.0, C7B_STRONG_GATE["hard_bottom25_dPSNR"] - hard)
    penalty += 1.5 * max(0.0, C7B_STRONG_GATE["positive_ratio"] - pos)
    penalty += 3.0 * max(0.0, -easy)
    penalty += 0.012 * max(0.0, severe - C7B_STRONG_GATE["severe_loss_per_600"])
    return fnum(row.get("mean_dPSNR")) + 0.8 * hard + 0.45 * pos + 0.1 * easy - 0.002 * severe - penalty


class PatchTable:
    def __init__(self, patch_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]):
        self.patch_rows = patch_rows
        self.image_rows = image_rows
        self.image_names = [str(row["name"]) for row in image_rows]
        self.image_index = {name: i for i, name in enumerate(self.image_names)}
        self.patch_image_idx = np.array([self.image_index[str(row["name"])] for row in patch_rows], dtype=np.int64)
        self.a0_psnr = np.array([fnum(row["A0_PSNR"]) for row in image_rows], dtype=np.float64)
        self.features = {feat: np.array([fnum(row[feat]) for row in patch_rows], dtype=np.float64) for feat in FEATURES}
        self.pixels = np.array([fnum(row["pixel_count"]) for row in patch_rows], dtype=np.float64)
        self.sse = np.zeros((len(patch_rows), len(ALPHAS)), dtype=np.float64)
        for j, alpha in enumerate(ALPHAS):
            self.sse[:, j] = np.array([fnum(row[f"sse_{alpha_key(alpha)}"]) for row in patch_rows], dtype=np.float64)
        self.alpha_to_idx = {alpha: i for i, alpha in enumerate(ALPHAS)}

    def subset_images(self, keep_images: np.ndarray) -> "PatchTable":
        keep_images = np.asarray(keep_images, dtype=bool)
        keep_names = {name for name, keep in zip(self.image_names, keep_images, strict=False) if keep}
        patch_rows = [row for row in self.patch_rows if str(row["name"]) in keep_names]
        image_rows = [row for row in self.image_rows if str(row["name"]) in keep_names]
        return PatchTable(patch_rows, image_rows)


def summarize_patch_actions(table: PatchTable, actions: np.ndarray) -> dict[str, Any]:
    image_sse = np.zeros(len(table.image_rows), dtype=np.float64)
    image_pix = np.zeros(len(table.image_rows), dtype=np.float64)
    image_selected = np.zeros(len(table.image_rows), dtype=bool)
    for pidx, action in enumerate(np.asarray(actions, dtype=np.int64)):
        iidx = table.patch_image_idx[pidx]
        image_sse[iidx] += table.sse[pidx, action]
        image_pix[iidx] += table.pixels[pidx]
        if action > 0:
            image_selected[iidx] = True
    psnr = np.array([psnr_from_mse(s / max(1.0, p)) for s, p in zip(image_sse, image_pix, strict=False)])
    deltas = psnr - table.a0_psnr
    selected_count = int(image_selected.sum())
    order = np.argsort(table.a0_psnr)
    bucket = max(1, len(table.image_rows) // 4)
    selected_d = deltas[image_selected]
    severe = int(np.sum(deltas <= -0.20))
    strong = int(np.sum(deltas <= -0.05))
    rec = {
        "count": len(table.image_rows),
        "selected_count": selected_count,
        "coverage": selected_count / max(1, len(table.image_rows)),
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean(deltas[order[:bucket]])),
        "easy_top25_dPSNR": float(np.mean(deltas[order[-bucket:]])),
        "positive_ratio": float(np.mean(deltas > 0.0)),
        "nonnegative_ratio": float(np.mean(deltas >= 0.0)),
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / max(1, len(deltas)) * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / max(1, len(deltas)) * 600.0,
        "selected_precision": float(np.mean(selected_d > 0.0)) if selected_d.size else 0.0,
        "selected_nonnegative_ratio": float(np.mean(selected_d >= 0.0)) if selected_d.size else 1.0,
        "selected_severe_count": int(np.sum(selected_d <= -0.20)) if selected_d.size else 0,
    }
    for alpha in ALPHAS:
        idx = table.alpha_to_idx[alpha]
        rec[f"patch_action_count_{alpha_key(alpha)}"] = int(np.sum(actions == idx))
        rec[f"patch_action_fraction_{alpha_key(alpha)}"] = float(np.mean(actions == idx)) if len(actions) else 0.0
    return rec


def condition_candidates(table: PatchTable, min_coverage: float = 0.03) -> list[dict[str, Any]]:
    out = [{"condition_id": "all_patches", "mask": np.ones(len(table.patch_rows), dtype=bool)}]
    for feat in FEATURES:
        vals = table.features[feat]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        for threshold in sorted({float(np.quantile(finite, q)) for q in QUANTILES}):
            for direction, mask in (("le", vals <= threshold), ("ge", vals >= threshold)):
                if float(mask.mean()) >= min_coverage:
                    out.append({"condition_id": f"{feat}_{direction}_{threshold:.8g}", "mask": mask})
    return out


def parse_condition(table: PatchTable, condition_id: str) -> np.ndarray:
    if condition_id == "all_patches":
        return np.ones(len(table.patch_rows), dtype=bool)
    if "_le_" in condition_id:
        feat, threshold = condition_id.rsplit("_le_", 1)
        return table.features[feat] <= float(threshold)
    if "_ge_" in condition_id:
        feat, threshold = condition_id.rsplit("_ge_", 1)
        return table.features[feat] >= float(threshold)
    return np.zeros(len(table.patch_rows), dtype=bool)


def apply_policy(table: PatchTable, policy_id: str) -> np.ndarray:
    actions = np.zeros(len(table.patch_rows), dtype=np.int64)
    if policy_id == "a0_anchor":
        return actions
    parts = policy_id.split("|")
    if parts[0] == "single":
        alpha = float(parts[1].removeprefix("alpha="))
        cond = parts[2].removeprefix("cond=")
        actions[parse_condition(table, cond)] = table.alpha_to_idx[alpha]
    elif parts[0] == "tier":
        low_alpha = float(parts[1].removeprefix("low_alpha="))
        low_cond = parts[2].removeprefix("low_cond=")
        high_alpha = float(parts[3].removeprefix("high_alpha="))
        high_cond = parts[4].removeprefix("high_cond=")
        actions[parse_condition(table, low_cond)] = table.alpha_to_idx[low_alpha]
        actions[parse_condition(table, high_cond)] = table.alpha_to_idx[high_alpha]
    return actions


def add_candidate(rows: list[dict[str, Any]], table: PatchTable, policy_id: str, actions: np.ndarray, complexity: int) -> None:
    rec = {"policy_id": policy_id, "complexity": complexity}
    rec.update(summarize_patch_actions(table, actions))
    rec["screen_proxy_pass"] = proxy_gate_pass(rec, C7B_SCREEN_GATE)
    rec["strong_proxy_pass"] = proxy_gate_pass(rec, C7B_STRONG_GATE)
    rec["score"] = score(rec)
    rows.append(rec)


def policy_grid(table: PatchTable, top_k: int, low_pool_limit: int, high_pool_limit: int) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    add_candidate(policies, table, "a0_anchor", np.zeros(len(table.patch_rows), dtype=np.int64), 0)
    conditions = condition_candidates(table)
    singles: list[tuple[dict[str, Any], np.ndarray, float, str]] = []
    for alpha in ALPHAS[1:]:
        for cond in conditions:
            actions = np.zeros(len(table.patch_rows), dtype=np.int64)
            actions[np.asarray(cond["mask"], dtype=bool)] = table.alpha_to_idx[alpha]
            policy_id = f"single|alpha={alpha}|cond={cond['condition_id']}"
            pos = len(policies)
            add_candidate(policies, table, policy_id, actions, 1)
            singles.append((policies[pos], actions, alpha, str(cond["condition_id"])))
    low_pool = []
    high_pool = []
    for rec, actions, alpha, cond in singles:
        patch_cov = 1.0 - fnum(rec.get(f"patch_action_fraction_{alpha_key(0.0)}"))
        if alpha in LOW_ALPHAS and 0.20 <= patch_cov <= 0.95 and fnum(rec["easy_top25_dPSNR"]) >= -0.01:
            low_pool.append((rec, actions, alpha, cond))
        if alpha in HIGH_ALPHAS and 0.03 <= patch_cov <= 0.65 and fnum(rec["selected_precision"]) >= 0.55 and fnum(rec["severe_loss_per_600"]) <= 120.0:
            high_pool.append((rec, actions, alpha, cond))
    low_pool.sort(key=lambda x: (bool(x[0]["strong_proxy_pass"]), bool(x[0]["screen_proxy_pass"]), fnum(x[0]["score"])), reverse=True)
    high_pool.sort(key=lambda x: (fnum(x[0]["hard_bottom25_dPSNR"]), fnum(x[0]["score"])), reverse=True)
    for _lr, low_actions, low_alpha, low_cond in low_pool[:low_pool_limit]:
        for _hr, high_actions, high_alpha, high_cond in high_pool[:high_pool_limit]:
            actions = low_actions.copy()
            actions[high_actions > 0] = table.alpha_to_idx[high_alpha]
            if float(np.mean(actions > 0)) < 0.03:
                continue
            policy_id = f"tier|low_alpha={low_alpha}|low_cond={low_cond}|high_alpha={high_alpha}|high_cond={high_cond}"
            add_candidate(policies, table, policy_id, actions, 2)
    policies.sort(key=lambda r: (bool(r["strong_proxy_pass"]), bool(r["screen_proxy_pass"]), fnum(r["score"])), reverse=True)
    keep = []
    for row in policies:
        if len(keep) < top_k or row["strong_proxy_pass"] or row["screen_proxy_pass"]:
            keep.append(row)
    return keep


def choose_policy(table: PatchTable, top_k: int, low_pool_limit: int, high_pool_limit: int) -> dict[str, Any]:
    grid = policy_grid(table, top_k, low_pool_limit, high_pool_limit)
    strong = [row for row in grid if row["strong_proxy_pass"]]
    if strong:
        return strong[0]
    screen = [row for row in grid if row["screen_proxy_pass"]]
    if screen:
        return screen[0]
    return grid[0]


def iter_patches(h: int, w: int, patch_size: int):
    patch_id = 0
    for y in range(0, h, patch_size):
        y2 = min(h, y + patch_size)
        for x in range(0, w, patch_size):
            x2 = min(w, x + patch_size)
            yield patch_id, y, y2, x, x2
            patch_id += 1


def render_patch_table(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _loader, build_convir_net = c2.load_convir_builders(Path(args.convir_its_dir))
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, ckpt_meta = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    image_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    start = time.time()
    factor = int(args.pad_factor)
    for split in args.splits:
        names = c2.load_split_names(Path(args.split_json), split)
        depth_split = "train" if args.split_json else args.depth_split
        with torch.no_grad():
            for idx, image_name in enumerate(names):
                input_img, label_img, depth = c2.load_sample(Path(args.data_dir), Path(args.depth_cache_dir), image_name, depth_split)
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
                a0_pred = c2.infer_one(a0_model, rgb_padded, h, w)
                udp_pred = c2.infer_one(udp_model, udp_input, h, w)
                a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
                image_rows.append({"name": image_name, "split": split, "A0_PSNR": a0_psnr, "A0_SSIM": a0_ssim, "h": h, "w": w})
                residual = udp_pred - a0_pred
                for patch_id, y, y2, x, x2 in iter_patches(h, w, int(args.patch_size)):
                    input_p = input_img[..., y:y2, x:x2]
                    depth_p = depth[..., y:y2, x:x2]
                    a0_p = a0_pred[..., y:y2, x:x2]
                    udp_p = udp_pred[..., y:y2, x:x2]
                    gt_p = label_img[..., y:y2, x:x2]
                    rec: dict[str, Any] = {
                        "name": image_name,
                        "split": split,
                        "patch_id": patch_id,
                        "y0": y,
                        "y1": y2,
                        "x0": x,
                        "x1": x2,
                        "pixel_count": int(gt_p.numel()),
                    }
                    rec.update(patch_features(input_p, depth_p, a0_p, udp_p))
                    for alpha in ALPHAS:
                        pred = torch.clamp(a0_p + alpha * residual[..., y:y2, x:x2], 0.0, 1.0)
                        rec[f"sse_{alpha_key(alpha)}"] = float(torch.sum((pred - gt_p) ** 2).item())
                    patch_rows.append(rec)
                if (idx + 1) % args.print_freq == 0:
                    print(f"patch_table {split} {idx + 1}/{len(names)} images={len(image_rows)} patches={len(patch_rows)}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    meta = {
        "elapsed_sec": time.time() - start,
        "device": str(device),
        "official_checkpoint_meta": ckpt_meta,
        "depth_normalization": "per_image_minmax_to_udpnet_depth2l_contract",
        "locked_test_touched": False,
    }
    return image_rows, patch_rows, meta


def patch_rows_for_image_tensors(
    image_name: str,
    split: str,
    input_img: torch.Tensor,
    depth: torch.Tensor,
    a0_pred: torch.Tensor,
    udp_pred: torch.Tensor,
    patch_size: int,
) -> list[dict[str, Any]]:
    h, w = input_img.shape[2], input_img.shape[3]
    rows: list[dict[str, Any]] = []
    for patch_id, y, y2, x, x2 in iter_patches(h, w, patch_size):
        rec: dict[str, Any] = {"name": image_name, "split": split, "patch_id": patch_id}
        rec.update(
            patch_features(
                input_img[..., y:y2, x:x2],
                depth[..., y:y2, x:x2],
                a0_pred[..., y:y2, x:x2],
                udp_pred[..., y:y2, x:x2],
            )
        )
        rows.append(rec)
    return rows


def actions_for_patch_rows(policy_id: str, rows: list[dict[str, Any]]) -> np.ndarray:
    image_rows = [{"name": "one", "split": "one", "A0_PSNR": 0.0}]
    table_rows = []
    for row in rows:
        rec = {"name": "one", "pixel_count": 1}
        rec.update({feat: row[feat] for feat in FEATURES})
        for alpha in ALPHAS:
            rec[f"sse_{alpha_key(alpha)}"] = 0.0
        table_rows.append(rec)
    table = PatchTable(table_rows, image_rows)
    return apply_policy(table, policy_id)


def eval_oof_actual(args: argparse.Namespace, fold_policies: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _loader, build_convir_net = c2.load_convir_builders(Path(args.convir_its_dir))
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, _ckpt_meta = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    for split in args.splits:
        names = c2.load_split_names(Path(args.split_json), split)
        depth_split = "train" if args.split_json else args.depth_split
        with torch.no_grad():
            for idx, image_name in enumerate(names):
                input_img, label_img, depth = c2.load_sample(Path(args.data_dir), Path(args.depth_cache_dir), image_name, depth_split)
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
                a0_pred = c2.infer_one(a0_model, rgb_padded, h, w)
                udp_pred = c2.infer_one(udp_model, udp_input, h, w)
                a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
                fold = fold_id(image_name)
                policy_id = fold_policies[fold]
                p_rows = patch_rows_for_image_tensors(image_name, split, input_img, depth, a0_pred, udp_pred, int(args.patch_size))
                actions = actions_for_patch_rows(policy_id, p_rows)
                out = a0_pred.clone()
                residual = udp_pred - a0_pred
                alpha_counts = {alpha: 0 for alpha in ALPHAS}
                for (patch_id, y, y2, x, x2), action in zip(iter_patches(h, w, int(args.patch_size)), actions, strict=False):
                    _ = patch_id
                    alpha = ALPHAS[int(action)]
                    alpha_counts[alpha] += 1
                    out[..., y:y2, x:x2] = torch.clamp(a0_pred[..., y:y2, x:x2] + alpha * residual[..., y:y2, x:x2], 0.0, 1.0)
                psnr, ssim = c2.metric_pair(out, label_img, (h_pad, w_pad))
                rec: dict[str, Any] = {
                    "name": image_name,
                    "split": split,
                    "fold": fold,
                    "policy_id": policy_id,
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "local_alpha_PSNR": psnr,
                    "local_alpha_SSIM": ssim,
                    "dPSNR": psnr - a0_psnr,
                    "dSSIM": ssim - a0_ssim,
                    "patch_count": int(sum(alpha_counts.values())),
                }
                for alpha in ALPHAS:
                    rec[f"patch_action_count_{alpha_key(alpha)}"] = alpha_counts[alpha]
                    rec[f"patch_action_fraction_{alpha_key(alpha)}"] = alpha_counts[alpha] / max(1, rec["patch_count"])
                rows.append(rec)
                if (idx + 1) % args.print_freq == 0:
                    print(f"actual_eval {split} {idx + 1}/{len(names)} rows={len(rows)}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    return rows, summarize_actual_rows(rows)


def summarize_actual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.array([fnum(row["dPSNR"]) for row in rows], dtype=np.float64)
    ssims = np.array([fnum(row["dSSIM"]) for row in rows], dtype=np.float64)
    a0 = np.array([fnum(row["A0_PSNR"]) for row in rows], dtype=np.float64)
    selected = np.array([fnum(row.get(f"patch_action_fraction_{alpha_key(0.0)}")) < 1.0 for row in rows], dtype=bool)
    order = np.argsort(a0)
    bucket = max(1, len(rows) // 4)
    selected_d = deltas[selected]
    rec = {
        "count": len(rows),
        "selected_count": int(selected.sum()),
        "coverage": float(np.mean(selected)),
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean(deltas[order[:bucket]])),
        "easy_top25_dPSNR": float(np.mean(deltas[order[-bucket:]])),
        "dSSIM": float(np.mean(ssims)),
        "positive_ratio": float(np.mean(deltas > 0.0)),
        "nonnegative_ratio": float(np.mean(deltas >= 0.0)),
        "severe_loss_count": int(np.sum(deltas <= -0.20)),
        "severe_loss_per_600": int(np.sum(deltas <= -0.20)) / max(1, len(rows)) * 600.0,
        "strong_loss_count": int(np.sum(deltas <= -0.05)),
        "strong_loss_per_600": int(np.sum(deltas <= -0.05)) / max(1, len(rows)) * 600.0,
        "selected_precision": float(np.mean(selected_d > 0.0)) if selected_d.size else 0.0,
        "selected_nonnegative_ratio": float(np.mean(selected_d >= 0.0)) if selected_d.size else 1.0,
        "selected_severe_count": int(np.sum(selected_d <= -0.20)) if selected_d.size else 0,
    }
    for alpha in ALPHAS:
        vals = [fnum(row.get(f"patch_action_fraction_{alpha_key(alpha)}")) for row in rows]
        rec[f"mean_patch_action_fraction_{alpha_key(alpha)}"] = float(np.mean(vals)) if vals else 0.0
    rec["screen_gate_pass"] = gate_pass(rec, C7B_SCREEN_GATE)
    rec["strong_gate_pass"] = gate_pass(rec, C7B_STRONG_GATE)
    rec["score"] = score(rec)
    return rec


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
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--low_pool_limit", type=int, default=80)
    parser.add_argument("--high_pool_limit", type=int, default=120)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_rows, patch_rows, meta = render_patch_table(args)
    patch_fields = ["name", "split", "patch_id", "y0", "y1", "x0", "x1", "pixel_count"] + FEATURES + [f"sse_{alpha_key(a)}" for a in ALPHAS]
    write_csv(args.out_dir / "v21_c7b_patch_feature_rows.csv", patch_rows, patch_fields)
    image_fields = ["name", "split", "A0_PSNR", "A0_SSIM", "h", "w"]
    write_csv(args.out_dir / "v21_c7b_image_rows.csv", image_rows, image_fields)

    full_table = PatchTable(patch_rows, image_rows)
    full_grid = policy_grid(full_table, args.top_k, args.low_pool_limit, args.high_pool_limit)
    policy_fields = [
        "policy_id", "complexity", "count", "selected_count", "coverage", "mean_dPSNR",
        "hard_bottom25_dPSNR", "easy_top25_dPSNR", "positive_ratio", "nonnegative_ratio",
        "severe_loss_count", "severe_loss_per_600", "strong_loss_count", "strong_loss_per_600",
        "selected_precision", "selected_nonnegative_ratio", "selected_severe_count",
        "screen_proxy_pass", "strong_proxy_pass", "score",
    ] + [f"patch_action_count_{alpha_key(a)}" for a in ALPHAS] + [f"patch_action_fraction_{alpha_key(a)}" for a in ALPHAS]
    write_csv(args.out_dir / "v21_c7b_local_alpha_policy_grid.csv", full_grid, policy_fields)

    fold_rows: list[dict[str, Any]] = []
    fold_policies: dict[int, str] = {}
    image_folds = np.array([fold_id(str(row["name"])) for row in image_rows], dtype=np.int64)
    for fold in range(5):
        train_table = full_table.subset_images(image_folds != fold)
        heldout_table = full_table.subset_images(image_folds == fold)
        chosen = choose_policy(train_table, args.top_k, args.low_pool_limit, args.high_pool_limit)
        fold_policies[fold] = str(chosen["policy_id"])
        heldout_actions = apply_policy(heldout_table, str(chosen["policy_id"]))
        rec: dict[str, Any] = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_screen_proxy_pass": chosen["screen_proxy_pass"],
            "train_strong_proxy_pass": chosen["strong_proxy_pass"],
            "train_score": chosen["score"],
            "train_count": len(train_table.image_rows),
            "heldout_count": len(heldout_table.image_rows),
        }
        rec.update(summarize_patch_actions(heldout_table, heldout_actions))
        rec["screen_proxy_pass"] = proxy_gate_pass(rec, C7B_SCREEN_GATE)
        rec["strong_proxy_pass"] = proxy_gate_pass(rec, C7B_STRONG_GATE)
        rec["score"] = score(rec)
        fold_rows.append(rec)
    fold_fields = ["fold", "train_policy_id", "train_screen_proxy_pass", "train_strong_proxy_pass", "train_score", "train_count", "heldout_count"] + policy_fields[2:]
    write_csv(args.out_dir / "v21_c7b_local_alpha_per_fold_proxy.csv", fold_rows, fold_fields)

    actual_rows, actual_summary = eval_oof_actual(args, fold_policies)
    actual_fields = [
        "name", "split", "fold", "policy_id", "A0_PSNR", "A0_SSIM", "local_alpha_PSNR",
        "local_alpha_SSIM", "dPSNR", "dSSIM", "patch_count",
    ] + [f"patch_action_count_{alpha_key(a)}" for a in ALPHAS] + [f"patch_action_fraction_{alpha_key(a)}" for a in ALPHAS]
    write_csv(args.out_dir / "v21_c7b_local_alpha_oof_per_image.csv", actual_rows, actual_fields)

    if actual_summary["strong_gate_pass"]:
        decision = "C7B_LOCAL_ALPHA_STRONG_CANDIDATE_PASS_START_C9_SHIFTED_STRONG"
    elif actual_summary["screen_gate_pass"]:
        decision = "C7B_LOCAL_ALPHA_SCREEN_PASS_STRONG_NOT_YET_REASSESS_POSITIVE"
    else:
        decision = "C7B_LOCAL_ALPHA_FAIL_START_C8_MULTIEXPERT_OR_RICHER_LOCAL_FEATURES"

    summary = {
        "route": "Haze4K-v2.1 SEG-Mix",
        "phase": "C7b Local-Alpha Deployable Prototype",
        "locked_test_touched": False,
        "meta": meta,
        "patch_size": args.patch_size,
        "alphas": ALPHAS,
        "screen_gate": C7B_SCREEN_GATE,
        "strong_gate": C7B_STRONG_GATE,
        "best_full_proxy_policy": full_grid[0] if full_grid else None,
        "fold_proxy_rows": fold_rows,
        "fold_policies": fold_policies,
        "actual_oof_summary": actual_summary,
        "decision": decision,
    }
    (args.out_dir / "v21_c7b_local_alpha_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Haze4K v2.1 C7b Local-Alpha Deployable Prototype",
        "",
        f"Decision: `{decision}`",
        "",
        "C7b trains transparent patch-level alpha policies using train-derived image-fold OOF and re-renders held-out images for true PSNR/SSIM. Locked test data was not touched.",
        "",
        "## Actual OOF Summary",
        "",
    ]
    for key, value in actual_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Interpretation", "", "C9 shifted-strong validation is authorized only if the C7b strong gate passes. Otherwise locked test and distillation remain blocked."])
    (args.out_dir / "v21_c7b_local_alpha_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V21_C7B_LOCAL_ALPHA_OK decision={decision} rows={len(actual_rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
