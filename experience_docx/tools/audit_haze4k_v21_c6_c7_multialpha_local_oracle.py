#!/usr/bin/env python3
"""C6/C7 exact multi-alpha router and patch-alpha oracle audit.

Renders A0 and official FullUDP once on train-derived internal validation
splits, evaluates exact A0-preserving alpha blends, searches a leakage-safe
multi-alpha OOF router, and computes a patch-level alpha oracle. It writes
text/CSV/JSON evidence only; raw images/tensors are not written and locked
Haze4K test data is not touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import POLICY_FEATURES, QUANTILES, fnum, fold_id, write_csv


ALPHAS = [0.125, 0.25, 0.375, 0.50, 0.75]
ACTION_ALPHAS = [0.0] + ALPHAS
LOW_ALPHAS = [0.125, 0.25, 0.375]
HIGH_ALPHAS = [0.375, 0.50, 0.75]

C6_SCREEN_GATE = {
    "mean_dPSNR": 0.20,
    "hard_bottom25_dPSNR": 0.28,
    "easy_top25_dPSNR": 0.0,
    "dSSIM": 0.0,
    "positive_ratio": 0.69,
    "severe_loss_per_600": 48.0,
}

C6_FORMAL_CANDIDATE_GATE = {
    "mean_dPSNR": 0.20,
    "hard_bottom25_dPSNR": 0.30,
    "easy_top25_dPSNR": 0.0,
    "dSSIM": 0.0,
    "positive_ratio": 0.70,
    "severe_loss_per_600": 48.0,
}

C7_PATCH_SIGNAL_GATE = {
    "hard_bottom25_dPSNR": 0.35,
    "positive_ratio": 0.72,
    "dSSIM": 0.0,
}


def alpha_key(alpha: float) -> str:
    text = ("%.6f" % alpha).rstrip("0").rstrip(".")
    return f"a{text.replace('.', 'p')}"


def alpha_label(alpha: float) -> str:
    return alpha_key(alpha)


def gate_pass(row: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= gate["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= gate["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= gate["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= gate["dSSIM"]
        and fnum(row.get("positive_ratio")) >= gate["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= gate["severe_loss_per_600"]
    )


def c6_score(row: dict[str, Any]) -> float:
    hard = fnum(row.get("hard_bottom25_dPSNR"))
    pos = fnum(row.get("positive_ratio"))
    easy = fnum(row.get("easy_top25_dPSNR"))
    dssim = fnum(row.get("dSSIM"))
    severe = fnum(row.get("severe_loss_per_600"))
    penalty = 0.0
    penalty += 2.0 * max(0.0, C6_FORMAL_CANDIDATE_GATE["hard_bottom25_dPSNR"] - hard)
    penalty += 1.5 * max(0.0, C6_FORMAL_CANDIDATE_GATE["positive_ratio"] - pos)
    penalty += 3.0 * max(0.0, C6_FORMAL_CANDIDATE_GATE["easy_top25_dPSNR"] - easy)
    penalty += 50.0 * max(0.0, C6_FORMAL_CANDIDATE_GATE["dSSIM"] - dssim)
    penalty += 0.01 * max(0.0, severe - C6_FORMAL_CANDIDATE_GATE["severe_loss_per_600"])
    return (
        fnum(row.get("mean_dPSNR"))
        + 0.80 * hard
        + 0.45 * pos
        + 0.10 * easy
        + 0.05 * fnum(row.get("selected_precision"))
        + 0.03 * fnum(row.get("coverage"))
        - 0.002 * severe
        - penalty
    )


class ActionTable:
    def __init__(self, rows: list[dict[str, Any]], alphas: list[float]):
        self.rows = rows
        self.alphas = alphas
        self.names = np.array([str(row["name"]) for row in rows])
        self.a0_psnr = np.array([fnum(row["A0_PSNR"]) for row in rows], dtype=np.float64)
        self.dpsnr = np.zeros((len(rows), len(alphas)), dtype=np.float64)
        self.dssim = np.zeros((len(rows), len(alphas)), dtype=np.float64)
        for col, alpha in enumerate(alphas):
            if alpha == 0.0:
                continue
            key = alpha_key(alpha)
            self.dpsnr[:, col] = np.array([fnum(row[f"dPSNR_{key}"]) for row in rows], dtype=np.float64)
            self.dssim[:, col] = np.array([fnum(row[f"dSSIM_{key}"]) for row in rows], dtype=np.float64)
        self.features = {feature: np.array([fnum(row[feature]) for row in rows], dtype=np.float64) for feature in POLICY_FEATURES}
        self.alpha_to_index = {alpha: idx for idx, alpha in enumerate(alphas)}

    def subset(self, mask: np.ndarray) -> "ActionTable":
        return ActionTable([row for row, keep in zip(self.rows, np.asarray(mask, dtype=bool), strict=False) if keep], self.alphas)


def summarize_actions(table: ActionTable, action_idx: np.ndarray) -> dict[str, Any]:
    count = len(table.rows)
    action_idx = np.asarray(action_idx, dtype=np.int64)
    if count == 0:
        return {
            "count": 0,
            "selected_count": 0,
            "coverage": 0.0,
            "mean_dPSNR": 0.0,
            "hard_bottom25_dPSNR": 0.0,
            "easy_top25_dPSNR": 0.0,
            "dSSIM": 0.0,
            "positive_ratio": 0.0,
            "nonnegative_ratio": 0.0,
            "severe_loss_count": 0,
            "severe_loss_per_600": 0.0,
            "strong_loss_count": 0,
            "strong_loss_per_600": 0.0,
            "selected_precision": 0.0,
            "selected_nonnegative_ratio": 1.0,
            "selected_severe_count": 0,
        }
    idx = np.arange(count)
    deltas = table.dpsnr[idx, action_idx]
    ssims = table.dssim[idx, action_idx]
    selected = action_idx > 0
    selected_count = int(selected.sum())
    order = np.argsort(table.a0_psnr)
    bucket = max(1, count // 4)
    selected_deltas = deltas[selected]
    severe = int(np.sum(deltas <= -0.20))
    strong = int(np.sum(deltas <= -0.05))
    selected_positive = int(np.sum(selected_deltas > 0.0)) if selected_count else 0
    selected_nonnegative = int(np.sum(selected_deltas >= 0.0)) if selected_count else 0
    selected_severe = int(np.sum(selected_deltas <= -0.20)) if selected_count else 0
    return {
        "count": count,
        "selected_count": selected_count,
        "coverage": selected_count / count,
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean(deltas[order[:bucket]])),
        "easy_top25_dPSNR": float(np.mean(deltas[order[-bucket:]])),
        "dSSIM": float(np.mean(ssims)),
        "positive_ratio": float(np.mean(deltas > 0.0)),
        "nonnegative_ratio": float(np.mean(deltas >= 0.0)),
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / count * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / count * 600.0,
        "selected_precision": selected_positive / selected_count if selected_count else 0.0,
        "selected_nonnegative_ratio": selected_nonnegative / selected_count if selected_count else 1.0,
        "selected_severe_count": selected_severe,
    }


def condition_candidates(table: ActionTable, min_coverage: float = 0.04) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"condition_id": "all_images", "mask": np.ones(len(table.rows), dtype=bool), "coverage": 1.0}]
    for feature in POLICY_FEATURES:
        vals = table.features[feature]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        thresholds = sorted({float(np.quantile(finite, q)) for q in QUANTILES})
        for threshold in thresholds:
            for direction, mask in (("le", vals <= threshold), ("ge", vals >= threshold)):
                coverage = float(mask.mean()) if len(mask) else 0.0
                if coverage < min_coverage:
                    continue
                out.append({"condition_id": f"{feature}_{direction}_{threshold:.8g}", "mask": mask, "coverage": coverage})
    return out


def parse_condition(table: ActionTable, condition_id: str) -> np.ndarray:
    if condition_id == "all_images":
        return np.ones(len(table.rows), dtype=bool)
    if "_le_" in condition_id:
        feature, threshold = condition_id.rsplit("_le_", 1)
        return table.features[feature] <= float(threshold)
    if "_ge_" in condition_id:
        feature, threshold = condition_id.rsplit("_ge_", 1)
        return table.features[feature] >= float(threshold)
    return np.zeros(len(table.rows), dtype=bool)


def apply_policy(table: ActionTable, policy_id: str) -> np.ndarray:
    actions = np.zeros(len(table.rows), dtype=np.int64)
    if policy_id == "a0_anchor":
        return actions
    parts = policy_id.split("|")
    if parts[0] == "single":
        alpha = float(parts[1].removeprefix("alpha="))
        condition = parts[2].removeprefix("cond=")
        actions[parse_condition(table, condition)] = table.alpha_to_index[alpha]
    elif parts[0] == "tier":
        low_alpha = float(parts[1].removeprefix("low_alpha="))
        low_cond = parts[2].removeprefix("low_cond=")
        high_alpha = float(parts[3].removeprefix("high_alpha="))
        high_cond = parts[4].removeprefix("high_cond=")
        actions[parse_condition(table, low_cond)] = table.alpha_to_index[low_alpha]
        actions[parse_condition(table, high_cond)] = table.alpha_to_index[high_alpha]
    return actions


def add_candidate(out: list[dict[str, Any]], table: ActionTable, policy_id: str, actions: np.ndarray, complexity: int) -> None:
    rec: dict[str, Any] = {"policy_id": policy_id, "complexity": complexity}
    rec.update(summarize_actions(table, actions))
    rec["screen_gate_pass"] = gate_pass(rec, C6_SCREEN_GATE)
    rec["formal_candidate_gate_pass"] = gate_pass(rec, C6_FORMAL_CANDIDATE_GATE)
    rec["score"] = c6_score(rec)
    for alpha in table.alphas:
        label = alpha_label(alpha)
        rec[f"action_count_{label}"] = int(np.sum(actions == table.alpha_to_index[alpha]))
        rec[f"action_fraction_{label}"] = float(np.mean(actions == table.alpha_to_index[alpha])) if len(actions) else 0.0
    out.append(rec)


def policy_grid(table: ActionTable, top_k: int, low_pool_limit: int, high_pool_limit: int) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    n = len(table.rows)
    add_candidate(policies, table, "a0_anchor", np.zeros(n, dtype=np.int64), 0)
    conditions = condition_candidates(table)
    single_by_policy: list[tuple[dict[str, Any], np.ndarray, float, str]] = []
    for alpha in ALPHAS:
        alpha_idx = table.alpha_to_index[alpha]
        for cond in conditions:
            actions = np.zeros(n, dtype=np.int64)
            actions[np.asarray(cond["mask"], dtype=bool)] = alpha_idx
            policy_id = f"single|alpha={alpha}|cond={cond['condition_id']}"
            before_len = len(policies)
            add_candidate(policies, table, policy_id, actions, 1)
            single_by_policy.append((policies[before_len], actions, alpha, str(cond["condition_id"])))

    low_pool: list[tuple[dict[str, Any], np.ndarray, float, str]] = []
    high_pool: list[tuple[dict[str, Any], np.ndarray, float, str]] = []
    for rec, actions, alpha, cond_id in single_by_policy:
        coverage = fnum(rec["coverage"])
        if alpha in LOW_ALPHAS and 0.35 <= coverage <= 0.95 and fnum(rec["easy_top25_dPSNR"]) >= -0.01 and fnum(rec["dSSIM"]) >= -0.0002:
            low_pool.append((rec, actions, alpha, cond_id))
        if alpha in HIGH_ALPHAS and 0.04 <= coverage <= 0.55 and fnum(rec["selected_precision"]) >= 0.55 and fnum(rec["severe_loss_per_600"]) <= 120.0:
            high_pool.append((rec, actions, alpha, cond_id))
    low_pool.sort(key=lambda item: (bool(item[0]["formal_candidate_gate_pass"]), bool(item[0]["screen_gate_pass"]), fnum(item[0]["score"])), reverse=True)
    high_pool.sort(key=lambda item: (fnum(item[0]["hard_bottom25_dPSNR"]), fnum(item[0]["score"])), reverse=True)
    low_pool = low_pool[:low_pool_limit]
    high_pool = high_pool[:high_pool_limit]

    for _low_rec, low_actions, low_alpha, low_cond in low_pool:
        for _high_rec, high_actions, high_alpha, high_cond in high_pool:
            if high_alpha <= low_alpha and high_cond == low_cond:
                continue
            actions = low_actions.copy()
            actions[high_actions > 0] = table.alpha_to_index[high_alpha]
            selected_fraction = float(np.mean(actions > 0)) if len(actions) else 0.0
            if selected_fraction < 0.08 or selected_fraction > 0.96:
                continue
            policy_id = f"tier|low_alpha={low_alpha}|low_cond={low_cond}|high_alpha={high_alpha}|high_cond={high_cond}"
            add_candidate(policies, table, policy_id, actions, 2)

    policies.sort(key=lambda row: (bool(row["formal_candidate_gate_pass"]), bool(row["screen_gate_pass"]), fnum(row["score"])), reverse=True)
    keep: list[dict[str, Any]] = []
    for row in policies:
        if len(keep) < top_k or row["formal_candidate_gate_pass"] or row["screen_gate_pass"]:
            keep.append(row)
    return keep


def choose_policy(train_table: ActionTable, top_k: int, low_pool_limit: int, high_pool_limit: int) -> dict[str, Any]:
    grid = policy_grid(train_table, top_k=top_k, low_pool_limit=low_pool_limit, high_pool_limit=high_pool_limit)
    formal = [row for row in grid if row["formal_candidate_gate_pass"]]
    if formal:
        return formal[0]
    screen = [row for row in grid if row["screen_gate_pass"]]
    if screen:
        return screen[0]
    return grid[0]


def image_oracle_actions(table: ActionTable) -> np.ndarray:
    best = np.argmax(table.dpsnr, axis=1)
    gains = table.dpsnr[np.arange(len(table.rows)), best]
    best[gains <= 0.0] = 0
    return best.astype(np.int64)


def action_distribution_rows(table: ActionTable, actions: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in table.alphas:
        idx = table.alpha_to_index[alpha]
        mask = actions == idx
        selected_dpsnr = table.dpsnr[:, idx][mask]
        rows.append(
            {
                "scope": prefix,
                "alpha": alpha,
                "action_count": int(mask.sum()),
                "action_fraction": float(mask.mean()) if len(mask) else 0.0,
                "mean_action_dPSNR": float(selected_dpsnr.mean()) if selected_dpsnr.size else 0.0,
                "positive_action_ratio": float(np.mean(selected_dpsnr > 0.0)) if selected_dpsnr.size else 0.0,
                "severe_action_count": int(np.sum(selected_dpsnr <= -0.20)) if selected_dpsnr.size else 0,
            }
        )
    return rows


def patch_oracle(
    a0_pred: torch.Tensor,
    udp_pred: torch.Tensor,
    label_img: torch.Tensor,
    alphas: list[float],
    patch_size: int,
    min_improve_mse: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, _, h, w = a0_pred.shape
    residual = udp_pred - a0_pred
    out = a0_pred.clone()
    counts = {alpha: 0 for alpha in alphas}
    total_improve = 0.0
    patch_count = 0
    for y in range(0, h, patch_size):
        y2 = min(h, y + patch_size)
        for x in range(0, w, patch_size):
            x2 = min(w, x + patch_size)
            a0_patch = a0_pred[..., y:y2, x:x2]
            res_patch = residual[..., y:y2, x:x2]
            gt_patch = label_img[..., y:y2, x:x2]
            base_mse = float(F.mse_loss(a0_patch, gt_patch).item())
            best_alpha = 0.0
            best_mse = base_mse
            for alpha in alphas:
                if alpha == 0.0:
                    continue
                blend = torch.clamp(a0_patch + alpha * res_patch, 0.0, 1.0)
                mse = float(F.mse_loss(blend, gt_patch).item())
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            if base_mse - best_mse <= min_improve_mse:
                best_alpha = 0.0
                best_mse = base_mse
            out[..., y:y2, x:x2] = torch.clamp(a0_patch + best_alpha * res_patch, 0.0, 1.0)
            counts[best_alpha] = counts.get(best_alpha, 0) + 1
            total_improve += base_mse - best_mse
            patch_count += 1
    stats: dict[str, Any] = {
        "patch_count": patch_count,
        "mean_patch_mse_improve": total_improve / max(1, patch_count),
        "selected_patch_count": sum(count for alpha, count in counts.items() if alpha > 0.0),
        "selected_patch_fraction": sum(count for alpha, count in counts.items() if alpha > 0.0) / max(1, patch_count),
    }
    for alpha in alphas:
        stats[f"patch_count_{alpha_label(alpha)}"] = counts.get(alpha, 0)
        stats[f"patch_fraction_{alpha_label(alpha)}"] = counts.get(alpha, 0) / max(1, patch_count)
    return out, stats


def render_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    convir_its_dir = Path(args.convir_its_dir)
    _test_dataloader, build_convir_net = c2.load_convir_builders(convir_its_dir)
    build_udpnet = c2.load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a0_model = c2.load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udpnet_model, ckpt_meta = c2.load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    start_time = time.time()
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
                udp_pred = c2.infer_one(udpnet_model, udp_input, h, w)
                a0_psnr, a0_ssim = c2.metric_pair(a0_pred, label_img, (h_pad, w_pad))
                rec: dict[str, Any] = {"name": image_name, "split": split, "A0_PSNR": a0_psnr, "A0_SSIM": a0_ssim}
                rec.update(c2.feature_dict(input_img, depth, a0_pred, udp_pred))
                for alpha in ALPHAS:
                    key = alpha_key(alpha)
                    blend = torch.clamp(a0_pred + alpha * (udp_pred - a0_pred), 0.0, 1.0)
                    psnr_val, ssim_val = c2.metric_pair(blend, label_img, (h_pad, w_pad))
                    rec[f"blend_PSNR_{key}"] = psnr_val
                    rec[f"blend_SSIM_{key}"] = ssim_val
                    rec[f"dPSNR_{key}"] = psnr_val - a0_psnr
                    rec[f"dSSIM_{key}"] = ssim_val - a0_ssim
                rows.append(rec)

                for patch_size in args.patch_sizes:
                    variants = [
                        ("oracle_max", ACTION_ALPHAS, 0.0),
                        ("oracle_riskcap0p5", [alpha for alpha in ACTION_ALPHAS if alpha <= 0.50], args.patch_min_improve_mse),
                    ]
                    for variant, patch_alphas, min_improve in variants:
                        patch_pred, patch_stats = patch_oracle(a0_pred, udp_pred, label_img, patch_alphas, int(patch_size), float(min_improve))
                        patch_psnr, patch_ssim = c2.metric_pair(patch_pred, label_img, (h_pad, w_pad))
                        prow = {
                            "name": image_name,
                            "split": split,
                            "A0_PSNR": a0_psnr,
                            "A0_SSIM": a0_ssim,
                            "patch_variant": variant,
                            "patch_size": int(patch_size),
                            "patch_PSNR": patch_psnr,
                            "patch_SSIM": patch_ssim,
                            "dPSNR": patch_psnr - a0_psnr,
                            "dSSIM": patch_ssim - a0_ssim,
                        }
                        prow.update(patch_stats)
                        patch_rows.append(prow)
                if (idx + 1) % args.print_freq == 0:
                    mean_delta = statistics.mean(float(r[f"dPSNR_{alpha_key(0.25)}"]) for r in rows)
                    print(f"{split} {idx + 1}/{len(names)} rows={len(rows)} alpha025_mean_delta={mean_delta:.4f}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break
    meta = {
        "elapsed_sec": time.time() - start_time,
        "device": str(device),
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": c2.sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint": str(args.official_checkpoint),
        "official_sha256": c2.sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "udp_repo": str(args.udp_repo),
        "depth_normalization": "per_image_minmax_to_udpnet_depth2l_contract",
        "alphas": ALPHAS,
        "patch_sizes": args.patch_sizes,
        "splits": args.splits,
        "split_json": args.split_json,
        "locked_test_touched": False,
    }
    return rows, patch_rows, meta


def summarize_patch_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = np.array([fnum(row["dPSNR"]) for row in rows], dtype=np.float64)
    ssims = np.array([fnum(row["dSSIM"]) for row in rows], dtype=np.float64)
    a0 = np.array([fnum(row["A0_PSNR"]) for row in rows], dtype=np.float64)
    selected = np.array([fnum(row.get("selected_patch_fraction")) > 0.0 for row in rows], dtype=bool)
    order = np.argsort(a0)
    bucket = max(1, len(rows) // 4)
    selected_deltas = deltas[selected]
    return {
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
        "severe_loss_per_600": int(np.sum(deltas <= -0.20)) / len(rows) * 600.0,
        "strong_loss_count": int(np.sum(deltas <= -0.05)),
        "strong_loss_per_600": int(np.sum(deltas <= -0.05)) / len(rows) * 600.0,
        "selected_precision": float(np.mean(selected_deltas > 0.0)) if selected_deltas.size else 0.0,
        "selected_nonnegative_ratio": float(np.mean(selected_deltas >= 0.0)) if selected_deltas.size else 1.0,
        "selected_severe_count": int(np.sum(selected_deltas <= -0.20)) if selected_deltas.size else 0,
        "mean_selected_patch_fraction": float(np.mean([fnum(row.get("selected_patch_fraction")) for row in rows])),
    }


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
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--patch_sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--patch_min_improve_mse", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--low_pool_limit", type=int, default=80)
    parser.add_argument("--high_pool_limit", type=int, default=120)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, patch_rows, meta = render_rows(args)
    fields = ["name", "split", "A0_PSNR", "A0_SSIM"] + POLICY_FEATURES
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        fields.extend([f"blend_PSNR_{key}", f"blend_SSIM_{key}", f"dPSNR_{key}", f"dSSIM_{key}"])
    write_csv(out_dir / "v21_c6_multialpha_feature_rows.csv", rows, fields)

    patch_fields = [
        "name",
        "split",
        "A0_PSNR",
        "A0_SSIM",
        "patch_variant",
        "patch_size",
        "patch_PSNR",
        "patch_SSIM",
        "dPSNR",
        "dSSIM",
        "patch_count",
        "selected_patch_count",
        "selected_patch_fraction",
        "mean_patch_mse_improve",
    ]
    for alpha in ACTION_ALPHAS:
        patch_fields.extend([f"patch_count_{alpha_label(alpha)}", f"patch_fraction_{alpha_label(alpha)}"])
    write_csv(out_dir / "v21_c7_patch_alpha_oracle.csv", patch_rows, patch_fields)

    table = ActionTable(rows, ACTION_ALPHAS)
    policies = policy_grid(table, top_k=args.top_k, low_pool_limit=args.low_pool_limit, high_pool_limit=args.high_pool_limit)
    policy_fields = [
        "policy_id",
        "complexity",
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
        "screen_gate_pass",
        "formal_candidate_gate_pass",
        "score",
    ]
    for alpha in ACTION_ALPHAS:
        policy_fields.extend([f"action_count_{alpha_label(alpha)}", f"action_fraction_{alpha_label(alpha)}"])
    write_csv(out_dir / "v21_c6_multialpha_policy_rows.csv", policies, policy_fields)

    fold_rows: list[dict[str, Any]] = []
    oof_actions = np.zeros(len(rows), dtype=np.int64)
    fold_ids = np.array([fold_id(str(row["name"])) for row in rows], dtype=np.int64)
    for fold in range(5):
        heldout_mask = fold_ids == fold
        train_table = table.subset(~heldout_mask)
        heldout_table = table.subset(heldout_mask)
        chosen = choose_policy(train_table, top_k=args.top_k, low_pool_limit=args.low_pool_limit, high_pool_limit=args.high_pool_limit)
        heldout_actions = apply_policy(heldout_table, str(chosen["policy_id"]))
        oof_actions[heldout_mask] = heldout_actions
        rec: dict[str, Any] = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_screen_gate_pass": chosen["screen_gate_pass"],
            "train_formal_candidate_gate_pass": chosen["formal_candidate_gate_pass"],
            "train_score": chosen["score"],
            "train_count": len(train_table.rows),
            "heldout_count": len(heldout_table.rows),
        }
        rec.update(summarize_actions(heldout_table, heldout_actions))
        rec["screen_gate_pass"] = gate_pass(rec, C6_SCREEN_GATE)
        rec["formal_candidate_gate_pass"] = gate_pass(rec, C6_FORMAL_CANDIDATE_GATE)
        rec["score"] = c6_score(rec)
        fold_rows.append(rec)
    fold_fields = [
        "fold",
        "train_policy_id",
        "train_screen_gate_pass",
        "train_formal_candidate_gate_pass",
        "train_score",
        "train_count",
        "heldout_count",
    ] + policy_fields[2:21]
    write_csv(out_dir / "v21_c6_multialpha_per_fold.csv", fold_rows, fold_fields)

    oof_summary = summarize_actions(table, oof_actions)
    oof_summary["screen_gate_pass"] = gate_pass(oof_summary, C6_SCREEN_GATE)
    oof_summary["formal_candidate_gate_pass"] = gate_pass(oof_summary, C6_FORMAL_CANDIDATE_GATE)
    oof_summary["score"] = c6_score(oof_summary)
    image_oracle = image_oracle_actions(table)
    image_oracle_summary = summarize_actions(table, image_oracle)
    image_oracle_summary["screen_gate_pass"] = gate_pass(image_oracle_summary, C6_SCREEN_GATE)
    image_oracle_summary["formal_candidate_gate_pass"] = gate_pass(image_oracle_summary, C6_FORMAL_CANDIDATE_GATE)
    image_oracle_summary["score"] = c6_score(image_oracle_summary)

    action_rows = action_distribution_rows(table, oof_actions, "c6_oof") + action_distribution_rows(table, image_oracle, "image_multialpha_oracle")
    write_csv(out_dir / "v21_c6_multialpha_action_distribution.csv", action_rows, list(action_rows[0].keys()))

    patch_summary_rows: list[dict[str, Any]] = []
    for patch_size in sorted({int(row["patch_size"]) for row in patch_rows}):
        for variant in sorted({str(row["patch_variant"]) for row in patch_rows}):
            subset = [row for row in patch_rows if int(row["patch_size"]) == patch_size and str(row["patch_variant"]) == variant]
            rec = {"patch_size": patch_size, "patch_variant": variant}
            rec.update(summarize_patch_group(subset))
            rec["patch_signal_gate_pass"] = (
                fnum(rec.get("hard_bottom25_dPSNR")) >= C7_PATCH_SIGNAL_GATE["hard_bottom25_dPSNR"]
                and fnum(rec.get("positive_ratio")) >= C7_PATCH_SIGNAL_GATE["positive_ratio"]
                and fnum(rec.get("dSSIM")) >= C7_PATCH_SIGNAL_GATE["dSSIM"]
            )
            patch_summary_rows.append(rec)
    patch_summary_fields = ["patch_size", "patch_variant"] + [
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
        "mean_selected_patch_fraction",
        "patch_signal_gate_pass",
    ]
    write_csv(out_dir / "v21_c7_patch_alpha_mask_stats.csv", patch_summary_rows, patch_summary_fields)

    if oof_summary["formal_candidate_gate_pass"]:
        c6_decision = "C6_MULTIALPHA_OOF_STRONG_CANDIDATE_PASS_START_C9_SHIFTED_STRONG"
    elif oof_summary["screen_gate_pass"]:
        c6_decision = "C6_MULTIALPHA_OOF_SCREEN_PASS_STRONG_TARGET_NOT_YET_START_C7_C8"
    else:
        c6_decision = "C6_MULTIALPHA_OOF_FAIL_USE_C7_PATCH_OR_C8_MULTIEXPERT"
    c7_signal = any(bool(row.get("patch_signal_gate_pass")) for row in patch_summary_rows)
    c7_decision = "C7_PATCH_ALPHA_ORACLE_STRONG_SIGNAL_START_LOCAL_ALPHA" if c7_signal else "C7_PATCH_ALPHA_ORACLE_WEAK_SIGNAL_PRIORITIZE_C8_OR_FEATURES"

    summary = {
        "route": "Haze4K-v2.1 SEG-Mix",
        "phase": "C6 Multi-Alpha OOF Router + C7 Patch-Alpha Oracle",
        "locked_test_touched": False,
        "meta": meta,
        "alphas": ALPHAS,
        "action_alphas": ACTION_ALPHAS,
        "c6_screen_gate": C6_SCREEN_GATE,
        "c6_formal_candidate_gate": C6_FORMAL_CANDIDATE_GATE,
        "c7_patch_signal_gate": C7_PATCH_SIGNAL_GATE,
        "best_in_sample_policy": policies[0] if policies else None,
        "best_formal_in_sample_policy": next((row for row in policies if row["formal_candidate_gate_pass"]), None),
        "best_screen_in_sample_policy": next((row for row in policies if row["screen_gate_pass"]), None),
        "fold_rows": fold_rows,
        "oof_summary": oof_summary,
        "image_multialpha_oracle_summary": image_oracle_summary,
        "patch_summary_rows": patch_summary_rows,
        "c6_decision": c6_decision,
        "c7_decision": c7_decision,
        "decision": f"{c6_decision}__{c7_decision}",
    }
    (out_dir / "v21_c6_c7_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "v21_c6_multialpha_strong_gate_decision.json").write_text(
        json.dumps(
            {
                "locked_test_touched": False,
                "decision": c6_decision,
                "oof_summary": oof_summary,
                "image_multialpha_oracle_summary": image_oracle_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    c6_lines = [
        "# Haze4K v2.1 C6 Risk-Bounded Multi-Alpha Router",
        "",
        f"Decision: `{c6_decision}`",
        "",
        "C6 evaluates exact A0-preserving FullUDP residual alphas and searches train-only OOF image-level multi-alpha policies. Locked test data was not touched.",
        "",
        "## OOF Summary",
        "",
    ]
    for key, value in oof_summary.items():
        c6_lines.append(f"- `{key}`: `{value}`")
    c6_lines.extend(["", "## Image Multi-Alpha Oracle", ""])
    for key, value in image_oracle_summary.items():
        c6_lines.append(f"- `{key}`: `{value}`")
    c6_lines.extend(["", "## Interpretation", "", "Only a C6 strong-candidate OOF pass can start C9 shifted-strong validation. It still does not authorize locked test."])
    (out_dir / "v21_c6_multialpha_decision.md").write_text("\n".join(c6_lines) + "\n", encoding="utf-8")

    c7_lines = [
        "# Haze4K v2.1 C7 Patch-Level Alpha Oracle",
        "",
        f"Decision: `{c7_decision}`",
        "",
        "C7 computes non-deployable patch-level alpha oracles to test whether local alpha can break the image-level router ceiling. It is capacity evidence only.",
        "",
        "| Variant | Patch | Mean | Hard | Easy | dSSIM | Positive | Severe/600 | Patch Signal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in patch_summary_rows:
        c7_lines.append(
            f"| `{row['patch_variant']}` | `{row['patch_size']}` | `{fnum(row['mean_dPSNR']):.6f}` | "
            f"`{fnum(row['hard_bottom25_dPSNR']):.6f}` | `{fnum(row['easy_top25_dPSNR']):.6f}` | "
            f"`{fnum(row['dSSIM']):.8f}` | `{fnum(row['positive_ratio']):.6f}` | "
            f"`{fnum(row['severe_loss_per_600']):.1f}` | `{row['patch_signal_gate_pass']}` |"
        )
    c7_lines.extend(["", "## Interpretation", "", "Patch oracle pass supports a local-alpha prototype; weak patch signal shifts priority toward candidate-zoo/multi-expert expansion."])
    (out_dir / "v21_c7_patch_alpha_decision.md").write_text("\n".join(c7_lines) + "\n", encoding="utf-8")

    print(f"V21_C6_C7_OK decision={summary['decision']} rows={len(rows)} patch_rows={len(patch_rows)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
