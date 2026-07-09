import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
for path in (str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2F_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2f_need_target_head_redesign.py"
spec_v2f = importlib.util.spec_from_file_location("chdrm_v2f_tool", V2F_TOOL)
v2f = importlib.util.module_from_spec(spec_v2f)
spec_v2f.loader.exec_module(v2f)

v2e = v2f.v2e
d7c = v2e.d7c
v2d = v2e.v2d
v2b = v2d.v2b

ROUTE_ID = "haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709"
SAFETY_FALSE_GLOBAL = 0.01
SAFETY_FALSE_P90 = 0.05
SAFETY_FALSE_P95 = 0.10
LDHN_RECALL_GATE = 0.10


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def finite_values(values):
    return [float(v) for v in values if not math.isnan(float(v))]


def finite_mean(values):
    vals = finite_values(values)
    return statistics.mean(vals) if vals else math.nan


def finite_quantile(values, q):
    vals = sorted(finite_values(values))
    if not vals:
        return math.nan
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return float(vals[idx])


def sample_map(x, size):
    if x.shape[-2:] != (size, size):
        x = F.adaptive_avg_pool2d(x, (size, size))
    return x.detach().flatten().float().cpu().numpy().astype(np.float32, copy=False)


def torch_bin_ids(values, edges):
    finite_edges = [float(x) for x in edges[1:-1]]
    edge_t = torch.as_tensor(finite_edges, device=values.device, dtype=values.dtype)
    return torch.bucketize(values.contiguous(), edge_t)


def torch_conditional_cdf(raw_need, density, fit, mode):
    out = torch.zeros_like(raw_need)
    bin_id = torch_bin_ids(density, fit["edges"])
    q_denom = float(len(fit["q_grid"]) - 1)
    for table, excess_table in zip(fit["tables"], fit["excess_tables"]):
        b = int(table["bin"])
        mask = bin_id == b
        if not bool(mask.any()):
            continue
        if mode == "excess":
            vals = raw_need[mask] - float(table["raw_need_mean"])
            grid_np = excess_table["excess_q"]
        elif mode == "conditional":
            vals = raw_need[mask]
            grid_np = table["raw_need_q"]
        else:
            raise ValueError(mode)
        grid = torch.as_tensor(grid_np, device=raw_need.device, dtype=raw_need.dtype)
        idx = torch.bucketize(vals.contiguous(), grid).clamp_(0, len(grid_np) - 1)
        out[mask] = idx.to(raw_need.dtype) / q_denom
    return out.clamp_(0.0, 1.0)


def density_weights(density_pred, edges, temperature):
    finite = [float(x) for x in edges[1:-1]]
    if not finite:
        return torch.ones(
            (density_pred.shape[0], 1, 1, density_pred.shape[-2], density_pred.shape[-1]),
            device=density_pred.device,
            dtype=density_pred.dtype,
        )
    left = finite[0] - max(finite[1] - finite[0] if len(finite) > 1 else 0.10, 1e-3)
    right = finite[-1] + max(finite[-1] - finite[-2] if len(finite) > 1 else 0.10, 1e-3)
    centers = [0.5 * (left + finite[0])]
    centers.extend(0.5 * (finite[i - 1] + finite[i]) for i in range(1, len(finite)))
    centers.append(0.5 * (finite[-1] + right))
    center_t = torch.as_tensor(centers, device=density_pred.device, dtype=density_pred.dtype).view(1, -1, 1, 1, 1)
    dist = torch.abs(density_pred.unsqueeze(1) - center_t)
    return torch.softmax(-dist / max(float(temperature), 1e-6), dim=1)


class DensityStratifiedNeedHead(nn.Module):
    def __init__(self, in_channels=234, density_bins=5, out_channels=4, hidden=96):
        super().__init__()
        self.density_bins = density_bins
        self.out_channels = out_channels
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.Conv2d(64, density_bins * out_channels, 1)

    def forward(self, context, edges, temperature):
        feat = self.shared(context)
        logits = self.heads(feat)
        b, _, h, w = logits.shape
        logits = logits.view(b, self.density_bins, self.out_channels, h, w)
        weights = density_weights(context[:, -1:].detach(), edges, temperature)
        mixed_logits = (weights * logits).sum(dim=1)
        pred_by_bin = torch.sigmoid(logits).mean(dim=2, keepdim=True)
        pred = (weights * pred_by_bin).sum(dim=1)
        return pred, mixed_logits, logits, weights.squeeze(2)


def masked_mean(values, mask):
    if mask is None:
        return values.mean()
    if mask.dim() == values.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_smooth_l1(pred, target, mask=None):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return masked_mean(loss, mask)


def masked_bce_with_logits(logits, labels, mask=None):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return masked_mean(loss, mask)


def variant_config(name):
    cfg = {
        "target_mode": "conditional",
        "core_only": True,
        "ldhn_protection": True,
        "tail_protection": True,
    }
    if name == "f4_global_strat_control":
        cfg.update({"target_mode": "global", "core_only": False})
    elif name == "f4_cond_strat_core":
        cfg.update({"target_mode": "conditional", "ldhn_protection": False})
    elif name == "f4_cond_strat_ldhn":
        cfg.update({"target_mode": "conditional"})
    elif name == "f4_excess_strat_ldhn":
        cfg.update({"target_mode": "excess"})
    else:
        raise ValueError(f"Unknown F4 variant: {name}")
    return cfg


def collect_fit_arrays(model, density_head, names, device, target_info, density_stats, args):
    dataset = v2e.Haze4KPairDataset(names, args.data_dir, max_items=args.fit_limit, seed=args.seed)
    raw_need_parts = []
    density_parts = []
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2b.v2.pad32(hazy)
            a0, _ = d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
            density = v2b.v2.normalize(
                v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            raw_need_parts.append(sample_map(raw_need, args.fit_grid))
            density_parts.append(sample_map(density, args.fit_grid))
            if (idx + 1) % args.progress_every == 0:
                print(f"f4_fit_collect {idx + 1}/{len(dataset)}", flush=True)
    raw_need = np.concatenate(raw_need_parts).astype(np.float32, copy=False)
    density = np.concatenate(density_parts).astype(np.float32, copy=False)
    return v2f.fit_cdf_tables(raw_need, density, args.density_bins)


def write_fit_summary(path, fit):
    rows = []
    q_points = [0, 20, 33, 50, 66, 80, 100]
    q_grid = np.asarray(fit["q_grid"])
    for table in fit["tables"]:
        row = {
            "bin": int(table["bin"]),
            "count": int(table["count"]),
            "raw_need_mean": float(table["raw_need_mean"]),
        }
        for q in q_points:
            idx = int(np.argmin(np.abs(q_grid - q / 100.0)))
            row[f"raw_need_q{q}"] = float(table["raw_need_q"][idx])
        rows.append(row)
    write_csv(path, rows)


def target_for_mode(raw_need, global_target, density_target, fit, mode):
    if mode == "global":
        return global_target
    if mode == "conditional":
        return torch_conditional_cdf(raw_need, density_target, fit, "conditional")
    if mode == "excess":
        return torch_conditional_cdf(raw_need, density_target, fit, "excess")
    raise ValueError(mode)


def train_variant(model, density_head, train_names, device, target_info, density_stats, fit, args, variant):
    cfg = variant_config(variant)
    head = DensityStratifiedNeedHead(density_bins=args.density_bins, hidden=args.head_hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = v2b.Haze4KPairDataset(
        train_names, args.data_dir, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=v2b.collate_pairs,
    )
    thresholds = [0.20, 0.33, 0.66, 0.80]
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    train_rows = []
    model.eval()
    density_head.eval()
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        smooth_losses = []
        ordinal_losses = []
        pair_losses = []
        ldhn_losses = []
        tail_losses = []
        start = time.time()
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, context = d7c.convir_a0_context(model, density_head, hazy)
                raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
                global_target = v2b.make_target(raw_need, target_info, "quantile")
                density_target = v2b.v2.normalize(
                    v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                )
                train_target = target_for_mode(raw_need, global_target, density_target, fit, cfg["target_mode"])
                density_bin = torch_bin_ids(density_target, fit["edges"])
            pred, mixed_logits, logits_by_bin, _ = head(context.detach(), fit["edges"], args.density_temperature)
            core_mask = (train_target <= args.cond_low) | (train_target >= args.cond_high)
            smooth = masked_smooth_l1(pred, train_target, core_mask if cfg["core_only"] else None)
            ordinal = pred.new_tensor(0.0)
            for i, thr in enumerate(thresholds):
                labels = (train_target >= thr).float()
                ordinal = ordinal + masked_bce_with_logits(mixed_logits[:, i : i + 1], labels)
                for b in range(args.density_bins):
                    ordinal = ordinal + args.stratum_ordinal_weight * masked_bce_with_logits(
                        logits_by_bin[:, b, i : i + 1],
                        labels,
                        density_bin == b,
                    )
            cond_pos_low = (train_target >= args.cond_high) & (density_target <= density_q33)
            cond_neg_low = (train_target <= args.cond_low) & (density_target <= density_q33)
            global_ldhn = (global_target >= q66) & (density_target <= density_q33)
            global_hn = (global_target <= q33) & (density_target <= density_q33)
            pos_mask = cond_pos_low | (global_ldhn if cfg["ldhn_protection"] else torch.zeros_like(global_ldhn))
            hn_mask = cond_neg_low | (global_hn if cfg["tail_protection"] else torch.zeros_like(global_hn))
            pair = v2d.pairwise_loss(pred, pos_mask, hn_mask, args.pair_margin, args.pair_sample)
            ldhn_loss = v2d.positive_response_loss(pred, pos_mask, args.tau_pos)
            tail_loss = (
                v2d.hard_negative_loss(pred, hn_mask, args.tau_neg)
                + v2d.topk_hn_loss(pred, hn_mask, args.tau_neg, args.topk_fraction)
            )
            loss = (
                args.smooth_weight * smooth
                + args.ordinal_weight * ordinal
                + args.pair_weight * pair
                + args.pos_weight * ldhn_loss
                + args.hn_weight * tail_loss
                + args.tv_weight * v2b.tv_loss(pred)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            smooth_losses.append(float(smooth.detach().cpu()))
            ordinal_losses.append(float(ordinal.detach().cpu()))
            pair_losses.append(float(pair.detach().cpu()))
            ldhn_losses.append(float(ldhn_loss.detach().cpu()))
            tail_losses.append(float(tail_loss.detach().cpu()))
            if step % args.progress_every == 0:
                print(
                    f"{variant} epoch={epoch} step={step}/{len(loader)} "
                    f"loss={statistics.mean(losses[-args.progress_every:]):.6f}",
                    flush=True,
                )
        row = {
            "variant": variant,
            "epoch": epoch,
            "steps": len(loader),
            "loss_mean": statistics.mean(losses),
            "smooth_l1_mean": statistics.mean(smooth_losses),
            "ordinal_loss_mean": statistics.mean(ordinal_losses),
            "pair_loss_mean": statistics.mean(pair_losses),
            "ldhn_positive_loss_mean": statistics.mean(ldhn_losses),
            "tail_loss_mean": statistics.mean(tail_losses),
            "elapsed_sec": time.time() - start,
            "target_mode": cfg["target_mode"],
            "core_only": cfg["core_only"],
            "ldhn_protection": cfg["ldhn_protection"],
        }
        train_rows.append(row)
        print(json.dumps(row), flush=True)
    return head, train_rows


def collect_prediction_pack(model, density_head, head, names, device, target_info, density_stats, fit, args, split_name, limit):
    dataset = v2e.Haze4KPairDataset(names, args.data_dir, max_items=limit, seed=args.seed)
    records = []
    concat = {k: [] for k in ["global_target", "cond_target", "excess_target", "density", "density_pred", "pred"]}
    head.eval()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2b.v2.pad32(hazy)
            a0, context = d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            context = context[:, :, :h, :w]
            raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
            global_target = v2b.make_target(raw_need, target_info, "quantile")
            density_target = v2b.v2.normalize(
                v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            cond_target = torch_conditional_cdf(raw_need, density_target, fit, "conditional")
            excess_target = torch_conditional_cdf(raw_need, density_target, fit, "excess")
            pred, _, _, _ = head(context, fit["edges"], args.density_temperature)
            density_pred = context[:, -1:]
            rec = {
                "name": name,
                "global_target": sample_map(global_target, args.metric_sample_size),
                "cond_target": sample_map(cond_target, args.metric_sample_size),
                "excess_target": sample_map(excess_target, args.metric_sample_size),
                "density": sample_map(density_target, args.metric_sample_size),
                "density_pred": sample_map(density_pred, args.metric_sample_size),
                "pred": sample_map(pred, args.metric_sample_size),
            }
            for key in concat:
                concat[key].append(rec[key])
            records.append(rec)
            if (idx + 1) % args.progress_every == 0:
                print(f"collect_{split_name} {idx + 1}/{len(dataset)}", flush=True)
    arrays = {key: np.concatenate(vals).astype(np.float32, copy=False) for key, vals in concat.items()}
    return {"records": records, "arrays": arrays}


def records_for_target(pack, target_key):
    return [{"name": rec["name"], "target": rec[target_key], "density": rec["density"]} for rec in pack["records"]]


def metric_curve(variant, train_score, val_pack, target_key, target_label, target_info, density_stats, args):
    if target_key == "global_target":
        q33 = float(target_info["quantile"]["q33"])
        q66 = float(target_info["quantile"]["q66"])
    else:
        q33 = args.cond_low
        q66 = args.cond_high
    density_q33 = float(density_stats["density"]["q33"])
    val_target = val_pack["arrays"][target_key]
    val_score = val_pack["arrays"]["pred"]
    val_density = val_pack["arrays"]["density"]
    val_records = records_for_target(val_pack, target_key)
    selected_threshold = v2e.matched_threshold(train_score, args.target_coverage)
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(train_score, np.linspace(0.0, 1.0, args.threshold_grid)),
                np.linspace(0.05, 0.95, 19),
                np.asarray([selected_threshold], dtype=np.float32),
            ]
        )
    )
    base = v2e.base_metrics(val_target, val_score, val_density, q33, q66, density_q33)
    rows = []
    for thr in sorted(grid):
        row = v2e.threshold_metrics(thr, val_target, val_score, val_density, val_records, q33, q66, density_q33, base)
        row.update({"variant": variant, "target_contract": target_label, "selected_threshold_train_coverage": selected_threshold})
        rows.append(row)
    return rows, selected_threshold


def summarize_curve(rows, selected_threshold):
    safe = [
        r
        for r in rows
        if r["false_global"] <= SAFETY_FALSE_GLOBAL
        and r["false_per_image_p90"] <= SAFETY_FALSE_P90
        and r["false_per_image_p95"] <= SAFETY_FALSE_P95
    ]
    ldhn = [r for r in rows if r["ldhn_recall"] >= LDHN_RECALL_GATE]
    both = [r for r in safe if r["ldhn_recall"] >= LDHN_RECALL_GATE]
    selected = min(rows, key=lambda r: abs(r["threshold"] - selected_threshold))
    best_safe = max(safe, key=lambda r: r["ldhn_recall"]) if safe else {}
    best_both = max(both, key=lambda r: r["ldhn_recall"]) if both else {}
    first_ldhn = min(ldhn, key=lambda r: (r["false_per_image_p95"], r["false_per_image_p90"], r["false_global"])) if ldhn else {}
    gate_pass = bool(
        selected.get("spearman", 0.0) >= 0.50
        and selected.get("auroc", 0.0) >= 0.83
        and selected.get("auprc", 0.0) >= 0.62
        and 0.25 <= selected.get("coverage", 0.0) <= 0.35
        and selected.get("false_global", 1.0) <= SAFETY_FALSE_GLOBAL
        and selected.get("false_per_image_p90", 1.0) <= SAFETY_FALSE_P90
        and selected.get("false_per_image_p95", 1.0) <= SAFETY_FALSE_P95
        and selected.get("ldhn_recall", 0.0) >= LDHN_RECALL_GATE
        and selected.get("ldhn_precision", 0.0) >= 0.55
    )
    return {
        "selected_threshold": selected.get("threshold", math.nan),
        "selected_coverage": selected.get("coverage", math.nan),
        "selected_spearman": selected.get("spearman", math.nan),
        "selected_auroc": selected.get("auroc", math.nan),
        "selected_auprc": selected.get("auprc", math.nan),
        "selected_precision": selected.get("precision", math.nan),
        "selected_recall": selected.get("recall", math.nan),
        "selected_false_global": selected.get("false_global", math.nan),
        "selected_false_p90": selected.get("false_per_image_p90", math.nan),
        "selected_false_p95": selected.get("false_per_image_p95", math.nan),
        "selected_ldhn_recall": selected.get("ldhn_recall", math.nan),
        "selected_ldhn_precision": selected.get("ldhn_precision", math.nan),
        "safe_points": len(safe),
        "ldhn_passing_points": len(ldhn),
        "safe_and_ldhn_points": len(both),
        "best_safe_ldhn_recall": best_safe.get("ldhn_recall", math.nan),
        "best_safe_false_p90": best_safe.get("false_per_image_p90", math.nan),
        "best_safe_false_p95": best_safe.get("false_per_image_p95", math.nan),
        "best_safe_and_ldhn_threshold": best_both.get("threshold", math.nan),
        "best_safe_and_ldhn_recall": best_both.get("ldhn_recall", math.nan),
        "first_ldhn_false_global": first_ldhn.get("false_global", math.nan),
        "first_ldhn_false_p90": first_ldhn.get("false_per_image_p90", math.nan),
        "first_ldhn_false_p95": first_ldhn.get("false_per_image_p95", math.nan),
        "selected_gate_pass": gate_pass,
    }


def per_image_safety_rows(variant, pack, target_key, threshold, target_info, density_stats, args):
    if target_key == "global_target":
        q33 = float(target_info["quantile"]["q33"])
        q66 = float(target_info["quantile"]["q66"])
    else:
        q33 = args.cond_low
        q66 = args.cond_high
    density_q33 = float(density_stats["density"]["q33"])
    rows = []
    for rec in pack["records"]:
        target = rec[target_key]
        density = rec["density"]
        pred = rec["pred"]
        hn = (target <= q33) & (density <= density_q33)
        ldhn = (target >= q66) & (density <= density_q33)
        pred_high = pred >= threshold
        rows.append(
            {
                "variant": variant,
                "target_contract": target_key,
                "name": rec["name"],
                "threshold": float(threshold),
                "hard_negative_pixels": int(hn.sum()),
                "ldhn_pixels": int(ldhn.sum()),
                "false_tail_rate": float(pred_high[hn].mean()) if int(hn.sum()) else math.nan,
                "ldhn_recall": float(pred_high[ldhn].mean()) if int(ldhn.sum()) else math.nan,
                "pred_high_coverage": float(pred_high.mean()),
            }
        )
    return rows


def density_only_controls(train_pack, val_pack, target_info, density_stats, args):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    train_score = train_pack["arrays"]["density_pred"]
    val_score = val_pack["arrays"]["density_pred"]
    val_target = val_pack["arrays"]["global_target"]
    val_density = val_pack["arrays"]["density"]
    val_records = records_for_target(val_pack, "global_target")
    thr = v2e.matched_threshold(train_score, args.target_coverage)
    row = v2e.fixed_threshold_summary(
        "density_only_matched_threshold_f4",
        val_target,
        val_score,
        val_density,
        val_records,
        q33,
        q66,
        density_q33,
        thr,
    )
    row["target_coverage"] = args.target_coverage
    stratum_rows = []
    bin_id = v2f.assign_bins(val_density, np.asarray(args.fit_edges, dtype=np.float64))
    for b in range(args.density_bins):
        mask = bin_id == b
        if int(mask.sum()) < 10:
            continue
        high = val_target >= q66
        low_or_high = (val_target <= q33) | high
        local = mask & low_or_high
        if int(local.sum()) < 10 or len(np.unique(high[local])) < 2:
            auroc = math.nan
        else:
            auroc = v2b.auroc(high[local], val_score[local])
        stratum_rows.append(
            {
                "variant": "density_only",
                "density_bin": b,
                "rows": int(mask.sum()),
                "spearman": v2b.spearman(val_score[mask], val_target[mask]),
                "auroc_high_vs_low": auroc,
                "auprc_high": v2b.auprc(high[mask], val_score[mask]) if len(np.unique(high[mask])) > 1 else math.nan,
            }
        )
    return row, stratum_rows


def score_stratum_rows(variant, val_pack, target_info, args):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    score = val_pack["arrays"]["pred"]
    target = val_pack["arrays"]["global_target"]
    density = val_pack["arrays"]["density"]
    bin_id = v2f.assign_bins(density, np.asarray(args.fit_edges, dtype=np.float64))
    rows = []
    for b in range(args.density_bins):
        mask = bin_id == b
        if int(mask.sum()) < 10:
            continue
        high = target >= q66
        low_or_high = (target <= q33) | high
        local = mask & low_or_high
        rows.append(
            {
                "variant": variant,
                "density_bin": b,
                "rows": int(mask.sum()),
                "spearman": v2b.spearman(score[mask], target[mask]),
                "auroc_high_vs_low": (
                    v2b.auroc(high[local], score[local]) if int(local.sum()) >= 10 and len(np.unique(high[local])) > 1 else math.nan
                ),
                "auprc_high": v2b.auprc(high[mask], score[mask]) if len(np.unique(high[mask])) > 1 else math.nan,
            }
        )
    return rows


def write_docs(output_dir, closeout, summaries):
    lines = [
        "# v2f F4 Stratified Head Canary Summary",
        "",
        f"Status: `{closeout['status']}`",
        "",
        "Policy:",
        "",
        "- ConvIR-B frozen: yes",
        "- D3 density frozen: yes",
        "- D2/v3/RARM: not run",
        "- Locked Haze4K test usage: none",
        "",
        "Original v2e gate is still the primary decision contract. The density-conditioned target is a training redesign, not a replacement for the safety/LDHN audit.",
        "",
        "| Variant | Gate | Spearman | AUROC | AUPRC | Coverage | False p95 | LDHN recall | Safe+LDHN points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if row["target_contract"] != "global_v2e_contract":
            continue
        lines.append(
            f"| {row['variant']} | {row['selected_gate_pass']} | {row['selected_spearman']:.4f} | "
            f"{row['selected_auroc']:.4f} | {row['selected_auprc']:.4f} | {row['selected_coverage']:.4f} | "
            f"{row['selected_false_p95']:.4f} | {row['selected_ldhn_recall']:.4f} | {row['safe_and_ldhn_points']} |"
        )
    lines.append("")
    output_dir.joinpath("v2f_f4_stratified_head_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--density_artifact", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--source_commit", default="")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--density_bins", type=int, default=5)
    ap.add_argument("--density_temperature", type=float, default=0.08)
    ap.add_argument("--head_hidden", type=int, default=96)
    ap.add_argument("--fit_grid", type=int, default=64)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--smooth_weight", type=float, default=0.50)
    ap.add_argument("--ordinal_weight", type=float, default=0.20)
    ap.add_argument("--stratum_ordinal_weight", type=float, default=0.25)
    ap.add_argument("--pair_weight", type=float, default=0.30)
    ap.add_argument("--pos_weight", type=float, default=1.00)
    ap.add_argument("--hn_weight", type=float, default=1.20)
    ap.add_argument("--tv_weight", type=float, default=0.02)
    ap.add_argument("--tau_neg", type=float, default=0.50)
    ap.add_argument("--tau_pos", type=float, default=0.66)
    ap.add_argument("--pair_margin", type=float, default=0.15)
    ap.add_argument("--pair_sample", type=int, default=4096)
    ap.add_argument("--topk_fraction", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--target_coverage", type=float, default=v2f.DEFAULT_TARGET_COVERAGE)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--cond_low", type=float, default=0.33)
    ap.add_argument("--cond_high", type=float, default=0.66)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--fit_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["f4_global_strat_control", "f4_cond_strat_core", "f4_cond_strat_ldhn", "f4_excess_strat_ldhn"],
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.txt"
    with status_path.open("a", encoding="utf-8") as f:
        f.write(f"F4_CANARY_RUNNING {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    v2b.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names, val_names, device, model, density_stats, target_info = v2d.load_runtime(args)
    density_head = v2d.load_density_head(args.density_artifact, device)
    for module in [model, density_head]:
        v2e.set_frozen(module)

    start = time.time()
    print("F4_FIT_TARGET_TABLES_BEGIN", flush=True)
    fit = collect_fit_arrays(model, density_head, train_names, device, target_info, density_stats, args)
    args.fit_edges = fit["edges"]
    write_fit_summary(args.output_dir / "stratified_head_density_conditioned_fit_summary.csv", fit)

    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    train_rows = []
    summary_rows = []
    curve_rows = []
    per_image_rows = []
    leakage_rows = []
    stratum_rows = []
    density_control_rows = []

    first_train_pack = None
    first_val_pack = None
    candidate_passes = []
    for variant in args.variants:
        print(f"F4_TRAIN_{variant}_BEGIN", flush=True)
        head, rows = train_variant(model, density_head, train_names, device, target_info, density_stats, fit, args, variant)
        train_rows.extend(rows)
        torch.save(
            {
                "variant": variant,
                "state_dict": head.state_dict(),
                "args": vars(args),
                "density_edges": [float(x) if math.isfinite(float(x)) else str(x) for x in fit["edges"]],
            },
            artifact_dir / f"{variant}_head.pt",
        )

        print(f"F4_COLLECT_{variant}_TRAIN", flush=True)
        train_pack = collect_prediction_pack(
            model, density_head, head, train_names, device, target_info, density_stats, fit, args, "train_inner", args.train_limit
        )
        print(f"F4_COLLECT_{variant}_VAL", flush=True)
        val_pack = collect_prediction_pack(
            model, density_head, head, val_names, device, target_info, density_stats, fit, args, "val_inner", args.val_limit
        )
        if first_train_pack is None:
            first_train_pack = train_pack
            first_val_pack = val_pack

        for target_key, target_label in [
            ("global_target", "global_v2e_contract"),
            ("cond_target", "density_conditioned_contract"),
        ]:
            train_score = train_pack["arrays"]["pred"]
            rows_curve, selected_thr = metric_curve(
                variant, train_score, val_pack, target_key, target_label, target_info, density_stats, args
            )
            curve_rows.extend(rows_curve)
            summary = summarize_curve(rows_curve, selected_thr)
            summary.update({"variant": variant, "target_contract": target_label})
            summary_rows.append(summary)
            if target_key == "global_target":
                candidate_passes.append(summary["selected_gate_pass"])
                per_image_rows.extend(
                    per_image_safety_rows(variant, val_pack, target_key, summary["selected_threshold"], target_info, density_stats, args)
                )
        leakage_rows.append(
            {
                "variant": variant,
                "score_density_spearman": v2b.spearman(val_pack["arrays"]["pred"], val_pack["arrays"]["density"]),
                "score_global_target_spearman": v2b.spearman(val_pack["arrays"]["pred"], val_pack["arrays"]["global_target"]),
                "score_cond_target_spearman": v2b.spearman(val_pack["arrays"]["pred"], val_pack["arrays"]["cond_target"]),
                "cond_target_density_spearman": v2b.spearman(val_pack["arrays"]["cond_target"], val_pack["arrays"]["density"]),
            }
        )
        stratum_rows.extend(score_stratum_rows(variant, val_pack, target_info, args))
        print(json.dumps(summary_rows[-2:], indent=2), flush=True)

    if first_train_pack is not None and first_val_pack is not None:
        density_row, density_strata = density_only_controls(first_train_pack, first_val_pack, target_info, density_stats, args)
        density_control_rows.append(density_row)
        stratum_rows.extend(density_strata)

    write_csv(args.output_dir / "stratified_head_train_log.csv", train_rows)
    write_csv(args.output_dir / "stratified_head_ablation_summary.csv", summary_rows)
    write_csv(args.output_dir / "stratified_head_threshold_curve.csv", curve_rows)
    write_csv(args.output_dir / "stratified_head_ldhn_recall_curve.csv", curve_rows)
    write_csv(args.output_dir / "stratified_head_false_tail_curve.csv", curve_rows)
    write_csv(args.output_dir / "stratified_head_per_image_safety_metrics.csv", per_image_rows)
    write_csv(args.output_dir / "density_only_matched_threshold_summary.csv", density_control_rows)
    write_csv(args.output_dir / "density_only_matched_within_stratum_summary.csv", stratum_rows)
    write_csv(args.output_dir / "target_transform_leakage_audit.csv", leakage_rows)
    write_json(
        args.output_dir / "target_transform_leakage_audit.json",
        {
            "rows": leakage_rows,
            "interpretation": "Low cond_target_density_spearman and moderate score_target_spearman support density-conditioned residual need rather than pure density proxy.",
            "locked_haze4k_test_usage": "none",
        },
    )
    write_csv(
        args.output_dir / "resource_summary.csv",
        [
            {
                "route_id": ROUTE_ID,
                "source_commit": args.source_commit,
                "device": str(device),
                "train_images": len(train_names),
                "val_images": len(val_names),
                "epochs": args.epochs,
                "variants": " ".join(args.variants),
                "locked_haze4k_test_usage": "none",
                "D2": "not_run",
                "RARM": "not_connected_or_trained",
                "v3": "not_run",
            }
        ],
    )

    closeout = {
        "status": "COMPLETED_GATE_PASS" if any(candidate_passes) else "COMPLETED_GATE_FAIL",
        "phase": "F4_STRATIFIED_HEAD_CANARY",
        "elapsed_sec": time.time() - start,
        "selected_gate_pass_any_variant": bool(any(candidate_passes)),
        "locked_haze4k_test_usage": "none",
        "D2": "not_run",
        "RARM": "not_connected_or_trained",
        "v3": "not_run",
        "ConvIR_B": "frozen",
        "D3_density": "frozen",
        "next_gate": "If F4 gate fails, do not run v3/RARM; inspect summary for target/head failure. If it passes, run F5 controls before any v3 no-op audit.",
    }
    write_json(args.output_dir / "v2f_f4_stratified_head_closeout.json", closeout)
    write_docs(args.output_dir, closeout, summary_rows)
    with status_path.open("a", encoding="utf-8") as f:
        f.write(f"F4_CANARY_DONE status={closeout['status']} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write("V2F_F4_STRATIFIED_HEAD_CANARY_OK\n")
    print(json.dumps(closeout, indent=2), flush=True)
    print("V2F_F4_STRATIFIED_HEAD_CANARY_OK", flush=True)


if __name__ == "__main__":
    main()
