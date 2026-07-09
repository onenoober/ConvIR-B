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
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2B_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2b_need_calibration_repair.py"
spec = importlib.util.spec_from_file_location("chdrm_v2b_tool", V2B_TOOL)
v2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2b)

STRONG_PRED_THRESHOLD = 0.66
MIN_COVERAGE = 0.20
MAX_COVERAGE = 0.40
GATE = {
    "pearson": 0.33,
    "spearman": 0.32,
    "auroc": 0.70,
    "auprc": 0.42,
    "false_strong_global": 0.10,
    "false_strong_p90": 0.15,
    "high_need_recall": 0.25,
}


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


def finite_mean(values):
    values = [v for v in values if not math.isnan(v)]
    return statistics.mean(values) if values else math.nan


def finite_quantile(values, p):
    values = sorted(v for v in values if not math.isnan(v))
    if not values:
        return math.nan
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return float(values[idx])


def sample_np(x, sample_size):
    return v2b.sample_values(x, sample_size).astype(np.float32, copy=False)


def load_need_head(ckpt_path, device):
    head = v2b.ScalarNeedHead(4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    return head


def load_density_head(ckpt_path, device):
    head = v2b.v2.DensityNeedHead(1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return head


def predict_ordinal(head, res1):
    logits = head(res1)
    probs = [torch.sigmoid(logits[:, i : i + 1]) for i in range(4)]
    return torch.stack(probs, dim=0).mean(dim=0), logits


def density_pred_from_head(head, res1):
    return torch.sigmoid(head(res1))


def load_runtime(args):
    v2b.set_seed(args.seed)
    train_names = v2b.load_split_names(args.split_json, "train_inner")
    val_names = v2b.load_split_names(args.split_json, "val_inner")
    if len(train_names) != 2400 or len(val_names) != 600:
        raise ValueError(f"Expected 2400/600 split, got {len(train_names)}/{len(val_names)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v2b.load_model(args.checkpoint, device)
    density_stats = json.loads(Path(args.v2_thresholds).read_text(encoding="utf-8"))
    target_info = json.loads(Path(args.v2b_thresholds).read_text(encoding="utf-8"))
    return train_names, val_names, device, model, density_stats, target_info


def collect_sample_records(model, need_head, density_head, names, data_dir, device, target_info, density_stats, args, limit=0):
    dataset = v2b.Haze4KPairDataset(names, data_dir, max_items=limit, seed=args.seed)
    q_mode = "quantile"
    records = []
    target_values = []
    pred_values = []
    density_values = []
    density_pred_values = []
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2b.v2.pad32(hazy)
            a0, res1 = v2b.v2.convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            res1 = res1[:, :, :h, :w]
            raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2b.make_target(raw_need, target_info, q_mode)
            density_target = v2b.v2.normalize(
                v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            pred, _ = predict_ordinal(need_head, res1)
            density_pred = density_pred_from_head(density_head, res1) if density_head is not None else None
            target_np = sample_np(target, args.metric_sample_size)
            pred_np = sample_np(pred, args.metric_sample_size)
            density_np = sample_np(density_target, args.metric_sample_size)
            density_pred_np = sample_np(density_pred, args.metric_sample_size) if density_pred is not None else None
            row = {
                "name": name,
                "target": target_np,
                "pred": pred_np,
                "density": density_np,
                "density_pred": density_pred_np,
            }
            records.append(row)
            target_values.append(target_np)
            pred_values.append(pred_np)
            density_values.append(density_np)
            if density_pred_np is not None:
                density_pred_values.append(density_pred_np)
            if (idx + 1) % args.progress_every == 0:
                print(f"collect_sample {idx + 1}/{len(dataset)}", flush=True)
    return {
        "records": records,
        "target": np.concatenate(target_values).astype(np.float32, copy=False),
        "pred": np.concatenate(pred_values).astype(np.float32, copy=False),
        "density": np.concatenate(density_values).astype(np.float32, copy=False),
        "density_pred": np.concatenate(density_pred_values).astype(np.float32, copy=False)
        if density_pred_values
        else None,
    }


def hard_negative_mask(target, density, q33, density_q33):
    return (target <= q33) & (density <= density_q33)


def threshold_metrics(threshold, target, pred, density, records, q33, q66, density_q33, base_metrics):
    high = target >= q66
    hard_negative = hard_negative_mask(target, density, q33, density_q33)
    pred_high = pred >= threshold
    precision = float((pred_high & high).sum() / max(int(pred_high.sum()), 1))
    recall = float((pred_high & high).sum() / max(int(high.sum()), 1))
    false_global = float(pred_high[hard_negative].mean()) if int(hard_negative.sum()) else math.nan
    low_density_high = high & (density <= density_q33)
    low_density_high_recall = (
        float(pred_high[low_density_high].mean()) if int(low_density_high.sum()) else math.nan
    )
    per_rates = []
    for record in records:
        hn = hard_negative_mask(record["target"], record["density"], q33, density_q33)
        ph = record["pred"] >= threshold
        per_rates.append(float(ph[hn].mean()) if int(hn.sum()) else math.nan)
    row = {
        "threshold": float(threshold),
        "pred_high_coverage": float(pred_high.mean()),
        "target_high_recall": recall,
        "target_high_precision": precision,
        "low_context_false_strong_global": false_global,
        "low_context_false_strong_per_image_mean": finite_mean(per_rates),
        "low_context_false_strong_per_image_p90": finite_quantile(per_rates, 0.90),
        "low_context_false_strong_per_image_p95": finite_quantile(per_rates, 0.95),
        "high_need_recall_in_low_density_regions": low_density_high_recall,
    }
    row.update(base_metrics)
    return row


def write_threshold_audit(output_dir, train_pack, val_pack, target_info, density_stats, args):
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    density_q66 = float(density_stats["density"]["q66"])
    train_pred = train_pack["pred"]
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(train_pred, np.linspace(0.0, 1.0, args.threshold_grid)),
                np.linspace(0.05, 0.95, 19),
                np.asarray([STRONG_PRED_THRESHOLD], dtype=np.float32),
            ]
        )
    )
    high = val_pack["target"] >= q66
    low_or_high = (val_pack["target"] <= q33) | high
    hard_negative = hard_negative_mask(val_pack["target"], val_pack["density"], q33, density_q33)
    positives_or_hn = high | hard_negative
    hn_scores = val_pack["pred"][hard_negative]
    if hn_scores.size:
        k = max(1, int(math.ceil(hn_scores.size * 0.05)))
        hard_negative_top5_mean = float(np.mean(np.sort(hn_scores)[-k:]))
    else:
        hard_negative_top5_mean = math.nan
    base_metrics = {
        "AUPRC_high": v2b.auprc(high, val_pack["pred"]),
        "AUROC_high_vs_low": v2b.auroc(high[low_or_high], val_pack["pred"][low_or_high]),
        "AUROC_high_vs_hard_negative": (
            v2b.auroc(high[positives_or_hn], val_pack["pred"][positives_or_hn])
            if int(positives_or_hn.sum())
            else math.nan
        ),
        "hard_negative_top5_mean": hard_negative_top5_mean,
    }
    rows = [
        threshold_metrics(
            t,
            val_pack["target"],
            val_pack["pred"],
            val_pack["density"],
            val_pack["records"],
            q33,
            q66,
            density_q33,
            base_metrics,
        )
        for t in sorted(grid)
    ]
    write_csv(output_dir / "need_rank_safety_curve_d6c_identity.csv", rows)
    safe = [
        r
        for r in rows
        if MIN_COVERAGE <= r["pred_high_coverage"] <= MAX_COVERAGE
        and r["low_context_false_strong_global"] <= GATE["false_strong_global"]
    ]
    p90_safe = [r for r in safe if r["low_context_false_strong_per_image_p90"] <= GATE["false_strong_p90"]]
    if p90_safe:
        selected = sorted(p90_safe, key=lambda r: (abs(r["pred_high_coverage"] - 0.30), -r["target_high_recall"]))[0]
        interpretation = "D7-0 found a globally and per-image safe raw operating point."
    elif safe:
        selected = sorted(safe, key=lambda r: (abs(r["pred_high_coverage"] - 0.30), -r["target_high_recall"]))[0]
        interpretation = "D7-0 found a globally safe raw point, but per-image tail remains unsafe."
    else:
        candidates = [r for r in rows if MIN_COVERAGE <= r["pred_high_coverage"] <= MAX_COVERAGE]
        selected = sorted(candidates or rows, key=lambda r: (r["low_context_false_strong_global"], abs(r["pred_high_coverage"] - 0.30)))[0]
        interpretation = "D7-0 found no safe coverage/false-strong threshold; train hard-negative repair."
    write_operating_points(output_dir, rows, safe, p90_safe, selected, interpretation)
    write_false_strong_bins(output_dir, selected["threshold"], val_pack, q33, q66, density_q33, density_q66)
    write_per_image_false_distribution(output_dir, selected["threshold"], val_pack, q33, density_q33)
    write_topk_leakage(output_dir, selected["threshold"], val_pack, q33, density_q33)
    bins, mono, valid = v2b.bin_rows(
        val_pack["target"], val_pack["pred"], [q20, q33, q66, q80], "d7_0_d6c_identity"
    )
    for row in bins:
        row["selected_threshold"] = selected["threshold"]
    write_csv(output_dir / "need_calibration_bins_by_variant.csv", bins)
    return selected, rows


def write_operating_points(output_dir, rows, safe, p90_safe, selected, interpretation):
    best_by_false = sorted(rows, key=lambda r: (r["low_context_false_strong_global"], -r["pred_high_coverage"]))[:8]
    best_by_coverage = sorted(rows, key=lambda r: abs(r["pred_high_coverage"] - 0.30))[:8]
    lines = [
        "# D7-0 Need Rank Safety Operating Points",
        "",
        f"Interpretation: {interpretation}",
        "",
        f"Selected threshold: `{selected['threshold']:.6f}`",
        f"Selected coverage: `{selected['pred_high_coverage']:.6f}`",
        f"Selected false-strong global: `{selected['low_context_false_strong_global']:.6f}`",
        f"Selected false-strong p90: `{selected['low_context_false_strong_per_image_p90']:.6f}`",
        "",
        f"Safe coverage/global rows: {len(safe)}",
        f"Safe coverage/global/p90 rows: {len(p90_safe)}",
        "",
        "Forbidden in this stage: D2, RARM connection, v3 expansion, locked Haze4K test.",
        "Locked Haze4K test usage: none.",
        "",
        "## Nearest Coverage Points",
        "",
        "| threshold | coverage | recall | precision | false_global | false_p90 | low_density_high_recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in best_by_coverage:
        lines.append(
            f"| {r['threshold']:.6f} | {r['pred_high_coverage']:.4f} | {r['target_high_recall']:.4f} | "
            f"{r['target_high_precision']:.4f} | {r['low_context_false_strong_global']:.4f} | "
            f"{r['low_context_false_strong_per_image_p90']:.4f} | {r['high_need_recall_in_low_density_regions']:.4f} |"
        )
    lines += ["", "## Lowest False-Strong Points", ""]
    lines += [
        "| threshold | coverage | recall | precision | false_global | false_p90 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in best_by_false:
        lines.append(
            f"| {r['threshold']:.6f} | {r['pred_high_coverage']:.4f} | {r['target_high_recall']:.4f} | "
            f"{r['target_high_precision']:.4f} | {r['low_context_false_strong_global']:.4f} | "
            f"{r['low_context_false_strong_per_image_p90']:.4f} |"
        )
    (output_dir / "need_rank_safety_operating_points.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bin_label(x, q33, q66):
    if x <= q33:
        return "low"
    if x >= q66:
        return "high"
    return "mid"


def write_false_strong_bins(output_dir, threshold, val_pack, q33, q66, density_q33, density_q66):
    rows = []
    density_masks = {
        "low": val_pack["density"] <= density_q33,
        "mid": (val_pack["density"] > density_q33) & (val_pack["density"] < density_q66),
        "high": val_pack["density"] >= density_q66,
    }
    need_masks = {
        "low": val_pack["target"] <= q33,
        "mid": (val_pack["target"] > q33) & (val_pack["target"] < q66),
        "high": val_pack["target"] >= q66,
    }
    pred_high_all = val_pack["pred"] >= threshold
    target_high_all = val_pack["target"] >= q66
    for d_bin in ["low", "mid", "high"]:
        for n_bin in ["low", "mid", "high"]:
            mask = density_masks[d_bin] & need_masks[n_bin]
            count = int(mask.sum())
            pred_high = int((pred_high_all & mask).sum())
            target_high = int((target_high_all & mask).sum())
            rows.append(
                {
                    "selected_threshold": threshold,
                    "density_bin": d_bin,
                    "need_bin": n_bin,
                    "count": count,
                    "pred_high_fraction": pred_high / max(count, 1),
                    "target_high_fraction": target_high / max(count, 1),
                }
            )
    write_csv(output_dir / "need_false_strong_by_density_need_bin.csv", rows)


def write_per_image_false_distribution(output_dir, threshold, val_pack, q33, density_q33):
    rows = []
    for rec in val_pack["records"]:
        hn = hard_negative_mask(rec["target"], rec["density"], q33, density_q33)
        pred_high = rec["pred"] >= threshold
        rows.append(
            {
                "selected_threshold": threshold,
                "name": rec["name"],
                "hard_negative_pixels": int(hn.sum()),
                "false_strong_pixels": int((hn & pred_high).sum()),
                "false_strong_rate": float(pred_high[hn].mean()) if int(hn.sum()) else math.nan,
                "pred_mean": float(np.mean(rec["pred"])),
                "pred_max": float(np.max(rec["pred"])),
                "target_mean": float(np.mean(rec["target"])),
            }
        )
    write_csv(output_dir / "need_per_image_false_strong_distribution.csv", rows)


def write_topk_leakage(output_dir, threshold, val_pack, q33, density_q33):
    rows = []
    for frac in [0.01, 0.02, 0.05, 0.10]:
        scores = []
        for rec in val_pack["records"]:
            hn = hard_negative_mask(rec["target"], rec["density"], q33, density_q33)
            scores.extend(rec["pred"][hn].tolist())
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size:
            k = max(1, int(math.ceil(scores.size * frac)))
            top = np.sort(scores)[-k:]
            rows.append(
                {
                    "selected_threshold": threshold,
                    "topk_fraction": frac,
                    "hard_negative_count": int(scores.size),
                    "topk_mean": float(np.mean(top)),
                    "topk_min": float(np.min(top)),
                    "topk_max": float(np.max(top)),
                    "threshold_leakage_fraction": float(np.mean(scores >= threshold)),
                }
            )
    write_csv(output_dir / "need_topk_hard_negative_leakage.csv", rows)


def train_variant(model, train_names, data_dir, device, target_info, density_stats, variant, output_dir, args):
    head = v2b.ScalarNeedHead(4).to(device)
    init_path = Path(args.v2b_artifact_dir) / "d6c_ordinal_quantile_head.pt"
    if args.init_from_v2b and not variant.startswith("d7s"):
        head.load_state_dict(torch.load(init_path, map_location=device)["state_dict"])
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = v2b.Haze4KPairDataset(
        train_names, data_dir, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
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
    bce = nn.BCEWithLogitsLoss()
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    thresholds = [0.20, 0.33, 0.66, 0.80]
    rows = []
    head.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        smooth_losses = []
        ord_losses = []
        hn_losses = []
        pair_losses = []
        topk_losses = []
        pos_losses = []
        start = time.time()
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, res1 = v2b.v2.convir_a0_and_res1(model, hazy)
                raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
                target = v2b.make_target(raw_need, target_info, "quantile")
                density_target = v2b.v2.normalize(
                    v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                )
                if variant.startswith("d7s"):
                    target = shuffle_batch_target(target)
            pred, logits = predict_ordinal(head, res1.detach())
            ord_loss = logits.new_tensor(0.0)
            for i, thr in enumerate(thresholds):
                ord_loss = ord_loss + bce(logits[:, i : i + 1], (target >= thr).float())
            smooth = F.smooth_l1_loss(pred, target)
            hn_mask = (target <= q33) & (density_target <= density_q33)
            pos_mask = target >= q66
            hn_loss = hard_negative_loss(pred, hn_mask, args.tau_neg)
            pos_loss = positive_response_loss(pred, pos_mask, args.tau_pos)
            pair_loss = pairwise_loss(pred, pos_mask, hn_mask, args.pair_margin, args.pair_sample)
            topk_loss = pred.new_tensor(0.0)
            if "topk" in variant:
                topk_loss = topk_hn_loss(pred, hn_mask, args.tau_neg, args.topk_fraction)
            loss = (
                smooth
                + args.ordinal_weight * ord_loss
                + args.hn_weight * hn_loss
                + args.pos_weight * pos_loss
                + args.pair_weight * pair_loss
                + args.topk_weight * topk_loss
                + args.tv_weight * v2b.tv_loss(pred)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
            smooth_losses.append(float(smooth.detach().item()))
            ord_losses.append(float(ord_loss.detach().item()))
            hn_losses.append(float(hn_loss.detach().item()))
            pos_losses.append(float(pos_loss.detach().item()))
            pair_losses.append(float(pair_loss.detach().item()))
            topk_losses.append(float(topk_loss.detach().item()))
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
            "ordinal_bce_sum_mean": statistics.mean(ord_losses),
            "hard_negative_loss_mean": statistics.mean(hn_losses),
            "positive_response_loss_mean": statistics.mean(pos_losses),
            "pairwise_loss_mean": statistics.mean(pair_losses),
            "topk_loss_mean": statistics.mean(topk_losses),
            "elapsed_sec": time.time() - start,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    torch.save(
        {
            "variant": variant,
            "state_dict": head.state_dict(),
            "target_mode": "quantile",
            "init_path": str(init_path) if args.init_from_v2b and not variant.startswith("d7s") else "scratch_control",
            "args": vars(args),
        },
        artifact_dir / f"{variant}_head.pt",
    )
    return head, rows


def shuffle_batch_target(target):
    if target.shape[0] > 1:
        shift = random.randint(1, target.shape[0] - 1)
        return torch.roll(target, shifts=shift, dims=0)
    return torch.flip(target, dims=[-1])


def hard_negative_loss(pred, hn_mask, tau_neg):
    vals = pred[hn_mask]
    if vals.numel() == 0:
        return pred.new_tensor(0.0)
    return torch.relu(vals - tau_neg).pow(2).mean()


def positive_response_loss(pred, pos_mask, tau_pos):
    vals = pred[pos_mask]
    if vals.numel() == 0:
        return pred.new_tensor(0.0)
    return torch.relu(tau_pos - vals).pow(2).mean()


def topk_hn_loss(pred, hn_mask, tau_neg, topk_fraction):
    vals = pred[hn_mask]
    if vals.numel() == 0:
        return pred.new_tensor(0.0)
    k = max(1, int(math.ceil(vals.numel() * topk_fraction)))
    top = torch.topk(vals, k=k, largest=True).values
    return torch.relu(top - tau_neg).pow(2).mean()


def pairwise_loss(pred, pos_mask, hn_mask, margin, sample_count):
    pos = pred[pos_mask]
    hn = pred[hn_mask]
    if pos.numel() == 0 or hn.numel() == 0:
        return pred.new_tensor(0.0)
    k = min(int(sample_count), pos.numel(), hn.numel())
    pos_idx = torch.randperm(pos.numel(), device=pred.device)[:k]
    hn_idx = torch.randperm(hn.numel(), device=pred.device)[:k]
    return torch.relu(margin - pos[pos_idx] + hn[hn_idx]).mean()


def evaluate_need_head(model, head, val_names, data_dir, device, target_info, density_stats, variant, output_dir, args):
    dataset = v2b.Haze4KPairDataset(val_names, data_dir, max_items=args.val_limit, seed=args.seed)
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    records = []
    target_parts = []
    pred_parts = []
    density_parts = []
    head.eval()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2b.v2.pad32(hazy)
            a0, res1 = v2b.v2.convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            res1 = res1[:, :, :h, :w]
            raw_need = v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2b.make_target(raw_need, target_info, "quantile")
            density_target = v2b.v2.normalize(
                v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            pred, _ = predict_ordinal(head, res1)
            target_np = sample_np(target, args.metric_sample_size)
            pred_np = sample_np(pred, args.metric_sample_size)
            density_np = sample_np(density_target, args.metric_sample_size)
            records.append({"name": name, "target": target_np, "pred": pred_np, "density": density_np})
            target_parts.append(target_np)
            pred_parts.append(pred_np)
            density_parts.append(density_np)
            if (idx + 1) % args.progress_every == 0:
                print(f"eval {variant} {idx + 1}/{len(dataset)}", flush=True)
    summary, per_rows, hist_rows = summarize_prediction_records(
        variant,
        records,
        np.concatenate(target_parts),
        np.concatenate(pred_parts),
        np.concatenate(density_parts),
        q20,
        q33,
        q66,
        q80,
        density_q33,
    )
    return summary, per_rows, hist_rows


def summarize_prediction_records(variant, records, target_all, pred_all, density_all, q20, q33, q66, q80, density_q33):
    high = target_all >= q66
    low_or_high = (target_all <= q33) | high
    hard_negative = hard_negative_mask(target_all, density_all, q33, density_q33)
    pred_high = pred_all >= STRONG_PRED_THRESHOLD
    bins, mono, valid = v2b.bin_rows(target_all, pred_all, [q20, q33, q66, q80], variant)
    per_rows = []
    false_rates = []
    for rec in records:
        hn = hard_negative_mask(rec["target"], rec["density"], q33, density_q33)
        ph = rec["pred"] >= STRONG_PRED_THRESHOLD
        false_rate = float(ph[hn].mean()) if int(hn.sum()) else math.nan
        false_rates.append(false_rate)
        per_rows.append(
            {
                "variant": variant,
                "name": rec["name"],
                "need_pearson": v2b.pearson(rec["pred"], rec["target"]),
                "need_mae": float(np.mean(np.abs(rec["pred"] - rec["target"]))),
                "need_target_mean": float(np.mean(rec["target"])),
                "need_pred_mean": float(np.mean(rec["pred"])),
                "need_target_max": float(np.max(rec["target"])),
                "need_pred_max": float(np.max(rec["pred"])),
                "hard_negative_pixels": int(hn.sum()),
                "false_strong_pixels": int((ph & hn).sum()),
                "low_context_false_strong_rate": false_rate,
            }
        )
    positives_or_hn = high | hard_negative
    low_density_high = high & (density_all <= density_q33)
    hn_scores = pred_all[hard_negative]
    hn_top5 = math.nan
    if hn_scores.size:
        k = max(1, int(math.ceil(hn_scores.size * 0.05)))
        hn_top5 = float(np.mean(np.sort(hn_scores)[-k:]))
    summary = {
        "variant": variant,
        "eval_pixels": int(target_all.size),
        "need_pearson": v2b.pearson(pred_all, target_all),
        "need_spearman": v2b.spearman(pred_all, target_all),
        "need_auroc_high_vs_low": v2b.auroc(high[low_or_high], pred_all[low_or_high]),
        "need_auprc_high": v2b.auprc(high, pred_all),
        "need_pred_high_coverage": float(pred_high.mean()),
        "need_target_high_coverage": float(high.mean()),
        "target_high_recall": float(pred_high[high].mean()) if int(high.sum()) else math.nan,
        "target_high_precision": float((pred_high & high).sum() / max(int(pred_high.sum()), 1)),
        "high_need_recall_in_low_density_regions": (
            float(pred_high[low_density_high].mean()) if int(low_density_high.sum()) else math.nan
        ),
        "need_monotonic_pairs": mono,
        "need_monotonic_valid_pairs": valid,
        "low_context_false_strong_global": (
            float(pred_high[hard_negative].mean()) if int(hard_negative.sum()) else math.nan
        ),
        "low_context_false_strong_per_image_mean": finite_mean(false_rates),
        "low_context_false_strong_per_image_p90": finite_quantile(false_rates, 0.90),
        "low_context_false_strong_per_image_p95": finite_quantile(false_rates, 0.95),
        "hard_negative_top5_mean": hn_top5,
        "AUROC_high_vs_hard_negative": (
            v2b.auroc(high[positives_or_hn], pred_all[positives_or_hn]) if int(positives_or_hn.sum()) else math.nan
        ),
    }
    hist_rows = []
    for row in bins:
        hist_rows.append(row)
    return summary, per_rows, hist_rows


def candidate_gate(summary):
    return bool(
        summary["need_pearson"] >= GATE["pearson"]
        and summary["need_spearman"] >= GATE["spearman"]
        and summary["need_auroc_high_vs_low"] >= GATE["auroc"]
        and summary["need_auprc_high"] >= GATE["auprc"]
        and summary["need_monotonic_pairs"] == summary["need_monotonic_valid_pairs"] == 4
        and MIN_COVERAGE <= summary["need_pred_high_coverage"] <= MAX_COVERAGE
        and summary["low_context_false_strong_global"] <= GATE["false_strong_global"]
        and summary["low_context_false_strong_per_image_p90"] <= GATE["false_strong_p90"]
        and summary["target_high_recall"] >= GATE["high_need_recall"]
    )


def control_gate(summary):
    return candidate_gate(summary)


def density_only_control(model, density_head, val_names, data_dir, device, target_info, density_stats, output_dir, args):
    pack = collect_sample_records(
        model,
        load_need_head(Path(args.v2b_artifact_dir) / "d6c_ordinal_quantile_head.pt", device),
        density_head,
        val_names,
        data_dir,
        device,
        target_info,
        density_stats,
        args,
        limit=args.val_limit,
    )
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    records = []
    for rec in pack["records"]:
        records.append(
            {
                "name": rec["name"],
                "target": rec["target"],
                "pred": rec["density_pred"],
                "density": rec["density"],
            }
        )
    target = np.concatenate([r["target"] for r in records])
    pred = np.concatenate([r["pred"] for r in records])
    density = np.concatenate([r["density"] for r in records])
    summary, per, hist = summarize_prediction_records(
        "d7s2_density_only_control", records, target, pred, density, q20, q33, q66, q80, density_q33
    )
    write_csv(output_dir / "density_only_control_summary.csv", [summary])
    return summary, per, hist


def write_run_documents(output_dir, args, selected_op, threshold_rows, summaries, train_rows, per_rows, hist_rows, density_control):
    write_csv(output_dir / "need_ablation_summary.csv", summaries)
    write_csv(output_dir / "need_feature_probe_summary.csv", summaries)
    write_csv(output_dir / "need_per_image_safety_metrics.csv", per_rows)
    write_csv(output_dir / "need_prediction_histogram_by_variant.csv", hist_rows)
    shuffled_rows = [s for s in summaries if s["variant"].startswith("d7s")]
    write_csv(output_dir / "shuffled_control_summary.csv", shuffled_rows)
    write_csv(output_dir / "need_repair_train_log_v2d.csv", train_rows)
    control_pass = any(control_gate(s) for s in shuffled_rows) or control_gate(density_control)
    candidate_pass = [s for s in summaries if not s["variant"].startswith("d7s") and candidate_gate(s)]
    if control_pass:
        decision = "PAUSE_V2D_CONTROL_INVALID"
        next_step = "Do not proceed; a shuffled or density-only control passed the full R_need gate."
    elif candidate_pass:
        decision = "COMPLETED_V2D_NEED_SPATIAL_SAFETY_PASS_PAUSE_BEFORE_V3_NOOP_AUDIT"
        next_step = "Only a v3 no-op RARM audit may be considered next; do not train or connect RARM directly."
    else:
        d7b = next((s for s in summaries if s["variant"] == "d7b_topk_hn_ordinal"), None)
        if d7b and d7b["need_spearman"] >= GATE["spearman"] and d7b["low_context_false_strong_global"] <= 0.12:
            decision = "PAUSE_V2D_D7B_NEAR_MISS_NEXT_D7C_MULTI_CONTEXT"
            next_step = "Run frozen multi-context need head before any D2/RARM/v3 step."
        else:
            decision = "PAUSE_V2D_D7A_D7B_NOT_ENOUGH_NEXT_D7C_OR_TARGET_AUDIT"
            next_step = "Do not run D2/RARM; inspect whether feature context or R_need target definition is limiting."
    a0_audit = {
        "status": "PASS_BY_CONSTRUCTION",
        "a0_output_changed": False,
        "max_abs_diff": 0.0,
        "reason": "ConvIR-B is frozen; v2d heads are trained/evaluated only as side heads and are not connected to the dehazing output.",
        "locked_haze4k_test_usage": "none",
    }
    (output_dir / "a0_equivalence_audit.json").write_text(json.dumps(a0_audit, indent=2), encoding="utf-8")
    run_summary = {
        "decision": {
            "decision": decision,
            "candidate_pass": [
                {"variant": s["variant"], "coverage": s["need_pred_high_coverage"]} for s in candidate_pass
            ],
            "control_pass": control_pass,
            "next_step": next_step,
        },
        "d7_0_selected_operating_point": selected_op,
        "summaries": summaries,
        "density_only_control": density_control,
        "args": vars(args),
        "policy": {
            "D2": "forbidden",
            "RARM": "forbidden",
            "v3": "forbidden unless v2d passes and only as no-op audit",
            "locked_haze4k_test": "not used",
        },
    }
    (output_dir / "v2d_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_result_summary(output_dir, run_summary, summaries, density_control)
    write_failure_atlas(output_dir, per_rows)
    write_route_design(output_dir)
    write_resource_summary(output_dir, args)
    (output_dir / "decision_record.md").write_text(
        "\n".join(
            [
                "# CHD-RM v2d Decision Record",
                "",
                f"Decision: `{decision}`",
                "",
                f"Next step: {next_step}",
                "",
                "Forbidden: D2, RARM connection/training, v3 expansion, locked Haze4K test.",
                "Locked Haze4K test usage: none.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# CHD-RM v2d Need Spatial Hard-Negative Evidence\n\n"
        f"Status: `{decision}`\n\n"
        "Start with `v2d_result_summary.md`, `decision_record.md`, `v2d_run_summary.json`, "
        "and `need_rank_safety_operating_points.md`.\n\n"
        "This route keeps ConvIR-B frozen, does not connect RARM, and does not use the locked Haze4K test.\n",
        encoding="utf-8",
    )
    return run_summary


def write_result_summary(output_dir, run_summary, summaries, density_control):
    lines = [
        "# CHD-RM v2d Need Spatial Hard-Negative Summary",
        "",
        f"Decision: `{run_summary['decision']['decision']}`",
        "",
        "| Variant | Pearson | Spearman | AUROC | AUPRC | Coverage | False global | False p90 | Recall | Mono | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summaries + [density_control]:
        gate = candidate_gate(s)
        lines.append(
            f"| {s['variant']} | {s['need_pearson']:.4f} | {s['need_spearman']:.4f} | "
            f"{s['need_auroc_high_vs_low']:.4f} | {s['need_auprc_high']:.4f} | "
            f"{s['need_pred_high_coverage']:.4f} | {s['low_context_false_strong_global']:.4f} | "
            f"{s['low_context_false_strong_per_image_p90']:.4f} | {s['target_high_recall']:.4f} | "
            f"{s['need_monotonic_pairs']}/{s['need_monotonic_valid_pairs']} | {'PASS' if gate else 'FAIL'} |"
        )
    lines += [
        "",
        "Gate: Pearson >= 0.33, Spearman >= 0.32, AUROC >= 0.70, AUPRC >= 0.42, monotonic 4/4, "
        "coverage in [0.20, 0.40], false global <= 0.10, false p90 <= 0.15, recall >= 0.25.",
        "",
        "Locked Haze4K test usage: none.",
        "D2/RARM/v3 expansion: forbidden in this stage.",
        "",
    ]
    (output_dir / "v2d_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_failure_atlas(output_dir, per_rows):
    rows = [
        r
        for r in per_rows
        if not math.isnan(r.get("low_context_false_strong_rate", math.nan))
        and not r["variant"].startswith("d7s")
    ]
    rows = sorted(rows, key=lambda r: r["low_context_false_strong_rate"], reverse=True)[:30]
    lines = [
        "# v2d Failure Atlas Text Summary",
        "",
        "Top low-context false-strong images among candidate variants.",
        "",
        "| variant | name | false_rate | pred_mean | pred_max | target_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['name']} | {r['low_context_false_strong_rate']:.4f} | "
            f"{r['need_pred_mean']:.4f} | {r['need_pred_max']:.4f} | {r['need_target_mean']:.4f} |"
        )
    (output_dir / "failure_atlas_text_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_design(output_dir):
    (output_dir / "v2d_route_design.md").write_text(
        "# CHD-RM v2d Route Design\n\n"
        "Objective: repair R_need spatial safety before any RARM or v3 work.\n\n"
        "- D7-0 audits frozen D6c raw threshold safety without training.\n"
        "- D7a keeps the D6c ordinal target and adds low-density/low-need hard-negative loss.\n"
        "- D7b adds top-k hard-negative tail pressure to D7a.\n"
        "- D7s shuffled and density-only controls must fail.\n"
        "- ConvIR-B/A0 remain frozen and disconnected from the side head.\n"
        "- Locked Haze4K test is not used.\n",
        encoding="utf-8",
    )
    (output_dir / "need_spatial_gate_definition.md").write_text(
        "# Need Spatial Gate Definition\n\n"
        "Hard negative = R_need target <= q33 and density target <= q33, with q33 derived from train_inner only.\n"
        "Strong prediction = R_need_pred >= 0.66 unless a D7-0 threshold row explicitly states otherwise.\n"
        "Pass requires ranking, coverage, safety, recall protection, controls, and A0 equivalence together.\n",
        encoding="utf-8",
    )
    (output_dir / "hard_negative_mask_definition.md").write_text(
        "# Hard-Negative Mask Definition\n\n"
        "`low_context_hard_negative = need_low AND density_low`.\n"
        "`need_low` is the quantile R_need target <= q33. `density_low` is the normalized density target <= q33.\n"
        "Both thresholds are train_inner-derived. Low density alone is not penalized.\n",
        encoding="utf-8",
    )


def write_resource_summary(output_dir, args):
    rows = [
        {"key": "data_dir", "value": args.data_dir},
        {"key": "checkpoint", "value": args.checkpoint},
        {"key": "split_json", "value": args.split_json},
        {"key": "v2_thresholds", "value": args.v2_thresholds},
        {"key": "v2b_thresholds", "value": args.v2b_thresholds},
        {"key": "v2b_artifact_dir", "value": args.v2b_artifact_dir},
        {"key": "density_artifact", "value": args.density_artifact},
        {"key": "output_dir", "value": args.output_dir},
        {"key": "locked_haze4k_test_usage", "value": "none"},
    ]
    write_csv(output_dir / "resource_summary.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--v2b_artifact_dir", required=True)
    ap.add_argument("--density_artifact", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ordinal_weight", type=float, default=0.25)
    ap.add_argument("--hn_weight", type=float, default=0.75)
    ap.add_argument("--pair_weight", type=float, default=0.20)
    ap.add_argument("--topk_weight", type=float, default=0.50)
    ap.add_argument("--pos_weight", type=float, default=0.35)
    ap.add_argument("--tv_weight", type=float, default=0.02)
    ap.add_argument("--tau_neg", type=float, default=0.50)
    ap.add_argument("--tau_pos", type=float, default=0.66)
    ap.add_argument("--pair_margin", type=float, default=0.15)
    ap.add_argument("--topk_fraction", type=float, default=0.05)
    ap.add_argument("--pair_sample", type=int, default=4096)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--init_from_v2b", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["d7a_hn_ordinal", "d7b_topk_hn_ordinal", "d7s_shuffled_topk"],
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.txt").write_text("RUNNING\n", encoding="utf-8")
    train_names, val_names, device, model, density_stats, target_info = load_runtime(args)
    d6c_head = load_need_head(Path(args.v2b_artifact_dir) / "d6c_ordinal_quantile_head.pt", device)
    d6c_head.eval()
    for param in d6c_head.parameters():
        param.requires_grad_(False)
    density_head = load_density_head(args.density_artifact, device)

    print("D7_0_COLLECT_TRAIN_BEGIN", flush=True)
    train_pack = collect_sample_records(
        model, d6c_head, None, train_names, args.data_dir, device, target_info, density_stats, args, limit=args.train_limit
    )
    print("D7_0_COLLECT_VAL_BEGIN", flush=True)
    val_pack = collect_sample_records(
        model, d6c_head, None, val_names, args.data_dir, device, target_info, density_stats, args, limit=args.val_limit
    )
    selected_op, threshold_rows = write_threshold_audit(output_dir, train_pack, val_pack, target_info, density_stats, args)
    density_control, density_per, density_hist = density_only_control(
        model, density_head, val_names, args.data_dir, device, target_info, density_stats, output_dir, args
    )

    train_rows = []
    summaries = []
    per_rows = []
    hist_rows = []
    if not args.skip_train:
        for variant in args.variants:
            print(f"TRAIN_{variant}_BEGIN", flush=True)
            head, rows = train_variant(model, train_names, args.data_dir, device, target_info, density_stats, variant, output_dir, args)
            train_rows.extend(rows)
            print(f"EVAL_{variant}_BEGIN", flush=True)
            summary, per, hist = evaluate_need_head(
                model, head, val_names, args.data_dir, device, target_info, density_stats, variant, output_dir, args
            )
            summaries.append(summary)
            per_rows.extend(per)
            hist_rows.extend(hist)
            print(json.dumps(summary), flush=True)
    per_rows.extend(density_per)
    hist_rows.extend(density_hist)
    run_summary = write_run_documents(
        output_dir, args, selected_op, threshold_rows, summaries, train_rows, per_rows, hist_rows, density_control
    )
    (output_dir / "thresholds_train_inner_v2d.json").write_text(
        json.dumps(
            {
                "need_q33": target_info["quantile"]["q33"],
                "need_q66": target_info["quantile"]["q66"],
                "density_q33": density_stats["density"]["q33"],
                "tau_neg": args.tau_neg,
                "strong_pred_threshold": STRONG_PRED_THRESHOLD,
                "source": "train_inner only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "status.txt").write_text(f"{run_summary['decision']['decision']}\n", encoding="utf-8")
    print(json.dumps({"decision": run_summary["decision"], "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
