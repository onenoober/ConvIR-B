import argparse
import importlib.util
import json
import math
import os
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

V2D_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2d_need_spatial_hard_negative.py"
spec = importlib.util.spec_from_file_location("chdrm_v2d_tool", V2D_TOOL)
v2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2d)


class MultiContextNeedHead(nn.Module):
    def __init__(self, in_channels=234, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


def convir_a0_context(model, density_head, x):
    x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x_4 = F.interpolate(x_2, scale_factor=0.5, mode="bilinear", align_corners=False)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x0 = model.feat_extract[0](x)
    res1 = model.Encoder[0](x0)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    bottleneck = model.Encoder[2](z)

    z = model.Decoder[0](bottleneck)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z = model.feat_extract[4](z)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z = model.feat_extract[5](z)
    a0 = torch.clamp(z + x, 0, 1)

    res2_up = F.interpolate(res2, size=res1.shape[-2:], mode="bilinear", align_corners=False)
    bottleneck_up = F.interpolate(bottleneck, size=res1.shape[-2:], mode="bilinear", align_corners=False)
    density_pred = v2d.density_pred_from_head(density_head, res1)
    context = torch.cat([res1, res2_up, bottleneck_up, x, a0, (x - a0).abs(), density_pred], dim=1)
    return a0, context


def predict_head(head, context):
    logits = head(context)
    probs = [torch.sigmoid(logits[:, i : i + 1]) for i in range(4)]
    pred = torch.stack(probs, dim=0).mean(dim=0)
    return pred, logits


def shuffle_target(target):
    if target.shape[0] > 1:
        return torch.roll(target, shifts=1, dims=0)
    return torch.flip(target, dims=[-1])


def train_variant(model, density_head, train_names, device, target_info, density_stats, args, variant):
    head = MultiContextNeedHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = v2d.v2b.Haze4KPairDataset(
        train_names, args.data_dir, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=v2d.v2b.collate_pairs,
    )
    bce = nn.BCEWithLogitsLoss()
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    thresholds = [0.20, 0.33, 0.66, 0.80]
    rows = []
    head.train()
    model.eval()
    density_head.eval()
    for epoch in range(1, args.epochs + 1):
        losses = []
        smooth_losses = []
        hn_losses = []
        pos_losses = []
        topk_losses = []
        start = time.time()
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, context = convir_a0_context(model, density_head, hazy)
                raw_need = v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
                target = v2d.v2b.make_target(raw_need, target_info, "quantile")
                density_target = v2d.v2b.v2.normalize(
                    v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                )
                if variant.startswith("d7c_random"):
                    target = torch.rand_like(target)
                elif variant.startswith("d7c_shuffled"):
                    target = shuffle_target(target)
            pred, logits = predict_head(head, context.detach())
            ord_loss = logits.new_tensor(0.0)
            for i, thr in enumerate(thresholds):
                ord_loss = ord_loss + bce(logits[:, i : i + 1], (target >= thr).float())
            smooth = F.smooth_l1_loss(pred, target)
            hn_mask = (target <= q33) & (density_target <= density_q33)
            pos_mask = target >= q66
            hn_loss = v2d.hard_negative_loss(pred, hn_mask, args.tau_neg)
            pos_loss = v2d.positive_response_loss(pred, pos_mask, args.tau_pos)
            pair_loss = v2d.pairwise_loss(pred, pos_mask, hn_mask, args.pair_margin, args.pair_sample)
            topk_loss = pred.new_tensor(0.0)
            if "topk" in variant:
                topk_loss = v2d.topk_hn_loss(pred, hn_mask, args.tau_neg, args.topk_fraction)
            loss = (
                smooth
                + args.ordinal_weight * ord_loss
                + args.hn_weight * hn_loss
                + args.pos_weight * pos_loss
                + args.pair_weight * pair_loss
                + args.topk_weight * topk_loss
                + args.tv_weight * v2d.v2b.tv_loss(pred)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
            smooth_losses.append(float(smooth.detach().item()))
            hn_losses.append(float(hn_loss.detach().item()))
            pos_losses.append(float(pos_loss.detach().item()))
            topk_losses.append(float(topk_loss.detach().item()))
            if step % args.progress_every == 0:
                print(f"{variant} epoch={epoch} step={step}/{len(loader)} loss={statistics.mean(losses[-args.progress_every:]):.6f}", flush=True)
        row = {
            "variant": variant,
            "epoch": epoch,
            "steps": len(loader),
            "loss_mean": statistics.mean(losses),
            "smooth_l1_mean": statistics.mean(smooth_losses),
            "hard_negative_loss_mean": statistics.mean(hn_losses),
            "positive_response_loss_mean": statistics.mean(pos_losses),
            "topk_loss_mean": statistics.mean(topk_losses),
            "elapsed_sec": time.time() - start,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return head, rows


def collect_pack(model, density_head, head, names, device, target_info, density_stats, args, limit):
    dataset = v2d.v2b.Haze4KPairDataset(names, args.data_dir, max_items=limit, seed=args.seed)
    records = []
    targets = []
    preds = []
    densities = []
    head.eval()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2d.v2b.v2.pad32(hazy)
            a0, context = convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            context = context[:, :, :h, :w]
            raw_need = v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2d.v2b.make_target(raw_need, target_info, "quantile")
            density_target = v2d.v2b.v2.normalize(
                v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            pred, _ = predict_head(head, context)
            target_np = v2d.sample_np(target, args.metric_sample_size)
            pred_np = v2d.sample_np(pred, args.metric_sample_size)
            density_np = v2d.sample_np(density_target, args.metric_sample_size)
            records.append({"name": name, "target": target_np, "pred": pred_np, "density": density_np})
            targets.append(target_np)
            preds.append(pred_np)
            densities.append(density_np)
            if (idx + 1) % args.progress_every == 0:
                print(f"collect {idx + 1}/{len(dataset)}", flush=True)
    return {
        "records": records,
        "target": np.concatenate(targets).astype(np.float32, copy=False),
        "pred": np.concatenate(preds).astype(np.float32, copy=False),
        "density": np.concatenate(densities).astype(np.float32, copy=False),
    }


def threshold_sweep(output_dir, variant, train_pack, val_pack, target_info, density_stats, args):
    q20 = float(target_info["quantile"]["q20"])
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    q80 = float(target_info["quantile"]["q80"])
    density_q33 = float(density_stats["density"]["q33"])
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(train_pack["pred"], np.linspace(0.0, 1.0, args.threshold_grid)),
                np.linspace(0.05, 0.95, 19),
                np.asarray([v2d.STRONG_PRED_THRESHOLD], dtype=np.float32),
            ]
        )
    )
    high = val_pack["target"] >= q66
    low_or_high = (val_pack["target"] <= q33) | high
    hard_negative = v2d.hard_negative_mask(val_pack["target"], val_pack["density"], q33, density_q33)
    positives_or_hn = high | hard_negative
    hn_scores = val_pack["pred"][hard_negative]
    if hn_scores.size:
        k = max(1, int(math.ceil(hn_scores.size * 0.05)))
        hn_top5 = float(np.mean(np.sort(hn_scores)[-k:]))
    else:
        hn_top5 = math.nan
    base = {
        "AUPRC_high": v2d.v2b.auprc(high, val_pack["pred"]),
        "AUROC_high_vs_low": v2d.v2b.auroc(high[low_or_high], val_pack["pred"][low_or_high]),
        "AUROC_high_vs_hard_negative": (
            v2d.v2b.auroc(high[positives_or_hn], val_pack["pred"][positives_or_hn])
            if int(positives_or_hn.sum())
            else math.nan
        ),
        "hard_negative_top5_mean": hn_top5,
    }
    rows = [
        v2d.threshold_metrics(
            t,
            val_pack["target"],
            val_pack["pred"],
            val_pack["density"],
            val_pack["records"],
            q33,
            q66,
            density_q33,
            base,
        )
        for t in sorted(grid)
    ]
    v2d.write_csv(output_dir / f"need_rank_safety_curve_{variant}.csv", rows)
    safe_global = [
        row
        for row in rows
        if v2d.MIN_COVERAGE <= row["pred_high_coverage"] <= v2d.MAX_COVERAGE
        and row["low_context_false_strong_global"] <= v2d.GATE["false_strong_global"]
    ]
    safe_p90 = [
        row
        for row in safe_global
        if row["low_context_false_strong_per_image_p90"] <= v2d.GATE["false_strong_p90"]
    ]
    pool = safe_p90 or safe_global or rows
    selected = sorted(pool, key=lambda row: (0 if row in safe_p90 else 1, abs(row["pred_high_coverage"] - 0.30), row["low_context_false_strong_per_image_p90"]))[0]
    raw66 = min(rows, key=lambda row: abs(row["threshold"] - v2d.STRONG_PRED_THRESHOLD))
    bins, mono, valid = v2d.v2b.bin_rows(val_pack["target"], val_pack["pred"], [q20, q33, q66, q80], variant)
    summary = {
        "variant": variant,
        "need_pearson": v2d.v2b.pearson(val_pack["pred"], val_pack["target"]),
        "need_spearman": v2d.v2b.spearman(val_pack["pred"], val_pack["target"]),
        "need_auroc_high_vs_low": base["AUROC_high_vs_low"],
        "need_auprc_high": base["AUPRC_high"],
        "need_monotonic_pairs": mono,
        "need_monotonic_valid_pairs": valid,
        "safe_global_points": len(safe_global),
        "safe_p90_points": len(safe_p90),
        "selected_threshold": selected["threshold"],
        "selected_coverage": selected["pred_high_coverage"],
        "selected_recall": selected["target_high_recall"],
        "selected_precision": selected["target_high_precision"],
        "selected_false_global": selected["low_context_false_strong_global"],
        "selected_false_p90": selected["low_context_false_strong_per_image_p90"],
        "selected_false_p95": selected["low_context_false_strong_per_image_p95"],
        "selected_low_density_high_recall": selected["high_need_recall_in_low_density_regions"],
        "raw66_coverage": raw66["pred_high_coverage"],
        "raw66_false_global": raw66["low_context_false_strong_global"],
        "hard_negative_top5_mean": base["hard_negative_top5_mean"],
    }
    return summary, bins


def write_summary(output_dir, summaries, train_rows):
    v2d.write_csv(output_dir / "d7c_multicontext_train_log.csv", train_rows)
    v2d.write_csv(output_dir / "d7c_multicontext_safety_summary.csv", summaries)
    lines = [
        "# CHD-RM v2d D7c Multi-Context Safety Summary",
        "",
        "| Variant | Spearman | AUROC | AUPRC | Mono | safe_global | safe_p90 | coverage | recall | false_global | false_p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['need_spearman']:.4f} | {row['need_auroc_high_vs_low']:.4f} | "
            f"{row['need_auprc_high']:.4f} | {row['need_monotonic_pairs']}/{row['need_monotonic_valid_pairs']} | "
            f"{row['safe_global_points']} | {row['safe_p90_points']} | {row['selected_coverage']:.4f} | "
            f"{row['selected_recall']:.4f} | {row['selected_false_global']:.4f} | {row['selected_false_p90']:.4f} |"
        )
    lines += ["", "Locked Haze4K test usage: none.", "D2/RARM/v3 remain forbidden.", ""]
    (output_dir / "d7c_multicontext_safety_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--density_artifact", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ordinal_weight", type=float, default=0.25)
    ap.add_argument("--hn_weight", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--pair_weight", type=float, default=0.2)
    ap.add_argument("--topk_weight", type=float, default=1.0)
    ap.add_argument("--tv_weight", type=float, default=0.02)
    ap.add_argument("--tau_neg", type=float, default=0.50)
    ap.add_argument("--tau_pos", type=float, default=0.66)
    ap.add_argument("--pair_margin", type=float, default=0.15)
    ap.add_argument("--pair_sample", type=int, default=4096)
    ap.add_argument("--topk_fraction", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["d7c_mc_hn_ordinal", "d7c_mc_topk_hn_ordinal", "d7c_shuffled_mc_topk"],
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v2d.v2b.set_seed(args.seed)
    train_names, val_names, device, model, density_stats, target_info = v2d.load_runtime(args)
    density_head = v2d.load_density_head(args.density_artifact, device)
    train_rows = []
    summaries = []
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    for variant in args.variants:
        print(f"TRAIN_{variant}_BEGIN", flush=True)
        head, rows = train_variant(model, density_head, train_names, device, target_info, density_stats, args, variant)
        train_rows.extend(rows)
        torch.save({"variant": variant, "state_dict": head.state_dict(), "args": vars(args)}, artifact_dir / f"{variant}_head.pt")
        print(f"AUDIT_{variant}_TRAIN_PACK", flush=True)
        train_pack = collect_pack(model, density_head, head, train_names, device, target_info, density_stats, args, args.train_limit)
        print(f"AUDIT_{variant}_VAL_PACK", flush=True)
        val_pack = collect_pack(model, density_head, head, val_names, device, target_info, density_stats, args, args.val_limit)
        summary, _ = threshold_sweep(args.output_dir, variant, train_pack, val_pack, target_info, density_stats, args)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_summary(args.output_dir, summaries, train_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
