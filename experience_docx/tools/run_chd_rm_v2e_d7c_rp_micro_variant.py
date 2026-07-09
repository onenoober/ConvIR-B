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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
for path in (str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2E_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2e_d7c_control_recall_audit.py"
spec_v2e = importlib.util.spec_from_file_location("chdrm_v2e_tool", V2E_TOOL)
v2e = importlib.util.module_from_spec(spec_v2e)
spec_v2e.loader.exec_module(v2e)


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


def set_trainable(module):
    module.train()
    for param in module.parameters():
        param.requires_grad_(True)


def load_initialized_rp_head(path, device):
    head = v2e.d7c.MultiContextNeedHead().to(device)
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    set_trainable(head)
    return head


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    k = int(radius) * 2 + 1
    return F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=int(radius)) > 0


class RecallProtectedCropDataset(Dataset):
    def __init__(self, names, data_dir, crop_size=256, max_items=0, seed=3407):
        names = list(sorted(names))
        if max_items is not None and max_items > 0 and max_items < len(names):
            rng = random.Random(seed)
            names = sorted(rng.sample(names, max_items))
        self.names = names
        self.haze_dir = Path(data_dir) / "train" / "haze"
        self.gt_dir = Path(data_dir) / "train" / "gt"
        self.crop_size = crop_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        hazy = v2e.load_tensor(self.haze_dir / name)
        gt = v2e.load_tensor(v2e.label_path(self.gt_dir, name))
        _, h, w = hazy.shape
        if h < self.crop_size or w < self.crop_size:
            raise ValueError(f"crop_size {self.crop_size} exceeds image shape {(h, w)} for {name}")
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        hazy = hazy[:, top : top + self.crop_size, left : left + self.crop_size]
        gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]
        return name, hazy, gt


def collate_rp(batch):
    names, hazy, gt = zip(*batch)
    return list(names), torch.stack(hazy, 0), torch.stack(gt, 0)


def compute_ldhn_support_weights(model, density_head, train_names, device, target_info, density_stats, args):
    dataset = v2e.Haze4KPairDataset(train_names, args.data_dir, max_items=args.train_limit, seed=args.seed)
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    rows = []
    model.eval()
    density_head.eval()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2e.v2d.v2b.v2.pad32(hazy)
            a0, _ = v2e.d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            raw_need = v2e.v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2e.v2d.v2b.make_target(raw_need, target_info, "quantile")
            density_target = v2e.v2d.v2b.v2.normalize(
                v2e.v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            ldhn = (target >= q66) & (density_target <= density_q33)
            rows.append(
                {
                    "name": name,
                    "ldhn_coverage": float(ldhn.float().mean().item()),
                    "target_mean": float(target.mean().item()),
                    "density_mean": float(density_target.mean().item()),
                }
            )
            if (idx + 1) % args.progress_every == 0:
                print(f"support_train {idx + 1}/{len(dataset)}", flush=True)
    coverages = [row["ldhn_coverage"] for row in rows]
    cutoff = v2e.finite_quantile(coverages, args.oversample_quantile)
    weights_by_name = {}
    for row in rows:
        weights_by_name[row["name"]] = args.oversample_weight if row["ldhn_coverage"] >= cutoff else 1.0
    summary = {
        "train_images": len(rows),
        "oversample_quantile": args.oversample_quantile,
        "oversample_cutoff": cutoff,
        "oversample_weight": args.oversample_weight,
        "images_ge_cutoff": int(sum(row["ldhn_coverage"] >= cutoff for row in rows)),
        "ldhn_p50": v2e.finite_quantile(coverages, 0.50),
        "ldhn_p75": v2e.finite_quantile(coverages, 0.75),
        "ldhn_p90": v2e.finite_quantile(coverages, 0.90),
        "ldhn_p95": v2e.finite_quantile(coverages, 0.95),
        "ldhn_max": max(coverages) if coverages else math.nan,
    }
    return rows, weights_by_name, summary


def make_loader(train_names, weights_by_name, args):
    dataset = RecallProtectedCropDataset(
        train_names, args.data_dir, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
    )
    weights = [float(weights_by_name.get(name, 1.0)) for name in dataset.names]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_rp,
    )


def train_rp_variant(model, density_head, train_names, weights_by_name, device, target_info, density_stats, args, spec):
    head = load_initialized_rp_head(args.d7c_topk_artifact, device)
    opt = torch.optim.AdamW(head.parameters(), lr=spec["lr"], weight_decay=args.weight_decay)
    loader = make_loader(train_names, weights_by_name, args)
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    thresholds = [0.20, 0.33, 0.66, 0.80]
    bce = nn.BCEWithLogitsLoss()
    rows = []
    model.eval()
    density_head.eval()
    for epoch in range(1, args.epochs + 1):
        losses = []
        ord_losses = []
        smooth_losses = []
        hn_losses = []
        ldhn_losses = []
        pair_losses = []
        topk_losses = []
        tv_losses = []
        start = time.time()
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, context = v2e.d7c.convir_a0_context(model, density_head, hazy)
                raw_need = v2e.v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
                target = v2e.v2d.v2b.make_target(raw_need, target_info, "quantile")
                density_target = v2e.v2d.v2b.v2.normalize(
                    v2e.v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                )
                need_high = target >= q66
                density_low = density_target <= density_q33
                ldhn_pos = need_high & density_low
                protected = dilate_mask(need_high, spec["protect_radius"])
                hard_negative = (target <= q33) & density_low & (~protected)
            pred, logits = v2e.d7c.predict_head(head, context.detach())
            ord_loss = logits.new_tensor(0.0)
            for i, thr in enumerate(thresholds):
                ord_loss = ord_loss + bce(logits[:, i : i + 1], (target >= thr).float())
            smooth = F.smooth_l1_loss(pred, target)
            hn_loss = v2e.v2d.hard_negative_loss(pred, hard_negative, args.tau_neg)
            ldhn_loss = v2e.v2d.positive_response_loss(pred, ldhn_pos, args.tau_pos)
            pair_loss = v2e.v2d.pairwise_loss(pred, need_high, hard_negative, args.pair_margin, args.pair_sample)
            topk_loss = v2e.v2d.topk_hn_loss(pred, hard_negative, args.tau_neg, spec["topk_fraction"])
            tv_loss = v2e.v2d.v2b.tv_loss(pred)
            loss = (
                smooth
                + spec["ordinal_weight"] * ord_loss
                + spec["hn_weight"] * hn_loss
                + spec["ldhn_weight"] * ldhn_loss
                + spec["pair_weight"] * pair_loss
                + spec["topk_weight"] * topk_loss
                + spec["tv_weight"] * tv_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
            ord_losses.append(float(ord_loss.detach().item()))
            smooth_losses.append(float(smooth.detach().item()))
            hn_losses.append(float(hn_loss.detach().item()))
            ldhn_losses.append(float(ldhn_loss.detach().item()))
            pair_losses.append(float(pair_loss.detach().item()))
            topk_losses.append(float(topk_loss.detach().item()))
            tv_losses.append(float(tv_loss.detach().item()))
            if step % args.progress_every == 0:
                print(
                    f"train_{spec['name']} epoch={epoch} step={step}/{len(loader)} "
                    f"loss={statistics.mean(losses[-args.progress_every:]):.6f}",
                    flush=True,
                )
        row = {
            "variant": spec["name"],
            "epoch": epoch,
            "steps": len(loader),
            "loss_mean": statistics.mean(losses),
            "smooth_l1_mean": statistics.mean(smooth_losses),
            "ordinal_bce_sum_mean": statistics.mean(ord_losses),
            "hard_negative_loss_mean": statistics.mean(hn_losses),
            "ldhn_positive_loss_mean": statistics.mean(ldhn_losses),
            "pairwise_loss_mean": statistics.mean(pair_losses),
            "topk_loss_mean": statistics.mean(topk_losses),
            "tv_loss_mean": statistics.mean(tv_losses),
            "elapsed_sec": time.time() - start,
            **{f"spec_{k}": v for k, v in spec.items() if k != "name"},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return head, rows


def parse_variant_specs(spec_text):
    specs = []
    for item in spec_text:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"Bad --rp_specs item {item!r}; expected name:ldhn_weight:protect_radius:lr")
        name, ldhn_weight, protect_radius, lr = parts
        specs.append(
            {
                "name": name,
                "ldhn_weight": float(ldhn_weight),
                "protect_radius": int(protect_radius),
                "lr": float(lr),
                "ordinal_weight": 0.25,
                "hn_weight": 1.0,
                "pair_weight": 0.2,
                "topk_weight": 1.0,
                "tv_weight": 0.02,
                "topk_fraction": 0.05,
            }
        )
    return specs


def select_summary_for_variant(variant, train_pack, val_pack, target_info, density_stats, args):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    val_target, val_pred, val_density = v2e.arrays_for_variant(val_pack, variant)
    train_target, train_pred, train_density = v2e.arrays_for_variant(train_pack, variant)
    del train_target, train_density
    threshold = v2e.matched_threshold(train_pred, args.target_coverage)
    summary = v2e.fixed_threshold_summary(
        variant,
        val_target,
        val_pred,
        val_density,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        threshold,
    )
    summary["selected_threshold"] = threshold
    summary["threshold_selection"] = "train_inner_matched_target_coverage"
    curve = v2e.build_curve(
        variant,
        train_pred,
        val_target,
        val_pred,
        val_density,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        args,
    )
    return summary, curve


def fixed_threshold_summary_for_variant(variant, pack, target_info, density_stats, threshold):
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    val_target, val_pred, val_density = v2e.arrays_for_variant(pack, variant)
    return v2e.fixed_threshold_summary(
        variant,
        val_target,
        val_pred,
        val_density,
        pack["records"],
        q33,
        q66,
        density_q33,
        threshold,
    )


def decision_from_rows(rows, baseline):
    candidates = []
    for row in rows:
        safety = (
            row["false_global"] <= v2e.SAFETY_FALSE_GLOBAL
            and row["false_per_image_p90"] <= v2e.SAFETY_FALSE_P90
            and row["false_per_image_p95"] <= v2e.SAFETY_FALSE_P95
        )
        coverage_ok = 0.20 <= row["coverage"] <= 0.40
        ldhn_ok = row["ldhn_recall"] >= 0.10
        ranking_ok = (
            row["spearman"] >= baseline["spearman"] - 0.04
            and row["auroc"] >= baseline["auroc"] - 0.03
            and row["auprc"] >= baseline["auprc"] - 0.04
            and row["precision"] >= 0.58
            and row["recall"] >= 0.40
            and row["monotonic_pairs"] == 4
            and row["monotonic_valid_pairs"] == 4
        )
        row["gate_safety_pass"] = bool(safety)
        row["gate_coverage_pass"] = bool(coverage_ok)
        row["gate_ldhn_pass"] = bool(ldhn_ok)
        row["gate_ranking_pass"] = bool(ranking_ok)
        row["gate_overall_pass"] = bool(safety and coverage_ok and ldhn_ok and ranking_ok)
        if row["gate_overall_pass"]:
            candidates.append(row)
    if candidates:
        best = sorted(candidates, key=lambda r: (-r["ldhn_recall"], r["false_per_image_p95"], -r["spearman"]))[0]
        return "COMPLETED_V2E_D7C_RP_PASS_AUTHORIZE_V3_NOOP_GATE_ONLY", best
    best = sorted(rows, key=lambda r: (-r["ldhn_recall"], r["false_per_image_p95"], -r["spearman"]))[0] if rows else None
    return "PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3", best


def write_docs(out_dir, decision, best, baseline, rows, support_summary):
    best_text = best["variant"] if best is not None else "none"
    lines = [
        "# CHD-RM v2e D7c-RP Micro-Variant Summary",
        "",
        f"Decision: `{decision}`",
        "",
        f"Best RP variant by gate ordering: `{best_text}`.",
        "",
        "Frozen and forbidden: ConvIR-B frozen, D3 density frozen, no D2, no RARM connection/training, no v3 runtime, no locked Haze4K test.",
        "",
        "## Baseline",
        "",
        (
            f"D7c top-k baseline Spearman `{baseline['spearman']:.4f}`, AUROC `{baseline['auroc']:.4f}`, "
            f"AUPRC `{baseline['auprc']:.4f}`, coverage `{baseline['coverage']:.4f}`, "
            f"false-p90 `{baseline['false_per_image_p90']:.4f}`, false-p95 `{baseline['false_per_image_p95']:.4f}`, "
            f"LDHN recall `{baseline['ldhn_recall']:.4f}`."
        ),
        "",
        "## RP Variants",
        "",
        "| variant | pass | Spearman | AUROC | AUPRC | coverage | recall | precision | false_global | false_p90 | false_p95 | LDHN recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['gate_overall_pass']} | {row['spearman']:.4f} | {row['auroc']:.4f} | "
            f"{row['auprc']:.4f} | {row['coverage']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} | "
            f"{row['false_global']:.4f} | {row['false_per_image_p90']:.4f} | {row['false_per_image_p95']:.4f} | "
            f"{row['ldhn_recall']:.4f} |"
        )
    lines += [
        "",
        "## Train LDHN Oversampling Support",
        "",
        f"Train LDHN p75 cutoff: `{support_summary['oversample_cutoff']:.6f}`.",
        f"Images at or above cutoff: `{support_summary['images_ge_cutoff']}` / `{support_summary['train_images']}`.",
        "",
    ]
    (out_dir / "d7c_rp_safety_recall_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "decision_record.md").write_text(
        f"# CHD-RM v2e D7c-RP Decision Record\n\nDecision: `{decision}`\n\n"
        "Locked Haze4K test usage: none.\nD2/RARM/v3 runtime: not run.\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# CHD-RM v2e D7c-RP Micro-Variant Evidence\n\n"
        f"Status: `{decision}`\n\n"
        "Primary files: `d7c_rp_safety_recall_summary.md`, `d7c_rp_ablation_summary.csv`, "
        "`d7c_rp_run_summary.json`, and `d7c_rp_train_log.csv`.\n\n"
        "This is a gated v2e follow-up after controls passed but LDHN recall failed. It does not use D2, RARM, v3 runtime, or locked test data.\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v2b_thresholds", required=True)
    ap.add_argument("--density_artifact", required=True)
    ap.add_argument("--d7c_topk_artifact", required=True)
    ap.add_argument("--d7c_hn_artifact", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau_neg", type=float, default=0.50)
    ap.add_argument("--tau_pos", type=float, default=0.66)
    ap.add_argument("--pair_margin", type=float, default=0.15)
    ap.add_argument("--pair_sample", type=int, default=4096)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--target_coverage", type=float, default=v2e.TARGET_COVERAGE)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--oversample_quantile", type=float, default=0.75)
    ap.add_argument("--oversample_weight", type=float, default=3.0)
    ap.add_argument(
        "--rp_specs",
        nargs="+",
        default=[
            "d7c_rp_lam05_r3:0.5:3:0.0002",
            "d7c_rp_lam10_r3:1.0:3:0.0002",
            "d7c_rp_lam20_r5:2.0:5:0.00015",
        ],
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.txt").write_text("RUNNING_D7C_RP\n", encoding="utf-8")
    v2e.v2d.v2b.set_seed(args.seed)
    specs = parse_variant_specs(args.rp_specs)

    train_names, val_names, device, model, density_stats, target_info = v2e.v2d.load_runtime(args)
    density_head = v2e.v2d.load_density_head(args.density_artifact, device)
    topk_head = v2e.load_head(args.d7c_topk_artifact, device)
    hn_head = v2e.load_head(args.d7c_hn_artifact, device)
    v2e.set_frozen(model)
    v2e.set_frozen(density_head)
    v2e.set_frozen(topk_head)
    v2e.set_frozen(hn_head)

    support_rows, weights_by_name, support_summary = compute_ldhn_support_weights(
        model, density_head, train_names, device, target_info, density_stats, args
    )
    write_csv(args.output_dir / "d7c_rp_train_ldhn_oversample_support.csv", support_rows)
    (args.output_dir / "d7c_rp_train_ldhn_oversample_summary.json").write_text(
        json.dumps(support_summary, indent=2), encoding="utf-8"
    )

    rp_heads = {}
    train_rows = []
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    for spec in specs:
        print(f"TRAIN_RP_{spec['name']}_BEGIN", flush=True)
        head, rows = train_rp_variant(model, density_head, train_names, weights_by_name, device, target_info, density_stats, args, spec)
        v2e.set_frozen(head)
        rp_heads[spec["name"]] = head
        train_rows.extend(rows)
        torch.save({"variant": spec["name"], "state_dict": head.state_dict(), "args": vars(args), "spec": spec}, artifact_dir / f"{spec['name']}_head.pt")
    write_csv(args.output_dir / "d7c_rp_train_log.csv", train_rows)

    heads = {
        "d7c_mc_topk_hn_ordinal": topk_head,
        "d7c_mc_hn_ordinal": hn_head,
    }
    heads.update(rp_heads)
    print("COLLECT_RP_TRAIN_BEGIN", flush=True)
    train_pack = v2e.collect_multi_pack(model, density_head, heads, train_names, device, target_info, density_stats, args, "rp_train_inner", args.train_limit)
    print("COLLECT_RP_VAL_BEGIN", flush=True)
    val_pack = v2e.collect_multi_pack(model, density_head, heads, val_names, device, target_info, density_stats, args, "rp_val_inner", args.val_limit)

    rows = []
    curves = []
    baseline, baseline_curve = select_summary_for_variant(
        "d7c_mc_topk_hn_ordinal", train_pack, val_pack, target_info, density_stats, args
    )
    baseline["variant"] = "d7c_mc_topk_hn_ordinal_baseline_matched_coverage"
    frozen_baseline = fixed_threshold_summary_for_variant(
        "d7c_mc_topk_hn_ordinal", val_pack, target_info, density_stats, v2e.DEFAULT_THRESHOLD
    )
    frozen_baseline["variant"] = "d7c_mc_topk_hn_ordinal_frozen_v2d_threshold"
    frozen_baseline["threshold_selection"] = "frozen_v2d_threshold"
    rows.append(frozen_baseline)
    rows.append(baseline)
    curves.extend(baseline_curve)
    hn_summary, hn_curve = select_summary_for_variant("d7c_mc_hn_ordinal", train_pack, val_pack, target_info, density_stats, args)
    hn_summary["variant"] = "d7c_mc_hn_ordinal_baseline"
    rows.append(hn_summary)
    curves.extend(hn_curve)
    rp_rows = []
    for spec in specs:
        summary, curve = select_summary_for_variant(spec["name"], train_pack, val_pack, target_info, density_stats, args)
        rp_rows.append(summary)
        rows.append(summary)
        curves.extend(curve)
    decision, best = decision_from_rows(rp_rows, baseline)

    write_csv(args.output_dir / "d7c_rp_ablation_summary.csv", rows)
    write_csv(args.output_dir / "d7c_rp_recall_vs_false_curve.csv", curves)
    no_test = {
        "locked_haze4k_test_usage": "none",
        "forbidden": ["D2", "RARM connection", "RARM training", "v3 runtime", "locked Haze4K test"],
        "convird_frozen": True,
        "density_head_frozen": True,
        "init": "d7c top-k head",
    }
    run_summary = {
        "decision": decision,
        "best_variant": best,
        "baseline": baseline,
        "d7c_hn_baseline": hn_summary,
        "rp_rows": rp_rows,
        "train_ldhn_oversample_summary": support_summary,
        "rp_specs": specs,
        "policy": no_test,
    }
    (args.output_dir / "no_locked_test_audit.json").write_text(json.dumps(no_test, indent=2), encoding="utf-8")
    (args.output_dir / "d7c_rp_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_docs(args.output_dir, decision, best, baseline, rp_rows, support_summary)
    (args.output_dir / "status.txt").write_text(f"{decision}\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "best_variant": best["variant"] if best else None, "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
