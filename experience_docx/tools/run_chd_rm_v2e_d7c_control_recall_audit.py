import argparse
import csv
import hashlib
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
for path in (str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

V2D_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2d_need_spatial_hard_negative.py"
V2D_D7C_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2d_d7c_multicontext.py"

spec_v2d = importlib.util.spec_from_file_location("chdrm_v2d_tool", V2D_TOOL)
v2d = importlib.util.module_from_spec(spec_v2d)
spec_v2d.loader.exec_module(v2d)

spec_d7c = importlib.util.spec_from_file_location("chdrm_v2d_d7c_tool", V2D_D7C_TOOL)
d7c = importlib.util.module_from_spec(spec_d7c)
spec_d7c.loader.exec_module(d7c)

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
DEFAULT_THRESHOLD = 0.5773006677627563
TARGET_COVERAGE = 0.3026953125
SAFETY_FALSE_GLOBAL = 0.01
SAFETY_FALSE_P90 = 0.05
SAFETY_FALSE_P95 = 0.10


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


def sha_array(values):
    arr = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def label_path(gt_dir, hazy_name):
    stem = Path(hazy_name).stem
    ext = Path(hazy_name).suffix
    candidates = [hazy_name]
    if "_" in stem:
        image_id = stem.split("_")[0]
        candidates.extend([f"{image_id}{ext}", f"{image_id}.png"])
    for name in candidates:
        path = gt_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT for {hazy_name}; tried {candidates}")


def load_tensor(path):
    return TF.to_tensor(Image.open(path).convert("RGB"))


class Haze4KPairDataset(Dataset):
    def __init__(self, names, data_dir, max_items=None, seed=3407):
        names = list(sorted(names))
        if max_items is not None and max_items > 0 and max_items < len(names):
            rng = random.Random(seed)
            names = sorted(rng.sample(names, max_items))
        self.names = names
        self.haze_dir = Path(data_dir) / "train" / "haze"
        self.gt_dir = Path(data_dir) / "train" / "gt"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        hazy = load_tensor(self.haze_dir / name)
        gt = load_tensor(label_path(self.gt_dir, name))
        return name, hazy, gt


class PermutedCropDataset(Dataset):
    def __init__(self, names, data_dir, perm_map, crop_size=256, max_items=None, seed=3407):
        names = list(sorted(names))
        if max_items is not None and max_items > 0 and max_items < len(names):
            rng = random.Random(seed)
            names = sorted(rng.sample(names, max_items))
        self.names = names
        self.perm_map = dict(perm_map)
        self.haze_dir = Path(data_dir) / "train" / "haze"
        self.gt_dir = Path(data_dir) / "train" / "gt"
        self.crop_size = crop_size

    def __len__(self):
        return len(self.names)

    def _load_pair(self, name):
        hazy = load_tensor(self.haze_dir / name)
        gt = load_tensor(label_path(self.gt_dir, name))
        return hazy, gt

    def __getitem__(self, idx):
        name = self.names[idx]
        assigned = self.perm_map[name]
        hazy, gt = self._load_pair(name)
        assigned_hazy, assigned_gt = self._load_pair(assigned)
        _, h, w = hazy.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        _, ah, aw = assigned_hazy.shape
        assigned_top = random.randint(0, ah - self.crop_size)
        assigned_left = random.randint(0, aw - self.crop_size)
        hazy = hazy[:, top : top + self.crop_size, left : left + self.crop_size]
        gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]
        assigned_hazy = assigned_hazy[
            :, assigned_top : assigned_top + self.crop_size, assigned_left : assigned_left + self.crop_size
        ]
        assigned_gt = assigned_gt[
            :, assigned_top : assigned_top + self.crop_size, assigned_left : assigned_left + self.crop_size
        ]
        return name, assigned, hazy, gt, assigned_hazy, assigned_gt


def collate_permuted(batch):
    names, assigned, hazy, gt, assigned_hazy, assigned_gt = zip(*batch)
    return (
        list(names),
        list(assigned),
        torch.stack(hazy, 0),
        torch.stack(gt, 0),
        torch.stack(assigned_hazy, 0),
        torch.stack(assigned_gt, 0),
    )


def load_head(path, device):
    head = d7c.MultiContextNeedHead().to(device)
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


def set_frozen(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def make_derangement(names, seed):
    names = list(sorted(names))
    rng = np.random.default_rng(seed)
    perm = np.arange(len(names))
    if len(perm) > 1:
        for _ in range(1000):
            rng.shuffle(perm)
            if not np.any(perm == np.arange(len(names))):
                break
        else:
            perm = np.roll(np.arange(len(names)), 1)
    return {names[i]: names[int(perm[i])] for i in range(len(names))}


def sample_np(x, sample_size):
    return v2d.sample_np(x, sample_size)


def collect_multi_pack(model, density_head, heads, names, device, target_info, density_stats, args, split_name, limit):
    dataset = Haze4KPairDataset(names, args.data_dir, max_items=limit, seed=args.seed)
    records = []
    concat = {"target": [], "density": [], "density_pred": []}
    for key in heads:
        concat[key] = []
    q_mode = "quantile"
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2d.v2b.v2.pad32(hazy)
            a0, context = d7c.convir_a0_context(model, density_head, padded)
            a0 = a0[:, :, :h, :w]
            context = context[:, :, :h, :w]
            raw_need = v2d.v2b.v2.raw_need(a0, gt, args.blur_kernel)
            target = v2d.v2b.make_target(raw_need, target_info, q_mode)
            density_target = v2d.v2b.v2.normalize(
                v2d.v2b.v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            density_pred = context[:, -1:]
            target_np = sample_np(target, args.metric_sample_size)
            density_np = sample_np(density_target, args.metric_sample_size)
            density_pred_np = sample_np(density_pred, args.metric_sample_size)
            rec = {
                "name": name,
                "target": target_np,
                "density": density_np,
                "density_pred": density_pred_np,
                "target_hash": sha_array(target_np),
                "density_hash": sha_array(density_np),
                "preds": {},
            }
            concat["target"].append(target_np)
            concat["density"].append(density_np)
            concat["density_pred"].append(density_pred_np)
            for key, head in heads.items():
                pred, _ = d7c.predict_head(head, context)
                pred_np = sample_np(pred, args.metric_sample_size)
                rec["preds"][key] = pred_np
                concat[key].append(pred_np)
            records.append(rec)
            if (idx + 1) % args.progress_every == 0:
                print(f"collect_{split_name} {idx + 1}/{len(dataset)}", flush=True)
    arrays = {key: np.concatenate(vals).astype(np.float32, copy=False) for key, vals in concat.items()}
    return {"records": records, "arrays": arrays}


def arrays_for_variant(pack, variant, target_records=None):
    if target_records is None:
        target = pack["arrays"]["target"]
        density = pack["arrays"]["density"]
    else:
        target = np.concatenate([rec["target"] for rec in target_records]).astype(np.float32, copy=False)
        density = np.concatenate([rec["density"] for rec in target_records]).astype(np.float32, copy=False)
    pred = pack["arrays"][variant] if variant != "density_pred" else pack["arrays"]["density_pred"]
    return target, pred, density


def threshold_metrics(threshold, target, pred, density, records, q33, q66, density_q33, base):
    high = target >= q66
    density_low = density <= density_q33
    hard_negative = (target <= q33) & density_low
    ldhn = high & density_low
    pred_high = pred >= threshold
    per_false = []
    per_ldhn_recall = []
    per_ldhn_count = []
    offset = 0
    for rec in records:
        n = rec["target"].size
        target_i = target[offset : offset + n]
        density_i = density[offset : offset + n]
        pred_i = pred[offset : offset + n]
        offset += n
        hn_i = (target_i <= q33) & (density_i <= density_q33)
        ldhn_i = (target_i >= q66) & (density_i <= density_q33)
        ph_i = pred_i >= threshold
        per_false.append(float(ph_i[hn_i].mean()) if int(hn_i.sum()) else math.nan)
        per_ldhn_count.append(int(ldhn_i.sum()))
        per_ldhn_recall.append(float(ph_i[ldhn_i].mean()) if int(ldhn_i.sum()) else math.nan)
    ldhn_pred = pred_high & ldhn
    density_low_pred = pred_high & density_low
    row = {
        "threshold": float(threshold),
        "coverage": float(pred_high.mean()),
        "recall": float(pred_high[high].mean()) if int(high.sum()) else math.nan,
        "precision": float((pred_high & high).sum() / max(int(pred_high.sum()), 1)),
        "false_global": float(pred_high[hard_negative].mean()) if int(hard_negative.sum()) else math.nan,
        "false_per_image_mean": finite_mean(per_false),
        "false_per_image_p90": finite_quantile(per_false, 0.90),
        "false_per_image_p95": finite_quantile(per_false, 0.95),
        "ldhn_recall": float(pred_high[ldhn].mean()) if int(ldhn.sum()) else math.nan,
        "ldhn_precision": float(ldhn_pred.sum() / max(int(density_low_pred.sum()), 1)),
        "ldhn_per_image_recall_p50": finite_quantile(per_ldhn_recall, 0.50),
        "ldhn_per_image_recall_p75": finite_quantile(per_ldhn_recall, 0.75),
        "ldhn_per_image_recall_p90": finite_quantile(per_ldhn_recall, 0.90),
        "ldhn_images_with_support": int(sum(c > 0 for c in per_ldhn_count)),
    }
    row.update(base)
    return row


def base_metrics(target, pred, density, q33, q66, density_q33):
    high = target >= q66
    low_or_high = (target <= q33) | high
    hard_negative = (target <= q33) & (density <= density_q33)
    positives_or_hn = high | hard_negative
    bins, mono, valid = v2d.v2b.bin_rows(target, pred, [0.20, q33, q66, 0.80], "tmp")
    return {
        "pearson": v2d.v2b.pearson(pred, target),
        "spearman": v2d.v2b.spearman(pred, target),
        "auroc": v2d.v2b.auroc(high[low_or_high], pred[low_or_high]),
        "auprc": v2d.v2b.auprc(high, pred),
        "auroc_high_vs_hard_negative": (
            v2d.v2b.auroc(high[positives_or_hn], pred[positives_or_hn]) if int(positives_or_hn.sum()) else math.nan
        ),
        "monotonic_pairs": mono,
        "monotonic_valid_pairs": valid,
        "target_high_prevalence": float(high.mean()),
    }


def fixed_threshold_summary(variant, target, pred, density, records, q33, q66, density_q33, threshold):
    base = base_metrics(target, pred, density, q33, q66, density_q33)
    row = threshold_metrics(threshold, target, pred, density, records, q33, q66, density_q33, base)
    row["variant"] = variant
    row["selected_threshold"] = float(threshold)
    return row


def build_curve(variant, train_score, val_target, val_score, val_density, val_records, q33, q66, density_q33, args):
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(train_score, np.linspace(0.0, 1.0, args.threshold_grid)),
                np.linspace(0.05, 0.95, 19),
                np.asarray([DEFAULT_THRESHOLD], dtype=np.float32),
            ]
        )
    )
    base = base_metrics(val_target, val_score, val_density, q33, q66, density_q33)
    rows = [
        {"variant": variant, **threshold_metrics(t, val_target, val_score, val_density, val_records, q33, q66, density_q33, base)}
        for t in sorted(grid)
    ]
    return rows


def matched_threshold(train_score, target_coverage):
    return float(np.quantile(train_score, max(0.0, min(1.0, 1.0 - target_coverage))))


def permutation_null(val_pack, variant, seeds, q33, q66, density_q33, args, out_dir):
    def sampled_records(perm_map=None):
        rows = []
        by_name = {rec["name"]: rec for rec in val_pack["records"]}
        for rec in val_pack["records"]:
            target_rec = by_name[perm_map[rec["name"]]] if perm_map is not None else rec
            n = rec["target"].size
            k = n if args.permutation_pixel_sample_per_image <= 0 else min(n, args.permutation_pixel_sample_per_image)
            seed_bytes = f"{args.seed}:{rec['name']}".encode("utf-8")
            sample_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little") % (2**32)
            rng = np.random.default_rng(sample_seed)
            idx = np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)
            rows.append(
                {
                    "name": rec["name"],
                    "target": target_rec["target"][idx],
                    "density": target_rec["density"][idx],
                    "preds": {variant: rec["preds"][variant][idx]},
                }
            )
        return rows

    original_records = sampled_records()
    original_target = np.concatenate([rec["target"] for rec in original_records]).astype(np.float32, copy=False)
    original_density = np.concatenate([rec["density"] for rec in original_records]).astype(np.float32, copy=False)
    original_pred = np.concatenate([rec["preds"][variant] for rec in original_records]).astype(np.float32, copy=False)
    original = fixed_threshold_summary(
        variant,
        original_target,
        original_pred,
        original_density,
        original_records,
        q33,
        q66,
        density_q33,
        args.candidate_threshold,
    )
    records = val_pack["records"]
    rows = []
    map_rows = []
    for seed in seeds:
        perm_map = make_derangement([rec["name"] for rec in records], seed)
        by_name = {rec["name"]: rec for rec in records}
        perm_records = sampled_records(perm_map)
        target = np.concatenate([rec["target"] for rec in perm_records]).astype(np.float32, copy=False)
        density = np.concatenate([rec["density"] for rec in perm_records]).astype(np.float32, copy=False)
        pred = np.concatenate([rec["preds"][variant] for rec in perm_records]).astype(np.float32, copy=False)
        row = fixed_threshold_summary(
            f"{variant}_perm_seed{seed}",
            target,
            pred,
            density,
            perm_records,
            q33,
            q66,
            density_q33,
            args.candidate_threshold,
        )
        row["seed"] = seed
        row["pixels_per_image"] = args.permutation_pixel_sample_per_image
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"permutation_null {len(rows)}/{len(seeds)}", flush=True)
        if seed == args.seed:
            for rec in records:
                assigned = by_name[perm_map[rec["name"]]]
                map_rows.append(
                    {
                        "seed": seed,
                        "image_id": rec["name"],
                        "assigned_target_image_id": assigned["name"],
                        "assigned_target_hash": assigned["target_hash"],
                        "assigned_density_hash": assigned["density_hash"],
                    }
                )
    metrics = ["spearman", "auroc", "auprc"]
    pvalues = {}
    for metric in metrics:
        vals = np.asarray([row[metric] for row in rows], dtype=np.float64)
        obs = float(original[metric])
        pvalues[f"{metric}_empirical_p"] = float((1 + np.sum(vals >= obs)) / (len(vals) + 1))
        pvalues[f"{metric}_null_median"] = float(np.median(vals))
        pvalues[f"{metric}_null_p95"] = float(np.quantile(vals, 0.95))
        pvalues[f"{metric}_null_p99"] = float(np.quantile(vals, 0.99))
        pvalues[f"{metric}_original"] = obs
        pvalues[f"{metric}_original_gt_p99"] = bool(obs > pvalues[f"{metric}_null_p99"])
    pvalues["permutation_count"] = len(rows)
    pvalues["pixels_per_image"] = args.permutation_pixel_sample_per_image
    pvalues["threshold"] = args.candidate_threshold
    pvalues["pass"] = bool(
        pvalues["spearman_original_gt_p99"]
        and pvalues["auroc_original_gt_p99"]
        and pvalues["auprc_original_gt_p99"]
        and pvalues["spearman_empirical_p"] <= 0.01
        and pvalues["auroc_empirical_p"] <= 0.01
        and pvalues["auprc_empirical_p"] <= 0.01
        and pvalues["spearman_null_median"] <= 0.05
        and pvalues["spearman_null_p95"] <= 0.10
    )
    write_csv(out_dir / "fixed_image_permutation_null_distribution.csv", rows)
    write_csv(out_dir / f"fixed_image_permutation_map_seed{args.seed}.csv", map_rows)
    (out_dir / "fixed_image_permutation_pvalues.json").write_text(json.dumps(pvalues, indent=2), encoding="utf-8")
    return original, rows, pvalues


def ldhn_support(records, q66, density_q33):
    rows = []
    total = 0
    ldhn_total = 0
    image_hits = 0
    for rec in records:
        ldhn = (rec["target"] >= q66) & (rec["density"] <= density_q33)
        count = int(ldhn.sum())
        total += int(rec["target"].size)
        ldhn_total += count
        if count > 0:
            image_hits += 1
        rows.append(
            {
                "name": rec["name"],
                "pixels": int(rec["target"].size),
                "ldhn_pixels": count,
                "ldhn_coverage": float(count / max(int(rec["target"].size), 1)),
                "target_mean": float(np.mean(rec["target"])),
                "density_mean": float(np.mean(rec["density"])),
            }
        )
    coverages = [r["ldhn_coverage"] for r in rows]
    summary = {
        "ldhn_pixel_coverage": float(ldhn_total / max(total, 1)),
        "ldhn_image_coverage": float(image_hits / max(len(records), 1)),
        "images": len(records),
        "images_ldhn_ge_1pct": int(sum(r["ldhn_coverage"] >= 0.01 for r in rows)),
        "images_ldhn_ge_3pct": int(sum(r["ldhn_coverage"] >= 0.03 for r in rows)),
        "images_ldhn_ge_5pct": int(sum(r["ldhn_coverage"] >= 0.05 for r in rows)),
        "per_image_ldhn_p50": finite_quantile(coverages, 0.50),
        "per_image_ldhn_p75": finite_quantile(coverages, 0.75),
        "per_image_ldhn_p90": finite_quantile(coverages, 0.90),
        "per_image_ldhn_p95": finite_quantile(coverages, 0.95),
        "per_image_ldhn_max": max(coverages) if coverages else math.nan,
    }
    return rows, summary


def train_permuted_head(model, density_head, train_names, train_perm_map, device, target_info, density_stats, args, variant):
    head = d7c.MultiContextNeedHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = PermutedCropDataset(
        train_names, args.data_dir, train_perm_map, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_permuted,
    )
    thresholds = [0.20, 0.33, 0.66, 0.80]
    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    bce = nn.BCEWithLogitsLoss()
    rows = []
    model.eval()
    density_head.eval()
    for epoch in range(1, args.control_epochs + 1):
        losses = []
        start = time.time()
        for step, (_, _, hazy, gt, assigned_hazy, assigned_gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            assigned_hazy = assigned_hazy.to(device, non_blocking=True)
            assigned_gt = assigned_gt.to(device, non_blocking=True)
            with torch.no_grad():
                _, context = d7c.convir_a0_context(model, density_head, hazy)
                assigned_a0, _ = d7c.convir_a0_context(model, density_head, assigned_hazy)
                raw_need = v2d.v2b.v2.raw_need(assigned_a0, assigned_gt, args.blur_kernel)
                target = v2d.v2b.make_target(raw_need, target_info, "quantile")
                if variant == "perm_all_supervision":
                    density_source_hazy = assigned_hazy
                    density_source_gt = assigned_gt
                else:
                    density_source_hazy = hazy
                    density_source_gt = gt
                density_target = v2d.v2b.v2.normalize(
                    v2d.v2b.v2.raw_density(density_source_hazy, density_source_gt, args.blur_kernel),
                    density_stats["density"]["raw_p1"],
                    density_stats["density"]["raw_p99"],
                )
            pred, logits = d7c.predict_head(head, context.detach())
            ord_loss = logits.new_tensor(0.0)
            for i, thr in enumerate(thresholds):
                ord_loss = ord_loss + bce(logits[:, i : i + 1], (target >= thr).float())
            smooth = F.smooth_l1_loss(pred, target)
            hn_mask = (target <= q33) & (density_target <= density_q33)
            pos_mask = target >= q66
            loss = (
                smooth
                + args.ordinal_weight * ord_loss
                + args.hn_weight * v2d.hard_negative_loss(pred, hn_mask, args.tau_neg)
                + args.pos_weight * v2d.positive_response_loss(pred, pos_mask, args.tau_pos)
                + args.pair_weight * v2d.pairwise_loss(pred, pos_mask, hn_mask, args.pair_margin, args.pair_sample)
                + args.topk_weight * v2d.topk_hn_loss(pred, hn_mask, args.tau_neg, args.topk_fraction)
                + args.tv_weight * v2d.v2b.tv_loss(pred)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
            if step % args.progress_every == 0:
                print(f"train_{variant} epoch={epoch} step={step}/{len(loader)} loss={statistics.mean(losses[-args.progress_every:]):.6f}", flush=True)
        rows.append(
            {
                "variant": variant,
                "epoch": epoch,
                "steps": len(loader),
                "loss_mean": statistics.mean(losses),
                "elapsed_sec": time.time() - start,
            }
        )
    return head, rows


def write_docs(out_dir, run_summary):
    (out_dir / "fixed_image_permutation_protocol.md").write_text(
        "# Fixed Image-Level Permutation Control\n\n"
        "Frozen D7c predictions are kept on each val_inner image. Target and density masks are reassigned by a fixed image-level derangement. "
        "The null distribution uses deterministic seeds and reports empirical p-values against the original D7c candidate.\n",
        encoding="utf-8",
    )
    (out_dir / "d7c_candidate_threshold_protocol.md").write_text(
        "# D7c Candidate Threshold Protocol\n\n"
        f"Candidate: `d7c_mc_topk_hn_ordinal`. Frozen threshold: `{run_summary['candidate_threshold']}`. "
        "The threshold is inherited from v2d and is not retuned on v2e. Locked Haze4K test usage: none.\n",
        encoding="utf-8",
    )
    (out_dir / "low_density_high_need_definition.md").write_text(
        "# Low-Density High-Need Definition\n\n"
        "`LDHN = density_target <= train_inner_density_q33 AND R_need_target >= train_inner_need_q66`.\n"
        "The audit reports support before interpreting recall.\n",
        encoding="utf-8",
    )
    (out_dir / "random_target_control_protocol.md").write_text(
        "# Random/Permutation Target Controls\n\n"
        "v2e uses fixed image-level permutation controls. v2d random and shuffled controls are retained as historical weak-control evidence, "
        "but v2e decisions are based on fixed target/mask pairing plus density-matched controls.\n",
        encoding="utf-8",
    )
    (out_dir / "v2e_gate_definition.md").write_text(
        "# CHD-RM v2e Gate Definition\n\n"
        "- Controls must be clean: fixed permutation p-values <= 0.01 and Spearman null median <= 0.05 / p95 <= 0.10.\n"
        "- D7c must beat density-only matched threshold by Spearman >= 0.15, AUROC >= 0.10, AUPRC >= 0.10, and precision >= 0.08.\n"
        "- Candidate safety target: false_global <= 0.01, false_p90 <= 0.05, false_p95 <= 0.10.\n"
        "- LDHN protection target: LDHN recall >= 0.10 preferred >= 0.12, with support reported.\n"
        "- D2, RARM, v3, and locked Haze4K test remain forbidden.\n",
        encoding="utf-8",
    )
    (out_dir / "v2e_route_design.md").write_text(
        "# CHD-RM v2e Route Design\n\n"
        "v2e is a D7c control and recall-protection audit. It keeps ConvIR-B frozen, keeps RARM disconnected, and does not use the locked test. "
        "The route first audits the frozen v2d D7c top-k candidate, then evaluates fixed image-level permutation controls, density-only matched-threshold controls, "
        "and low-density high-need recall. D7c-RP is authorized only if controls are clean but LDHN recall is the remaining blocker.\n",
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
    ap.add_argument("--permutation_count", type=int, default=100)
    ap.add_argument("--permutation_pixel_sample_per_image", type=int, default=512)
    ap.add_argument("--candidate_threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--target_coverage", type=float, default=TARGET_COVERAGE)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--threshold_grid", type=int, default=121)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--control_epochs", type=int, default=6)
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
    ap.add_argument("--skip_train_controls", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.txt").write_text("RUNNING_AUDIT\n", encoding="utf-8")
    v2d.v2b.set_seed(args.seed)

    train_names, val_names, device, model, density_stats, target_info = v2d.load_runtime(args)
    density_head = v2d.load_density_head(args.density_artifact, device)
    topk_head = load_head(args.d7c_topk_artifact, device)
    hn_head = load_head(args.d7c_hn_artifact, device)
    set_frozen(model)
    set_frozen(density_head)
    set_frozen(topk_head)
    set_frozen(hn_head)

    q33 = float(target_info["quantile"]["q33"])
    q66 = float(target_info["quantile"]["q66"])
    density_q33 = float(density_stats["density"]["q33"])
    heads = {
        "d7c_mc_topk_hn_ordinal": topk_head,
        "d7c_mc_hn_ordinal": hn_head,
    }
    print("COLLECT_TRAIN_BEGIN", flush=True)
    train_pack = collect_multi_pack(model, density_head, heads, train_names, device, target_info, density_stats, args, "train_inner", args.train_limit)
    print("COLLECT_VAL_BEGIN", flush=True)
    val_pack = collect_multi_pack(model, density_head, heads, val_names, device, target_info, density_stats, args, "val_inner", args.val_limit)

    target, pred_topk, density = arrays_for_variant(val_pack, "d7c_mc_topk_hn_ordinal")
    candidate_summary = fixed_threshold_summary(
        "d7c_mc_topk_hn_ordinal", target, pred_topk, density, val_pack["records"], q33, q66, density_q33, args.candidate_threshold
    )
    target_hn, pred_hn, density_hn = arrays_for_variant(val_pack, "d7c_mc_hn_ordinal")
    hn_summary = fixed_threshold_summary(
        "d7c_mc_hn_ordinal_at_topk_threshold", target_hn, pred_hn, density_hn, val_pack["records"], q33, q66, density_q33, args.candidate_threshold
    )
    train_target, train_pred_topk, train_density = arrays_for_variant(train_pack, "d7c_mc_topk_hn_ordinal")
    train_density_score = train_pack["arrays"]["density_pred"]
    val_density_score = val_pack["arrays"]["density_pred"]
    density_threshold = matched_threshold(train_density_score, args.target_coverage)
    density_summary = fixed_threshold_summary(
        "density_only_matched_threshold",
        target,
        val_density_score,
        density,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        density_threshold,
    )
    density_summary["selected_threshold"] = density_threshold
    density_summary["target_coverage"] = args.target_coverage

    seeds = [args.seed + i for i in range(args.permutation_count)]
    original, perm_rows, pvalues = permutation_null(val_pack, "d7c_mc_topk_hn_ordinal", seeds, q33, q66, density_q33, args, args.output_dir)

    candidate_curve = build_curve(
        "d7c_mc_topk_hn_ordinal",
        train_pred_topk,
        target,
        pred_topk,
        density,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        args,
    )
    train_target_hn, train_pred_hn, _ = arrays_for_variant(train_pack, "d7c_mc_hn_ordinal")
    hn_curve = build_curve(
        "d7c_mc_hn_ordinal",
        train_pred_hn,
        target_hn,
        pred_hn,
        density_hn,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        args,
    )
    density_curve = build_curve(
        "density_only_matched_threshold",
        train_density_score,
        target,
        val_density_score,
        density,
        val_pack["records"],
        q33,
        q66,
        density_q33,
        args,
    )
    support_rows, support_summary = ldhn_support(val_pack["records"], q66, density_q33)
    write_csv(args.output_dir / "low_density_high_need_support.csv", support_rows)
    (args.output_dir / "low_density_high_need_support_summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "low_density_high_need_recall_curve.csv", candidate_curve + hn_curve + density_curve)
    write_csv(args.output_dir / "ldhn_recall_vs_false_strong_tradeoff.csv", candidate_curve + hn_curve + density_curve)
    per_image_rows = []
    for rec in val_pack["records"]:
        ldhn = (rec["target"] >= q66) & (rec["density"] <= density_q33)
        for variant, score in [
            ("d7c_mc_topk_hn_ordinal", rec["preds"]["d7c_mc_topk_hn_ordinal"]),
            ("d7c_mc_hn_ordinal", rec["preds"]["d7c_mc_hn_ordinal"]),
            ("density_only_matched_threshold", rec["density_pred"]),
        ]:
            threshold = args.candidate_threshold if variant != "density_only_matched_threshold" else density_threshold
            ph = score >= threshold
            per_image_rows.append(
                {
                    "variant": variant,
                    "name": rec["name"],
                    "threshold": threshold,
                    "ldhn_pixels": int(ldhn.sum()),
                    "ldhn_recall": float(ph[ldhn].mean()) if int(ldhn.sum()) else math.nan,
                    "pred_high_coverage": float(ph.mean()),
                }
            )
    write_csv(args.output_dir / "per_image_ldhn_recall_distribution.csv", per_image_rows)

    gap = {
        "candidate": "d7c_mc_topk_hn_ordinal",
        "control": "density_only_matched_threshold",
        "delta_spearman": candidate_summary["spearman"] - density_summary["spearman"],
        "delta_auroc": candidate_summary["auroc"] - density_summary["auroc"],
        "delta_auprc": candidate_summary["auprc"] - density_summary["auprc"],
        "delta_precision": candidate_summary["precision"] - density_summary["precision"],
        "candidate_false_global": candidate_summary["false_global"],
        "density_false_global": density_summary["false_global"],
        "candidate_false_p90": candidate_summary["false_per_image_p90"],
        "density_false_p90": density_summary["false_per_image_p90"],
    }
    gap["pass"] = bool(
        gap["delta_spearman"] >= 0.15
        and gap["delta_auroc"] >= 0.10
        and gap["delta_auprc"] >= 0.10
        and gap["delta_precision"] >= 0.08
        and candidate_summary["false_global"] <= density_summary["false_global"] + 0.01
        and candidate_summary["false_per_image_p90"] <= density_summary["false_per_image_p90"] + 0.03
    )
    write_csv(args.output_dir / "density_only_vs_d7c_gap_summary.csv", [gap])
    write_csv(args.output_dir / "density_only_matched_threshold_curve.csv", density_curve)
    (args.output_dir / "density_only_matched_threshold_summary.md").write_text(
        "# Density-Only Matched-Threshold Summary\n\n"
        f"Matched threshold: `{density_threshold:.6f}` for target coverage `{args.target_coverage:.6f}`.\n\n"
        f"D7c minus density-only gaps: Spearman `{gap['delta_spearman']:.4f}`, AUROC `{gap['delta_auroc']:.4f}`, "
        f"AUPRC `{gap['delta_auprc']:.4f}`, precision `{gap['delta_precision']:.4f}`.\n\n"
        f"Decision: `{'PASS' if gap['pass'] else 'FAIL'}`.\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "d7c_candidate_metrics_reproduced.csv", [candidate_summary, hn_summary])
    manifest = {
        "candidate": "d7c_mc_topk_hn_ordinal",
        "artifact": args.d7c_topk_artifact,
        "selected_threshold": args.candidate_threshold,
        "target_coverage": args.target_coverage,
        "reproduced_summary": candidate_summary,
        "locked_haze4k_test_usage": "none",
        "D2": "not_run",
        "RARM": "not_connected_or_trained",
        "v3": "not_run",
    }
    (args.output_dir / "d7c_candidate_frozen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    control_summaries = []
    train_logs = []
    if not args.skip_train_controls:
        artifact_dir = args.output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        train_perm = make_derangement(train_names, args.seed)
        for variant in ["perm_all_supervision", "perm_need_keep_density"]:
            print(f"TRAIN_CONTROL_{variant}_BEGIN", flush=True)
            head, rows = train_permuted_head(model, density_head, train_names, train_perm, device, target_info, density_stats, args, variant)
            train_logs.extend(rows)
            torch.save({"variant": variant, "state_dict": head.state_dict(), "args": vars(args)}, artifact_dir / f"{variant}_head.pt")
            set_frozen(head)
            control_pack = collect_multi_pack(
                model, density_head, {variant: head}, val_names, device, target_info, density_stats, args, f"val_{variant}", args.val_limit
            )
            c_target, c_pred, c_density = arrays_for_variant(control_pack, variant)
            c_summary = fixed_threshold_summary(
                variant, c_target, c_pred, c_density, control_pack["records"], q33, q66, density_q33, args.candidate_threshold
            )
            control_summaries.append(c_summary)
    write_csv(args.output_dir / "perm_all_supervision_control_summary.csv", [r for r in control_summaries if r["variant"] == "perm_all_supervision"])
    write_csv(args.output_dir / "perm_need_keep_density_control_summary.csv", [r for r in control_summaries if r["variant"] == "perm_need_keep_density"])
    write_csv(args.output_dir / "v2e_control_train_log.csv", train_logs)

    control_clean = bool(pvalues["pass"])
    density_clean = bool(gap["pass"])
    ldhn_pass = bool(candidate_summary["ldhn_recall"] >= 0.10)
    safety_pass = bool(
        candidate_summary["false_global"] <= SAFETY_FALSE_GLOBAL
        and candidate_summary["false_per_image_p90"] <= SAFETY_FALSE_P90
        and candidate_summary["false_per_image_p95"] <= SAFETY_FALSE_P95
    )
    ranking_pass = bool(
        candidate_summary["spearman"] >= 0.50
        and candidate_summary["auroc"] >= 0.83
        and candidate_summary["auprc"] >= 0.62
        and candidate_summary["precision"] >= 0.60
        and candidate_summary["recall"] >= 0.42
        and candidate_summary["monotonic_pairs"] == 4
        and candidate_summary["monotonic_valid_pairs"] == 4
    )
    if not control_clean:
        decision = "PAUSE_V2E_FIXED_PERMUTATION_CONTROL_NOT_CLEAN_NO_V3"
    elif not density_clean:
        decision = "PAUSE_V2E_DENSITY_MATCHED_CONTROL_TOO_STRONG_NO_V3"
    elif not safety_pass or not ranking_pass:
        decision = "PAUSE_V2E_D7C_CANDIDATE_GATE_FAIL_NO_V3"
    elif not ldhn_pass:
        decision = "PAUSE_V2E_CONTROLS_CLEAN_BUT_LDHN_RECALL_LOW_RUN_D7C_RP"
    else:
        decision = "COMPLETED_V2E_D7C_CONTROL_RECALL_AUDIT_PASS_AUTHORIZE_V3_NOOP_GATE_ONLY"

    a0_audit = {
        "status": "PASS_BY_CONSTRUCTION",
        "a0_output_changed": False,
        "reason": "v2e trains/evaluates side heads only; ConvIR-B dehazing output is frozen and RARM is disconnected.",
        "locked_haze4k_test_usage": "none",
    }
    no_test = {
        "locked_haze4k_test_usage": "none",
        "data_dir": args.data_dir,
        "split_json": args.split_json,
        "forbidden": ["D2", "RARM connection", "RARM training", "v3", "locked Haze4K test"],
    }
    (args.output_dir / "a0_equivalence_audit.json").write_text(json.dumps(a0_audit, indent=2), encoding="utf-8")
    (args.output_dir / "no_locked_test_audit.json").write_text(json.dumps(no_test, indent=2), encoding="utf-8")
    resource_rows = [
        {"key": "data_dir", "value": args.data_dir},
        {"key": "checkpoint", "value": args.checkpoint},
        {"key": "split_json", "value": args.split_json},
        {"key": "density_artifact", "value": args.density_artifact},
        {"key": "d7c_topk_artifact", "value": args.d7c_topk_artifact},
        {"key": "d7c_hn_artifact", "value": args.d7c_hn_artifact},
        {"key": "output_dir", "value": str(args.output_dir)},
        {"key": "locked_haze4k_test_usage", "value": "none"},
    ]
    write_csv(args.output_dir / "resource_summary.csv", resource_rows)

    run_summary = {
        "decision": decision,
        "candidate_threshold": args.candidate_threshold,
        "candidate_summary": candidate_summary,
        "d7c_hn_at_topk_threshold": hn_summary,
        "fixed_permutation_pvalues": pvalues,
        "density_only_matched_summary": density_summary,
        "density_gap": gap,
        "ldhn_support": support_summary,
        "train_time_controls": control_summaries,
        "gate": {
            "fixed_permutation_clean": control_clean,
            "density_matched_clean": density_clean,
            "safety_pass": safety_pass,
            "ranking_pass": ranking_pass,
            "ldhn_recall_pass": ldhn_pass,
        },
        "policy": no_test,
    }
    (args.output_dir / "v2e_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (args.output_dir / "v2e_overall_result_summary.md").write_text(
        "# CHD-RM v2e D7c Control Recall Audit Summary\n\n"
        f"Decision: `{decision}`\n\n"
        f"D7c top-k reproduced Spearman `{candidate_summary['spearman']:.4f}`, AUROC `{candidate_summary['auroc']:.4f}`, "
        f"AUPRC `{candidate_summary['auprc']:.4f}`, coverage `{candidate_summary['coverage']:.4f}`, "
        f"false-global `{candidate_summary['false_global']:.4f}`, false-p90 `{candidate_summary['false_per_image_p90']:.4f}`, "
        f"false-p95 `{candidate_summary['false_per_image_p95']:.4f}`, LDHN recall `{candidate_summary['ldhn_recall']:.4f}`.\n\n"
        f"Fixed permutation clean: `{control_clean}`. Density matched clean: `{density_clean}`. "
        f"Safety pass: `{safety_pass}`. Ranking pass: `{ranking_pass}`. LDHN recall pass: `{ldhn_pass}`.\n\n"
        "Forbidden and not used: D2, RARM connection/training, v3, locked Haze4K test.\n",
        encoding="utf-8",
    )
    (args.output_dir / "decision_record.md").write_text(
        f"# CHD-RM v2e Decision Record\n\nDecision: `{decision}`\n\nLocked Haze4K test usage: none.\nD2/RARM/v3: not run.\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# CHD-RM v2e D7c Control Recall Audit Evidence\n\n"
        f"Status: `{decision}`\n\n"
        "Primary files: `v2e_overall_result_summary.md`, `v2e_run_summary.json`, "
        "`fixed_image_permutation_pvalues.json`, `density_only_vs_d7c_gap_summary.csv`, "
        "and `low_density_high_need_recall_curve.csv`.\n\n"
        "This route keeps ConvIR-B frozen, keeps RARM disconnected, and does not use the locked Haze4K test.\n",
        encoding="utf-8",
    )
    write_docs(args.output_dir, run_summary)
    (args.output_dir / "status.txt").write_text(f"{decision}\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
