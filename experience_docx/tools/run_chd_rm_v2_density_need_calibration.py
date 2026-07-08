import argparse
import csv
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
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ConvIR import build_net

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
MIN_PRED_HIGH_COVERAGE = 0.01
MAX_PRED_HIGH_COVERAGE = 0.90


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    def __init__(self, names, data_dir, crop_size=None, max_items=None, seed=3407):
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
        hazy = load_tensor(self.haze_dir / name)
        gt = load_tensor(label_path(self.gt_dir, name))
        if self.crop_size is not None:
            _, h, w = hazy.shape
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"{name} is smaller than crop {self.crop_size}: {(h, w)}")
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            hazy = hazy[:, top : top + self.crop_size, left : left + self.crop_size]
            gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]
        return name, hazy, gt


def collate_pairs(batch):
    names, hazy, gt = zip(*batch)
    return list(names), torch.stack(hazy, dim=0), torch.stack(gt, dim=0)


def load_split_names(split_json, split_name):
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    names = split["splits"][split_name]
    return sorted(names)


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def pad32(x):
    _, _, h, w = x.shape
    factor = 32
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    pad_h = padded_h - h if h % factor != 0 else 0
    pad_w = padded_w - w if w % factor != 0 else 0
    return F.pad(x, (0, pad_w, 0, pad_h), "reflect"), h, w


def convir_a0_and_res1(model, x):
    x_2 = F.interpolate(x, scale_factor=0.5)
    x_4 = F.interpolate(x_2, scale_factor=0.5)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)

    x_ = model.feat_extract[0](x)
    res1 = model.Encoder[0](x_)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    z = model.Encoder[2](z)

    z = model.Decoder[0](z)
    z_ = model.ConvsOut[0](z)
    z = model.feat_extract[3](z)

    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z_ = model.ConvsOut[1](z)
    z = model.feat_extract[4](z)

    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z = model.feat_extract[5](z)
    a0 = z + x
    return torch.clamp(a0, 0, 1), res1


def gray_abs(a, b):
    weights = GRAY_WEIGHTS.to(device=a.device, dtype=a.dtype)
    return ((a - b).abs() * weights).sum(dim=1, keepdim=True)


def smooth_map(x, kernel):
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)


def raw_density(hazy, gt, kernel):
    return smooth_map(gray_abs(hazy, gt), kernel)


def raw_need(a0, gt, kernel):
    return smooth_map(gray_abs(a0, gt), kernel)


def raw_dark_density(hazy, kernel):
    dark = hazy.min(dim=1, keepdim=True).values
    return smooth_map(dark, kernel)


def normalize(raw, lo, hi):
    denom = max(float(hi) - float(lo), 1e-6)
    return torch.clamp((raw - float(lo)) / denom, 0.0, 1.0)


def downsample_values(x, sample_size):
    if x.shape[-2] != sample_size or x.shape[-1] != sample_size:
        x = F.adaptive_avg_pool2d(x, (sample_size, sample_size))
    return x.detach().flatten().float().cpu().numpy()


def percentile(values, pct):
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def collect_thresholds(model, names, data_dir, device, args):
    dataset = Haze4KPairDataset(
        names,
        data_dir,
        crop_size=None,
        max_items=args.threshold_limit,
        seed=args.seed,
    )
    density_raw = []
    need_raw = []
    dark_raw = []
    start = time.time()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = pad32(hazy)
            a0, _ = convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            density_raw.append(downsample_values(raw_density(hazy, gt, args.blur_kernel), args.metric_sample_size))
            need_raw.append(downsample_values(raw_need(a0, gt, args.blur_kernel), args.metric_sample_size))
            dark_raw.append(downsample_values(raw_dark_density(hazy, args.dark_kernel), args.metric_sample_size))
            if (idx + 1) % args.progress_every == 0:
                print(f"thresholds {idx + 1}/{len(dataset)}", flush=True)
    density_raw = np.concatenate(density_raw)
    need_raw = np.concatenate(need_raw)
    dark_raw = np.concatenate(dark_raw)

    def build_stats(raw):
        lo = percentile(raw, 1)
        hi = percentile(raw, 99)
        norm = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return {
            "raw_p1": lo,
            "raw_p99": hi,
            "q20": percentile(norm, 20),
            "q33": percentile(norm, 33),
            "q40": percentile(norm, 40),
            "q60": percentile(norm, 60),
            "q66": percentile(norm, 66),
            "q80": percentile(norm, 80),
        }

    stats = {
        "source": "train_inner only",
        "count_images": len(dataset),
        "sample_size_per_image": args.metric_sample_size,
        "blur_kernel": args.blur_kernel,
        "dark_kernel": args.dark_kernel,
        "density": build_stats(density_raw),
        "need": build_stats(need_raw),
        "d0_dark_density": build_stats(dark_raw),
        "elapsed_sec": time.time() - start,
    }
    return stats


class DensityNeedHead(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def tv_loss(x):
    if x.shape[-2] < 2 or x.shape[-1] < 2:
        return x.new_tensor(0.0)
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (
        x[:, :, :, 1:] - x[:, :, :, :-1]
    ).abs().mean()


def target_maps(hazy, gt, a0, stats, args):
    d_raw = raw_density(hazy, gt, args.blur_kernel)
    n_raw = raw_need(a0, gt, args.blur_kernel)
    d = normalize(d_raw, stats["density"]["raw_p1"], stats["density"]["raw_p99"])
    n = normalize(n_raw, stats["need"]["raw_p1"], stats["need"]["raw_p99"])
    return d, n


def train_variant(model, train_names, data_dir, device, stats, variant, output_dir, args):
    active_density = variant in ("d1_dual", "d3_density_only", "d5_shuffled_dual")
    active_need = variant in ("d1_dual", "d4_need_only", "d5_shuffled_dual")
    out_channels = int(active_density) + int(active_need)
    head = DensityNeedHead(out_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = Haze4KPairDataset(
        train_names,
        data_dir,
        crop_size=args.crop_size,
        max_items=args.train_limit,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pairs,
    )
    bce = nn.BCEWithLogitsLoss()
    train_rows = []
    head.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        d_losses = []
        n_losses = []
        start = time.time()
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, res1 = convir_a0_and_res1(model, hazy)
                density_target, need_target = target_maps(hazy, gt, a0, stats, args)
            if variant == "d5_shuffled_dual":
                shift = random.randint(1, hazy.shape[0] - 1)
                density_target = torch.roll(density_target, shifts=shift, dims=0)
                need_target = torch.roll(need_target, shifts=shift, dims=0)

            logits = head(res1.detach())
            loss = hazy.new_tensor(0.0)
            cursor = 0
            density_loss_value = math.nan
            need_loss_value = math.nan
            if active_density:
                d_logit = logits[:, cursor : cursor + 1]
                d_pred = torch.sigmoid(d_logit)
                d_label = (density_target >= float(stats["density"]["q66"])).float()
                d_loss = (
                    F.smooth_l1_loss(d_pred, density_target)
                    + args.bce_weight * bce(d_logit, d_label)
                    + args.tv_weight * tv_loss(d_pred)
                )
                loss = loss + d_loss
                density_loss_value = float(d_loss.detach().item())
                d_losses.append(density_loss_value)
                cursor += 1
            if active_need:
                n_logit = logits[:, cursor : cursor + 1]
                n_pred = torch.sigmoid(n_logit)
                n_label = (need_target >= float(stats["need"]["q66"])).float()
                n_loss = (
                    F.smooth_l1_loss(n_pred, need_target)
                    + args.bce_weight * bce(n_logit, n_label)
                    + args.tv_weight * tv_loss(n_pred)
                )
                loss = loss + args.need_weight * n_loss
                need_loss_value = float(n_loss.detach().item())
                n_losses.append(need_loss_value)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
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
            "density_loss_mean": statistics.mean(d_losses) if d_losses else math.nan,
            "need_loss_mean": statistics.mean(n_losses) if n_losses else math.nan,
            "elapsed_sec": time.time() - start,
        }
        train_rows.append(row)
        print(json.dumps(row), flush=True)

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifact_dir / f"{variant}_head.pt"
    torch.save(
        {
            "variant": variant,
            "state_dict": head.state_dict(),
            "active_density": active_density,
            "active_need": active_need,
            "args": vars(args),
        },
        ckpt_path,
    )
    return head, train_rows, ckpt_path


def rankdata_average(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 2:
        return math.nan
    return pearson(rankdata_average(np.asarray(x)), rankdata_average(np.asarray(y)))


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata_average(scores)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(labels, scores):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    total_pos = int(labels.sum())
    if total_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(~labels_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def bin_means(target, pred, cuts):
    edges = [0.0] + list(cuts) + [1.0 + 1e-6]
    rows = []
    means = []
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == 0:
            mask = target <= hi
        else:
            mask = (target > lo) & (target <= hi)
        count = int(mask.sum())
        mean_pred = float(np.mean(pred[mask])) if count else math.nan
        mean_target = float(np.mean(target[mask])) if count else math.nan
        means.append(mean_pred)
        rows.append(
            {
                "bin_index": idx,
                "target_lo": lo,
                "target_hi": hi,
                "count": count,
                "target_mean": mean_target,
                "pred_mean": mean_pred,
            }
        )
    monotonic_pairs = 0
    valid_pairs = 0
    for a, b in zip(means[:-1], means[1:]):
        if not math.isnan(a) and not math.isnan(b):
            valid_pairs += 1
            if b >= a:
                monotonic_pairs += 1
    return rows, monotonic_pairs, valid_pairs


def summarize_signal(target, pred, low_cut, high_cut, pred_strong_cut):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    low_or_high = (target <= low_cut) | (target >= high_cut)
    heavy_label = target >= high_cut
    return {
        "pearson": pearson(pred, target),
        "spearman": spearman(pred, target),
        "auroc_high_vs_low": auroc(heavy_label[low_or_high], pred[low_or_high]),
        "auprc_high": auprc(heavy_label, pred),
        "pred_high_coverage": float(np.mean(pred >= pred_strong_cut)),
        "target_high_coverage": float(np.mean(target >= high_cut)),
    }


def predict_variant(model, head, hazy, stats, variant, args):
    if variant == "d0_handcrafted":
        d_raw = raw_dark_density(hazy, args.dark_kernel)
        density = normalize(
            d_raw,
            stats["d0_dark_density"]["raw_p1"],
            stats["d0_dark_density"]["raw_p99"],
        )
        return density, None

    a0, res1 = convir_a0_and_res1(model, hazy)
    logits = head(res1)
    active_density = variant in ("d1_dual", "d3_density_only", "d5_shuffled_dual")
    active_need = variant in ("d1_dual", "d4_need_only", "d5_shuffled_dual")
    cursor = 0
    density = None
    need = None
    if active_density:
        density = torch.sigmoid(logits[:, cursor : cursor + 1])
        cursor += 1
    if active_need:
        need = torch.sigmoid(logits[:, cursor : cursor + 1])
    return density, need


def evaluate_variant(model, head, val_names, data_dir, device, stats, variant, output_dir, args):
    dataset = Haze4KPairDataset(val_names, data_dir, crop_size=None, max_items=args.val_limit, seed=args.seed)
    density_target_values = []
    density_pred_values = []
    need_target_values = []
    need_pred_values = []
    per_image_rows = []
    false_rows = []
    start = time.time()
    model.eval()
    if head is not None:
        head.eval()
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = pad32(hazy)
            a0, _ = convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            density_target, need_target = target_maps(hazy, gt, a0, stats, args)

            density_pred, need_pred = predict_variant(model, head, padded, stats, variant, args)
            if density_pred is not None:
                density_pred = density_pred[:, :, :h, :w]
            if need_pred is not None:
                need_pred = need_pred[:, :, :h, :w]

            dt = downsample_values(density_target, args.metric_sample_size)
            nt = downsample_values(need_target, args.metric_sample_size)
            row = {"variant": variant, "name": name}
            if density_pred is not None:
                dp = downsample_values(density_pred, args.metric_sample_size)
                density_target_values.append(dt)
                density_pred_values.append(dp)
                row["density_pearson"] = pearson(dp, dt)
                row["density_mae"] = float(np.mean(np.abs(dp - dt)))
            else:
                row["density_pearson"] = math.nan
                row["density_mae"] = math.nan
            if need_pred is not None:
                npred = downsample_values(need_pred, args.metric_sample_size)
                need_target_values.append(nt)
                need_pred_values.append(npred)
                row["need_pearson"] = pearson(npred, nt)
                row["need_mae"] = float(np.mean(np.abs(npred - nt)))
            else:
                row["need_pearson"] = math.nan
                row["need_mae"] = math.nan
            if density_pred is not None and need_pred is not None:
                dp_full = downsample_values(density_pred, args.metric_sample_size)
                np_full = downsample_values(need_pred, args.metric_sample_size)
                low_mask = (dt <= float(stats["density"]["q33"])) & (nt <= float(stats["need"]["q33"]))
                strong_mask = (dp_full >= args.strong_pred_threshold) & (
                    np_full >= args.strong_pred_threshold
                )
                false_rate = float(np.mean(strong_mask[low_mask])) if int(low_mask.sum()) else math.nan
                row["low_haze_false_strong_rate"] = false_rate
                false_rows.append(
                    {
                        "variant": variant,
                        "name": name,
                        "low_haze_pixels": int(low_mask.sum()),
                        "false_strong_pixels": int((strong_mask & low_mask).sum()),
                        "false_strong_rate": false_rate,
                    }
                )
            else:
                row["low_haze_false_strong_rate"] = math.nan
            per_image_rows.append(row)
            if (idx + 1) % args.progress_every == 0:
                print(f"eval {variant} {idx + 1}/{len(dataset)}", flush=True)

    summary = {"variant": variant, "eval_images": len(dataset), "elapsed_sec": time.time() - start}
    density_bins = []
    need_bins = []
    if density_target_values:
        d_target = np.concatenate(density_target_values)
        d_pred = np.concatenate(density_pred_values)
        summary.update({f"density_{k}": v for k, v in summarize_signal(
            d_target,
            d_pred,
            float(stats["density"]["q33"]),
            float(stats["density"]["q66"]),
            args.strong_pred_threshold,
        ).items()})
        rows, mono, valid = bin_means(
            d_target,
            d_pred,
            [
                float(stats["density"]["q20"]),
                float(stats["density"]["q33"]),
                float(stats["density"]["q66"]),
                float(stats["density"]["q80"]),
            ],
        )
        summary["density_monotonic_pairs"] = mono
        summary["density_monotonic_valid_pairs"] = valid
        for r in rows:
            r["variant"] = variant
            density_bins.append(r)
    if need_target_values:
        n_target = np.concatenate(need_target_values)
        n_pred = np.concatenate(need_pred_values)
        summary.update({f"need_{k}": v for k, v in summarize_signal(
            n_target,
            n_pred,
            float(stats["need"]["q33"]),
            float(stats["need"]["q66"]),
            args.strong_pred_threshold,
        ).items()})
        rows, mono, valid = bin_means(
            n_target,
            n_pred,
            [
                float(stats["need"]["q20"]),
                float(stats["need"]["q33"]),
                float(stats["need"]["q66"]),
                float(stats["need"]["q80"]),
            ],
        )
        summary["need_monotonic_pairs"] = mono
        summary["need_monotonic_valid_pairs"] = valid
        for r in rows:
            r["variant"] = variant
            need_bins.append(r)
    if false_rows:
        vals = [r["false_strong_rate"] for r in false_rows if not math.isnan(r["false_strong_rate"])]
        summary["low_haze_false_strong_rate"] = statistics.mean(vals) if vals else math.nan
    return summary, density_bins, need_bins, per_image_rows, false_rows


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gate_pass(summary):
    d_ok = (
        summary.get("density_pearson", math.nan) >= 0.25
        and summary.get("density_spearman", math.nan) >= 0.30
        and summary.get("density_auroc_high_vs_low", math.nan) >= 0.65
        and summary.get("density_monotonic_pairs", 0) >= 4
        and MIN_PRED_HIGH_COVERAGE
        <= summary.get("density_pred_high_coverage", 0.0)
        <= MAX_PRED_HIGH_COVERAGE
    )
    n_ok = (
        summary.get("need_pearson", math.nan) >= 0.20
        and summary.get("need_spearman", math.nan) >= 0.25
        and summary.get("need_auroc_high_vs_low", math.nan) >= 0.65
        and summary.get("need_monotonic_pairs", 0) >= 4
        and MIN_PRED_HIGH_COVERAGE
        <= summary.get("need_pred_high_coverage", 0.0)
        <= MAX_PRED_HIGH_COVERAGE
    )
    false_ok = summary.get("low_haze_false_strong_rate", 1.0) <= 0.10
    return bool(d_ok and n_ok and false_ok)


def write_failure_cases(output_dir, per_image_rows):
    ranked = sorted(
        [r for r in per_image_rows if r["variant"] in ("d1_dual", "d5_shuffled_dual")],
        key=lambda r: (
            -1 if math.isnan(float(r.get("density_pearson", math.nan))) else float(r.get("density_pearson", 0.0))
        ),
    )[:20]
    lines = [
        "# CHD-RM v2 Failure Case Notes",
        "",
        "This file lists lowest per-image density correlation cases from the current v2 run.",
        "Raw images are not committed; names point back to the Haze4K train split.",
        "",
        "| variant | name | density_pearson | need_pearson | low_haze_false_strong_rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['variant']} | {row['name']} | {row.get('density_pearson', math.nan)} | "
            f"{row.get('need_pearson', math.nan)} | {row.get('low_haze_false_strong_rate', math.nan)} |"
        )
    (output_dir / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(output_dir, summaries):
    by_variant = {row["variant"]: row for row in summaries}
    d1 = by_variant.get("d1_dual", {})
    d5 = by_variant.get("d5_shuffled_dual", {})
    d1_pass = gate_pass(d1) if d1 else False
    d5_pass = gate_pass(d5) if d5 else False
    if d1_pass and not d5_pass:
        decision = "COMPLETED_V2_GATE_PASS_PAUSE_BEFORE_V3"
        next_step = "Pause before v3 no-op RARM audit branch creation."
    elif not d1_pass and not d5_pass:
        decision = "PAUSE_V2_DUAL_HEAD_NOT_PASSED"
        next_step = "Inspect D3/D4 single-head evidence; run D2 only if single-head learnability is clear."
    else:
        decision = "PAUSE_V2_CONTROL_INVALID"
        next_step = "Do not proceed; shuffled target control did not fail."
    lines = [
        "# CHD-RM v2 Decision Record",
        "",
        f"Decision: `{decision}`",
        "",
        "Gate contract:",
        "",
        "- density Pearson >= 0.25, Spearman >= 0.30, AUROC >= 0.65, 4/4 monotonic pairs.",
        "- need Pearson >= 0.20, Spearman >= 0.25, AUROC >= 0.65, 4/4 monotonic pairs.",
        "- density and need strong-response coverage must be non-degenerate: 0.01 to 0.90.",
        "- low-haze false-strong-recovery rate <= 0.10.",
        "- shuffled target control must not pass the same gate.",
        "",
        f"Next step: {next_step}",
        "",
        "Locked Haze4K test usage: none.",
        "",
    ]
    (output_dir / "decision_record.md").write_text("\n".join(lines), encoding="utf-8")
    return {"decision": decision, "d1_gate_pass": d1_pass, "d5_gate_pass": d5_pass, "next_step": next_step}


def write_readme(output_dir, result):
    lines = [
        "# CHD-RM v2 Density-Need Calibration Evidence",
        "",
        f"Status: `{result['decision']}`",
        "",
        "Scope:",
        "",
        "- Route: CHD-RM v5.",
        "- Stage: v2 density/need calibration.",
        "- Dataset view: Haze4K train split only, using v1 train_inner/val_inner split.",
        "- Locked Haze4K test: not used.",
        "- Baseline: official ConvIR-B Haze4K checkpoint.",
        "",
        "Executed variants:",
        "",
        "- V2-D0: handcrafted dark-channel density proxy.",
        "- V2-D1: frozen ConvIR-B res1 feature plus dual density/need head.",
        "- V2-D3: density-only head.",
        "- V2-D4: need-only head.",
        "- V2-D5: shuffled-target dual-head control.",
        "",
        "Compact evidence files in this directory are text-only. Head checkpoints under `artifacts/` are cloud-only and ignored by Git.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--need_weight", type=float, default=1.0)
    ap.add_argument("--bce_weight", type=float, default=0.2)
    ap.add_argument("--tv_weight", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--dark_kernel", type=int, default=15)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--strong_pred_threshold", type=float, default=0.66)
    ap.add_argument("--threshold_limit", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--variants", nargs="*", default=["d0_handcrafted", "d1_dual", "d3_density_only", "d4_need_only", "d5_shuffled_dual"])
    args = ap.parse_args()

    if args.blur_kernel % 2 != 1 or args.dark_kernel % 2 != 1:
        raise ValueError("blur kernels must be odd")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    (output_dir / "artifacts" / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    train_names = load_split_names(args.split_json, "train_inner")
    val_names = load_split_names(args.split_json, "val_inner")
    if len(train_names) != 2400 or len(val_names) != 600:
        raise ValueError(f"Expected 2400/600 split, got {len(train_names)}/{len(val_names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_state(Path(args.checkpoint), device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    stats = collect_thresholds(model, train_names, args.data_dir, device, args)
    (output_dir / "density_need_thresholds.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    train_rows = []
    summaries = []
    density_bin_rows = []
    need_bin_rows = []
    per_image_rows = []
    false_rows = []
    heads = {}
    for variant in args.variants:
        head = None
        if variant != "d0_handcrafted":
            head, rows, ckpt_path = train_variant(
                model, train_names, args.data_dir, device, stats, variant, output_dir, args
            )
            for row in rows:
                row["checkpoint_path"] = str(ckpt_path)
            train_rows.extend(rows)
            heads[variant] = head
        summary, d_bins, n_bins, per_rows, false_variant_rows = evaluate_variant(
            model, head, val_names, args.data_dir, device, stats, variant, output_dir, args
        )
        summaries.append(summary)
        density_bin_rows.extend(d_bins)
        need_bin_rows.extend(n_bins)
        per_image_rows.extend(per_rows)
        false_rows.extend(false_variant_rows)
        print(json.dumps(summary), flush=True)

    write_csv(output_dir / "density_need_train_log.csv", train_rows)
    density_summaries = [row for row in summaries if any(k.startswith("density_") for k in row)]
    need_summaries = [row for row in summaries if any(k.startswith("need_") for k in row)]
    write_csv(output_dir / "density_calibration_summary.csv", density_summaries)
    write_csv(output_dir / "need_calibration_summary.csv", need_summaries)
    write_csv(output_dir / "density_calibration_bins.csv", density_bin_rows)
    write_csv(output_dir / "need_calibration_bins.csv", need_bin_rows)
    write_csv(output_dir / "density_need_per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "false_strong_recovery_audit.csv", false_rows)
    write_failure_cases(output_dir, per_image_rows)
    decision = write_decision(output_dir, summaries)
    write_readme(output_dir, decision)
    (output_dir / "v2_run_summary.json").write_text(
        json.dumps({"decision": decision, "summaries": summaries, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
