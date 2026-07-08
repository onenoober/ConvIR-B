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

V2_TOOL = REPO_ROOT / "experience_docx" / "tools" / "run_chd_rm_v2_density_need_calibration.py"
spec = importlib.util.spec_from_file_location("chdrm_v2_tool", V2_TOOL)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
STRONG_PRED_THRESHOLD = 0.66
MIN_STRONG_COVERAGE = 0.01
MAX_STRONG_COVERAGE = 0.90


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
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            hazy = hazy[:, top : top + self.crop_size, left : left + self.crop_size]
            gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]
        return name, hazy, gt


def collate_pairs(batch):
    names, hazy, gt = zip(*batch)
    return list(names), torch.stack(hazy, 0), torch.stack(gt, 0)


def load_split_names(split_json, split_name):
    data = json.loads(Path(split_json).read_text(encoding="utf-8"))
    return sorted(data["splits"][split_name])


def percentile(values, pct):
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def rankdata_average(values):
    return v2.rankdata_average(np.asarray(values))


def pearson(x, y):
    return v2.pearson(x, y)


def spearman(x, y):
    return v2.spearman(x, y)


def auroc(labels, scores):
    return v2.auroc(labels, scores)


def auprc(labels, scores):
    return v2.auprc(labels, scores)


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


def load_model(checkpoint, device):
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(v2.load_state(Path(checkpoint), device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def sample_values(x, sample_size):
    if x.shape[-2] != sample_size or x.shape[-1] != sample_size:
        x = F.adaptive_avg_pool2d(x, (sample_size, sample_size))
    return x.detach().flatten().float().cpu().numpy()


def collect_raw_need_density(model, names, data_dir, device, args, limit=0):
    dataset = Haze4KPairDataset(names, data_dir, max_items=limit, seed=args.seed)
    raw_need_values = []
    raw_density_values = []
    image_rows = []
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2.pad32(hazy)
            a0, _ = v2.convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            raw_need = v2.raw_need(a0, gt, args.blur_kernel)
            raw_density = v2.raw_density(hazy, gt, args.blur_kernel)
            n = sample_values(raw_need, args.metric_sample_size)
            d = sample_values(raw_density, args.metric_sample_size)
            raw_need_values.append(n)
            raw_density_values.append(d)
            image_rows.append(
                {
                    "name": name,
                    "raw_need_mean": float(np.mean(n)),
                    "raw_need_max_sampled": float(np.max(n)),
                    "raw_density_mean": float(np.mean(d)),
                    "raw_density_max_sampled": float(np.max(d)),
                }
            )
            if (idx + 1) % args.progress_every == 0:
                print(f"audit_raw {idx + 1}/{len(dataset)}", flush=True)
    return np.concatenate(raw_need_values), np.concatenate(raw_density_values), image_rows


def build_target_info(train_raw_need, train_raw_density):
    need_sorted = np.sort(train_raw_need.astype(np.float64))
    density_sorted = np.sort(train_raw_density.astype(np.float64))
    q_grid = np.linspace(0.0, 1.0, 1001)
    need_quantile_raw = np.quantile(need_sorted, q_grid)
    raw_need_median = percentile(train_raw_need, 50)
    log_values = np.log1p(train_raw_need / max(raw_need_median, 1e-8))
    log_p1 = percentile(log_values, 1)
    log_p99 = percentile(log_values, 99)
    log_norm = np.clip((log_values - log_p1) / max(log_p99 - log_p1, 1e-8), 0.0, 1.0)
    return {
        "raw_need": {
            f"p{p}": percentile(train_raw_need, p)
            for p in [0, 1, 5, 10, 20, 33, 50, 66, 80, 90, 95, 99, 99.5, 100]
        },
        "raw_density": {
            f"p{p}": percentile(train_raw_density, p)
            for p in [0, 1, 5, 10, 20, 33, 50, 66, 80, 90, 95, 99, 99.5, 100]
        },
        "quantile": {
            "raw_grid": need_quantile_raw.tolist(),
            "q20": 0.20,
            "q33": 0.33,
            "q66": 0.66,
            "q80": 0.80,
        },
        "log": {
            "raw_need_median": raw_need_median,
            "log_p1": log_p1,
            "log_p99": log_p99,
            "q20": percentile(log_norm, 20),
            "q33": percentile(log_norm, 33),
            "q66": percentile(log_norm, 66),
            "q80": percentile(log_norm, 80),
        },
    }


def quantile_target(raw_need, target_info):
    grid = torch.as_tensor(
        target_info["quantile"]["raw_grid"],
        device=raw_need.device,
        dtype=raw_need.dtype,
    )
    idx = torch.bucketize(raw_need.contiguous(), grid)
    idx = idx.clamp(0, len(target_info["quantile"]["raw_grid"]) - 1)
    return idx.to(raw_need.dtype) / float(len(target_info["quantile"]["raw_grid"]) - 1)


def log_target(raw_need, target_info):
    median = max(float(target_info["log"]["raw_need_median"]), 1e-8)
    lo = float(target_info["log"]["log_p1"])
    hi = float(target_info["log"]["log_p99"])
    x = torch.log1p(raw_need / median)
    return torch.clamp((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def make_target(raw_need, target_info, mode):
    if mode == "quantile":
        return quantile_target(raw_need, target_info)
    if mode == "log":
        return log_target(raw_need, target_info)
    raise ValueError(mode)


class ScalarNeedHead(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


def tv_loss(x):
    return v2.tv_loss(x)


def train_variant(model, train_names, data_dir, device, target_info, variant, output_dir, args):
    ordinal = variant == "d6c_ordinal_quantile"
    shuffled = variant == "d6s_shuffled_quantile"
    target_mode = "log" if variant == "d6b_log" else "quantile"
    out_channels = 4 if ordinal else 1
    head = ScalarNeedHead(out_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = Haze4KPairDataset(
        train_names, data_dir, crop_size=args.crop_size, max_items=args.train_limit, seed=args.seed
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
    rows = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        smooth_losses = []
        ord_losses = []
        for step, (_, hazy, gt) in enumerate(loader, start=1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            with torch.no_grad():
                a0, res1 = v2.convir_a0_and_res1(model, hazy)
                raw_need = v2.raw_need(a0, gt, args.blur_kernel)
                target = make_target(raw_need, target_info, target_mode)
                if shuffled:
                    shift = random.randint(1, hazy.shape[0] - 1)
                    target = torch.roll(target, shifts=shift, dims=0)
            logits = head(res1.detach())
            if ordinal:
                thresholds = [0.20, 0.33, 0.66, 0.80]
                ord_loss = logits.new_tensor(0.0)
                probs = []
                for i, thr in enumerate(thresholds):
                    label = (target >= thr).float()
                    ord_loss = ord_loss + bce(logits[:, i : i + 1], label)
                    probs.append(torch.sigmoid(logits[:, i : i + 1]))
                pred = torch.stack(probs, dim=0).mean(dim=0)
                smooth = F.smooth_l1_loss(pred, target)
                loss = smooth + args.ordinal_weight * ord_loss + args.tv_weight * tv_loss(pred)
                ord_losses.append(float(ord_loss.detach().item()))
            else:
                pred = torch.sigmoid(logits)
                high_thr = float(target_info[target_mode]["q66"])
                smooth = F.smooth_l1_loss(pred, target)
                loss = (
                    smooth
                    + args.bce_weight * bce(logits, (target >= high_thr).float())
                    + args.tv_weight * tv_loss(pred)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
            smooth_losses.append(float(smooth.detach().item()))
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
            "ordinal_bce_sum_mean": statistics.mean(ord_losses) if ord_losses else math.nan,
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
            "target_mode": target_mode,
            "ordinal": ordinal,
        },
        artifact_dir / f"{variant}_head.pt",
    )
    return head, rows


def predict(head, res1, variant):
    logits = head(res1)
    if variant == "d6c_ordinal_quantile":
        return torch.stack([torch.sigmoid(logits[:, i : i + 1]) for i in range(4)], dim=0).mean(dim=0)
    return torch.sigmoid(logits)


def bin_rows(target, pred, cuts, variant):
    rows, mono, valid = v2.bin_means(target, pred, cuts)
    for row in rows:
        row["variant"] = variant
    return rows, mono, valid


def summarize(target, pred, q33, q66):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    low_or_high = (target <= q33) | (target >= q66)
    high_label = target >= q66
    return {
        "need_pearson": pearson(pred, target),
        "need_spearman": spearman(pred, target),
        "need_auroc_high_vs_low": auroc(high_label[low_or_high], pred[low_or_high]),
        "need_auprc_high": auprc(high_label, pred),
        "need_pred_high_coverage": float(np.mean(pred >= STRONG_PRED_THRESHOLD)),
        "need_target_high_coverage": float(np.mean(target >= q66)),
    }


def evaluate_variant(model, head, val_names, data_dir, device, target_info, density_stats, variant, output_dir, args):
    target_mode = "log" if variant == "d6b_log" else "quantile"
    q20 = float(target_info[target_mode]["q20"])
    q33 = float(target_info[target_mode]["q33"])
    q66 = float(target_info[target_mode]["q66"])
    q80 = float(target_info[target_mode]["q80"])
    dataset = Haze4KPairDataset(val_names, data_dir, max_items=args.val_limit, seed=args.seed)
    target_values = []
    pred_values = []
    per_rows = []
    false_rows = []
    with torch.no_grad():
        for idx, (name, hazy, gt) in enumerate(dataset):
            hazy = hazy.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            padded, h, w = v2.pad32(hazy)
            a0, res1 = v2.convir_a0_and_res1(model, padded)
            a0 = a0[:, :, :h, :w]
            res1 = res1[:, :, :h, :w]
            raw_need = v2.raw_need(a0, gt, args.blur_kernel)
            target = make_target(raw_need, target_info, target_mode)
            pred = predict(head, res1, variant)
            density_target = v2.normalize(
                v2.raw_density(hazy, gt, args.blur_kernel),
                density_stats["density"]["raw_p1"],
                density_stats["density"]["raw_p99"],
            )
            tv = sample_values(target, args.metric_sample_size)
            pv = sample_values(pred, args.metric_sample_size)
            dv = sample_values(density_target, args.metric_sample_size)
            target_values.append(tv)
            pred_values.append(pv)
            low_context = (dv <= float(density_stats["density"]["q33"])) & (tv <= q33)
            false_strong = pv >= STRONG_PRED_THRESHOLD
            false_rate = float(np.mean(false_strong[low_context])) if int(low_context.sum()) else math.nan
            per_rows.append(
                {
                    "variant": variant,
                    "name": name,
                    "target_mode": target_mode,
                    "need_pearson": pearson(pv, tv),
                    "need_mae": float(np.mean(np.abs(pv - tv))),
                    "need_target_mean": float(np.mean(tv)),
                    "need_pred_mean": float(np.mean(pv)),
                    "need_target_max": float(np.max(tv)),
                    "need_pred_max": float(np.max(pv)),
                    "low_context_false_strong_rate": false_rate,
                }
            )
            false_rows.append(
                {
                    "variant": variant,
                    "name": name,
                    "low_context_pixels": int(low_context.sum()),
                    "false_strong_pixels": int((false_strong & low_context).sum()),
                    "false_strong_rate": false_rate,
                }
            )
            if (idx + 1) % args.progress_every == 0:
                print(f"eval {variant} {idx + 1}/{len(dataset)}", flush=True)
    target_all = np.concatenate(target_values)
    pred_all = np.concatenate(pred_values)
    summary = {
        "variant": variant,
        "target_mode": target_mode,
        "eval_images": len(dataset),
        **summarize(target_all, pred_all, q33, q66),
    }
    bins, mono, valid = bin_rows(target_all, pred_all, [q20, q33, q66, q80], variant)
    summary["need_monotonic_pairs"] = mono
    summary["need_monotonic_valid_pairs"] = valid
    vals = [r["false_strong_rate"] for r in false_rows if not math.isnan(r["false_strong_rate"])]
    summary["low_context_false_strong_rate"] = statistics.mean(vals) if vals else math.nan
    return summary, bins, per_rows, false_rows


def gate_pass(summary):
    return bool(
        summary.get("need_pearson", math.nan) >= 0.20
        and summary.get("need_spearman", math.nan) >= 0.25
        and summary.get("need_auroc_high_vs_low", math.nan) >= 0.65
        and summary.get("need_monotonic_pairs", 0) >= 4
        and MIN_STRONG_COVERAGE
        <= summary.get("need_pred_high_coverage", 0.0)
        <= MAX_STRONG_COVERAGE
        and summary.get("low_context_false_strong_rate", 1.0) <= 0.10
    )


def write_target_audit(output_dir, target_info, train_rows, val_rows, val_a0_csv):
    dist_rows = []
    for family in ["raw_need", "raw_density"]:
        row = {"family": family}
        row.update(target_info[family])
        dist_rows.append(row)
    write_csv(output_dir / "need_target_distribution.csv", dist_rows)
    thresholds = {
        "raw_need_q66": target_info["raw_need"]["p66"],
        "raw_need_q80": target_info["raw_need"]["p80"],
        "raw_need_abs_0_66": 0.66,
        "quantile_q66": 0.66,
        "log_q66": target_info["log"]["q66"],
    }
    coverage_rows = []
    for split, rows in [("train_inner", train_rows), ("val_inner", val_rows)]:
        raw_need = np.asarray([r["raw_need_mean"] for r in rows], dtype=np.float64)
        raw_density = np.asarray([r["raw_density_mean"] for r in rows], dtype=np.float64)
        coverage_rows.append(
            {
                "split": split,
                "image_count": len(rows),
                "mean_raw_need_mean": float(np.mean(raw_need)),
                "mean_raw_density_mean": float(np.mean(raw_density)),
                "image_mean_need_ge_raw_q66": float(np.mean(raw_need >= thresholds["raw_need_q66"])),
                "image_mean_need_ge_raw_q80": float(np.mean(raw_need >= thresholds["raw_need_q80"])),
                "image_mean_need_ge_abs_0_66": float(np.mean(raw_need >= 0.66)),
            }
        )
    write_csv(output_dir / "need_target_spatial_coverage.csv", coverage_rows)
    relation_rows = []
    for split, rows in [("train_inner", train_rows), ("val_inner", val_rows)]:
        need = np.asarray([r["raw_need_mean"] for r in rows], dtype=np.float64)
        density = np.asarray([r["raw_density_mean"] for r in rows], dtype=np.float64)
        relation_rows.append(
            {
                "split": split,
                "image_count": len(rows),
                "pearson_need_density": pearson(need, density),
                "spearman_need_density": spearman(need, density),
            }
        )
    write_csv(output_dir / "need_density_relation.csv", relation_rows)
    a0_rows = []
    if val_a0_csv and Path(val_a0_csv).is_file():
        a0_by_name = {r["name"]: r for r in csv.DictReader(Path(val_a0_csv).open())}
        joined = []
        for r in val_rows:
            a = a0_by_name.get(r["name"])
            if a:
                joined.append({**r, "a0_psnr": float(a["a0_psnr"]), "a0_ssim": float(a["a0_ssim"])})
        if joined:
            need = np.asarray([r["raw_need_mean"] for r in joined], dtype=np.float64)
            psnr = np.asarray([r["a0_psnr"] for r in joined], dtype=np.float64)
            ssim = np.asarray([r["a0_ssim"] for r in joined], dtype=np.float64)
            a0_rows.append(
                {
                    "split": "val_inner",
                    "count": len(joined),
                    "pearson_need_mean_vs_a0_psnr": pearson(need, psnr),
                    "spearman_need_mean_vs_a0_psnr": spearman(need, psnr),
                    "pearson_need_mean_vs_a0_ssim": pearson(need, ssim),
                    "spearman_need_mean_vs_a0_ssim": spearman(need, ssim),
                }
            )
    write_csv(output_dir / "need_a0_error_relation.csv", a0_rows)
    summary = {
        "raw_need_p99": target_info["raw_need"]["p99"],
        "raw_need_median": target_info["raw_need"]["p50"],
        "log_q66": target_info["log"]["q66"],
        "quantile_strong_threshold": 0.66,
        "audit_interpretation": "raw need has low absolute dynamic range; quantile/log transforms are tested before any D2 or RARM connection.",
    }
    (output_dir / "need_target_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_decision(output_dir, summaries):
    by_variant = {s["variant"]: s for s in summaries}
    control = by_variant.get("d6s_shuffled_quantile", {})
    passing = [s for s in summaries if s["variant"] != "d6s_shuffled_quantile" and gate_pass(s)]
    control_pass = gate_pass(control) if control else False
    if passing and not control_pass:
        decision = "COMPLETED_V2B_NEED_REPAIR_GATE_PASS_PAUSE_BEFORE_V3"
        next_step = "Run one narrow confirmation, then v3 no-op RARM audit only after written approval."
    elif control_pass:
        decision = "PAUSE_V2B_CONTROL_INVALID"
        next_step = "Do not proceed; repair control passed and metric contract must be audited."
    else:
        decision = "PAUSE_V2B_NEED_REPAIR_NOT_PASSED"
        next_step = "Do not run D2 yet unless a near-pass variant shows non-degenerate coverage and control failure."
    lines = [
        "# CHD-RM v2b Decision Record",
        "",
        f"Decision: `{decision}`",
        "",
        f"Next step: {next_step}",
        "",
        "Locked Haze4K test usage: none.",
        "",
    ]
    (output_dir / "decision_record.md").write_text("\n".join(lines), encoding="utf-8")
    return {"decision": decision, "passing_variants": [s["variant"] for s in passing], "control_pass": control_pass, "next_step": next_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--v2_thresholds", required=True)
    ap.add_argument("--v1_a0_per_image_csv", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--bce_weight", type=float, default=0.2)
    ap.add_argument("--ordinal_weight", type=float, default=0.25)
    ap.add_argument("--tv_weight", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--blur_kernel", type=int, default=9)
    ap.add_argument("--metric_sample_size", type=int, default=64)
    ap.add_argument("--audit_limit", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--variants", nargs="*", default=["d6a_quantile", "d6b_log", "d6c_ordinal_quantile", "d6s_shuffled_quantile"])
    args = ap.parse_args()

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
    model = load_model(args.checkpoint, device)
    density_stats = json.loads(Path(args.v2_thresholds).read_text(encoding="utf-8"))

    train_raw_need, train_raw_density, train_audit_rows = collect_raw_need_density(
        model, train_names, args.data_dir, device, args, limit=args.audit_limit
    )
    val_raw_need, val_raw_density, val_audit_rows = collect_raw_need_density(
        model, val_names, args.data_dir, device, args, limit=args.val_limit
    )
    target_info = build_target_info(train_raw_need, train_raw_density)
    (output_dir / "need_thresholds_v2b.json").write_text(json.dumps(target_info, indent=2), encoding="utf-8")
    write_target_audit(output_dir, target_info, train_audit_rows, val_audit_rows, args.v1_a0_per_image_csv)
    (output_dir / "need_target_transform_definitions.md").write_text(
        "# CHD-RM v2b Need Target Transform Definitions\n\n"
        "D6a quantile target maps raw need through the train_inner empirical CDF.\n"
        "D6b log target uses log1p(raw_need / train_median_raw_need), then train_inner p1/p99 normalization.\n"
        "D6c ordinal target uses the quantile target and BCE labels at q20/q33/q66/q80.\n"
        "All statistics are computed from train_inner only. Locked Haze4K test is not used.\n",
        encoding="utf-8",
    )

    train_rows = []
    summaries = []
    bin_rows = []
    per_rows = []
    false_rows = []
    for variant in args.variants:
        head, rows = train_variant(model, train_names, args.data_dir, device, target_info, variant, output_dir, args)
        train_rows.extend(rows)
        summary, bins, per, false = evaluate_variant(
            model, head, val_names, args.data_dir, device, target_info, density_stats, variant, output_dir, args
        )
        summaries.append(summary)
        bin_rows.extend(bins)
        per_rows.extend(per)
        false_rows.extend(false)
        print(json.dumps(summary), flush=True)

    write_csv(output_dir / "need_repair_train_log.csv", train_rows)
    write_csv(output_dir / "need_repair_calibration_summary.csv", summaries)
    write_csv(output_dir / "need_repair_calibration_bins.csv", bin_rows)
    write_csv(output_dir / "need_repair_per_image_metrics.csv", per_rows)
    write_csv(output_dir / "need_repair_false_strong_audit.csv", false_rows)
    hist_rows = []
    for s in summaries:
        hist_rows.append(
            {
                "variant": s["variant"],
                "target_mode": s["target_mode"],
                "need_pred_high_coverage": s["need_pred_high_coverage"],
                "need_target_high_coverage": s["need_target_high_coverage"],
            }
        )
    write_csv(output_dir / "need_repair_prediction_histogram.csv", hist_rows)
    decision = write_decision(output_dir, summaries)
    (output_dir / "v2b_run_summary.json").write_text(
        json.dumps({"decision": decision, "summaries": summaries, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    result_lines = [
        "# CHD-RM v2b Need Calibration Repair Summary",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "| Variant | Target | Pearson | Spearman | AUROC | Pred high coverage | Monotonic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        result_lines.append(
            f"| {s['variant']} | {s['target_mode']} | {s['need_pearson']:.4f} | "
            f"{s['need_spearman']:.4f} | {s['need_auroc_high_vs_low']:.4f} | "
            f"{s['need_pred_high_coverage']:.4f} | {s['need_monotonic_pairs']}/{s['need_monotonic_valid_pairs']} |"
        )
    result_lines += ["", "Locked Haze4K test usage: none.", ""]
    (output_dir / "v2b_result_summary.md").write_text("\n".join(result_lines), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# CHD-RM v2b Need Calibration Repair Evidence\n\n"
        f"Status: `{decision['decision']}`\n\n"
        "This stage audits and repairs R_need target calibration while keeping ConvIR-B frozen and leaving the dehazing output unchanged.\n"
        "Start with `v2b_result_summary.md`, `decision_record.md`, and `v2b_run_summary.json`.\n\n"
        "Locked Haze4K test usage: none.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
