#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import build_d7c_gate_producer  # noqa: E402
from models.ConvIR import build_net  # noqa: E402
from train import _configure_train_scope  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3e_matched_utility_mechanism_audit_20260710"
V3D_ROUTE_ID = "haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
D7C_THRESHOLD = 0.5773006677627563


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnum(value, default=float("nan")):
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return float(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values),
        "min": min(values) if values else float("nan"),
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else float("nan"),
    }


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x_vals, y_vals = zip(*pairs)
    mx = mean(x_vals)
    my = mean(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def auprc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def bootstrap_ci(values, rng, iterations):
    values = list(values)
    n = len(values)
    samples = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        samples.append(total / n)
    return [percentile(samples, 2.5), percentile(samples, 50), percentile(samples, 97.5)]


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [hazy_name, f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"]
    label_dir = Path(data_dir) / source_split / "gt"
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT match for {hazy_name}; tried {candidates} under {label_dir}")


def load_pair(data_dir, source_split, hazy_name):
    hazy_path = Path(data_dir) / source_split / "haze" / hazy_name
    label_path = label_path_for_hazy(data_dir, source_split, hazy_name)
    if not hazy_path.is_file():
        raise FileNotFoundError(f"Hazy image missing: {hazy_path}")
    image = Image.open(hazy_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    return TF.to_tensor(image), TF.to_tensor(label)


def center_crop_pair(input_tensor, label_tensor, crop_size):
    if crop_size <= 0:
        return input_tensor, label_tensor
    _, height, width = input_tensor.shape
    if height < crop_size or width < crop_size:
        raise ValueError(f"Image too small for crop_size={crop_size}: {height}x{width}")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return (
        input_tensor[:, top : top + crop_size, left : left + crop_size],
        label_tensor[:, top : top + crop_size, left : left + crop_size],
    )


def load_names(split_json, split_key, max_samples):
    split = read_json(split_json)
    names = sorted(split["splits"][split_key])
    if max_samples > 0:
        names = names[:max_samples]
    if not names:
        raise ValueError(f"No samples for split {split_key}")
    return names


def pad_to_factor(input_img, factor=32):
    height, width = input_img.shape[2], input_img.shape[3]
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    pad_height = padded_height - height if height % factor != 0 else 0
    pad_width = padded_width - width if width % factor != 0 else 0
    if pad_height or pad_width:
        input_img = F.pad(input_img, (0, pad_width, 0, pad_height), "reflect")
    return input_img, height, width


def pad_batch_pair_to_factor(input_img, label_img, factor=32):
    height, width = input_img.shape[2], input_img.shape[3]
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    pad_height = padded_height - height if height % factor != 0 else 0
    pad_width = padded_width - width if width % factor != 0 else 0
    if pad_height or pad_width:
        input_img = F.pad(input_img, (0, pad_width, 0, pad_height), "reflect")
        label_img = F.pad(label_img, (0, pad_width, 0, pad_height), "reflect")
    return input_img, label_img


def load_checkpoint_state(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def metric_summary(pred, label):
    pred = torch.clamp(pred, 0.0, 1.0)
    label = torch.clamp(label, 0.0, 1.0)
    mse = F.mse_loss(pred, label)
    psnr = float("inf") if mse.item() == 0 else (10.0 * torch.log10(1.0 / mse)).item()
    height, width = pred.shape[2], pred.shape[3]
    down_ratio = max(1, round(min(height, width) / 256))
    pooled_pred = F.adaptive_avg_pool2d(pred, (int(height / down_ratio), int(width / down_ratio)))
    pooled_label = F.adaptive_avg_pool2d(label, (int(height / down_ratio), int(width / down_ratio)))
    ssim_val = ssim(pooled_pred, pooled_label, data_range=1, size_average=False).mean().item()
    return psnr, ssim_val


class D7CBuildArgs:
    pass


def build_gate_producer(args, device):
    build_args = D7CBuildArgs()
    build_args.version = "base"
    build_args.data = "Haze4K"
    build_args.fam_mode = "fam2_d7c_noop"
    build_args.d7c_gate_mode = "d7c_fixed"
    build_args.d7c_base_checkpoint = args.a0_checkpoint
    build_args.d7c_density_artifact = args.density_artifact
    build_args.d7c_need_artifact = args.d7c_artifact
    build_args.d7c_threshold = args.d7c_threshold
    return build_d7c_gate_producer(build_args, device)


def build_candidate(fam_mode, checkpoint_path, device):
    model = build_net("base", "Haze4K", fam_mode).to(device).eval()
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device), strict=True)
    return model


def qbin(values, idx):
    cuts = [percentile(values, 25), percentile(values, 50), percentile(values, 75)]
    v = values[idx]
    if v <= cuts[0]:
        return 1
    if v <= cuts[1]:
        return 2
    if v <= cuts[2]:
        return 3
    return 4


def paired_reanalysis(args):
    old = Path(args.old_logdir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d7 = read_csv(old / "v3d_stage1_5epoch_val_inner_per_image.csv")
    control = read_csv(old / "v3d_fam2modres_control_5epoch_val_inner_per_image.csv")
    control_by = {row["name"]: row for row in control}
    missing = [row["name"] for row in d7 if row["name"] not in control_by]
    if missing:
        raise ValueError(f"Missing control rows: {missing[:5]}")
    rows = []
    for row in d7:
        c = control_by[row["name"]]
        d7_delta = fnum(row["psnr_delta"])
        control_delta = fnum(c["psnr_delta"])
        pair = d7_delta - control_delta
        rows.append(
            {
                "name": row["name"],
                "base_psnr": fnum(row["base_psnr"]),
                "gate_coverage": fnum(row["gate_coverage"]),
                "d7c_psnr_delta": d7_delta,
                "control_psnr_delta": control_delta,
                "d7c_minus_control_psnr_delta": pair,
                "d7c_output_mean_abs_diff": fnum(row["output_mean_abs_diff"]),
                "control_output_mean_abs_diff": fnum(c["output_mean_abs_diff"]),
                "d7c_better": pair > 0,
                "d7c_regression_le_0p2": d7_delta <= -0.2,
                "control_regression_le_0p2": control_delta <= -0.2,
            }
        )
    pair = [r["d7c_minus_control_psnr_delta"] for r in rows]
    d7_delta = [r["d7c_psnr_delta"] for r in rows]
    control_delta = [r["control_psnr_delta"] for r in rows]
    gate = [r["gate_coverage"] for r in rows]
    base = [r["base_psnr"] for r in rows]
    rng = random.Random(args.seed)
    boot_mean = bootstrap_ci(pair, rng, args.bootstrap)
    tail_reductions = []
    wins = []
    n = len(rows)
    for _ in range(args.bootstrap):
        d7_reg = 0
        control_reg = 0
        win = 0
        for _ in range(n):
            j = rng.randrange(n)
            d7_reg += d7_delta[j] <= -0.2
            control_reg += control_delta[j] <= -0.2
            win += pair[j] > 0
        tail_reductions.append(control_reg - d7_reg)
        wins.append(win / n)
    subgroup_rows = []
    for axis_name, vals in [("gate_coverage", gate), ("base_psnr", base)]:
        for bin_id in [1, 2, 3, 4]:
            idxs = [i for i in range(n) if qbin(vals, i) == bin_id]
            subgroup_rows.append(
                {
                    "axis": axis_name,
                    "bin": bin_id,
                    "n": len(idxs),
                    "axis_min": min(vals[i] for i in idxs),
                    "axis_max": max(vals[i] for i in idxs),
                    "d7c_mean": mean(d7_delta[i] for i in idxs),
                    "control_mean": mean(control_delta[i] for i in idxs),
                    "paired_mean": mean(pair[i] for i in idxs),
                    "d7c_win_rate": mean(1.0 if pair[i] > 0 else 0.0 for i in idxs),
                    "d7c_reg_le_0p2": sum(d7_delta[i] <= -0.2 for i in idxs),
                    "control_reg_le_0p2": sum(control_delta[i] <= -0.2 for i in idxs),
                }
            )
    categories = Counter()
    for a, b, p in zip(d7_delta, control_delta, pair):
        categories["d7c_better" if p > 0 else "control_better"] += 1
        categories["both_positive" if a >= 0 and b >= 0 else "both_negative" if a < 0 and b < 0 else "mixed"] += 1
        if a > -0.2 and b <= -0.2:
            categories["d7c_avoids_control_0p2_regression"] += 1
        if b > -0.2 and a <= -0.2:
            categories["control_avoids_d7c_0p2_regression"] += 1
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3e-A paired reanalysis",
        "source_v3d_logdir": str(old),
        "sample_count": n,
        "bootstrap_iterations": args.bootstrap,
        "d7c_delta": summarize(d7_delta),
        "control_delta": summarize(control_delta),
        "paired_d7c_minus_control": summarize(pair),
        "paired_mean_ci95": boot_mean,
        "tail_regression_reduction_ci95": [
            percentile(tail_reductions, 2.5),
            percentile(tail_reductions, 50),
            percentile(tail_reductions, 97.5),
        ],
        "d7c_win_rate_ci95": [
            percentile(wins, 2.5),
            percentile(wins, 50),
            percentile(wins, 97.5),
        ],
        "category_counts": dict(categories),
        "correlations": {
            "gate_vs_d7c_delta": pearson(gate, d7_delta),
            "gate_vs_control_delta": pearson(gate, control_delta),
            "gate_vs_d7c_minus_control": pearson(gate, pair),
            "base_psnr_vs_d7c_minus_control": pearson(base, pair),
        },
        "decision": (
            "V3E_A_SINGLE_SEED_MEAN_INCONCLUSIVE_TAIL_SAFETY_STABLE"
            if boot_mean[0] <= 0 <= boot_mean[2] and percentile(tail_reductions, 2.5) > 0
            else "V3E_A_REVIEW_REQUIRED"
        ),
        "locked_test_touched": False,
    }
    write_csv(out / "v3e_a_paired_reanalysis_per_image.csv", rows)
    write_csv(out / "v3e_a_paired_subgroup_summary.csv", subgroup_rows)
    write_json(out / "v3e_a_paired_reanalysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_variant(pred, base_pred, label):
    psnr_val, ssim_val = metric_summary(pred, label)
    base_psnr, base_ssim = metric_summary(base_pred, label)
    output_diff = (pred - base_pred).abs()
    return {
        "psnr": psnr_val,
        "psnr_delta": psnr_val - base_psnr,
        "ssim": ssim_val,
        "ssim_delta": ssim_val - base_ssim,
        "output_mean_abs_diff": output_diff.mean().item(),
        "output_max_abs_diff": output_diff.max().item(),
    }


def replay_2x2(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    base = build_candidate("original", args.a0_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    models = {
        "W_D_G_D": build_candidate("fam2_d7c_noop", args.d7c_checkpoint, device),
        "W_D_G_1": build_candidate("fam2_modres", args.d7c_checkpoint, device),
        "W_U_G_D": build_candidate("fam2_d7c_noop", args.control_checkpoint, device),
        "W_U_G_1": build_candidate("fam2_modres", args.control_checkpoint, device),
    }
    names = load_names(args.split_json, args.split_key, args.max_samples)
    rows = []
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            base_pred = base(padded)[2][:, :, :height, :width]
            gate, score, _ = gate_producer(padded)
            row = {"index": index, "name": name, "gate_coverage": gate.mean().item(), "score_mean": score.mean().item()}
            for variant, model in models.items():
                if variant.endswith("G_D"):
                    pred = model(padded, d7c_gate=gate)[2][:, :, :height, :width]
                else:
                    pred = model(padded)[2][:, :, :height, :width]
                stats = evaluate_variant(pred, base_pred, label)
                for key, value in stats.items():
                    row[f"{variant}_{key}"] = value
            rows.append(row)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"replay_progress {index + 1}/{len(names)}", flush=True)
    variants = ["W_D_G_D", "W_D_G_1", "W_U_G_D", "W_U_G_1"]
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3e-B weight x gate 2x2 replay",
        "sample_count": len(rows),
        "locked_test_touched": False,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "d7c_checkpoint": args.d7c_checkpoint,
        "d7c_checkpoint_sha256": sha256_file(args.d7c_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "variants": {},
    }
    for variant in variants:
        deltas = [r[f"{variant}_psnr_delta"] for r in rows]
        summary["variants"][variant] = {
            **summarize(deltas),
            "positive_ratio": mean(1.0 if v > 0 else 0.0 for v in deltas),
            "regression_le_0p2_count": sum(v <= -0.2 for v in deltas),
            "regression_le_0p5_count": sum(v <= -0.5 for v in deltas),
            "output_mean_abs_diff": mean(r[f"{variant}_output_mean_abs_diff"] for r in rows),
        }
    control_mean = summary["variants"]["W_U_G_1"]["mean"]
    wu_gd_mean = summary["variants"]["W_U_G_D"]["mean"]
    wd_gd_mean = summary["variants"]["W_D_G_D"]["mean"]
    wd_g1_mean = summary["variants"]["W_D_G_1"]["mean"]
    flags = {
        "ungated_weights_survive_d7c_gate": wu_gd_mean >= control_mean - args.noninferiority_margin,
        "hard_gate_restricts_d7c_weights": wd_g1_mean > wd_gd_mean + args.mean_gap_margin,
        "d7c_gate_drops_ungated_utility": wu_gd_mean < control_mean - args.mean_gap_margin,
        "gated_training_undertrains_operator": wd_g1_mean < control_mean - args.mean_gap_margin,
    }
    pair_rows = []
    for row in rows:
        pair_row = {"name": row["name"], "gate_coverage": row["gate_coverage"]}
        for a, b in [("W_D_G_D", "W_U_G_1"), ("W_U_G_D", "W_U_G_1"), ("W_D_G_1", "W_D_G_D"), ("W_D_G_1", "W_U_G_1")]:
            pair_row[f"{a}_minus_{b}"] = row[f"{a}_psnr_delta"] - row[f"{b}_psnr_delta"]
        pair_rows.append(pair_row)
    summary["mechanism_flags"] = flags
    summary["decision"] = "V3E_B_MECHANISM_REPLAY_DONE_REVIEW_FLAGS"
    write_csv(out / "v3e_b_weight_gate_2x2_per_image.csv", rows)
    write_csv(out / "v3e_b_weight_gate_2x2_pairwise.csv", pair_rows)
    write_json(out / "v3e_b_weight_gate_2x2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def erode(mask, radius):
    if radius <= 0:
        return mask
    inv = 1.0 - mask
    dil_inv = F.max_pool2d(inv, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (1.0 - dil_inv).clamp(0, 1)


def dilate(mask, radius):
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def masked_mean(tensor, mask):
    denom = mask.sum().item()
    if denom <= 0:
        return float("nan")
    return (tensor * mask).sum().item() / denom


def gain_alignment(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base = build_candidate("original", args.a0_checkpoint, device)
    d7c = build_candidate("fam2_d7c_noop", args.d7c_checkpoint, device)
    control = build_candidate("fam2_modres", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    names = load_names(args.split_json, args.split_key, args.max_samples)
    region_rows = []
    score_samples = []
    gain_u_samples = []
    gain_d_samples = []
    gate_samples = []
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            base_pred = torch.clamp(base(padded)[2][:, :, :height, :width], 0, 1)
            gate, score, _ = gate_producer(padded)
            d_pred = torch.clamp(d7c(padded, d7c_gate=gate)[2][:, :, :height, :width], 0, 1)
            u_pred = torch.clamp(control(padded)[2][:, :, :height, :width], 0, 1)
            label = label[:, :, :height, :width]
            base_err = (base_pred - label).abs().mean(dim=1, keepdim=True)
            d_gain = base_err - (d_pred - label).abs().mean(dim=1, keepdim=True)
            u_gain = base_err - (u_pred - label).abs().mean(dim=1, keepdim=True)
            score_map = score[:, :, : score.shape[-2], : score.shape[-1]]
            gate_map = gate[:, :, : gate.shape[-2], : gate.shape[-1]]
            u_gain_score = F.adaptive_avg_pool2d(u_gain, score_map.shape[-2:])
            d_gain_score = F.adaptive_avg_pool2d(d_gain, score_map.shape[-2:])
            flat_n = score_map.numel()
            take = min(args.pixel_sample_per_image, flat_n)
            choice = rng.choice(flat_n, size=take, replace=False)
            score_samples.append(score_map.detach().cpu().numpy().reshape(-1)[choice].astype(np.float32))
            gain_u_samples.append(u_gain_score.detach().cpu().numpy().reshape(-1)[choice].astype(np.float32))
            gain_d_samples.append(d_gain_score.detach().cpu().numpy().reshape(-1)[choice].astype(np.float32))
            gate_samples.append(gate_map.detach().cpu().numpy().reshape(-1)[choice].astype(np.uint8))
            out_gate = F.interpolate(gate, size=(height, width), mode="nearest")
            interior = erode(out_gate, 1)
            boundary1 = (dilate(out_gate, 1) - erode(out_gate, 1)).clamp(0, 1)
            exterior1 = (dilate(out_gate, 1) - out_gate).clamp(0, 1)
            exterior2 = (dilate(out_gate, 2) - dilate(out_gate, 1)).clamp(0, 1)
            exterior4 = (dilate(out_gate, 4) - dilate(out_gate, 2)).clamp(0, 1)
            exterior8 = (dilate(out_gate, 8) - dilate(out_gate, 4)).clamp(0, 1)
            exterior_far = (1.0 - dilate(out_gate, 8)).clamp(0, 1)
            d_change = (d_pred - base_pred).abs().mean(dim=1, keepdim=True)
            u_change = (u_pred - base_pred).abs().mean(dim=1, keepdim=True)
            for zone, mask in [
                ("gate_interior", interior),
                ("gate_boundary_r1", boundary1),
                ("exterior_r1", exterior1),
                ("exterior_r2", exterior2),
                ("exterior_r4", exterior4),
                ("exterior_r8", exterior8),
                ("exterior_far_gt8", exterior_far),
            ]:
                region_rows.append(
                    {
                        "index": index,
                        "name": name,
                        "zone": zone,
                        "pixel_count": int(mask.sum().item()),
                        "d7c_output_change_mean": masked_mean(d_change, mask),
                        "control_output_change_mean": masked_mean(u_change, mask),
                        "d7c_gain_l1_mean": masked_mean(d_gain, mask),
                        "control_gain_l1_mean": masked_mean(u_gain, mask),
                    }
                )
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"gain_progress {index + 1}/{len(names)}", flush=True)
    scores = np.concatenate(score_samples)
    gain_u = np.concatenate(gain_u_samples)
    gain_d = np.concatenate(gain_d_samples)
    gates = np.concatenate(gate_samples).astype(bool)
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3e-C operator gain alignment and boundary leakage",
        "sample_count": len(names),
        "pixel_sample_count": int(len(scores)),
        "pixel_sample_per_image": args.pixel_sample_per_image,
        "locked_test_touched": False,
        "score_vs_ungated_positive_gain": {
            "auroc": auroc(scores, gain_u > 0),
            "auprc": auprc(scores, gain_u > 0),
            "spearman_gain": spearman(scores, gain_u),
            "positive_gain_rate": float((gain_u > 0).mean()),
        },
        "score_vs_d7c_positive_gain": {
            "auroc": auroc(scores, gain_d > 0),
            "auprc": auprc(scores, gain_d > 0),
            "spearman_gain": spearman(scores, gain_d),
            "positive_gain_rate": float((gain_d > 0).mean()),
        },
        "hard_gate": {
            "coverage_sample": float(gates.mean()),
            "ungated_positive_gain_precision_in_gate": float((gain_u[gates] > 0).mean()) if gates.any() else float("nan"),
            "d7c_positive_gain_precision_in_gate": float((gain_d[gates] > 0).mean()) if gates.any() else float("nan"),
            "ungated_positive_gain_rate_outside_gate": float((gain_u[~gates] > 0).mean()) if (~gates).any() else float("nan"),
            "d7c_positive_gain_rate_outside_gate": float((gain_d[~gates] > 0).mean()) if (~gates).any() else float("nan"),
        },
        "decision": "V3E_C_OPERATOR_GAIN_ALIGNMENT_DONE_REVIEW_SCORE_AND_BOUNDARY",
    }
    zone_summary = []
    for zone in sorted({row["zone"] for row in region_rows}):
        rows = [row for row in region_rows if row["zone"] == zone and row["pixel_count"] > 0]
        zone_summary.append(
            {
                "zone": zone,
                "images": len(rows),
                "mean_pixel_count": mean(row["pixel_count"] for row in rows),
                "d7c_output_change_mean": mean(row["d7c_output_change_mean"] for row in rows if math.isfinite(row["d7c_output_change_mean"])),
                "control_output_change_mean": mean(row["control_output_change_mean"] for row in rows if math.isfinite(row["control_output_change_mean"])),
                "d7c_gain_l1_mean": mean(row["d7c_gain_l1_mean"] for row in rows if math.isfinite(row["d7c_gain_l1_mean"])),
                "control_gain_l1_mean": mean(row["control_gain_l1_mean"] for row in rows if math.isfinite(row["control_gain_l1_mean"])),
            }
        )
    summary["zone_summary"] = zone_summary
    write_csv(out / "v3e_c_boundary_leakage_by_image_zone.csv", region_rows)
    write_csv(out / "v3e_c_boundary_leakage_summary.csv", zone_summary)
    write_json(out / "v3e_c_operator_gain_alignment_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


class TrainArgs:
    pass


def train_loss_components(model, fam_mode, gate_producer, base_model, input_img, label_img):
    gate = None
    score = None
    if gate_producer is not None:
        gate, score, _ = gate_producer(input_img)
    if fam_mode == "fam2_d7c_noop":
        pred_img = model(input_img, d7c_gate=gate)
    else:
        pred_img = model(input_img)
    criterion = torch.nn.L1Loss()
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    l1 = criterion(pred_img[0], label_img4)
    l2 = criterion(pred_img[1], label_img2)
    l3 = criterion(pred_img[2], label_img)
    loss_content = l1 + l2 + l3
    fft_losses = []
    for pred, label in [(pred_img[0], label_img4), (pred_img[1], label_img2), (pred_img[2], label_img)]:
        label_fft = torch.fft.fft2(label, dim=(-2, -1))
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_losses.append(criterion(pred_fft, label_fft))
    loss_fft = sum(fft_losses)
    total = loss_content + 0.1 * loss_fft
    with torch.no_grad():
        base_pred = base_model(input_img)[2]
    if gate is None:
        mask = torch.ones((input_img.shape[0], 1, input_img.shape[2], input_img.shape[3]), device=input_img.device)
    else:
        mask = F.interpolate(gate, size=input_img.shape[-2:], mode="nearest")
    negative = 1.0 - mask
    action_loss = ((pred_img[2] - label_img).abs() * mask).sum() / (mask.sum() * pred_img[2].shape[1] + 1e-8)
    preserve_loss = ((pred_img[2] - base_pred).abs() * negative).sum() / (negative.sum() * pred_img[2].shape[1] + 1e-8)
    return {
        "total": total,
        "content": loss_content,
        "fft": loss_fft,
        "action": action_loss,
        "preserve": preserve_loss,
        "gate_mean": mask.mean().item(),
        "score_mean": score.mean().item() if score is not None else None,
    }


def flatten_grads(loss, params, retain_graph=True):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            flat.append(grad.detach().reshape(-1))
    return torch.cat(flat)


def cosine(a, b):
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return float("nan")
    return (torch.dot(a, b) / denom).item()


def gradient_contract(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base_model = build_candidate("original", args.a0_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    variants = {
        "D7C_W_D_G_D": ("fam2_d7c_noop", args.d7c_checkpoint),
        "CONTROL_W_U_G_1": ("fam2_modres", args.control_checkpoint),
    }
    names = load_names(args.split_json, args.split_key, args.max_samples)
    batches = []
    for start in range(0, min(len(names), args.batch_size * args.num_batches), args.batch_size):
        batch_names = names[start : start + args.batch_size]
        inputs = []
        labels = []
        for name in batch_names:
            inp, lab = load_pair(args.data_dir, args.source_split, name)
            inp, lab = center_crop_pair(inp, lab, args.crop_size)
            inputs.append(inp)
            labels.append(lab)
        batches.append((batch_names, torch.stack(inputs).to(device), torch.stack(labels).to(device)))
    rows = []
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3e-D no-step gradient and optimizer contract audit",
        "locked_test_touched": False,
        "batch_count": len(batches),
        "batch_size": args.batch_size,
        "cli_weight_decay": args.weight_decay,
        "effective_optimizer_weight_decay": {},
        "checkpoint_key_audit": {},
        "scheduler_contract": {},
    }
    for ckpt_label, ckpt_path in [("d7c_final", args.d7c_checkpoint), ("control_final", args.control_checkpoint), ("d7c_resume", args.d7c_resume_checkpoint), ("control_resume", args.control_resume_checkpoint)]:
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            summary["checkpoint_key_audit"][ckpt_label] = {
                "path": ckpt_path,
                "keys": sorted(state.keys()) if isinstance(state, dict) else ["<raw_state_dict>"],
                "has_scheduler": isinstance(state, dict) and "scheduler" in state,
                "has_optimizer": isinstance(state, dict) and "optimizer" in state,
                "epoch": state.get("epoch") if isinstance(state, dict) else None,
            }
    for variant, (fam_mode, ckpt_path) in variants.items():
        model = build_candidate(fam_mode, ckpt_path, device)
        train_args = TrainArgs()
        train_args.rarm_train_scope = "fam2_modulator_only"
        params = _configure_train_scope(model, train_args)
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        summary["effective_optimizer_weight_decay"][variant] = optimizer.param_groups[0].get("weight_decay", None)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch - 3, eta_min=1e-6)
        from warmup_scheduler import GradualWarmupScheduler

        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
        scheduler.step()
        summary["scheduler_contract"][variant] = {
            "after_initial_scheduler_step_get_lr": scheduler.get_lr()[0],
            "optimizer_lr_after_initial_scheduler_step": optimizer.param_groups[0]["lr"],
            "scheduler_step_called_before_optimizer_step_matches_train_py": True,
        }
        model.eval()
        model.FAM2.modulator.train()
        for batch_index, (batch_names, input_img, label_img) in enumerate(batches):
            input_img, label_img = pad_batch_pair_to_factor(input_img, label_img)
            components = train_loss_components(model, fam_mode, gate_producer if fam_mode == "fam2_d7c_noop" else gate_producer, base_model, input_img, label_img)
            grads = {}
            for key in ["total", "content", "fft", "action", "preserve"]:
                grads[key] = flatten_grads(components[key], params, retain_graph=True)
            norms = {key: torch.linalg.vector_norm(value).item() for key, value in grads.items()}
            row = {
                "variant": variant,
                "batch_index": batch_index,
                "names": "|".join(batch_names),
                "gate_mean": components["gate_mean"],
                "score_mean": components["score_mean"],
                "loss_total": components["total"].item(),
                "loss_content": components["content"].item(),
                "loss_fft": components["fft"].item(),
                "loss_action": components["action"].item(),
                "loss_preserve": components["preserve"].item(),
                "pre_clip_norm_total": norms["total"],
                "post_clip_norm_total_if_train_contract": min(norms["total"], args.grad_clip_norm),
                "clip_scale_total_if_train_contract": min(1.0, args.grad_clip_norm / norms["total"]) if norms["total"] > 0 else float("nan"),
                "grad_norm_content": norms["content"],
                "grad_norm_fft": norms["fft"],
                "grad_norm_action": norms["action"],
                "grad_norm_preserve": norms["preserve"],
                "cos_content_fft": cosine(grads["content"], grads["fft"]),
                "cos_action_preserve": cosine(grads["action"], grads["preserve"]),
                "cos_total_action": cosine(grads["total"], grads["action"]),
                "cos_total_preserve": cosine(grads["total"], grads["preserve"]),
            }
            rows.append(row)
            model.zero_grad(set_to_none=True)
    by_variant = {}
    for variant in variants:
        vrows = [r for r in rows if r["variant"] == variant]
        by_variant[variant] = {
            "mean_pre_clip_norm_total": mean(r["pre_clip_norm_total"] for r in vrows),
            "mean_clip_scale_total": mean(r["clip_scale_total_if_train_contract"] for r in vrows),
            "clipped_batch_ratio": mean(1.0 if r["pre_clip_norm_total"] > args.grad_clip_norm else 0.0 for r in vrows),
            "mean_cos_content_fft": mean(r["cos_content_fft"] for r in vrows if math.isfinite(r["cos_content_fft"])),
            "mean_cos_action_preserve": mean(r["cos_action_preserve"] for r in vrows if math.isfinite(r["cos_action_preserve"])),
        }
    summary["variant_summary"] = by_variant
    summary["decision"] = "V3E_D_GRADIENT_CONTRACT_AUDIT_DONE_REVIEW_CLIP_AND_COSINES"
    write_csv(out / "v3e_d_gradient_norm_trace.csv", rows)
    write_json(out / "v3e_d_gradient_contract_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paired")
    p.add_argument("--old_logdir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=3407)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--a0_checkpoint", required=True)
    common.add_argument("--d7c_checkpoint", required=True)
    common.add_argument("--control_checkpoint", required=True)
    common.add_argument("--data_dir", required=True)
    common.add_argument("--split_json", required=True)
    common.add_argument("--density_artifact", required=True)
    common.add_argument("--d7c_artifact", required=True)
    common.add_argument("--output_dir", required=True)
    common.add_argument("--source_split", default="train")
    common.add_argument("--split_key", default="val_inner")
    common.add_argument("--max_samples", type=int, default=600)
    common.add_argument("--seed", type=int, default=3407)
    common.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    common.add_argument("--progress_every", type=int, default=50)

    p = sub.add_parser("replay", parents=[common])
    p.add_argument("--noninferiority_margin", type=float, default=0.01)
    p.add_argument("--mean_gap_margin", type=float, default=0.02)

    p = sub.add_parser("gain", parents=[common])
    p.add_argument("--pixel_sample_per_image", type=int, default=8192)

    p = sub.add_parser("grad", parents=[common])
    p.add_argument("--d7c_resume_checkpoint", default="")
    p.add_argument("--control_resume_checkpoint", default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=0.001)
    p.add_argument("--num_epoch", type=int, default=1000)

    args = parser.parse_args()
    if args.cmd == "paired":
        paired_reanalysis(args)
    elif args.cmd == "replay":
        replay_2x2(args)
    elif args.cmd == "gain":
        gain_alignment(args)
    elif args.cmd == "grad":
        gradient_contract(args)


if __name__ == "__main__":
    main()
