import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())

from data.data_load import DeblurDataset
from models.APDRConvIR import build_apdr_net


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def psnr(pred, target):
    mse = f.mse_loss(pred, target).clamp_min(1e-12)
    return (10 * torch.log10(1 / mse)).item()


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        return f.pad(input_img, (0, padw, 0, padh), "reflect"), h, w
    return input_img, h, w


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_lowpass(x, kernel_size=31, sigma=7.0):
    if kernel_size <= 1:
        return x
    kernel = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    pad = kernel_size // 2
    x_pad = f.pad(x, (pad, pad, 0, 0), mode="reflect")
    x_blur = f.conv2d(x_pad, kx, groups=channels)
    x_pad = f.pad(x_blur, (0, 0, pad, pad), mode="reflect")
    return f.conv2d(x_pad, ky, groups=channels)


def channel_stats(prefix, tensor):
    flat = tensor.flatten(2)
    result = {}
    for idx in range(tensor.shape[1]):
        values = flat[:, idx]
        result[f"{prefix}_c{idx}_mean"] = values.mean().item()
        result[f"{prefix}_c{idx}_std"] = values.std(unbiased=False).item()
        result[f"{prefix}_c{idx}_p10"] = torch.quantile(values.flatten(), 0.10).item()
        result[f"{prefix}_c{idx}_p90"] = torch.quantile(values.flatten(), 0.90).item()
    return result


def scalar_stats(prefix, tensor):
    values = tensor.flatten()
    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_std": values.std(unbiased=False).item(),
        f"{prefix}_p05": torch.quantile(values, 0.05).item(),
        f"{prefix}_p50": torch.quantile(values, 0.50).item(),
        f"{prefix}_p95": torch.quantile(values, 0.95).item(),
        f"{prefix}_max": values.max().item(),
    }


def rank_values(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = rank
        i = j
    return ranks


def pearson(x, y):
    if len(x) < 2:
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(x, y):
    return pearson(rank_values(x), rank_values(y))


def auc_score(scores, labels):
    pairs = [(score, label) for score, label in zip(scores, labels) if label in (0, 1)]
    pos = sum(label == 1 for _, label in pairs)
    neg = sum(label == 0 for _, label in pairs)
    if pos == 0 or neg == 0:
        return None
    ranked = rank_values([score for score, _ in pairs])
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranked, pairs) if label == 1)
    return (rank_sum_pos - pos * (pos - 1) / 2.0) / (pos * neg)


class TabularBenefitProxy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


def build_loader(data_dir, split, count, num_workers):
    image_dir = Path(data_dir) / split
    dataset = DeblurDataset(str(image_dir), "Haze4K", transform=None, is_test=True)
    if count > 0:
        dataset = Subset(dataset, list(range(min(count, len(dataset)))))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_apdr_model(selector_checkpoint, residual_max, device):
    model = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=residual_max,
        apdr_gate_max=1.0,
        apdr_gate_init=0.01,
        apdr_force_zero_gate=False,
        apdr_active_scales="full",
        apdr_selector_mode="v0_2r",
        apdr_residual_capacity="linear",
    ).to(device)
    model.load_state_dict(load_model_state(selector_checkpoint, device), strict=True)
    model.eval()
    return model


def collect_rows(model, loader, device, args):
    rows = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            model(padded)
            full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
            anchor = full["anchor"][:, :, :h, :w].detach().clamp(0, 1)
            m_safe = full["gate"][:, :, :h, :w].detach().clamp(0, 1)
            spatial = full.get("spatial_gate", m_safe)[:, :, :h, :w].detach().clamp(0, 1)
            priors = full["prior"][:, :, :h, :w].detach()
            global_budget = full.get("global_budget_unit", full.get("global_gate"))
            global_score = full.get("global_score_unit", full.get("global_logits"))
            residual_proxy = (input_img - anchor).abs()
            delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max)
            low_delta = gaussian_lowpass(delta_star, args.kernel_size, args.sigma)
            anchor_psnr = psnr(anchor, label_img)
            low_pred = (anchor + m_safe * low_delta).clamp(0, 1)
            low_gain = psnr(low_pred, label_img) - anchor_psnr

            feature_values = {}
            feature_values.update(channel_stats("hazy", input_img))
            feature_values.update(channel_stats("anchor", anchor))
            feature_values.update(channel_stats("hazy_minus_anchor_abs", residual_proxy))
            feature_values.update(channel_stats("prior", priors))
            feature_values.update(scalar_stats("m_safe", m_safe))
            feature_values.update(scalar_stats("spatial_gate", spatial))
            feature_values["global_budget_mean"] = global_budget.detach().float().mean().item()
            feature_values["global_score_mean"] = global_score.detach().float().mean().item()
            feature_values["hazy_anchor_l1"] = residual_proxy.mean().item()
            feature_values["hazy_anchor_l2"] = torch.sqrt(residual_proxy.square().mean()).item()

            row = {
                "name": name[0],
                "index": idx,
                "anchor_psnr": anchor_psnr,
                "low_oracle_gain": low_gain,
                "benefit_label": 1 if low_gain >= args.positive_gain else (0 if low_gain <= args.negative_gain else -1),
                **feature_values,
            }
            rows.append(row)
            if args.progress_freq and (idx + 1) % args.progress_freq == 0:
                print(f"collected={idx + 1}", flush=True)
    return rows


def split_indices(count, train_fraction, seed):
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(round(count * train_fraction))
    return indices[:split], indices[split:]


def matrix_from_rows(rows, feature_names, indices, device):
    data = [[rows[idx][name] for name in feature_names] for idx in indices]
    return torch.tensor(data, dtype=torch.float32, device=device)


def train_proxy(rows, feature_names, train_idx, valid_idx, args, device):
    train_labeled = [idx for idx in train_idx if rows[idx]["benefit_label"] in (0, 1)]
    valid_labeled = [idx for idx in valid_idx if rows[idx]["benefit_label"] in (0, 1)]
    if not train_labeled or not valid_labeled:
        raise RuntimeError("Not enough labeled positive/negative rows for proxy audit.")
    train_y = torch.tensor(
        [rows[idx]["benefit_label"] for idx in train_labeled],
        dtype=torch.float32,
        device=device,
    )
    pos = train_y.sum().item()
    neg = train_y.numel() - pos
    if pos == 0 or neg == 0:
        raise RuntimeError("Training split has only one benefit class.")

    train_x_all = matrix_from_rows(rows, feature_names, train_idx, device)
    mean = train_x_all.mean(dim=0, keepdim=True)
    std = train_x_all.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (matrix_from_rows(rows, feature_names, train_labeled, device) - mean) / std

    model = TabularBenefitProxy(len(feature_names), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
        if args.progress_freq and step % args.progress_freq == 0:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(train_x)).detach().cpu().tolist()
            auc = auc_score(scores, [rows[idx]["benefit_label"] for idx in train_labeled])
            history.append({"step": step, "train_loss": loss.item(), "train_auc": auc})
            print(f"step={step} train_loss={loss.item():.6f} train_auc={auc:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        all_x = (matrix_from_rows(rows, feature_names, list(range(len(rows))), device) - mean) / std
        all_scores = torch.sigmoid(model(all_x)).detach().cpu().tolist()
    return all_scores, history, train_labeled, valid_labeled


def summarize(rows, scores, train_labeled, valid_labeled, train_idx, valid_idx, args):
    for row, score in zip(rows, scores):
        row["proxy_score"] = score
    train_scores = [scores[idx] for idx in train_labeled]
    train_labels = [rows[idx]["benefit_label"] for idx in train_labeled]
    valid_scores = [scores[idx] for idx in valid_labeled]
    valid_labels = [rows[idx]["benefit_label"] for idx in valid_labeled]
    valid_gains = [rows[idx]["low_oracle_gain"] for idx in valid_idx]
    valid_all_scores = [scores[idx] for idx in valid_idx]
    hard_cut = percentile([rows[idx]["anchor_psnr"] for idx in valid_idx], 25)
    easy_cut = percentile([rows[idx]["anchor_psnr"] for idx in valid_idx], 75)
    easy_rows = [idx for idx in valid_idx if rows[idx]["anchor_psnr"] >= easy_cut]
    positive_hard_rows = [
        idx
        for idx in valid_idx
        if rows[idx]["anchor_psnr"] <= hard_cut and rows[idx]["low_oracle_gain"] >= args.positive_gain
    ]
    valid_pos = sum(label == 1 for label in valid_labels)
    valid_neg = sum(label == 0 for label in valid_labels)
    summary = {
        "count": len(rows),
        "train_count": len(train_idx),
        "valid_count": len(valid_idx),
        "train_labeled_count": len(train_labeled),
        "valid_labeled_count": len(valid_labeled),
        "valid_positive_count": valid_pos,
        "valid_negative_count": valid_neg,
        "positive_gain_threshold": args.positive_gain,
        "negative_gain_threshold": args.negative_gain,
        "train_auc": auc_score(train_scores, train_labels),
        "valid_auc": auc_score(valid_scores, valid_labels),
        "valid_spearman_score_low_gain": spearman(valid_all_scores, valid_gains),
        "valid_mean_score_easy_top25": statistics.mean(scores[idx] for idx in easy_rows) if easy_rows else None,
        "valid_mean_score_oracle_positive_hard": (
            statistics.mean(scores[idx] for idx in positive_hard_rows) if positive_hard_rows else None
        ),
        "valid_positive_hard_count": len(positive_hard_rows),
        "valid_hard_anchor_psnr_cut": hard_cut,
        "valid_easy_anchor_psnr_cut": easy_cut,
        "valid_mean_low_oracle_gain": statistics.mean(valid_gains),
        "valid_mean_score": statistics.mean(valid_all_scores),
    }
    checks = {
        "valid_auc": {
            "observed": summary["valid_auc"],
            "required": ">= 0.80",
            "pass": summary["valid_auc"] is not None and summary["valid_auc"] >= 0.80,
        },
        "valid_spearman_score_low_gain": {
            "observed": summary["valid_spearman_score_low_gain"],
            "required": ">= 0.45",
            "pass": (
                summary["valid_spearman_score_low_gain"] is not None
                and summary["valid_spearman_score_low_gain"] >= 0.45
            ),
        },
        "valid_mean_score_easy_top25": {
            "observed": summary["valid_mean_score_easy_top25"],
            "required": "<= 0.10",
            "pass": (
                summary["valid_mean_score_easy_top25"] is not None
                and summary["valid_mean_score_easy_top25"] <= 0.10
            ),
        },
        "valid_mean_score_oracle_positive_hard": {
            "observed": summary["valid_mean_score_oracle_positive_hard"],
            "required": ">= 0.50",
            "pass": (
                summary["valid_mean_score_oracle_positive_hard"] is not None
                and summary["valid_mean_score_oracle_positive_hard"] >= 0.50
            ),
        },
    }
    return summary, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_correctability_proxy")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--positive_gain", type=float, default=0.10)
    parser.add_argument("--negative_gain", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    loader = build_loader(args.data_dir, args.split, args.max_images, args.num_workers)
    rows = collect_rows(model, loader, device, args)
    if len(rows) < 20:
        raise RuntimeError("Too few rows for held-out proxy audit.")
    feature_names = [
        name
        for name in rows[0].keys()
        if name not in {"name", "index", "anchor_psnr", "low_oracle_gain", "benefit_label"}
    ]
    train_idx, valid_idx = split_indices(len(rows), args.train_fraction, args.seed)
    scores, history, train_labeled, valid_labeled = train_proxy(
        rows,
        feature_names,
        train_idx,
        valid_idx,
        args,
        device,
    )
    summary, checks = summarize(
        rows,
        scores,
        train_labeled,
        valid_labeled,
        train_idx,
        valid_idx,
        args,
    )
    result = {
        "stage": "APDR-v0.3 CorrectabilityOpen tabular proxy audit",
        "tag": args.tag,
        "feature_names": feature_names,
        "summary": summary,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "history": history,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"correctability_proxy_{args.tag}.json"
    csv_path = output_dir / f"correctability_proxy_per_image_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
