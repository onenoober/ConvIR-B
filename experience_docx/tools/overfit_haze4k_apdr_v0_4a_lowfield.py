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
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
    correlation,
    frozen_apdr_tensors,
    gaussian_lowpass,
    percentile,
    psnr,
)


def tv_loss(x):
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return x.new_tensor(0.0)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


def gradient_magnitude(x):
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def lowfield_features(x, anchor, m_safe, p_benefit):
    max_rgb = x.max(dim=1, keepdim=True).values
    min_rgb = x.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    grad = gradient_magnitude(x)
    diff = x - anchor
    p_map = torch.full_like(m_safe, float(p_benefit))
    return torch.cat(
        [
            x,
            anchor,
            diff,
            diff.abs(),
            min_rgb,
            max_rgb,
            saturation,
            grad,
            m_safe,
            p_map,
        ],
        dim=1,
    )


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class LowFieldNetV1(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=48, residual_max=0.04):
        super().__init__()
        self.residual_max = float(residual_max)
        h = int(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(h, h * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h * 2, kernel_size=7, dilation=1),
            DepthwiseResidualBlock(h * 2, kernel_size=7, dilation=2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(h * 2, h * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h * 4, kernel_size=11, dilation=1),
            DepthwiseResidualBlock(h * 4, kernel_size=7, dilation=4),
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(h * 4, h * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h * 4, h * 4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(h * 4 + h * 2, h * 2, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h * 2, kernel_size=7, dilation=1),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(h * 2 + h, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
        )
        self.head = nn.Conv2d(h, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, kernel_size, sigma):
        s0 = self.stem(features)
        s1 = self.down1(s0)
        bottleneck = self.down2(s1)
        bottleneck = bottleneck * self.context_gate(bottleneck)
        up = F.interpolate(bottleneck, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up1(torch.cat([up, s1], dim=1))
        up = F.interpolate(up, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up2(torch.cat([up, s0], dim=1))
        bounded = self.residual_max * torch.tanh(self.head(up))
        return gaussian_lowpass(bounded, kernel_size, sigma)


class IDEmbeddingLowField(nn.Module):
    def __init__(self, count, low_size=32, residual_max=0.04):
        super().__init__()
        self.low_size = int(low_size)
        self.residual_max = float(residual_max)
        self.raw = nn.Parameter(torch.zeros(int(count), 3, self.low_size, self.low_size))

    def forward_for_index(self, index, size, kernel_size, sigma):
        raw = self.raw[int(index) : int(index) + 1]
        delta = self.residual_max * torch.tanh(raw)
        delta = F.interpolate(delta, size=size, mode="bilinear", align_corners=False)
        return gaussian_lowpass(delta, kernel_size, sigma)


class BasisMixtureLowField(nn.Module):
    def __init__(
        self,
        in_channels=18,
        hidden_channels=48,
        num_bases=16,
        basis_size=32,
        residual_max=0.04,
    ):
        super().__init__()
        self.residual_max = float(residual_max)
        self.basis_size = int(basis_size)
        self.bases = nn.Parameter(torch.zeros(int(num_bases), 3, self.basis_size, self.basis_size))
        h = int(hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
            nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.weight_head = nn.Conv2d(h, int(num_bases), kernel_size=1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def basis_delta(self, features):
        weights = torch.softmax(self.weight_head(self.encoder(features)).flatten(1), dim=1)
        bounded_bases = self.residual_max * torch.tanh(self.bases)
        delta = torch.einsum("bk,kchw->bchw", weights, bounded_bases)
        return F.interpolate(delta, size=features.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features, kernel_size, sigma):
        return gaussian_lowpass(self.basis_delta(features), kernel_size, sigma)


class BasisLocalLowField(BasisMixtureLowField):
    def __init__(
        self,
        in_channels=18,
        hidden_channels=48,
        num_bases=16,
        basis_size=32,
        residual_max=0.04,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_bases=num_bases,
            basis_size=basis_size,
            residual_max=residual_max,
        )
        h = int(hidden_channels)
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=2),
        )
        self.local_head = nn.Conv2d(h, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)

    def forward(self, features, kernel_size, sigma):
        basis = self.basis_delta(features)
        local = 0.5 * self.residual_max * torch.tanh(self.local_head(self.local(features)))
        delta = (basis + local).clamp(-self.residual_max, self.residual_max)
        return gaussian_lowpass(delta, kernel_size, sigma)


class PhysicsVeilLowField(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=48, residual_max=0.04):
        super().__init__()
        self.residual_max = float(residual_max)
        h = int(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
        )
        self.down = nn.Sequential(
            nn.Conv2d(h, h * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h * 2, kernel_size=7, dilation=2),
            nn.Conv2d(h * 2, h * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h * 2, kernel_size=11, dilation=1),
        )
        self.up = nn.Sequential(
            nn.Conv2d(h * 2 + h, h, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseResidualBlock(h, kernel_size=7, dilation=1),
        )
        self.veil_head = nn.Conv2d(h, 1, kernel_size=3, padding=1)
        self.epsilon_head = nn.Conv2d(h, 3, kernel_size=3, padding=1)
        self.color_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(h * 2, h, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h, 3, kernel_size=1),
        )
        nn.init.zeros_(self.veil_head.weight)
        nn.init.zeros_(self.veil_head.bias)
        nn.init.zeros_(self.epsilon_head.weight)
        nn.init.zeros_(self.epsilon_head.bias)
        nn.init.zeros_(self.color_head[-1].weight)
        nn.init.zeros_(self.color_head[-1].bias)

    def forward(self, features, kernel_size, sigma):
        shallow = self.stem(features)
        deep = self.down(shallow)
        up = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        context = self.up(torch.cat([up, shallow], dim=1))
        veil = torch.tanh(self.veil_head(context))
        epsilon = 0.25 * torch.tanh(self.epsilon_head(context))
        color = 0.5 + 0.5 * torch.tanh(self.color_head(deep))
        delta = self.residual_max * (veil * color + epsilon)
        return gaussian_lowpass(delta, kernel_size, sigma)


def build_probe(args, record_count):
    if args.model_type == "lowfield":
        return LowFieldNetV1(hidden_channels=args.hidden_channels, residual_max=args.residual_max)
    if args.model_type == "id_embedding":
        return IDEmbeddingLowField(record_count, low_size=args.low_size, residual_max=args.residual_max)
    if args.model_type == "basis":
        return BasisMixtureLowField(
            hidden_channels=args.hidden_channels,
            num_bases=args.num_bases,
            basis_size=args.low_size,
            residual_max=args.residual_max,
        )
    if args.model_type == "basis_local":
        return BasisLocalLowField(
            hidden_channels=args.hidden_channels,
            num_bases=args.num_bases,
            basis_size=args.low_size,
            residual_max=args.residual_max,
        )
    if args.model_type == "veil":
        return PhysicsVeilLowField(hidden_channels=args.hidden_channels, residual_max=args.residual_max)
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def predict_delta(model, features, args, record_index):
    if hasattr(model, "forward_for_index"):
        return model.forward_for_index(record_index, features.shape[-2:], args.kernel_size, args.sigma)
    return model(features, args.kernel_size, args.sigma)


def read_correctability(correctability_json, correctability_csv):
    data = json.loads(Path(correctability_json).read_text(encoding="utf-8"))
    tau = data["summary"]["train_oof_tau"]
    scores = {}
    with Path(correctability_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scores[row["name"]] = float(row["proxy_score"])
    return tau, scores


def choose_crop(rng, height, width, crop_size):
    crop_h = min(int(crop_size), height)
    crop_w = min(int(crop_size), width)
    y = 0 if height <= crop_h else rng.randint(0, height - crop_h)
    x = 0 if width <= crop_w else rng.randint(0, width - crop_w)
    return y, x, crop_h, crop_w


def crop_tensor(x, y, x0, h, w):
    return x[:, :, y : y + h, x0 : x0 + w]


def load_records(apdr_model, loader, device, args, tau, scores, cache_dir):
    records = []
    cache_rows = []
    rng = random.Random(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, (input_img, label_img, name) in enumerate(loader):
        image_name = name[0]
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        anchor, m_safe, _delta_star, low_target, _color_target = frozen_apdr_tensors(
            apdr_model,
            input_img,
            label_img,
            args,
        )
        score = scores.get(image_name)
        if score is None:
            raise KeyError(f"Missing correctability score for {image_name}")
        p_benefit = 1.0 if score >= tau else 0.0
        record = {
            "index": index,
            "name": image_name,
            "input": input_img.detach().cpu(),
            "label": label_img.detach().cpu(),
            "anchor": anchor.detach().cpu(),
            "m_safe": m_safe.detach().cpu(),
            "target": low_target.detach().cpu(),
            "p_benefit": p_benefit,
            "proxy_score": score,
        }
        cache_path = cache_dir / f"{index:04d}_{Path(image_name).stem}.pt"
        torch.save(
            {
                "name": image_name,
                "input": record["input"].squeeze(0).float(),
                "label": record["label"].squeeze(0).float(),
                "anchor": record["anchor"].squeeze(0).float(),
                "m_safe": record["m_safe"].squeeze(0).float(),
                "target": record["target"].squeeze(0).float(),
                "p_benefit": p_benefit,
                "proxy_score": score,
            },
            cache_path,
        )
        loaded = torch.load(cache_path, map_location="cpu")
        height, width = record["anchor"].shape[-2:]
        y, x0, crop_h, crop_w = choose_crop(rng, height, width, args.crop_size)
        cache_rows.append(
            {
                "name": image_name,
                "index": index,
                "crop_x": x0,
                "crop_y": y,
                "crop_h": crop_h,
                "crop_w": crop_w,
                "cached_patch_anchor_max_abs_diff": (
                    loaded["anchor"][:, y : y + crop_h, x0 : x0 + crop_w]
                    - record["anchor"].squeeze(0)[:, y : y + crop_h, x0 : x0 + crop_w]
                )
                .abs()
                .max()
                .item(),
                "cached_patch_m_safe_max_abs_diff": (
                    loaded["m_safe"][:, y : y + crop_h, x0 : x0 + crop_w]
                    - record["m_safe"].squeeze(0)[:, y : y + crop_h, x0 : x0 + crop_w]
                )
                .abs()
                .max()
                .item(),
                "cached_patch_low_target_max_abs_diff": (
                    loaded["target"][:, y : y + crop_h, x0 : x0 + crop_w]
                    - record["target"].squeeze(0)[:, y : y + crop_h, x0 : x0 + crop_w]
                )
                .abs()
                .max()
                .item(),
            }
        )
        records.append(record)
    return records, cache_rows


def weighted_smooth_l1(pred, target, weight):
    expanded = weight.expand_as(pred)
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
    return (expanded * loss).sum() / expanded.sum().clamp_min(1e-12)


def train_step(model, record, device, args, rng):
    height, width = record["anchor"].shape[-2:]
    y, x0, crop_h, crop_w = choose_crop(rng, height, width, args.crop_size)
    input_img = crop_tensor(record["input"], y, x0, crop_h, crop_w).to(device)
    anchor = crop_tensor(record["anchor"], y, x0, crop_h, crop_w).to(device)
    m_safe = crop_tensor(record["m_safe"], y, x0, crop_h, crop_w).to(device)
    target = crop_tensor(record["target"], y, x0, crop_h, crop_w).to(device)
    weight = m_safe * float(record["p_benefit"])
    if weight.sum().item() <= 1e-8:
        return None
    features = lowfield_features(input_img, anchor, m_safe, record["p_benefit"])
    pred = predict_delta(model, features, args, record["index"])
    loss_delta = weighted_smooth_l1(pred, target, weight)
    lowpass = gaussian_lowpass(pred, args.kernel_size, args.sigma)
    loss_lowpass = (pred - lowpass).abs().mean()
    loss_tv = tv_loss(pred)
    loss = loss_delta + args.lowpass_lambda * loss_lowpass + args.tv_lambda * loss_tv
    return loss, {
        "loss_delta": loss_delta.item(),
        "loss_lowpass": loss_lowpass.item(),
        "loss_tv": loss_tv.item(),
    }


def evaluate(model, records, device, args):
    rows = []
    weighted_l1_num = 0.0
    weighted_l1_den = 0.0
    corrs = []
    amplitude_values = []
    outside_values = []
    lowpass_consistency = []
    with torch.no_grad():
        for record in records:
            input_img = record["input"].to(device)
            label = record["label"].to(device)
            anchor = record["anchor"].to(device)
            m_safe = record["m_safe"].to(device)
            target = record["target"].to(device)
            weight = m_safe * float(record["p_benefit"])
            features = lowfield_features(input_img, anchor, m_safe, record["p_benefit"])
            pred = predict_delta(model, features, args, record["index"])
            expanded = weight.expand_as(pred)
            weighted_l1_num += (expanded * (pred - target).abs()).sum().item()
            weighted_l1_den += expanded.sum().item()
            corr = correlation(pred.cpu(), target.cpu(), expanded.cpu())
            if corr is not None:
                corrs.append(corr)
            applied = weight * pred
            output = (anchor + applied).clamp(0, 1)
            oracle = (anchor + weight * target).clamp(0, 1)
            anchor_psnr = psnr(anchor, label)
            output_psnr = psnr(output, label)
            oracle_psnr = psnr(oracle, label)
            pred_abs = pred.abs().detach().flatten().cpu()
            amplitude_values.extend(pred_abs.tolist())
            outside_mask = (weight <= 1e-8).expand_as(applied)
            if outside_mask.any():
                outside_values.append(applied[outside_mask].abs().max().item())
            lowpass_consistency.append((pred - gaussian_lowpass(pred, args.kernel_size, args.sigma)).abs().mean().item())
            rows.append(
                {
                    "name": record["name"],
                    "index": record["index"],
                    "anchor_psnr": anchor_psnr,
                    "output_psnr": output_psnr,
                    "oracle_psnr": oracle_psnr,
                    "output_gain": output_psnr - anchor_psnr,
                    "oracle_gain": oracle_psnr - anchor_psnr,
                    "gain_delta": output_psnr - anchor_psnr,
                    "output_max_abs_diff_vs_anchor": (output - anchor).abs().max().item(),
                    "corr": corr,
                    "M_safe_mean": m_safe.mean().item(),
                    "P_benefit": record["p_benefit"],
                    "proxy_score": record["proxy_score"],
                    "target_abs_mean": target.abs().mean().item(),
                    "pred_abs_mean": pred.abs().mean().item(),
                    "applied_abs_mean": applied.abs().mean().item(),
                }
            )
    amplitude_values = sorted(amplitude_values)
    p95 = amplitude_values[int(0.95 * (len(amplitude_values) - 1))] if amplitude_values else None
    amplitude = {
        "residual_abs_mean": statistics.mean(amplitude_values) if amplitude_values else None,
        "residual_abs_p95": p95,
        "lowpass_consistency": statistics.mean(lowpass_consistency) if lowpass_consistency else None,
        "outside_mask_residual_max": max(outside_values) if outside_values else 0.0,
    }
    return rows, weighted_l1_num / max(weighted_l1_den, 1e-12), (statistics.mean(corrs) if corrs else None), amplitude


def opened_closed_groups(rows):
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    hard_cut = ordered[max(0, len(ordered) // 4 - 1)]["anchor_psnr"]
    easy_cut = ordered[3 * len(ordered) // 4]["anchor_psnr"]
    groups = {
        "open_hard": [row for row in rows if row["P_benefit"] >= 0.5 and row["anchor_psnr"] <= hard_cut],
        "closed_hard": [row for row in rows if row["P_benefit"] < 0.5 and row["anchor_psnr"] <= hard_cut],
        "open_easy": [row for row in rows if row["P_benefit"] >= 0.5 and row["anchor_psnr"] >= easy_cut],
        "closed_easy": [row for row in rows if row["P_benefit"] < 0.5 and row["anchor_psnr"] >= easy_cut],
    }
    out = []
    for name, members in groups.items():
        out.append(
            {
                "group": name,
                "count": len(members),
                "mean_output_gain": statistics.mean([row["output_gain"] for row in members]) if members else None,
                "mean_oracle_gain": statistics.mean([row["oracle_gain"] for row in members]) if members else None,
            }
        )
    return out


def summarize(rows, initial_loss, final_loss, final_corr):
    gains = [row["output_gain"] for row in rows]
    oracle_gains = [row["oracle_gain"] for row in rows]
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    easy = ordered[3 * len(ordered) // 4 :]
    positive = [row for row in rows if row["oracle_gain"] > 1e-6]
    recovery = sum(row["output_gain"] for row in positive) / max(
        sum(row["oracle_gain"] for row in positive),
        1e-12,
    )
    strong_cut = percentile([row["anchor_psnr"] for row in rows], 75)
    strong = [row for row in rows if row["anchor_psnr"] >= strong_cut]
    return {
        "count": len(rows),
        "open_count": sum(row["P_benefit"] >= 0.5 for row in rows),
        "initial_weighted_delta_l1": initial_loss,
        "final_weighted_delta_l1": final_loss,
        "weighted_delta_l1_drop": (initial_loss - final_loss) / max(initial_loss, 1e-12),
        "pred_target_corr": final_corr,
        "mean_output_gain": statistics.mean(gains),
        "mean_oracle_gain": statistics.mean(oracle_gains),
        "oracle_recovery": recovery,
        "hard_bottom25_output_gain": statistics.mean(row["output_gain"] for row in hard),
        "hard_bottom25_oracle_gain": statistics.mean(row["oracle_gain"] for row in hard),
        "easy_top25_output_gain": statistics.mean(row["output_gain"] for row in easy),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row["output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["output_gain"] <= -0.20 for row in rows),
    }


def gate_checks(summary, noop, cache_rows):
    cache_max = 0.0
    for row in cache_rows:
        for key, value in row.items():
            if key.endswith("_max_abs_diff"):
                cache_max = max(cache_max, float(value))
    return {
        "initial_noop_max_abs_diff": {
            "observed": noop["initial_output_max_abs_diff_vs_anchor"],
            "required": "<= 1e-6",
            "pass": noop["initial_output_max_abs_diff_vs_anchor"] <= 1e-6,
        },
        "cache_patch_max_abs_diff": {
            "observed": cache_max,
            "required": "<= 1e-8",
            "pass": cache_max <= 1e-8,
        },
        "backbone_selector_trainable_params": {
            "observed": noop["apdr_trainable_params"],
            "required": "== 0",
            "pass": noop["apdr_trainable_params"] == 0,
        },
        "weighted_delta_l1_drop": {
            "observed": summary["weighted_delta_l1_drop"],
            "required": ">= 0.50",
            "pass": summary["weighted_delta_l1_drop"] >= 0.50,
        },
        "pred_target_corr": {
            "observed": summary["pred_target_corr"],
            "required": ">= 0.50",
            "pass": summary["pred_target_corr"] is not None and summary["pred_target_corr"] >= 0.50,
        },
        "oracle_recovery": {
            "observed": summary["oracle_recovery"],
            "required": ">= 0.30",
            "pass": summary["oracle_recovery"] >= 0.30,
        },
        "hard_train_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": ">= +0.30 dB",
            "pass": summary["hard_bottom25_output_gain"] >= 0.30,
        },
        "easy_train_gain": {
            "observed": summary["easy_top25_output_gain"],
            "required": ">= -0.010 dB",
            "pass": summary["easy_top25_output_gain"] >= -0.010,
        },
        "strong_reference_regressions": {
            "observed": summary["strong_reference_regressions"],
            "required": "== 0",
            "pass": summary["strong_reference_regressions"] == 0,
        },
        "severe_regressions": {
            "observed": summary["severe_regressions"],
            "required": "== 0",
            "pass": summary["severe_regressions"] == 0,
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4a_lowfield_overfit32_seed3407")
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--hidden_channels", type=int, default=48)
    parser.add_argument(
        "--model_type",
        default="lowfield",
        choices=("lowfield", "id_embedding", "basis", "basis_local", "veil"),
    )
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--num_bases", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lowpass_lambda", type=float, default=0.05)
    parser.add_argument("--tv_lambda", type=float, default=0.001)
    parser.add_argument("--progress_freq", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau, scores = read_correctability(args.correctability_json, args.correctability_train_csv)
    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    apdr_trainable = sum(param.numel() for param in apdr_model.parameters() if param.requires_grad)
    loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=False)
    records, cache_rows = load_records(
        apdr_model,
        loader,
        device,
        args,
        tau,
        scores,
        output_dir / "tensor_cache" / args.tag,
    )

    model = build_probe(args, len(records)).to(device)
    trainable_lowfield = sum(param.numel() for param in model.parameters() if param.requires_grad)
    initial_rows, initial_loss, initial_corr, initial_amp = evaluate(model, records, device, args)
    noop = {
        "initial_output_max_abs_diff_vs_anchor": max(
            row["output_max_abs_diff_vs_anchor"] for row in initial_rows
        ),
        "initial_weighted_delta_l1": initial_loss,
        "initial_pred_target_corr": initial_corr,
        "apdr_trainable_params": apdr_trainable,
        "lowfield_trainable_params": trainable_lowfield,
        "correctability_tau": tau,
        "open_count": sum(row["P_benefit"] >= 0.5 for row in initial_rows),
        "amplitude": initial_amp,
    }

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed + 17)
    history = [
        {
            "step": 0,
            "weighted_delta_l1": initial_loss,
            "pred_target_corr": initial_corr,
            "mean_output_gain": statistics.mean(row["output_gain"] for row in initial_rows),
        }
    ]
    active_indices = [idx for idx, record in enumerate(records) if record["p_benefit"] >= 0.5]
    if not active_indices:
        raise RuntimeError("No open samples in overfit32; check correctability threshold.")

    for step in range(1, args.steps + 1):
        record = records[active_indices[(step - 1) % len(active_indices)]]
        item = train_step(model, record, device, args, rng)
        if item is None:
            continue
        loss, _parts = item
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if args.progress_freq and step % args.progress_freq == 0:
            rows, eval_loss, eval_corr, _amp = evaluate(model, records, device, args)
            mean_gain = statistics.mean(row["output_gain"] for row in rows)
            history.append(
                {
                    "step": step,
                    "weighted_delta_l1": eval_loss,
                    "pred_target_corr": eval_corr,
                    "mean_output_gain": mean_gain,
                }
            )
            print(
                f"step={step} weighted_delta_l1={eval_loss:.6f} "
                f"corr={eval_corr if eval_corr is not None else float('nan'):.4f} "
                f"mean_gain={mean_gain:.4f}",
                flush=True,
            )

    final_rows, final_loss, final_corr, amplitude = evaluate(model, records, device, args)
    summary = summarize(final_rows, initial_loss, final_loss, final_corr)
    groups = opened_closed_groups(final_rows)
    checks = gate_checks(summary, noop, cache_rows)
    result = {
        "stage": "APDR-v0.4A LowFieldNet Gate A/B overfit32",
        "tag": args.tag,
        "model_type": args.model_type,
        "summary": summary,
        "gate_checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "noop_cache_preflight": noop,
        "amplitude_audit": amplitude,
        "opened_closed_groups": groups,
        "history": history,
        "args": vars(args),
    }

    (output_dir / f"lowfield_overfit32_summary_{args.tag}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"preflight_noop_cache_{args.tag}.json").write_text(
        json.dumps(noop, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"lowfield_amplitude_audit_{args.tag}.json").write_text(
        json.dumps(amplitude, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / f"lowfield_overfit32_per_image_{args.tag}.csv", final_rows)
    write_csv(output_dir / f"lowfield_overfit32_history_{args.tag}.csv", history)
    write_csv(output_dir / f"opened_closed_groups_{args.tag}.csv", groups)
    write_csv(output_dir / f"cache_usage_audit_{args.tag}.csv", cache_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
