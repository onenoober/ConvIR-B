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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())

from data.data_load import DeblurDataset
from models.APDRConvIR import build_apdr_net


def psnr(pred, target):
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return (10 * torch.log10(1 / mse)).item()


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


def correlation(x, y, weight=None):
    x = x.flatten()
    y = y.flatten()
    if weight is not None:
        weight = weight.flatten().clamp_min(0)
        keep = weight > 0
        x = x[keep]
        y = y[keep]
        weight = weight[keep]
    if x.numel() < 2:
        return None
    if weight is None:
        x = x - x.mean()
        y = y - y.mean()
        denom = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denom.item() == 0:
            return None
        return (x * y).sum().div(denom).item()
    weight = weight / weight.sum().clamp_min(1e-12)
    mean_x = (weight * x).sum()
    mean_y = (weight * y).sum()
    xc = x - mean_x
    yc = y - mean_y
    denom = (weight * xc.square()).sum().sqrt() * (weight * yc.square()).sum().sqrt()
    if denom.item() == 0:
        return None
    return (weight * xc * yc).sum().div(denom).item()


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_lowpass(x, kernel_size, sigma):
    if kernel_size <= 1:
        return x
    kernel = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x_blur = F.conv2d(x_pad, kx, groups=channels)
    x_pad = F.pad(x_blur, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x_pad, ky, groups=channels)


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        return F.pad(input_img, (0, padw, 0, padh), "reflect"), h, w
    return input_img, h, w


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


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
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_loader(data_dir, count, num_workers, shuffle):
    train_dir = Path(data_dir) / "train"
    dataset = DeblurDataset(str(train_dir), "Haze4K", transform=None, is_test=True)
    if count > 0:
        dataset = Subset(dataset, list(range(min(count, len(dataset)))))
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def fit_weighted_channel_affine(anchor, target, weight, residual_max):
    weights = weight.expand_as(anchor).clamp_min(1e-6)
    xs = anchor.flatten(2)
    ys = target.flatten(2)
    ws = weights.flatten(2)
    sum_w = ws.sum(dim=2).clamp_min(1e-12)
    sum_x = (ws * xs).sum(dim=2)
    sum_y = (ws * ys).sum(dim=2)
    sum_xx = (ws * xs * xs).sum(dim=2)
    sum_xy = (ws * xs * ys).sum(dim=2)
    denom = (sum_w * sum_xx - sum_x * sum_x).clamp_min(1e-12)
    scale = (sum_w * sum_xy - sum_x * sum_y) / denom
    bias = (sum_y - scale * sum_x) / sum_w
    scale = scale.view(anchor.shape[0], anchor.shape[1], 1, 1)
    bias = bias.view(anchor.shape[0], anchor.shape[1], 1, 1)
    corrected = scale * anchor + bias
    return (corrected - anchor).clamp(-residual_max, residual_max)


def frozen_apdr_tensors(model, input_img, label_img, args):
    padded, h, w = pad_to_factor(input_img)
    with torch.no_grad():
        model(padded)
        full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
        anchor = full["anchor"][:, :, :h, :w].detach().clamp(0, 1)
        m_safe = full["gate"][:, :, :h, :w].detach().clamp(0, 1)
        delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max)
        low_delta = gaussian_lowpass(delta_star, args.kernel_size, args.sigma)
        color_delta = fit_weighted_channel_affine(anchor, label_img, m_safe, args.residual_max)
    return anchor, m_safe, delta_star, low_delta, color_delta


class FreeParamLowColor(nn.Module):
    def __init__(self, count, low_size, residual_max):
        super().__init__()
        self.low_size = int(low_size)
        self.residual_max = float(residual_max)
        self.low_raw = nn.Parameter(torch.zeros(count, 3, self.low_size, self.low_size))
        self.color_scale_raw = nn.Parameter(torch.zeros(count, 3))
        self.color_bias_raw = nn.Parameter(torch.zeros(count, 3))

    def forward(self, index, anchor):
        raw = self.low_raw[index : index + 1]
        low_delta = self.residual_max * torch.tanh(raw)
        low_delta = F.interpolate(low_delta, size=anchor.shape[2:], mode="bilinear", align_corners=False)
        scale = 1.0 + 0.20 * torch.tanh(self.color_scale_raw[index : index + 1]).view(1, 3, 1, 1)
        bias = self.residual_max * torch.tanh(self.color_bias_raw[index : index + 1]).view(1, 3, 1, 1)
        color_delta = (scale * anchor + bias - anchor).clamp(-self.residual_max, self.residual_max)
        return low_delta, color_delta


def collect_eval(apdr_model, probe, loader, device, args, mode):
    rows = []
    loss_num = 0.0
    loss_den = 0.0
    corrs = []
    with torch.no_grad():
        for input_img, label_img, name in loader:
            index = len(rows)
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            anchor, m_safe, delta_star, low_target, color_target = frozen_apdr_tensors(
                apdr_model,
                input_img,
                label_img,
                args,
            )
            pred_low, pred_color = probe(index, anchor)
            if mode == "low":
                pred_delta = pred_low
                target = low_target
            elif mode == "color":
                pred_delta = pred_color
                target = color_target
            else:
                pred_delta = (pred_low + pred_color).clamp(-args.residual_max, args.residual_max)
                target = (low_target + color_target).clamp(-args.residual_max, args.residual_max)
            weight = m_safe.expand_as(pred_delta)
            loss_num += (weight * (pred_delta - target).abs()).sum().item()
            loss_den += weight.sum().item()
            corr = correlation(pred_delta.cpu(), target.cpu(), weight.cpu())
            if corr is not None:
                corrs.append(corr)
            output = (anchor + m_safe * pred_delta).clamp(0, 1)
            oracle = (anchor + m_safe * target).clamp(0, 1)
            full_oracle = (anchor + m_safe * delta_star).clamp(0, 1)
            anchor_psnr = psnr(anchor, label_img)
            output_psnr = psnr(output, label_img)
            oracle_psnr = psnr(oracle, label_img)
            rows.append(
                {
                    "name": name[0],
                    "index": index,
                    "anchor_psnr": anchor_psnr,
                    "output_psnr": output_psnr,
                    "oracle_psnr": oracle_psnr,
                    "full_oracle_psnr": psnr(full_oracle, label_img),
                    "output_gain": output_psnr - anchor_psnr,
                    "oracle_gain": oracle_psnr - anchor_psnr,
                    "full_oracle_gain": psnr(full_oracle, label_img) - anchor_psnr,
                    "corr": corr,
                    "m_safe_mean": m_safe.mean().item(),
                    "pred_delta_abs_mean": pred_delta.abs().mean().item(),
                    "target_abs_mean": target.abs().mean().item(),
                }
            )
    return rows, loss_num / max(loss_den, 1e-12), (statistics.mean(corrs) if corrs else None)


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
        "initial_weighted_target_l1": initial_loss,
        "final_weighted_target_l1": final_loss,
        "loss_drop_fraction": (initial_loss - final_loss) / max(initial_loss, 1e-12),
        "mean_output_gain": statistics.mean(gains),
        "mean_oracle_gain": statistics.mean(oracle_gains),
        "oracle_gain_recovery": recovery,
        "mean_corr_pred_target": final_corr,
        "hard_bottom25_output_gain": statistics.mean(row["output_gain"] for row in hard),
        "hard_bottom25_oracle_gain": statistics.mean(row["oracle_gain"] for row in hard),
        "easy_top25_output_gain": statistics.mean(row["output_gain"] for row in easy),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row["output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["output_gain"] <= -0.20 for row in rows),
        "mean_pred_delta_abs": statistics.mean(row["pred_delta_abs_mean"] for row in rows),
        "mean_target_abs": statistics.mean(row["target_abs_mean"] for row in rows),
    }


def gate_checks(summary, mode):
    if mode == "color":
        hard_required = 0.35
    elif mode == "low":
        hard_required = 0.60
    else:
        hard_required = 0.60
    return {
        "loss_drop_fraction": {
            "observed": summary["loss_drop_fraction"],
            "required": ">= 0.80",
            "pass": summary["loss_drop_fraction"] >= 0.80,
        },
        "oracle_gain_recovery": {
            "observed": summary["oracle_gain_recovery"],
            "required": ">= 0.80",
            "pass": summary["oracle_gain_recovery"] >= 0.80,
        },
        "corr_pred_target": {
            "observed": summary["mean_corr_pred_target"],
            "required": ">= 0.70",
            "pass": summary["mean_corr_pred_target"] is not None and summary["mean_corr_pred_target"] >= 0.70,
        },
        "hard_train_psnr_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": f">= +{hard_required:.2f} dB",
            "pass": summary["hard_bottom25_output_gain"] >= hard_required,
        },
        "easy_train_psnr_gain": {
            "observed": summary["easy_top25_output_gain"],
            "required": ">= -0.010 dB",
            "pass": summary["easy_top25_output_gain"] >= -0.010,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4_freeparam_lowcolor_32")
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--mode", default="low_color", choices=("low", "color", "low_color"))
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--progress_freq", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    eval_loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=False)
    probe = FreeParamLowColor(len(eval_loader), args.low_size, args.residual_max).to(device)
    optimizer = torch.optim.Adam([param for param in probe.parameters() if param.requires_grad], lr=args.learning_rate)

    initial_rows, initial_loss, initial_corr = collect_eval(
        apdr_model,
        probe,
        eval_loader,
        device,
        args,
        args.mode,
    )
    history = [
        {
            "step": 0,
            "weighted_target_l1": initial_loss,
            "corr": initial_corr,
            "mean_output_gain": statistics.mean(row["output_gain"] for row in initial_rows),
        }
    ]

    cached_batches = []
    with torch.no_grad():
        for input_img, label_img, name in eval_loader:
            cached_batches.append((input_img.to(device), label_img.to(device), name[0]))

    for step in range(1, args.steps + 1):
        index = (step - 1) % len(cached_batches)
        input_img, label_img, _ = cached_batches[index]
        anchor, m_safe, _, low_target, color_target = frozen_apdr_tensors(
            apdr_model,
            input_img,
            label_img,
            args,
        )
        pred_low, pred_color = probe(index, anchor)
        if args.mode == "low":
            pred_delta = pred_low
            target = low_target
        elif args.mode == "color":
            pred_delta = pred_color
            target = color_target
        else:
            pred_delta = (pred_low + pred_color).clamp(-args.residual_max, args.residual_max)
            target = (low_target + color_target).clamp(-args.residual_max, args.residual_max)
        weight = m_safe.expand_as(pred_delta)
        loss = (weight * (pred_delta - target).abs()).sum() / weight.sum().clamp_min(1e-12)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip_norm)
        optimizer.step()
        if args.progress_freq and step % args.progress_freq == 0:
            rows, eval_loss, eval_corr = collect_eval(apdr_model, probe, eval_loader, device, args, args.mode)
            mean_gain = statistics.mean(row["output_gain"] for row in rows)
            history.append(
                {
                    "step": step,
                    "weighted_target_l1": eval_loss,
                    "corr": eval_corr,
                    "mean_output_gain": mean_gain,
                }
            )
            print(
                f"step={step} weighted_target_l1={eval_loss:.6f} "
                f"corr={eval_corr if eval_corr is not None else float('nan'):.4f} "
                f"mean_gain={mean_gain:.4f}",
                flush=True,
            )

    final_rows, final_loss, final_corr = collect_eval(apdr_model, probe, eval_loader, device, args, args.mode)
    summary = summarize(final_rows, initial_loss, final_loss, final_corr)
    summary["initial_corr_pred_target"] = initial_corr
    checks = gate_checks(summary, args.mode)
    result = {
        "stage": "APDR-v0.4 CCLF free-parameter low/color sanity",
        "tag": args.tag,
        "mode": args.mode,
        "summary": summary,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "history": history,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"freeparam_lowcolor_{args.tag}.json"
    csv_path = output_dir / f"freeparam_lowcolor_per_image_{args.tag}.csv"
    history_path = output_dir / f"freeparam_lowcolor_history_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {history_path}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
