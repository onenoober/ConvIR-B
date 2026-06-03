import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim

sys.path.insert(0, os.getcwd())

from data import test_dataloader
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


def ssim_value(pred, target):
    h, w = pred.shape[2], pred.shape[3]
    down_ratio = max(1, round(min(h, w) / 256))
    return ssim(
        f.adaptive_avg_pool2d(pred, (int(h / down_ratio), int(w / down_ratio))),
        f.adaptive_avg_pool2d(target, (int(h / down_ratio), int(w / down_ratio))),
        data_range=1,
        size_average=False,
    ).mean().item()


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
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


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


def summarize_variant(rows, variant):
    deltas = [row[f"{variant}_delta_psnr"] for row in rows]
    ssim_deltas = [row[f"{variant}_delta_ssim"] for row in rows]
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    count = len(rows)
    hard = ordered[: count // 4]
    easy = ordered[3 * count // 4 :]
    strong_cut = percentile([row["anchor_psnr"] for row in rows], 75)
    strong = [row for row in rows if row["anchor_psnr"] >= strong_cut]
    summary = {
        "count": count,
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p5_psnr_delta": percentile(deltas, 5),
        "p95_psnr_delta": percentile(deltas, 95),
        "hard_bottom25_mean_delta": statistics.mean(row[f"{variant}_delta_psnr"] for row in hard),
        "easy_top25_mean_delta": statistics.mean(row[f"{variant}_delta_psnr"] for row in easy),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row[f"{variant}_delta_psnr"] <= -0.05 for row in strong),
        "severe_regressions": sum(delta <= -0.20 for delta in deltas),
        "worst10_image_mean_delta": statistics.mean(sorted(deltas)[:10]),
        "best10_image_mean_delta": statistics.mean(sorted(deltas)[-10:]),
    }
    checks = {
        "oracle_mean_psnr_delta": {
            "observed": summary["mean_psnr_delta"],
            "required": ">= +0.050",
            "pass": summary["mean_psnr_delta"] >= 0.050,
        },
        "oracle_hard_bottom25_delta": {
            "observed": summary["hard_bottom25_mean_delta"],
            "required": ">= +0.150",
            "pass": summary["hard_bottom25_mean_delta"] >= 0.150,
        },
        "oracle_easy_top25_delta": {
            "observed": summary["easy_top25_mean_delta"],
            "required": ">= -0.005",
            "pass": summary["easy_top25_mean_delta"] >= -0.005,
        },
        "oracle_strong_reference_regressions": {
            "observed": summary["strong_reference_regressions"],
            "required": "<= 5 / 250",
            "pass": summary["strong_reference_regressions"] <= 5,
        },
        "oracle_severe_regressions": {
            "observed": summary["severe_regressions"],
            "required": "== 0 / 1000",
            "pass": summary["severe_regressions"] == 0,
        },
    }
    return {"summary": summary, "checks": checks, "pass": all(item["pass"] for item in checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_3_residual_source_oracle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=args.residual_max,
        apdr_gate_max=1.0,
        apdr_gate_init=0.01,
        apdr_force_zero_gate=False,
        apdr_active_scales="full",
        apdr_selector_mode="v0_2r",
        apdr_residual_capacity="linear",
    ).to(device)
    model.load_state_dict(load_model_state(args.selector_checkpoint, device), strict=True)
    model.eval()

    dataloader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    times = []
    variants = (
        "O_full_delta",
        "O_low",
        "O_high",
        "O_color",
        "O_low_plus_color",
    )

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images and idx >= args.max_images:
                break
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            model(padded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
            full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
            anchor = full["anchor"][:, :, :h, :w].clamp(0, 1)
            m_safe = full["gate"][:, :, :h, :w].clamp(0, 1)
            delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max)
            low_delta = gaussian_lowpass(delta_star, args.kernel_size, args.sigma)
            high_delta = (delta_star - low_delta).clamp(-args.residual_max, args.residual_max)
            color_delta = fit_weighted_channel_affine(anchor, label_img, m_safe, args.residual_max)
            color_residual = (delta_star - color_delta).clamp(-args.residual_max, args.residual_max)
            low_plus_color_delta = (
                color_delta + gaussian_lowpass(color_residual, args.kernel_size, args.sigma)
            ).clamp(-args.residual_max, args.residual_max)
            variant_deltas = {
                "O_full_delta": delta_star,
                "O_low": low_delta,
                "O_high": high_delta,
                "O_color": color_delta,
                "O_low_plus_color": low_plus_color_delta,
            }

            row = {
                "name": name[0],
                "anchor_psnr": psnr(anchor, label_img),
                "anchor_ssim": ssim_value(anchor, label_img),
                "m_safe_mean": m_safe.mean().item(),
                "m_safe_p95": torch.quantile(m_safe.flatten(), 0.95).item(),
                "delta_star_abs_mean": delta_star.abs().mean().item(),
                "low_delta_abs_mean": low_delta.abs().mean().item(),
                "high_delta_abs_mean": high_delta.abs().mean().item(),
                "color_delta_abs_mean": color_delta.abs().mean().item(),
                "low_plus_color_delta_abs_mean": low_plus_color_delta.abs().mean().item(),
            }
            for variant, delta in variant_deltas.items():
                pred = (anchor + m_safe * delta).clamp(0, 1)
                row[f"{variant}_psnr"] = psnr(pred, label_img)
                row[f"{variant}_ssim"] = ssim_value(pred, label_img)
                row[f"{variant}_delta_psnr"] = row[f"{variant}_psnr"] - row["anchor_psnr"]
                row[f"{variant}_delta_ssim"] = row[f"{variant}_ssim"] - row["anchor_ssim"]
            rows.append(row)
            if args.progress_freq and (idx + 1) % args.progress_freq == 0:
                mean_full = statistics.mean(row["O_full_delta_delta_psnr"] for row in rows)
                mean_low_color = statistics.mean(row["O_low_plus_color_delta_psnr"] for row in rows)
                print(
                    f"{idx + 1}/{len(dataloader)} "
                    f"full={mean_full:.4f} low_color={mean_low_color:.4f}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No rows were evaluated.")

    variant_results = {variant: summarize_variant(rows, variant) for variant in variants}
    summary = {
        "stage": "APDR-v0.3 residual source oracle ablation",
        "tag": args.tag,
        "variant_results": variant_results,
        "mask_stats": {
            "m_safe_mean": statistics.mean(row["m_safe_mean"] for row in rows),
            "m_safe_p95_mean": statistics.mean(row["m_safe_p95"] for row in rows),
            "delta_star_abs_mean": statistics.mean(row["delta_star_abs_mean"] for row in rows),
            "low_delta_abs_mean": statistics.mean(row["low_delta_abs_mean"] for row in rows),
            "high_delta_abs_mean": statistics.mean(row["high_delta_abs_mean"] for row in rows),
            "color_delta_abs_mean": statistics.mean(row["color_delta_abs_mean"] for row in rows),
            "low_plus_color_delta_abs_mean": statistics.mean(
                row["low_plus_color_delta_abs_mean"] for row in rows
            ),
        },
        "runtime": {
            "count": len(rows),
            "avg_apdr_forward_sec": statistics.mean(times),
            "median_apdr_forward_sec": statistics.median(times),
        },
        "args": vars(args),
    }

    json_path = output_dir / f"oracle_residual_source_summary_{args.tag}.json"
    csv_path = output_dir / f"oracle_residual_source_per_image_{args.tag}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
