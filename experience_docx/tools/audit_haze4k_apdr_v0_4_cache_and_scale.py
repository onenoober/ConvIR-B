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


def build_loader(data_dir, split, count, num_workers):
    image_dir = Path(data_dir) / split
    dataset = DeblurDataset(str(image_dir), "Haze4K", transform=None, is_test=True)
    if count > 0:
        dataset = Subset(dataset, list(range(min(count, len(dataset)))))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


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
    return (corrected - anchor).clamp(-residual_max, residual_max), scale, bias


def full_tensors(model, input_img, label_img, args):
    padded, h, w = pad_to_factor(input_img)
    model(padded)
    full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
    anchor = full["anchor"][:, :, :h, :w].detach().clamp(0, 1)
    m_safe = full["gate"][:, :, :h, :w].detach().clamp(0, 1)
    spatial = full.get("spatial_gate", m_safe)[:, :, :h, :w].detach().clamp(0, 1)
    global_budget = full.get("global_budget_unit", full.get("global_gate"))
    global_score = full.get("global_score_unit", full.get("global_logits"))
    delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max)
    color_delta, color_scale, color_bias = fit_weighted_channel_affine(
        anchor,
        label_img,
        m_safe,
        args.residual_max,
    )
    return {
        "anchor": anchor,
        "m_safe": m_safe,
        "spatial_gate": spatial,
        "global_budget": global_budget.detach().float().mean().item() if global_budget is not None else None,
        "global_score": global_score.detach().float().mean().item() if global_score is not None else None,
        "delta_star": delta_star,
        "color_delta": color_delta,
        "color_scale": color_scale,
        "color_bias": color_bias,
    }


def safe_stem(name):
    return Path(name).stem.replace("/", "_").replace("\\", "_").replace(" ", "_")


def choose_crop(rng, height, width, crop_size):
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y = 0 if height <= crop_h else rng.randint(0, height - crop_h)
    x = 0 if width <= crop_w else rng.randint(0, width - crop_w)
    return y, x, crop_h, crop_w


def write_cache(cache_path, name, split, tensors, low_targets, args):
    payload = {
        "name": name,
        "split": split,
        "residual_max": args.residual_max,
        "anchor": tensors["anchor"].squeeze(0).float().cpu(),
        "m_safe": tensors["m_safe"].squeeze(0).float().cpu(),
        "spatial_gate": tensors["spatial_gate"].squeeze(0).float().cpu(),
        "delta_star": tensors["delta_star"].squeeze(0).float().cpu(),
        "color_delta": tensors["color_delta"].squeeze(0).float().cpu(),
        "color_scale": tensors["color_scale"].squeeze(0).float().cpu(),
        "color_bias": tensors["color_bias"].squeeze(0).float().cpu(),
        "low_targets": {
            key: value.squeeze(0).float().cpu() for key, value in low_targets.items()
        },
    }
    torch.save(payload, cache_path)


def cache_roundtrip_audit(cache_path, tensors, low_targets, rng, crop_size):
    loaded = torch.load(cache_path, map_location="cpu")
    anchor = tensors["anchor"].squeeze(0).cpu()
    m_safe = tensors["m_safe"].squeeze(0).cpu()
    color_delta = tensors["color_delta"].squeeze(0).cpu()
    height, width = anchor.shape[1], anchor.shape[2]
    y, x, crop_h, crop_w = choose_crop(rng, height, width, crop_size)
    result = {
        "cache_anchor_max_abs_diff": (loaded["anchor"].float() - anchor).abs().max().item(),
        "cache_m_safe_max_abs_diff": (loaded["m_safe"].float() - m_safe).abs().max().item(),
        "cache_color_max_abs_diff": (loaded["color_delta"].float() - color_delta).abs().max().item(),
        "crop_x": x,
        "crop_y": y,
        "crop_h": crop_h,
        "crop_w": crop_w,
    }
    full_patch = m_safe[:, y : y + crop_h, x : x + crop_w]
    cache_patch = loaded["m_safe"].float()[:, y : y + crop_h, x : x + crop_w]
    result["cached_crop_mask_max_abs_diff"] = (full_patch - cache_patch).abs().max().item()
    for key, value in low_targets.items():
        full_low_patch = value.squeeze(0).cpu()[:, y : y + crop_h, x : x + crop_w]
        cached_low_patch = loaded["low_targets"][key].float()[:, y : y + crop_h, x : x + crop_w]
        result[f"cached_crop_{key}_max_abs_diff"] = (
            full_low_patch - cached_low_patch
        ).abs().max().item()
    return result


def summarize_scale(rows, scale_keys):
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    easy = ordered[3 * len(ordered) // 4 :]
    summary = {}
    for key in scale_keys:
        deltas = [row[f"{key}_gain"] for row in rows]
        summary[key] = {
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "p05_delta": percentile(deltas, 5),
            "p95_delta": percentile(deltas, 95),
            "hard_bottom25_delta": statistics.mean(row[f"{key}_gain"] for row in hard),
            "easy_top25_delta": statistics.mean(row[f"{key}_gain"] for row in easy),
            "target_abs_mean": statistics.mean(row[f"{key}_target_abs_mean"] for row in rows),
            "leftover_abs_mean": statistics.mean(row[f"{key}_leftover_abs_mean"] for row in rows),
        }
    return summary


def summarize_cache(cache_rows):
    if not cache_rows:
        return {"enabled": False}
    max_keys = [key for key in cache_rows[0] if key.endswith("_max_abs_diff")]
    summary = {"enabled": True, "count": len(cache_rows)}
    for key in max_keys:
        values = [row[key] for row in cache_rows]
        summary[key] = {
            "max": max(values),
            "mean": statistics.mean(values),
        }
    checks = {
        "cached_crop_mask_exact": {
            "observed": summary["cached_crop_mask_max_abs_diff"]["max"],
            "required": "<= 1e-8 after float32 cache roundtrip",
            "pass": summary["cached_crop_mask_max_abs_diff"]["max"] <= 1e-8,
        }
    }
    low_keys = [key for key in max_keys if key.startswith("cached_crop_low_")]
    for key in low_keys:
        checks[f"{key}_exact"] = {
            "observed": summary[key]["max"],
            "required": "<= 1e-8 after float32 cache roundtrip",
            "pass": summary[key]["max"] <= 1e-8,
        }
    summary["checks"] = checks
    summary["pass"] = all(item["pass"] for item in checks.values())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--tag", default="apdr_v0_4_cache_scale")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--write_cache", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sigmas", default="3,5,7,11,15")
    parser.add_argument("--kernel_size_factor", type=float, default=4.0)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--progress_freq", type=int, default=25)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    sigmas = [float(item) for item in args.sigmas.split(",") if item.strip()]
    scale_keys = [f"low_sigma_{str(sigma).replace('.', 'p')}" for sigma in sigmas]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "tensor_cache"
    if args.write_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    loader = build_loader(args.data_dir, args.split, args.max_images, args.num_workers)
    rows = []
    cache_rows = []
    manifest = []

    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tensors = full_tensors(model, input_img, label_img, args)
            anchor = tensors["anchor"]
            m_safe = tensors["m_safe"]
            delta_star = tensors["delta_star"]
            color_delta = tensors["color_delta"]
            anchor_psnr = psnr(anchor, label_img)
            color_pred = (anchor + m_safe * color_delta).clamp(0, 1)
            row = {
                "name": name[0],
                "index": idx,
                "anchor_psnr": anchor_psnr,
                "m_safe_mean": m_safe.mean().item(),
                "m_safe_p95": torch.quantile(m_safe.flatten(), 0.95).item(),
                "delta_star_abs_mean": delta_star.abs().mean().item(),
                "color_gain": psnr(color_pred, label_img) - anchor_psnr,
                "color_target_abs_mean": color_delta.abs().mean().item(),
                "global_budget": tensors["global_budget"],
                "global_score": tensors["global_score"],
            }
            low_targets = {}
            for sigma, key in zip(sigmas, scale_keys):
                kernel_size = int(math.ceil(args.kernel_size_factor * sigma) * 2 + 1)
                low_delta = gaussian_lowpass(delta_star, kernel_size, sigma)
                pred = (anchor + m_safe * low_delta).clamp(0, 1)
                row[f"{key}_gain"] = psnr(pred, label_img) - anchor_psnr
                row[f"{key}_target_abs_mean"] = low_delta.abs().mean().item()
                row[f"{key}_leftover_abs_mean"] = (delta_star - low_delta).abs().mean().item()
                low_targets[key] = low_delta
            rows.append(row)

            cache_rel = ""
            if args.write_cache:
                cache_path = cache_dir / f"{args.split}_{idx:05d}_{safe_stem(name[0])}.pt"
                write_cache(cache_path, name[0], args.split, tensors, low_targets, args)
                cache_rel = os.path.relpath(cache_path, output_dir)
                audit_row = {"name": name[0], "index": idx, **cache_roundtrip_audit(cache_path, tensors, low_targets, rng, args.crop_size)}
                cache_rows.append(audit_row)
            manifest.append({"name": name[0], "index": idx, "cache_path": cache_rel})

            if args.progress_freq and (idx + 1) % args.progress_freq == 0:
                best_key = max(scale_keys, key=lambda key: statistics.mean(r[f"{key}_gain"] for r in rows))
                print(
                    f"processed={idx + 1} best_scale={best_key} "
                    f"mean_gain={statistics.mean(r[f'{best_key}_gain'] for r in rows):.4f}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No rows were evaluated.")

    result = {
        "stage": "APDR-v0.4 CCLF cache sanity and lowpass scale sweep",
        "tag": args.tag,
        "summary": {
            "count": len(rows),
            "split": args.split,
            "anchor_psnr_mean": statistics.mean(row["anchor_psnr"] for row in rows),
            "m_safe_mean": statistics.mean(row["m_safe_mean"] for row in rows),
            "color": {
                "mean_delta": statistics.mean(row["color_gain"] for row in rows),
                "hard_bottom25_delta": statistics.mean(
                    row["color_gain"] for row in sorted(rows, key=lambda row: row["anchor_psnr"])[: max(1, len(rows) // 4)]
                ),
                "easy_top25_delta": statistics.mean(
                    row["color_gain"] for row in sorted(rows, key=lambda row: row["anchor_psnr"])[3 * len(rows) // 4 :]
                ),
                "target_abs_mean": statistics.mean(row["color_target_abs_mean"] for row in rows),
            },
            "lowpass": summarize_scale(rows, scale_keys),
            "cache": summarize_cache(cache_rows),
        },
        "manifest": manifest,
        "args": vars(args),
    }

    json_path = output_dir / f"cache_scale_summary_{args.tag}.json"
    csv_path = output_dir / f"cache_scale_per_image_{args.tag}.csv"
    manifest_path = output_dir / f"cache_manifest_{args.tag}.json"
    cache_audit_path = output_dir / f"cache_roundtrip_audit_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if cache_rows:
        with cache_audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cache_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cache_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
    if cache_rows:
        print(f"wrote {cache_audit_path}")
    cache_summary = result["summary"]["cache"]
    if cache_summary.get("enabled") and not cache_summary.get("pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
