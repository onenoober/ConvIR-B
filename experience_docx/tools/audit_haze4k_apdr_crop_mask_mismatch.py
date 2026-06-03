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
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())

from data.data_load import DeblurDataset
from models.APDRConvIR import build_apdr_net


def psnr(pred, target):
    mse = f.mse_loss(pred, target).clamp_min(1e-12)
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


def correlation(x, y):
    x = x.flatten()
    y = y.flatten()
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() == 0:
        return None
    return (x * y).sum().div(denom).item()


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


def build_loader(data_dir, count, num_workers):
    train_dir = Path(data_dir) / "train"
    dataset = DeblurDataset(str(train_dir), "Haze4K", transform=None, is_test=True)
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


def full_tensors(model, input_img):
    padded, h, w = pad_to_factor(input_img)
    model(padded)
    full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
    result = {}
    for key in (
        "anchor",
        "gate",
        "spatial_gate",
        "global_gate",
        "global_budget_unit",
        "global_score_unit",
    ):
        if key in full:
            value = full[key]
            if value.ndim == 4 and value.shape[2] >= h and value.shape[3] >= w:
                value = value[:, :, :h, :w]
            result[key] = value.detach().clamp(0, 1)
    return result


def choose_crop(rng, height, width, crop_size):
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y = 0 if height <= crop_h else rng.randint(0, height - crop_h)
    x = 0 if width <= crop_w else rng.randint(0, width - crop_w)
    return y, x, crop_h, crop_w


def summarize(rows):
    corr_values = [row["mask_corr"] for row in rows if row["mask_corr"] is not None]
    abs_diffs = [row["mask_mean_abs_diff"] for row in rows]
    mean_ratios = [row["crop_recompute_mask_mean"] / max(row["full_patch_mask_mean"], 1e-12) for row in rows]
    hard_cut = percentile([row["anchor_psnr"] for row in rows], 25)
    hard_rows = [row for row in rows if row["anchor_psnr"] <= hard_cut]
    drop_rows = [
        row
        for row in hard_rows
        if row["crop_recompute_mask_mean"] < max(0.005, 0.25 * row["full_patch_mask_mean"])
    ]
    summary = {
        "crop_count": len(rows),
        "image_count": len({row["name"] for row in rows}),
        "mean_mask_corr": statistics.mean(corr_values) if corr_values else None,
        "median_mask_corr": statistics.median(corr_values) if corr_values else None,
        "p10_mask_corr": percentile(corr_values, 10) if corr_values else None,
        "mean_mask_abs_diff": statistics.mean(abs_diffs),
        "median_mask_abs_diff": statistics.median(abs_diffs),
        "mean_crop_to_full_mask_mean_ratio": statistics.mean(mean_ratios),
        "median_crop_to_full_mask_mean_ratio": statistics.median(mean_ratios),
        "hard_anchor_psnr_cut": hard_cut,
        "hard_crop_count": len(hard_rows),
        "hard_crop_budget_drop_fraction": len(drop_rows) / max(len(hard_rows), 1),
        "near_zero_crop_mask_fraction": sum(
            row["crop_recompute_mask_mean"] < 0.005 for row in rows
        )
        / max(len(rows), 1),
        "mean_full_global_budget": statistics.mean(row["full_global_budget"] for row in rows),
        "mean_crop_global_budget": statistics.mean(row["crop_global_budget"] for row in rows),
    }
    checks = {
        "mean_mask_corr": {
            "observed": summary["mean_mask_corr"],
            "required": ">= 0.80",
            "pass": summary["mean_mask_corr"] is not None and summary["mean_mask_corr"] >= 0.80,
        },
        "p10_mask_corr": {
            "observed": summary["p10_mask_corr"],
            "required": ">= 0.60",
            "pass": summary["p10_mask_corr"] is not None and summary["p10_mask_corr"] >= 0.60,
        },
        "mean_mask_abs_diff": {
            "observed": summary["mean_mask_abs_diff"],
            "required": "<= 0.020",
            "pass": summary["mean_mask_abs_diff"] <= 0.020,
        },
        "hard_crop_budget_drop_fraction": {
            "observed": summary["hard_crop_budget_drop_fraction"],
            "required": "<= 0.10",
            "pass": summary["hard_crop_budget_drop_fraction"] <= 0.10,
        },
        "near_zero_crop_mask_fraction": {
            "observed": summary["near_zero_crop_mask_fraction"],
            "required": "<= 0.10",
            "pass": summary["near_zero_crop_mask_fraction"] <= 0.10,
        },
    }
    return summary, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_crop_mask_mismatch")
    parser.add_argument("--num_images", type=int, default=128)
    parser.add_argument("--crops_per_image", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--progress_freq", type=int, default=16)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    loader = build_loader(args.data_dir, args.num_images, args.num_workers)
    rows = []
    with torch.no_grad():
        for image_idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tensors = full_tensors(model, input_img)
            anchor = tensors["anchor"]
            gate = tensors["gate"]
            spatial_gate = tensors.get("spatial_gate", gate)
            full_global = tensors.get("global_budget_unit", tensors.get("global_gate"))
            full_global_value = full_global.mean().item() if full_global is not None else float("nan")
            anchor_psnr = psnr(anchor, label_img)
            _, _, height, width = input_img.shape
            for crop_idx in range(args.crops_per_image):
                y, x, crop_h, crop_w = choose_crop(rng, height, width, args.crop_size)
                crop_input = input_img[:, :, y : y + crop_h, x : x + crop_w]
                crop_tensors = full_tensors(model, crop_input)
                crop_gate = crop_tensors["gate"]
                crop_spatial = crop_tensors.get("spatial_gate", crop_gate)
                crop_global = crop_tensors.get("global_budget_unit", crop_tensors.get("global_gate"))
                crop_global_value = (
                    crop_global.mean().item() if crop_global is not None else float("nan")
                )
                full_patch_gate = gate[:, :, y : y + crop_h, x : x + crop_w]
                full_patch_spatial = spatial_gate[:, :, y : y + crop_h, x : x + crop_w]
                mask_corr = correlation(full_patch_gate.cpu(), crop_gate.cpu())
                spatial_corr = correlation(full_patch_spatial.cpu(), crop_spatial.cpu())
                row = {
                    "name": name[0],
                    "image_idx": image_idx,
                    "crop_idx": crop_idx,
                    "x": x,
                    "y": y,
                    "crop_h": crop_h,
                    "crop_w": crop_w,
                    "anchor_psnr": anchor_psnr,
                    "full_global_budget": full_global_value,
                    "crop_global_budget": crop_global_value,
                    "full_patch_mask_mean": full_patch_gate.mean().item(),
                    "crop_recompute_mask_mean": crop_gate.mean().item(),
                    "mask_corr": mask_corr,
                    "mask_mean_abs_diff": (full_patch_gate - crop_gate).abs().mean().item(),
                    "full_patch_spatial_mean": full_patch_spatial.mean().item(),
                    "crop_spatial_mean": crop_spatial.mean().item(),
                    "spatial_corr": spatial_corr,
                    "spatial_mean_abs_diff": (full_patch_spatial - crop_spatial).abs().mean().item(),
                }
                rows.append(row)
            if args.progress_freq and (image_idx + 1) % args.progress_freq == 0:
                corr_values = [row["mask_corr"] for row in rows if row["mask_corr"] is not None]
                mean_corr = statistics.mean(corr_values) if corr_values else float("nan")
                mean_diff = statistics.mean(row["mask_mean_abs_diff"] for row in rows)
                print(
                    f"images={image_idx + 1} crops={len(rows)} "
                    f"mean_mask_corr={mean_corr:.4f} mean_abs_diff={mean_diff:.5f}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No crops were evaluated.")
    summary, checks = summarize(rows)
    result = {
        "stage": "APDR-v0.2RC full-image mask vs crop recompute mismatch audit",
        "tag": args.tag,
        "summary": summary,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"crop_mask_mismatch_{args.tag}.json"
    csv_path = output_dir / f"crop_mask_mismatch_per_crop_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
