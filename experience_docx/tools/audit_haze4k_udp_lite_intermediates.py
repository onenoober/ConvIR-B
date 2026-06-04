#!/usr/bin/env python3
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
import torch.nn.functional as F
from pytorch_msssim import ssim


def add_repo_imports(its_dir):
    its_dir = os.path.abspath(its_dir)
    if its_dir not in sys.path:
        sys.path.insert(0, its_dir)


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


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def psnr_from_mse(mse):
    return 10.0 * torch.log10(torch.tensor(1.0, device=mse.device, dtype=mse.dtype) / mse)


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_magnitude(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def dark_channel(x, patch=15):
    dark = x.min(dim=1, keepdim=True).values
    return -F.max_pool2d(-dark, kernel_size=patch, stride=1, padding=patch // 2)


def make_input_depth_stats(input_img, depth, anchor=None):
    gray = luma(input_img)
    grad = gradient_magnitude(gray)
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    depth_grad = gradient_magnitude(depth)
    sky_proxy = ((gray >= 0.66) & (grad <= 0.05) & (saturation <= 0.20)).float()
    stats = {
        "brightness": float(gray.mean().item()),
        "saturation": float(saturation.mean().item()),
        "gradient": float(grad.mean().item()),
        "dark_channel": float(dark_channel(input_img).mean().item()),
        "sky_proxy": float(sky_proxy.mean().item()),
        "depth_mean": float(depth.mean().item()),
        "depth_gradient": float(depth_grad.mean().item()),
    }
    if anchor is not None:
        stats["input_anchor_residual_l1"] = float((input_img - anchor).abs().mean().item())
        stats["input_anchor_residual_l2"] = float(torch.sqrt(torch.mean((input_img - anchor) ** 2) + 1e-12).item())
    return stats


def pad_to_factor(input_img, depth, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")
        depth = F.pad(depth, (0, padw, 0, padh), "reflect")
    return input_img, depth, h, w, padded_h, padded_w


def forward_model(model, arch, input_img, depth):
    if arch == "dpga":
        return model(input_img, depth)[2]
    return model(input_img)[2]


def build_dataloader(args):
    add_repo_imports(args.its_dir)
    from data import test_dataloader

    depth_split = args.dpga_eval_depth_split
    if args.split_json and args.split_name and depth_split == "test":
        depth_split = "train"
    return test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.dpga_depth_cache_dir,
        depth_split=depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )


def build_models(args, device):
    add_repo_imports(args.its_dir)
    from models.ConvIR import build_net as build_convir_net
    from models.DPGAConvIR import build_dpga_net

    original = build_convir_net("base", "Haze4K", "original").to(device)
    original.load_state_dict(load_model_state(args.original_checkpoint, device))
    original.eval()

    def make_candidate(active_adapters, udp_components):
        model = build_dpga_net(
            "base",
            "Haze4K",
            prior_embed_channels=args.dpga_prior_embed_channels,
            adapter_reduction=args.dpga_adapter_reduction,
            adapter_residual_scale=args.dpga_adapter_residual_scale,
            adapter_scale_init=args.dpga_adapter_scale_init,
            adapter_bootstrap_scale=args.dpga_adapter_bootstrap_scale,
            active_adapters=active_adapters,
            scale_multiplier=args.dpga_scale_multiplier,
            fusion_mode=args.dpga_fusion_mode,
            udp_components=udp_components,
            udp_window_size=args.dpga_udp_window_size,
            udp_num_heads=args.dpga_udp_num_heads,
            agf_gate_limit=args.dpga_agf_gate_limit,
        ).to(device)
        model.eval()
        return model

    return original, make_candidate


def load_udp_lite_state(model, state, allow_partial=False):
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if allow_partial:
        bad_missing = [key for key in missing if not key.startswith("DPGA_")]
    else:
        bad_missing = missing
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Unexpected UDP-Lite checkpoint load result: missing={missing}, unexpected={unexpected}"
        )


def eval_model(label, model, arch, dataloader, device, max_images=0):
    rows = []
    times = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if max_images > 0 and idx >= max_images:
                break
            input_img, label_img, depth, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device)
            padded, padded_depth, h, w, padded_h, padded_w = pad_to_factor(input_img, depth)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = forward_model(model, arch, padded, padded_depth)[:, :, :h, :w]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            pred = torch.clamp(pred, 0, 1)
            mse = F.mse_loss(pred, label_img)
            psnr_val = float(psnr_from_mse(mse).item())
            down_ratio = max(1, round(min(padded_h, padded_w) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(pred, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
                F.adaptive_avg_pool2d(label_img, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
                data_range=1,
                size_average=False,
            ).mean().item()

            row = {
                "name": name[0],
                "psnr": psnr_val,
                "ssim": float(ssim_val),
                "time_sec": elapsed,
            }
            if arch == "convir":
                row.update(make_input_depth_stats(input_img, depth, anchor=pred))
            rows.append(row)
            times.append(elapsed)
            if (idx + 1) % 100 == 0:
                mean_psnr = statistics.mean(row["psnr"] for row in rows)
                print(f"{label} {idx + 1}/{len(dataloader)} mean_psnr={mean_psnr:.4f}", flush=True)
    return rows, {
        "label": label,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }


def compare_rows(original_rows, candidate_rows, checkpoint_label, variant, active_adapters, udp_components):
    original = {row["name"]: row for row in original_rows}
    candidate = {row["name"]: row for row in candidate_rows}
    common = [name for name in original if name in candidate]
    deltas = [candidate[name]["psnr"] - original[name]["psnr"] for name in common]
    ssim_deltas = [candidate[name]["ssim"] - original[name]["ssim"] for name in common]
    sorted_by_original = sorted(common, key=lambda name: original[name]["psnr"])
    bucket_count = max(1, len(common) // 4)
    hard = sorted_by_original[:bucket_count]
    easy = sorted_by_original[-bucket_count:]
    strong_cut = percentile([original[name]["psnr"] for name in common], 75)
    strong = [name for name in common if original[name]["psnr"] >= strong_cut]
    return {
        "checkpoint": checkpoint_label,
        "variant": variant,
        "active_adapters": active_adapters,
        "udp_components": udp_components,
        "common_count": len(common),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "hard_bottom25_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in hard),
        "easy_top25_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in easy),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_ref_regressions": sum((candidate[name]["psnr"] - original[name]["psnr"]) <= -0.05 for name in strong),
        "strong_regression_ratio": (
            sum((candidate[name]["psnr"] - original[name]["psnr"]) <= -0.05 for name in strong) / len(strong)
            if strong
            else 0.0
        ),
        "worst_0p20_count": sum(delta <= -0.20 for delta in deltas),
        "p5_delta": percentile(deltas, 5),
        "p95_delta": percentile(deltas, 95),
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "checkpoint",
        "variant",
        "active_adapters",
        "udp_components",
        "common_count",
        "mean_delta",
        "median_delta",
        "hard_bottom25_delta",
        "easy_top25_delta",
        "mean_ssim_delta",
        "positive_ratio",
        "strong_reference_cut_psnr",
        "strong_ref_regressions",
        "strong_regression_ratio",
        "worst_0p20_count",
        "p5_delta",
        "p95_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_image_csv(path, original_rows, variant_results):
    original = {row["name"]: row for row in original_rows}
    fieldnames = [
        "checkpoint",
        "variant",
        "active_adapters",
        "udp_components",
        "name",
        "original_psnr",
        "candidate_psnr",
        "delta_psnr",
        "original_ssim",
        "candidate_ssim",
        "delta_ssim",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for meta, rows in variant_results:
            for row in rows:
                base = original[row["name"]]
                writer.writerow(
                    {
                        **meta,
                        "name": row["name"],
                        "original_psnr": base["psnr"],
                        "candidate_psnr": row["psnr"],
                        "delta_psnr": row["psnr"] - base["psnr"],
                        "original_ssim": base["ssim"],
                        "candidate_ssim": row["ssim"],
                        "delta_ssim": row["ssim"] - base["ssim"],
                    }
                )


def failure_type(delta, is_strong, bucket):
    tags = []
    if delta <= -0.20:
        tags.append("worst_le_-0.20")
    if is_strong and delta <= -0.05:
        tags.append("strong_ref_le_-0.05")
    if bucket == "hard":
        tags.append("hard_bucket")
    if not tags:
        tags.append("non_failure_or_regular")
    return "|".join(tags)


def write_failure_audit(path, original_rows, variant_results):
    original = {row["name"]: row for row in original_rows}
    original_psnrs = [row["psnr"] for row in original_rows]
    hard_cut = percentile(original_psnrs, 25)
    easy_cut = percentile(original_psnrs, 75)
    fieldnames = [
        "checkpoint",
        "variant",
        "name",
        "a0_psnr",
        "candidate_psnr",
        "candidate_delta",
        "a0_ssim",
        "candidate_ssim",
        "ssim_delta",
        "a0_bucket",
        "is_strong_ref",
        "failure_type",
        "depth_mean",
        "depth_gradient",
        "sky_proxy",
        "brightness",
        "saturation",
        "dark_channel",
        "gradient",
        "input_anchor_residual_l1",
        "input_anchor_residual_l2",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for meta, rows in variant_results:
            for row in rows:
                base = original[row["name"]]
                delta = row["psnr"] - base["psnr"]
                bucket = "hard" if base["psnr"] <= hard_cut else "easy" if base["psnr"] >= easy_cut else "medium"
                is_strong = base["psnr"] >= easy_cut
                writer.writerow(
                    {
                        "checkpoint": meta["checkpoint"],
                        "variant": meta["variant"],
                        "name": row["name"],
                        "a0_psnr": base["psnr"],
                        "candidate_psnr": row["psnr"],
                        "candidate_delta": delta,
                        "a0_ssim": base["ssim"],
                        "candidate_ssim": row["ssim"],
                        "ssim_delta": row["ssim"] - base["ssim"],
                        "a0_bucket": bucket,
                        "is_strong_ref": int(is_strong),
                        "failure_type": failure_type(delta, is_strong, bucket),
                        "depth_mean": base.get("depth_mean"),
                        "depth_gradient": base.get("depth_gradient"),
                        "sky_proxy": base.get("sky_proxy"),
                        "brightness": base.get("brightness"),
                        "saturation": base.get("saturation"),
                        "dark_channel": base.get("dark_channel"),
                        "gradient": base.get("gradient"),
                        "input_anchor_residual_l1": base.get("input_anchor_residual_l1"),
                        "input_anchor_residual_l2": base.get("input_anchor_residual_l2"),
                    }
                )


def parse_checkpoints(values):
    checkpoints = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint spec must be label=path, got {value}")
        label, path = value.split("=", 1)
        checkpoints.append((label, path))
    return checkpoints


def variant_rows(preset):
    if preset == "v14b_runtime_component_matrix":
        return [
            ("dpfm1_channel_only", "dpfm1", "channel"),
            ("dpfm1_cross_only", "dpfm1", "cross"),
            ("dpfm1_all", "dpfm1", "all"),
            ("dpfm4_only", "dpfm4", "all"),
            ("dpfm1_plus_dpfm4", "dpfm1,dpfm4", "all"),
            ("dpfm1_plus_dpfm2", "dpfm1,dpfm2", "all"),
            ("dpfm1_plus_dpfm2_plus_dpfm4", "dpfm1,dpfm2,dpfm4", "all"),
        ]
    return [
        ("dgca_only", "dpfm", "channel"),
        ("dpfm1_only", "dpfm1", "all"),
        ("dpfm2_only", "dpfm2", "all"),
        ("dpfm4_only", "dpfm4", "all"),
        ("dpfm1_2_4", "dpfm", "all"),
        ("dpfm1_2_4_agf_lite", "dpfm,agf", "all"),
    ]


def output_names(preset):
    if preset == "v14b_runtime_component_matrix":
        return {
            "summary_csv": "v14b_runtime_component_matrix.csv",
            "per_image_csv": "v14b_runtime_component_matrix_per_image.csv",
            "failure_csv": "v14b_runtime_component_matrix_failure_audit.csv",
            "summary_json": "v14b_runtime_component_matrix_summary.json",
        }
    return {
        "summary_csv": "v14_depth_fusion_module_ablation_val.csv",
        "per_image_csv": "v14_depth_fusion_module_ablation_per_image.csv",
        "failure_csv": "v14_depth_quality_failure_audit.csv",
        "summary_json": "v14_udp_lite_intermediate_summary.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--its_dir", default="Dehazing/ITS")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", action="append", required=True, help="Repeat as label=path.")
    parser.add_argument("--dpga_depth_cache_dir", required=True)
    parser.add_argument("--dpga_eval_depth_split", default="test")
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--dpga_prior_embed_channels", type=int, default=16)
    parser.add_argument("--dpga_adapter_reduction", type=int, default=2)
    parser.add_argument("--dpga_adapter_residual_scale", type=float, default=0.1)
    parser.add_argument("--dpga_adapter_scale_init", type=float, default=0.0)
    parser.add_argument("--dpga_adapter_bootstrap_scale", type=float, default=0.01)
    parser.add_argument("--dpga_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--dpga_fusion_mode", default="udp_lite", choices=["udp_lite", "udp_bi"])
    parser.add_argument("--dpga_udp_window_size", type=int, default=8)
    parser.add_argument("--dpga_udp_num_heads", type=int, default=4)
    parser.add_argument("--dpga_agf_gate_limit", type=float, default=0.25)
    parser.add_argument("--allow_partial_dpga_load", action="store_true")
    parser.add_argument(
        "--variant_preset",
        default="v14_default",
        choices=["v14_default", "v14b_runtime_component_matrix"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model, make_candidate = build_models(args, device)
    checkpoints = parse_checkpoints(args.candidate_checkpoint)

    print("eval original A0", flush=True)
    original_rows, original_summary = eval_model(
        "a0",
        original_model,
        "convir",
        build_dataloader(args),
        device,
        max_images=args.max_images,
    )

    variants = variant_rows(args.variant_preset)
    names = output_names(args.variant_preset)
    summary_rows = []
    per_image_rows = []
    summaries = {"original": original_summary, "variants": {}}

    for checkpoint_label, checkpoint_path in checkpoints:
        state = load_model_state(checkpoint_path, device)
        for variant, active_adapters, udp_components in variants:
            label = f"{checkpoint_label}_{variant}"
            print(f"eval {label}", flush=True)
            candidate = make_candidate(active_adapters, udp_components)
            load_udp_lite_state(candidate, state, allow_partial=args.allow_partial_dpga_load)
            candidate_rows, candidate_summary = eval_model(
                label,
                candidate,
                "dpga",
                build_dataloader(args),
                device,
                max_images=args.max_images,
            )
            meta = {
                "checkpoint": checkpoint_label,
                "variant": variant,
                "active_adapters": active_adapters,
                "udp_components": udp_components,
            }
            summary = compare_rows(
                original_rows,
                candidate_rows,
                checkpoint_label,
                variant,
                active_adapters,
                udp_components,
            )
            summary_rows.append(summary)
            per_image_rows.append((meta, candidate_rows))
            summaries["variants"][label] = {
                "checkpoint": checkpoint_path,
                "active_adapters": active_adapters,
                "udp_components": udp_components,
                "eval": candidate_summary,
                "comparison": summary,
            }
            del candidate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summary_csv(output_dir / names["summary_csv"], summary_rows)
    write_per_image_csv(output_dir / names["per_image_csv"], original_rows, per_image_rows)
    write_failure_audit(output_dir / names["failure_csv"], original_rows, per_image_rows)
    manifest = {
        "args": vars(args),
        "outputs": [
            names["summary_csv"],
            names["per_image_csv"],
            names["failure_csv"],
            names["summary_json"],
        ],
        "summaries": summaries,
    }
    (output_dir / names["summary_json"]).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
