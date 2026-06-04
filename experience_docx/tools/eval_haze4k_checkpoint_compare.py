import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim

sys.path.insert(0, os.getcwd())

from data import test_dataloader
from models.APDRConvIR import build_apdr_net
from models.ConvIR import build_net as build_convir_net
from models.DPGAConvIR import build_dpga_net


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


def build_model(arch, mode, args, prefix):
    if arch == "convir":
        return build_convir_net("base", "Haze4K", mode)
    if arch == "dpga":
        return build_dpga_net(
            "base",
            "Haze4K",
            prior_embed_channels=getattr(args, f"{prefix}_dpga_prior_embed_channels"),
            adapter_reduction=getattr(args, f"{prefix}_dpga_adapter_reduction"),
            adapter_residual_scale=getattr(args, f"{prefix}_dpga_adapter_residual_scale"),
            adapter_scale_init=getattr(args, f"{prefix}_dpga_adapter_scale_init"),
            adapter_bootstrap_scale=getattr(args, f"{prefix}_dpga_adapter_bootstrap_scale"),
            hard_gate_init_bias=getattr(args, f"{prefix}_dpga_hard_gate_init_bias"),
            dark_patch=getattr(args, f"{prefix}_dpga_dark_patch"),
            local_patch=getattr(args, f"{prefix}_dpga_local_patch"),
            active_adapters=getattr(args, f"{prefix}_dpga_active_adapters"),
            scale_multiplier=getattr(args, f"{prefix}_dpga_scale_multiplier"),
            hard_gate_mode=getattr(args, f"{prefix}_dpga_hard_gate_mode"),
            shallow_scale_multiplier=getattr(args, f"{prefix}_dpga_shallow_scale_multiplier"),
            bottleneck_scale_multiplier=getattr(args, f"{prefix}_dpga_bottleneck_scale_multiplier"),
            skip_scale_multiplier=getattr(args, f"{prefix}_dpga_skip_scale_multiplier"),
            fusion_mode=getattr(args, f"{prefix}_dpga_fusion_mode"),
            udp_components=getattr(args, f"{prefix}_dpga_udp_components"),
            udp_window_size=getattr(args, f"{prefix}_dpga_udp_window_size"),
            udp_num_heads=getattr(args, f"{prefix}_dpga_udp_num_heads"),
            agf_gate_limit=getattr(args, f"{prefix}_dpga_agf_gate_limit"),
        )
    if arch == "apdr":
        return build_apdr_net(
            "base",
            "Haze4K",
            apdr_prior_mode=getattr(args, f"{prefix}_apdr_prior_mode"),
            apdr_residual_max=getattr(args, f"{prefix}_apdr_residual_max"),
            apdr_gate_max=getattr(args, f"{prefix}_apdr_gate_max"),
            apdr_gate_init=getattr(args, f"{prefix}_apdr_gate_init"),
            apdr_force_zero_gate=getattr(args, f"{prefix}_apdr_force_zero_gate"),
            apdr_active_scales=getattr(args, f"{prefix}_apdr_active_scales"),
            apdr_selector_mode=getattr(args, f"{prefix}_apdr_selector_mode"),
            apdr_residual_capacity=getattr(args, f"{prefix}_apdr_residual_capacity"),
        )
    raise ValueError(f"Unsupported arch: {arch}")


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_candidate_state(model, checkpoint, device, arch):
    state = load_model_state(checkpoint, device)
    if arch != "dpga":
        model.load_state_dict(state)
        return
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("DPGA_hard_gate.")]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected DPGA checkpoint load result: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def dpga_depth_cache_dir(args, prefix, arch):
    if arch != "dpga":
        return ""
    depth_cache_dir = getattr(args, f"{prefix}_dpga_depth_cache_dir") or args.dpga_depth_cache_dir
    if not depth_cache_dir:
        raise ValueError(f"--{prefix}_dpga_depth_cache_dir or --dpga_depth_cache_dir is required for DPGA eval")
    return depth_cache_dir


def forward_model(model, arch, input_img, depth):
    if arch == "dpga":
        return model(input_img, depth)
    return model(input_img)


def eval_one(label, arch, mode, checkpoint, data_dir, args, prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(arch, mode, args, prefix).to(device)
    load_candidate_state(model, checkpoint, device, arch)
    model.eval()

    depth_split = args.dpga_eval_depth_split
    if arch == "dpga" and args.split_json and args.split_name and depth_split == "test":
        depth_split = "train"
    dataloader = test_dataloader(
        data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=dpga_depth_cache_dir(args, prefix, arch),
        depth_split=depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    factor = 32
    rows = []
    times = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if arch == "dpga":
                input_img, label_img, depth, name = data
                depth = depth.to(device)
            else:
                input_img, label_img, name = data
                depth = None
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            padded = f.pad(input_img, (0, padw, 0, padh), "reflect")
            if depth is not None:
                depth = f.pad(depth, (0, padw, 0, padh), "reflect")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = forward_model(model, arch, padded, depth)[2][:, :, :h, :w]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            pred = torch.clamp(pred, 0, 1)
            mse = f.mse_loss(pred, label_img)
            psnr_val = (10 * torch.log10(1 / mse)).item()

            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                f.adaptive_avg_pool2d(pred, (int(H / down_ratio), int(W / down_ratio))),
                f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1,
                size_average=False,
            ).mean().item()

            times.append(elapsed)
            rows.append(
                {
                    "name": name[0],
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "time_sec": elapsed,
                }
            )
            if (idx + 1) % 100 == 0:
                mean_psnr = statistics.mean(row["psnr"] for row in rows)
                print(f"{label} {idx + 1}/{len(dataloader)} psnr={mean_psnr:.4f}", flush=True)

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    summary = {
        "label": label,
        "arch": arch,
        "mode": mode,
        "checkpoint": checkpoint,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": peak_mem,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--original_arch", default="convir", choices=["convir", "apdr", "dpga"])
    parser.add_argument("--original_mode", default="original")
    parser.add_argument("--original_name", default="original")
    parser.add_argument("--modres_checkpoint")
    parser.add_argument("--candidate_checkpoint")
    parser.add_argument("--candidate_arch", default="convir", choices=["convir", "apdr", "dpga"])
    parser.add_argument("--candidate_mode", default="modres")
    parser.add_argument("--candidate_name")
    parser.add_argument("--original_apdr_prior_mode", default="rgb_haze", choices=["rgb_haze"])
    parser.add_argument("--original_apdr_residual_max", type=float, default=0.04)
    parser.add_argument("--original_apdr_gate_max", type=float, default=0.5)
    parser.add_argument("--original_apdr_gate_init", type=float, default=0.02)
    parser.add_argument("--original_apdr_force_zero_gate", action="store_true")
    parser.add_argument("--original_apdr_active_scales", default="all", choices=["all", "full"])
    parser.add_argument("--original_apdr_selector_mode", default="v0", choices=["v0", "v0_2", "v0_2r"])
    parser.add_argument("--original_apdr_residual_capacity", default="linear", choices=["linear", "shallow_mlp"])
    parser.add_argument("--candidate_apdr_prior_mode", default="rgb_haze", choices=["rgb_haze"])
    parser.add_argument("--candidate_apdr_residual_max", type=float, default=0.04)
    parser.add_argument("--candidate_apdr_gate_max", type=float, default=0.5)
    parser.add_argument("--candidate_apdr_gate_init", type=float, default=0.02)
    parser.add_argument("--candidate_apdr_force_zero_gate", action="store_true")
    parser.add_argument("--candidate_apdr_active_scales", default="all", choices=["all", "full"])
    parser.add_argument("--candidate_apdr_selector_mode", default="v0", choices=["v0", "v0_2", "v0_2r"])
    parser.add_argument("--candidate_apdr_residual_capacity", default="linear", choices=["linear", "shallow_mlp"])
    parser.add_argument("--dpga_depth_cache_dir", default="")
    parser.add_argument("--dpga_eval_depth_split", default="test")
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--original_dpga_depth_cache_dir", default="")
    parser.add_argument("--candidate_dpga_depth_cache_dir", default="")
    parser.add_argument("--original_dpga_prior_embed_channels", type=int, default=16)
    parser.add_argument("--candidate_dpga_prior_embed_channels", type=int, default=16)
    parser.add_argument("--original_dpga_adapter_reduction", type=int, default=2)
    parser.add_argument("--candidate_dpga_adapter_reduction", type=int, default=2)
    parser.add_argument("--original_dpga_adapter_residual_scale", type=float, default=0.1)
    parser.add_argument("--candidate_dpga_adapter_residual_scale", type=float, default=0.1)
    parser.add_argument("--original_dpga_adapter_scale_init", type=float, default=0.0)
    parser.add_argument("--candidate_dpga_adapter_scale_init", type=float, default=0.0)
    parser.add_argument("--original_dpga_adapter_bootstrap_scale", type=float, default=0.01)
    parser.add_argument("--candidate_dpga_adapter_bootstrap_scale", type=float, default=0.01)
    parser.add_argument("--original_dpga_hard_gate_init_bias", type=float, default=-3.0)
    parser.add_argument("--candidate_dpga_hard_gate_init_bias", type=float, default=-3.0)
    parser.add_argument("--original_dpga_dark_patch", type=int, default=15)
    parser.add_argument("--candidate_dpga_dark_patch", type=int, default=15)
    parser.add_argument("--original_dpga_local_patch", type=int, default=31)
    parser.add_argument("--candidate_dpga_local_patch", type=int, default=31)
    parser.add_argument("--original_dpga_active_adapters", default="all")
    parser.add_argument("--candidate_dpga_active_adapters", default="all")
    parser.add_argument("--original_dpga_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--candidate_dpga_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--original_dpga_hard_gate_mode", default="off", choices=["off", "bottleneck"])
    parser.add_argument("--candidate_dpga_hard_gate_mode", default="off", choices=["off", "bottleneck"])
    parser.add_argument("--original_dpga_shallow_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--candidate_dpga_shallow_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--original_dpga_bottleneck_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--candidate_dpga_bottleneck_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--original_dpga_skip_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--candidate_dpga_skip_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--original_dpga_fusion_mode", default="legacy", choices=["legacy", "udp_lite"])
    parser.add_argument("--candidate_dpga_fusion_mode", default="legacy", choices=["legacy", "udp_lite"])
    parser.add_argument("--original_dpga_udp_components", default="all")
    parser.add_argument("--candidate_dpga_udp_components", default="all")
    parser.add_argument("--original_dpga_udp_window_size", type=int, default=8)
    parser.add_argument("--candidate_dpga_udp_window_size", type=int, default=8)
    parser.add_argument("--original_dpga_udp_num_heads", type=int, default=4)
    parser.add_argument("--candidate_dpga_udp_num_heads", type=int, default=4)
    parser.add_argument("--original_dpga_agf_gate_limit", type=float, default=0.25)
    parser.add_argument("--candidate_dpga_agf_gate_limit", type=float, default=0.25)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="seed3407")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_checkpoint = args.candidate_checkpoint or args.modres_checkpoint
    if not candidate_checkpoint:
        raise ValueError("Provide --candidate_checkpoint or --modres_checkpoint")
    candidate_name = args.candidate_name or args.candidate_mode
    runs = [
        (args.original_name, args.original_arch, args.original_mode, args.original_checkpoint, "original"),
        (candidate_name, args.candidate_arch, args.candidate_mode, candidate_checkpoint, "candidate"),
    ]

    all_rows = {}
    summaries = {}
    for label, arch, mode, checkpoint, prefix in runs:
        rows, summary = eval_one(label, arch, mode, checkpoint, args.data_dir, args, prefix)
        all_rows[label] = rows
        summaries[label] = summary

    original = {row["name"]: row for row in all_rows[args.original_name]}
    candidate = {row["name"]: row for row in all_rows[candidate_name]}
    common = [name for name in original if name in candidate]
    deltas = [candidate[name]["psnr"] - original[name]["psnr"] for name in common]
    ssim_deltas = [candidate[name]["ssim"] - original[name]["ssim"] for name in common]

    sorted_by_original = sorted(common, key=lambda name: original[name]["psnr"])
    bucket_count = max(1, len(common) // 4)
    hard = sorted_by_original[:bucket_count]
    easy = sorted_by_original[-bucket_count:]
    strong_cut = percentile([original[name]["psnr"] for name in common], 75)
    strong = [name for name in common if original[name]["psnr"] >= strong_cut]
    strong_regressions = [
        name for name in strong if (candidate[name]["psnr"] - original[name]["psnr"]) <= -0.05
    ]
    worst_regressions = [
        name for name in common if (candidate[name]["psnr"] - original[name]["psnr"]) <= -0.20
    ]

    tail_count = max(1, len(deltas) // 10)
    sorted_deltas = sorted(deltas)
    summary = {
        "runs": summaries,
        "comparison": {
            "common_count": len(common),
            "mean_psnr_delta": statistics.mean(deltas),
            "median_psnr_delta": statistics.median(deltas),
            "p5_psnr_delta": percentile(deltas, 5),
            "p95_psnr_delta": percentile(deltas, 95),
            "hard_bottom25_psnr_delta": statistics.mean(
                candidate[name]["psnr"] - original[name]["psnr"] for name in hard
            ),
            "easy_top25_psnr_delta": statistics.mean(
                candidate[name]["psnr"] - original[name]["psnr"] for name in easy
            ),
            "worst10pct_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
            "best10pct_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
            "worst10img_mean_psnr_delta": statistics.mean(sorted_deltas[:10]),
            "best10img_mean_psnr_delta": statistics.mean(sorted_deltas[-10:]),
            "worst10_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
            "best10_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
            "mean_ssim_delta": statistics.mean(ssim_deltas),
            "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
            "strong_reference_cut_psnr": strong_cut,
            "strong_reference_count": len(strong),
            "strong_regression_count_delta_le_-0.05": len(strong_regressions),
            "worst_regression_count_delta_le_-0.20": len(worst_regressions),
        },
    }

    json_path = os.path.join(args.output_dir, f"scout_eval_compare_{args.tag}.json")
    csv_path = os.path.join(args.output_dir, f"scout_eval_per_image_{args.tag}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "original_psnr",
                f"{candidate_name}_psnr",
                "delta_psnr",
                "original_ssim",
                f"{candidate_name}_ssim",
                "delta_ssim",
                "original_time_sec",
                f"{candidate_name}_time_sec",
            ]
        )
        for name in common:
            writer.writerow(
                [
                    name,
                    original[name]["psnr"],
                    candidate[name]["psnr"],
                    candidate[name]["psnr"] - original[name]["psnr"],
                    original[name]["ssim"],
                    candidate[name]["ssim"],
                    candidate[name]["ssim"] - original[name]["ssim"],
                    original[name]["time_sec"],
                    candidate[name]["time_sec"],
                ]
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
