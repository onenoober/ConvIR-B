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
    return its_dir


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


def load_dpga_state(model, state):
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("DPGA_hard_gate.")]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected DPGA checkpoint load result: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def psnr_from_mse(mse):
    return 10.0 * torch.log10(torch.tensor(1.0, device=mse.device, dtype=mse.dtype) / mse)


def masked_psnr(pred, target, mask):
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.sum().item() < 16:
        return None
    mse = ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[1])
    if mse.item() <= 1e-12:
        return 99.0
    return float(psnr_from_mse(mse).item())


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_magnitude(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def make_tail_masks(input_img):
    gray = luma(input_img)
    grad = gradient_magnitude(gray)
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    return {
        "bright_low_gradient": ((gray >= 0.62) & (grad <= 0.035)).float(),
        "low_saturation_bright": ((gray >= 0.58) & (saturation <= 0.12)).float(),
        "sky_bright_proxy": ((gray >= 0.66) & (grad <= 0.05) & (saturation <= 0.20)).float(),
    }


def pad_to_factor(input_img, depth, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")
        if depth is not None:
            depth = F.pad(depth, (0, padw, 0, padh), "reflect")
    return input_img, depth, h, w, padded_h, padded_w


def build_models(args, device):
    add_repo_imports(args.its_dir)
    from models.ConvIR import build_net as build_convir_net
    from models.DPGAConvIR import build_dpga_net

    original = build_convir_net("base", "Haze4K", "original").to(device)
    original.load_state_dict(load_model_state(args.original_checkpoint, device))
    original.eval()

    def make_candidate(active_adapters, scale_multiplier):
        model = build_dpga_net(
            "base",
            "Haze4K",
            prior_embed_channels=args.dpga_prior_embed_channels,
            adapter_reduction=args.dpga_adapter_reduction,
            adapter_residual_scale=args.dpga_adapter_residual_scale,
            adapter_scale_init=args.dpga_adapter_scale_init,
            adapter_bootstrap_scale=args.dpga_adapter_bootstrap_scale,
            hard_gate_init_bias=args.dpga_hard_gate_init_bias,
            dark_patch=args.dpga_dark_patch,
            local_patch=args.dpga_local_patch,
            active_adapters=active_adapters,
            scale_multiplier=scale_multiplier,
            hard_gate_mode=args.dpga_hard_gate_mode,
            shallow_scale_multiplier=args.dpga_shallow_scale_multiplier,
            bottleneck_scale_multiplier=args.dpga_bottleneck_scale_multiplier,
            skip_scale_multiplier=args.dpga_skip_scale_multiplier,
        ).to(device)
        model.eval()
        return model

    return original, make_candidate


def build_dataloader(args):
    add_repo_imports(args.its_dir)
    from data import test_dataloader

    return test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.dpga_depth_cache_dir,
        depth_split=args.dpga_eval_depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )


def forward_model(model, arch, input_img, depth):
    if arch == "dpga":
        return model(input_img, depth)[2]
    return model(input_img)[2]


def eval_model(label, model, arch, dataloader, device, max_images=0):
    rows = []
    times = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if max_images > 0 and idx >= max_images:
                break
            input_img, label_img, depth, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device) if arch == "dpga" else None
            masks = make_tail_masks(input_img)
            padded, depth, h, w, padded_h, padded_w = pad_to_factor(input_img, depth)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = forward_model(model, arch, padded, depth)[:, :, :h, :w]
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
            for mask_name, mask in masks.items():
                row[f"{mask_name}_psnr"] = masked_psnr(pred, label_img, mask)
                row[f"{mask_name}_pixel_ratio"] = float(mask.mean().item())
            rows.append(row)
            times.append(elapsed)
            if (idx + 1) % 100 == 0:
                print(f"{label} {idx + 1}/{len(dataloader)} mean_psnr={statistics.mean(r['psnr'] for r in rows):.4f}", flush=True)

    return rows, {
        "label": label,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }


def compare_rows(original_rows, candidate_rows, checkpoint_label, variant, active_adapters, scale_multiplier):
    original = {row["name"]: row for row in original_rows}
    candidate = {row["name"]: row for row in candidate_rows}
    common = [name for name in original if name in candidate]
    deltas = [candidate[name]["psnr"] - original[name]["psnr"] for name in common]
    ssim_deltas = [candidate[name]["ssim"] - original[name]["ssim"] for name in common]
    sorted_by_original = sorted(common, key=lambda name: original[name]["psnr"])
    hard = sorted_by_original[: max(1, len(common) // 4)]
    easy = sorted_by_original[-max(1, len(common) // 4) :]
    strong_cut = percentile([original[name]["psnr"] for name in common], 75)
    strong = [name for name in common if original[name]["psnr"] >= strong_cut]

    def mean_delta_for(names, key):
        values = [
            candidate[name][key] - original[name][key]
            for name in names
            if candidate[name].get(key) is not None and original[name].get(key) is not None
        ]
        return statistics.mean(values) if values else None

    return {
        "checkpoint": checkpoint_label,
        "variant": variant,
        "active_adapters": active_adapters,
        "scale_multiplier": scale_multiplier,
        "common_count": len(common),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "hard_bottom25_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in hard),
        "easy_top25_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in easy),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_ref_regressions": sum(
            (candidate[name]["psnr"] - original[name]["psnr"]) <= -0.05 for name in strong
        ),
        "worst_0p20_count": sum(delta <= -0.20 for delta in deltas),
        "p5_delta": percentile(deltas, 5),
        "p95_delta": percentile(deltas, 95),
        "bright_low_gradient_delta": mean_delta_for(common, "bright_low_gradient_psnr"),
        "low_saturation_bright_delta": mean_delta_for(common, "low_saturation_bright_psnr"),
        "sky_bright_proxy_delta": mean_delta_for(common, "sky_bright_proxy_psnr"),
        "hard_bright_low_gradient_delta": mean_delta_for(hard, "bright_low_gradient_psnr"),
        "easy_bright_low_gradient_delta": mean_delta_for(easy, "bright_low_gradient_psnr"),
    }


def write_summary_csv(path, rows):
    fieldnames = [
        "checkpoint",
        "variant",
        "active_adapters",
        "scale_multiplier",
        "common_count",
        "mean_delta",
        "median_delta",
        "hard_bottom25_delta",
        "easy_top25_delta",
        "mean_ssim_delta",
        "positive_ratio",
        "strong_reference_cut_psnr",
        "strong_ref_regressions",
        "worst_0p20_count",
        "p5_delta",
        "p95_delta",
        "bright_low_gradient_delta",
        "low_saturation_bright_delta",
        "sky_bright_proxy_delta",
        "hard_bright_low_gradient_delta",
        "easy_bright_low_gradient_delta",
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
        "scale_multiplier",
        "name",
        "original_psnr",
        "candidate_psnr",
        "delta_psnr",
        "original_ssim",
        "candidate_ssim",
        "delta_ssim",
        "original_bright_low_gradient_psnr",
        "candidate_bright_low_gradient_psnr",
        "delta_bright_low_gradient_psnr",
        "original_sky_bright_proxy_psnr",
        "candidate_sky_bright_proxy_psnr",
        "delta_sky_bright_proxy_psnr",
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
                        "original_bright_low_gradient_psnr": base.get("bright_low_gradient_psnr"),
                        "candidate_bright_low_gradient_psnr": row.get("bright_low_gradient_psnr"),
                        "delta_bright_low_gradient_psnr": (
                            row.get("bright_low_gradient_psnr") - base.get("bright_low_gradient_psnr")
                            if row.get("bright_low_gradient_psnr") is not None
                            and base.get("bright_low_gradient_psnr") is not None
                            else None
                        ),
                        "original_sky_bright_proxy_psnr": base.get("sky_bright_proxy_psnr"),
                        "candidate_sky_bright_proxy_psnr": row.get("sky_bright_proxy_psnr"),
                        "delta_sky_bright_proxy_psnr": (
                            row.get("sky_bright_proxy_psnr") - base.get("sky_bright_proxy_psnr")
                            if row.get("sky_bright_proxy_psnr") is not None
                            and base.get("sky_bright_proxy_psnr") is not None
                            else None
                        ),
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
    parser.add_argument("--dpga_hard_gate_init_bias", type=float, default=-3.0)
    parser.add_argument("--dpga_dark_patch", type=int, default=15)
    parser.add_argument("--dpga_local_patch", type=int, default=31)
    parser.add_argument("--dpga_hard_gate_mode", default="off", choices=["off", "bottleneck"])
    parser.add_argument("--dpga_shallow_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--dpga_bottleneck_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--dpga_skip_scale_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--dpga_module_scale_multiplier",
        type=float,
        default=1.0,
        help="Runtime scale used by module-ablation rows. Keep scale-sweep values independent.",
    )
    parser.add_argument("--run_module_ablation", action="store_true")
    parser.add_argument("--run_scale_sweep", action="store_true")
    args = parser.parse_args()

    if not args.run_module_ablation and not args.run_scale_sweep:
        args.run_module_ablation = True
        args.run_scale_sweep = True

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

    module_specs = [
        ("all_adapters", "all", args.dpga_module_scale_multiplier),
        ("no_shallow", "bottleneck,skip", args.dpga_module_scale_multiplier),
        ("no_bottleneck", "shallow,skip", args.dpga_module_scale_multiplier),
        ("no_skip", "shallow,bottleneck", args.dpga_module_scale_multiplier),
        ("shallow_only", "shallow", args.dpga_module_scale_multiplier),
        ("bottleneck_only", "bottleneck", args.dpga_module_scale_multiplier),
        ("skip_only", "skip", args.dpga_module_scale_multiplier),
    ]
    scale_specs = [(f"scale_{value:g}", "all", value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)]

    module_rows = []
    scale_rows = []
    module_per_image = []
    scale_per_image = []
    summaries = {"original": original_summary, "variants": {}}

    for checkpoint_label, checkpoint_path in checkpoints:
        state = load_model_state(checkpoint_path, device)
        for group, specs, summary_rows, per_image_rows in (
            ("module_ablation", module_specs if args.run_module_ablation else [], module_rows, module_per_image),
            ("scale_sweep", scale_specs if args.run_scale_sweep else [], scale_rows, scale_per_image),
        ):
            for variant, active_adapters, scale_multiplier in specs:
                label = f"{checkpoint_label}_{variant}"
                print(f"eval {group} {label}", flush=True)
                candidate = make_candidate(active_adapters, scale_multiplier)
                load_dpga_state(candidate, state)
                candidate.eval()
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
                    "scale_multiplier": scale_multiplier,
                }
                summary = compare_rows(
                    original_rows,
                    candidate_rows,
                    checkpoint_label,
                    variant,
                    active_adapters,
                    scale_multiplier,
                )
                summary_rows.append(summary)
                per_image_rows.append((meta, candidate_rows))
                summaries["variants"][label] = {
                    "group": group,
                    "checkpoint": checkpoint_path,
                    "active_adapters": active_adapters,
                    "scale_multiplier": scale_multiplier,
                    "eval": candidate_summary,
                    "comparison": summary,
                }
                del candidate
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if module_rows:
        write_summary_csv(output_dir / "dpga_module_ablation_best_final.csv", module_rows)
        write_per_image_csv(output_dir / "dpga_module_ablation_per_image.csv", original_rows, module_per_image)
    if scale_rows:
        write_summary_csv(output_dir / "dpga_scale_sweep_best_final.csv", scale_rows)
        write_per_image_csv(output_dir / "dpga_scale_sweep_per_image.csv", original_rows, scale_per_image)

    (output_dir / "dpga_runtime_variants_summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
