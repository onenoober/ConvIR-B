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

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader
from models.ConvIR import build_net
from models.ConvIRWD import build_convir_wd_net


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


def cvar_low(values, pct):
    if not values:
        return None
    count = max(1, math.ceil(len(values) * pct / 100.0))
    return statistics.mean(sorted(values)[:count])


def build_model(arch):
    if arch in ("official_convir", "convir"):
        return build_net("base", "Haze4K", "original")
    if arch == "convir_wd_lite":
        return build_convir_wd_net("base", "Haze4K")
    raise ValueError(f"Unsupported arch: {arch}")


def load_checkpoint_model(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def eval_one(label, arch, checkpoint, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(arch).to(device)
    model.load_state_dict(load_checkpoint_model(checkpoint, device), strict=True)
    model.eval()

    dataloader = test_dataloader(data_dir, "Haze4K", batch_size=1, num_workers=0)
    rows = []
    times = []
    factor = 32
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            h_pad = ((h + factor) // factor) * factor
            w_pad = ((w + factor) // factor) * factor
            pad_h = h_pad - h if h % factor != 0 else 0
            pad_w = w_pad - w if w % factor != 0 else 0
            padded = F.pad(input_img, (0, pad_w, 0, pad_h), "reflect")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = model(padded)[2][:, :, :h, :w]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            pred = torch.clamp(pred, 0, 1)
            mse = F.mse_loss(pred, label_img)
            psnr_val = (10 * torch.log10(1 / mse)).item()
            down_ratio = max(1, round(min(h_pad, w_pad) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(pred, (int(h_pad / down_ratio), int(w_pad / down_ratio))),
                F.adaptive_avg_pool2d(label_img, (int(h_pad / down_ratio), int(w_pad / down_ratio))),
                data_range=1,
                size_average=False,
            ).mean().item()

            times.append(elapsed)
            rows.append({
                "name": name[0],
                "psnr": psnr_val,
                "ssim": ssim_val,
                "time_sec": elapsed,
            })
            if (idx + 1) % 40 == 0:
                print(f"{label} {idx + 1}/{len(dataloader)} mean_psnr={statistics.mean(r['psnr'] for r in rows):.4f}", flush=True)

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    summary = {
        "label": label,
        "arch": arch,
        "checkpoint": checkpoint,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": peak_mem,
    }
    return rows, summary


def summarize_comparison(original_rows, candidate_rows, original_name, candidate_name):
    original = {row["name"]: row for row in original_rows}
    candidate = {row["name"]: row for row in candidate_rows}
    common = [name for name in original if name in candidate]
    if not common:
        raise ValueError("No common images between original and candidate eval rows")
    deltas = [candidate[name]["psnr"] - original[name]["psnr"] for name in common]
    ssim_deltas = [candidate[name]["ssim"] - original[name]["ssim"] for name in common]
    sorted_by_original = sorted(common, key=lambda name: original[name]["psnr"])
    bucket_count = max(1, len(common) // 4)
    hard = sorted_by_original[:bucket_count]
    easy = sorted_by_original[-bucket_count:]
    catastrophic = [
        name for name in common
        if candidate[name]["psnr"] - original[name]["psnr"] <= -2.0
        or candidate[name]["ssim"] - original[name]["ssim"] <= -0.02
    ]
    strong_cut = percentile([original[name]["psnr"] for name in common], 75)
    strong = [name for name in common if original[name]["psnr"] >= strong_cut]
    strong_regressions = [
        name for name in strong if candidate[name]["psnr"] - original[name]["psnr"] <= -0.05
    ]
    return {
        "common_count": len(common),
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p05_psnr_delta": percentile(deltas, 5),
        "cvar5_psnr_delta": cvar_low(deltas, 5),
        "hard_bottom25_psnr_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in hard),
        "easy_top25_psnr_delta": statistics.mean(candidate[name]["psnr"] - original[name]["psnr"] for name in easy),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "median_ssim_delta": statistics.median(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "catastrophic_proxy_count": len(catastrophic),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_count": len(strong),
        "strong_regression_count_delta_le_neg_0p05": len(strong_regressions),
        "original_name": original_name,
        "candidate_name": candidate_name,
    }


def load_v31_rows(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["image_name"]: row for row in csv.DictReader(handle)}


def summarize_v31(v31_rows, names, original_rows):
    if not v31_rows:
        return {"available": False}
    original = {row["name"]: row for row in original_rows}
    common = [name for name in names if name in v31_rows and name in original]
    sorted_by_original = sorted(common, key=lambda name: original[name]["psnr"])
    bucket_count = max(1, len(common) // 4)
    hard = set(sorted_by_original[:bucket_count])
    easy = set(sorted_by_original[-bucket_count:])
    candidates = {
        "WDMamba_standalone_fullimage": "WDMamba_standalone_fullimage_delta_vs_A0",
        "ConvIR-L_standalone_fullimage": "ConvIR-L_standalone_fullimage_delta_vs_A0",
        "FullUDP_standalone_fullimage": "FullUDP_standalone_fullimage_delta_vs_A0",
    }
    out = {"available": True, "common_count": len(common), "candidates": {}}
    for label, delta_col in candidates.items():
        deltas = [float(v31_rows[name][delta_col]) for name in common]
        out["candidates"][label] = {
            "mean_psnr_delta": statistics.mean(deltas),
            "median_psnr_delta": statistics.median(deltas),
            "p05_psnr_delta": percentile(deltas, 5),
            "cvar5_psnr_delta": cvar_low(deltas, 5),
            "hard_bottom25_psnr_delta": statistics.mean(float(v31_rows[name][delta_col]) for name in common if name in hard),
            "easy_top25_psnr_delta": statistics.mean(float(v31_rows[name][delta_col]) for name in common if name in easy),
        }
    return out


def pareto_competitive(comparison, v31_summary):
    if not v31_summary.get("available"):
        return {
            "available": False,
            "competitive": False,
            "reason": "v3.1 per-image table unavailable",
        }
    keys = [
        "mean_psnr_delta",
        "hard_bottom25_psnr_delta",
        "easy_top25_psnr_delta",
        "p05_psnr_delta",
        "cvar5_psnr_delta",
    ]
    dominated_by = []
    eps = 0.02
    for label, metrics in v31_summary["candidates"].items():
        if label == "FullUDP_standalone_fullimage":
            continue
        no_worse = all(metrics[key] >= comparison[key] - eps for key in keys)
        strictly_better = any(metrics[key] > comparison[key] + eps for key in keys)
        if no_worse and strictly_better:
            dominated_by.append(label)
    return {
        "available": True,
        "competitive": not dominated_by,
        "dominated_by": dominated_by,
        "metrics_checked": keys,
        "epsilon_db": eps,
    }


def gate_result(comparison, v31_summary):
    checks = {
        "mean_psnr_delta_ge_0p30": comparison["mean_psnr_delta"] >= 0.30,
        "hard_bottom25_psnr_delta_ge_0p50": comparison["hard_bottom25_psnr_delta"] >= 0.50,
        "easy_top25_psnr_delta_ge_neg0p05": comparison["easy_top25_psnr_delta"] >= -0.05,
        "p05_psnr_delta_ge_neg0p30": comparison["p05_psnr_delta"] >= -0.30,
        "cvar5_psnr_delta_ge_neg0p50": comparison["cvar5_psnr_delta"] >= -0.50,
        "mean_ssim_delta_ge_neg0p001": comparison["mean_ssim_delta"] >= -0.001,
        "catastrophic_proxy_count_eq_0": comparison["catastrophic_proxy_count"] == 0,
    }
    pareto = pareto_competitive(comparison, v31_summary)
    quality_pass = all(checks.values())
    continue_allowed = quality_pass and pareto.get("competitive", False)
    return {
        "checks": checks,
        "p2_quality_pass": quality_pass,
        "pareto_competitive_vs_v31": pareto,
        "continue_allowed_to_p3_design": continue_allowed,
        "locked_test_allowed": False,
        "decision": (
            "V32_P2_TRAIN_DERIVED_VALIDATION_PASS_P3_DESIGN_ALLOWED_LOCKED_TEST_BLOCKED"
            if continue_allowed
            else "V32_P2_TRAIN_DERIVED_VALIDATION_FAIL_OR_NOT_COMPETITIVE_LOCKED_TEST_BLOCKED"
        ),
    }


def write_per_image(path, original_rows, candidate_rows, v31_rows):
    original = {row["name"]: row for row in original_rows}
    candidate = {row["name"]: row for row in candidate_rows}
    common = [name for name in original if name in candidate]
    fieldnames = [
        "name",
        "a0_psnr",
        "candidate_psnr",
        "delta_psnr",
        "a0_ssim",
        "candidate_ssim",
        "delta_ssim",
        "wdmamba_delta_vs_a0",
        "convirl_delta_vs_a0",
        "fulludp_delta_vs_a0",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in common:
            v31 = v31_rows.get(name, {})
            writer.writerow({
                "name": name,
                "a0_psnr": original[name]["psnr"],
                "candidate_psnr": candidate[name]["psnr"],
                "delta_psnr": candidate[name]["psnr"] - original[name]["psnr"],
                "a0_ssim": original[name]["ssim"],
                "candidate_ssim": candidate[name]["ssim"],
                "delta_ssim": candidate[name]["ssim"] - original[name]["ssim"],
                "wdmamba_delta_vs_a0": v31.get("WDMamba_standalone_fullimage_delta_vs_A0", ""),
                "convirl_delta_vs_a0": v31.get("ConvIR-L_standalone_fullimage_delta_vs_A0", ""),
                "fulludp_delta_vs_a0": v31.get("FullUDP_standalone_fullimage_delta_vs_A0", ""),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--candidate_arch", default="convir_wd_lite", choices=["convir_wd_lite", "official_convir", "convir"])
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_per_image", required=True)
    parser.add_argument("--v31_per_image_csv", default="")
    args = parser.parse_args()

    original_rows, original_summary = eval_one("official_A0", "official_convir", args.original_checkpoint, args.data_dir)
    candidate_rows, candidate_summary = eval_one(args.candidate_name, args.candidate_arch, args.candidate_checkpoint, args.data_dir)
    comparison = summarize_comparison(original_rows, candidate_rows, "official_A0", args.candidate_name)
    names = [row["name"] for row in original_rows]
    v31_rows = load_v31_rows(args.v31_per_image_csv)
    v31_summary = summarize_v31(v31_rows, names, original_rows)
    gate = gate_result(comparison, v31_summary)

    write_per_image(args.output_per_image, original_rows, candidate_rows, v31_rows)
    summary = {
        "route_id": "haze4k_v3_2_convir_wd_full_model_line_20260707",
        "phase": "P2 train-derived validation eval",
        "data_dir": args.data_dir,
        "original": original_summary,
        "candidate": candidate_summary,
        "comparison": comparison,
        "v31_same_names": v31_summary,
        "gate": gate,
        "per_image_csv_cloud_only": args.output_per_image,
        "locked_test_touched": False,
        "quality_claim": "P2 train-derived validation only; no locked-test or deployment claim.",
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
