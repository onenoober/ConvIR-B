import argparse
import csv
import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim


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


def purge_repo_modules():
    prefixes = ("data", "models")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(prefixes[0] + ".") or name.startswith(prefixes[1] + "."):
            del sys.modules[name]


@contextmanager
def repo_import_context(its_root):
    its_root = str(Path(its_root).resolve())
    previous_cwd = os.getcwd()
    previous_path = list(sys.path)
    purge_repo_modules()
    os.chdir(its_root)
    sys.path.insert(0, its_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        purge_repo_modules()


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_state(model, checkpoint, device, strict=True):
    state = load_model_state(checkpoint, device)
    result = model.load_state_dict(state, strict=strict)
    return {
        "missing": list(getattr(result, "missing_keys", [])),
        "unexpected": list(getattr(result, "unexpected_keys", [])),
    }


def build_apdr_model(args, device):
    with repo_import_context(args.apdr_its_root):
        module = importlib.import_module("models.APDRConvIR")
        model = module.build_apdr_net(
            "base",
            "Haze4K",
            apdr_prior_mode=args.apdr_prior_mode,
            apdr_residual_max=args.residual_max,
            apdr_gate_max=args.apdr_gate_max,
            apdr_gate_init=args.apdr_gate_init,
            apdr_force_zero_gate=False,
            apdr_active_scales=args.apdr_active_scales,
            apdr_selector_mode=args.apdr_selector_mode,
            apdr_residual_capacity=args.apdr_residual_capacity,
        ).to(device)
        load_info = load_state(model, args.apdr_selector_checkpoint, device, strict=True)
        model.eval()
        return model, load_info


def build_expert_model(args, device):
    with repo_import_context(args.expert_its_root):
        if args.expert_arch == "convir":
            module = importlib.import_module("models.ConvIR")
            build_net = module.build_net
            signature = inspect.signature(build_net)
            kwargs = {}
            if "fam_mode" in signature.parameters:
                kwargs["fam_mode"] = args.expert_mode
            if "scm_mode" in signature.parameters:
                kwargs["scm_mode"] = args.expert_scm_mode
            if kwargs:
                model = build_net("base", "Haze4K", **kwargs).to(device)
            else:
                model = build_net("base", "Haze4K", args.expert_mode).to(device)
        elif args.expert_arch == "pfd":
            module = importlib.import_module("models.PFDConvIR")
            model = module.build_pfd_net(
                "base",
                "Haze4K",
                pfd_rhfd=bool(args.expert_pfd_rhfd),
                pfd_hscm=bool(args.expert_pfd_hscm),
                pfd_pffb=bool(args.expert_pfd_pffb),
                pfd_pffb_high=bool(args.expert_pfd_pffb_high),
                pfd_teacher=bool(args.expert_pfd_teacher),
            ).to(device)
        else:
            raise ValueError(f"Unsupported expert_arch: {args.expert_arch}")
        load_info = load_state(model, args.expert_checkpoint, device, strict=True)
        model.eval()
        return model, load_info


def import_test_dataloader(apdr_its_root):
    with repo_import_context(apdr_its_root):
        module = importlib.import_module("data")
        return module.test_dataloader


def summarize_deltas(rows, key):
    deltas = [row[key] for row in rows]
    ssim_deltas = [row[key.replace("psnr", "ssim")] for row in rows]
    sorted_deltas = sorted(deltas)
    tail_count = max(1, len(deltas) // 10)
    return {
        "common_count": len(rows),
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p5_psnr_delta": percentile(deltas, 5),
        "p95_psnr_delta": percentile(deltas, 95),
        "worst10pct_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
        "best10pct_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
        "worst10img_mean_psnr_delta": statistics.mean(sorted_deltas[:10]),
        "best10img_mean_psnr_delta": statistics.mean(sorted_deltas[-10:]),
        "worst10_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
        "best10_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "strong_regression_count_delta_le_-0.05": None,
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
    }


def attach_strong_counts(summary, rows, key, strong_cut):
    strong = [row for row in rows if row["original_psnr"] >= strong_cut]
    summary["strong_reference_cut_psnr"] = strong_cut
    summary["strong_reference_count"] = len(strong)
    summary["strong_regression_count_delta_le_-0.05"] = sum(row[key] <= -0.05 for row in strong)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--apdr_its_root", required=True)
    parser.add_argument("--apdr_selector_checkpoint", required=True)
    parser.add_argument("--apdr_prior_mode", default="rgb_haze")
    parser.add_argument("--apdr_selector_mode", default="v0_2r")
    parser.add_argument("--apdr_active_scales", default="full")
    parser.add_argument("--apdr_residual_capacity", default="linear")
    parser.add_argument("--apdr_gate_max", type=float, default=1.0)
    parser.add_argument("--apdr_gate_init", type=float, default=0.01)
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--expert_name", required=True)
    parser.add_argument("--expert_its_root", required=True)
    parser.add_argument("--expert_checkpoint", required=True)
    parser.add_argument("--expert_arch", choices=["convir", "pfd"], default="convir")
    parser.add_argument("--expert_mode", default="original")
    parser.add_argument("--expert_scm_mode", default="original")
    parser.add_argument("--expert_pfd_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--expert_pfd_hscm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--expert_pfd_pffb", type=int, default=0, choices=[0, 1])
    parser.add_argument("--expert_pfd_pffb_high", type=int, default=0, choices=[0, 1])
    parser.add_argument("--expert_pfd_teacher", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = f"safe_expert_{args.expert_name}"

    test_dataloader = import_test_dataloader(args.apdr_its_root)
    apdr_model, apdr_load = build_apdr_model(args, device)
    expert_model, expert_load = build_expert_model(args, device)
    dataloader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=args.num_workers)

    rows = []
    factor = 32
    apdr_times = []
    expert_times = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images and idx >= args.max_images:
                break
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            padded_h = ((h + factor) // factor) * factor
            padded_w = ((w + factor) // factor) * factor
            padh = padded_h - h if h % factor != 0 else 0
            padw = padded_w - w if w % factor != 0 else 0
            padded = f.pad(input_img, (0, padw, 0, padh), "reflect")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            apdr_model(padded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            apdr_elapsed = time.time() - start
            apdr_times.append(apdr_elapsed)

            tensors = apdr_model._last_apdr_tensors or []
            active_full = [item for item in tensors if item.get("scale") == "full"]
            if not active_full:
                raise RuntimeError("APDR model did not expose a full-scale tensor record.")
            full = active_full[0]
            anchor = full["anchor"][:, :, :h, :w].clamp(0, 1)
            m_safe = full["gate"][:, :, :h, :w].clamp(0, 1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            expert_pred = expert_model(padded)[2][:, :, :h, :w].clamp(0, 1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            expert_elapsed = time.time() - start
            expert_times.append(expert_elapsed)

            delta_expert = (expert_pred - anchor).clamp(-args.residual_max, args.residual_max)
            safe_pred = (anchor + m_safe * delta_expert).clamp(0, 1)

            original_psnr = psnr(anchor, label_img)
            raw_expert_psnr = psnr(expert_pred, label_img)
            safe_psnr = psnr(safe_pred, label_img)
            original_ssim = ssim_value(anchor, label_img)
            raw_expert_ssim = ssim_value(expert_pred, label_img)
            safe_ssim = ssim_value(safe_pred, label_img)
            rows.append(
                {
                    "name": name[0],
                    "original_psnr": original_psnr,
                    "raw_expert_psnr": raw_expert_psnr,
                    "candidate_psnr": safe_psnr,
                    "delta_psnr": safe_psnr - original_psnr,
                    "raw_expert_delta_psnr": raw_expert_psnr - original_psnr,
                    "original_ssim": original_ssim,
                    "raw_expert_ssim": raw_expert_ssim,
                    "candidate_ssim": safe_ssim,
                    "delta_ssim": safe_ssim - original_ssim,
                    "raw_expert_delta_ssim": raw_expert_ssim - original_ssim,
                    "original_time_sec": apdr_elapsed,
                    "candidate_time_sec": apdr_elapsed + expert_elapsed,
                    "expert_time_sec": expert_elapsed,
                    "m_safe_mean": m_safe.mean().item(),
                    "m_safe_p95": torch.quantile(m_safe.flatten(), 0.95).item(),
                    "m_safe_active_gt_0_01": (m_safe > 0.01).float().mean().item(),
                    "delta_expert_abs_mean": delta_expert.abs().mean().item(),
                    "delta_expert_abs_max": delta_expert.abs().max().item(),
                }
            )
            if args.progress_freq and (idx + 1) % args.progress_freq == 0:
                mean_delta = statistics.mean(row["delta_psnr"] for row in rows)
                print(f"{args.expert_name} {idx + 1}/{len(dataloader)} safe_delta={mean_delta:.4f}", flush=True)

    if not rows:
        raise RuntimeError("No rows were evaluated.")

    strong_cut = percentile([row["original_psnr"] for row in rows], 75)
    safe_summary = summarize_deltas(rows, "delta_psnr")
    raw_summary = summarize_deltas(rows, "raw_expert_delta_psnr")
    attach_strong_counts(safe_summary, rows, "delta_psnr", strong_cut)
    attach_strong_counts(raw_summary, rows, "raw_expert_delta_psnr", strong_cut)

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    summary = {
        "runs": {
            "a0_anchor_from_apdr": {
                "label": "a0_anchor_from_apdr",
                "arch": "apdr",
                "mode": "anchor",
                "checkpoint": args.apdr_selector_checkpoint,
                "count": len(rows),
                "mean_psnr": statistics.mean(row["original_psnr"] for row in rows),
                "mean_ssim": statistics.mean(row["original_ssim"] for row in rows),
                "avg_time_sec_sync": statistics.mean(apdr_times),
                "median_time_sec_sync": statistics.median(apdr_times),
                "peak_cuda_mem_mib": peak_mem,
            },
            candidate_name: {
                "label": candidate_name,
                "arch": "safe_expert_delta",
                "expert_arch": args.expert_arch,
                "expert_mode": args.expert_mode,
                "expert_scm_mode": args.expert_scm_mode,
                "checkpoint": args.expert_checkpoint,
                "count": len(rows),
                "mean_psnr": statistics.mean(row["candidate_psnr"] for row in rows),
                "mean_ssim": statistics.mean(row["candidate_ssim"] for row in rows),
                "avg_time_sec_sync": statistics.mean(row["candidate_time_sec"] for row in rows),
                "median_time_sec_sync": statistics.median(row["candidate_time_sec"] for row in rows),
                "peak_cuda_mem_mib": peak_mem,
            },
        },
        "comparison": safe_summary,
        "raw_expert_comparison": raw_summary,
        "mask_stats": {
            "m_safe_mean": statistics.mean(row["m_safe_mean"] for row in rows),
            "m_safe_p95_mean": statistics.mean(row["m_safe_p95"] for row in rows),
            "m_safe_active_gt_0_01_mean": statistics.mean(row["m_safe_active_gt_0_01"] for row in rows),
            "delta_expert_abs_mean": statistics.mean(row["delta_expert_abs_mean"] for row in rows),
            "delta_expert_abs_max": max(row["delta_expert_abs_max"] for row in rows),
        },
        "load_info": {
            "apdr": apdr_load,
            "expert": expert_load,
        },
        "args": vars(args),
    }

    compare_path = output_dir / f"scout_eval_compare_{args.tag}.json"
    csv_path = output_dir / f"scout_eval_per_image_{args.tag}.csv"
    with compare_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
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
                "raw_expert_psnr",
                "raw_expert_delta_psnr",
                "raw_expert_ssim",
                "raw_expert_delta_ssim",
                "expert_time_sec",
                "m_safe_mean",
                "m_safe_p95",
                "m_safe_active_gt_0_01",
                "delta_expert_abs_mean",
                "delta_expert_abs_max",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["original_psnr"],
                    row["candidate_psnr"],
                    row["delta_psnr"],
                    row["original_ssim"],
                    row["candidate_ssim"],
                    row["delta_ssim"],
                    row["original_time_sec"],
                    row["candidate_time_sec"],
                    row["raw_expert_psnr"],
                    row["raw_expert_delta_psnr"],
                    row["raw_expert_ssim"],
                    row["raw_expert_delta_ssim"],
                    row["expert_time_sec"],
                    row["m_safe_mean"],
                    row["m_safe_p95"],
                    row["m_safe_active_gt_0_01"],
                    row["delta_expert_abs_mean"],
                    row["delta_expert_abs_max"],
                ]
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {compare_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
