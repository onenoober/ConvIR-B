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
from PIL import Image
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[3]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for p in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.ConvIR import build_net


def label_path(gt_dir: Path, hazy_name: str) -> Path:
    stem = Path(hazy_name).stem
    ext = Path(hazy_name).suffix
    candidates = [hazy_name]
    if "_" in stem:
        image_id = stem.split("_")[0]
        candidates.extend([f"{image_id}{ext}", f"{image_id}.png"])
    for name in candidates:
        p = gt_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No GT for {hazy_name}; tried {candidates}")


def load_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    return (10 * torch.log10(1 / mse)).item()


def ssim_value(pred: torch.Tensor, target: torch.Tensor, h: int, w: int) -> float:
    down_ratio = max(1, round(min(h, w) / 256))
    return ssim(
        F.adaptive_avg_pool2d(pred, (int(h / down_ratio), int(w / down_ratio))),
        F.adaptive_avg_pool2d(target, (int(h / down_ratio), int(w / down_ratio))),
        data_range=1,
        size_average=False,
    ).mean().item()


def load_state(path: Path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def run_pass(model, names, data_dir: Path, device, pass_name: str):
    haze_dir = data_dir / "train" / "haze"
    gt_dir = data_dir / "train" / "gt"
    factor = 32
    rows = []
    times = []
    with torch.no_grad():
        for idx, name in enumerate(names):
            inp = load_tensor(haze_dir / name).to(device)
            gt = load_tensor(label_path(gt_dir, name)).to(device)
            h, w = inp.shape[2], inp.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            padded = F.pad(inp, (0, padw, 0, padh), "reflect")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = model(padded)[2][:, :, :h, :w]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            pred = torch.clamp(pred, 0, 1)
            rows.append({
                "name": name,
                f"{pass_name}_psnr": psnr(pred, gt),
                f"{pass_name}_ssim": ssim_value(pred, gt, h, w),
                f"{pass_name}_time_sec": elapsed,
            })
            times.append(elapsed)
            if (idx + 1) % 100 == 0:
                mean_psnr = statistics.mean(r[f"{pass_name}_psnr"] for r in rows)
                print(f"{pass_name} {idx + 1}/{len(names)} psnr={mean_psnr:.4f}", flush=True)
    return rows, times


def percentile(values, pct):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--split_name", default="val_inner")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split_json).read_text())
    names = sorted(split["splits"][args.split_name])
    if len(names) != 600:
        raise ValueError(f"Expected 600 val_inner names, got {len(names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_state(Path(args.checkpoint), device))
    model.eval()

    rows1, times1 = run_pass(model, names, Path(args.data_dir), device, "a0")
    rows2, times2 = run_pass(model, names, Path(args.data_dir), device, "a0_repeat")
    rows2_by_name = {r["name"]: r for r in rows2}
    merged = []
    for r in rows1:
        s = rows2_by_name[r["name"]]
        merged.append({
            "name": r["name"],
            "a0_psnr": r["a0_psnr"],
            "a0_ssim": r["a0_ssim"],
            "a0_repeat_psnr": s["a0_repeat_psnr"],
            "a0_repeat_ssim": s["a0_repeat_ssim"],
            "delta_psnr": s["a0_repeat_psnr"] - r["a0_psnr"],
            "delta_ssim": s["a0_repeat_ssim"] - r["a0_ssim"],
            "a0_time_sec": r["a0_time_sec"],
            "a0_repeat_time_sec": s["a0_repeat_time_sec"],
        })

    with (out / "a0_val600_per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "a0_psnr", "a0_ssim", "a0_repeat_psnr", "a0_repeat_ssim", "delta_psnr", "delta_ssim", "a0_time_sec", "a0_repeat_time_sec"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(merged)

    psnrs = [r["a0_psnr"] for r in merged]
    ssims = [r["a0_ssim"] for r in merged]
    dpsnr = [r["delta_psnr"] for r in merged]
    dssim = [r["delta_ssim"] for r in merged]
    global_row = {
        "split": args.split_name,
        "count": len(merged),
        "mean_psnr": statistics.mean(psnrs),
        "median_psnr": statistics.median(psnrs),
        "p5_psnr": percentile(psnrs, 5),
        "mean_ssim": statistics.mean(ssims),
        "repeat_mean_delta_psnr": statistics.mean(dpsnr),
        "repeat_max_abs_delta_psnr": max(abs(v) for v in dpsnr),
        "repeat_mean_delta_ssim": statistics.mean(dssim),
        "repeat_max_abs_delta_ssim": max(abs(v) for v in dssim),
    }
    with (out / "a0_val600_global_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(global_row))
        w.writeheader(); w.writerow(global_row)
    metric_audit = {
        "count": len(merged),
        "repeat_mean_delta_psnr": global_row["repeat_mean_delta_psnr"],
        "repeat_max_abs_delta_psnr": global_row["repeat_max_abs_delta_psnr"],
        "repeat_mean_delta_ssim": global_row["repeat_mean_delta_ssim"],
        "repeat_max_abs_delta_ssim": global_row["repeat_max_abs_delta_ssim"],
        "pass": global_row["repeat_max_abs_delta_psnr"] <= 1e-8 and global_row["repeat_max_abs_delta_ssim"] <= 1e-10,
    }
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    efficiency = {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "avg_time_sec_pass1": statistics.mean(times1),
        "median_time_sec_pass1": statistics.median(times1),
        "avg_time_sec_pass2": statistics.mean(times2),
        "median_time_sec_pass2": statistics.median(times2),
        "peak_cuda_mem_mib": peak,
    }
    (out / "metric_repro_audit.json").write_text(json.dumps(metric_audit, indent=2), encoding="utf-8")
    (out / "a0_efficiency_metrics.json").write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    print(json.dumps({"global": global_row, "metric_repro_audit": metric_audit, "efficiency": efficiency}, indent=2))


if __name__ == "__main__":
    main()
