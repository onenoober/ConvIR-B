#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF


SEVERE = -0.20
GATE = {
    "mean_dPSNR": 1.00,
    "hard_bottom25_dPSNR": 1.00,
    "easy_top25_dPSNR": 0.80,
    "positive_ratio": 0.90,
    "severe_loss_per_600": 36.0,
    "dSSIM": 0.0,
}


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"no GT for {image_name}")


def load_convir(convir_dir: Path):
    sys.path.insert(0, str(convir_dir))
    from models.ConvIR import build_net  # type: ignore

    return build_net


def load_model(convir_dir: Path, checkpoint: Path, device: torch.device):
    build_net = load_convir(convir_dir)
    model = build_net("base", "Haze4K", "original").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), "reflect"), h, w, h + ph, w + pw


def infer(model, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    out = model(x)
    pred = out[2] if isinstance(out, (list, tuple)) else out
    return torch.clamp(pred[:, :, :h, :w], 0, 1)


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    psnr = float((10 * torch.log10(1 / mse)).item())
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ds = [float(r["dPSNR"]) for r in rows]
    ss = [float(r["dSSIM"]) for r in rows]
    ordered = sorted(rows, key=lambda r: float(r["A0_PSNR"]))
    k = max(1, n // 4)
    hard = [float(r["dPSNR"]) for r in ordered[:k]]
    easy = [float(r["dPSNR"]) for r in ordered[-k:]]
    severe = sum(d <= SEVERE for d in ds)
    return {
        "count": n,
        "mean_dPSNR": statistics.mean(ds),
        "median_dPSNR": statistics.median(ds),
        "hard_bottom25_dPSNR": statistics.mean(hard),
        "easy_top25_dPSNR": statistics.mean(easy),
        "dSSIM": statistics.mean(ss),
        "positive_ratio": sum(d > 0 for d in ds) / n,
        "nonnegative_ratio": sum(d >= 0 for d in ds) / n,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / n * 600.0,
        "p05_dPSNR": sorted(ds)[max(0, math.floor(0.05 * n) - 1)],
        "p95_dPSNR": sorted(ds)[min(n - 1, math.ceil(0.95 * n) - 1)],
    }


def gate_pass(summary: dict[str, Any]) -> bool:
    return (
        summary["mean_dPSNR"] >= GATE["mean_dPSNR"]
        and summary["hard_bottom25_dPSNR"] >= GATE["hard_bottom25_dPSNR"]
        and summary["easy_top25_dPSNR"] >= GATE["easy_top25_dPSNR"]
        and summary["positive_ratio"] >= GATE["positive_ratio"]
        and summary["severe_loss_per_600"] <= GATE["severe_loss_per_600"]
        and summary["dSSIM"] >= GATE["dSSIM"]
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checkpoint-name", default="Best")
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--student-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--print-freq", type=int, default=50)
    args = ap.parse_args()

    payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    val = list(payload["val"])
    if args.max_images:
        val = val[: args.max_images]
    input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = load_model(args.convir_dir, args.a0_checkpoint, device)
    student = load_model(args.convir_dir, args.student_checkpoint, device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, rec in enumerate(val, 1):
            name = rec["name"]
            split = rec["split"]
            hazy = TVF.to_tensor(Image.open(input_dir / name).convert("RGB")).unsqueeze(0).to(device)
            label = TVF.to_tensor(Image.open(label_path(gt_dir, name)).convert("RGB")).unsqueeze(0).to(device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = infer(a0, x, h, w)
            st_pred = infer(student, x, h, w)
            a0_psnr, a0_ssim = metric(a0_pred, label, hp, wp)
            st_psnr, st_ssim = metric(st_pred, label, hp, wp)
            rows.append({
                "name": name,
                "split": split,
                "variant": args.variant,
                "checkpoint": args.checkpoint_name,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "student_PSNR": st_psnr,
                "student_SSIM": st_ssim,
                "dPSNR": st_psnr - a0_psnr,
                "dSSIM": st_ssim - a0_ssim,
            })
            if idx % args.print_freq == 0:
                print(f"C12_EVAL {args.variant} {idx}/{len(val)}")
    summary = summarize(rows)
    summary.update({
        "variant": args.variant,
        "checkpoint": args.checkpoint_name,
        "gate_pass": gate_pass(summary),
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "student_checkpoint": str(args.student_checkpoint),
    })
    prefix = f"v24_c12_eval_{args.variant}_{args.checkpoint_name}"
    write_csv(args.out_dir / f"{prefix}_per_image.csv", rows)
    write_json(args.out_dir / f"{prefix}_summary.json", summary)
    print("C12_EVAL_OK", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
