#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from c13_common import (
    first_dir,
    image_tensor,
    label_path,
    load_a0_model,
    load_c13_model,
    load_split_manifest,
    metric,
    pad_to,
    summarize_rows,
    write_csv,
    write_json,
)


def eval_rows(
    model,
    a0_model,
    data_dir: Path,
    rows: list[dict[str, Any]],
    device: torch.device,
    split_name: str,
):
    input_dir = first_dir(data_dir / "train", ("IN", "haze", "hazy"))
    gt_dir = first_dir(data_dir / "train", ("GT", "gt"))
    output_rows = []
    with torch.no_grad():
        for idx, rec in enumerate(rows, 1):
            name = rec["name"]
            hazy = image_tensor(input_dir / name).to(device)
            label = image_tensor(label_path(gt_dir, name)).to(device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = a0_model(x)[2]
            pred = model(x)[2]
            a0_psnr, a0_ssim = metric(a0_pred[:, :, :h, :w], label, hp, wp)
            pred_psnr, pred_ssim = metric(pred[:, :, :h, :w], label, hp, wp)
            output_rows.append(
                {
                    "name": name,
                    "split": split_name,
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "student_PSNR": pred_psnr,
                    "student_SSIM": pred_ssim,
                    "dPSNR": pred_psnr - a0_psnr,
                    "dSSIM": pred_ssim - a0_ssim,
                    "max_abs_vs_A0": float((pred[:, :, :h, :w] - a0_pred[:, :, :h, :w]).abs().max().item()),
                }
            )
    return output_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checkpoint", default="Best")
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--student-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-mode", default="rgb_wavelet", choices=["rgb", "rgb_wavelet"])
    ap.add_argument("--adapter-width", type=int, default=32)
    ap.add_argument("--adapter-depth", type=int, default=3)
    ap.add_argument("--bootstrap-scale", type=float, default=0.01)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    args = ap.parse_args()

    payload = load_split_manifest(args.split_manifest)
    train_rows = list(payload["train_core"])
    val_rows = list(payload["val"])
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.max_val:
        val_rows = val_rows[: args.max_val]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = load_a0_model(args.convir_dir, args.a0_checkpoint, device)
    student = load_c13_model(
        args.convir_dir,
        args.a0_checkpoint,
        args.student_checkpoint,
        device,
        args.feature_mode,
        args.adapter_width,
        args.adapter_depth,
        args.bootstrap_scale,
    )
    train_eval_rows = eval_rows(student, a0_model, args.data_dir, [{"name": name} for name in train_rows], device, "train_core")
    val_eval_rows = eval_rows(student, a0_model, args.data_dir, val_rows, device, "val")
    all_rows = train_eval_rows + val_eval_rows
    train_summary = summarize_rows(train_eval_rows)
    val_summary = summarize_rows(val_eval_rows)
    summary = summarize_rows(all_rows)
    summary.update(
        {
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "locked_test_touched": False,
            "locked_per_image_read": False,
            "student_checkpoint": str(args.student_checkpoint),
            "a0_checkpoint": str(args.a0_checkpoint),
            "train_core_count": len(train_eval_rows),
            "val_count": len(val_eval_rows),
            "model0_a0_parity_pass": max((float(row["max_abs_vs_A0"]) for row in all_rows), default=0.0) <= 1e-7,
            "train_core_summary": train_summary,
            "val_summary": val_summary,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v25_c13_eval_{args.variant}_{args.checkpoint}"
    write_csv(args.out_dir / f"{prefix}_per_image.csv", all_rows)
    write_json(args.out_dir / f"{prefix}_summary.json", summary)
    print("C13_EVAL_OK", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
