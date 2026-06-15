#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from c13_common import (
    first_dir,
    image_tensor,
    label_path,
    load_a0_model,
    load_c13_model,
    load_split_manifest,
    load_teacher_metric_map,
    metric,
    pad_to,
    write_csv,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--teacher-metrics", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--c12-student-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-mode", default="rgb_wavelet")
    ap.add_argument("--adapter-width", type=int, default=32)
    ap.add_argument("--adapter-depth", type=int, default=3)
    ap.add_argument("--bootstrap-scale", type=float, default=0.01)
    ap.add_argument("--max-train", type=int, default=32)
    ap.add_argument("--max-val", type=int, default=64)
    args = ap.parse_args()

    payload = load_split_manifest(args.split_manifest)
    train_core = list(payload["train_core"])
    val_rows = list(payload["val"])
    train_core = train_core[: args.max_train] if args.max_train else train_core
    val_rows = val_rows[: args.max_val] if args.max_val else val_rows
    teacher_metrics = load_teacher_metric_map(args.teacher_metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = load_a0_model(args.convir_dir, args.a0_checkpoint, device)
    c12_student = load_a0_model(args.convir_dir, args.c12_student_checkpoint, device)
    c13 = load_c13_model(
        args.convir_dir,
        args.a0_checkpoint,
        None,
        device,
        args.feature_mode,
        args.adapter_width,
        args.adapter_depth,
        args.bootstrap_scale,
    )

    train_input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    train_gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))
    rows = []
    with torch.no_grad():
        for name in train_core:
            hazy = image_tensor(train_input_dir / name).to(device)
            label = image_tensor(label_path(train_gt_dir, name)).to(device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = a0(x)[2]
            c12_pred = c12_student(x)[2]
            c13_raw = c13.route_forward(hazy)["outputs"][-1]
            c13_pred = c13(x)[2]
            teacher_row = teacher_metrics.get(name, {})
            a0_psnr, a0_ssim = metric(a0_pred[:, :, :h, :w], label, hp, wp)
            c12_psnr, c12_ssim = metric(c12_pred[:, :, :h, :w], label, hp, wp)
            c13_psnr, c13_ssim = metric(c13_pred[:, :, :h, :w], label, hp, wp)
            rows.append(
                {
                    "name": name,
                    "split": "train_core",
                    "teacher_dPSNR": float(teacher_row.get("WD0375_dPSNR", 0.0)),
                    "teacher_dSSIM": float(teacher_row.get("WD0375_dSSIM", 0.0)),
                    "teacher_positive": float(teacher_row.get("WD0375_dPSNR", 0.0)) > 0,
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "C12_PSNR": c12_psnr,
                    "C12_SSIM": c12_ssim,
                    "C12_dPSNR": c12_psnr - a0_psnr,
                    "C12_dSSIM": c12_ssim - a0_ssim,
                    "C13_PSNR": c13_psnr,
                    "C13_SSIM": c13_ssim,
                    "C13_dPSNR": c13_psnr - a0_psnr,
                    "C13_dSSIM": c13_ssim - a0_ssim,
                    "C13_max_abs_vs_A0": float((c13_raw[:, :, :h, :w] - a0_pred[:, :, :h, :w]).abs().max().item()),
                }
            )

    val_input_dir = first_dir(args.data_dir / "train", ("IN", "haze", "hazy"))
    val_gt_dir = first_dir(args.data_dir / "train", ("GT", "gt"))
    for rec in val_rows:
        name = rec["name"]
        split = rec["split"]
        hazy = image_tensor(val_input_dir / name).to(device)
        label = image_tensor(label_path(val_gt_dir, name)).to(device)
        x, h, w, hp, wp = pad_to(hazy, 32)
        a0_pred = a0(x)[2]
        c12_pred = c12_student(x)[2]
        c13_raw = c13.route_forward(hazy)["outputs"][-1]
        c13_pred = c13(x)[2]
        a0_psnr, a0_ssim = metric(a0_pred[:, :, :h, :w], label, hp, wp)
        c12_psnr, c12_ssim = metric(c12_pred[:, :, :h, :w], label, hp, wp)
        c13_psnr, c13_ssim = metric(c13_pred[:, :, :h, :w], label, hp, wp)
        rows.append(
            {
                "name": name,
                "split": split,
                "teacher_dPSNR": 0.0,
                "teacher_dSSIM": 0.0,
                "teacher_positive": None,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "C12_PSNR": c12_psnr,
                "C12_SSIM": c12_ssim,
                "C12_dPSNR": c12_psnr - a0_psnr,
                "C12_dSSIM": c12_ssim - a0_ssim,
                "C13_PSNR": c13_psnr,
                "C13_SSIM": c13_ssim,
                "C13_dPSNR": c13_psnr - a0_psnr,
                "C13_dSSIM": c13_ssim - a0_ssim,
                "C13_max_abs_vs_A0": float((c13_raw[:, :, :h, :w] - a0_pred[:, :, :h, :w]).abs().max().item()),
            }
        )

    c13_stats = c13.collect_route_stats(hazy)
    train_rows = [row for row in rows if row["split"] == "train_core"]
    val_rows_all = [row for row in rows if row["split"] != "train_core"]
    payload = {
        "route": "Haze4K v2.5 C13 A0-frozen residual distillation",
        "source": "C12 failure audit",
        "locked_test_touched": False,
        "locked_per_image_read": False,
        "feature_mode": args.feature_mode,
        "adapter_width": args.adapter_width,
        "adapter_depth": args.adapter_depth,
        "bootstrap_scale": args.bootstrap_scale,
        "train_core_count": len(train_rows),
        "val_count": len(val_rows_all),
        "train_core_mean_teacher_dPSNR": sum(r["teacher_dPSNR"] for r in train_rows) / max(1, len(train_rows)),
        "train_core_positive_ratio": sum(r["teacher_positive"] for r in train_rows if r["teacher_positive"] is not None) / max(1, len(train_rows)),
        "train_core_mean_C12_dPSNR": sum(r["C12_dPSNR"] for r in train_rows) / max(1, len(train_rows)),
        "train_core_mean_C13_dPSNR": sum(r["C13_dPSNR"] for r in train_rows) / max(1, len(train_rows)),
        "val_mean_C12_dPSNR": sum(r["C12_dPSNR"] for r in val_rows_all) / max(1, len(val_rows_all)),
        "val_mean_C13_dPSNR": sum(r["C13_dPSNR"] for r in val_rows_all) / max(1, len(val_rows_all)),
        "c13_max_abs_vs_A0": max((float(r["C13_max_abs_vs_A0"]) for r in rows), default=0.0),
        "c13_model0_a0_parity_pass": max((float(r["C13_max_abs_vs_A0"]) for r in rows), default=0.0) <= 1e-7,
        "route_stats": c13_stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "v25_c13_0_c12_failure_audit.csv", rows)
    write_json(args.out_dir / "v25_c13_0_c12_failure_audit.json", payload)
    (args.out_dir / "v25_c13_0_c12_failure_audit.md").write_text(
        "\n".join(
            [
                "# Haze4K v2.5 C13 C12 Failure Audit",
                "",
                f"Feature mode: `{args.feature_mode}`",
                f"Adapter width/depth: `{args.adapter_width}` / `{args.adapter_depth}`",
                f"Bootstrap scale: `{args.bootstrap_scale}`",
                f"Train-core count: `{len(train_rows)}`",
                f"Val count: `{len(val_rows_all)}`",
                f"Train-core mean teacher dPSNR: `{payload['train_core_mean_teacher_dPSNR']:.6f}`",
                f"Train-core positive ratio: `{payload['train_core_positive_ratio']:.6f}`",
                f"Train-core mean C12 dPSNR: `{payload['train_core_mean_C12_dPSNR']:.6f}`",
                f"Train-core mean C13 dPSNR: `{payload['train_core_mean_C13_dPSNR']:.6f}`",
                f"Val mean C12 dPSNR: `{payload['val_mean_C12_dPSNR']:.6f}`",
                f"Val mean C13 dPSNR: `{payload['val_mean_C13_dPSNR']:.6f}`",
                f"C13 max abs vs A0: `{payload['c13_max_abs_vs_A0']:.12g}`",
                f"C13 model0 parity pass: `{payload['c13_model0_a0_parity_pass']}`",
                f"Route stats: `{json.dumps(c13_stats, sort_keys=True)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("C13_C12_FAILURE_AUDIT_OK", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
