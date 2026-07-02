#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "experience_docx" / "tools"
for path in (str(TOOLS), str(REPO_ROOT / "Dehazing" / "ITS"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nopost_common import (  # noqa: E402
    build_nopost,
    build_official,
    image_tensor,
    infer_final,
    label_path,
    load_official_checkpoint,
    metric,
    names_for_scope,
    pad_to,
    partial_load_nopost,
    sha256,
    summarize_delta_rows,
    train_dirs,
    write_csv,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scope", default="fold_val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--gate-bias", type=float, default=-3.0)
    ap.add_argument("--use-detail", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official = build_official(device)
    nopost = build_nopost(device, use_detail=args.use_detail, gate_bias=args.gate_bias)
    load_official_checkpoint(official, args.checkpoint, device)
    load_report = partial_load_nopost(nopost, args.checkpoint, device)
    official.eval()
    nopost.eval()

    input_dir, gt_dir = train_dirs(args.data_dir)
    rows = []
    max_abs = 0.0
    with torch.no_grad():
        for rec in names_for_scope(args.split_manifest, args.scope, fold=args.fold, max_images=args.max_images):
            hazy = image_tensor(input_dir / rec["name"], device)
            label = image_tensor(label_path(gt_dir, rec["name"]), device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0 = infer_final(official, x, h, w)
            candidate = infer_final(nopost, x, h, w)
            diff = float((a0 - candidate).abs().max().detach().cpu())
            max_abs = max(max_abs, diff)
            a0_psnr, a0_ssim = metric(a0, label, hp, wp)
            cand_psnr, cand_ssim = metric(candidate, label, hp, wp)
            rows.append(
                {
                    "name": rec["name"],
                    "split": rec["split"],
                    "max_abs_vs_A0": diff,
                    "A0_PSNR": a0_psnr,
                    "NoPost_PSNR": cand_psnr,
                    "dPSNR": cand_psnr - a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "NoPost_SSIM": cand_ssim,
                    "dSSIM": cand_ssim - a0_ssim,
                }
            )

    trainable = []
    frozen = []
    for name, param in nopost.named_parameters():
        rec = {"name": name, "numel": param.numel(), "requires_grad": name.startswith("nopost_adapter.")}
        if rec["requires_grad"]:
            trainable.append(rec)
        else:
            frozen.append(rec)
    for name, param in nopost.named_parameters():
        param.requires_grad = name.startswith("nopost_adapter.")

    param_groups = {
        "trainable_prefixes": ["nopost_adapter."],
        "trainable_param_count": sum(r["numel"] for r in trainable),
        "frozen_param_count": sum(r["numel"] for r in frozen),
        "trainable": trainable,
    }
    delta_summary = summarize_delta_rows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "scope": args.scope,
        "fold": args.fold,
        "count": len(rows),
        "max_abs_vs_A0": max_abs,
        "delta_summary": delta_summary,
        "partial_load": load_report,
        "param_groups": {k: v for k, v in param_groups.items() if k != "trainable"},
        "locked_test_touched": False,
        "pass": max_abs <= 1e-7,
    }
    write_csv(args.out_dir / "v213_n2_identity_per_image.csv", rows)
    write_json(args.out_dir / "v213_n2_identity_summary.json", summary)
    write_json(args.out_dir / "v213_n2_param_groups.json", param_groups)
    (args.out_dir / "v213_n2_state_dict_load_report.txt").write_text(
        "NOPOST_PARTIAL_LOAD\n"
        f"checkpoint={args.checkpoint}\n"
        f"loaded_count={load_report['loaded_count']}\n"
        f"missing_new_count={len(load_report['missing_new_modules'])}\n"
        f"unexpected={load_report['unexpected']}\n"
        f"shape_mismatch={load_report['shape_mismatch']}\n"
        f"max_abs_vs_A0={max_abs:.12g}\n",
        encoding="utf-8",
    )
    print("N2_NOPOST_IDENTITY_PASS" if summary["pass"] else "N2_NOPOST_IDENTITY_FAIL")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
