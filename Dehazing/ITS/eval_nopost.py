#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "experience_docx" / "tools"
for path in (str(Path(__file__).resolve().parent), str(ROOT), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ConvIR import build_net as build_official  # noqa: E402
from models.NoPostFGAConvIR import build_net as build_nopost  # noqa: E402
from nopost_common import (  # noqa: E402
    image_tensor,
    infer_final,
    label_path,
    load_official_checkpoint,
    load_state,
    metric,
    names_for_scope,
    pad_to,
    summarize_delta_rows,
    train_dirs,
    write_csv,
    write_json,
)


N4_GATE = {
    "mean_dPSNR": 0.25,
    "hard_bottom25_dPSNR": 0.35,
    "easy_top25_dPSNR": 0.0,
    "positive_ratio": 0.70,
    "severe_loss_per_600": 72.0,
    "dSSIM": 0.0,
}


N5_GATE = {
    "mean_dPSNR": 0.50,
    "hard_bottom25_dPSNR": 0.70,
    "easy_top25_dPSNR": 0.30,
    "positive_ratio": 0.80,
    "severe_loss_per_600": 72.0,
    "dSSIM": 0.0,
}


def gate_pass(summary, gate):
    return (
        summary["mean_dPSNR"] >= gate["mean_dPSNR"]
        and summary["hard_bottom25_dPSNR"] >= gate["hard_bottom25_dPSNR"]
        and summary["easy_top25_dPSNR"] >= gate["easy_top25_dPSNR"]
        and summary["positive_ratio"] >= gate["positive_ratio"]
        and summary["severe_loss_per_600"] <= gate["severe_loss_per_600"]
        and summary["dSSIM"] >= gate["dSSIM"]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--candidate-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scope", default="fold_val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--use-detail", action="store_true")
    ap.add_argument("--gate-bias", type=float, default=-3.0)
    ap.add_argument("--gate", default="n4", choices=["n4", "n5", "none"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = build_official("base", "Haze4K", "original").to(device)
    candidate = build_nopost(
        "base",
        "Haze4K",
        "original",
        nopost_use_low=True,
        nopost_use_detail=args.use_detail,
        nopost_gate_bias=args.gate_bias,
    ).to(device)
    load_official_checkpoint(a0, args.a0_checkpoint, device)
    candidate.load_state_dict(load_state(args.candidate_checkpoint, device), strict=True)
    a0.eval()
    candidate.eval()

    input_dir, gt_dir = train_dirs(args.data_dir)
    rows = []
    with torch.no_grad():
        for rec in names_for_scope(args.split_manifest, args.scope, fold=args.fold, max_images=args.max_images):
            hazy = image_tensor(input_dir / rec["name"], device)
            label = image_tensor(label_path(gt_dir, rec["name"]), device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = infer_final(a0, x, h, w)
            cand_pred = infer_final(candidate, x, h, w)
            a0_psnr, a0_ssim = metric(a0_pred, label, hp, wp)
            cand_psnr, cand_ssim = metric(cand_pred, label, hp, wp)
            rows.append(
                {
                    "name": rec["name"],
                    "split": rec["split"],
                    "A0_PSNR": a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "candidate_PSNR": cand_psnr,
                    "candidate_SSIM": cand_ssim,
                    "dPSNR": cand_psnr - a0_psnr,
                    "dSSIM": cand_ssim - a0_ssim,
                }
            )
    summary = summarize_delta_rows(rows)
    gate = None
    if args.gate == "n4":
        gate = N4_GATE
    elif args.gate == "n5":
        gate = N5_GATE
    summary.update(
        {
            "tag": args.tag,
            "scope": args.scope,
            "fold": args.fold,
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "gate": args.gate,
            "gate_thresholds": gate,
            "gate_pass": True if gate is None else gate_pass(summary, gate),
            "locked_test_touched": False,
        }
    )
    write_csv(args.out_dir / f"{args.tag}_per_image.csv", rows)
    write_json(args.out_dir / f"{args.tag}_summary.json", summary)
    print("NOPOST_EVAL_PASS" if summary["gate_pass"] else "NOPOST_EVAL_FAIL")


if __name__ == "__main__":
    main()
