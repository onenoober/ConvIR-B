#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from nopost_common import (
    build_nopost,
    build_official,
    first_dir,
    image_tensor,
    infer_final,
    label_path,
    load_official_checkpoint,
    metric,
    pad_to,
    partial_load_nopost,
    sha256,
    train_dirs,
    write_json,
)


FORBIDDEN_PATTERNS = [
    "anchor_side",
    "teacher - A0",
    "expert - A0",
    "anchor - hazy",
    "anchor_side - hazy",
    "A0 + rgb_delta",
    "A0_output",
    "teacher_image",
    "expert_image",
]


SCAN_FILES = [
    "Dehazing/ITS/models/NoPostFGAConvIR.py",
    "Dehazing/ITS/models/nopost_fga.py",
    "Dehazing/ITS/main.py",
    "Dehazing/ITS/train.py",
]


def git_value(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def source_scan(repo_root: Path) -> dict[str, Any]:
    hits = []
    output_lines = []
    for rel in SCAN_FILES:
        path = repo_root / rel
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in line:
                    hits.append({"file": rel, "line": lineno, "pattern": pattern, "text": line.strip()})
                    output_lines.append(f"{rel}:{lineno}:{pattern}:{line.strip()}")
    return {"files": SCAN_FILES, "forbidden_patterns": FORBIDDEN_PATTERNS, "hits": hits, "text": "\n".join(output_lines)}


def final_plus_x_count(repo_root: Path) -> int:
    text = (repo_root / "Dehazing/ITS/models/NoPostFGAConvIR.py").read_text(encoding="utf-8")
    return text.count("outputs.append(rgb_residual + x)")


def compare_models(args, device: torch.device) -> dict[str, Any]:
    official = build_official(device)
    nopost = build_nopost(device, use_detail=args.use_detail, gate_bias=args.gate_bias)
    load_official_checkpoint(official, args.checkpoint, device)
    partial_report = partial_load_nopost(nopost, args.checkpoint, device)
    official.eval()
    nopost.eval()

    synthetic = torch.rand(1, 3, args.synthetic_size, args.synthetic_size, device=device)
    with torch.no_grad():
        official_syn = infer_final(official, synthetic, args.synthetic_size, args.synthetic_size)
        nopost_syn = infer_final(nopost, synthetic, args.synthetic_size, args.synthetic_size)
    synthetic_max_abs = float((official_syn - nopost_syn).abs().max().detach().cpu())

    per_image = []
    real_max_abs = 0.0
    if args.split_manifest and args.max_images > 0:
        from nopost_common import names_for_scope

        input_dir, gt_dir = train_dirs(args.data_dir)
        rows = names_for_scope(args.split_manifest, args.scope, fold=args.fold, max_images=args.max_images)
        with torch.no_grad():
            for rec in rows:
                hazy = image_tensor(input_dir / rec["name"], device)
                label = image_tensor(label_path(gt_dir, rec["name"]), device)
                x, h, w, hp, wp = pad_to(hazy, 32)
                official_pred = infer_final(official, x, h, w)
                nopost_pred = infer_final(nopost, x, h, w)
                diff = float((official_pred - nopost_pred).abs().max().detach().cpu())
                real_max_abs = max(real_max_abs, diff)
                official_psnr, official_ssim = metric(official_pred, label, hp, wp)
                nopost_psnr, nopost_ssim = metric(nopost_pred, label, hp, wp)
                per_image.append(
                    {
                        "name": rec["name"],
                        "split": rec["split"],
                        "max_abs_vs_A0": diff,
                        "A0_PSNR": official_psnr,
                        "NoPost_PSNR": nopost_psnr,
                        "dPSNR": nopost_psnr - official_psnr,
                        "A0_SSIM": official_ssim,
                        "NoPost_SSIM": nopost_ssim,
                        "dSSIM": nopost_ssim - official_ssim,
                    }
                )
    return {
        "partial_load": partial_report,
        "synthetic_max_abs_vs_A0": synthetic_max_abs,
        "real_max_abs_vs_A0": real_max_abs,
        "per_image": per_image,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument("--scope", default="fold_val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--synthetic-size", type=int, default=64)
    ap.add_argument("--gate-bias", type=float, default=-3.0)
    ap.add_argument("--use-detail", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scan = source_scan(repo_root)
    compare = compare_models(args, device)

    from models.NoPostFGAConvIR import NoPostFGAConvIR
    from models.nopost_fga import NoPostPBCFGA

    signature = {
        "NoPostFGAConvIR.forward": str(inspect.signature(NoPostFGAConvIR.forward)),
        "NoPostPBCFGA.forward": str(inspect.signature(NoPostPBCFGA.forward)),
        "adapter_forward_forbidden_args_present": any(
            token in str(inspect.signature(NoPostPBCFGA.forward))
            for token in ["anchor", "teacher", "expert", "A0"]
        ),
        "final_rgb_plus_x_count": final_plus_x_count(repo_root),
    }

    source_audit = {
        "route": "haze4k-v2-13-nopost-feature-gated-adapter",
        "branch": git_value(["branch", "--show-current"], repo_root),
        "commit": git_value(["rev-parse", "--short", "HEAD"], repo_root),
        "status_short": git_value(["status", "--short"], repo_root),
        "python": str(Path(os.sys.executable)),
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "scan": {k: v for k, v in scan.items() if k != "text"},
        "signature": signature,
        "parity": {k: v for k, v in compare.items() if k != "per_image"},
        "locked_test_touched": False,
    }
    pass_contract = (
        not scan["hits"]
        and not signature["adapter_forward_forbidden_args_present"]
        and signature["final_rgb_plus_x_count"] == 1
        and compare["synthetic_max_abs_vs_A0"] <= 1e-7
        and compare["real_max_abs_vs_A0"] <= 1e-7
    )
    source_audit["pass"] = pass_contract

    (args.out_dir / "v213_n0_forbidden_symbol_scan.txt").write_text(
        scan["text"] if scan["text"] else "NO_FORBIDDEN_SYMBOL_HITS\n",
        encoding="utf-8",
    )
    write_json(args.out_dir / "v213_n0_source_audit.json", source_audit)
    write_json(args.out_dir / "v213_n0_forward_signature.json", signature)
    if compare["per_image"]:
        from nopost_common import write_csv

        write_csv(args.out_dir / "v213_n0_identity_sample_per_image.csv", compare["per_image"])

    (args.out_dir / "v213_n0_nopost_contract.md").write_text(
        "# v2.13 N0 NoPost Contract Audit\n\n"
        f"Decision: `{'N0_CONTRACT_PASS' if pass_contract else 'N0_CONTRACT_FAIL'}`\n\n"
        f"- forbidden symbol hits: `{len(scan['hits'])}`\n"
        f"- adapter forbidden args present: `{signature['adapter_forward_forbidden_args_present']}`\n"
        f"- final `rgb_residual + x` count: `{signature['final_rgb_plus_x_count']}`\n"
        f"- synthetic max_abs_vs_A0: `{compare['synthetic_max_abs_vs_A0']:.12g}`\n"
        f"- real sample max_abs_vs_A0: `{compare['real_max_abs_vs_A0']:.12g}`\n"
        f"- locked test touched: `false`\n",
        encoding="utf-8",
    )
    print("N0_NOPOST_CONTRACT_PASS" if pass_contract else "N0_NOPOST_CONTRACT_FAIL")
    print(json.dumps(source_audit, indent=2, sort_keys=True))
    if not pass_contract:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
