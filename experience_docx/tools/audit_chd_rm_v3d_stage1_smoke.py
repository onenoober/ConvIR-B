#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from d7c_gate import (  # noqa: E402
    build_d7c_gate_producer,
    collect_modulation_stats_with_optional_d7c,
    forward_with_optional_d7c,
)
from models.ConvIR import build_net  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
DEFAULT_DECISION_PASS = "V3D_RARM_STAGE1_1EPOCH_PASS_AUTHORIZE_STAGE1_5EPOCH_ADAPTER_ONLY_DECISION"
DEFAULT_DECISION_FAIL = "V3D_RARM_STAGE1_1EPOCH_FAIL_NO_CONTINUATION"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_state(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    if "splits" not in split:
        raise ValueError(f"Split JSON has no 'splits': {path}")
    return split


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [hazy_name, f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"]
    label_dir = Path(data_dir) / source_split / "gt"
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT match for {hazy_name}; tried {candidates} under {label_dir}")


def load_pair(data_dir, source_split, hazy_name):
    hazy_path = Path(data_dir) / source_split / "haze" / hazy_name
    label_path = label_path_for_hazy(data_dir, source_split, hazy_name)
    if not hazy_path.is_file():
        raise FileNotFoundError(f"Hazy image missing: {hazy_path}")
    image = Image.open(hazy_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    return TF.to_tensor(image), TF.to_tensor(label)


def pad_to_factor(input_img, factor=32):
    height, width = input_img.shape[2], input_img.shape[3]
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    pad_height = padded_height - height if height % factor != 0 else 0
    pad_width = padded_width - width if width % factor != 0 else 0
    if pad_height or pad_width:
        input_img = F.pad(input_img, (0, pad_width, 0, pad_height), "reflect")
    return input_img, height, width


def metric_summary(pred, label):
    pred = torch.clamp(pred, 0.0, 1.0)
    label = torch.clamp(label, 0.0, 1.0)
    mse = F.mse_loss(pred, label)
    psnr = float("inf") if mse.item() == 0 else (10.0 * torch.log10(1.0 / mse)).item()
    height, width = pred.shape[2], pred.shape[3]
    down_ratio = max(1, round(min(height, width) / 256))
    pooled_pred = F.adaptive_avg_pool2d(pred, (int(height / down_ratio), int(width / down_ratio)))
    pooled_label = F.adaptive_avg_pool2d(label, (int(height / down_ratio), int(width / down_ratio)))
    ssim_val = ssim(pooled_pred, pooled_label, data_range=1, size_average=False).mean().item()
    return psnr, ssim_val


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


def make_gate_args(args, gate_producer):
    class GateArgs:
        pass

    gate_args = GateArgs()
    gate_args.d7c_gate_producer = gate_producer
    return gate_args


def make_gate_build_args(args):
    class BuildArgs:
        pass

    build_args = BuildArgs()
    build_args.version = "base"
    build_args.data = "Haze4K"
    build_args.fam_mode = "fam2_d7c_noop"
    build_args.d7c_gate_mode = "d7c_fixed"
    build_args.d7c_base_checkpoint = args.a0_checkpoint
    build_args.d7c_density_artifact = args.density_artifact
    build_args.d7c_need_artifact = args.d7c_artifact
    build_args.d7c_threshold = args.d7c_threshold
    return build_args


def summarize_mod_stats(mod_stats_rows):
    if not mod_stats_rows:
        return {}
    keys = [key for key in mod_stats_rows[0] if key != "fam_name"]
    summary = {}
    for key in keys:
        values = [
            row[key]
            for row in mod_stats_rows
            if isinstance(row.get(key), (int, float))
        ]
        if values:
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_max"] = max(values)
    return summary


def audit(args):
    if args.source_split != "train":
        raise ValueError(f"v3d Stage 1 audit may only sample from Haze4K train, got {args.source_split}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        args.a0_checkpoint,
        args.candidate_checkpoint,
        args.data_dir,
        args.split_json,
        args.density_artifact,
        args.d7c_artifact,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", "fam2_d7c_noop").to(device).eval()
    base.load_state_dict(load_checkpoint_state(args.a0_checkpoint, device), strict=True)
    candidate.load_state_dict(load_checkpoint_state(args.candidate_checkpoint, device), strict=True)
    gate_producer = build_d7c_gate_producer(make_gate_build_args(args), device)
    gate_args = make_gate_args(args, gate_producer)

    split = load_split(args.split_json)
    names = sorted(split["splits"][args.split_key])[: args.max_samples]
    if not names:
        raise ValueError(f"No names in split {args.split_key}")

    rows = []
    mod_stats_rows = []
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            base_pred = base(padded)[2][:, :, :height, :width]
            cand_pred = forward_with_optional_d7c(candidate, gate_args, padded)[2][:, :, :height, :width]
            base_psnr, base_ssim = metric_summary(base_pred, label)
            cand_psnr, cand_ssim = metric_summary(cand_pred, label)
            output_diff = (cand_pred - base_pred).abs()
            gate, _, _ = gate_producer(padded)
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "base_psnr": base_psnr,
                    "candidate_psnr": cand_psnr,
                    "psnr_delta": cand_psnr - base_psnr,
                    "base_ssim": base_ssim,
                    "candidate_ssim": cand_ssim,
                    "ssim_delta": cand_ssim - base_ssim,
                    "output_max_abs_diff": output_diff.max().item(),
                    "output_mean_abs_diff": output_diff.mean().item(),
                    "gate_coverage": gate.mean().item(),
                }
            )
            mod_stats = collect_modulation_stats_with_optional_d7c(candidate, gate_args, padded)
            for fam_name, fam_stats in mod_stats.items():
                row = {"fam_name": fam_name}
                row.update(fam_stats)
                mod_stats_rows.append(row)

    deltas = [row["psnr_delta"] for row in rows]
    output_means = [row["output_mean_abs_diff"] for row in rows]
    output_maxes = [row["output_max_abs_diff"] for row in rows]
    summary = {
        "route_id": ROUTE_ID,
        "branch": os.popen("git branch --show-current").read().strip(),
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "candidate_checkpoint": args.candidate_checkpoint,
        "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
        "data_dir": args.data_dir,
        "split_json": args.split_json,
        "split_key": args.split_key,
        "source_split": args.source_split,
        "sample_count": len(rows),
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p10_psnr_delta": percentile(deltas, 10),
        "worst_psnr_delta": min(deltas),
        "positive_psnr_ratio": sum(1 for value in deltas if value > 0) / len(deltas),
        "regression_le_0p2_count": sum(1 for value in deltas if value <= -0.2),
        "regression_le_1p0_count": sum(1 for value in deltas if value <= -1.0),
        "mean_output_mean_abs_diff": statistics.mean(output_means),
        "max_output_max_abs_diff": max(output_maxes),
        "modulation_stats": summarize_mod_stats(mod_stats_rows),
        "locked_test_touched": False,
    }
    pass_checks = {
        "sample_count": len(rows) == args.max_samples,
        "finite_metrics": all(math.isfinite(row["candidate_psnr"]) for row in rows),
        "mean_not_collapsed": summary["mean_psnr_delta"] >= args.min_mean_delta,
        "p10_not_collapsed": summary["p10_psnr_delta"] >= args.min_p10_delta,
        "worst_not_catastrophic": summary["worst_psnr_delta"] >= args.min_worst_delta,
        "activity_nonzero": summary["mean_output_mean_abs_diff"] > args.min_mean_output_diff,
        "activity_bounded": summary["max_output_max_abs_diff"] <= args.max_output_diff,
        "locked_test_touched_false": True,
    }
    summary["pass_checks"] = pass_checks
    summary["pass"] = all(pass_checks.values())
    summary["decision"] = args.decision_pass if summary["pass"] else args.decision_fail
    summary["next_action"] = (
        args.next_action_pass
        if summary["pass"]
        else args.next_action_fail
    )

    prefix = f"v3d_{args.run_label}"
    write_csv(output_dir / f"{prefix}_val_inner_per_image.csv", rows)
    write_csv(output_dir / f"{prefix}_modulation_stats.csv", mod_stats_rows)
    write_json(output_dir / f"{prefix}_audit_summary.json", summary)
    write_json(
        output_dir / f"{prefix}_closeout.json",
        {
            "route_id": ROUTE_ID,
            "decision": summary["decision"],
            "pass": summary["pass"],
            "next_action": summary["next_action"],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--min_mean_delta", type=float, default=-0.30)
    parser.add_argument("--min_p10_delta", type=float, default=-1.0)
    parser.add_argument("--min_worst_delta", type=float, default=-3.0)
    parser.add_argument("--min_mean_output_diff", type=float, default=1e-8)
    parser.add_argument("--max_output_diff", type=float, default=0.10)
    parser.add_argument("--run_label", default="stage1_1epoch")
    parser.add_argument("--decision_pass", default=DEFAULT_DECISION_PASS)
    parser.add_argument("--decision_fail", default=DEFAULT_DECISION_FAIL)
    parser.add_argument(
        "--next_action_pass",
        default="Write a separate Stage 1 5-epoch adapter-only decision before continuation.",
    )
    parser.add_argument(
        "--next_action_fail",
        default="Stop v3d training continuation and inspect Stage 1 failure.",
    )
    args = parser.parse_args()
    raise SystemExit(audit(args))


if __name__ == "__main__":
    main()
