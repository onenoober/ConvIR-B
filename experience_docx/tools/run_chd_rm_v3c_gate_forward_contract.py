#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
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
    forward_with_optional_d7c,
    collect_modulation_stats_with_optional_d7c,
    get_d7c_gate,
    partial_load_model_state,
)
from models.ConvIR import build_net  # noqa: E402


ROUTE_ID = "haze4k_v5_chd_rm_v3c_gate_forward_contract_20260710"
DECISION_PASS = "V3C_GATE_FORWARD_CONTRACT_PASS_AUTHORIZE_NO_TRAINING_ENTRYPOINT_PREFLIGHT_ONLY"
DECISION_FAIL = "V3C_GATE_FORWARD_CONTRACT_FAIL_PAUSE"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_models(args, device):
    base = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", "fam2_d7c_noop").to(device).eval()
    checkpoint_state = load_checkpoint_state(args.checkpoint)
    base.load_state_dict(checkpoint_state, strict=True)
    candidate_load = partial_load_model_state(
        candidate,
        checkpoint_state,
        {"FAM2.modulator.weight", "FAM2.modulator.bias"},
    )
    return base, candidate, candidate_load


def load_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    if "splits" not in split:
        raise ValueError(f"Split JSON has no 'splits' object: {path}")
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


def crop_outputs(outputs, height, width):
    cropped = []
    for index, tensor in enumerate(outputs):
        scale = 4 if index == 0 else 2 if index == 1 else 1
        cropped.append(tensor[:, :, : math.ceil(height / scale), : math.ceil(width / scale)])
    return cropped


def diff_outputs(base_outputs, cand_outputs):
    diffs = []
    for index, (base, cand) in enumerate(zip(base_outputs, cand_outputs)):
        delta = (base - cand).abs()
        diffs.append(
            {
                "output_index": index,
                "shape": list(base.shape),
                "max_abs_diff": delta.max().item(),
                "mean_abs_diff": delta.mean().item(),
            }
        )
    return {
        "outputs": diffs,
        "max_abs_diff": max(item["max_abs_diff"] for item in diffs),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in diffs),
    }


def tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "max_abs": tensor.abs().max().item(),
    }


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


def gate_summary(gate):
    return {
        "gate": tensor_stats(gate),
        "selected_coverage": gate.mean().item(),
        "gate_shape": list(gate.shape),
        "nontrivial_gate": bool(gate.min().item() < gate.max().item()),
    }


def make_gate_args(args):
    class GateArgs:
        pass

    gate_args = GateArgs()
    gate_args.version = "base"
    gate_args.data = "Haze4K"
    gate_args.fam_mode = "fam2_d7c_noop"
    gate_args.d7c_gate_mode = "d7c_fixed"
    gate_args.d7c_base_checkpoint = args.checkpoint
    gate_args.d7c_density_artifact = args.density_artifact
    gate_args.d7c_need_artifact = args.d7c_artifact
    gate_args.d7c_threshold = args.d7c_threshold
    return gate_args


def static_source_contract(repo_root):
    checks = []
    expected = [
        ("Dehazing/ITS/train.py", "forward_with_optional_d7c(model, args, input_img)"),
        ("Dehazing/ITS/train.py", "collect_modulation_stats_with_optional_d7c(model, args, input_img)"),
        ("Dehazing/ITS/valid.py", "forward_with_optional_d7c(model, args, input_img)[2]"),
        ("Dehazing/ITS/eval.py", "forward_with_optional_d7c(model, args, input_img)[2]"),
        ("Dehazing/ITS/main.py", "args.d7c_gate_producer = build_d7c_gate_producer(args, device)"),
        ("Dehazing/ITS/main.py", "--allow_fam2_partial_init"),
    ]
    for relative_path, needle in expected:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        checks.append({"path": relative_path, "needle": needle, "pass": needle in text})
    return {"checks": checks, "pass": all(item["pass"] for item in checks)}


def run_contract_audit(args):
    assert_no_locked_test_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [args.checkpoint, args.data_dir, args.split_json, args.density_artifact, args.d7c_artifact]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    base, candidate, candidate_load = load_models(args, device)
    gate_args = make_gate_args(args)
    gate_args.d7c_gate_producer = build_d7c_gate_producer(gate_args, device)

    static_contract = static_source_contract(REPO_ROOT)
    write_json(output_dir / "v3c_static_source_contract.json", static_contract)

    split = load_split(args.split_json)
    names = sorted(split["splits"][args.split_key])[: args.max_samples]
    rows = []
    max_abs = 0.0
    max_psnr_delta = 0.0
    max_ssim_delta = 0.0
    nontrivial_count = 0
    mod_stats_seen = False
    gate_coverages = []

    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            gate = get_d7c_gate(gate_args, padded)
            base_outputs = crop_outputs(base(padded), height, width)
            cand_outputs = crop_outputs(forward_with_optional_d7c(candidate, gate_args, padded), height, width)
            mod_stats = collect_modulation_stats_with_optional_d7c(candidate, gate_args, padded)
            diff = diff_outputs(base_outputs, cand_outputs)
            base_psnr, base_ssim = metric_summary(base_outputs[-1], label)
            cand_psnr, cand_ssim = metric_summary(cand_outputs[-1], label)
            psnr_delta = cand_psnr - base_psnr
            ssim_delta = cand_ssim - base_ssim
            coverage = gate.mean().item()
            gate_coverages.append(coverage)
            max_abs = max(max_abs, diff["max_abs_diff"])
            max_psnr_delta = max(max_psnr_delta, abs(psnr_delta))
            max_ssim_delta = max(max_ssim_delta, abs(ssim_delta))
            if gate.min().item() < gate.max().item():
                nontrivial_count += 1
            if "FAM2" in mod_stats and "d7c_gate_mean" in mod_stats["FAM2"]:
                mod_stats_seen = True
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "height": height,
                    "width": width,
                    "max_abs_diff": diff["max_abs_diff"],
                    "mean_abs_diff": diff["mean_abs_diff"],
                    "base_psnr": base_psnr,
                    "candidate_psnr": cand_psnr,
                    "psnr_delta": psnr_delta,
                    "base_ssim": base_ssim,
                    "candidate_ssim": cand_ssim,
                    "ssim_delta": ssim_delta,
                    "d7c_selected_coverage": coverage,
                    "d7c_gate_nontrivial": bool(gate.min().item() < gate.max().item()),
                }
            )
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(f"v3c_forward_contract {index + 1}/{len(names)} max_abs={max_abs:.3e}", flush=True)

    candidate_missing_ok = sorted(candidate_load["missing_candidate_keys"]) == [
        "FAM2.modulator.bias",
        "FAM2.modulator.weight",
    ]
    summary = {
        "route_id": ROUTE_ID,
        "samples": len(rows),
        "source_split": args.source_split,
        "split_key": args.split_key,
        "candidate_load": candidate_load,
        "candidate_missing_ok": candidate_missing_ok,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "density_artifact_sha256": sha256_file(args.density_artifact),
        "d7c_artifact_sha256": sha256_file(args.d7c_artifact),
        "d7c_threshold": args.d7c_threshold,
        "max_abs_diff": max_abs,
        "psnr_delta_max_abs": max_psnr_delta,
        "ssim_delta_max_abs": max_ssim_delta,
        "d7c_selected_coverage_mean": float(np.mean(gate_coverages)) if gate_coverages else math.nan,
        "d7c_selected_coverage_min": float(np.min(gate_coverages)) if gate_coverages else math.nan,
        "d7c_selected_coverage_max": float(np.max(gate_coverages)) if gate_coverages else math.nan,
        "nontrivial_gate_images": nontrivial_count,
        "modulation_stats_gate_seen": mod_stats_seen,
        "static_source_contract_pass": static_contract["pass"],
        "pass": (
            candidate_missing_ok
            and static_contract["pass"]
            and mod_stats_seen
            and len(rows) == args.max_samples
            and nontrivial_count > 0
            and max_abs <= args.max_abs_threshold
            and max_psnr_delta <= args.metric_delta_threshold
            and max_ssim_delta <= args.metric_delta_threshold
        ),
    }
    write_json(output_dir / "v3c_gate_forward_contract_summary.json", summary)
    write_csv(output_dir / "v3c_gate_forward_contract_per_image.csv", rows)
    write_json(
        output_dir / "forbidden_flow_audit.json",
        {
            "training": "none",
            "RARM": "not_connected_or_trained",
            "adapter_training": "none",
            "ConvIR_B_unfreeze": "none",
            "locked_haze4k_test_usage": "none",
            "canary_expansion": "none",
            "gate_mode": "d7c_fixed",
        },
    )
    closeout = {
        "route_id": ROUTE_ID,
        "decision_label": DECISION_PASS if summary["pass"] else DECISION_FAIL,
        "pass": summary["pass"],
        "samples": summary["samples"],
        "max_abs_diff": max_abs,
        "psnr_delta_max_abs": max_psnr_delta,
        "ssim_delta_max_abs": max_ssim_delta,
        "nontrivial_gate_images": nontrivial_count,
        "modulation_stats_gate_seen": mod_stats_seen,
        "training": "none",
        "RARM": "not_connected_or_trained",
        "adapter_training": "none",
        "locked_haze4k_test_usage": "none",
        "next_authorized_stage": (
            "no-training entrypoint preflight is passed; RARM/training still requires a separate written decision"
            if summary["pass"]
            else "pause; inspect gate forward contract"
        ),
    }
    write_json(output_dir / "v3c_gate_forward_contract_closeout.json", closeout)
    print("V3C_GATE_FORWARD_CONTRACT_OK" if summary["pass"] else "V3C_GATE_FORWARD_CONTRACT_FAILED")
    if not summary["pass"]:
        raise SystemExit(1)


def assert_no_locked_test_paths(args):
    forbidden = []
    for label, value in vars(args).items():
        if value is None:
            continue
        text = str(value).lower().replace("\\", "/")
        if "/test/" in text or "locked" in text:
            forbidden.append({"arg": label, "value": str(value)})
    if forbidden:
        raise RuntimeError(f"locked/test path detected in arguments: {forbidden}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--max_abs_threshold", type=float, default=1e-7)
    parser.add_argument("--metric_delta_threshold", type=float, default=1e-10)
    parser.add_argument("--progress_every", type=int, default=8)
    return parser.parse_args()


def main():
    run_contract_audit(parse_args())


if __name__ == "__main__":
    main()
