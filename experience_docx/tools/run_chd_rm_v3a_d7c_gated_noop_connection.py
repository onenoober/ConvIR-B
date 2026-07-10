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
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from models.ConvIR import build_net  # noqa: E402


GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
D7C_FIXED_THRESHOLD = 0.5773006677627563


class DensityNeedHead(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class MultiContextNeedHead(nn.Module):
    def __init__(self, in_channels=234, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def load_checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def partial_load_official_checkpoint(model, checkpoint_path, allowed_missing):
    checkpoint_state = load_checkpoint_state(checkpoint_path)
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if key not in allowed_missing]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected}, shape_mismatch={shape_mismatch}, "
            f"bad_missing={bad_missing}, missing={missing}"
        )

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "missing_candidate_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "bad_missing": bad_missing,
    }


def load_density_head(path, device):
    head = DensityNeedHead(1).to(device)
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return head


def load_d7c_head(path, device):
    head = MultiContextNeedHead().to(device)
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return head


def tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "max_abs": tensor.abs().max().item(),
    }


def gray_abs(a, b):
    weights = GRAY_WEIGHTS.to(device=a.device, dtype=a.dtype)
    return ((a - b).abs() * weights).sum(dim=1, keepdim=True)


def smooth_map(x, kernel):
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)


def raw_density(hazy, gt, kernel):
    return smooth_map(gray_abs(hazy, gt), kernel)


def normalize(raw, lo, hi):
    denom = max(float(hi) - float(lo), 1e-6)
    return torch.clamp((raw - float(lo)) / denom, 0.0, 1.0)


def density_pred_from_head(head, res1):
    return torch.sigmoid(head(res1))


def convir_a0_context(model, density_head, x):
    x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x_4 = F.interpolate(x_2, scale_factor=0.5, mode="bilinear", align_corners=False)
    z2 = model.SCM2(x_2)
    z4 = model.SCM1(x_4)
    x0 = model.feat_extract[0](x)
    res1 = model.Encoder[0](x0)
    z = model.feat_extract[1](res1)
    z = model.FAM2(z, z2)
    res2 = model.Encoder[1](z)
    z = model.feat_extract[2](res2)
    z = model.FAM1(z, z4)
    bottleneck = model.Encoder[2](z)

    z = model.Decoder[0](bottleneck)
    z = model.feat_extract[3](z)
    z = torch.cat([z, res2], dim=1)
    z = model.Convs[0](z)
    z = model.Decoder[1](z)
    z = model.feat_extract[4](z)
    z = torch.cat([z, res1], dim=1)
    z = model.Convs[1](z)
    z = model.Decoder[2](z)
    z = model.feat_extract[5](z)
    a0 = torch.clamp(z + x, 0, 1)

    res2_up = F.interpolate(res2, size=res1.shape[-2:], mode="bilinear", align_corners=False)
    bottleneck_up = F.interpolate(bottleneck, size=res1.shape[-2:], mode="bilinear", align_corners=False)
    density_pred = density_pred_from_head(density_head, res1)
    context = torch.cat([res1, res2_up, bottleneck_up, x, a0, (x - a0).abs(), density_pred], dim=1)
    return a0, context


def predict_d7c(head, context):
    logits = head(context)
    probs = [torch.sigmoid(logits[:, i : i + 1]) for i in range(4)]
    pred = torch.stack(probs, dim=0).mean(dim=0)
    return pred, logits


def make_d7c_gate(base_model, density_head, d7c_head, x, threshold):
    _, context = convir_a0_context(base_model, density_head, x)
    score, logits = predict_d7c(d7c_head, context)
    gate = (score >= threshold).to(dtype=x.dtype)
    return gate, score, logits


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


def metric_summary(pred, label):
    pred = torch.clamp(pred, 0.0, 1.0)
    label = torch.clamp(label, 0.0, 1.0)
    mse = F.mse_loss(pred, label)
    if mse.item() == 0:
        psnr = float("inf")
    else:
        psnr = (10.0 * torch.log10(1.0 / mse)).item()

    height, width = pred.shape[2], pred.shape[3]
    down_ratio = max(1, round(min(height, width) / 256))
    pooled_pred = F.adaptive_avg_pool2d(pred, (int(height / down_ratio), int(width / down_ratio)))
    pooled_label = F.adaptive_avg_pool2d(label, (int(height / down_ratio), int(width / down_ratio)))
    ssim_val = ssim(pooled_pred, pooled_label, data_range=1, size_average=False).mean().item()
    return psnr, ssim_val


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


def load_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    if "splits" not in split:
        raise ValueError(f"Split JSON has no 'splits' object: {path}")
    return split


def label_path_for_hazy(data_dir, source_split, hazy_name):
    stem, ext = os.path.splitext(hazy_name)
    candidates = [
        hazy_name,
        f"{stem.split('_')[0]}{ext}",
        f"{stem.split('_')[0]}.png",
    ]
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


def load_image_batch(data_dir, source_split, names):
    images = []
    labels = []
    for name in names:
        image, label = load_pair(data_dir, source_split, name)
        images.append(image)
        labels.append(label)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def gate_summary(gate, score):
    return {
        "gate": tensor_stats(gate),
        "score": tensor_stats(score),
        "selected_coverage": gate.mean().item(),
        "threshold": D7C_FIXED_THRESHOLD,
        "gate_shape": list(gate.shape),
        "score_shape": list(score.shape),
        "nontrivial_gate": bool(gate.min().item() < gate.max().item()),
    }


def modulation_weight_stats(candidate):
    modulator = candidate.FAM2.modulator
    return {
        "FAM2.modulator.weight": tensor_stats(modulator.weight),
        "FAM2.modulator.bias": tensor_stats(modulator.bias),
        "fam1_has_modulator": hasattr(candidate.FAM1, "modulator"),
        "fam2_has_modulator": hasattr(candidate.FAM2, "modulator"),
        "fam2_mode": getattr(candidate.FAM2, "mode", None),
        "fam2_modulator_in_channels": int(modulator.in_channels),
        "fam2_modulator_out_channels": int(modulator.out_channels),
    }


def make_models(args, device):
    allowed_missing = {"FAM2.modulator.weight", "FAM2.modulator.bias"}
    original = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", "fam2_d7c_noop").to(device).eval()
    original_load = partial_load_official_checkpoint(original, args.checkpoint, set())
    candidate_load = partial_load_official_checkpoint(candidate, args.checkpoint, allowed_missing)
    return original, candidate, original_load, candidate_load


def random_equivalence(original, candidate, density_head, d7c_head, args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x = torch.rand(1, 3, args.random_height, args.random_width, generator=generator, device=device)
    with torch.no_grad():
        gate, score, _ = make_d7c_gate(original, density_head, d7c_head, x, D7C_FIXED_THRESHOLD)
        base_outputs = original(x)
        cand_outputs = candidate(x, d7c_gate=gate)
        mod_stats = candidate.collect_modulation_stats(x, d7c_gate=gate)
    result = diff_outputs(base_outputs, cand_outputs)
    result.update(
        {
            "input_shape": list(x.shape),
            "seed": args.seed,
            "d7c": gate_summary(gate, score),
            "modulation_forward_stats": mod_stats,
            "pass": result["max_abs_diff"] <= args.max_abs_threshold,
        }
    )
    return result


def real_batch_equivalence(original, candidate, density_head, d7c_head, args, device):
    split = load_split(args.split_json)
    names = sorted(split["splits"][args.real_batch_split])[: args.real_batch_size]
    input_img, _ = load_image_batch(args.data_dir, args.source_split, names)
    input_img = input_img.to(device)
    padded, height, width = pad_to_factor(input_img)
    with torch.no_grad():
        gate, score, _ = make_d7c_gate(original, density_head, d7c_head, padded, D7C_FIXED_THRESHOLD)
        base_outputs = crop_outputs(original(padded), height, width)
        cand_outputs = crop_outputs(candidate(padded, d7c_gate=gate), height, width)
        mod_stats = candidate.collect_modulation_stats(padded, d7c_gate=gate)
    result = diff_outputs(base_outputs, cand_outputs)
    result.update(
        {
            "input_shape": list(input_img.shape),
            "padded_shape": list(padded.shape),
            "batch_size": args.real_batch_size,
            "names": names,
            "source_split": args.source_split,
            "split_key": args.real_batch_split,
            "d7c": gate_summary(gate, score),
            "modulation_forward_stats": mod_stats,
            "pass": result["max_abs_diff"] <= args.max_abs_threshold,
        }
    )
    return result


def full_internal_val_equivalence(original, candidate, density_head, d7c_head, args, device):
    split = load_split(args.split_json)
    names = sorted(split["splits"][args.full_split])
    rows = []
    max_abs = 0.0
    max_mean = 0.0
    max_psnr_delta = 0.0
    max_ssim_delta = 0.0
    selected_coverages = []
    nontrivial_count = 0
    with torch.no_grad():
        for index, name in enumerate(names):
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            gate, score, _ = make_d7c_gate(original, density_head, d7c_head, padded, D7C_FIXED_THRESHOLD)
            base_outputs = crop_outputs(original(padded), height, width)
            cand_outputs = crop_outputs(candidate(padded, d7c_gate=gate), height, width)
            diff = diff_outputs(base_outputs, cand_outputs)
            base_final = base_outputs[-1]
            cand_final = cand_outputs[-1]
            base_psnr, base_ssim = metric_summary(base_final, label)
            cand_psnr, cand_ssim = metric_summary(cand_final, label)
            psnr_delta = cand_psnr - base_psnr
            ssim_delta = cand_ssim - base_ssim
            coverage = gate.mean().item()
            selected_coverages.append(coverage)
            if gate.min().item() < gate.max().item():
                nontrivial_count += 1
            row = {
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
                "d7c_score_mean": score.mean().item(),
                "d7c_score_min": score.min().item(),
                "d7c_score_max": score.max().item(),
            }
            rows.append(row)
            max_abs = max(max_abs, diff["max_abs_diff"])
            max_mean = max(max_mean, diff["mean_abs_diff"])
            max_psnr_delta = max(max_psnr_delta, abs(psnr_delta))
            max_ssim_delta = max(max_ssim_delta, abs(ssim_delta))
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(f"val_equivalence {index + 1}/{len(names)} max_abs={max_abs:.3e}", flush=True)
    summary = {
        "samples": len(rows),
        "source_split": args.source_split,
        "split_key": args.full_split,
        "max_abs_diff": max_abs,
        "max_mean_abs_diff": max_mean,
        "psnr_delta_max_abs": max_psnr_delta,
        "ssim_delta_max_abs": max_ssim_delta,
        "d7c_selected_coverage_mean": float(np.mean(selected_coverages)) if selected_coverages else math.nan,
        "d7c_selected_coverage_min": float(np.min(selected_coverages)) if selected_coverages else math.nan,
        "d7c_selected_coverage_max": float(np.max(selected_coverages)) if selected_coverages else math.nan,
        "nontrivial_gate_images": nontrivial_count,
        "pass": (
            max_abs <= args.max_abs_threshold
            and max_psnr_delta <= args.metric_delta_threshold
            and max_ssim_delta <= args.metric_delta_threshold
            and nontrivial_count > 0
        ),
    }
    return summary, rows


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
    parser.add_argument("--real_batch_split", default="val_inner")
    parser.add_argument("--full_split", default="val_inner")
    parser.add_argument("--real_batch_size", type=int, default=4)
    parser.add_argument("--random_height", type=int, default=256)
    parser.add_argument("--random_width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_abs_threshold", type=float, default=1e-7)
    parser.add_argument("--metric_delta_threshold", type=float, default=1e-10)
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
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

    original, candidate, original_load, candidate_load = make_models(args, device)
    density_head = load_density_head(args.density_artifact, device)
    d7c_head = load_d7c_head(args.d7c_artifact, device)

    base_params = count_parameters(original)
    cand_params = count_parameters(candidate)
    param_delta = cand_params - base_params
    expected_missing = ["FAM2.modulator.weight", "FAM2.modulator.bias"]
    expected_shapes = {
        "FAM2.modulator.weight": [128, 64, 1, 1],
        "FAM2.modulator.bias": [128],
    }
    state_ok = (
        sorted(candidate_load["missing_candidate_keys"]) == expected_missing
        and candidate_load["unexpected_keys"] == []
        and candidate_load["shape_mismatch"] == []
        and param_delta == 8320
        and list(candidate.FAM2.modulator.weight.shape) == expected_shapes["FAM2.modulator.weight"]
        and list(candidate.FAM2.modulator.bias.shape) == expected_shapes["FAM2.modulator.bias"]
    )

    write_json(
        output_dir / "d7c_noop_state_dict_compatibility.json",
        {
            "pass": state_ok,
            "original_load": original_load,
            "candidate_load": candidate_load,
            "expected_missing_candidate_keys": expected_missing,
            "expected_new_param_shapes": expected_shapes,
            "base_params": base_params,
            "candidate_params": cand_params,
            "param_delta": param_delta,
            "expected_param_delta": 8320,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "density_artifact_sha256": sha256_file(args.density_artifact),
            "d7c_artifact_sha256": sha256_file(args.d7c_artifact),
        },
    )

    mod_zero = modulation_weight_stats(candidate)
    mod_zero["pass"] = (
        mod_zero["FAM2.modulator.weight"]["max_abs"] == 0.0
        and mod_zero["FAM2.modulator.bias"]["max_abs"] == 0.0
        and mod_zero["fam2_modulator_in_channels"] == 64
    )
    write_json(output_dir / "d7c_noop_modulation_zero_stats.json", mod_zero)

    random_result = random_equivalence(original, candidate, density_head, d7c_head, args, device)
    write_json(output_dir / "d7c_noop_random_equivalence.json", random_result)

    real_batch_result = real_batch_equivalence(original, candidate, density_head, d7c_head, args, device)
    write_json(output_dir / "d7c_noop_real_batch_equivalence.json", real_batch_result)

    full_summary, full_rows = full_internal_val_equivalence(original, candidate, density_head, d7c_head, args, device)
    write_json(output_dir / "d7c_noop_internal_val600_summary.json", full_summary)
    write_csv(output_dir / "d7c_noop_per_image_diff_summary.csv", full_rows)

    forbidden_flow = {
        "training": "none",
        "RARM": "not_connected_or_trained",
        "adapter_training": "none",
        "ConvIR_B_unfreeze": "none",
        "locked_haze4k_test_usage": "none",
        "D7c_gate_forward_connection": "connected_as_external_gate_tensor",
        "final_modulation": "zero_init_noop",
        "source_split": args.source_split,
    }
    write_json(output_dir / "forbidden_flow_audit.json", forbidden_flow)

    closeout_pass = (
        state_ok
        and mod_zero["pass"]
        and random_result["pass"]
        and real_batch_result["pass"]
        and full_summary["pass"]
    )
    closeout = {
        "route_id": "haze4k_v5_chd_rm_v3a_d7c_gated_noop_connection_audit_20260710",
        "source_branch": "github/codex/haze4k-official-arch-anchor",
        "candidate_mode": "fam2_d7c_noop",
        "decision_label": (
            "V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY"
            if closeout_pass
            else "V3A_D7C_GATED_NOOP_CONNECTION_FAIL_PAUSE"
        ),
        "pass": closeout_pass,
        "param_delta": param_delta,
        "expected_missing_candidate_keys": expected_missing,
        "d7c_gate_forward_connection": "connected",
        "d7c_fixed_threshold": D7C_FIXED_THRESHOLD,
        "random_noop_pass": random_result["pass"],
        "real_batch_noop_pass": real_batch_result["pass"],
        "internal_val600_noop_pass": full_summary["pass"],
        "internal_val600_nontrivial_gate_images": full_summary["nontrivial_gate_images"],
        "internal_val600_max_abs_diff": full_summary["max_abs_diff"],
        "internal_val600_psnr_delta_max_abs": full_summary["psnr_delta_max_abs"],
        "internal_val600_ssim_delta_max_abs": full_summary["ssim_delta_max_abs"],
        "training": "none",
        "RARM": "not_connected_or_trained",
        "adapter_training": "none",
        "locked_haze4k_test_usage": "none",
        "next_authorized_stage": (
            "RARM/training still blocked; only a separate preflight design decision may be written"
            if closeout_pass
            else "pause; inspect D7c gate connection or no-op implementation"
        ),
    }
    write_json(output_dir / "d7c_noop_closeout.json", closeout)
    print("V3A_D7C_GATED_NOOP_CONNECTION_OK" if closeout_pass else "V3A_D7C_GATED_NOOP_CONNECTION_FAILED")
    if not closeout_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
