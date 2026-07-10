import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


def repo_root_from_script():
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root_from_script()
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from models.ConvIR import build_net  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "max_abs": tensor.abs().max().item(),
    }


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


def modulation_weight_stats(candidate):
    modulator = candidate.FAM2.modulator
    return {
        "FAM2.modulator.weight": tensor_stats(modulator.weight),
        "FAM2.modulator.bias": tensor_stats(modulator.bias),
        "fam1_has_modulator": hasattr(candidate.FAM1, "modulator"),
        "fam2_has_modulator": hasattr(candidate.FAM2, "modulator"),
    }


def make_models(args, device):
    allowed_missing = {"FAM2.modulator.weight", "FAM2.modulator.bias"}
    original = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", "fam2_modres").to(device).eval()
    original_load = partial_load_official_checkpoint(original, args.checkpoint, set())
    candidate_load = partial_load_official_checkpoint(candidate, args.checkpoint, allowed_missing)
    return original, candidate, original_load, candidate_load


def random_equivalence(original, candidate, args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x = torch.rand(
        1,
        3,
        args.random_height,
        args.random_width,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        base_outputs = original(x)
        cand_outputs = candidate(x)
        mod_stats = candidate.collect_modulation_stats(x)
    result = diff_outputs(base_outputs, cand_outputs)
    result.update(
        {
            "input_shape": list(x.shape),
            "seed": args.seed,
            "modulation_forward_stats": mod_stats,
            "pass": result["max_abs_diff"] <= args.max_abs_threshold,
        }
    )
    return result


def real_batch_equivalence(original, candidate, args, device):
    split = load_split(args.split_json)
    names = sorted(split["splits"][args.real_batch_split])[: args.real_batch_size]
    input_img, _ = load_image_batch(args.data_dir, args.source_split, names)
    input_img = input_img.to(device)
    padded, height, width = pad_to_factor(input_img)
    with torch.no_grad():
        base_outputs = crop_outputs(original(padded), height, width)
        cand_outputs = crop_outputs(candidate(padded), height, width)
        mod_stats = candidate.collect_modulation_stats(padded)
    result = diff_outputs(base_outputs, cand_outputs)
    result.update(
        {
            "input_shape": list(input_img.shape),
            "padded_shape": list(padded.shape),
            "batch_size": args.real_batch_size,
            "split_json": args.split_json,
            "split": args.real_batch_split,
            "image_names": names,
            "modulation_forward_stats": mod_stats,
            "pass": result["max_abs_diff"] <= args.max_abs_threshold,
        }
    )
    return result


def internal_val_equivalence(original, candidate, args, device):
    split = load_split(args.split_json)
    names = sorted(split["splits"][args.full_val_split])
    if args.val_limit > 0:
        names = names[: args.val_limit]
    rows = []
    max_abs = 0.0
    mean_abs_sum = 0.0
    count = 0
    psnr_delta_max_abs = 0.0
    ssim_delta_max_abs = 0.0
    factor = 32

    with torch.no_grad():
        for index, name in enumerate(names):
            input_tensor, label_tensor = load_pair(args.data_dir, args.source_split, name)
            input_img = input_tensor.unsqueeze(0).to(device)
            label_img = label_tensor.unsqueeze(0).to(device)
            height, width = input_img.shape[2], input_img.shape[3]
            padded, height, width = pad_to_factor(input_img, factor)

            base_pred = original(padded)[2][:, :, :height, :width]
            cand_pred = candidate(padded)[2][:, :, :height, :width]
            delta = (base_pred - cand_pred).abs()
            item_max = delta.max().item()
            item_mean = delta.mean().item()
            base_psnr, base_ssim = metric_summary(base_pred, label_img)
            cand_psnr, cand_ssim = metric_summary(cand_pred, label_img)
            psnr_delta = cand_psnr - base_psnr
            ssim_delta = cand_ssim - base_ssim
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "height": height,
                    "width": width,
                    "max_abs_diff": item_max,
                    "mean_abs_diff": item_mean,
                    "base_psnr": base_psnr,
                    "candidate_psnr": cand_psnr,
                    "psnr_delta": psnr_delta,
                    "base_ssim": base_ssim,
                    "candidate_ssim": cand_ssim,
                    "ssim_delta": ssim_delta,
                }
            )
            max_abs = max(max_abs, item_max)
            mean_abs_sum += item_mean
            psnr_delta_max_abs = max(psnr_delta_max_abs, abs(psnr_delta))
            ssim_delta_max_abs = max(ssim_delta_max_abs, abs(ssim_delta))
            count += 1

    mean_abs = mean_abs_sum / count if count else math.nan
    summary = {
        "samples": count,
        "split_json": args.split_json,
        "split": args.full_val_split,
        "source_split": args.source_split,
        "max_abs_diff": max_abs,
        "mean_abs_diff_mean": mean_abs,
        "psnr_delta_max_abs": psnr_delta_max_abs,
        "ssim_delta_max_abs": ssim_delta_max_abs,
        "psnr_delta_mean": sum(row["psnr_delta"] for row in rows) / count if count else math.nan,
        "ssim_delta_mean": sum(row["ssim_delta"] for row in rows) / count if count else math.nan,
        "pass": (
            count == args.expected_val_samples
            and max_abs <= args.max_abs_threshold
            and psnr_delta_max_abs <= args.metric_delta_threshold
            and ssim_delta_max_abs <= args.metric_delta_threshold
        ),
    }
    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--real_batch_split", default="val_inner")
    parser.add_argument("--full_val_split", default="val_inner")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--real_batch_size", type=int, default=2)
    parser.add_argument("--random_height", type=int, default=256)
    parser.add_argument("--random_width", type=int, default=256)
    parser.add_argument("--max_abs_threshold", type=float, default=1e-7)
    parser.add_argument("--metric_delta_threshold", type=float, default=1e-10)
    parser.add_argument("--expected_val_samples", type=int, default=600)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, candidate, original_load, candidate_load = make_models(args, device)
    original_params = count_parameters(original)
    candidate_params = count_parameters(candidate)
    param_delta = candidate_params - original_params

    state_dict_payload = {
        "candidate_mode": "fam2_modres",
        "original_mode": "original",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "original_partial_load": {
            "loaded_count": original_load["loaded_count"],
            "missing_candidate_keys": original_load["missing_candidate_keys"],
            "unexpected_keys": original_load["unexpected_keys"],
            "shape_mismatch": original_load["shape_mismatch"],
            "bad_missing": original_load["bad_missing"],
        },
        "candidate_partial_load": {
            "loaded_count": candidate_load["loaded_count"],
            "missing_candidate_keys": candidate_load["missing_candidate_keys"],
            "unexpected_keys": candidate_load["unexpected_keys"],
            "shape_mismatch": candidate_load["shape_mismatch"],
            "bad_missing": candidate_load["bad_missing"],
        },
        "original_params": original_params,
        "candidate_params": candidate_params,
        "param_delta": param_delta,
        "param_delta_pct": param_delta / original_params * 100.0,
        "expected_param_delta": 8320,
        "pass": (
            candidate_load["missing_candidate_keys"] == ["FAM2.modulator.weight", "FAM2.modulator.bias"]
            and not candidate_load["unexpected_keys"]
            and not candidate_load["shape_mismatch"]
            and param_delta == 8320
        ),
    }
    param_payload = {
        "original_params": original_params,
        "candidate_params": candidate_params,
        "param_delta": param_delta,
        "param_delta_pct": param_delta / original_params * 100.0,
        "expected_param_delta": 8320,
        "pass": param_delta == 8320,
    }
    zero_stats = modulation_weight_stats(candidate)
    zero_stats["pass"] = (
        zero_stats["FAM2.modulator.weight"]["max_abs"] == 0.0
        and zero_stats["FAM2.modulator.bias"]["max_abs"] == 0.0
        and zero_stats["fam1_has_modulator"] is False
        and zero_stats["fam2_has_modulator"] is True
    )

    random_result = random_equivalence(original, candidate, args, device)
    real_batch_result = real_batch_equivalence(original, candidate, args, device)
    full_val_summary, full_val_rows = internal_val_equivalence(original, candidate, args, device)

    write_json(output_dir / "fam2_state_dict_compatibility.json", state_dict_payload)
    write_json(output_dir / "fam2_param_delta_audit.json", param_payload)
    write_json(output_dir / "fam2_modulation_zero_stats.json", zero_stats)
    write_json(output_dir / "fam2_noop_random_equivalence.json", random_result)
    write_json(output_dir / "fam2_noop_real_batch_equivalence.json", real_batch_result)
    write_json(output_dir / "fam2_noop_internal_val600_summary.json", full_val_summary)
    write_csv(output_dir / "fam2_noop_per_scale_diff_summary.csv", full_val_rows)

    pass_all = all(
        [
            state_dict_payload["pass"],
            param_payload["pass"],
            zero_stats["pass"],
            random_result["pass"],
            real_batch_result["pass"],
            full_val_summary["pass"],
        ]
    )
    closeout = {
        "route_id": "haze4k_v5_chd_rm_v2i_fam2_noop_arch_equivalence_20260710",
        "source_branch": "github/codex/haze4k-official-arch-anchor",
        "candidate_mode": "fam2_modres",
        "training": "none",
        "locked_haze4k_test_usage": "none",
        "RARM": "not_connected_or_trained",
        "D7c_gate": "not_connected",
        "adapter_training": "none",
        "missing_candidate_keys_exact": candidate_load["missing_candidate_keys"],
        "unexpected_keys": candidate_load["unexpected_keys"],
        "param_delta": param_delta,
        "fam1_original": not hasattr(candidate.FAM1, "modulator"),
        "fam2_zero_init": zero_stats["pass"],
        "random_noop_pass": random_result["pass"],
        "real_batch_noop_pass": real_batch_result["pass"],
        "internal_val600_noop_pass": full_val_summary["pass"],
        "full_val600_noop_pass": full_val_summary["pass"],
        "full_val600_semantics": "train-derived internal val_inner 600, not locked Haze4K test",
        "max_abs_diff_gate": args.max_abs_threshold,
        "metric_delta_gate": args.metric_delta_threshold,
        "pass": pass_all,
        "decision_label": (
            "V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY"
            if pass_all
            else "V2I_FAM2_NOOP_ARCH_EQUIVALENCE_FAIL_FIX_ARCH_ONLY_NO_RARM"
        ),
    }
    write_json(output_dir / "fam2_noop_closeout.json", closeout)

    print(json.dumps(closeout, indent=2, sort_keys=True))
    if not pass_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
