#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as tvf
from PIL import Image


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _add_its_path():
    its = _repo_root() / "Dehazing" / "ITS"
    sys.path.insert(0, str(its))
    return its


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_value(args):
    return subprocess.check_output(args, cwd=_repo_root(), text=True).strip()


def _load_checkpoint_model(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _partial_load(model, state, allowed_new_prefixes):
    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            f"partial-load failed: unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}, bad_missing={bad_missing}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded": sorted(loaded),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def _count_train_files(data_dir):
    train_in = Path(data_dir) / "train" / "IN"
    if not train_in.is_dir():
        train_in = Path(data_dir) / "train" / "haze"
    if not train_in.is_dir():
        train_in = Path(data_dir) / "train" / "hazy"
    return len([p for p in train_in.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


def _first_train_pair(data_dir):
    input_dir = Path(data_dir) / "train" / "IN"
    label_dir = Path(data_dir) / "train" / "GT"
    if not input_dir.is_dir():
        input_dir = Path(data_dir) / "train" / "haze"
    if not input_dir.is_dir():
        input_dir = Path(data_dir) / "train" / "hazy"
    if not label_dir.is_dir():
        label_dir = Path(data_dir) / "train" / "gt"
    name = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[0]
    stem, ext = os.path.splitext(name)
    label_candidates = [name]
    if "_" in stem:
        label_candidates.append(f"{stem.split('_')[0]}{ext}")
        label_candidates.append(f"{stem.split('_')[0]}.png")
    label_path = None
    for candidate in label_candidates:
        candidate_path = label_dir / candidate
        if candidate_path.is_file():
            label_path = candidate_path
            break
    if label_path is None:
        raise FileNotFoundError(f"No matching label for {name}; tried {label_candidates}")
    image = Image.open(input_dir / name).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    image_t = tvf.to_tensor(image).unsqueeze(0)
    label_t = tvf.to_tensor(label).unsqueeze(0)
    crop_h = min(256, image_t.shape[-2], label_t.shape[-2])
    crop_w = min(256, image_t.shape[-1], label_t.shape[-1])
    crop_h = max(32, crop_h - crop_h % 32)
    crop_w = max(32, crop_w - crop_w % 32)
    return image_t[:, :, :crop_h, :crop_w], label_t[:, :, :crop_h, :crop_w], name


def _scope_count(model, scope):
    def is_wd(name):
        return name.startswith("WD_")

    def is_decoder(name):
        return (
            name.startswith("Decoder.")
            or name.startswith("Convs.")
            or name.startswith("ConvsOut.")
            or name.startswith("feat_extract.3.")
            or name.startswith("feat_extract.4.")
            or name.startswith("feat_extract.5.")
        )

    total = 0
    for name, param in model.named_parameters():
        active = scope == "all" or is_wd(name) or (scope == "wd_decoder" and is_decoder(name))
        if active:
            total += param.numel()
    return total


def _multiscale_loss(outputs, label):
    label2 = F.interpolate(label, scale_factor=0.5, mode="bilinear", align_corners=False)
    label4 = F.interpolate(label, scale_factor=0.25, mode="bilinear", align_corners=False)
    return (
        F.l1_loss(outputs[0], label4)
        + F.l1_loss(outputs[1], label2)
        + F.l1_loss(outputs[2], label)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="base")
    parser.add_argument("--data", default="Haze4K")
    args = parser.parse_args()

    _add_its_path()
    from models.ConvIR import build_net
    from models.ConvIRWD import build_convir_wd_net

    torch.manual_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = _load_checkpoint_model(args.checkpoint)

    official = build_net(args.version, args.data, "original").to(device)
    official.load_state_dict(state, strict=True)
    official.eval()

    model = build_convir_wd_net(args.version, args.data).to(device)
    report = _partial_load(model, state, ("WD_",))
    model.eval()

    synthetic = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        official_outputs = official(synthetic)
        wd_outputs = model(synthetic)
    noop_diffs = [
        float((wd - base).abs().max().detach().cpu())
        for wd, base in zip(wd_outputs, official_outputs)
    ]
    synthetic_finite = all(torch.isfinite(out).all().item() for out in wd_outputs)
    synthetic_shapes = [list(out.shape) for out in wd_outputs]

    input_img, label_img, sample_name = _first_train_pair(args.data_dir)
    input_img = input_img.to(device)
    label_img = label_img.to(device)
    with torch.no_grad():
        batch_outputs = model(input_img)
        one_batch_loss = float(_multiscale_loss(batch_outputs, label_img).detach().cpu())
    one_batch_finite = all(torch.isfinite(out).all().item() for out in batch_outputs)

    total_params = sum(p.numel() for p in model.parameters())
    wd_params = _scope_count(model, "wd_only")
    wd_decoder_params = _scope_count(model, "wd_decoder")

    result = {
        "route_id": "haze4k_v3_2_convir_wd_full_model_line_20260707",
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "--short", "HEAD"]),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "data_dir": args.data_dir,
        "train_sample_count": _count_train_files(args.data_dir),
        "validation_sample_count": None,
        "validation_sample_count_note": "P0 does not read locked/test validation data.",
        "partial_load": {
            "loaded_count": len(report["loaded"]),
            "missing_new_modules": report["missing_new_modules"],
            "unexpected": report["unexpected"],
            "shape_mismatch": report["shape_mismatch"],
        },
        "parameter_count_total": total_params,
        "parameter_count_trainable_by_scope": {
            "wd_only": wd_params,
            "wd_decoder": wd_decoder_params,
            "all": total_params,
        },
        "synthetic_output_shapes": synthetic_shapes,
        "synthetic_forward_finite": synthetic_finite,
        "noop_or_bounded_diff_vs_a0": {
            "max_abs_by_output": noop_diffs,
            "max_abs": max(noop_diffs),
            "contract": "exact no-op expected from zero-init WD projections",
        },
        "one_batch_sample": sample_name,
        "one_batch_forward_finite": one_batch_finite,
        "one_batch_loss": one_batch_loss,
        "locked_test_touched": False,
        "pass": bool(
            synthetic_finite
            and one_batch_finite
            and max(noop_diffs) == 0.0
            and not report["unexpected"]
            and not report["shape_mismatch"]
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
