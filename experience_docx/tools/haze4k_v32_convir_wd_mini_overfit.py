#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as tvf


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _add_its_path():
    its = _repo_root() / "Dehazing" / "ITS"
    sys.path.insert(0, str(its))
    return its


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


def _image_dirs(data_dir):
    input_dir = Path(data_dir) / "train" / "IN"
    label_dir = Path(data_dir) / "train" / "GT"
    if not input_dir.is_dir():
        input_dir = Path(data_dir) / "train" / "haze"
    if not input_dir.is_dir():
        input_dir = Path(data_dir) / "train" / "hazy"
    if not label_dir.is_dir():
        label_dir = Path(data_dir) / "train" / "gt"
    return input_dir, label_dir


def _load_pairs(data_dir, sample_count, crop_size):
    input_dir, label_dir = _image_dirs(data_dir)
    names = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    names = names[:sample_count]
    inputs = []
    labels = []
    for name in names:
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
        image = tvf.to_tensor(Image.open(input_dir / name).convert("RGB"))
        label = tvf.to_tensor(Image.open(label_path).convert("RGB"))
        h = min(image.shape[-2], label.shape[-2])
        w = min(image.shape[-1], label.shape[-1])
        crop = min(crop_size, h, w)
        crop = max(32, crop - crop % 32)
        top = max(0, (h - crop) // 2)
        left = max(0, (w - crop) // 2)
        inputs.append(image[:, top:top + crop, left:left + crop])
        labels.append(label[:, top:top + crop, left:left + crop])
    return torch.stack(inputs, dim=0), torch.stack(labels, dim=0), names


def _is_wd(name):
    return name.startswith("WD_")


def _is_decoder(name):
    return (
        name.startswith("Decoder.")
        or name.startswith("Convs.")
        or name.startswith("ConvsOut.")
        or name.startswith("feat_extract.3.")
        or name.startswith("feat_extract.4.")
        or name.startswith("feat_extract.5.")
    )


def _set_scope(model, scope, wd_lr, decoder_lr):
    for _, param in model.named_parameters():
        param.requires_grad = False
    wd_params = []
    decoder_params = []
    all_params = []
    for name, param in model.named_parameters():
        if scope == "all":
            param.requires_grad = True
            all_params.append(param)
        elif _is_wd(name):
            param.requires_grad = True
            wd_params.append(param)
        elif scope == "wd_decoder" and _is_decoder(name):
            param.requires_grad = True
            decoder_params.append(param)
    groups = []
    if wd_params:
        groups.append({"params": wd_params, "lr": wd_lr, "name": "WD"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder"})
    if all_params:
        groups.append({"params": all_params, "lr": wd_lr, "name": "all"})
    if not groups:
        raise ValueError(f"No trainable parameters for scope={scope}")
    return groups


def _haar_dwt2(x):
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


def _rgb_to_y(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _loss(outputs, label):
    label2 = F.interpolate(label, scale_factor=0.5, mode="bilinear", align_corners=False)
    label4 = F.interpolate(label, scale_factor=0.25, mode="bilinear", align_corners=False)
    content = (
        F.l1_loss(outputs[0], label4)
        + F.l1_loss(outputs[1], label2)
        + F.l1_loss(outputs[2], label)
    )
    final = outputs[2]
    pred_ll, pred_lh, pred_hl, pred_hh = _haar_dwt2(final)
    label_ll, label_lh, label_hl, label_hh = _haar_dwt2(label)
    dwt_low = F.l1_loss(pred_ll, label_ll)
    dwt_high = (
        F.l1_loss(pred_lh, label_lh)
        + F.l1_loss(pred_hl, label_hl)
        + F.l1_loss(pred_hh, label_hh)
    ) / 3.0
    y_loss = F.l1_loss(_rgb_to_y(final), _rgb_to_y(label))
    total = content + 0.05 * dwt_low + 0.01 * dwt_high + 0.05 * y_loss
    return total, {
        "content": float(content.detach().cpu()),
        "dwt_low": float(dwt_low.detach().cpu()),
        "dwt_high": float(dwt_high.detach().cpu()),
        "y": float(y_loss.detach().cpu()),
    }


def _mean_activity(stats):
    values = []
    for block in stats.values():
        values.extend(abs(v) for v in block.values())
    return float(sum(values) / max(1, len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--scope", default="wd_decoder", choices=["wd_only", "wd_decoder", "all"])
    parser.add_argument("--wd_lr", type=float, default=2e-4)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gate_loss_ratio", type=float, default=0.95)
    parser.add_argument("--version", default="base")
    parser.add_argument("--data", default="Haze4K")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _add_its_path()
    from models.ConvIRWD import build_convir_wd_net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_convir_wd_net(args.version, args.data).to(device)
    report = _partial_load(model, _load_checkpoint_model(args.checkpoint), ("WD_",))
    groups = _set_scope(model, args.scope, args.wd_lr, args.decoder_lr)
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

    inputs, labels, names = _load_pairs(args.data_dir, args.samples, args.crop_size)
    inputs = inputs.to(device)
    labels = labels.to(device)
    model.train()

    with torch.no_grad():
        initial_outputs = model(inputs[:args.batch_size])
        initial_loss, initial_parts = _loss(initial_outputs, labels[:args.batch_size])
        initial_stats = model.collect_wd_stats(inputs[:args.batch_size])

    history = []
    n = inputs.shape[0]
    for step in range(1, args.steps + 1):
        start = ((step - 1) * args.batch_size) % n
        end = start + args.batch_size
        if end <= n:
            batch_x = inputs[start:end]
            batch_y = labels[start:end]
        else:
            batch_x = torch.cat([inputs[start:n], inputs[0:end - n]], dim=0)
            batch_y = torch.cat([labels[start:n], labels[0:end - n]], dim=0)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss, parts = _loss(outputs, batch_y)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip_norm,
            )
        optimizer.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 6) == 0:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                **parts,
            })
            print(f"STEP {step:04d} loss={history[-1]['loss']:.8f}")

    model.eval()
    with torch.no_grad():
        final_outputs = model(inputs[:args.batch_size])
        final_loss, final_parts = _loss(final_outputs, labels[:args.batch_size])
        final_stats = model.collect_wd_stats(inputs[:args.batch_size])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    finite = all(torch.isfinite(out).all().item() for out in final_outputs)
    initial_value = float(initial_loss.detach().cpu())
    final_value = float(final_loss.detach().cpu())
    loss_ratio = final_value / initial_value if initial_value > 0 else None
    activity_delta = _mean_activity(final_stats) - _mean_activity(initial_stats)

    result = {
        "route_id": "haze4k_v3_2_convir_wd_full_model_line_20260707",
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "--short", "HEAD"]),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "sample_names": names,
        "sample_contract": f"{args.samples} train-derived center crops, crop_size={args.crop_size}",
        "scope": args.scope,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "lr_groups": [{"name": g.get("name"), "lr": g["lr"]} for g in groups],
        "partial_load": {
            "loaded_count": len(report["loaded"]),
            "missing_new_count": len(report["missing_new_modules"]),
            "unexpected": report["unexpected"],
            "shape_mismatch": report["shape_mismatch"],
        },
        "initial_loss": initial_value,
        "initial_parts": initial_parts,
        "final_loss": final_value,
        "final_parts": final_parts,
        "loss_ratio": loss_ratio,
        "gate_loss_ratio": args.gate_loss_ratio,
        "history": history,
        "initial_wd_stats": initial_stats,
        "final_wd_stats": final_stats,
        "wd_activity_delta": activity_delta,
        "outputs_finite": finite,
        "locked_test_touched": False,
        "quality_claim": "none; mini-overfit numerical/trainability sanity only",
        "pass": bool(finite and loss_ratio is not None and loss_ratio <= args.gate_loss_ratio and activity_delta > 0.0),
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
