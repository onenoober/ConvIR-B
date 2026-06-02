import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from data.data_load import DeblurDataset, train_dataloader
from models.APDRConvIR import build_apdr_net
from models.ConvIR import build_net as build_convir_net

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(path):
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMG_EXTENSIONS
    )


def label_path(dataset, image_name):
    try:
        return Path(dataset._label_path(image_name))
    except FileNotFoundError:
        return None


def audit_split(data_dir, split, sample_sizes=32):
    image_dir = Path(data_dir) / split
    dataset = DeblurDataset(str(image_dir), "Haze4K", transform=None)
    inputs = list_images(dataset.input_dir)
    labels = list_images(dataset.label_dir)
    input_stems = Counter(Path(name).stem for name in inputs)
    label_stems = Counter(Path(name).stem for name in labels)

    missing = []
    mapped = defaultdict(list)
    for name in inputs:
        target = label_path(dataset, name)
        if target is None:
            missing.append(name)
        else:
            mapped[target.name].append(name)

    duplicate_targets = {
        target: names for target, names in mapped.items() if len(names) > 1
    }
    mapped_targets = set(mapped)
    orphan_labels = [name for name in labels if name not in mapped_targets]

    crop_compatible_failures = []
    for name in inputs[:sample_sizes]:
        input_path = Path(dataset.input_dir) / name
        target = label_path(dataset, name)
        if target is None:
            continue
        from PIL import Image

        with Image.open(input_path) as image, Image.open(target) as label:
            if image.size != label.size:
                crop_compatible_failures.append(
                    {"input": name, "input_size": image.size, "label_size": label.size}
                )
            if min(image.size) < 256:
                crop_compatible_failures.append(
                    {"input": name, "input_size": image.size, "reason": "smaller_than_crop256"}
                )

    duplicate_input_stems = {
        stem: count for stem, count in input_stems.items() if count > 1
    }
    duplicate_label_stems = {
        stem: count for stem, count in label_stems.items() if count > 1
    }
    fatal = bool(missing or duplicate_targets or crop_compatible_failures)
    return {
        "split": split,
        "image_dir": str(image_dir),
        "input_dir": str(dataset.input_dir),
        "label_dir": str(dataset.label_dir),
        "input_count": len(inputs),
        "label_count": len(labels),
        "mapped_target_count": len(mapped_targets),
        "missing_label_count": len(missing),
        "missing_labels": missing[:20],
        "orphan_label_count": len(orphan_labels),
        "orphan_labels": orphan_labels[:20],
        "duplicate_target_mapping_count": len(duplicate_targets),
        "duplicate_target_mappings": dict(list(duplicate_targets.items())[:20]),
        "duplicate_input_stem_count": len(duplicate_input_stems),
        "duplicate_input_stems": dict(list(duplicate_input_stems.items())[:20]),
        "duplicate_label_stem_count": len(duplicate_label_stems),
        "duplicate_label_stems": dict(list(duplicate_label_stems.items())[:20]),
        "crop_compatible_sample_count": min(sample_sizes, len(inputs)),
        "crop_compatible_failure_count": len(crop_compatible_failures),
        "crop_compatible_failures": crop_compatible_failures[:20],
        "pass": not fatal,
    }


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_original(model, state):
    result = model.load_state_dict(state, strict=True)
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def load_apdr(model, state):
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("APDR_")]
    return {
        "missing": missing,
        "unexpected": unexpected,
        "bad_missing": bad_missing,
        "pass": not unexpected and not bad_missing,
    }


def output_diffs(original, candidate, x):
    with torch.no_grad():
        original_out = original(x)
        candidate_out = candidate(x)
    diffs = []
    for idx, (base_tensor, cand_tensor) in enumerate(zip(original_out, candidate_out)):
        delta = (base_tensor - cand_tensor).abs()
        diffs.append(
            {
                "output_index": idx,
                "shape": list(base_tensor.shape),
                "max_abs_diff": delta.max().item(),
                "mean_abs_diff": delta.mean().item(),
            }
        )
    return diffs


def summarize_diffs(diffs):
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in diffs),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in diffs),
    }


def configure_apdr_only(model):
    trainable = []
    frozen = []
    active_prefixes = ("APDR_",)
    if hasattr(model, "active_apdr_prefixes"):
        active_prefixes = model.active_apdr_prefixes()
    for name, param in model.named_parameters():
        if name.startswith("APDR_"):
            param.requires_grad = any(name.startswith(prefix) for prefix in active_prefixes)
        else:
            param.requires_grad = False
        if param.requires_grad:
            trainable.append((name, param))
        else:
            frozen.append((name, param))
    return trainable, frozen


def finite_backward(model, x, target, args):
    if args.apdr_loss_scales == "full_only" and args.apdr_active_scales != "full":
        raise ValueError("--apdr_loss_scales full_only requires --apdr_active_scales full")

    trainable, frozen = configure_apdr_only(model)
    model.eval()
    for name, module in model.named_modules():
        if name.startswith("APDR_"):
            module.train()
    outputs = model(x)
    if args.apdr_loss_scales == "full_only":
        scale_pairs = [(outputs[2], target)]
        apdr_targets = [target, target, target]
    else:
        label2 = F.interpolate(target, scale_factor=0.5, mode="bilinear")
        label4 = F.interpolate(target, scale_factor=0.25, mode="bilinear")
        scale_pairs = [(outputs[0], label4), (outputs[1], label2), (outputs[2], target)]
        apdr_targets = [label4, label2, target]

    loss_content = sum(F.l1_loss(pred, label) for pred, label in scale_pairs)
    loss = loss_content
    apdr_train_reg = {}
    if hasattr(model, "apdr_training_regularization"):
        apdr_train_reg = model.apdr_training_regularization(
            apdr_targets,
            risk_temperature=args.apdr_risk_temperature,
        )
        loss = (
            loss
            + args.apdr_anchor_lambda * apdr_train_reg.get("apdr_anchor", 0.0)
            + args.apdr_gate_lambda * apdr_train_reg.get("apdr_gate", 0.0)
            + args.apdr_residual_lambda * apdr_train_reg.get("apdr_residual", 0.0)
            + args.apdr_gate_supervision_lambda * apdr_train_reg.get("apdr_gate_supervision", 0.0)
        )
    loss.backward()

    nonzero_grad = []
    zero_grad = []
    nonfinite_grad = []
    for name, param in trainable:
        grad = param.grad
        if grad is None:
            zero_grad.append(name)
            continue
        if not torch.isfinite(grad).all():
            nonfinite_grad.append(name)
        elif grad.abs().max().item() > 0:
            nonzero_grad.append(name)
        else:
            zero_grad.append(name)

    frozen_with_grad = [
        name
        for name, param in frozen
        if param.grad is not None and param.grad.abs().max().item() > 0
    ]
    return {
        "loss": loss.item(),
        "loss_content": loss_content.item(),
        "loss_finite": math.isfinite(loss.item()),
        "apdr_training_regularization": {
            key: value.detach().item()
            for key, value in apdr_train_reg.items()
            if torch.is_tensor(value)
        },
        "trainable_param_count": sum(param.numel() for _, param in trainable),
        "frozen_param_count": sum(param.numel() for _, param in frozen),
        "trainable_tensor_count": len(trainable),
        "nonzero_grad_tensor_count": len(nonzero_grad),
        "nonzero_grad_tensors": nonzero_grad[:30],
        "zero_grad_tensor_count": len(zero_grad),
        "zero_grad_tensors_first30": zero_grad[:30],
        "nonfinite_grad_tensors": nonfinite_grad[:30],
        "frozen_with_grad_count": len(frozen_with_grad),
        "frozen_with_grad_first30": frozen_with_grad[:30],
        "pass": (
            math.isfinite(loss.item())
            and len(nonfinite_grad) == 0
            and len(nonzero_grad) > 0
            and len(frozen_with_grad) == 0
        ),
    }


def build_pair(args, device):
    set_seed(args.seed)
    original = build_convir_net("base", "Haze4K", "original").to(device).eval()
    set_seed(args.seed)
    apdr = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode=args.apdr_prior_mode,
        apdr_residual_max=args.apdr_residual_max,
        apdr_gate_max=args.apdr_gate_max,
        apdr_gate_init=args.apdr_gate_init,
        apdr_force_zero_gate=False,
        apdr_active_scales=args.apdr_active_scales,
        apdr_selector_mode=args.apdr_selector_mode,
    ).to(device).eval()
    return original, apdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="apdr_v0_1_preflight")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--mean_abs_threshold", type=float, default=1e-7)
    parser.add_argument("--apdr_prior_mode", default="rgb_haze", choices=["rgb_haze"])
    parser.add_argument("--apdr_residual_max", type=float, default=0.04)
    parser.add_argument("--apdr_gate_max", type=float, default=0.5)
    parser.add_argument("--apdr_gate_init", type=float, default=0.02)
    parser.add_argument("--apdr_selector_mode", default="v0", choices=["v0", "v0_2"])
    parser.add_argument("--apdr_active_scales", default="all", choices=["all", "full"])
    parser.add_argument("--apdr_loss_scales", default="all", choices=["all", "full_only"])
    parser.add_argument("--apdr_anchor_lambda", type=float, default=0.0)
    parser.add_argument("--apdr_gate_supervision_lambda", type=float, default=0.0)
    parser.add_argument("--apdr_gate_lambda", type=float, default=0.0)
    parser.add_argument("--apdr_residual_lambda", type=float, default=0.0)
    parser.add_argument("--apdr_risk_temperature", type=float, default=5.0)
    args = parser.parse_args()
    if args.apdr_loss_scales == "full_only" and args.apdr_active_scales != "full":
        raise ValueError("--apdr_loss_scales full_only requires --apdr_active_scales full")

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    original, apdr = build_pair(args, device)
    checkpoint_load = {"status": "skipped", "pass": True}
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            checkpoint_load = {
                "status": "missing_checkpoint",
                "checkpoint": str(checkpoint),
                "pass": False,
            }
        else:
            state = load_model_state(checkpoint, device)
            original_load = load_original(original, state)
            apdr_load = load_apdr(apdr, state)
            checkpoint_load = {
                "status": "loaded",
                "checkpoint": str(checkpoint),
                "original": original_load,
                "apdr": apdr_load,
                "pass": (
                    not original_load["missing"]
                    and not original_load["unexpected"]
                    and apdr_load["pass"]
                ),
            }

    pair_audit = {"status": "skipped", "pass": True}
    real_batch = None
    real_target = None
    if args.data_dir:
        data_root = Path(args.data_dir)
        if not data_root.is_dir():
            pair_audit = {
                "status": "missing_data_dir",
                "data_dir": str(data_root),
                "pass": False,
            }
        else:
            pair_audit = {
                "status": "checked",
                "data_dir": str(data_root),
                "train": audit_split(data_root, "train"),
                "test": audit_split(data_root, "test"),
            }
            pair_audit["pass"] = pair_audit["train"]["pass"] and pair_audit["test"]["pass"]
            loader = train_dataloader(
                str(data_root),
                batch_size=args.batch_size,
                num_workers=0,
                data="Haze4K",
                use_transform=True,
            )
            real_batch, real_target = next(iter(loader))
            real_batch = real_batch.to(device)
            real_target = real_target.to(device)

    set_seed(args.seed)
    random_input = torch.rand(1, 3, args.height, args.width, device=device)
    random_target = torch.rand(1, 3, args.height, args.width, device=device)
    random_diffs = output_diffs(original, apdr, random_input)
    random_summary = summarize_diffs(random_diffs)

    real_equivalence = {"status": "skipped", "pass": True}
    if real_batch is not None:
        real_diffs = output_diffs(original, apdr, real_batch)
        real_summary = summarize_diffs(real_diffs)
        real_equivalence = {
            "status": "checked",
            **real_summary,
            "input_shape": list(real_batch.shape),
            "outputs": real_diffs,
            "pass": (
                real_summary["max_abs_diff"] < args.max_abs_threshold
                and real_summary["mean_abs_diff"] < args.mean_abs_threshold
            ),
        }

    backward_input = real_batch if real_batch is not None else random_input
    backward_target = real_target if real_target is not None else random_target
    backward = finite_backward(apdr, backward_input, backward_target, args)
    stats_input = real_batch if real_batch is not None else random_input
    stats = apdr.collect_apdr_stats(stats_input)

    param_original = count_parameters(original)
    param_apdr = count_parameters(apdr)
    result = {
        "stage": args.stage,
        "seed": args.seed,
        "device": str(device),
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "apdr_config": {
            "prior_mode": args.apdr_prior_mode,
            "residual_max": args.apdr_residual_max,
            "gate_max": args.apdr_gate_max,
            "gate_init": args.apdr_gate_init,
            "selector_mode": args.apdr_selector_mode,
            "active_scales": args.apdr_active_scales,
            "loss_scales": args.apdr_loss_scales,
            "anchor_lambda": args.apdr_anchor_lambda,
            "gate_supervision_lambda": args.apdr_gate_supervision_lambda,
            "gate_lambda": args.apdr_gate_lambda,
            "residual_lambda": args.apdr_residual_lambda,
            "risk_temperature": args.apdr_risk_temperature,
        },
        "pair_audit": pair_audit,
        "checkpoint_load": checkpoint_load,
        "random_tensor_equivalence": {
            **random_summary,
            "outputs": random_diffs,
            "pass": (
                random_summary["max_abs_diff"] < args.max_abs_threshold
                and random_summary["mean_abs_diff"] < args.mean_abs_threshold
            ),
        },
        "real_batch_equivalence": real_equivalence,
        "finite_backward": backward,
        "apdr_stats_initial": stats,
        "params": {
            "original": param_original,
            "apdr_candidate": param_apdr,
            "delta": param_apdr - param_original,
            "delta_pct": (param_apdr - param_original) / param_original * 100.0,
        },
    }
    result["pass"] = (
        pair_audit["pass"]
        and checkpoint_load["pass"]
        and result["random_tensor_equivalence"]["pass"]
        and real_equivalence["pass"]
        and backward["pass"]
        and math.isfinite(result["params"]["delta_pct"])
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
