import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from data.data_load import DeblurDataset, train_dataloader
from models.ConvIR import build_net as build_convir_net
from models.PFDConvIR import build_pfd_net

IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def list_images(path):
    return sorted(
        item.name
        for item in Path(path).iterdir()
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
    for idx, name in enumerate(inputs[:sample_sizes]):
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


def load_pfd(model, state):
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("PFD_")]
    return {
        "missing": missing,
        "unexpected": unexpected,
        "bad_missing": bad_missing,
        "pass": not unexpected and not bad_missing,
    }


def trainable_summary(model, enabled_prefixes=None):
    enabled_prefixes = tuple(enabled_prefixes or ["PFD_"])
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if name.startswith(enabled_prefixes):
            trainable.append((name, param.numel()))
        else:
            frozen.append((name, param.numel()))
    return {
        "enabled_prefixes": list(enabled_prefixes),
        "trainable_param_count": sum(numel for _, numel in trainable),
        "frozen_param_count": sum(numel for _, numel in frozen),
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable_names_sample": [name for name, _ in trainable[:20]],
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


def finite(value):
    return value is not None and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--mean_abs_threshold", type=float, default=1e-7)
    parser.add_argument("--pfd_decoder_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pfd_decoder_rhfd_scale", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    pair_audit = {
        "train": audit_split(args.data_dir, "train"),
        "test": audit_split(args.data_dir, "test"),
    }
    pair_pass = pair_audit["train"]["pass"] and pair_audit["test"]["pass"]

    state = load_model_state(args.checkpoint, device)
    original = build_convir_net("base", "Haze4K", "original").to(device).eval()
    pfd = build_pfd_net(
        "base",
        "Haze4K",
        pfd_rhfd=False,
        pfd_hscm=False,
        pfd_pffb=False,
        pfd_pffb_high=False,
        pfd_teacher=False,
        pfd_decoder_rhfd=args.pfd_decoder_rhfd,
        pfd_decoder_rhfd_scale=args.pfd_decoder_rhfd_scale,
    ).to(device).eval()

    original_load = load_original(original, state)
    pfd_load = load_pfd(pfd, state)

    torch.manual_seed(args.seed)
    random_input = torch.rand(1, 3, args.height, args.width, device=device)
    random_diffs = output_diffs(original, pfd, random_input)
    random_summary = summarize_diffs(random_diffs)

    loader = train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        data="Haze4K",
        use_transform=True,
    )
    real_batch = next(iter(loader))[0].to(device)
    real_diffs = output_diffs(original, pfd, real_batch)
    real_summary = summarize_diffs(real_diffs)

    param_original = count_parameters(original)
    param_pfd = count_parameters(pfd)
    diff_pass = (
        random_summary["max_abs_diff"] < args.max_abs_threshold
        and random_summary["mean_abs_diff"] < args.mean_abs_threshold
        and real_summary["max_abs_diff"] < args.max_abs_threshold
        and real_summary["mean_abs_diff"] < args.mean_abs_threshold
    )
    result = {
        "stage": "pfd_b1r_decoder_rhfd_preflight" if args.pfd_decoder_rhfd else "pfd_v0_preflight",
        "seed": args.seed,
        "device": str(device),
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "candidate_flags": {
            "pfd_decoder_rhfd": bool(args.pfd_decoder_rhfd),
            "pfd_decoder_rhfd_scale": args.pfd_decoder_rhfd_scale,
        },
        "pair_audit": pair_audit,
        "checkpoint_load": {
            "original": original_load,
            "pfd": pfd_load,
            "pass": not original_load["missing"] and not original_load["unexpected"] and pfd_load["pass"],
        },
        "random_tensor_equivalence": {
            **random_summary,
            "outputs": random_diffs,
            "pass": random_summary["max_abs_diff"] < args.max_abs_threshold
            and random_summary["mean_abs_diff"] < args.mean_abs_threshold,
        },
        "real_batch_equivalence": {
            **real_summary,
            "input_shape": list(real_batch.shape),
            "outputs": real_diffs,
            "pass": real_summary["max_abs_diff"] < args.max_abs_threshold
            and real_summary["mean_abs_diff"] < args.mean_abs_threshold,
        },
        "params": {
            "original": param_original,
            "pfd_candidate": param_pfd,
            "delta": param_pfd - param_original,
            "delta_pct": (param_pfd - param_original) / param_original * 100.0,
        },
        "adapter_only_trainable_summary": trainable_summary(
            pfd,
            ["PFD_DECODER_RHFD"] if args.pfd_decoder_rhfd else ["PFD_"],
        ),
    }
    result["pass"] = (
        pair_pass
        and result["checkpoint_load"]["pass"]
        and diff_pass
        and finite(result["params"]["delta_pct"])
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
