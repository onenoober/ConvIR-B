#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def add_repo_imports(its_dir):
    its_dir = os.path.abspath(its_dir)
    if its_dir not in sys.path:
        sys.path.insert(0, its_dir)


def list_images(image_dir):
    return sorted(
        name
        for name in os.listdir(image_dir)
        if name.lower().endswith(IMG_EXTENSIONS)
        and os.path.isfile(os.path.join(image_dir, name))
    )


def first_existing_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"None of {names} found under {root}")


def label_path(label_dir, image_name):
    candidates = [image_name]
    stem, ext = os.path.splitext(image_name)
    if "_" in stem:
        candidates.append(f"{stem.split('_')[0]}{ext}")
        candidates.append(f"{stem.split('_')[0]}.png")
    for candidate in candidates:
        path = os.path.join(label_dir, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No GT match for {image_name}; tried {candidates}")


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def luma_np(image):
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def gradient_mean_np(gray):
    grad_x = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    grad_y = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    return float((grad_x + grad_y) * 0.5)


def image_stats(path):
    image = Image.open(path).convert("RGB").resize((96, 96), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    gray = luma_np(arr)
    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)
    saturation = max_rgb - min_rgb
    dark = min_rgb
    gradient = gradient_mean_np(gray)
    brightness = float(gray.mean())
    saturation_mean = float(saturation.mean())
    dark_mean = float(dark.mean())
    return {
        "brightness": brightness,
        "saturation": saturation_mean,
        "gradient": gradient,
        "dark_channel": dark_mean,
        "bright_low_gradient_proxy": int(brightness >= 0.62 and gradient <= 0.035),
        "low_saturation_bright_proxy": int(brightness >= 0.58 and saturation_mean <= 0.12),
        "sky_bright_proxy": int(brightness >= 0.66 and gradient <= 0.05 and saturation_mean <= 0.20),
    }


def depth_cache_path(cache_dir, split, image_name):
    return os.path.join(cache_dir, split, image_name.replace("/", "__") + ".npy")


def depth_stats(cache_dir, split, image_name):
    if not cache_dir:
        return {"depth_mean": None, "depth_gradient": None}
    path = depth_cache_path(cache_dir, split, image_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing depth cache for {image_name}: {path}")
    depth = np.load(path).astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return {"depth_mean": 0.0, "depth_gradient": 0.0}
    values = depth[finite]
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo > 1e-12:
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    else:
        depth = np.zeros_like(depth)
    return {
        "depth_mean": float(np.mean(depth)),
        "depth_gradient": gradient_mean_np(depth),
    }


def tensor_from_image(path, device):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def pad_to_factor(x, factor=32):
    h, w = x.shape[-2:]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), "reflect")
    return x, h, w


def psnr(pred, target):
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return float((10.0 * torch.log10(torch.tensor(1.0, device=mse.device) / mse)).item())


def compute_a0_psnr(rows, args, input_dir, label_dir):
    add_repo_imports(args.its_dir)
    from models.ConvIR import build_net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_model_state(args.a0_checkpoint, device))
    model.eval()
    with torch.no_grad():
        for idx, row in enumerate(rows):
            image_name = row["name"]
            input_img = tensor_from_image(os.path.join(input_dir, image_name), device)
            label_img = tensor_from_image(label_path(label_dir, image_name), device)
            padded, h, w = pad_to_factor(input_img)
            pred = torch.clamp(model(padded)[2][:, :, :h, :w], 0, 1)
            row["a0_psnr"] = psnr(pred, label_img)
            if (idx + 1) % 100 == 0:
                print(f"a0_psnr {idx + 1}/{len(rows)}", flush=True)


def quantile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def assign_buckets(rows):
    hard_cut = quantile([row["a0_psnr"] for row in rows], 0.25)
    easy_cut = quantile([row["a0_psnr"] for row in rows], 0.75)
    for row in rows:
        if row["a0_psnr"] <= hard_cut:
            row["a0_bucket"] = "hard"
            row["dpga_hard_bucket"] = 0
        elif row["a0_psnr"] >= easy_cut:
            row["a0_bucket"] = "easy"
            row["dpga_hard_bucket"] = 2
        else:
            row["a0_bucket"] = "medium"
            row["dpga_hard_bucket"] = 1
    return hard_cut, easy_cut


def quantile_bucket(value, cuts):
    bucket = 0
    for cut in cuts:
        if value is not None and value > cut:
            bucket += 1
    return bucket


def quantile_cuts(values, bucket_count):
    finite = [value for value in values if value is not None]
    if bucket_count <= 1 or not finite:
        return []
    return [quantile(finite, idx / bucket_count) for idx in range(1, bucket_count)]


def stratified_select(rows, count, seed, key_fn):
    if count <= 0:
        return []
    strata = defaultdict(list)
    for row in rows:
        strata[key_fn(row)].append(row["name"])
    ng = random.Random(seed)
    selected = []
    for key, names in sorted(strata.items()):
        quota = round(len(names) * count / max(1, len(rows)))
        if quota <= 0:
            continue
        names = list(names)
        ng.shuffle(names)
        selected.extend(names[:quota])
    selected = sorted(set(selected))
    if len(selected) < count:
        remaining = sorted({row["name"] for row in rows}.difference(selected))
        ng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    elif len(selected) > count:
        ng.shuffle(selected)
        selected = selected[:count]
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--its_dir", default="Dehazing/ITS")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--depth_cache_dir", default="")
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val_regular_count", type=int, default=300)
    parser.add_argument("--val_hard_count", type=int, default=300)
    parser.add_argument("--hard_pool_quantile", type=float, default=0.35)
    args = parser.parse_args()

    train_root = Path(args.data_dir) / "train"
    input_dir = first_existing_dir(train_root, ("IN", "haze", "hazy"))
    label_dir = first_existing_dir(train_root, ("GT", "gt"))
    names = list_images(input_dir)
    if args.val_regular_count + args.val_hard_count >= len(names):
        raise ValueError("regular + hard validation counts must leave train_inner samples")

    rows = []
    for name in names:
        row = {"name": name}
        row.update(image_stats(os.path.join(input_dir, name)))
        row.update(depth_stats(args.depth_cache_dir, args.depth_split, name))
        rows.append(row)
    compute_a0_psnr(rows, args, str(input_dir), str(label_dir))
    hard_cut, easy_cut = assign_buckets(rows)

    brightness_cuts = quantile_cuts([row["brightness"] for row in rows], 5)
    gradient_cuts = quantile_cuts([row["gradient"] for row in rows], 3)
    depth_cuts = quantile_cuts([row["depth_mean"] for row in rows], 3)
    regular = stratified_select(
        rows,
        args.val_regular_count,
        args.seed,
        lambda row: (
            quantile_bucket(row["brightness"], brightness_cuts),
            quantile_bucket(row["gradient"], gradient_cuts),
            row["bright_low_gradient_proxy"],
            row["low_saturation_bright_proxy"],
        ),
    )
    regular_set = set(regular)
    remaining = [row for row in rows if row["name"] not in regular_set]
    hard_pool_count = max(args.val_hard_count, round(len(rows) * args.hard_pool_quantile))
    hard_pool_names = {
        row["name"]
        for row in sorted(rows, key=lambda item: item["a0_psnr"])[:hard_pool_count]
        if row["name"] not in regular_set
    }
    hard_pool = [row for row in remaining if row["name"] in hard_pool_names]
    if len(hard_pool) < args.val_hard_count:
        hard_pool = remaining
    val_hard = stratified_select(
        hard_pool,
        args.val_hard_count,
        args.seed + 17,
        lambda row: (
            row["a0_bucket"],
            quantile_bucket(row["depth_mean"], depth_cuts),
            row["sky_bright_proxy"],
            quantile_bucket(row["gradient"], gradient_cuts),
        ),
    )
    val_hard_set = set(val_hard)
    train_inner = sorted(set(names).difference(regular_set).difference(val_hard_set))

    payload = {
        "meta": {
            "data_dir": args.data_dir,
            "source_split": "train",
            "seed": args.seed,
            "total_count": len(names),
            "train_inner_count": len(train_inner),
            "val_regular_count": len(regular),
            "val_hard_count": len(val_hard),
            "a0_hard_psnr_cut": hard_cut,
            "a0_easy_psnr_cut": easy_cut,
            "stratification": {
                "val_regular": [
                    "input brightness quantile",
                    "input gradient quantile",
                    "bright-low-gradient proxy",
                    "low-saturation-bright proxy",
                ],
                "val_hard": [
                    "A0 bottom hard pool",
                    "A0 bucket",
                    "depth mean quantile",
                    "sky proxy",
                    "input gradient quantile",
                ],
            },
            "note": "Use val_regular for comparability and val_hard for v1.3 hard-gain gate. Locked Haze4K test remains blocked until both pass.",
        },
        "splits": {
            "train_inner": train_inner,
            "val_regular": regular,
            "val_hard": val_hard,
            "val_inner_regular": regular,
            "val_inner_hard": val_hard,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
