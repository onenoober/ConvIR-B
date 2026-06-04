import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


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


def image_stats(path):
    image = Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR)
    pixels = list(image.getdata())
    count = len(pixels)
    mean_r = sum(pixel[0] for pixel in pixels) / (255.0 * count)
    mean_g = sum(pixel[1] for pixel in pixels) / (255.0 * count)
    mean_b = sum(pixel[2] for pixel in pixels) / (255.0 * count)
    brightness = (0.299 * mean_r) + (0.587 * mean_g) + (0.114 * mean_b)
    saturation = sum((max(pixel) - min(pixel)) for pixel in pixels) / (255.0 * count)

    gray = [0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2] for pixel in pixels]
    width, height = image.size
    grad_sum = 0.0
    grad_count = 0
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width - 1):
            grad_sum += abs(gray[row + x + 1] - gray[row + x])
            grad_sum += abs(gray[next_row + x] - gray[row + x])
            grad_count += 2
    gradient = grad_sum / (255.0 * max(1, grad_count))
    return {
        "brightness": brightness,
        "saturation": saturation,
        "gradient": gradient,
        "bright_low_gradient_proxy": int(brightness >= 0.62 and gradient <= 0.035),
        "low_saturation_bright_proxy": int(brightness >= 0.58 and saturation <= 0.12),
    }


def quantile_bucket(value, cuts):
    bucket = 0
    for cut in cuts:
        if value > cut:
            bucket += 1
    return bucket


def quantile_cuts(values, bucket_count):
    if bucket_count <= 1:
        return []
    ordered = sorted(values)
    cuts = []
    for idx in range(1, bucket_count):
        pos = round((len(ordered) - 1) * idx / bucket_count)
        cuts.append(ordered[pos])
    return cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--brightness_buckets", type=int, default=5)
    parser.add_argument("--gradient_buckets", type=int, default=3)
    args = parser.parse_args()

    train_root = Path(args.data_dir) / "train"
    input_dir = first_existing_dir(train_root, ("IN", "haze", "hazy"))
    image_names = list_images(input_dir)
    if args.val_count <= 0 or args.val_count >= len(image_names):
        raise ValueError(f"--val_count must be in [1, {len(image_names) - 1}]")

    rows = []
    for name in image_names:
        stats = image_stats(os.path.join(input_dir, name))
        rows.append({"name": name, **stats})

    brightness_cuts = quantile_cuts([row["brightness"] for row in rows], args.brightness_buckets)
    gradient_cuts = quantile_cuts([row["gradient"] for row in rows], args.gradient_buckets)
    strata = defaultdict(list)
    for row in rows:
        key = (
            quantile_bucket(row["brightness"], brightness_cuts),
            quantile_bucket(row["gradient"], gradient_cuts),
            row["bright_low_gradient_proxy"],
            row["low_saturation_bright_proxy"],
        )
        strata[key].append(row["name"])

    rng = random.Random(args.seed)
    selected = []
    targets = {}
    for key, names in sorted(strata.items()):
        quota = round(len(names) * args.val_count / len(image_names))
        if quota > 0:
            rng.shuffle(names)
            chosen = sorted(names[:quota])
            selected.extend(chosen)
            targets[str(key)] = {"count": len(names), "val_selected": len(chosen)}

    selected = sorted(set(selected))
    if len(selected) < args.val_count:
        remaining = sorted(set(image_names).difference(selected))
        rng.shuffle(remaining)
        selected = sorted(selected + remaining[: args.val_count - len(selected)])
    elif len(selected) > args.val_count:
        rng.shuffle(selected)
        selected = sorted(selected[: args.val_count])

    train_inner = sorted(set(image_names).difference(selected))
    output = {
        "meta": {
            "data_dir": args.data_dir,
            "source_split": "train",
            "seed": args.seed,
            "total_count": len(image_names),
            "train_inner_count": len(train_inner),
            "val_inner_count": len(selected),
            "val_count_requested": args.val_count,
            "stratification": [
                "input brightness quantile",
                "input gradient quantile",
                "bright-low-gradient proxy",
                "low-saturation-bright proxy",
            ],
            "note": "Use train_inner for training and val_inner for checkpoint/scale selection. Do not select Best on Haze4K test.",
        },
        "splits": {
            "train_inner": train_inner,
            "val_inner": selected,
        },
        "strata": targets,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["meta"], indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
