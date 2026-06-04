import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def list_haze_images(data_dir, split):
    haze_dir = Path(data_dir) / split / "haze"
    if not haze_dir.is_dir():
        raise FileNotFoundError(f"Missing Haze4K haze directory: {haze_dir}")
    return sorted(
        path for path in haze_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def depth_cache_path(output_dir, split, image_path):
    return Path(output_dir) / split / (image_path.name.replace("/", "__") + ".npy")


def predict_depth(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    depth = F.interpolate(
        output.predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = depth.detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_count", type=int, default=0)
    parser.add_argument("--progress_freq", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    total_written = 0
    total_skipped = 0

    for split in splits:
        images = list_haze_images(args.data_dir, split)
        if args.max_count > 0:
            images = images[:args.max_count]
        split_out = Path(args.output_dir) / split
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"split={split} images={len(images)} output={split_out}")
        written = 0
        skipped = 0
        for idx, image_path in enumerate(images, start=1):
            out_path = depth_cache_path(args.output_dir, split, image_path)
            if out_path.is_file() and not args.overwrite:
                skipped += 1
            else:
                depth = predict_depth(model, processor, image_path, device)
                np.save(out_path, depth)
                written += 1
            if args.progress_freq > 0 and (idx % args.progress_freq == 0 or idx == len(images)):
                print(
                    f"progress split={split} processed={idx}/{len(images)} "
                    f"written={written} skipped={skipped}"
                )
        total_written += written
        total_skipped += skipped
        print(f"done split={split} written={written} skipped={skipped}")

    print(f"complete written={total_written} skipped={total_skipped}")


if __name__ == "__main__":
    main()
