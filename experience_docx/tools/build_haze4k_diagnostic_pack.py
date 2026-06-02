import argparse
import csv
import json
import math
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as TVF

REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from data import test_dataloader
from models.ConvIR import build_net
from models.PFDConvIR import build_pfd_net


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _float(raw, default=None):
    if raw in (None, ""):
        return default
    return float(raw)


def read_per_image_rows(csv_path, original_name, candidate_name):
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        original_psnr_key = "original_psnr"
        if original_psnr_key not in fieldnames:
            original_psnr_key = f"{original_name}_psnr"
        candidate_psnr_key = f"{candidate_name}_psnr"
        if candidate_psnr_key not in fieldnames:
            candidates = sorted(
                key
                for key in fieldnames
                if key.endswith("_psnr")
                and key not in {"original_psnr", f"{original_name}_psnr"}
            )
            if not candidates:
                raise ValueError(f"Could not infer candidate PSNR column from {csv_path}")
            candidate_psnr_key = candidates[0]

        original_ssim_key = "original_ssim"
        if original_ssim_key not in fieldnames:
            original_ssim_key = f"{original_name}_ssim"
        candidate_ssim_key = f"{candidate_name}_ssim"
        if candidate_ssim_key not in fieldnames:
            candidate_ssim_key = next(
                (
                    key
                    for key in sorted(fieldnames)
                    if key.endswith("_ssim")
                    and key not in {"original_ssim", f"{original_name}_ssim"}
                ),
                None,
            )

        for raw in reader:
            original_psnr = float(raw[original_psnr_key])
            candidate_psnr = float(raw[candidate_psnr_key])
            rows.append(
                {
                    "name": raw["name"],
                    "original_psnr": original_psnr,
                    "candidate_psnr": candidate_psnr,
                    "delta_psnr": _float(
                        raw.get("delta_psnr"),
                        candidate_psnr - original_psnr,
                    ),
                    "original_ssim": _float(raw.get(original_ssim_key)),
                    "candidate_ssim": _float(
                        raw.get(candidate_ssim_key) if candidate_ssim_key else None
                    ),
                    "delta_ssim": _float(raw.get("delta_ssim")),
                }
            )
    if not rows:
        raise ValueError(f"No per-image rows found in {csv_path}")
    return rows


def assign_difficulty_buckets(rows):
    ordered = sorted(rows, key=lambda row: row["original_psnr"])
    count = len(ordered)
    hard_end = count // 4
    easy_start = 3 * count // 4
    for idx, row in enumerate(ordered):
        if idx < hard_end:
            row["bucket"] = "hard_bottom_25pct"
        elif idx >= easy_start:
            row["bucket"] = "easy_top_25pct"
        else:
            row["bucket"] = "medium_middle_50pct"


def _add_category(selected, category, candidates, limit):
    for row in candidates[:limit]:
        item = selected.get(row["name"])
        if item is None:
            item = dict(row)
            item["categories"] = []
            selected[row["name"]] = item
        if category not in item["categories"]:
            item["categories"].append(category)


def select_rows(rows, category_limit, max_samples):
    assign_difficulty_buckets(rows)
    by_delta = sorted(rows, key=lambda row: row["delta_psnr"])
    selected = OrderedDict()

    _add_category(selected, "worst_regression_top10", by_delta, category_limit)
    _add_category(
        selected,
        "catastrophic_worst",
        [row for row in by_delta if row["delta_psnr"] <= -10.0],
        category_limit,
    )
    _add_category(
        selected,
        "best_gain_top10",
        sorted(rows, key=lambda row: row["delta_psnr"], reverse=True),
        category_limit,
    )
    _add_category(
        selected,
        "hard_gain",
        sorted(
            (
                row
                for row in rows
                if row["bucket"] == "hard_bottom_25pct" and row["delta_psnr"] > 0
            ),
            key=lambda row: row["delta_psnr"],
            reverse=True,
        ),
        category_limit,
    )
    _add_category(
        selected,
        "hard_regression",
        [
            row
            for row in by_delta
            if row["bucket"] == "hard_bottom_25pct" and row["delta_psnr"] <= -0.05
        ],
        category_limit,
    )
    _add_category(
        selected,
        "easy_preserved",
        sorted(
            (
                row
                for row in rows
                if row["bucket"] == "easy_top_25pct"
                and abs(row["delta_psnr"]) <= 0.05
            ),
            key=lambda row: abs(row["delta_psnr"]),
        ),
        category_limit,
    )
    _add_category(
        selected,
        "easy_regression",
        [
            row
            for row in by_delta
            if row["bucket"] == "easy_top_25pct" and row["delta_psnr"] <= -0.05
        ],
        category_limit,
    )
    medium_regressions = [
        row
        for row in rows
        if row["bucket"] == "medium_middle_50pct"
        and -1.0 <= row["delta_psnr"] <= -0.05
    ]
    if not medium_regressions:
        medium_regressions = [
            row
            for row in rows
            if row["bucket"] == "medium_middle_50pct" and row["delta_psnr"] <= -0.05
        ]
    _add_category(
        selected,
        "medium_ordinary_regression",
        sorted(medium_regressions, key=lambda row: abs(row["delta_psnr"] + 0.2)),
        category_limit,
    )

    chosen = list(selected.values())
    if max_samples > 0:
        chosen = chosen[:max_samples]
    for rank, row in enumerate(chosen, start=1):
        row["rank"] = rank
    return chosen


def build_eval_model(arch, mode, args, prefix):
    if arch == "convir":
        return build_net("base", "Haze4K", mode)
    return build_pfd_net(
        "base",
        "Haze4K",
        pfd_rhfd=getattr(args, f"{prefix}_pfd_rhfd"),
        pfd_hscm=getattr(args, f"{prefix}_pfd_hscm"),
        pfd_pffb=getattr(args, f"{prefix}_pfd_pffb"),
        pfd_pffb_high=getattr(args, f"{prefix}_pfd_pffb_high"),
        pfd_teacher=getattr(args, f"{prefix}_pfd_teacher"),
        pfd_safe_rhfd=getattr(args, f"{prefix}_pfd_safe_rhfd"),
        pfd_safe_rhfd_gate_max=getattr(args, f"{prefix}_pfd_safe_rhfd_gate_max"),
        pfd_safe_rhfd_norm_cap=getattr(args, f"{prefix}_pfd_safe_rhfd_norm_cap"),
        pfd_safe_rhfd_lowpass_ratio=getattr(args, f"{prefix}_pfd_safe_rhfd_lowpass_ratio"),
    )


def load_model(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    return F.pad(input_img, (0, padw, 0, padh), "reflect"), h, w


def predict_multiscale(model, padded, h, w, label_img):
    outputs = model(padded)
    label_half = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_quarter = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    qh, qw = label_quarter.shape[2], label_quarter.shape[3]
    hh, hw = label_half.shape[2], label_half.shape[3]
    return {
        "quarter": outputs[0][:, :, :qh, :qw],
        "half": outputs[1][:, :, :hh, :hw],
        "full": outputs[2][:, :, :h, :w],
    }


def tensor_to_pil(tensor):
    image = tensor.detach().squeeze(0).cpu().clamp(0, 1)
    return TVF.to_pil_image(image)


def save_tensor_image(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def signed_residual_vis(tensor, gain):
    return 0.5 + tensor * gain


def diff_vis(tensor, gain):
    return tensor.abs() * gain


def saturation_mean(tensor):
    clipped = tensor.clamp(0, 1)
    max_rgb = clipped.max(dim=1, keepdim=True)[0]
    min_rgb = clipped.min(dim=1, keepdim=True)[0]
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    return saturation.mean().item()


def luma(tensor):
    return (
        0.299 * tensor[:, 0:1]
        + 0.587 * tensor[:, 1:2]
        + 0.114 * tensor[:, 2:3]
    )


def output_safety_stats(name, row, model_label, pred, target):
    pred_detached = pred.detach()
    target_detached = target.detach()
    pred_rgb = pred_detached.mean(dim=(0, 2, 3))
    target_rgb = target_detached.mean(dim=(0, 2, 3))
    rgb_shift = pred_rgb - target_rgb
    target_sat = saturation_mean(target_detached)
    return {
        "name": name,
        "categories": "|".join(row["categories"]),
        "bucket": row["bucket"],
        "model": model_label,
        "pred_min": pred_detached.min().item(),
        "pred_max": pred_detached.max().item(),
        "pred_mean": pred_detached.mean().item(),
        "pred_std": pred_detached.std(unbiased=False).item(),
        "ratio_pred_lt_0": (pred_detached < 0).float().mean().item(),
        "ratio_pred_gt_1": (pred_detached > 1).float().mean().item(),
        "rgb_mean_shift_r": rgb_shift[0].item(),
        "rgb_mean_shift_g": rgb_shift[1].item(),
        "rgb_mean_shift_b": rgb_shift[2].item(),
        "rgb_mean_shift_abs_mean": rgb_shift.abs().mean().item(),
        "luma_mean_shift": (luma(pred_detached) - luma(target_detached)).mean().item(),
        "saturation_ratio": saturation_mean(pred_detached) / max(target_sat, 1e-12),
        "original_psnr": row["original_psnr"],
        "candidate_psnr": row["candidate_psnr"],
        "delta_psnr": row["delta_psnr"],
    }


def flatten_pfd_stats(stats):
    flat = {}
    for module_name, module_stats in stats.items():
        if module_name == "flags":
            continue
        prefix = module_name.lower()
        for key, value in module_stats.items():
            if isinstance(value, (int, float)):
                flat[f"{prefix}_{key}"] = value
    return flat


def aggregate_numeric(rows):
    grouped = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                grouped[key].append(value)
    result = {"count": len(rows)}
    for key, values in sorted(grouped.items()):
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_max"] = max(values)
    return result


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_tile(image, caption, thumb_size, caption_height):
    image = ImageOps.contain(image.convert("RGB"), (thumb_size, thumb_size))
    tile = Image.new("RGB", (thumb_size, thumb_size + caption_height), "white")
    tile.paste(image, ((thumb_size - image.width) // 2, caption_height))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 3), caption[:28], fill=(0, 0, 0))
    return tile


def build_panel(path, panel_rows, thumb_size):
    if not panel_rows:
        return
    captions = [
        "hazy input",
        "GT clear",
        "original full",
        "candidate full",
        "|candidate-original|",
        "|original-GT|",
        "|candidate-GT|",
        "input-original",
        "input-candidate",
    ]
    caption_height = 20
    width = len(captions) * thumb_size
    height = len(panel_rows) * (thumb_size + caption_height)
    panel = Image.new("RGB", (width, height), "white")
    for row_idx, panel_item in enumerate(panel_rows):
        row = panel_item["row"]
        images = panel_item["images"]
        row_captions = list(captions)
        row_captions[0] = f"hazy #{row['rank']} dPSNR {row['delta_psnr']:+.2f}"
        for col_idx, (caption, image) in enumerate(zip(row_captions, images)):
            tile = make_tile(image, caption, thumb_size, caption_height)
            panel.paste(tile, (col_idx * thumb_size, row_idx * (thumb_size + caption_height)))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def write_visual_notes(path, selected):
    lines = [
        "# Fixed Visual Notes",
        "",
        "| Rank | Name | Bucket | Categories | Delta PSNR | Notes |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            "| {rank} | `{name}` | {bucket} | {categories} | {delta:.4f} |  |".format(
                rank=row["rank"],
                name=row["name"],
                bucket=row["bucket"],
                categories=", ".join(row["categories"]),
                delta=row["delta_psnr"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--bucket_json", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--original_arch", choices=["convir", "pfd"], default="convir")
    parser.add_argument("--original_mode", default="original")
    parser.add_argument("--original_name", default="original")
    parser.add_argument("--original_pfd_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_hscm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_pffb", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_pffb_high", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_teacher", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_safe_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--original_pfd_safe_rhfd_gate_max", type=float, default=1.0)
    parser.add_argument("--original_pfd_safe_rhfd_norm_cap", type=float, default=0.0035)
    parser.add_argument("--original_pfd_safe_rhfd_lowpass_ratio", type=float, default=0.20)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--candidate_arch", choices=["convir", "pfd"], default="convir")
    parser.add_argument("--candidate_mode", default="original")
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--candidate_pfd_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_hscm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_pffb", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_pffb_high", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_teacher", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_safe_rhfd", type=int, default=0, choices=[0, 1])
    parser.add_argument("--candidate_pfd_safe_rhfd_gate_max", type=float, default=1.0)
    parser.add_argument("--candidate_pfd_safe_rhfd_norm_cap", type=float, default=0.0035)
    parser.add_argument("--candidate_pfd_safe_rhfd_lowpass_ratio", type=float, default=0.20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--category_limit", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=70)
    parser.add_argument("--panel_limit", type=int, default=20)
    parser.add_argument("--thumb_size", type=int, default=160)
    parser.add_argument("--diff_gain", type=float, default=4.0)
    parser.add_argument("--residual_gain", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_summary = json.loads(Path(args.bucket_json).read_text(encoding="utf-8"))
    rows = read_per_image_rows(args.per_image_csv, args.original_name, args.candidate_name)
    selected = select_rows(rows, args.category_limit, args.max_samples)
    selected_by_name = {row["name"]: row for row in selected}

    original_model = load_model(
        build_eval_model(args.original_arch, args.original_mode, args, "original").to(device),
        args.original_checkpoint,
        device,
    )
    candidate_model = load_model(
        build_eval_model(args.candidate_arch, args.candidate_mode, args, "candidate").to(device),
        args.candidate_checkpoint,
        device,
    )

    manifest_rows = []
    safety_rows = []
    pfd_rows = []
    panel_rows = []
    missing = []
    loader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)

    with torch.no_grad():
        for input_img, label_img, name in loader:
            sample_name = name[0]
            if sample_name not in selected_by_name:
                continue
            row = selected_by_name[sample_name]
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)

            original_outputs = predict_multiscale(original_model, padded, h, w, label_img)
            candidate_outputs = predict_multiscale(candidate_model, padded, h, w, label_img)
            original_full = original_outputs["full"]
            candidate_full = candidate_outputs["full"]

            safe_stem = f"{row['rank']:02d}_{Path(sample_name).stem}"
            sample_dir = output_dir / "samples" / safe_stem
            save_tensor_image(sample_dir / "hazy_input.png", input_img)
            save_tensor_image(sample_dir / "gt_clear.png", label_img)
            for prefix, outputs in (
                (args.original_name, original_outputs),
                (args.candidate_name, candidate_outputs),
            ):
                save_tensor_image(sample_dir / f"{prefix}_out_1_4.png", outputs["quarter"])
                save_tensor_image(sample_dir / f"{prefix}_out_1_2.png", outputs["half"])
                save_tensor_image(sample_dir / f"{prefix}_out_full.png", outputs["full"])
            save_tensor_image(
                sample_dir / f"diff_{args.candidate_name}_minus_{args.original_name}.png",
                diff_vis(candidate_full - original_full, args.diff_gain),
            )
            save_tensor_image(
                sample_dir / f"abs_error_{args.original_name}.png",
                diff_vis(original_full - label_img, args.diff_gain),
            )
            save_tensor_image(
                sample_dir / f"abs_error_{args.candidate_name}.png",
                diff_vis(candidate_full - label_img, args.diff_gain),
            )
            save_tensor_image(
                sample_dir / f"residual_input_minus_{args.original_name}.png",
                signed_residual_vis(input_img - original_full, args.residual_gain),
            )
            save_tensor_image(
                sample_dir / f"residual_input_minus_{args.candidate_name}.png",
                signed_residual_vis(input_img - candidate_full, args.residual_gain),
            )

            if len(panel_rows) < args.panel_limit:
                panel_rows.append(
                    {
                        "row": row,
                        "images": [
                            tensor_to_pil(input_img),
                            tensor_to_pil(label_img),
                            tensor_to_pil(original_full),
                            tensor_to_pil(candidate_full),
                            tensor_to_pil(diff_vis(candidate_full - original_full, args.diff_gain)),
                            tensor_to_pil(diff_vis(original_full - label_img, args.diff_gain)),
                            tensor_to_pil(diff_vis(candidate_full - label_img, args.diff_gain)),
                            tensor_to_pil(
                                signed_residual_vis(input_img - original_full, args.residual_gain)
                            ),
                            tensor_to_pil(
                                signed_residual_vis(input_img - candidate_full, args.residual_gain)
                            ),
                        ],
                    }
                )

            for model_label, model, full_output in (
                (args.original_name, original_model, original_full),
                (args.candidate_name, candidate_model, candidate_full),
            ):
                safety_rows.append(
                    output_safety_stats(sample_name, row, model_label, full_output, label_img)
                )
                if hasattr(model, "collect_pfd_stats"):
                    flat_stats = flatten_pfd_stats(model.collect_pfd_stats(padded))
                    if flat_stats:
                        flat_stats.update(
                            {
                                "name": sample_name,
                                "model": model_label,
                                "bucket": row["bucket"],
                                "categories": "|".join(row["categories"]),
                                "original_psnr": row["original_psnr"],
                                "candidate_psnr": row["candidate_psnr"],
                                "delta_psnr": row["delta_psnr"],
                            }
                        )
                        pfd_rows.append(flat_stats)

            manifest_rows.append(
                {
                    "rank": row["rank"],
                    "name": sample_name,
                    "bucket": row["bucket"],
                    "categories": "|".join(row["categories"]),
                    "original_psnr": row["original_psnr"],
                    "candidate_psnr": row["candidate_psnr"],
                    "delta_psnr": row["delta_psnr"],
                    "panel_included": int(row["rank"] <= args.panel_limit),
                    "sample_dir": str(sample_dir.relative_to(output_dir)),
                }
            )

    seen = {row["name"] for row in manifest_rows}
    missing = [name for name in selected_by_name if name not in seen]

    write_csv(output_dir / "sample_manifest.csv", manifest_rows)
    write_csv(output_dir / "output_safety_stats.csv", safety_rows)
    write_csv(output_dir / "pfd_branch_stats.csv", pfd_rows)
    build_panel(output_dir / "visual_panel_20.png", panel_rows, args.thumb_size)
    write_visual_notes(output_dir / "visual_notes_template.md", selected)

    pfd_summary = {}
    for model_label in sorted({row["model"] for row in pfd_rows}):
        model_rows = [row for row in pfd_rows if row["model"] == model_label]
        pfd_summary[model_label] = {}
        categories = sorted(
            {
                category
                for row in model_rows
                for category in row["categories"].split("|")
                if category
            }
        )
        for category in categories:
            category_rows = [
                row
                for row in model_rows
                if category in row["categories"].split("|")
            ]
            pfd_summary[model_label][category] = aggregate_numeric(category_rows)

    summary = {
        "data_dir": args.data_dir,
        "per_image_csv": args.per_image_csv,
        "bucket_json": args.bucket_json,
        "bucket_json_candidate": bucket_summary.get("candidate_name"),
        "original": {
            "name": args.original_name,
            "arch": args.original_arch,
            "checkpoint": args.original_checkpoint,
        },
        "candidate": {
            "name": args.candidate_name,
            "arch": args.candidate_arch,
            "checkpoint": args.candidate_checkpoint,
            "pfd_flags": {
                "rhfd": args.candidate_pfd_rhfd,
                "hscm": args.candidate_pfd_hscm,
                "pffb": args.candidate_pfd_pffb,
                "pffb_high": args.candidate_pfd_pffb_high,
                "teacher": args.candidate_pfd_teacher,
                "safe_rhfd": args.candidate_pfd_safe_rhfd,
                "safe_rhfd_gate_max": args.candidate_pfd_safe_rhfd_gate_max,
                "safe_rhfd_norm_cap": args.candidate_pfd_safe_rhfd_norm_cap,
                "safe_rhfd_lowpass_ratio": args.candidate_pfd_safe_rhfd_lowpass_ratio,
            },
        },
        "selected_count": len(manifest_rows),
        "missing_selected_names": missing,
        "panel_limit": args.panel_limit,
        "category_limit": args.category_limit,
        "max_samples": args.max_samples,
        "files": {
            "visual_panel": "visual_panel_20.png",
            "sample_manifest": "sample_manifest.csv",
            "output_safety_stats": "output_safety_stats.csv",
            "pfd_branch_stats": "pfd_branch_stats.csv",
            "pfd_branch_stats_by_category": "pfd_branch_stats_by_category.json",
            "visual_notes_template": "visual_notes_template.md",
        },
    }
    (output_dir / "pfd_branch_stats_by_category.json").write_text(
        json.dumps(pfd_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
