#!/usr/bin/env python3
"""Extract v1.9 patch-level A0/UDP alpha-oracle evidence.

This cloud-only runtime tool runs frozen A0 and official UDPNet on train-derived
splits, builds a tile-level alpha oracle for

    Y = A0 + alpha * (UDP - A0)

and writes only text/structured evidence. It does not touch locked Haze4K test
data and does not save predictions or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_udpnet_v15_phase0_repro import (  # noqa: E402
    infer_one,
    load_a0_model,
    load_convir_builders,
    load_udpnet_builder,
    load_udpnet_model,
    metric_pair,
    sha256_file,
)
from extract_haze4k_udp_switch_features import (  # noqa: E402
    gradient_mag,
    parse_filename_features,
    tensor_stats,
)


DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
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


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def tile_positions(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), tile_size))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def patch_features(input_patch: torch.Tensor, depth_patch: torch.Tensor) -> dict[str, float]:
    rgb = input_patch.detach().float()
    depth = depth_patch.detach().float()
    lum = luma(rgb)
    rgb_max = rgb.max(dim=1, keepdim=True).values
    rgb_min = rgb.min(dim=1, keepdim=True).values
    saturation = rgb_max - rgb_min
    lum_grad = gradient_mag(lum)
    depth_grad = gradient_mag(depth)
    out: dict[str, float] = {}
    out.update(tensor_stats("tile_luma", lum))
    out.update(tensor_stats("tile_saturation", saturation))
    out.update(tensor_stats("tile_dark_channel", rgb_min))
    out.update(tensor_stats("tile_bright_channel", rgb_max))
    out.update(tensor_stats("tile_depth", depth))
    out.update(
        {
            "tile_luma_grad_mean": float(lum_grad.detach().mean().item()),
            "tile_depth_grad_mean": float(depth_grad.detach().mean().item()),
            "tile_depth_near_ratio_gt_0_66": float((depth > 0.66).float().mean().item()),
            "tile_depth_far_ratio_lt_0_33": float((depth < 0.33).float().mean().item()),
            "tile_sky_proxy_ratio": float(((lum > 0.66) & (lum_grad < 0.05) & (saturation < 0.20)).float().mean().item()),
        }
    )
    return out


def summarize_per_image(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["patch_oracle_delta_psnr"]) for row in rows]
    ssim_deltas = [float(row["patch_oracle_delta_ssim"]) for row in rows]
    hard_sorted = sorted(rows, key=lambda row: float(row["a0_psnr"]))
    n = max(1, len(rows) // 4)
    hard = hard_sorted[:n]
    easy = hard_sorted[-n:]
    worst = [delta for delta in deltas if delta <= -0.20]
    strong_reg = [row for row in easy if float(row["patch_oracle_delta_psnr"]) <= -0.05]
    tail_n = max(1, len(deltas) // 10)
    ordered = sorted(deltas)
    return {
        "count": len(rows),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "hard_bottom25_delta": mean([float(row["patch_oracle_delta_psnr"]) for row in hard]),
        "easy_top25_delta": mean([float(row["patch_oracle_delta_psnr"]) for row in easy]),
        "worst10pct_delta": mean(ordered[:tail_n]),
        "mean_ssim_delta": mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "worst_regression_ratio": len(worst) / max(1, len(rows)),
        "strong_regression_ratio": len(strong_reg) / max(1, len(easy)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    convir_its_dir = Path(args.convir_its_dir)
    test_dataloader, build_convir_net = load_convir_builders(convir_its_dir)
    build_udpnet = load_udpnet_builder(Path(args.udp_repo))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, ckpt_meta = load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    alphas = [float(item) for item in args.alphas]

    tile_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    with torch.no_grad():
        for split in args.splits:
            dataloader = test_dataloader(
                args.data_dir,
                "Haze4K",
                batch_size=1,
                num_workers=args.num_workers,
                depth_cache_dir=args.depth_cache_dir,
                depth_split=args.depth_split,
                split_json=args.split_json,
                split_name=split,
            )
            for idx, data in enumerate(dataloader):
                input_img, label_img, depth, name = data
                image_name = name[0] if isinstance(name, (list, tuple)) else str(name)
                input_img = input_img.to(device)
                label_img = label_img.to(device)
                depth = depth.to(device)
                h, w = input_img.shape[2], input_img.shape[3]
                h_pad = ((h + factor) // factor) * factor
                w_pad = ((w + factor) // factor) * factor
                padh = h_pad - h if h % factor != 0 else 0
                padw = w_pad - w if w % factor != 0 else 0
                rgb_padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
                depth_padded = F.pad(depth, (0, padw, 0, padh), "reflect")
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)

                a0_pred = infer_one(a0_model, rgb_padded, h, w)
                udp_pred = infer_one(udp_model, udp_input, h, w)
                a0_psnr, a0_ssim = metric_pair(a0_pred, label_img, (h_pad, w_pad))

                alpha_mses = {}
                alpha_preds = {}
                for alpha in alphas:
                    if alpha == 0.0:
                        pred = a0_pred
                    elif alpha == 1.0:
                        pred = udp_pred
                    else:
                        pred = torch.clamp(a0_pred + alpha * (udp_pred - a0_pred), 0.0, 1.0)
                    alpha_preds[alpha] = pred
                    alpha_mses[alpha] = torch.mean((pred - label_img) ** 2, dim=1, keepdim=True)

                patch_pred = torch.zeros_like(a0_pred)
                alpha_map = torch.zeros((1, 1, h, w), device=device, dtype=a0_pred.dtype)
                tile_count = 0
                positive_tiles = 0
                for top in tile_positions(h, args.tile_size):
                    for left in tile_positions(w, args.tile_size):
                        bottom = min(h, top + args.tile_size)
                        right = min(w, left + args.tile_size)
                        a0_mse_tile = float(alpha_mses[0.0][:, :, top:bottom, left:right].mean().item())
                        best_alpha = 0.0
                        best_mse = a0_mse_tile
                        for alpha in alphas:
                            mse = float(alpha_mses[alpha][:, :, top:bottom, left:right].mean().item())
                            if mse < best_mse:
                                best_mse = mse
                                best_alpha = alpha
                        patch_pred[:, :, top:bottom, left:right] = alpha_preds[best_alpha][:, :, top:bottom, left:right]
                        alpha_map[:, :, top:bottom, left:right] = best_alpha
                        positive_tiles += int(best_alpha > 0)
                        tile_count += 1
                        tile_row = {
                            "split": split,
                            "name": image_name,
                            "tile_index": tile_count - 1,
                            "tile_top": top,
                            "tile_left": left,
                            "tile_bottom": bottom,
                            "tile_right": right,
                            "tile_height": bottom - top,
                            "tile_width": right - left,
                            "a0_tile_mse": a0_mse_tile,
                            "best_alpha": best_alpha,
                            "best_tile_mse": best_mse,
                            "best_tile_psnr": psnr_from_mse(best_mse),
                            "a0_tile_psnr": psnr_from_mse(a0_mse_tile),
                            "best_alpha_delta_psnr": psnr_from_mse(best_mse) - psnr_from_mse(a0_mse_tile),
                            "teacher_positive": int(best_alpha > 0 and best_mse < a0_mse_tile),
                        }
                        for alpha in alphas:
                            tag = alpha_tag(alpha)
                            mse = float(alpha_mses[alpha][:, :, top:bottom, left:right].mean().item())
                            tile_row[f"alpha_{tag}_tile_mse"] = mse
                            tile_row[f"alpha_{tag}_tile_delta_psnr"] = psnr_from_mse(mse) - psnr_from_mse(a0_mse_tile)
                        tile_row.update(parse_filename_features(image_name))
                        tile_row.update(
                            patch_features(
                                input_img[:, :, top:bottom, left:right],
                                depth[:, :, top:bottom, left:right],
                            )
                        )
                        tile_rows.append(tile_row)

                patch_pred = torch.clamp(patch_pred, 0.0, 1.0)
                patch_psnr, patch_ssim = metric_pair(patch_pred, label_img, (h_pad, w_pad))
                image_row = {
                    "split": split,
                    "name": image_name,
                    "a0_psnr": a0_psnr,
                    "a0_ssim": a0_ssim,
                    "patch_oracle_psnr": patch_psnr,
                    "patch_oracle_ssim": patch_ssim,
                    "patch_oracle_delta_psnr": patch_psnr - a0_psnr,
                    "patch_oracle_delta_ssim": patch_ssim - a0_ssim,
                    "tile_count": tile_count,
                    "positive_tile_count": positive_tiles,
                    "positive_tile_ratio": positive_tiles / max(1, tile_count),
                    "alpha_map_mean": float(alpha_map.mean().item()),
                    "alpha_map_std": float(alpha_map.std(unbiased=False).item()),
                }
                for alpha in alphas:
                    tag = alpha_tag(alpha)
                    alpha_psnr, alpha_ssim = metric_pair(alpha_preds[alpha], label_img, (h_pad, w_pad))
                    image_row[f"image_alpha_{tag}_psnr"] = alpha_psnr
                    image_row[f"image_alpha_{tag}_ssim"] = alpha_ssim
                    image_row[f"image_alpha_{tag}_delta_psnr"] = alpha_psnr - a0_psnr
                    image_row[f"image_alpha_{tag}_delta_ssim"] = alpha_ssim - a0_ssim
                image_rows.append(image_row)

                if (idx + 1) % args.print_freq == 0:
                    split_rows = [row for row in image_rows if row["split"] == split]
                    mean_delta = statistics.mean(float(row["patch_oracle_delta_psnr"]) for row in split_rows)
                    print(f"{split} {idx + 1}/{len(dataloader)} patch_oracle_mean_delta={mean_delta:.4f}", flush=True)
                if args.max_images and idx + 1 >= args.max_images:
                    break

    split_summaries = {
        split: summarize_per_image([row for row in image_rows if row["split"] == split])
        for split in sorted({str(row["split"]) for row in image_rows})
    }
    summary = {
        "route": "ConvIR-Dehaze-v1.9-ConditionalTeacherGuided",
        "stage": "patch alpha oracle extraction",
        "locked_test_touched": False,
        "status": "PATCH_ALPHA_ORACLE_COMPLETE" if not args.max_images else "PATCH_ALPHA_ORACLE_PARTIAL_PREFLIGHT",
        "count_images": len(image_rows),
        "count_tiles": len(tile_rows),
        "alphas": alphas,
        "tile_size": args.tile_size,
        "data_dir": args.data_dir,
        "depth_cache_dir": args.depth_cache_dir,
        "split_json": args.split_json,
        "splits": args.splits,
        "a0_checkpoint_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint_sha256": sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "combined_summary": summarize_per_image(image_rows),
        "split_summaries": split_summaries,
    }
    return tile_rows, image_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convir_its_dir", required=True)
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--splits", nargs="+", default=["train_inner", "val_regular", "val_hard"])
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tile_rows, image_rows, summary = evaluate(args)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "v19_patch_alpha_oracle_tiles.csv", tile_rows)
    write_csv(output_dir / "v19_patch_alpha_oracle_per_image.csv", image_rows)
    write_json(output_dir / "v19_patch_alpha_oracle_summary.json", summary)
    readme = """# Haze4K v1.9 Patch Alpha Oracle

Primary files:

- `v19_patch_alpha_oracle_tiles.csv`: tile-level alpha oracle labels and
  deployable local features.
- `v19_patch_alpha_oracle_per_image.csv`: image-level metrics after applying
  the tile oracle.
- `v19_patch_alpha_oracle_summary.json`: split summaries and checkpoint hashes.

Locked Haze4K test touched: no.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("V19_PATCH_ALPHA_ORACLE_OK", flush=True)


if __name__ == "__main__":
    main()
