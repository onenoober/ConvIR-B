#!/usr/bin/env python3
"""Render Haze4K contact sheets for A0 vs DTA candidates."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TVF

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.data_load import DeblurDataset
from models.ConvIR import build_net


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_state(path: str, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_candidate(args, device):
    model = build_net(
        "base",
        "Haze4K",
        "original",
        arch=args.candidate_arch,
        dta_variant=args.dta_variant,
        dta_prior_channels=args.dta_prior_channels,
        dta_gate_bias=args.dta_gate_bias,
        dta_gate_limit=args.dta_gate_limit,
        dta_gamma_limit=args.dta_gamma_limit,
        dta_beta_limit=args.dta_beta_limit,
        dta_alpha_init=args.dta_alpha_init,
        dta_depth_mode=args.dta_depth_mode,
        dta_confidence_floor=args.dta_confidence_floor,
        dta_confidence_local_scale=args.dta_confidence_local_scale,
        dta_r0_residual_scale=args.dta_r0_residual_scale,
        dta_depth_residual_scale=args.dta_depth_residual_scale,
        dta_depth_mask_easy_budget=args.dta_depth_mask_easy_budget,
        dta_depth_mask_dense_budget=args.dta_depth_mask_dense_budget,
        dta_depth_mask_density_thresh=args.dta_depth_mask_density_thresh,
        dta_depth_mask_bias=args.dta_depth_mask_bias,
        dta_phys_t_min=args.dta_phys_t_min,
        dta_phase=args.dta_phase,
        dta_ablation=args.dta_ablation,
        dta_safe_mix_enabled=args.dta_safe_mix_enabled,
        dta_safe_mix_delta_clip=args.dta_safe_mix_delta_clip,
        dta_safe_mix_phys_weight=args.dta_safe_mix_phys_weight,
        dta_safe_mix_learned_weight=args.dta_safe_mix_learned_weight,
        dta_safe_mix_gate_limit=args.dta_safe_mix_gate_limit,
        dta_safe_mix_gate_bias=args.dta_safe_mix_gate_bias,
    ).to(device)
    result = model.load_state_dict(load_state(args.candidate_checkpoint, device), strict=False)
    allowed_missing = ("DTA.trans_uncertainty_head.", "DTA.safe_residual_head.", "DTA.safe_gate_head.")
    missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Unexpected DTA-v3 checkpoint load: missing={missing} unexpected={result.unexpected_keys}")
    model.eval()
    return model


def build_a0(args, device):
    model = build_net("base", "Haze4K", "original", arch="official_convir").to(device)
    model.load_state_dict(load_state(args.a0_checkpoint, device))
    model.eval()
    return model


def unpack(data):
    name = data[-1] if isinstance(data[-1], str) else None
    if name is not None:
        data = data[:-1]
    image, label = data[0], data[1]
    depth = data[2] if len(data) >= 3 else None
    airlight = None
    if len(data) >= 4 and torch.is_tensor(data[3]) and data[3].dim() < 3:
        airlight = data[3]
    elif len(data) >= 5:
        airlight = data[4]
    return (
        image.unsqueeze(0),
        label.unsqueeze(0),
        depth.unsqueeze(0) if depth is not None else None,
        airlight.unsqueeze(0) if torch.is_tensor(airlight) and airlight.dim() == 0 else airlight,
        name,
    )


def pad32(x):
    h, w = x.shape[-2:]
    H = ((h + 32) // 32) * 32
    W = ((w + 32) // 32) * 32
    padh = H - h if h % 32 != 0 else 0
    padw = W - w if w % 32 != 0 else 0
    return F.pad(x, (0, padw, 0, padh), "reflect"), h, w


def to_thumb(tensor: torch.Tensor, size: int) -> Image.Image:
    image = TVF.to_pil_image(tensor.squeeze(0).detach().cpu().clamp(0, 1))
    image.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def heatmap(pred: torch.Tensor, label: torch.Tensor, size: int) -> Image.Image:
    err = (pred - label).abs().mean(dim=1, keepdim=True)
    err = (err / err.max().clamp_min(1e-6)).repeat(1, 3, 1, 1)
    return to_thumb(err, size)


def select_names(per_image_csv: Path, count: int) -> tuple[list[str], list[str]]:
    rows = read_rows(per_image_csv)
    rows = [row for row in rows if row.get("name")]
    rows.sort(key=lambda row: float(row.get("delta_psnr", "0")))
    losses = [row["name"] for row in rows[:count]]
    wins = [row["name"] for row in rows[-count:]][::-1]
    return losses, wins


def render_sheet(title: str, names: list[str], dataset, a0, candidate, args, device) -> Image.Image:
    cell = args.thumb_size
    labels = ["hazy", "A0", "candidate", "GT", "|candidate-GT|"]
    sheet = Image.new("RGB", (cell * len(labels), cell * (len(names) + 1) + 28), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=(0, 0, 0))
    for col, label in enumerate(labels):
        draw.text((col * cell + 6, 28), label, fill=(0, 0, 0))
    name_to_idx = {name: idx for idx, name in enumerate(dataset.image_list)}
    with torch.no_grad():
        for row_idx, name in enumerate(names):
            image, label, depth, airlight, _ = unpack(dataset[name_to_idx[name]])
            image = image.to(device)
            label = label.to(device)
            depth = depth.to(device) if depth is not None else None
            airlight = airlight.to(device) if airlight is not None and hasattr(airlight, "to") else airlight
            padded, h, w = pad32(image)
            if depth is not None:
                depth, _, _ = pad32(depth)
            pred_a0 = a0(padded)[2][:, :, :h, :w].clamp(0, 1)
            if args.candidate_arch in ("dta", "dta_v2", "dta_v3"):
                if args.candidate_arch == "dta_v3" and args.dta_airlight_mode == "gt":
                    pred_candidate = candidate(padded, depth, airlight=airlight)[2][:, :, :h, :w].clamp(0, 1)
                else:
                    pred_candidate = candidate(padded, depth)[2][:, :, :h, :w].clamp(0, 1)
            else:
                pred_candidate = candidate(padded)[2][:, :, :h, :w].clamp(0, 1)
            tiles = [
                to_thumb(image, cell),
                to_thumb(pred_a0, cell),
                to_thumb(pred_candidate, cell),
                to_thumb(label, cell),
                heatmap(pred_candidate, label, cell),
            ]
            y = cell * (row_idx + 1) + 28
            for col, tile in enumerate(tiles):
                sheet.paste(tile, (col * cell, y))
            draw.text((6, y + 4), name, fill=(255, 255, 255))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--root_split", default="train", choices=["train", "test"])
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--candidate_arch", default="dta_v3", choices=["official_convir", "convir", "dta", "dta_v2", "dta_v3"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="dta_v3")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--thumb_size", type=int, default=180)
    parser.add_argument("--dta_variant", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--dta_depth_mode", default="invert", choices=["normal", "invert", "zero", "shuffle"])
    parser.add_argument("--dta_airlight_mode", default="fallback", choices=["fallback", "gt"])
    parser.add_argument("--dta_prior_channels", type=int, default=32)
    parser.add_argument("--dta_gate_bias", type=float, default=-5.0)
    parser.add_argument("--dta_gate_limit", type=float, default=0.10)
    parser.add_argument("--dta_gamma_limit", type=float, default=0.16)
    parser.add_argument("--dta_beta_limit", type=float, default=0.08)
    parser.add_argument("--dta_alpha_init", type=float, default=1.0)
    parser.add_argument("--dta_confidence_floor", type=float, default=0.30)
    parser.add_argument("--dta_confidence_local_scale", type=float, default=6.0)
    parser.add_argument("--dta_r0_residual_scale", type=float, default=0.04)
    parser.add_argument("--dta_depth_residual_scale", type=float, default=0.08)
    parser.add_argument("--dta_depth_mask_easy_budget", type=float, default=0.04)
    parser.add_argument("--dta_depth_mask_dense_budget", type=float, default=0.12)
    parser.add_argument("--dta_depth_mask_density_thresh", type=float, default=0.35)
    parser.add_argument("--dta_depth_mask_bias", type=float, default=-4.0)
    parser.add_argument("--dta_phys_t_min", type=float, default=0.10)
    parser.add_argument("--dta_phase", default="joint", choices=["r0", "depth", "joint"])
    parser.add_argument("--dta_ablation", default="full", choices=["full", "r0_only", "film_only_no_output_refine", "trans_head_only_no_rgb_residual", "phys_blend_only"])
    parser.add_argument("--dta_safe_mix_enabled", action="store_true")
    parser.add_argument("--dta_safe_mix_delta_clip", type=float, default=0.08)
    parser.add_argument("--dta_safe_mix_phys_weight", type=float, default=1.0)
    parser.add_argument("--dta_safe_mix_learned_weight", type=float, default=0.0)
    parser.add_argument("--dta_safe_mix_gate_limit", type=float, default=1.0)
    parser.add_argument("--dta_safe_mix_gate_bias", type=float, default=-3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeblurDataset(
        str(Path(args.data_dir) / args.root_split),
        "Haze4K",
        is_test=True,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        return_meta=(args.dta_airlight_mode == "gt"),
        split_json=args.split_json,
        split_name=args.split_name,
    )
    losses, wins = select_names(Path(args.per_image_csv), args.count)
    a0 = build_a0(args, device)
    candidate = build_candidate(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / f"{args.tag}_worst_regressions_contact_sheet.png"
    win_path = output_dir / f"{args.tag}_best_wins_contact_sheet.png"
    render_sheet(f"{args.tag} worst regressions", losses, dataset, a0, candidate, args, device).save(loss_path)
    render_sheet(f"{args.tag} best wins", wins, dataset, a0, candidate, args, device).save(win_path)
    print(f"wrote {loss_path}")
    print(f"wrote {win_path}")
    print("DTA_CONTACT_SHEETS_OK")


if __name__ == "__main__":
    main()
