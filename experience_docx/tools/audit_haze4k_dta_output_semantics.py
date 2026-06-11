#!/usr/bin/env python3
"""Audit DTA-v3 no-op equivalence and refine-output residual semantics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader
from models.ConvIR import build_net


def load_state(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def pad_to_factor(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), "reflect")
    return x, h, w


def unpack_batch(data):
    name = data[-1]
    if isinstance(name, str):
        name = [name]
    data = data[:-1]
    input_img, label_img = data[0], data[1]
    depth = data[2] if len(data) >= 3 else None
    airlight = None
    if len(data) >= 4 and torch.is_tensor(data[3]) and data[3].dim() < 3:
        airlight = data[3]
    elif len(data) >= 5:
        airlight = data[4]
    return input_img, label_img, depth, airlight, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--eval_root_split", default="train", choices=["train", "test"])
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--max_images", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_state(args.checkpoint, device)

    official = build_net("base", "Haze4K", "original", arch="official_convir").to(device)
    official.load_state_dict(state)
    official.eval()

    dta = build_net(
        "base",
        "Haze4K",
        "original",
        arch="dta_v3",
        dta_variant="v3",
        dta_gate_limit=0.0,
        dta_gamma_limit=0.0,
        dta_beta_limit=0.0,
        dta_r0_residual_scale=0.0,
        dta_depth_residual_scale=0.0,
        dta_depth_mask_easy_budget=0.0,
        dta_depth_mask_dense_budget=0.0,
        dta_phase="joint",
        dta_ablation="full",
    ).to(device)
    load_result = dta.load_state_dict(state, strict=False)
    bad_missing = [key for key in load_result.missing_keys if not key.startswith("DTA.")]
    if bad_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected DTA no-op load result: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    dta.eval()

    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        root_split=args.eval_root_split,
        return_meta=True,
        split_json=args.split_json,
        split_name=args.split_name,
    )

    noop_diffs = []
    residual_semantic_diffs = []
    output_alone_diffs = []
    names = []
    original_refine = dta.DTA.refine_output
    capture = {}

    def wrapped_refine(final_feat, output, depth, hazy=None, airlight=None):
        capture["output"] = output.detach()
        capture["hazy"] = hazy.detach() if hazy is not None else None
        return original_refine(final_feat, output, depth, hazy=hazy, airlight=airlight)

    dta.DTA.refine_output = wrapped_refine
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, _, depth, airlight, name = unpack_batch(data)
            input_img = input_img.to(device)
            depth = depth.to(device) if depth is not None else None
            airlight = airlight.to(device) if airlight is not None and hasattr(airlight, "to") else airlight
            padded, h, w = pad_to_factor(input_img)
            if depth is not None:
                depth, _, _ = pad_to_factor(depth)

            official_pred = official(padded)[2]
            capture.clear()
            dta_pred = dta(padded, depth, airlight=airlight)[2]
            noop_diffs.append(float((official_pred[:, :, :h, :w] - dta_pred[:, :, :h, :w]).abs().max().cpu()))
            if capture.get("hazy") is not None:
                residual_base = capture["output"] + capture["hazy"]
                residual_semantic_diffs.append(float((residual_base - dta_pred).abs().max().cpu()))
                output_alone_diffs.append(float((capture["output"] - dta_pred).abs().mean().cpu()))
            names.append(name[0])

    result = {
        "official_A0_output_semantics": "final_image",
        "DTA_refine_input_semantics": "residual" if residual_semantic_diffs and max(residual_semantic_diffs) < 1e-6 else "unknown",
        "base_img_formula_checked": bool(residual_semantic_diffs),
        "recommendation": "use out + hazy for DTA-v3 physical base image",
        "sample_count": len(names),
        "sample_names": names,
        "max_abs_noop_diff": max(noop_diffs) if noop_diffs else None,
        "mean_abs_noop_diff": sum(noop_diffs) / len(noop_diffs) if noop_diffs else None,
        "max_abs_residual_plus_hazy_vs_final": max(residual_semantic_diffs) if residual_semantic_diffs else None,
        "mean_abs_output_alone_vs_final": sum(output_alone_diffs) / len(output_alone_diffs) if output_alone_diffs else None,
        "missing_keys_allowed_prefix": "DTA.",
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(load_result.unexpected_keys),
        "locked_test_touched": False,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("DTA_V3_OUTPUT_SEMANTICS_AUDIT_OK")


if __name__ == "__main__":
    main()
