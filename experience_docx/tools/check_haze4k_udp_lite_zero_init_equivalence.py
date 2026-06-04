#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def add_repo_imports(its_dir):
    its_dir = os.path.abspath(its_dir)
    if its_dir not in sys.path:
        sys.path.insert(0, its_dir)


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_convir_into_udp_lite(convir, udp_lite, checkpoint, device):
    if checkpoint:
        state = load_model_state(checkpoint, device)
        convir.load_state_dict(state)
    result = udp_lite.load_state_dict(convir.state_dict(), strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("DPGA_")]
    if unexpected or bad_missing:
        raise RuntimeError(f"Bad UDP-Lite init load: missing={missing}, unexpected={unexpected}")
    return missing


def make_synthetic_input(args, device):
    torch.manual_seed(args.seed)
    input_img = torch.rand(1, 3, args.size, args.size, device=device)
    label_img = torch.rand(1, 3, args.size, args.size, device=device)
    depth = torch.rand(1, 1, args.size, args.size, device=device)
    return input_img, label_img, depth, "synthetic"


def make_data_input(args, device):
    from data import test_dataloader

    loader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        split_json=args.split_json,
        split_name=args.split_name,
    )
    input_img, label_img, depth, name = next(iter(loader))
    return input_img.to(device), label_img.to(device), depth.to(device), name[0]


def max_output_diffs(convir, udp_lite, input_img, depth):
    convir.eval()
    udp_lite.eval()
    with torch.no_grad():
        ref = convir(input_img)
        out = udp_lite(input_img, depth)
    rows = []
    for idx, (a, b) in enumerate(zip(ref, out)):
        diff = (a - b).abs()
        rows.append(
            {
                "scale_index": idx,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
            }
        )
    return rows


def grad_probe(udp_lite, input_img, label_img, depth):
    udp_lite.eval()
    for name, module in udp_lite.named_modules():
        if name.startswith("DPGA_"):
            module.train()
    udp_lite.zero_grad(set_to_none=True)
    pred = udp_lite(input_img, depth)[-1]
    loss = F.l1_loss(pred, label_img)
    loss.backward()
    rows = []
    interesting_suffixes = (
        "channel.mlp.2.weight",
        "cross.project.weight",
        "spatial_gate.2.weight",
        "channel_gate.2.weight",
    )
    for name, param in udp_lite.named_parameters():
        if not name.startswith("DPGA_") or not name.endswith(interesting_suffixes):
            continue
        rows.append(
            {
                "name": name,
                "grad_abs_sum": 0.0 if param.grad is None else float(param.grad.abs().sum().item()),
                "grad_abs_max": 0.0 if param.grad is None else float(param.grad.abs().max().item()),
            }
        )
    return float(loss.item()), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--its_dir", default="Dehazing/ITS")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--depth_cache_dir", default="")
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--version", default="base", choices=["small", "base", "large"])
    parser.add_argument("--data", default="Haze4K")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--dpga_prior_embed_channels", type=int, default=16)
    parser.add_argument("--dpga_adapter_reduction", type=int, default=2)
    parser.add_argument("--dpga_adapter_residual_scale", type=float, default=0.1)
    parser.add_argument("--dpga_adapter_scale_init", type=float, default=0.0)
    parser.add_argument("--dpga_adapter_bootstrap_scale", type=float, default=0.01)
    parser.add_argument("--dpga_udp_components", default="all")
    parser.add_argument("--dpga_udp_window_size", type=int, default=8)
    parser.add_argument("--dpga_udp_num_heads", type=int, default=4)
    parser.add_argument("--dpga_active_adapters", default="all")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.data_dir and not args.depth_cache_dir:
        raise ValueError("--depth_cache_dir is required when --data_dir is used")

    add_repo_imports(args.its_dir)
    from models.ConvIR import build_net as build_convir_net
    from models.DPGAConvIR import build_dpga_net

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    convir = build_convir_net(args.version, args.data, "original").to(device)
    udp_lite = build_dpga_net(
        args.version,
        args.data,
        prior_embed_channels=args.dpga_prior_embed_channels,
        adapter_reduction=args.dpga_adapter_reduction,
        adapter_residual_scale=args.dpga_adapter_residual_scale,
        adapter_scale_init=args.dpga_adapter_scale_init,
        adapter_bootstrap_scale=args.dpga_adapter_bootstrap_scale,
        active_adapters=args.dpga_active_adapters,
        fusion_mode="udp_lite",
        udp_components=args.dpga_udp_components,
        udp_window_size=args.dpga_udp_window_size,
        udp_num_heads=args.dpga_udp_num_heads,
    ).to(device)
    missing = load_convir_into_udp_lite(convir, udp_lite, args.checkpoint, device)
    if args.data_dir:
        input_img, label_img, depth, sample = make_data_input(args, device)
    else:
        input_img, label_img, depth, sample = make_synthetic_input(args, device)

    diff_rows = max_output_diffs(convir, udp_lite, input_img, depth)
    loss, grad_rows = grad_probe(udp_lite, input_img, label_img, depth)
    max_abs = max(row["max_abs"] for row in diff_rows)
    grad_active = any(row["grad_abs_sum"] > 0 for row in grad_rows)
    payload = {
        "sample": sample,
        "checkpoint": args.checkpoint,
        "fusion_mode": "udp_lite",
        "active_adapters": args.dpga_active_adapters,
        "udp_components": args.dpga_udp_components,
        "missing_dpga_key_count": len(missing),
        "missing_dpga_key_prefix_ok": all(key.startswith("DPGA_") for key in missing),
        "output_diffs": diff_rows,
        "max_abs_diff": max_abs,
        "tolerance": args.tolerance,
        "equivalence_pass": max_abs <= args.tolerance,
        "grad_probe_loss": loss,
        "grad_rows": grad_rows,
        "grad_probe_pass": grad_active,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if max_abs > args.tolerance:
        raise SystemExit(f"UDP-Lite zero-init mismatch: max_abs={max_abs} tolerance={args.tolerance}")
    if not grad_active:
        raise SystemExit("UDP-Lite zero-init grad probe found no active zero-init projection gradients.")


if __name__ == "__main__":
    main()
