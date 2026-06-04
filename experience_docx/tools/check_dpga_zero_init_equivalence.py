import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from data import test_dataloader  # noqa: E402
from models.ConvIR import build_net as build_convir_net  # noqa: E402
from models.DPGAConvIR import build_dpga_net  # noqa: E402


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_convir_into_dpga(convir, dpga, checkpoint, device):
    if checkpoint:
        state = load_model_state(checkpoint, device)
        convir.load_state_dict(state)
    state = convir.state_dict()
    result = dpga.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("DPGA_")]
    if unexpected or bad_missing:
        raise RuntimeError(f"Bad DPGA init load: missing={missing}, unexpected={unexpected}")
    return missing


def make_input(args, device):
    if args.data_dir:
        loader = test_dataloader(
            args.data_dir,
            "Haze4K",
            batch_size=1,
            num_workers=0,
            depth_cache_dir=args.depth_cache_dir,
            depth_split=args.depth_split,
        )
        input_img, label_img, depth, name = next(iter(loader))
        return input_img.to(device), label_img.to(device), depth.to(device), name[0]

    torch.manual_seed(args.seed)
    input_img = torch.rand(1, 3, args.size, args.size, device=device)
    label_img = torch.rand(1, 3, args.size, args.size, device=device)
    depth = torch.rand(1, 1, args.size, args.size, device=device)
    return input_img, label_img, depth, "synthetic"


def max_output_diffs(convir, dpga, input_img, depth):
    convir.eval()
    dpga.eval()
    with torch.no_grad():
        ref = convir(input_img)
        out = dpga(input_img, depth)
    rows = []
    for idx, (a, b) in enumerate(zip(ref, out)):
        diff = (a - b).abs()
        rows.append(
            {
                "scale": idx,
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
            }
        )
    return rows


def adapter_grad_norms(dpga, input_img, label_img, depth):
    dpga.eval()
    for name, module in dpga.named_modules():
        if name.startswith("DPGA_"):
            module.train()
    dpga.zero_grad(set_to_none=True)
    pred = dpga(input_img, depth)[-1]
    loss = F.l1_loss(pred, label_img)
    loss.backward()
    rows = []
    for name, param in dpga.named_parameters():
        if name.startswith("DPGA_") and name.endswith("project.weight"):
            rows.append(
                {
                    "name": name,
                    "grad_abs_sum": 0.0 if param.grad is None else param.grad.abs().sum().item(),
                    "grad_abs_max": 0.0 if param.grad is None else param.grad.abs().max().item(),
                }
            )
    return loss.item(), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--depth_cache_dir", default="")
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--version", default="base", choices=["small", "base", "large"])
    parser.add_argument("--data", default="Haze4K")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--dpga_adapter_bootstrap_scale", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.data_dir and not args.depth_cache_dir:
        raise ValueError("--depth_cache_dir is required when --data_dir is used")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    convir = build_convir_net(args.version, args.data, "original").to(device)
    dpga = build_dpga_net(
        args.version,
        args.data,
        adapter_bootstrap_scale=args.dpga_adapter_bootstrap_scale,
    ).to(device)
    missing = load_convir_into_dpga(convir, dpga, args.checkpoint, device)
    input_img, label_img, depth, name = make_input(args, device)
    diff_rows = max_output_diffs(convir, dpga, input_img, depth)
    loss, grad_rows = adapter_grad_norms(dpga, input_img, label_img, depth)

    print(f"sample={name}")
    print(f"missing_dpga_keys={len(missing)}")
    for row in diff_rows:
        print(
            "output_scale=%d max_abs=%.12g mean_abs=%.12g"
            % (row["scale"], row["max_abs"], row["mean_abs"])
        )
    print(f"grad_probe_loss={loss:.8f}")
    for row in grad_rows:
        print(
            "grad %s abs_sum=%.12g abs_max=%.12g"
            % (row["name"], row["grad_abs_sum"], row["grad_abs_max"])
        )

    max_abs = max(row["max_abs"] for row in diff_rows)
    if max_abs > args.tolerance:
        raise SystemExit(f"DPGA zero-init mismatch: max_abs={max_abs} tolerance={args.tolerance}")
    if not any(row["grad_abs_sum"] > 0 for row in grad_rows):
        raise SystemExit("DPGA adapter projection gradients are all zero; bootstrap/startup is dead.")


if __name__ == "__main__":
    main()
