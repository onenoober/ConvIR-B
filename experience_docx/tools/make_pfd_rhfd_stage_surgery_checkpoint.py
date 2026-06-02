#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from models.PFDConvIR import build_pfd_net  # noqa: E402


def unwrap_state(path):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "net", "params"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def strip_module_prefix(state):
    if not state:
        return state
    if all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def scale_for_key(key, rhfd1_scale, rhfd2_scale):
    if key.startswith("PFD_RHFD1."):
        return rhfd1_scale
    if key.startswith("PFD_RHFD2."):
        return rhfd2_scale
    return 1.0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a stage-wise SafeRHFD checkpoint by combining an A0 ConvIR "
            "backbone with B1 RHFD branches and independent RHFD1/RHFD2 scales."
        )
    )
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--b1_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rhfd1_scale", type=float, required=True)
    parser.add_argument("--rhfd2_scale", type=float, required=True)
    parser.add_argument("--version", default="base")
    parser.add_argument("--data", default="Haze4K")
    args = parser.parse_args()

    if args.rhfd1_scale < 0 or args.rhfd2_scale < 0:
        raise ValueError("RHFD scales must be non-negative.")

    model = build_pfd_net(
        args.version,
        args.data,
        pfd_rhfd=True,
        pfd_hscm=False,
        pfd_pffb=False,
        pfd_pffb_high=False,
        pfd_teacher=False,
    )

    a0_state = strip_module_prefix(unwrap_state(args.a0_checkpoint))
    b1_state = strip_module_prefix(unwrap_state(args.b1_checkpoint))

    load_result = model.load_state_dict(a0_state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)

    bad_missing = [key for key in missing if not key.startswith("PFD_")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "A0 checkpoint did not cleanly load into the PFD wrapper.\n"
            f"Bad missing keys: {bad_missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    out_state = model.state_dict()
    copied_keys = []
    scaled_keys = []

    for key, value in b1_state.items():
        if not (key.startswith("PFD_RHFD1.") or key.startswith("PFD_RHFD2.")):
            continue

        if key not in out_state:
            raise RuntimeError(f"B1 RHFD key not found in target model: {key}")

        tensor = value.detach().cpu().clone()
        if key.endswith(".body.4.weight") or key.endswith(".body.4.bias"):
            branch_scale = scale_for_key(key, args.rhfd1_scale, args.rhfd2_scale)
            tensor.mul_(branch_scale)
            scaled_keys.append(key)

        out_state[key] = tensor
        copied_keys.append(key)

    if not copied_keys:
        raise RuntimeError("No PFD_RHFD1/PFD_RHFD2 keys were copied from B1 checkpoint.")

    expected_scaled = {
        "PFD_RHFD1.body.4.weight",
        "PFD_RHFD1.body.4.bias",
        "PFD_RHFD2.body.4.weight",
        "PFD_RHFD2.body.4.bias",
    }
    if set(scaled_keys) != expected_scaled:
        raise RuntimeError(
            "Unexpected RHFD final-conv scale keys.\n"
            f"Expected: {sorted(expected_scaled)}\n"
            f"Found: {sorted(scaled_keys)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": out_state,
            "source": {
                "a0_checkpoint": args.a0_checkpoint,
                "b1_checkpoint": args.b1_checkpoint,
                "copied_prefixes": ["PFD_RHFD1", "PFD_RHFD2"],
                "rhfd1_final_conv_scale": args.rhfd1_scale,
                "rhfd2_final_conv_scale": args.rhfd2_scale,
                "copied_key_count": len(copied_keys),
                "scaled_keys": scaled_keys,
                "surgery_rule": (
                    "A0 ConvIR backbone plus B1 RHFD branches with independent "
                    "RHFD1/RHFD2 final-conv scales"
                ),
            },
        },
        output,
    )

    print(f"Saved stage-wise SafeRHFD checkpoint: {output}")
    print(f"Copied RHFD tensors: {len(copied_keys)}")
    print(f"Scaled RHFD final-conv tensors: {len(scaled_keys)}")
    print(f"RHFD1 final conv scale: {args.rhfd1_scale}")
    print(f"RHFD2 final conv scale: {args.rhfd2_scale}")


if __name__ == "__main__":
    main()
