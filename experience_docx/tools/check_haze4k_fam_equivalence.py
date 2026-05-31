import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.getcwd())

from models.ConvIR import build_net


def load_shared_weights(original, candidate, checkpoint, device):
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)["model"]
        original.load_state_dict(state)
    shared = original.state_dict()
    result = candidate.load_state_dict(shared, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    bad_missing = [key for key in missing if ".modulator." not in key]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Unexpected shared-weight load result: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return missing


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def check_fresh_shared_init(candidate_mode, seed):
    torch.manual_seed(seed)
    original = build_net("base", "Haze4K", "original")
    torch.manual_seed(seed)
    candidate = build_net("base", "Haze4K", candidate_mode)

    candidate_state = candidate.state_dict()
    max_abs = 0.0
    bad_keys = []
    checked = 0
    for key, tensor in original.state_dict().items():
        if key not in candidate_state:
            bad_keys.append(key)
            continue
        delta = (tensor - candidate_state[key]).abs().max().item()
        max_abs = max(max_abs, delta)
        checked += 1
        if delta != 0.0:
            bad_keys.append(key)
    return {
        "checked_shared_keys": checked,
        "max_abs_diff": max_abs,
        "bad_shared_keys": bad_keys[:20],
        "bad_shared_key_count": len(bad_keys),
        "pass": max_abs == 0.0 and len(bad_keys) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--candidate_mode", default="fam2_modres")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    parser.add_argument("--max_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--mean_abs_threshold", type=float, default=1e-7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    fresh_init = check_fresh_shared_init(args.candidate_mode, args.seed)
    original = build_net("base", "Haze4K", "original").to(device).eval()
    candidate = build_net("base", "Haze4K", args.candidate_mode).to(device).eval()
    missing = load_shared_weights(original, candidate, args.checkpoint, device)

    x = torch.rand(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        original_out = original(x)
        candidate_out = candidate(x)

    diffs = []
    for idx, (base_tensor, cand_tensor) in enumerate(zip(original_out, candidate_out)):
        delta = (base_tensor - cand_tensor).abs()
        diffs.append(
            {
                "output_index": idx,
                "max_abs_diff": delta.max().item(),
                "mean_abs_diff": delta.mean().item(),
            }
        )

    max_abs = max(item["max_abs_diff"] for item in diffs)
    mean_abs = max(item["mean_abs_diff"] for item in diffs)
    result = {
        "candidate_mode": args.candidate_mode,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "input_shape": [1, 3, args.height, args.width],
        "missing_candidate_keys": missing,
        "original_params": count_parameters(original),
        "candidate_params": count_parameters(candidate),
        "param_delta": count_parameters(candidate) - count_parameters(original),
        "param_delta_pct": (
            (count_parameters(candidate) - count_parameters(original))
            / count_parameters(original)
            * 100.0
        ),
        "fresh_shared_init": fresh_init,
        "outputs": diffs,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "pass": (
            max_abs < args.max_abs_threshold
            and mean_abs < args.mean_abs_threshold
            and fresh_init["pass"]
        ),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
