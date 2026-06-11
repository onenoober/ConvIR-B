import argparse
import hashlib
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

from data import train_dataloader
from models.ConvIR import build_net


def unpack_batch(batch):
    input_img, label_img = batch[0], batch[1]
    depth = batch[2] if len(batch) >= 3 else None
    trans = batch[3] if len(batch) >= 4 else None
    airlight = batch[4] if len(batch) >= 5 else None
    return input_img, label_img, depth, trans, airlight


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def partial_load_dta(model, state):
    result = model.load_state_dict(state, strict=False)
    missing_bad = [key for key in result.missing_keys if not key.startswith("DTA.")]
    unexpected = list(result.unexpected_keys)
    if missing_bad or unexpected:
        raise RuntimeError(
            f"DTA partial load failed: missing_bad={missing_bad} unexpected={unexpected}"
        )
    loaded = [key for key in state if key in model.state_dict()]
    return {
        "loaded_count": len(loaded),
        "missing_count": len(result.missing_keys),
        "missing_new_prefix_ok": all(key.startswith("DTA.") for key in result.missing_keys),
        "missing_examples": result.missing_keys[:10],
        "unexpected": unexpected,
    }


def build_dta(args):
    return build_net(
        "base",
        "Haze4K",
        "original",
        arch=args.arch,
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
        dta_output_residual_scale=args.dta_output_residual_scale,
    )


def synthetic_noop_check(original, dta, device):
    torch.manual_seed(3407)
    # ConvIR's deep pooling path uses reflection padding, so keep the synthetic
    # preflight image large enough to exercise the official runtime shape safely.
    x = torch.rand(1, 3, 256, 256, device=device)
    depth = torch.rand(1, 1, 256, 256, device=device)
    original.eval()
    dta.eval()
    with torch.no_grad():
        out_original = original(x)
        out_dta = dta(x, depth)
    diffs = [
        float((a - b).abs().max().detach().cpu())
        for a, b in zip(out_original, out_dta)
    ]
    stats = dta.DTA.stats()
    return {
        "output_shapes": [list(item.shape) for item in out_dta],
        "max_abs_diffs": diffs,
        "max_abs_diff": max(diffs),
        "dta_stats": stats,
    }


def real_batch_backward_check(model, args, device):
    if not args.data_dir or not args.depth_cache_dir:
        return {"skipped": True, "reason": "data_dir or depth_cache_dir not provided"}
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("DTA.")
    model.train()
    loader = train_dataloader(
        args.data_dir,
        # BatchNorm layers can see 1x1 pooled tensors in train mode; two samples
        # keep the real-batch gradient probe aligned with actual smoke training.
        batch_size=2,
        num_workers=0,
        data="Haze4K",
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        return_trans=args.use_trans_gt,
        return_meta=args.use_trans_gt,
    )
    batch = next(iter(loader))
    input_img, label_img, depth, trans, airlight = unpack_batch(batch)
    if depth is None:
        raise RuntimeError("Expected train batch with image, label, depth")
    input_img = input_img.to(device)
    label_img = label_img.to(device)
    depth = depth.to(device)
    trans = trans.to(device) if trans is not None else None
    airlight = airlight.to(device) if airlight is not None and hasattr(airlight, "to") else airlight
    outputs = model(input_img, depth)
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    loss_content = (
        F.l1_loss(outputs[0], label_img4)
        + F.l1_loss(outputs[1], label_img2)
        + F.l1_loss(outputs[2], label_img)
    )
    aux = model.dta_auxiliary_losses(rank_pairs=args.rank_pairs, min_depth_gap=0.03)
    sup = {"trans": input_img.new_zeros(()), "phys": input_img.new_zeros(())}
    if hasattr(model, "dta_supervised_losses") and trans is not None:
        sup = model.dta_supervised_losses(
            trans_gt=trans,
            hazy=input_img,
            dehazed=outputs[2],
            airlight=airlight,
        )
    loss = (
        loss_content
        + args.rank_weight * aux["rank"]
        + args.tv_weight * aux["tv"]
        + args.trans_weight * sup["trans"]
        + args.phys_weight * sup["phys"]
    )
    loss.backward()
    grad_abs_sum = 0.0
    grad_nonzero = 0
    for name, param in model.named_parameters():
        if name.startswith("DTA.") and param.grad is not None:
            grad_abs = float(param.grad.detach().abs().sum().cpu())
            grad_abs_sum += grad_abs
            if grad_abs > 0:
                grad_nonzero += 1
    return {
        "skipped": False,
        "loss_content": float(loss_content.detach().cpu()),
        "loss_rank": float(aux["rank"].detach().cpu()),
        "loss_tv": float(aux["tv"].detach().cpu()),
        "loss_trans": float(sup["trans"].detach().cpu()),
        "loss_phys": float(sup["phys"].detach().cpu()),
        "loss_total": float(loss.detach().cpu()),
        "grad_abs_sum": grad_abs_sum,
        "grad_nonzero_tensors": grad_nonzero,
        "input_shape": list(input_img.shape),
        "depth_shape": list(depth.shape),
        "trans_shape": list(trans.shape) if trans is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--depth_cache_dir", default="")
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--arch", default="dta", choices=["dta", "dta_v2"])
    parser.add_argument("--dta_variant", default="v1", choices=["v1", "v2"])
    parser.add_argument("--dta_depth_mode", default="normal", choices=["normal", "invert", "zero", "shuffle"])
    parser.add_argument("--dta_prior_channels", type=int, default=16)
    parser.add_argument("--dta_gate_bias", type=float, default=-7.0)
    parser.add_argument("--dta_gate_limit", type=float, default=0.03)
    parser.add_argument("--dta_gamma_limit", type=float, default=0.10)
    parser.add_argument("--dta_beta_limit", type=float, default=0.05)
    parser.add_argument("--dta_alpha_init", type=float, default=1.0)
    parser.add_argument("--dta_confidence_floor", type=float, default=0.25)
    parser.add_argument("--dta_confidence_local_scale", type=float, default=6.0)
    parser.add_argument("--dta_output_residual_scale", type=float, default=0.03)
    parser.add_argument("--use_trans_gt", action="store_true")
    parser.add_argument("--rank_weight", type=float, default=0.003)
    parser.add_argument("--tv_weight", type=float, default=0.0003)
    parser.add_argument("--trans_weight", type=float, default=0.0)
    parser.add_argument("--phys_weight", type=float, default=0.0)
    parser.add_argument("--rank_pairs", type=int, default=256)
    parser.add_argument("--noop_tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_model_state(args.checkpoint, device)
    original = build_net("base", "Haze4K", "original", arch="official_convir").to(device)
    original.load_state_dict(state)
    dta = build_dta(args).to(device)
    partial = partial_load_dta(dta, state)
    synthetic = synthetic_noop_check(original, dta, device)
    real_batch = real_batch_backward_check(dta, args, device)

    ok = (
        partial["missing_new_prefix_ok"]
        and not partial["unexpected"]
        and synthetic["max_abs_diff"] <= args.noop_tolerance
        and (real_batch["skipped"] or real_batch["grad_abs_sum"] > 0.0)
    )
    payload = {
        "ok": ok,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_size": os.path.getsize(args.checkpoint),
        "partial_load": partial,
        "synthetic_noop": synthetic,
        "real_batch_backward": real_batch,
        "dta_config": {
            "arch": args.arch,
            "dta_variant": args.dta_variant,
            "dta_depth_mode": args.dta_depth_mode,
            "dta_prior_channels": args.dta_prior_channels,
            "dta_gate_bias": args.dta_gate_bias,
            "dta_gate_limit": args.dta_gate_limit,
            "dta_gamma_limit": args.dta_gamma_limit,
            "dta_beta_limit": args.dta_beta_limit,
            "dta_alpha_init": args.dta_alpha_init,
            "dta_confidence_floor": args.dta_confidence_floor,
            "dta_confidence_local_scale": args.dta_confidence_local_scale,
            "dta_output_residual_scale": args.dta_output_residual_scale,
            "use_trans_gt": args.use_trans_gt,
            "rank_weight": args.rank_weight,
            "tv_weight": args.tv_weight,
            "trans_weight": args.trans_weight,
            "phys_weight": args.phys_weight,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not ok:
        raise SystemExit("DTA_PREFLIGHT_FAILED")
    print("DTA_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
