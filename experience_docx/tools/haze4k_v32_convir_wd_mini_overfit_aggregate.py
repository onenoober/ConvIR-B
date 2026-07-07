#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import torch

from haze4k_v32_convir_wd_mini_overfit import (
    _add_its_path,
    _git_value,
    _load_checkpoint_model,
    _load_pairs,
    _mean_activity,
    _partial_load,
    _set_scope,
    _loss,
)


def _add_weighted_parts(acc, parts, weight):
    for key, value in parts.items():
        acc[key] = acc.get(key, 0.0) + float(value) * weight


def _add_weighted_stats(acc, stats, weight):
    for block_name, block in stats.items():
        out_block = acc.setdefault(block_name, {})
        for key, value in block.items():
            out_block[key] = out_block.get(key, 0.0) + float(value) * weight


def _finalize_parts(acc, total_weight):
    return {key: value / total_weight for key, value in sorted(acc.items())}


def _finalize_stats(acc, total_weight):
    return {
        block_name: {
            key: value / total_weight
            for key, value in sorted(block.items())
        }
        for block_name, block in sorted(acc.items())
    }


def _aggregate_eval(model, inputs, labels, batch_size):
    n = inputs.shape[0]
    total_loss = 0.0
    total_parts = {}
    total_stats = {}
    outputs_finite = True
    batches = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_x = inputs[start:end]
            batch_y = labels[start:end]
            outputs = model(batch_x)
            loss, parts = _loss(outputs, batch_y)
            stats = model.collect_wd_stats(batch_x)
            weight = end - start
            loss_value = float(loss.detach().cpu())
            total_loss += loss_value * weight
            _add_weighted_parts(total_parts, parts, weight)
            _add_weighted_stats(total_stats, stats, weight)
            finite = all(torch.isfinite(out).all().item() for out in outputs)
            outputs_finite = outputs_finite and bool(finite)
            batches.append({
                "start": start,
                "end": end,
                "count": weight,
                "loss": loss_value,
                "outputs_finite": bool(finite),
                **parts,
            })

    return {
        "loss": total_loss / n,
        "parts": _finalize_parts(total_parts, n),
        "wd_stats": _finalize_stats(total_stats, n),
        "outputs_finite": bool(outputs_finite),
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--scope", default="wd_decoder", choices=["wd_only", "wd_decoder", "all"])
    parser.add_argument("--wd_lr", type=float, default=2e-4)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gate_loss_ratio", type=float, default=0.95)
    parser.add_argument("--version", default="base")
    parser.add_argument("--data", default="Haze4K")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _add_its_path()
    from models.ConvIRWD import build_convir_wd_net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_convir_wd_net(args.version, args.data).to(device)
    report = _partial_load(model, _load_checkpoint_model(args.checkpoint), ("WD_",))
    groups = _set_scope(model, args.scope, args.wd_lr, args.decoder_lr)
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

    inputs, labels, names = _load_pairs(args.data_dir, args.samples, args.crop_size)
    inputs = inputs.to(device)
    labels = labels.to(device)

    initial_eval = _aggregate_eval(model, inputs, labels, args.batch_size)

    history = []
    n = inputs.shape[0]
    model.train()
    for step in range(1, args.steps + 1):
        start = ((step - 1) * args.batch_size) % n
        end = start + args.batch_size
        if end <= n:
            batch_x = inputs[start:end]
            batch_y = labels[start:end]
        else:
            batch_x = torch.cat([inputs[start:n], inputs[0:end - n]], dim=0)
            batch_y = torch.cat([labels[start:n], labels[0:end - n]], dim=0)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss, parts = _loss(outputs, batch_y)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip_norm,
            )
        optimizer.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 6) == 0:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                **parts,
            })
            print(f"STEP {step:04d} loss={history[-1]['loss']:.8f}")

    final_eval = _aggregate_eval(model, inputs, labels, args.batch_size)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    initial_value = initial_eval["loss"]
    final_value = final_eval["loss"]
    loss_ratio = final_value / initial_value if initial_value > 0 else None
    activity_delta = _mean_activity(final_eval["wd_stats"]) - _mean_activity(initial_eval["wd_stats"])
    finite = initial_eval["outputs_finite"] and final_eval["outputs_finite"]

    result = {
        "route_id": "haze4k_v3_2_convir_wd_full_model_line_20260707",
        "phase": "P1b aggregate mini-overfit sanity",
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "--short", "HEAD"]),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "sample_names": names,
        "sample_contract": f"{args.samples} train-derived center crops, crop_size={args.crop_size}",
        "aggregate_metric_contract": (
            "initial/final loss, output finite check, and WD activity are measured "
            "over every loaded train-derived crop in eval mode, chunked by batch_size"
        ),
        "corrects_prior_p1_gate_scope": (
            "v32_p1_mini_overfit.json measured the final gate on inputs[:batch_size]; "
            "this P1b result measures the gate on all loaded samples"
        ),
        "aggregate_sample_count": n,
        "aggregate_batch_size": args.batch_size,
        "scope": args.scope,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "lr_groups": [{"name": g.get("name"), "lr": g["lr"]} for g in groups],
        "partial_load": {
            "loaded_count": len(report["loaded"]),
            "missing_new_count": len(report["missing_new_modules"]),
            "unexpected": report["unexpected"],
            "shape_mismatch": report["shape_mismatch"],
        },
        "initial_loss": initial_value,
        "initial_parts": initial_eval["parts"],
        "initial_batches": initial_eval["batches"],
        "final_loss": final_value,
        "final_parts": final_eval["parts"],
        "final_batches": final_eval["batches"],
        "loss_ratio": loss_ratio,
        "gate_loss_ratio": args.gate_loss_ratio,
        "history": history,
        "initial_wd_stats": initial_eval["wd_stats"],
        "final_wd_stats": final_eval["wd_stats"],
        "wd_activity_delta": activity_delta,
        "outputs_finite": bool(finite),
        "locked_test_touched": False,
        "quality_claim": "none; aggregate mini-overfit numerical/trainability sanity only",
        "pass": bool(finite and loss_ratio is not None and loss_ratio <= args.gate_loss_ratio and activity_delta > 0.0),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
