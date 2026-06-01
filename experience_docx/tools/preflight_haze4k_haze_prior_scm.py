import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from data import train_dataloader
from models.ConvIR import build_net


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def per_image_l1(pred, target):
    return (pred - target).abs().flatten(1).mean(dim=1)


def hard_rank_weights(per_image_loss):
    batch_size = per_image_loss.numel()
    if batch_size <= 1:
        return torch.ones_like(per_image_loss)
    order = torch.argsort(per_image_loss, descending=True)
    ranks = torch.empty_like(per_image_loss)
    ranks[order] = torch.arange(batch_size, device=per_image_loss.device, dtype=per_image_loss.dtype)
    return 1.0 - ranks / float(batch_size - 1)


def restore_losses(pred_img, label_img):
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    l1_per_image = (
        per_image_l1(pred_img[0], label_img4)
        + per_image_l1(pred_img[1], label_img2)
        + per_image_l1(pred_img[2], label_img)
    )

    fft_per_image = []
    for pred, target in ((pred_img[0], label_img4), (pred_img[1], label_img2), (pred_img[2], label_img)):
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_per_image.append(per_image_l1(pred_fft, target_fft))
    fft_per_image = sum(fft_per_image)

    restore_per_image = l1_per_image + 0.1 * fft_per_image
    restore_loss = restore_per_image.mean()
    hard_weight = hard_rank_weights(restore_per_image.detach())
    focus_weight = 1.0 + hard_weight
    focus_weight = focus_weight / focus_weight.mean().clamp_min(1e-6)
    hard_focus_loss = (focus_weight * restore_per_image).mean()
    return {
        "content": l1_per_image.mean(),
        "fft": fft_per_image.mean(),
        "restore": restore_loss,
        "hard_focus": hard_focus_loss,
    }


def finite_grad_summary(model):
    total_norm_sq = 0.0
    max_abs = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return {"finite": False}
        total_norm_sq += grad.float().pow(2).sum().item()
        max_abs = max(max_abs, grad.abs().max().item())
        param_count += grad.numel()
    return {
        "finite": True,
        "grad_param_count": param_count,
        "grad_l2_norm": math.sqrt(total_norm_sq),
        "grad_max_abs": max_abs,
    }


def prior_grad_summary(model):
    total_norm_sq = 0.0
    max_abs = 0.0
    param_count = 0
    nonzero_count = 0
    for name, param in model.named_parameters():
        if "prior_branch" not in name or param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return {"finite": False}
        abs_grad = grad.abs()
        total_norm_sq += grad.float().pow(2).sum().item()
        max_abs = max(max_abs, abs_grad.max().item())
        param_count += grad.numel()
        nonzero_count += (abs_grad > 0).sum().item()
    return {
        "finite": True,
        "grad_param_count": param_count,
        "nonzero_grad_count": nonzero_count,
        "grad_l2_norm": math.sqrt(total_norm_sq),
        "grad_max_abs": max_abs,
    }


def load_batch(args, device):
    if not args.data_dir:
        input_img = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
        label_img = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
        return input_img, label_img, "synthetic"

    loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, "Haze4K")
    input_img, label_img = next(iter(loader))
    return input_img.to(device), label_img.to(device), "real_haze4k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--loss_mode", default="hard_aux", choices=["original", "hard_aux"])
    parser.add_argument("--hard_aux_lambda", type=float, default=0.25)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(args.seed)
    original = build_net("base", "Haze4K", "original", "original").to(device)
    torch.manual_seed(args.seed)
    candidate = build_net("base", "Haze4K", "original", "haze_prior").to(device)

    input_img, label_img, batch_source = load_batch(args, device)
    original.eval()
    candidate.eval()
    with torch.no_grad():
        original_outputs = original(input_img)
        candidate_outputs = candidate(input_img)
        max_abs_diffs = [
            (cand - orig).abs().max().item()
            for cand, orig in zip(candidate_outputs, original_outputs)
        ]
        scm_stats = candidate.collect_scm_stats(input_img)

    candidate.train()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=4e-4, betas=(0.9, 0.999), eps=1e-8)
    optimizer.zero_grad()
    pred_img = candidate(input_img)
    losses = restore_losses(pred_img, label_img)
    loss = losses["restore"]
    if args.loss_mode == "hard_aux":
        loss = losses["restore"] + args.hard_aux_lambda * (losses["hard_focus"] - losses["restore"])
    loss.backward()
    grad = finite_grad_summary(candidate)
    prior_grad = prior_grad_summary(candidate)
    optimizer.step()

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    original_params = count_parameters(original)
    candidate_params = count_parameters(candidate)
    result = {
        "seed": args.seed,
        "batch_source": batch_source,
        "batch_size": args.batch_size,
        "input_shape": list(input_img.shape),
        "loss_mode": args.loss_mode,
        "hard_aux_lambda": args.hard_aux_lambda if args.loss_mode == "hard_aux" else 0.0,
        "output_shapes": [list(tensor.shape) for tensor in pred_img],
        "neutral_init_max_abs_diffs": max_abs_diffs,
        "neutral_init_pass": max(max_abs_diffs) <= 1e-6,
        "loss_content": losses["content"].item(),
        "loss_fft": losses["fft"].item(),
        "loss_restore": losses["restore"].item(),
        "loss_hard_focus": losses["hard_focus"].item(),
        "loss_total": loss.item(),
        "loss_finite": torch.isfinite(loss).item(),
        "grad": grad,
        "prior_branch_grad": prior_grad,
        "original_params": original_params,
        "candidate_params": candidate_params,
        "param_delta": candidate_params - original_params,
        "param_delta_pct": (candidate_params - original_params) / original_params * 100.0,
        "scm_stats": scm_stats,
        "peak_cuda_mem_mib": peak_mem,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if not result["neutral_init_pass"] or not result["loss_finite"] or not grad.get("finite", False):
        raise SystemExit(1)
    if not prior_grad.get("finite", False) or prior_grad.get("nonzero_grad_count", 0) == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
