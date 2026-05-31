import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from data import train_dataloader
from models.ConvIR import build_net


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", default="fam2_modres")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    original = build_net("base", "Haze4K", "original")
    candidate = build_net("base", "Haze4K", args.mode).to(device)
    candidate.train()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=4e-4, betas=(0.9, 0.999), eps=1e-8)
    criterion = torch.nn.L1Loss()
    loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, "Haze4K")
    input_img, label_img = next(iter(loader))
    input_img = input_img.to(device)
    label_img = label_img.to(device)

    optimizer.zero_grad()
    pred_img = candidate(input_img)
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
    loss_content = (
        criterion(pred_img[0], label_img4)
        + criterion(pred_img[1], label_img2)
        + criterion(pred_img[2], label_img)
    )

    fft_losses = []
    for pred, target in ((pred_img[0], label_img4), (pred_img[1], label_img2), (pred_img[2], label_img)):
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_losses.append(criterion(pred_fft, target_fft))
    loss_fft = sum(fft_losses)
    loss = loss_content + 0.1 * loss_fft
    loss.backward()
    grad = finite_grad_summary(candidate)
    optimizer.step()

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    candidate_params = count_parameters(candidate)
    original_params = count_parameters(original)
    result = {
        "mode": args.mode,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_shape": list(input_img.shape),
        "output_shapes": [list(tensor.shape) for tensor in pred_img],
        "loss_content": loss_content.item(),
        "loss_fft": loss_fft.item(),
        "loss_total": loss.item(),
        "loss_finite": torch.isfinite(loss).item(),
        "grad": grad,
        "original_params": original_params,
        "candidate_params": candidate_params,
        "param_delta": candidate_params - original_params,
        "param_delta_pct": (candidate_params - original_params) / original_params * 100.0,
        "peak_cuda_mem_mib": peak_mem,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if not result["loss_finite"] or not grad.get("finite", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
