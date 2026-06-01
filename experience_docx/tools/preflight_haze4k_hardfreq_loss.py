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


def per_image_l1(pred, target):
    return (pred - target).abs().flatten(1).mean(dim=1)


def hard_rank_weights(per_image_loss):
    batch_size = per_image_loss.numel()
    if batch_size <= 1:
        return torch.zeros_like(per_image_loss)
    order = torch.argsort(per_image_loss, descending=False)
    ranks = torch.empty_like(per_image_loss)
    ranks[order] = torch.arange(batch_size, device=per_image_loss.device, dtype=per_image_loss.dtype)
    return ranks / float(batch_size - 1)


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--hard_fft_lambda", type=float, default=0.02)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_net("base", "Haze4K", "original").to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999), eps=1e-8)
    loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, "Haze4K")
    input_img, label_img = next(iter(loader))
    input_img = input_img.to(device)
    label_img = label_img.to(device)

    optimizer.zero_grad()
    pred_img = model(input_img)
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")

    l1_per_image = per_image_l1(pred_img[0], label_img4)
    l2_per_image = per_image_l1(pred_img[1], label_img2)
    l3_per_image = per_image_l1(pred_img[2], label_img)
    loss_content_per_image = l1_per_image + l2_per_image + l3_per_image
    loss_content = loss_content_per_image.mean()

    fft_per_image = []
    for pred, target in ((pred_img[0], label_img4), (pred_img[1], label_img2), (pred_img[2], label_img)):
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_per_image.append(per_image_l1(pred_fft, target_fft))
    loss_fft_per_image = sum(fft_per_image)
    loss_fft = loss_fft_per_image.mean()

    restore_loss_per_image = (loss_content_per_image + 0.1 * loss_fft_per_image).detach()
    hard_weight = hard_rank_weights(restore_loss_per_image)
    hard_fft_loss = (hard_weight * loss_fft_per_image).mean()
    loss = loss_content + 0.1 * loss_fft + args.hard_fft_lambda * hard_fft_loss
    loss.backward()
    grad = finite_grad_summary(model)
    optimizer.step()

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    result = {
        "mode": "original",
        "loss_mode": "hard_fft_boost",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_shape": list(input_img.shape),
        "output_shapes": [list(tensor.shape) for tensor in pred_img],
        "loss_content": loss_content.item(),
        "loss_fft": loss_fft.item(),
        "hard_fft_loss": hard_fft_loss.item(),
        "hard_fft_lambda": args.hard_fft_lambda,
        "loss_total": loss.item(),
        "loss_finite": torch.isfinite(loss).item(),
        "hard_weight_mean": hard_weight.mean().item(),
        "hard_weight_min": hard_weight.min().item(),
        "hard_weight_max": hard_weight.max().item(),
        "restore_loss_min": restore_loss_per_image.min().item(),
        "restore_loss_max": restore_loss_per_image.max().item(),
        "grad": grad,
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
