#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "experience_docx" / "tools"
for path in (str(Path(__file__).resolve().parent), str(ROOT), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor  # noqa: E402
from models.ConvIR import build_net as build_official  # noqa: E402
from models.NoPostFGAConvIR import build_net as build_nopost  # noqa: E402
from nopost_common import (  # noqa: E402
    image_tensor,
    infer_final,
    label_path,
    load_official_checkpoint,
    metric,
    names_for_scope,
    pad_to,
    partial_load_nopost,
    summarize_delta_rows,
    train_dirs,
    write_csv,
    write_json,
)


class NamedHaze4KDataset(Dataset):
    def __init__(self, data_dir: Path, names: list[str], transform=None):
        self.input_dir, self.gt_dir = train_dirs(data_dir)
        self.names = names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image = Image.open(self.input_dir / name).convert("RGB")
        label = Image.open(label_path(self.gt_dir, name)).convert("RGB")
        if self.transform is not None:
            image, label = self.transform(image, label)
        else:
            image = TVF.to_tensor(image)
            label = TVF.to_tensor(label)
        return image, label, name


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fft_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
    label_fft = torch.fft.fft2(label, dim=(-2, -1))
    label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
    return F.l1_loss(pred_fft, label_fft)


def psnr_per_sample(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - label) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10 * torch.log10(1 / mse)


def freeze_adapter_only(model) -> dict[str, int]:
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("nopost_adapter."):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return {"trainable": trainable, "frozen": frozen}


def set_adapter_train_mode(model) -> None:
    for name, module in model.named_children():
        if name == "nopost_adapter":
            module.train()
        else:
            module.eval()


def evaluate(model, a0_model, data_dir: Path, split_manifest: Path, scope: str, fold: int, max_images: int, device):
    input_dir, gt_dir = train_dirs(data_dir)
    rows = []
    model.eval()
    a0_model.eval()
    with torch.no_grad():
        for rec in names_for_scope(split_manifest, scope, fold=fold, max_images=max_images):
            hazy = image_tensor(input_dir / rec["name"], device)
            label = image_tensor(label_path(gt_dir, rec["name"]), device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0 = infer_final(a0_model, x, h, w)
            pred = infer_final(model, x, h, w)
            a0_psnr, a0_ssim = metric(a0, label, hp, wp)
            pred_psnr, pred_ssim = metric(pred, label, hp, wp)
            rows.append(
                {
                    "name": rec["name"],
                    "split": rec["split"],
                    "A0_PSNR": a0_psnr,
                    "candidate_PSNR": pred_psnr,
                    "dPSNR": pred_psnr - a0_psnr,
                    "A0_SSIM": a0_ssim,
                    "candidate_SSIM": pred_ssim,
                    "dSSIM": pred_ssim - a0_ssim,
                }
            )
    return rows, summarize_delta_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--train-scope", default="fold_train")
    ap.add_argument("--eval-scope", default="fold_val")
    ap.add_argument("--max-train-images", type=int, default=0)
    ap.add_argument("--max-eval-images", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=0.001)
    ap.add_argument("--action-budget", type=float, default=1e-4)
    ap.add_argument("--gate-budget", type=float, default=1e-5)
    ap.add_argument("--preserve-weight", type=float, default=0.0)
    ap.add_argument("--preserve-psnr-threshold", type=float, default=0.0)
    ap.add_argument("--use-detail", action="store_true")
    ap.add_argument("--gate-bias", type=float, default=-3.0)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints" / args.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_nopost(
        "base",
        "Haze4K",
        "original",
        nopost_use_low=True,
        nopost_use_detail=args.use_detail,
        nopost_gate_bias=args.gate_bias,
    ).to(device)
    a0_model = build_official("base", "Haze4K", "original").to(device)
    load_official_checkpoint(a0_model, args.a0_checkpoint, device)
    load_report = partial_load_nopost(model, args.a0_checkpoint, device)
    a0_model.eval()
    for p in a0_model.parameters():
        p.requires_grad = False
    param_counts = freeze_adapter_only(model)
    set_adapter_train_mode(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    names = [r["name"] for r in names_for_scope(args.split_manifest, args.train_scope, fold=args.fold, max_images=args.max_train_images)]
    transform = PairCompose([PairRandomCrop(256), PairRandomHorizontalFilp(), PairToTensor()])
    loader = DataLoader(
        NamedHaze4KDataset(args.data_dir, names, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    history = []
    gate_rows = []
    best_eval_psnr = -1e9
    for epoch in range(1, args.epochs + 1):
        set_adapter_train_mode(model)
        loss_sum = 0.0
        content_sum = 0.0
        fft_sum = 0.0
        batches = 0
        for input_img, label_img, _ in loader:
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            optimizer.zero_grad()
            pred = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
            loss_content = (
                F.l1_loss(pred[0], label_img4)
                + F.l1_loss(pred[1], label_img2)
                + F.l1_loss(pred[2], label_img)
            )
            loss_fft = fft_loss(pred[0], label_img4) + fft_loss(pred[1], label_img2) + fft_loss(pred[2], label_img)
            loss = loss_content + 0.1 * loss_fft
            reg = model.nopost_regularization()
            loss = loss + args.action_budget * reg["raw_action_abs_mean"] + args.gate_budget * reg["gate_mean"]
            if args.preserve_weight > 0:
                with torch.no_grad():
                    a0 = a0_model(input_img)[2]
                    if args.preserve_psnr_threshold > 0:
                        keep = psnr_per_sample(a0.clamp(0, 1), label_img) >= args.preserve_psnr_threshold
                    else:
                        keep = torch.ones(input_img.shape[0], dtype=torch.bool, device=device)
                if keep.any():
                    loss = loss + args.preserve_weight * torch.mean(torch.abs(pred[2][keep] - a0[keep]))
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip_norm)
            optimizer.step()
            stats = model.nopost_stats()
            gate_rows.append({"epoch": epoch, "batch": batches + 1, **stats})
            loss_sum += float(loss.detach().cpu())
            content_sum += float(loss_content.detach().cpu())
            fft_sum += float(loss_fft.detach().cpu())
            batches += 1

        eval_rows, eval_summary = evaluate(
            model,
            a0_model,
            args.data_dir,
            args.split_manifest,
            args.eval_scope,
            args.fold,
            args.max_eval_images,
            device,
        )
        eval_psnr = eval_summary["mean_dPSNR"]
        row = {
            "run_id": args.run_id,
            "epoch": epoch,
            "seed": args.seed,
            "fold": args.fold,
            "train_images": len(names),
            "batches": batches,
            "loss": loss_sum / max(1, batches),
            "loss_content": content_sum / max(1, batches),
            "loss_fft": fft_sum / max(1, batches),
            **{f"eval_{k}": v for k, v in eval_summary.items()},
        }
        history.append(row)
        write_csv(args.out_dir / f"{args.run_id}_eval_epoch{epoch:03d}_per_image.csv", eval_rows)
        torch.save({"model": model.state_dict()}, ckpt_dir / f"model_{epoch}.pkl")
        if eval_psnr >= best_eval_psnr:
            best_eval_psnr = eval_psnr
            torch.save({"model": model.state_dict()}, ckpt_dir / "Best.pkl")
        print(f"NOPOST_TRAIN_EPOCH {args.run_id} epoch={epoch} eval_mean_dPSNR={eval_psnr:.6f}", flush=True)

    torch.save({"model": model.state_dict()}, ckpt_dir / "Final.pkl")
    write_csv(args.out_dir / f"{args.run_id}_history.csv", history)
    write_csv(args.out_dir / f"{args.run_id}_gate_action_stats.csv", gate_rows)
    write_json(
        args.out_dir / f"{args.run_id}_train_summary.json",
        {
            "run_id": args.run_id,
            "seed": args.seed,
            "fold": args.fold,
            "train_scope": args.train_scope,
            "eval_scope": args.eval_scope,
            "max_train_images": args.max_train_images,
            "max_eval_images": args.max_eval_images,
            "epochs": args.epochs,
            "param_counts": param_counts,
            "partial_load": load_report,
            "best_eval_mean_dPSNR": best_eval_psnr,
            "locked_test_touched": False,
        },
    )
    print("NOPOST_TRAIN_OK", args.run_id)


if __name__ == "__main__":
    main()
