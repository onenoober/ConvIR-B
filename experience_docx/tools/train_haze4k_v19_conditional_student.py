#!/usr/bin/env python3
"""Train a v1.9 conditional teacher-guided DPGA student.

The loss uses GT reconstruction everywhere, UDP teacher distillation only where
UDP is locally better than A0, and A0 preservation where UDP is not locally
positive. This is cloud-runtime code; locked Haze4K test is not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for item in (str(ITS_ROOT), str(TOOL_PATH.parent), str(REPO_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from data import train_dataloader  # noqa: E402
from eval_udpnet_v15_phase0_repro import load_a0_model, load_udpnet_builder, load_udpnet_model  # noqa: E402
from models.ConvIR import build_net as build_convir_net  # noqa: E402
from models.DPGAConvIR import build_dpga_net  # noqa: E402


def load_student(path: str, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    model = build_dpga_net(
        "base",
        "Haze4K",
        prior_embed_channels=args.prior_embed_channels,
        adapter_residual_scale=args.adapter_residual_scale,
        adapter_bootstrap_scale=args.adapter_bootstrap_scale,
        active_adapters=args.active_adapters,
        scale_multiplier=args.scale_multiplier,
        fusion_mode=args.fusion_mode,
        udp_components=args.udp_components,
    ).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    result = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("DPGA_")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"student init mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}")
    return model


def set_train_scope(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter] | list[dict[str, object]]:
    neighbor_prefixes = ("FAM1", "FAM2", "SCM1", "SCM2", "Convs.0", "Convs.1")
    dpga_params = []
    neighbor_params = []
    for name, param in model.named_parameters():
        is_dpga = name.startswith("DPGA_")
        is_neighbor = scope == "fusion_neighbor" and any(name.startswith(prefix) for prefix in neighbor_prefixes)
        param.requires_grad = is_dpga or is_neighbor or scope == "all"
        if param.requires_grad and is_dpga:
            dpga_params.append(param)
        elif param.requires_grad:
            neighbor_params.append(param)
    if scope == "all":
        return [param for param in model.parameters() if param.requires_grad]
    if not dpga_params and not neighbor_params:
        raise RuntimeError("No trainable parameters selected")
    if scope == "fusion_neighbor" and neighbor_params:
        return [{"params": dpga_params}, {"params": neighbor_params, "lr": 1e-5}]
    return dpga_params


def set_training_mode(model: torch.nn.Module, scope: str) -> None:
    if scope == "all":
        model.train()
        return
    model.eval()
    neighbor_prefixes = ("FAM1", "FAM2", "SCM1", "SCM2", "Convs.0", "Convs.1")
    for name, module in model.named_modules():
        if name.startswith("DPGA_") or (scope == "fusion_neighbor" and any(name.startswith(prefix) for prefix in neighbor_prefixes)):
            module.train()


def l1_weighted(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).abs()
    if weight is None:
        return loss.mean()
    if weight.shape[1] == 1 and pred.shape[1] != 1:
        weight = weight.expand(-1, pred.shape[1], -1, -1)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def build_teacher_mask(a0: torch.Tensor, udp: torch.Tensor, gt: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    a0_err = (a0 - gt).abs().mean(dim=1, keepdim=True)
    udp_err = (udp - gt).abs().mean(dim=1, keepdim=True)
    positive = (udp_err + args.teacher_margin < a0_err).float()
    if args.mask_pool > 1:
        positive = F.avg_pool2d(positive, kernel_size=args.mask_pool, stride=1, padding=args.mask_pool // 2)
    return positive.clamp(0, 1)


def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convir_its_dir", default=str(ITS_ROOT))
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--train_split", default="train_inner")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.01)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--fusion_mode", default="udp_bi", choices=["udp_lite", "udp_bi"])
    parser.add_argument("--active_adapters", default="dpfm1,agf1")
    parser.add_argument("--udp_components", default="all")
    parser.add_argument("--train_scope", default="fusion_neighbor", choices=["active_adapter_only", "fusion_neighbor", "all"])
    parser.add_argument("--prior_embed_channels", type=int, default=24)
    parser.add_argument("--adapter_residual_scale", type=float, default=0.2)
    parser.add_argument("--adapter_bootstrap_scale", type=float, default=0.02)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--w_gt", type=float, default=1.0)
    parser.add_argument("--w_teacher", type=float, default=0.25)
    parser.add_argument("--w_preserve", type=float, default=0.20)
    parser.add_argument("--teacher_margin", type=float, default=0.002)
    parser.add_argument("--mask_pool", type=int, default=17)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    a0_model = load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, udp_meta = load_udpnet_model(load_udpnet_builder(Path(args.udp_repo)), Path(args.official_checkpoint), device)
    student = load_student(args.a0_checkpoint, device, args)
    ema_student = load_student(args.a0_checkpoint, device, args) if args.ema else None
    trainable = set_train_scope(student, args.train_scope)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    loader = train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data="Haze4K",
        depth_cache_dir=args.depth_cache_dir,
        depth_split="train",
        split_json=args.split_json,
        split_name=args.train_split,
    )

    status_rows = []
    for epoch in range(1, args.epochs + 1):
        set_training_mode(student, args.train_scope)
        sums = {"loss": 0.0, "gt": 0.0, "teacher": 0.0, "preserve": 0.0, "mask": 0.0}
        count = 0
        for batch_idx, batch in enumerate(loader):
            hazy, gt, depth = batch[:3]
            hazy = hazy.to(device)
            gt = gt.to(device)
            depth = depth.to(device)
            with torch.no_grad():
                a0 = torch.clamp(a0_model(hazy)[2], 0, 1)
                udp = torch.clamp(udp_model(torch.cat([hazy, depth], dim=1))[0][2], 0, 1)
                mask = build_teacher_mask(a0, udp, gt, args)
            pred = torch.clamp(student(hazy, depth)[2], 0, 1)
            gt_loss = l1_weighted(pred, gt)
            teacher_loss = l1_weighted(pred, udp, mask)
            preserve_loss = l1_weighted(pred, a0, 1.0 - mask)
            loss = args.w_gt * gt_loss + args.w_teacher * teacher_loss + args.w_preserve * preserve_loss
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], args.grad_clip_norm)
            optimizer.step()
            if ema_student is not None:
                ema_update(ema_student, student, args.ema_decay)
            sums["loss"] += float(loss.detach().item())
            sums["gt"] += float(gt_loss.detach().item())
            sums["teacher"] += float(teacher_loss.detach().item())
            sums["preserve"] += float(preserve_loss.detach().item())
            sums["mask"] += float(mask.detach().mean().item())
            count += 1
            if (batch_idx + 1) % args.print_freq == 0:
                print(
                    f"epoch={epoch} iter={batch_idx+1}/{len(loader)} "
                    f"loss={sums['loss']/count:.6f} mask={sums['mask']/count:.6f}",
                    flush=True,
                )
        row = {
            "epoch": epoch,
            "loss": sums["loss"] / max(1, count),
            "gt_loss": sums["gt"] / max(1, count),
            "teacher_loss": sums["teacher"] / max(1, count),
            "preserve_loss": sums["preserve"] / max(1, count),
            "mask_mean": sums["mask"] / max(1, count),
        }
        status_rows.append(row)
        if epoch % 5 == 0:
            torch.save({"model": student.state_dict()}, ckpt_dir / f"model_{epoch}.pkl")
            if ema_student is not None:
                torch.save({"model": ema_student.state_dict()}, ckpt_dir / f"model_{epoch}_ema.pkl")
        print("EPOCH_SUMMARY " + json.dumps(row, sort_keys=True), flush=True)
    torch.save({"model": student.state_dict()}, ckpt_dir / "Final.pkl")
    if ema_student is not None:
        torch.save({"model": ema_student.state_dict()}, ckpt_dir / "Final_ema.pkl")

    payload = {
        "route": "ConvIR-Dehaze-v1.9-ConditionalTeacherGuided",
        "stage": "conditional teacher-guided student train",
        "locked_test_touched": False,
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_dir": str(ckpt_dir),
        "udp_checkpoint_meta": udp_meta,
        "rows": status_rows,
        "decision": "TRAIN_COMPLETE_PENDING_INTERNAL_EVAL",
    }
    (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("V19_CONDITIONAL_STUDENT_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
