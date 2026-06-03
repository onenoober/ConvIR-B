import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())

from data.data_load import DeblurDataset
from models.APDRConvIR import build_apdr_net


def psnr(pred, target):
    mse = f.mse_loss(pred, target).clamp_min(1e-12)
    return (10 * torch.log10(1 / mse)).item()


def correlation(x, y, weight=None):
    x = x.flatten()
    y = y.flatten()
    if weight is not None:
        weight = weight.flatten().clamp_min(0)
        keep = weight > 0
        x = x[keep]
        y = y[keep]
        weight = weight[keep]
    if x.numel() < 2:
        return None
    if weight is None:
        x = x - x.mean()
        y = y - y.mean()
        denom = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denom.item() == 0:
            return None
        return (x * y).sum().div(denom).item()
    weight = weight / weight.sum().clamp_min(1e-12)
    mean_x = (weight * x).sum()
    mean_y = (weight * y).sum()
    xc = x - mean_x
    yc = y - mean_y
    denom = (weight * xc.square()).sum().sqrt() * (weight * yc.square()).sum().sqrt()
    if denom.item() == 0:
        return None
    return (weight * xc * yc).sum().div(denom).item()


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        return f.pad(input_img, (0, padw, 0, padh), "reflect"), h, w
    return input_img, h, w


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def configure_residual_only(model):
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("APDR_1.") and (
            ".residual_body." in name or ".residual_head." in name
        )
        if param.requires_grad:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No APDR full-scale residual parameters are trainable.")
    model.eval()
    model.APDR_1.train()
    return trainable


def build_loader(data_dir, count, num_workers):
    train_dir = Path(data_dir) / "train"
    dataset = DeblurDataset(str(train_dir), "Haze4K", transform=None, is_test=True)
    if count > 0:
        dataset = Subset(dataset, list(range(min(count, len(dataset)))))
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)


def collect_eval(model, loader, device, residual_max):
    rows = []
    total_loss_num = 0.0
    total_loss_den = 0.0
    corrs = []
    with torch.no_grad():
        for input_img, label_img, name in loader:
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            model(padded)
            full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
            anchor = full["anchor"][:, :, :h, :w].clamp(0, 1)
            m_safe = full["gate"][:, :, :h, :w].clamp(0, 1)
            residual_raw = full["residual_raw"][:, :, :h, :w]
            delta_star = (label_img - anchor).clamp(-residual_max, residual_max)
            weight = m_safe.expand_as(residual_raw)
            diff = (residual_raw - delta_star).abs()
            total_loss_num += (weight * diff).sum().item()
            total_loss_den += weight.sum().item()
            corr = correlation(residual_raw.detach().cpu(), delta_star.detach().cpu(), weight.detach().cpu())
            if corr is not None:
                corrs.append(corr)
            output = (anchor + m_safe * residual_raw).clamp(0, 1)
            oracle = (anchor + m_safe * delta_star).clamp(0, 1)
            anchor_psnr = psnr(anchor, label_img)
            output_psnr = psnr(output, label_img)
            oracle_psnr = psnr(oracle, label_img)
            rows.append(
                {
                    "name": name[0],
                    "anchor_psnr": anchor_psnr,
                    "output_psnr": output_psnr,
                    "oracle_psnr": oracle_psnr,
                    "output_gain": output_psnr - anchor_psnr,
                    "oracle_gain": oracle_psnr - anchor_psnr,
                    "corr": corr,
                    "m_safe_mean": m_safe.mean().item(),
                    "residual_raw_abs_mean": residual_raw.abs().mean().item(),
                    "delta_star_abs_mean": delta_star.abs().mean().item(),
                }
            )
    weighted_l1 = total_loss_num / max(total_loss_den, 1e-12)
    return rows, weighted_l1, (statistics.mean(corrs) if corrs else None)


def summarize(rows, initial_loss, final_loss, final_corr):
    gains = [row["output_gain"] for row in rows]
    oracle_gains = [row["oracle_gain"] for row in rows]
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    oracle_positive = [row for row in rows if row["oracle_gain"] > 1e-6]
    recovery = sum(row["output_gain"] for row in oracle_positive) / max(
        sum(row["oracle_gain"] for row in oracle_positive),
        1e-12,
    )
    loss_drop = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    return {
        "count": len(rows),
        "initial_weighted_delta_l1": initial_loss,
        "final_weighted_delta_l1": final_loss,
        "loss_drop_fraction": loss_drop,
        "mean_output_gain": statistics.mean(gains),
        "mean_oracle_gain": statistics.mean(oracle_gains),
        "oracle_gain_recovery": recovery,
        "mean_corr_residual_raw_delta_star": final_corr,
        "hard_bottom25_output_gain": statistics.mean(row["output_gain"] for row in hard),
        "hard_bottom25_oracle_gain": statistics.mean(row["oracle_gain"] for row in hard),
        "mean_m_safe": statistics.mean(row["m_safe_mean"] for row in rows),
        "mean_residual_raw_abs": statistics.mean(row["residual_raw_abs_mean"] for row in rows),
        "mean_delta_star_abs": statistics.mean(row["delta_star_abs_mean"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_delta_learnability_32")
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--progress_freq", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=args.residual_max,
        apdr_gate_max=1.0,
        apdr_gate_init=0.01,
        apdr_force_zero_gate=False,
        apdr_active_scales="full",
        apdr_selector_mode="v0_2r",
        apdr_residual_capacity="linear",
    ).to(device)
    model.load_state_dict(load_model_state(args.selector_checkpoint, device), strict=True)
    trainable = configure_residual_only(model)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loader = build_loader(args.data_dir, args.num_images, args.num_workers)
    eval_loader = build_loader(args.data_dir, args.num_images, args.num_workers)

    initial_rows, initial_loss, initial_corr = collect_eval(model, eval_loader, device, args.residual_max)
    history = [
        {
            "step": 0,
            "weighted_delta_l1": initial_loss,
            "corr": initial_corr,
            "mean_output_gain": statistics.mean(row["output_gain"] for row in initial_rows),
        }
    ]
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            input_img, label_img, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            input_img, label_img, _ = next(iterator)
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        padded, h, w = pad_to_factor(input_img)
        optimizer.zero_grad()
        model(padded)
        full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
        anchor = full["anchor"][:, :, :h, :w].detach()
        m_safe = full["gate"][:, :, :h, :w].detach().clamp(0, 1)
        residual_raw = full["residual_raw"][:, :, :h, :w]
        delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max).detach()
        weight = m_safe.expand_as(residual_raw)
        loss = (weight * (residual_raw - delta_star).abs()).sum() / weight.sum().clamp_min(1e-12)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
        optimizer.step()

        if args.progress_freq > 0 and step % args.progress_freq == 0:
            rows, eval_loss, eval_corr = collect_eval(model, eval_loader, device, args.residual_max)
            mean_gain = statistics.mean(row["output_gain"] for row in rows)
            history.append(
                {
                    "step": step,
                    "weighted_delta_l1": eval_loss,
                    "corr": eval_corr,
                    "mean_output_gain": mean_gain,
                }
            )
            print(
                f"step={step} weighted_delta_l1={eval_loss:.6f} "
                f"corr={eval_corr if eval_corr is not None else float('nan'):.4f} "
                f"mean_gain={mean_gain:.4f}",
                flush=True,
            )

    final_rows, final_loss, final_corr = collect_eval(model, eval_loader, device, args.residual_max)
    summary = summarize(final_rows, initial_loss, final_loss, final_corr)
    summary["initial_corr_residual_raw_delta_star"] = initial_corr
    summary["history"] = history
    summary["args"] = vars(args)
    checks = {
        "loss_drop_fraction": {
            "observed": summary["loss_drop_fraction"],
            "required": ">= 0.30",
            "pass": summary["loss_drop_fraction"] >= 0.30,
        },
        "oracle_gain_recovery": {
            "observed": summary["oracle_gain_recovery"],
            "required": ">= 0.50",
            "pass": summary["oracle_gain_recovery"] >= 0.50,
        },
        "corr_residual_raw_delta_star": {
            "observed": summary["mean_corr_residual_raw_delta_star"],
            "required": ">= 0.40",
            "pass": (
                summary["mean_corr_residual_raw_delta_star"] is not None
                and summary["mean_corr_residual_raw_delta_star"] >= 0.40
            ),
        },
        "hard_train_psnr_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": ">= +0.30 dB",
            "pass": summary["hard_bottom25_output_gain"] >= 0.30,
        },
    }
    result = {
        "stage": "APDR-v0.2RC delta learnability overfit",
        "summary": summary,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"delta_learnability_{args.tag}.json"
    csv_path = output_dir / f"delta_learnability_per_image_{args.tag}.csv"
    history_path = output_dir / f"delta_learnability_history_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {history_path}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
