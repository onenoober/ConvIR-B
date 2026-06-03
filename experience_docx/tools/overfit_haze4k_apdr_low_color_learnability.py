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
import torch.nn as nn
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


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


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


def build_loader(data_dir, count, num_workers, shuffle):
    train_dir = Path(data_dir) / "train"
    dataset = DeblurDataset(str(train_dir), "Haze4K", transform=None, is_test=True)
    if count > 0:
        dataset = Subset(dataset, list(range(min(count, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class LowColorResidualProbe(nn.Module):
    def __init__(self, residual_max=0.04, low_size=32, hidden=32):
        super().__init__()
        self.residual_max = float(residual_max)
        self.low_size = int(low_size)
        in_channels = 3 + 3 + 3 + 1
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
        )
        self.color_pool = nn.AdaptiveAvgPool2d(1)
        self.color_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 6, kernel_size=1),
        )
        nn.init.zeros_(self.low_branch[-1].weight)
        nn.init.zeros_(self.low_branch[-1].bias)
        nn.init.zeros_(self.color_head[-1].weight)
        nn.init.zeros_(self.color_head[-1].bias)

    def forward(self, hazy, anchor, m_safe):
        features = torch.cat([hazy, anchor, hazy - anchor, m_safe], dim=1)
        low_features = f.adaptive_avg_pool2d(features, (self.low_size, self.low_size))
        low_raw = self.residual_max * torch.tanh(self.low_branch(low_features))
        low_delta = f.interpolate(low_raw, size=anchor.shape[2:], mode="bilinear", align_corners=False)

        color_params = self.color_head(self.color_pool(features)).flatten(1)
        scale = 1.0 + 0.10 * torch.tanh(color_params[:, 0:3]).view(-1, 3, 1, 1)
        bias = self.residual_max * torch.tanh(color_params[:, 3:6]).view(-1, 3, 1, 1)
        color_delta = (scale * anchor + bias - anchor).clamp(-self.residual_max, self.residual_max)

        return (low_delta + color_delta).clamp(-self.residual_max, self.residual_max), low_delta, color_delta


def build_apdr_model(selector_checkpoint, residual_max, device):
    model = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=residual_max,
        apdr_gate_max=1.0,
        apdr_gate_init=0.01,
        apdr_force_zero_gate=False,
        apdr_active_scales="full",
        apdr_selector_mode="v0_2r",
        apdr_residual_capacity="linear",
    ).to(device)
    model.load_state_dict(load_model_state(selector_checkpoint, device), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def frozen_apdr_tensors(model, input_img):
    padded, h, w = pad_to_factor(input_img)
    with torch.no_grad():
        model(padded)
        full = [item for item in model._last_apdr_tensors if item.get("scale") == "full"][0]
        anchor = full["anchor"][:, :, :h, :w].detach().clamp(0, 1)
        m_safe = full["gate"][:, :, :h, :w].detach().clamp(0, 1)
    return anchor, m_safe


def collect_eval(apdr_model, probe, loader, device, residual_max):
    rows = []
    total_loss_num = 0.0
    total_loss_den = 0.0
    corrs = []
    with torch.no_grad():
        for input_img, label_img, name in loader:
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            anchor, m_safe = frozen_apdr_tensors(apdr_model, input_img)
            pred_delta, low_delta, color_delta = probe(input_img, anchor, m_safe)
            delta_star = (label_img - anchor).clamp(-residual_max, residual_max)
            weight = m_safe.expand_as(pred_delta)
            diff = (pred_delta - delta_star).abs()
            total_loss_num += (weight * diff).sum().item()
            total_loss_den += weight.sum().item()
            corr = correlation(pred_delta.cpu(), delta_star.cpu(), weight.cpu())
            if corr is not None:
                corrs.append(corr)
            output = (anchor + m_safe * pred_delta).clamp(0, 1)
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
                    "pred_delta_abs_mean": pred_delta.abs().mean().item(),
                    "low_delta_abs_mean": low_delta.abs().mean().item(),
                    "color_delta_abs_mean": color_delta.abs().mean().item(),
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
    easy = ordered[3 * len(ordered) // 4 :]
    oracle_positive = [row for row in rows if row["oracle_gain"] > 1e-6]
    recovery = sum(row["output_gain"] for row in oracle_positive) / max(
        sum(row["oracle_gain"] for row in oracle_positive),
        1e-12,
    )
    loss_drop = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    strong_cut = percentile([row["anchor_psnr"] for row in rows], 75)
    strong = [row for row in rows if row["anchor_psnr"] >= strong_cut]
    return {
        "count": len(rows),
        "initial_weighted_delta_l1": initial_loss,
        "final_weighted_delta_l1": final_loss,
        "loss_drop_fraction": loss_drop,
        "mean_output_gain": statistics.mean(gains),
        "mean_oracle_gain": statistics.mean(oracle_gains),
        "oracle_gain_recovery": recovery,
        "mean_corr_pred_delta_star": final_corr,
        "hard_bottom25_output_gain": statistics.mean(row["output_gain"] for row in hard),
        "hard_bottom25_oracle_gain": statistics.mean(row["oracle_gain"] for row in hard),
        "easy_top25_output_gain": statistics.mean(row["output_gain"] for row in easy),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row["output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["output_gain"] <= -0.20 for row in rows),
        "mean_m_safe": statistics.mean(row["m_safe_mean"] for row in rows),
        "mean_pred_delta_abs": statistics.mean(row["pred_delta_abs_mean"] for row in rows),
        "mean_low_delta_abs": statistics.mean(row["low_delta_abs_mean"] for row in rows),
        "mean_color_delta_abs": statistics.mean(row["color_delta_abs_mean"] for row in rows),
        "mean_delta_star_abs": statistics.mean(row["delta_star_abs_mean"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_low_color_learnability_32")
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--progress_freq", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    probe = LowColorResidualProbe(
        residual_max=args.residual_max,
        low_size=args.low_size,
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=True)
    eval_loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=False)

    initial_rows, initial_loss, initial_corr = collect_eval(
        apdr_model,
        probe,
        eval_loader,
        device,
        args.residual_max,
    )
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
        anchor, m_safe = frozen_apdr_tensors(apdr_model, input_img)
        delta_star = (label_img - anchor).clamp(-args.residual_max, args.residual_max).detach()
        pred_delta, _, _ = probe(input_img, anchor, m_safe)
        weight = m_safe.expand_as(pred_delta)
        loss = (weight * (pred_delta - delta_star).abs()).sum() / weight.sum().clamp_min(1e-12)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip_norm)
        optimizer.step()

        if args.progress_freq > 0 and step % args.progress_freq == 0:
            rows, eval_loss, eval_corr = collect_eval(
                apdr_model,
                probe,
                eval_loader,
                device,
                args.residual_max,
            )
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

    final_rows, final_loss, final_corr = collect_eval(
        apdr_model,
        probe,
        eval_loader,
        device,
        args.residual_max,
    )
    summary = summarize(final_rows, initial_loss, final_loss, final_corr)
    summary["initial_corr_pred_delta_star"] = initial_corr
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
            "required": ">= 0.30",
            "pass": summary["oracle_gain_recovery"] >= 0.30,
        },
        "corr_pred_delta_star": {
            "observed": summary["mean_corr_pred_delta_star"],
            "required": ">= 0.35",
            "pass": (
                summary["mean_corr_pred_delta_star"] is not None
                and summary["mean_corr_pred_delta_star"] >= 0.35
            ),
        },
        "hard_train_psnr_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": ">= +0.20 dB",
            "pass": summary["hard_bottom25_output_gain"] >= 0.20,
        },
        "easy_train_psnr_gain": {
            "observed": summary["easy_top25_output_gain"],
            "required": ">= -0.010 dB",
            "pass": summary["easy_top25_output_gain"] >= -0.010,
        },
    }
    result = {
        "stage": "APDR-v0.3 low-color residual learnability overfit",
        "summary": summary,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"low_color_learnability_{args.tag}.json"
    csv_path = output_dir / f"low_color_learnability_per_image_{args.tag}.csv"
    history_path = output_dir / f"low_color_learnability_history_{args.tag}.csv"
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
