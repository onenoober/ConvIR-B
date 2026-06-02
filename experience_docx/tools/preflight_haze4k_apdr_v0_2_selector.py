import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
sys.path.insert(0, str(ITS_ROOT))

from data.data_load import DeblurDataset, test_dataloader, train_dataloader
from models.APDRConvIR import build_apdr_net
from models.ConvIR import build_net as build_convir_net


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")
    return input_img, h, w


def crop_to(tensor, h, w):
    return tensor[:, :, :h, :w]


def full_item(model):
    for item in model._last_apdr_tensors or []:
        if item.get("scale") == "full":
            return item
    raise RuntimeError("Missing APDR full-scale selector tensors.")


def build_models(args, device):
    state = load_model_state(args.checkpoint, device)
    original = build_convir_net("base", "Haze4K", "original").to(device).eval()
    original.load_state_dict(state, strict=True)

    apdr = build_apdr_net(
        "base",
        "Haze4K",
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=args.apdr_residual_max,
        apdr_gate_max=args.apdr_gate_max,
        apdr_gate_init=args.apdr_gate_init,
        apdr_force_zero_gate=True,
        apdr_active_scales="full",
        apdr_selector_mode="v0_2",
    ).to(device)
    result = apdr.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("APDR_")]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Unexpected APDR load result: missing={missing}, unexpected={unexpected}"
        )
    return original, apdr, {"missing": missing, "unexpected": unexpected}


def configure_selector_only(apdr):
    trainable = []
    frozen = []
    for name, param in apdr.named_parameters():
        is_full_selector = name.startswith("APDR_1.") and "residual_head" not in name
        param.requires_grad = is_full_selector
        if param.requires_grad:
            trainable.append((name, param))
        else:
            frozen.append((name, param))
    if not trainable:
        raise RuntimeError("No v0.2 selector parameters are trainable.")
    return trainable, frozen


def set_selector_train_mode(apdr):
    apdr.eval()
    for name, module in apdr.named_modules():
        if name.startswith("APDR_1"):
            module.train()


def set_selector_eval_mode(apdr):
    apdr.eval()


def tensor_quantile(values, q):
    if not values:
        raise ValueError("Cannot compute quantile from an empty list.")
    tensor = torch.tensor(values, dtype=torch.float32)
    return torch.quantile(tensor, q).item()


def sample_flat_tensor(flat, sample_count):
    if sample_count <= 0 or flat.numel() <= sample_count:
        return flat
    idx = torch.linspace(0, flat.numel() - 1, steps=sample_count).long()
    return flat[idx]


def compute_calibration(apdr, args, device):
    dataset = DeblurDataset(str(Path(args.data_dir) / "train"), "Haze4K", transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    rmse_values = []
    pixel_samples = []
    set_selector_eval_mode(apdr)
    with torch.no_grad():
        for idx, (input_img, label_img) in enumerate(loader):
            if args.calibration_images > 0 and idx >= args.calibration_images:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            apdr(padded)
            anchor = crop_to(full_item(apdr)["anchor"], h, w)
            error = anchor - label_img
            rmse_values.append(torch.sqrt(error.square().mean()).item())
            error_map = error.abs().mean(dim=1, keepdim=True).flatten().detach().cpu()
            pixel_samples.append(sample_flat_tensor(error_map, args.pixel_samples_per_image))
            if (idx + 1) % args.progress_freq == 0:
                print(f"calibration {idx + 1}/{len(loader)}", flush=True)

    pixels = torch.cat(pixel_samples) if pixel_samples else torch.empty(0)
    if pixels.numel() == 0:
        raise RuntimeError("No pixel samples collected for selector calibration.")
    q70_pixel = torch.quantile(pixels.float(), 0.70).item()
    q90_pixel = torch.quantile(pixels.float(), 0.90).item()
    spatial_tau = args.spatial_tau
    if spatial_tau <= 0:
        spatial_tau = max(q90_pixel - q70_pixel, 1e-4)
    return {
        "train_image_count": len(rmse_values),
        "pixel_sample_count": int(pixels.numel()),
        "rmse_q50_train": tensor_quantile(rmse_values, 0.50),
        "rmse_q90_train": tensor_quantile(rmse_values, 0.90),
        "pixel_error_q70_train": q70_pixel,
        "pixel_error_q90_train": q90_pixel,
        "spatial_tau": spatial_tau,
    }


def make_targets(anchor, label, calibration):
    error = anchor.detach() - label
    mse_per_image = error.square().flatten(1).mean(dim=1)
    rmse = torch.sqrt(mse_per_image.clamp_min(1e-12))
    denom = max(
        calibration["rmse_q90_train"] - calibration["rmse_q50_train"],
        1e-8,
    )
    hard_target = ((rmse - calibration["rmse_q50_train"]) / denom).clamp(0.0, 1.0)
    pixel_error = error.abs().mean(dim=1, keepdim=True)
    spatial_target = torch.sigmoid(
        (pixel_error - calibration["pixel_error_q70_train"]) / calibration["spatial_tau"]
    )
    return hard_target.detach(), spatial_target.detach(), rmse.detach()


def selector_rank_loss(scores, targets, margin):
    if scores.numel() < 2:
        return scores.new_zeros(())
    order = torch.argsort(targets)
    k = max(1, scores.numel() // 4)
    easy_idx = order[:k]
    hard_idx = order[-k:]
    if torch.allclose(targets[hard_idx].mean(), targets[easy_idx].mean()):
        return scores.new_zeros(())
    return torch.relu(margin - scores[hard_idx].mean() + scores[easy_idx].mean())


def evaluate_selector_loss(apdr, args, device, calibration):
    dataset = DeblurDataset(str(Path(args.data_dir) / "train"), "Haze4K", transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    hard_losses = []
    spatial_losses = []
    set_selector_eval_mode(apdr)
    with torch.no_grad():
        for idx, (input_img, label_img) in enumerate(loader):
            if args.loss_eval_images > 0 and idx >= args.loss_eval_images:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            apdr(padded)
            item = full_item(apdr)
            anchor = crop_to(item["anchor"], h, w)
            hard_target, spatial_target, _ = make_targets(anchor, label_img, calibration)
            hard_logits = item["global_logits"].view(-1)
            spatial_logits = crop_to(item["spatial_logits"], h, w)
            hard_losses.append(
                F.binary_cross_entropy_with_logits(hard_logits, hard_target).item()
            )
            spatial_losses.append(
                F.binary_cross_entropy_with_logits(spatial_logits, spatial_target).item()
            )
    return {
        "image_count": len(hard_losses),
        "hard_bce": statistics.mean(hard_losses),
        "spatial_bce": statistics.mean(spatial_losses),
    }


def train_selector(apdr, args, device, calibration):
    trainable, frozen = configure_selector_only(apdr)
    optimizer = torch.optim.Adam(
        [param for _, param in trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    loader = train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        data="Haze4K",
        use_transform=True,
    )
    history = [
        {
            "epoch": 0,
            "train_subset": evaluate_selector_loss(apdr, args, device, calibration),
        }
    ]
    for epoch in range(1, args.epochs + 1):
        set_selector_train_mode(apdr)
        sums = {
            "loss": 0.0,
            "hard_bce": 0.0,
            "spatial_bce": 0.0,
            "rank": 0.0,
            "mask": 0.0,
        }
        count = 0
        for batch_idx, (input_img, label_img) in enumerate(loader):
            if args.train_batches_per_epoch > 0 and batch_idx >= args.train_batches_per_epoch:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad(set_to_none=True)
            apdr(input_img)
            item = full_item(apdr)
            anchor = item["anchor"]
            hard_target, spatial_target, _ = make_targets(anchor, label_img, calibration)

            hard_logits = item["global_logits"].view(-1)
            spatial_logits = item["spatial_logits"]
            hard_scores = torch.sigmoid(hard_logits)
            hard_bce = F.binary_cross_entropy_with_logits(hard_logits, hard_target)
            spatial_bce = F.binary_cross_entropy_with_logits(spatial_logits, spatial_target)
            rank = selector_rank_loss(hard_scores, hard_target, args.rank_margin)
            mask = (item["global_gate_unit"] * item["spatial_gate_unit"]).mean()
            loss = (
                args.hard_selector_lambda * hard_bce
                + args.spatial_selector_lambda * spatial_bce
                + args.rank_lambda * rank
                + args.gate_lambda * mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for _, param in trainable], args.grad_clip)
            optimizer.step()

            sums["loss"] += loss.item()
            sums["hard_bce"] += hard_bce.item()
            sums["spatial_bce"] += spatial_bce.item()
            sums["rank"] += rank.item()
            sums["mask"] += mask.item()
            count += 1
            if (batch_idx + 1) % args.progress_freq == 0:
                print(
                    "epoch %03d iter %04d loss=%.6f hard_bce=%.6f spatial_bce=%.6f"
                    % (
                        epoch,
                        batch_idx + 1,
                        sums["loss"] / count,
                        sums["hard_bce"] / count,
                        sums["spatial_bce"] / count,
                    ),
                    flush=True,
                )

        if count == 0:
            raise RuntimeError("No selector training batches were processed.")
        epoch_row = {
            "epoch": epoch,
            "train": {key: value / count for key, value in sums.items()},
            "train_subset": evaluate_selector_loss(apdr, args, device, calibration),
        }
        print(json.dumps(epoch_row, indent=2), flush=True)
        history.append(epoch_row)
    return history, {
        "trainable_tensor_count": len(trainable),
        "trainable_param_count": sum(param.numel() for _, param in trainable),
        "frozen_param_count": sum(param.numel() for _, param in frozen),
        "trainable_tensors_first40": [name for name, _ in trainable[:40]],
    }


def psnr_from_prediction(pred, label):
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + end - 1) / 2.0 + 1.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def auc_score(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def mean_or_none(values):
    return statistics.mean(values) if values else None


def evaluate_test(original, apdr, args, device, calibration, csv_path):
    loader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)
    rows = []
    max_abs_diff = 0.0
    spatial_bces = []
    set_selector_eval_mode(apdr)
    original.eval()
    with torch.no_grad(), csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fieldnames = [
            "name",
            "a0_psnr",
            "h_img",
            "s_pixel_mean",
            "mask_mean",
            "rmse0",
            "spatial_bce",
            "zero_residual_max_abs_diff_vs_a0",
        ]
        writer.writerow(fieldnames)
        for idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            a0 = crop_to(original(padded)[2], h, w)
            apdr_out = crop_to(apdr(padded)[2], h, w)
            item = full_item(apdr)
            anchor = crop_to(item["anchor"], h, w)
            hard_target, spatial_target, rmse = make_targets(anchor, label_img, calibration)
            spatial_logits = crop_to(item["spatial_logits"], h, w)
            spatial_bce = F.binary_cross_entropy_with_logits(
                spatial_logits,
                spatial_target,
            ).item()
            diff = (apdr_out - a0).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            h_img = item["global_gate_unit"].view(-1).item()
            s_pixel_mean = crop_to(item["spatial_gate_unit"], h, w).mean().item()
            mask_mean = (h_img * crop_to(item["spatial_gate_unit"], h, w)).mean().item()
            row = {
                "name": name[0],
                "a0_psnr": psnr_from_prediction(a0, label_img),
                "h_img": h_img,
                "s_pixel_mean": s_pixel_mean,
                "mask_mean": mask_mean,
                "rmse0": rmse.view(-1).item(),
                "spatial_bce": spatial_bce,
                "zero_residual_max_abs_diff_vs_a0": diff,
            }
            rows.append(row)
            spatial_bces.append(spatial_bce)
            writer.writerow([row[key] for key in fieldnames])
            if (idx + 1) % args.progress_freq == 0:
                print(f"test_eval {idx + 1}/{len(loader)}", flush=True)

    by_psnr = sorted(rows, key=lambda row: row["a0_psnr"])
    count = len(by_psnr)
    bucket_count = max(1, count // 4)
    hard = by_psnr[:bucket_count]
    easy = by_psnr[-bucket_count:]
    hard_mean = mean_or_none([row["h_img"] for row in hard])
    easy_mean = mean_or_none([row["h_img"] for row in easy])
    ratio = hard_mean / max(easy_mean, 1e-12) if hard_mean is not None else None
    auc = auc_score([row["h_img"] for row in hard], [row["h_img"] for row in easy])
    spearman_value = spearman(
        [row["h_img"] for row in rows],
        [row["a0_psnr"] for row in rows],
    )
    return rows, {
        "count": count,
        "hard_bottom_25pct": {
            "count": len(hard),
            "a0_psnr_range": [hard[0]["a0_psnr"], hard[-1]["a0_psnr"]],
            "mean_h_img": hard_mean,
        },
        "easy_top_25pct": {
            "count": len(easy),
            "a0_psnr_range": [easy[0]["a0_psnr"], easy[-1]["a0_psnr"]],
            "mean_h_img": easy_mean,
        },
        "hard_easy_h_img_ratio": ratio,
        "spearman_h_img_vs_a0_psnr": spearman_value,
        "auc_hard_vs_easy_by_h_img": auc,
        "strong_reference_count": len(easy),
        "strong_reference_mean_h_img": easy_mean,
        "mean_spatial_bce": statistics.mean(spatial_bces),
        "zero_residual_output": {
            "max_abs_diff_vs_a0": max_abs_diff,
            "threshold": args.zero_diff_threshold,
            "pass": max_abs_diff < args.zero_diff_threshold,
        },
    }


def write_history_csv(history, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_hard_bce", "train_spatial_bce", "subset_hard_bce", "subset_spatial_bce"])
        for row in history:
            train = row.get("train", {})
            subset = row["train_subset"]
            writer.writerow(
                [
                    row["epoch"],
                    train.get("loss"),
                    train.get("hard_bce"),
                    train.get("spatial_bce"),
                    subset["hard_bce"],
                    subset["spatial_bce"],
                ]
            )


def build_gate(test_summary, history):
    first_spatial = history[0]["train_subset"]["spatial_bce"]
    last_spatial = history[-1]["train_subset"]["spatial_bce"]
    checks = {
        "hard_easy_h_img_ratio": {
            "observed": test_summary["hard_easy_h_img_ratio"],
            "required": ">= 3.0",
            "pass": test_summary["hard_easy_h_img_ratio"] is not None
            and test_summary["hard_easy_h_img_ratio"] >= 3.0,
        },
        "spearman_h_img_vs_a0_psnr": {
            "observed": test_summary["spearman_h_img_vs_a0_psnr"],
            "required": "<= -0.45",
            "pass": test_summary["spearman_h_img_vs_a0_psnr"] is not None
            and test_summary["spearman_h_img_vs_a0_psnr"] <= -0.45,
        },
        "auc_hard_vs_easy_by_h_img": {
            "observed": test_summary["auc_hard_vs_easy_by_h_img"],
            "required": ">= 0.75",
            "pass": test_summary["auc_hard_vs_easy_by_h_img"] is not None
            and test_summary["auc_hard_vs_easy_by_h_img"] >= 0.75,
        },
        "strong_reference_mean_h_img": {
            "observed": test_summary["strong_reference_mean_h_img"],
            "required": "<= 0.05",
            "pass": test_summary["strong_reference_mean_h_img"] is not None
            and test_summary["strong_reference_mean_h_img"] <= 0.05,
        },
        "spatial_bce_decreased": {
            "observed": {"first": first_spatial, "last": last_spatial},
            "required": "last < first",
            "pass": last_spatial < first_spatial,
        },
        "zero_residual_output": {
            "observed": test_summary["zero_residual_output"]["max_abs_diff_vs_a0"],
            "required": f"< {test_summary['zero_residual_output']['threshold']}",
            "pass": test_summary["zero_residual_output"]["pass"],
        },
    }
    return {
        "stage": "APDR-v0.2 selector-only preflight",
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_2_selector_seed3407")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--calibration_images", type=int, default=0)
    parser.add_argument("--pixel_samples_per_image", type=int, default=2048)
    parser.add_argument("--loss_eval_images", type=int, default=128)
    parser.add_argument("--spatial_tau", type=float, default=0.0)
    parser.add_argument("--hard_selector_lambda", type=float, default=1.0)
    parser.add_argument("--spatial_selector_lambda", type=float, default=1.0)
    parser.add_argument("--rank_lambda", type=float, default=0.10)
    parser.add_argument("--gate_lambda", type=float, default=0.002)
    parser.add_argument("--rank_margin", type=float, default=0.20)
    parser.add_argument("--apdr_residual_max", type=float, default=0.04)
    parser.add_argument("--apdr_gate_max", type=float, default=0.5)
    parser.add_argument("--apdr_gate_init", type=float, default=0.01)
    parser.add_argument("--zero_diff_threshold", type=float, default=1e-6)
    parser.add_argument("--progress_freq", type=int, default=100)
    parser.add_argument("--fail_on_gate", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, apdr, checkpoint_load = build_models(args, device)
    print("checkpoint loaded", json.dumps(checkpoint_load), flush=True)

    calibration = compute_calibration(apdr, args, device)
    print("calibration", json.dumps(calibration, indent=2), flush=True)

    history, train_scope = train_selector(apdr, args, device, calibration)

    per_image_csv = output_dir / f"selector_per_image_{args.tag}.csv"
    _, test_summary = evaluate_test(original, apdr, args, device, calibration, per_image_csv)
    gate = build_gate(test_summary, history)

    history_csv = output_dir / f"selector_history_{args.tag}.csv"
    write_history_csv(history, history_csv)

    result = {
        "stage": "apdr_v0_2_selector_preflight",
        "tag": args.tag,
        "seed": args.seed,
        "device": str(device),
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "apdr_config": {
            "selector_mode": "v0_2",
            "active_scales": "full",
            "residual_forced_zero": True,
            "residual_head_trainable": False,
            "gate_init": args.apdr_gate_init,
            "gate_max": args.apdr_gate_max,
            "residual_max": args.apdr_residual_max,
        },
        "loss_config": {
            "hard_selector_lambda": args.hard_selector_lambda,
            "spatial_selector_lambda": args.spatial_selector_lambda,
            "rank_lambda": args.rank_lambda,
            "gate_lambda": args.gate_lambda,
            "rank_margin": args.rank_margin,
        },
        "checkpoint_load": checkpoint_load,
        "calibration": calibration,
        "train_scope": train_scope,
        "history": history,
        "test_summary": test_summary,
        "gate": gate,
        "artifacts": {
            "per_image_csv": str(per_image_csv),
            "history_csv": str(history_csv),
        },
        "pass": gate["pass"],
    }
    summary_json = output_dir / f"selector_summary_{args.tag}.json"
    gate_json = output_dir / f"gate_{args.tag}.json"
    summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    gate_json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.fail_on_gate and not gate["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
