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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

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


class NamedDeblurDataset(DeblurDataset):
    def __init__(self, image_dir, data, resize=0):
        super().__init__(image_dir, data, transform=None, is_test=True)
        self.resize = int(resize)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        image = Image.open(Path(self.input_dir) / name).convert("RGB")
        label = Image.open(self._label_path(name)).convert("RGB")
        if self.resize > 0:
            image = TF.resize(image, [self.resize, self.resize])
            label = TF.resize(label, [self.resize, self.resize])
        return TF.to_tensor(image), TF.to_tensor(label), name


def named_loader(args, split, resize=0, batch_size=1, shuffle=False, num_workers=0):
    if resize <= 0 and batch_size != 1:
        raise ValueError("Unresized full-image loader requires batch_size=1.")
    dataset = NamedDeblurDataset(str(Path(args.data_dir) / split), "Haze4K", resize=resize)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


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
        apdr_selector_mode="v0_2r",
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


def configure_phase_a_global(apdr):
    trainable = []
    frozen = []
    for name, param in apdr.named_parameters():
        param.requires_grad = name.startswith("APDR_1.global_router")
        if param.requires_grad:
            trainable.append((name, param))
        else:
            frozen.append((name, param))
    if not trainable:
        raise RuntimeError("No v0.2R global-router parameters are trainable.")
    return trainable, frozen


def configure_phase_b_spatial(apdr):
    trainable_prefixes = (
        "APDR_1.image_context",
        "APDR_1.feature_context",
        "APDR_1.context",
        "APDR_1.spatial_gate_head",
    )
    trainable = []
    frozen = []
    for name, param in apdr.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)
        if param.requires_grad:
            trainable.append((name, param))
        else:
            frozen.append((name, param))
    if not trainable:
        raise RuntimeError("No v0.2R spatial selector parameters are trainable.")
    return trainable, frozen


def set_selector_eval_mode(apdr):
    apdr.eval()


def set_phase_train_mode(apdr, phase):
    apdr.eval()
    if phase == "global":
        apdr.APDR_1.global_router.train()
    elif phase == "spatial":
        apdr.APDR_1.image_context.train()
        apdr.APDR_1.feature_context.train()
        apdr.APDR_1.context.train()
        apdr.APDR_1.spatial_gate_head.train()
    else:
        raise ValueError(f"Unknown phase: {phase}")


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
    loader = named_loader(args, "train", resize=0, batch_size=1, shuffle=False, num_workers=0)
    records = []
    pixel_samples = []
    set_selector_eval_mode(apdr)
    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(loader):
            if args.calibration_images > 0 and idx >= args.calibration_images:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            apdr(padded)
            anchor = crop_to(full_item(apdr)["anchor"], h, w)
            error = anchor - label_img
            rmse = torch.sqrt(error.square().mean()).item()
            error_map = error.abs().mean(dim=1, keepdim=True).flatten().detach().cpu()
            pixel_samples.append(sample_flat_tensor(error_map, args.pixel_samples_per_image))
            records.append({"name": name[0], "rmse": rmse})
            if (idx + 1) % args.progress_freq == 0:
                print(f"calibration {idx + 1}/{len(loader)}", flush=True)

    if not records:
        raise RuntimeError("No train images collected for calibration.")
    pixels = torch.cat(pixel_samples) if pixel_samples else torch.empty(0)
    if pixels.numel() == 0:
        raise RuntimeError("No pixel samples collected for spatial calibration.")

    rmse_values = [row["rmse"] for row in records]
    rmse_q25 = tensor_quantile(rmse_values, 0.25)
    rmse_q50 = tensor_quantile(rmse_values, 0.50)
    rmse_q75 = tensor_quantile(rmse_values, 0.75)
    rmse_q90 = tensor_quantile(rmse_values, 0.90)
    denom = max(rmse_q90 - rmse_q50, 1e-8)
    by_name = {}
    for row in records:
        hard_soft = min(max((row["rmse"] - rmse_q50) / denom, 0.0), 1.0)
        hard_binary = None
        if row["rmse"] >= rmse_q75:
            hard_binary = 1.0
        elif row["rmse"] <= rmse_q25:
            hard_binary = 0.0
        by_name[row["name"]] = {
            "rmse": row["rmse"],
            "hard_soft": hard_soft,
            "hard_binary": hard_binary,
        }

    q70_pixel = torch.quantile(pixels.float(), 0.70).item()
    q90_pixel = torch.quantile(pixels.float(), 0.90).item()
    spatial_tau = args.spatial_tau
    if spatial_tau <= 0:
        spatial_tau = max(q90_pixel - q70_pixel, 1e-4)
    return {
        "summary": {
            "train_image_count": len(records),
            "pixel_sample_count": int(pixels.numel()),
            "rmse_q25_train": rmse_q25,
            "rmse_q50_train": rmse_q50,
            "rmse_q75_train": rmse_q75,
            "rmse_q90_train": rmse_q90,
            "pixel_error_q70_train": q70_pixel,
            "pixel_error_q90_train": q90_pixel,
            "spatial_tau": spatial_tau,
        },
        "by_name": by_name,
    }


def hard_targets_for_names(names, calibration, device):
    targets = [calibration["by_name"][name]["hard_soft"] for name in names]
    return torch.tensor(targets, dtype=torch.float32, device=device)


def hard_binary_for_names(names, calibration, device):
    values = []
    mask = []
    for name in names:
        binary = calibration["by_name"][name]["hard_binary"]
        if binary is None:
            values.append(0.0)
            mask.append(False)
        else:
            values.append(binary)
            mask.append(True)
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def make_spatial_target(anchor, label, calibration):
    error = anchor.detach() - label
    pixel_error = error.abs().mean(dim=1, keepdim=True)
    summary = calibration["summary"]
    return torch.sigmoid(
        (pixel_error - summary["pixel_error_q70_train"]) / summary["spatial_tau"]
    ).detach()


def focal_bce_with_logits(logits, targets, mask, gamma):
    if mask.sum().item() == 0:
        return logits.new_zeros(())
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
    prob = torch.sigmoid(selected_logits)
    pt = prob * selected_targets + (1.0 - prob) * (1.0 - selected_targets)
    return ((1.0 - pt).pow(gamma) * bce).mean()


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


def evaluate_train_subset(apdr, args, device, calibration):
    global_losses = []
    spatial_losses = []
    loader = named_loader(args, "train", resize=0, batch_size=1, shuffle=False, num_workers=0)
    set_selector_eval_mode(apdr)
    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(loader):
            if args.loss_eval_images > 0 and idx >= args.loss_eval_images:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            apdr(padded)
            item = full_item(apdr)
            hard_target = hard_targets_for_names(name, calibration, device)
            hard_logits = item["global_logits"].view(-1)
            global_losses.append(
                F.binary_cross_entropy_with_logits(hard_logits, hard_target).item()
            )
            anchor = crop_to(item["anchor"], h, w)
            spatial_logits = crop_to(item["spatial_logits"], h, w)
            spatial_target = make_spatial_target(anchor, label_img, calibration)
            spatial_losses.append(
                F.binary_cross_entropy_with_logits(spatial_logits, spatial_target).item()
            )
    return {
        "image_count": len(global_losses),
        "hard_bce": statistics.mean(global_losses),
        "spatial_bce": statistics.mean(spatial_losses),
    }


def train_global_router(apdr, args, device, calibration):
    trainable, frozen = configure_phase_a_global(apdr)
    optimizer = torch.optim.Adam(
        [param for _, param in trainable],
        lr=args.global_learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    loader = named_loader(
        args,
        "train",
        resize=args.global_resize,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_worker,
    )
    history = [
        {
            "phase": "global",
            "epoch": 0,
            "train_subset": evaluate_train_subset(apdr, args, device, calibration),
        }
    ]
    for epoch in range(1, args.global_epochs + 1):
        set_phase_train_mode(apdr, "global")
        sums = {"loss": 0.0, "hard_bce": 0.0, "focal_bce": 0.0, "rank": 0.0}
        count = 0
        for batch_idx, (input_img, label_img, names) in enumerate(loader):
            if args.global_train_batches_per_epoch > 0 and batch_idx >= args.global_train_batches_per_epoch:
                break
            input_img = input_img.to(device)
            optimizer.zero_grad(set_to_none=True)
            padded, _, _ = pad_to_factor(input_img)
            apdr(padded)
            item = full_item(apdr)
            hard_logits = item["global_logits"].view(-1)
            hard_target = hard_targets_for_names(names, calibration, device)
            binary_target, binary_mask = hard_binary_for_names(names, calibration, device)
            hard_bce = F.binary_cross_entropy_with_logits(hard_logits, hard_target)
            focal = focal_bce_with_logits(
                hard_logits,
                binary_target,
                binary_mask,
                args.focal_gamma,
            )
            rank = selector_rank_loss(hard_logits, hard_target, args.rank_margin)
            loss = (
                args.global_bce_lambda * hard_bce
                + args.global_focal_lambda * focal
                + args.global_rank_lambda * rank
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for _, param in trainable], args.grad_clip)
            optimizer.step()

            sums["loss"] += loss.item()
            sums["hard_bce"] += hard_bce.item()
            sums["focal_bce"] += focal.item()
            sums["rank"] += rank.item()
            count += 1
            if (batch_idx + 1) % args.progress_freq == 0:
                print(
                    "global epoch %03d iter %04d loss=%.6f hard_bce=%.6f"
                    % (epoch, batch_idx + 1, sums["loss"] / count, sums["hard_bce"] / count),
                    flush=True,
                )
        if count == 0:
            raise RuntimeError("No global-router training batches were processed.")
        row = {
            "phase": "global",
            "epoch": epoch,
            "train": {key: value / count for key, value in sums.items()},
            "train_subset": evaluate_train_subset(apdr, args, device, calibration),
        }
        print(json.dumps(row, indent=2), flush=True)
        history.append(row)
    return history, {
        "phase": "global",
        "trainable_tensor_count": len(trainable),
        "trainable_param_count": sum(param.numel() for _, param in trainable),
        "frozen_param_count": sum(param.numel() for _, param in frozen),
        "trainable_tensors": [name for name, _ in trainable],
    }


def collect_train_scores(apdr, args, device, calibration):
    loader = named_loader(args, "train", resize=0, batch_size=1, shuffle=False, num_workers=0)
    records = []
    set_selector_eval_mode(apdr)
    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(loader):
            if args.budget_calibration_images > 0 and idx >= args.budget_calibration_images:
                break
            input_img = input_img.to(device)
            padded, _, _ = pad_to_factor(input_img)
            apdr(padded)
            item = full_item(apdr)
            z_img = item["global_logits"].view(-1).item()
            info = calibration["by_name"][name[0]]
            records.append(
                {
                    "name": name[0],
                    "z_img": z_img,
                    "rmse": info["rmse"],
                    "hard_soft": info["hard_soft"],
                    "hard_binary": info["hard_binary"],
                }
            )
            if (idx + 1) % args.progress_freq == 0:
                print(f"budget_calibration {idx + 1}/{len(loader)}", flush=True)
    return records


def calibrate_budget(apdr, args, train_scores):
    if not train_scores:
        raise RuntimeError("No train scores available for budget calibration.")
    hard_scores = [row["z_img"] for row in train_scores if row["hard_binary"] == 1.0]
    easy_scores = [row["z_img"] for row in train_scores if row["hard_binary"] == 0.0]
    all_scores = [row["z_img"] for row in train_scores]
    if hard_scores and easy_scores and statistics.mean(hard_scores) > statistics.mean(easy_scores):
        hard_mean = statistics.mean(hard_scores)
        easy_mean = statistics.mean(easy_scores)
        tau = 0.5 * (hard_mean + easy_mean)
        temperature = max((hard_mean - easy_mean) / 4.0, args.budget_temperature_floor)
    else:
        hard_mean = statistics.mean(hard_scores) if hard_scores else None
        easy_mean = statistics.mean(easy_scores) if easy_scores else None
        tau = statistics.median(all_scores)
        temperature = args.budget_temperature_floor
    if hasattr(apdr.APDR_1, "set_global_budget_calibration"):
        apdr.APDR_1.set_global_budget_calibration(tau, temperature)

    def budget(z):
        return 1.0 / (1.0 + math.exp(-(z - tau) / max(temperature, 1e-4)))

    hard_budget = [budget(z) for z in hard_scores]
    easy_budget = [budget(z) for z in easy_scores]
    return {
        "train_score_count": len(train_scores),
        "tau_train": tau,
        "temperature_train": temperature,
        "temperature_floor": args.budget_temperature_floor,
        "hard_top25_score_mean": hard_mean,
        "easy_bottom25_score_mean": easy_mean,
        "hard_top25_budget_mean": statistics.mean(hard_budget) if hard_budget else None,
        "easy_bottom25_budget_mean": statistics.mean(easy_budget) if easy_budget else None,
        "hard_easy_budget_ratio_train": (
            statistics.mean(hard_budget) / max(statistics.mean(easy_budget), 1e-12)
            if hard_budget and easy_budget
            else None
        ),
    }


def train_spatial_gate(apdr, args, device, calibration):
    trainable, frozen = configure_phase_b_spatial(apdr)
    optimizer = torch.optim.Adam(
        [param for _, param in trainable],
        lr=args.spatial_learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    loader = train_dataloader(
        args.data_dir,
        batch_size=args.spatial_batch_size,
        num_workers=args.num_worker,
        data="Haze4K",
        use_transform=True,
    )
    history = [
        {
            "phase": "spatial",
            "epoch": 0,
            "train_subset": evaluate_train_subset(apdr, args, device, calibration),
        }
    ]
    for epoch in range(1, args.spatial_epochs + 1):
        set_phase_train_mode(apdr, "spatial")
        sums = {"loss": 0.0, "spatial_bce": 0.0}
        count = 0
        for batch_idx, (input_img, label_img) in enumerate(loader):
            if args.spatial_train_batches_per_epoch > 0 and batch_idx >= args.spatial_train_batches_per_epoch:
                break
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            optimizer.zero_grad(set_to_none=True)
            apdr(input_img)
            item = full_item(apdr)
            anchor = item["anchor"]
            spatial_target = make_spatial_target(anchor, label_img, calibration)
            spatial_bce = F.binary_cross_entropy_with_logits(
                item["spatial_logits"],
                spatial_target,
            )
            loss = args.spatial_bce_lambda * spatial_bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for _, param in trainable], args.grad_clip)
            optimizer.step()

            sums["loss"] += loss.item()
            sums["spatial_bce"] += spatial_bce.item()
            count += 1
            if (batch_idx + 1) % args.progress_freq == 0:
                print(
                    "spatial epoch %03d iter %04d loss=%.6f spatial_bce=%.6f"
                    % (epoch, batch_idx + 1, sums["loss"] / count, sums["spatial_bce"] / count),
                    flush=True,
                )
        if count == 0:
            raise RuntimeError("No spatial training batches were processed.")
        row = {
            "phase": "spatial",
            "epoch": epoch,
            "train": {key: value / count for key, value in sums.items()},
            "train_subset": evaluate_train_subset(apdr, args, device, calibration),
        }
        print(json.dumps(row, indent=2), flush=True)
        history.append(row)
    return history, {
        "phase": "spatial",
        "trainable_tensor_count": len(trainable),
        "trainable_param_count": sum(param.numel() for _, param in trainable),
        "frozen_param_count": sum(param.numel() for _, param in frozen),
        "trainable_tensors": [name for name, _ in trainable],
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
        fieldnames = [
            "name",
            "a0_psnr",
            "z_img",
            "global_score_unit",
            "b_img",
            "s_pixel_mean",
            "mask_mean",
            "rmse0",
            "hard_target_full_image",
            "spatial_bce",
            "zero_residual_max_abs_diff_vs_a0",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (input_img, label_img, name) in enumerate(loader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            a0 = crop_to(original(padded)[2], h, w)
            apdr_out = crop_to(apdr(padded)[2], h, w)
            item = full_item(apdr)
            anchor = crop_to(item["anchor"], h, w)
            error = anchor - label_img
            rmse = torch.sqrt(error.square().mean()).item()
            summary = calibration["summary"]
            denom = max(summary["rmse_q90_train"] - summary["rmse_q50_train"], 1e-8)
            hard_target = min(max((rmse - summary["rmse_q50_train"]) / denom, 0.0), 1.0)
            spatial_target = make_spatial_target(anchor, label_img, calibration)
            spatial_logits = crop_to(item["spatial_logits"], h, w)
            spatial_bce = F.binary_cross_entropy_with_logits(
                spatial_logits,
                spatial_target,
            ).item()
            diff = (apdr_out - a0).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            z_img = item["global_logits"].view(-1).item()
            global_score_unit = item["global_score_unit"].view(-1).item()
            b_img = item["global_budget_unit"].view(-1).item()
            s_pixel = crop_to(item["spatial_gate_unit"], h, w)
            s_pixel_mean = s_pixel.mean().item()
            row = {
                "name": name[0],
                "a0_psnr": psnr_from_prediction(a0, label_img),
                "z_img": z_img,
                "global_score_unit": global_score_unit,
                "b_img": b_img,
                "s_pixel_mean": s_pixel_mean,
                "mask_mean": (b_img * s_pixel).mean().item(),
                "rmse0": rmse,
                "hard_target_full_image": hard_target,
                "spatial_bce": spatial_bce,
                "zero_residual_max_abs_diff_vs_a0": diff,
            }
            rows.append(row)
            spatial_bces.append(spatial_bce)
            writer.writerow(row)
            if (idx + 1) % args.progress_freq == 0:
                print(f"test_eval {idx + 1}/{len(loader)}", flush=True)

    by_psnr = sorted(rows, key=lambda row: row["a0_psnr"])
    count = len(by_psnr)
    bucket_count = max(1, count // 4)
    hard = by_psnr[:bucket_count]
    easy = by_psnr[-bucket_count:]
    hard_b_mean = mean_or_none([row["b_img"] for row in hard])
    easy_b_mean = mean_or_none([row["b_img"] for row in easy])
    ratio = hard_b_mean / max(easy_b_mean, 1e-12) if hard_b_mean is not None else None
    auc = auc_score([row["z_img"] for row in hard], [row["z_img"] for row in easy])
    spearman_value = spearman(
        [row["z_img"] for row in rows],
        [row["a0_psnr"] for row in rows],
    )
    return rows, {
        "count": count,
        "hard_bottom_25pct": {
            "count": len(hard),
            "a0_psnr_range": [hard[0]["a0_psnr"], hard[-1]["a0_psnr"]],
            "mean_z_img": mean_or_none([row["z_img"] for row in hard]),
            "mean_b_img": hard_b_mean,
        },
        "easy_top_25pct": {
            "count": len(easy),
            "a0_psnr_range": [easy[0]["a0_psnr"], easy[-1]["a0_psnr"]],
            "mean_z_img": mean_or_none([row["z_img"] for row in easy]),
            "mean_b_img": easy_b_mean,
        },
        "hard_easy_b_img_ratio": ratio,
        "spearman_z_img_vs_a0_psnr": spearman_value,
        "auc_hard_vs_easy_by_z_img": auc,
        "strong_reference_count": len(easy),
        "strong_reference_mean_b_img": easy_b_mean,
        "mean_spatial_bce": statistics.mean(spatial_bces),
        "zero_residual_output": {
            "max_abs_diff_vs_a0": max_abs_diff,
            "threshold": args.zero_diff_threshold,
            "pass": max_abs_diff < args.zero_diff_threshold,
        },
    }


def write_history_csv(global_history, spatial_history, path):
    fieldnames = [
        "phase",
        "epoch",
        "train_loss",
        "train_hard_bce",
        "train_focal_bce",
        "train_rank",
        "train_spatial_bce",
        "subset_hard_bce",
        "subset_spatial_bce",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in global_history + spatial_history:
            train = row.get("train", {})
            subset = row["train_subset"]
            writer.writerow(
                {
                    "phase": row["phase"],
                    "epoch": row["epoch"],
                    "train_loss": train.get("loss"),
                    "train_hard_bce": train.get("hard_bce"),
                    "train_focal_bce": train.get("focal_bce"),
                    "train_rank": train.get("rank"),
                    "train_spatial_bce": train.get("spatial_bce"),
                    "subset_hard_bce": subset["hard_bce"],
                    "subset_spatial_bce": subset["spatial_bce"],
                }
            )


def build_gate(test_summary, global_history, spatial_history, args):
    global_first = global_history[0]["train_subset"]["hard_bce"]
    global_last = global_history[-1]["train_subset"]["hard_bce"]
    spatial_first = spatial_history[0]["train_subset"]["spatial_bce"]
    spatial_last = spatial_history[-1]["train_subset"]["spatial_bce"]
    checks = {
        "zero_residual_output": {
            "observed": test_summary["zero_residual_output"]["max_abs_diff_vs_a0"],
            "required": f"< {test_summary['zero_residual_output']['threshold']}",
            "pass": test_summary["zero_residual_output"]["pass"],
        },
        "deterministic_full_image_hard_bce": {
            "observed": {"first": global_first, "last": global_last},
            "required": f"last <= {args.hard_bce_gate} and first-last >= {args.hard_bce_min_delta}",
            "pass": global_last <= args.hard_bce_gate
            and (global_first - global_last) >= args.hard_bce_min_delta,
        },
        "auc_hard_vs_easy_by_z_img": {
            "observed": test_summary["auc_hard_vs_easy_by_z_img"],
            "required": ">= 0.82",
            "pass": test_summary["auc_hard_vs_easy_by_z_img"] is not None
            and test_summary["auc_hard_vs_easy_by_z_img"] >= 0.82,
        },
        "spearman_z_img_vs_a0_psnr": {
            "observed": test_summary["spearman_z_img_vs_a0_psnr"],
            "required": "<= -0.50",
            "pass": test_summary["spearman_z_img_vs_a0_psnr"] is not None
            and test_summary["spearman_z_img_vs_a0_psnr"] <= -0.50,
        },
        "hard_bottom25_mean_b_img": {
            "observed": test_summary["hard_bottom_25pct"]["mean_b_img"],
            "required": ">= 0.20",
            "pass": test_summary["hard_bottom_25pct"]["mean_b_img"] is not None
            and test_summary["hard_bottom_25pct"]["mean_b_img"] >= 0.20,
        },
        "easy_top25_mean_b_img": {
            "observed": test_summary["easy_top_25pct"]["mean_b_img"],
            "required": "<= 0.05",
            "pass": test_summary["easy_top_25pct"]["mean_b_img"] is not None
            and test_summary["easy_top_25pct"]["mean_b_img"] <= 0.05,
        },
        "hard_easy_b_img_ratio": {
            "observed": test_summary["hard_easy_b_img_ratio"],
            "required": ">= 4.0",
            "pass": test_summary["hard_easy_b_img_ratio"] is not None
            and test_summary["hard_easy_b_img_ratio"] >= 4.0,
        },
        "spatial_bce_decreased": {
            "observed": {"first": spatial_first, "last": spatial_last},
            "required": "last < first",
            "pass": spatial_last < spatial_first,
        },
        "test_mean_spatial_bce": {
            "observed": test_summary["mean_spatial_bce"],
            "required": "<= 0.80",
            "pass": test_summary["mean_spatial_bce"] <= 0.80,
        },
    }
    return {
        "stage": "APDR-v0.2R full-image calibrated selector-only preflight",
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_2r_selector_seed3407")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--global_epochs", type=int, default=5)
    parser.add_argument("--spatial_epochs", type=int, default=3)
    parser.add_argument("--global_batch_size", type=int, default=4)
    parser.add_argument("--spatial_batch_size", type=int, default=8)
    parser.add_argument("--global_resize", type=int, default=384)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--global_learning_rate", type=float, default=2e-4)
    parser.add_argument("--spatial_learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--global_train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--spatial_train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--calibration_images", type=int, default=0)
    parser.add_argument("--budget_calibration_images", type=int, default=0)
    parser.add_argument("--pixel_samples_per_image", type=int, default=2048)
    parser.add_argument("--loss_eval_images", type=int, default=256)
    parser.add_argument("--spatial_tau", type=float, default=0.0)
    parser.add_argument("--global_bce_lambda", type=float, default=1.0)
    parser.add_argument("--global_focal_lambda", type=float, default=0.5)
    parser.add_argument("--global_rank_lambda", type=float, default=0.2)
    parser.add_argument("--spatial_bce_lambda", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--budget_temperature_floor", type=float, default=0.05)
    parser.add_argument("--hard_bce_gate", type=float, default=0.55)
    parser.add_argument("--hard_bce_min_delta", type=float, default=0.05)
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
    print("calibration", json.dumps(calibration["summary"], indent=2), flush=True)

    global_history, global_scope = train_global_router(apdr, args, device, calibration)
    train_scores = collect_train_scores(apdr, args, device, calibration)
    budget_calibration = calibrate_budget(apdr, args, train_scores)
    print("budget_calibration", json.dumps(budget_calibration, indent=2), flush=True)
    spatial_history, spatial_scope = train_spatial_gate(apdr, args, device, calibration)

    per_image_csv = output_dir / f"selector_per_image_{args.tag}.csv"
    _, test_summary = evaluate_test(original, apdr, args, device, calibration, per_image_csv)
    gate = build_gate(test_summary, global_history, spatial_history, args)

    history_csv = output_dir / f"selector_history_{args.tag}.csv"
    write_history_csv(global_history, spatial_history, history_csv)

    result = {
        "stage": "apdr_v0_2r_selector_preflight",
        "tag": args.tag,
        "seed": args.seed,
        "device": str(device),
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "apdr_config": {
            "selector_mode": "v0_2r",
            "active_scales": "full",
            "residual_forced_zero": True,
            "residual_head_trainable": False,
            "gate_init": args.apdr_gate_init,
            "gate_max": args.apdr_gate_max,
            "residual_max": args.apdr_residual_max,
            "global_router_decoupled_from_spatial_context": True,
        },
        "phase_config": {
            "global_epochs": args.global_epochs,
            "spatial_epochs": args.spatial_epochs,
            "global_resize": args.global_resize,
            "global_batch_size": args.global_batch_size,
            "spatial_batch_size": args.spatial_batch_size,
            "global_train_batches_per_epoch": args.global_train_batches_per_epoch,
            "spatial_train_batches_per_epoch": args.spatial_train_batches_per_epoch,
            "loss_eval_images": args.loss_eval_images,
        },
        "loss_config": {
            "global_bce_lambda": args.global_bce_lambda,
            "global_focal_lambda": args.global_focal_lambda,
            "global_rank_lambda": args.global_rank_lambda,
            "spatial_bce_lambda": args.spatial_bce_lambda,
            "gate_sparsity_lambda_selector_only": 0.0,
            "rank_margin": args.rank_margin,
        },
        "checkpoint_load": checkpoint_load,
        "calibration": calibration["summary"],
        "budget_calibration": budget_calibration,
        "train_scope": {
            "global": global_scope,
            "spatial": spatial_scope,
        },
        "global_history": global_history,
        "spatial_history": spatial_history,
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
