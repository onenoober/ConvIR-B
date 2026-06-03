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
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
    correlation,
    frozen_apdr_tensors,
    gaussian_lowpass,
    percentile,
    psnr,
)
from overfit_haze4k_apdr_v0_4a_lowfield import (  # noqa: E402
    gradient_magnitude,
    read_correctability,
)


def parse_int_list(value):
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError(f"Empty integer list: {value}")
    return out


def sigma_label(sigma):
    value = float(sigma)
    if value.is_integer():
        return f"sigma{int(value)}"
    return f"sigma{str(value).replace('.', 'p')}"


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def tensor_stats(prefix, tensor):
    flat = tensor.detach().float().flatten()
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_std": flat.std(unbiased=False).item(),
        f"{prefix}_min": flat.min().item(),
        f"{prefix}_max": flat.max().item(),
    }


def add_channel_stats(row, prefix, tensor):
    for channel in range(tensor.shape[1]):
        row.update(tensor_stats(f"{prefix}_c{channel}", tensor[:, channel : channel + 1]))


def feature_row(index, name, input_img, anchor, label, m_safe, p_benefit, proxy_score):
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    diff = input_img - anchor
    grad = gradient_magnitude(input_img)
    row = {
        "index": index,
        "name": name,
        "P_benefit": float(p_benefit),
        "proxy_score": float(proxy_score),
        "anchor_psnr": psnr(anchor, label),
        "M_safe_mean": m_safe.mean().item(),
        "M_safe_nonzero_frac": (m_safe > 1e-6).float().mean().item(),
    }
    for key, value in (
        ("x", input_img),
        ("anchor", anchor),
        ("absdiff", diff.abs()),
        ("dark", min_rgb),
        ("bright", max_rgb),
        ("saturation", saturation),
        ("gradient", grad),
        ("m_safe", m_safe),
    ):
        row.update(tensor_stats(key, value))
    add_channel_stats(row, "x", input_img)
    add_channel_stats(row, "anchor", anchor)
    add_channel_stats(row, "diff", diff)
    return row


def feature_matrix(rows):
    excluded = {"index", "name"}
    names = [name for name in rows[0].keys() if name not in excluded]
    matrix = []
    for row in rows:
        values = []
        for name in names:
            value = row[name]
            values.append(0.0 if value is None else float(value))
        matrix.append(values)
    return names, torch.tensor(matrix, dtype=torch.float32)


def collect_low_targets(apdr_model, loader, device, args, tau, scores, low_sizes):
    low_targets = {size: [] for size in low_sizes}
    low_weights = {size: [] for size in low_sizes}
    features = []
    meta_rows = []
    for index, (input_img, label_img, name) in enumerate(loader):
        image_name = name[0]
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        anchor, m_safe, _delta_star, low_target, _color_target = frozen_apdr_tensors(
            apdr_model,
            input_img,
            label_img,
            args,
        )
        score = scores.get(image_name)
        if score is None:
            raise KeyError(f"Missing correctability score for {image_name}")
        p_benefit = 1.0 if score >= tau else 0.0
        weight = m_safe * p_benefit
        for size in low_sizes:
            low_targets[size].append(
                F.interpolate(
                    low_target,
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .float()
            )
            low_weights[size].append(
                F.interpolate(
                    weight,
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .float()
            )
        features.append(
            feature_row(index, image_name, input_img, anchor, label_img, m_safe, p_benefit, score)
        )
        meta_rows.append(
            {
                "index": index,
                "name": image_name,
                "P_benefit": p_benefit,
                "proxy_score": score,
                "M_safe_mean": m_safe.mean().item(),
                "anchor_psnr": psnr(anchor, label_img),
                "low_target_abs_mean": low_target.abs().mean().item(),
                "low_weight_sum": weight.sum().item(),
            }
        )
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"collected={index + 1}", flush=True)
    for size in low_sizes:
        low_targets[size] = torch.stack(low_targets[size], dim=0)
        low_weights[size] = torch.stack(low_weights[size], dim=0)
    return low_targets, low_weights, features, meta_rows


def pca_bases(targets, weights, k_values, args):
    count = targets.shape[0]
    flat_targets = targets.flatten(1)
    flat_weights = weights.repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    active = flat_weights.sum(dim=1) > 1e-8
    active_count = int(active.sum().item())
    if active_count < 2:
        raise RuntimeError("Not enough open/weighted samples to derive bases.")
    weighted_den = flat_weights[active].sum(dim=0).clamp_min(1e-12)
    mean = (flat_targets[active] * flat_weights[active]).sum(dim=0) / weighted_den
    weighted_centered = (flat_targets[active] - mean) * flat_weights[active].sqrt()
    max_k = min(max(k_values), weighted_centered.shape[0], weighted_centered.shape[1])
    q = min(max_k + int(args.pca_oversample), weighted_centered.shape[0], weighted_centered.shape[1])
    q = max(2, q)
    _, singular_values, v = torch.pca_lowrank(
        weighted_centered.to(args.pca_device),
        q=q,
        center=False,
        niter=args.pca_niter,
    )
    bases = v[:, :max_k].t().detach().cpu().contiguous()
    singular_values = singular_values.detach().cpu()
    total_energy = weighted_centered.square().sum().item()
    explained = {}
    for k in k_values:
        if k <= bases.shape[0]:
            explained[k] = singular_values[:k].square().sum().item() / max(total_energy, 1e-12)
    return {
        "count": count,
        "active": active,
        "active_count": active_count,
        "mean": mean.detach().cpu(),
        "bases": bases,
        "explained": explained,
        "total_energy": total_energy,
    }


def weighted_project(flat_targets, flat_weights, mean, bases, ridge):
    count = flat_targets.shape[0]
    k = bases.shape[0]
    coeffs = torch.zeros(count, k, dtype=torch.float32)
    recons = torch.empty_like(flat_targets)
    centered = flat_targets - mean
    eye = torch.eye(k, dtype=torch.float32)
    for index in range(count):
        weight = flat_weights[index]
        if weight.sum().item() <= 1e-8:
            recons[index] = mean
            continue
        weighted_bases = bases * weight.unsqueeze(0)
        gram = weighted_bases @ bases.t() + float(ridge) * eye
        rhs = weighted_bases @ centered[index]
        coeff = torch.linalg.solve(gram, rhs)
        coeffs[index] = coeff
        recons[index] = mean + coeff @ bases
    return coeffs, recons


def corr_flat(x, y):
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() <= 1e-12:
        return None
    return (x * y).sum().div(denom).item()


def split_folds(indices, folds, seed):
    shuffled = list(indices)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    fold_count = max(2, min(int(folds), len(shuffled)))
    return [shuffled[idx::fold_count] for idx in range(fold_count)]


def fit_ridge(x_train, y_train, ridge):
    ones = torch.ones(x_train.shape[0], 1, dtype=torch.float32)
    x_aug = torch.cat([x_train, ones], dim=1)
    gram = x_aug.t() @ x_aug
    reg = torch.eye(gram.shape[0], dtype=torch.float32) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = x_aug.t() @ y_train
    return torch.linalg.solve(gram + reg, rhs)


def predict_ridge(x, weights):
    ones = torch.ones(x.shape[0], 1, dtype=torch.float32)
    return torch.cat([x, ones], dim=1) @ weights


def standardize(train_x, valid_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (valid_x - mean) / std


def coeff_predictability_rows(feature_x, projections, low_data, active_indices, args):
    rows = []
    folds = split_folds(active_indices, args.folds, args.seed)
    active_set = set(active_indices)
    for (low_size, k), projection in projections.items():
        coeffs = projection["coeffs"]
        targets = low_data[low_size]["flat_targets"]
        weights = low_data[low_size]["flat_weights"]
        bases = projection["bases"]
        mean = projection["mean"]
        fold_rows = []
        for fold_idx, valid_indices in enumerate(folds):
            valid_indices = [idx for idx in valid_indices if idx in active_set]
            train_indices = [idx for idx in active_indices if idx not in set(valid_indices)]
            if not valid_indices or not train_indices:
                continue
            train_x_raw = feature_x[train_indices]
            valid_x_raw = feature_x[valid_indices]
            train_x, valid_x = standardize(train_x_raw, valid_x_raw)
            model = fit_ridge(train_x, coeffs[train_indices], args.coeff_ridge)
            pred = predict_ridge(valid_x, model)
            truth = coeffs[valid_indices]
            mse = (pred - truth).square().mean().item()
            var = (truth - truth.mean(dim=0, keepdim=True)).square().mean().item()
            r2 = 1.0 - mse / max(var, 1e-12)
            coeff_corr = corr_flat(pred, truth)
            field_l1_num = 0.0
            field_l1_den = 0.0
            for row_pos, image_index in enumerate(valid_indices):
                recon = mean + pred[row_pos] @ bases
                weight = weights[image_index]
                field_l1_num += (weight * (recon - targets[image_index]).abs()).sum().item()
                field_l1_den += weight.sum().item()
            fold_row = {
                "low_size": low_size,
                "K": k,
                "fold": fold_idx,
                "valid_count": len(valid_indices),
                "coeff_mse": mse,
                "coeff_r2": r2,
                "coeff_corr": coeff_corr,
                "lowspace_weighted_field_l1": field_l1_num / max(field_l1_den, 1e-12),
            }
            fold_rows.append(fold_row)
            rows.append(fold_row)
        if fold_rows:
            rows.append(
                {
                    "low_size": low_size,
                    "K": k,
                    "fold": "mean",
                    "valid_count": sum(row["valid_count"] for row in fold_rows),
                    "coeff_mse": safe_mean([row["coeff_mse"] for row in fold_rows]),
                    "coeff_r2": safe_mean([row["coeff_r2"] for row in fold_rows]),
                    "coeff_corr": safe_mean([row["coeff_corr"] for row in fold_rows]),
                    "lowspace_weighted_field_l1": safe_mean(
                        [row["lowspace_weighted_field_l1"] for row in fold_rows]
                    ),
                }
            )
    return rows


def router_overfit_predictions(feature_x, projections, low_data, active_indices, args):
    router = {}
    chosen = active_indices[: max(1, min(args.router_probe_count, len(active_indices)))]
    for (low_size, k), projection in projections.items():
        coeffs = projection["coeffs"]
        train_x_raw = feature_x[chosen]
        train_x, _ = standardize(train_x_raw, train_x_raw)
        model = fit_ridge(train_x, coeffs[chosen], args.router_ridge)
        pred_coeffs = predict_ridge(train_x, model)
        bases = projection["bases"]
        mean = projection["mean"]
        targets = low_data[low_size]["flat_targets"]
        weights = low_data[low_size]["flat_weights"]
        items = {}
        for pos, image_index in enumerate(chosen):
            recon = mean + pred_coeffs[pos] @ bases
            true_coeff = coeffs[image_index]
            weight = weights[image_index]
            target = targets[image_index]
            items[image_index] = {
                "recon": recon,
                "coeff_mse": (pred_coeffs[pos] - true_coeff).square().mean().item(),
                "coeff_mae": (pred_coeffs[pos] - true_coeff).abs().mean().item(),
                "lowspace_weighted_field_l1": (
                    weight * (recon - target).abs()
                ).sum().item()
                / max(weight.sum().item(), 1e-12),
                "lowspace_field_corr": corr_flat(recon * weight, target * weight),
            }
        router[(low_size, k)] = items
    return router


def group_name(row, hard_cut, easy_cut):
    if row["anchor_psnr"] <= hard_cut and row["P_benefit"] >= 0.5:
        return "open_hard"
    if row["anchor_psnr"] <= hard_cut:
        return "closed_hard"
    if row["anchor_psnr"] >= easy_cut and row["P_benefit"] >= 0.5:
        return "open_easy"
    if row["anchor_psnr"] >= easy_cut:
        return "closed_easy"
    return "middle"


def summarize_rows(rows, initial_l1, final_l1, final_corr):
    gains = [row["output_gain"] for row in rows]
    oracle_gains = [row["oracle_gain"] for row in rows]
    ordered = sorted(rows, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    easy = ordered[3 * len(ordered) // 4 :]
    positive = [row for row in rows if row["oracle_gain"] > 1e-6]
    recovery = sum(row["output_gain"] for row in positive) / max(
        sum(row["oracle_gain"] for row in positive),
        1e-12,
    )
    strong_cut = percentile([row["anchor_psnr"] for row in rows], 75)
    strong = [row for row in rows if row["anchor_psnr"] >= strong_cut]
    return {
        "count": len(rows),
        "open_count": sum(row["P_benefit"] >= 0.5 for row in rows),
        "initial_weighted_delta_l1": initial_l1,
        "projection_weighted_delta_l1": final_l1,
        "weighted_delta_l1_drop": (initial_l1 - final_l1) / max(initial_l1, 1e-12),
        "pred_target_corr": final_corr,
        "mean_output_gain": statistics.mean(gains),
        "mean_oracle_gain": statistics.mean(oracle_gains),
        "oracle_recovery": recovery,
        "hard_bottom25_output_gain": statistics.mean(row["output_gain"] for row in hard),
        "hard_bottom25_oracle_gain": statistics.mean(row["oracle_gain"] for row in hard),
        "easy_top25_output_gain": statistics.mean(row["output_gain"] for row in easy),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regressions": sum(row["output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["output_gain"] <= -0.20 for row in rows),
    }


def gate0_checks(summary):
    return {
        "weighted_delta_l1_drop": {
            "observed": summary["weighted_delta_l1_drop"],
            "required": ">= 0.60",
            "pass": summary["weighted_delta_l1_drop"] >= 0.60,
        },
        "pred_target_corr": {
            "observed": summary["pred_target_corr"],
            "required": ">= 0.75",
            "pass": summary["pred_target_corr"] is not None and summary["pred_target_corr"] >= 0.75,
        },
        "oracle_recovery": {
            "observed": summary["oracle_recovery"],
            "required": ">= 0.50",
            "pass": summary["oracle_recovery"] >= 0.50,
        },
        "hard_bottom25_gain": {
            "observed": summary["hard_bottom25_output_gain"],
            "required": ">= +0.60 dB",
            "pass": summary["hard_bottom25_output_gain"] >= 0.60,
        },
        "easy_top25_gain": {
            "observed": summary["easy_top25_output_gain"],
            "required": ">= -0.010 dB",
            "pass": summary["easy_top25_output_gain"] >= -0.010,
        },
        "strong_reference_regressions": {
            "observed": summary["strong_reference_regressions"],
            "required": "== 0",
            "pass": summary["strong_reference_regressions"] == 0,
        },
        "severe_regressions": {
            "observed": summary["severe_regressions"],
            "required": "== 0",
            "pass": summary["severe_regressions"] == 0,
        },
    }


def evaluate_full_resolution(apdr_model, loader, device, args, tau, scores, projections, router_predictions):
    combo_rows = {combo: [] for combo in projections}
    combo_sums = {
        combo: {"initial_num": 0.0, "final_num": 0.0, "den": 0.0, "corrs": []}
        for combo in projections
    }
    router_rows = []
    for index, (input_img, label_img, name) in enumerate(loader):
        image_name = name[0]
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        anchor, m_safe, _delta_star, low_target, _color_target = frozen_apdr_tensors(
            apdr_model,
            input_img,
            label_img,
            args,
        )
        score = scores.get(image_name)
        if score is None:
            raise KeyError(f"Missing correctability score for {image_name}")
        p_benefit = 1.0 if score >= tau else 0.0
        weight = m_safe * p_benefit
        expanded = weight.expand_as(low_target)
        oracle = (anchor + weight * low_target).clamp(0, 1)
        anchor_psnr = psnr(anchor, label_img)
        oracle_psnr = psnr(oracle, label_img)
        for combo, projection in projections.items():
            low_size, k = combo
            recon = projection["recons"][index].view(1, 3, low_size, low_size).to(device)
            pred = F.interpolate(
                recon,
                size=anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = gaussian_lowpass(pred, args.kernel_size, args.sigma)
            output = (anchor + weight * pred).clamp(0, 1)
            output_psnr = psnr(output, label_img)
            corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
            if corr is not None:
                combo_sums[combo]["corrs"].append(corr)
            combo_sums[combo]["initial_num"] += (expanded * low_target.abs()).sum().item()
            combo_sums[combo]["final_num"] += (expanded * (pred - low_target).abs()).sum().item()
            combo_sums[combo]["den"] += expanded.sum().item()
            row = {
                "low_size": low_size,
                "K": k,
                "index": index,
                "name": image_name,
                "anchor_psnr": anchor_psnr,
                "output_psnr": output_psnr,
                "oracle_psnr": oracle_psnr,
                "output_gain": output_psnr - anchor_psnr,
                "oracle_gain": oracle_psnr - anchor_psnr,
                "corr": corr,
                "P_benefit": p_benefit,
                "proxy_score": score,
                "M_safe_mean": m_safe.mean().item(),
                "target_abs_mean": low_target.abs().mean().item(),
                "projection_abs_mean": pred.abs().mean().item(),
                "residual_abs_mean": (pred - low_target).abs().mean().item(),
                "projection_weighted_l1": (
                    expanded * (pred - low_target).abs()
                ).sum().item()
                / max(expanded.sum().item(), 1e-12),
            }
            combo_rows[combo].append(row)
            router_item = router_predictions.get(combo, {}).get(index)
            if router_item is not None:
                router_recon = router_item["recon"].view(1, 3, low_size, low_size).to(device)
                router_pred = F.interpolate(
                    router_recon,
                    size=anchor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                router_pred = gaussian_lowpass(router_pred, args.kernel_size, args.sigma)
                router_output = (anchor + weight * router_pred).clamp(0, 1)
                router_output_psnr = psnr(router_output, label_img)
                router_rows.append(
                    {
                        "low_size": low_size,
                        "K": k,
                        "index": index,
                        "name": image_name,
                        "anchor_psnr": anchor_psnr,
                        "router_output_psnr": router_output_psnr,
                        "oracle_psnr": oracle_psnr,
                        "router_output_gain": router_output_psnr - anchor_psnr,
                        "oracle_gain": oracle_psnr - anchor_psnr,
                        "coeff_mse": router_item["coeff_mse"],
                        "coeff_mae": router_item["coeff_mae"],
                        "lowspace_weighted_field_l1": router_item["lowspace_weighted_field_l1"],
                        "lowspace_field_corr": router_item["lowspace_field_corr"],
                        "full_field_corr": correlation(router_pred.cpu(), low_target.cpu(), expanded.cpu()),
                        "P_benefit": p_benefit,
                        "proxy_score": score,
                        "M_safe_mean": m_safe.mean().item(),
                        "target_abs_mean": low_target.abs().mean().item(),
                        "pred_abs_mean": router_pred.abs().mean().item(),
                    }
                )
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"evaluated={index + 1}", flush=True)
    summaries = {}
    residual_rows = []
    for combo, rows in combo_rows.items():
        sums = combo_sums[combo]
        initial_l1 = sums["initial_num"] / max(sums["den"], 1e-12)
        final_l1 = sums["final_num"] / max(sums["den"], 1e-12)
        summary = summarize_rows(rows, initial_l1, final_l1, safe_mean(sums["corrs"]))
        hard_cut = percentile([row["anchor_psnr"] for row in rows], 25)
        easy_cut = percentile([row["anchor_psnr"] for row in rows], 75)
        for row in rows:
            residual_rows.append({**row, "group": group_name(row, hard_cut, easy_cut)})
        summaries[combo] = summary
    return summaries, residual_rows, router_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4b_derived_basis_sigma3_seed3407")
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_sizes", default="32,48")
    parser.add_argument("--k_values", default="4,8,16,32,48")
    parser.add_argument("--projection_ridge", type=float, default=1e-5)
    parser.add_argument("--coeff_ridge", type=float, default=1e-3)
    parser.add_argument("--router_ridge", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--router_probe_count", type=int, default=32)
    parser.add_argument("--pca_oversample", type=int, default=8)
    parser.add_argument("--pca_niter", type=int, default=4)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.pca_device == "cuda" and not torch.cuda.is_available():
        args.pca_device = "cpu"

    low_sizes = parse_int_list(args.low_sizes)
    k_values = parse_int_list(args.k_values)
    label = sigma_label(args.sigma)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau, scores = read_correctability(args.correctability_json, args.correctability_train_csv)
    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    collect_loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=False)
    low_targets, low_weights, feature_rows, meta_rows = collect_low_targets(
        apdr_model,
        collect_loader,
        device,
        args,
        tau,
        scores,
        low_sizes,
    )
    feature_names, features = feature_matrix(feature_rows)
    write_csv(output_dir / f"derived_basis_feature_rows_{label}.csv", feature_rows)

    low_data = {}
    projections = {}
    spectrum_rows = []
    active_indices = [row["index"] for row in meta_rows if row["low_weight_sum"] > 1e-8]
    for low_size in low_sizes:
        flat_targets = low_targets[low_size].flatten(1)
        flat_weights = low_weights[low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
        pca = pca_bases(low_targets[low_size], low_weights[low_size], k_values, args)
        low_data[low_size] = {
            "flat_targets": flat_targets,
            "flat_weights": flat_weights,
            "pca": pca,
        }
        valid_k = [k for k in k_values if k <= pca["bases"].shape[0]]
        for k in valid_k:
            bases = pca["bases"][:k]
            coeffs, recons = weighted_project(
                flat_targets,
                flat_weights,
                pca["mean"],
                bases,
                args.projection_ridge,
            )
            projections[(low_size, k)] = {
                "low_size": low_size,
                "K": k,
                "coeffs": coeffs,
                "recons": recons,
                "bases": bases,
                "mean": pca["mean"],
                "explained_weighted_energy": pca["explained"].get(k),
            }

    coeff_rows = coeff_predictability_rows(features, projections, low_data, active_indices, args)
    router_predictions = router_overfit_predictions(features, projections, low_data, active_indices, args)

    eval_loader = build_loader(args.data_dir, args.num_images, args.num_workers, shuffle=False)
    summaries, residual_rows, router_rows = evaluate_full_resolution(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        projections,
        router_predictions,
    )

    summary_items = []
    for combo, summary in summaries.items():
        low_size, k = combo
        checks = gate0_checks(summary)
        item = {
            "low_size": low_size,
            "K": k,
            "active_count": low_data[low_size]["pca"]["active_count"],
            "explained_weighted_energy": projections[combo]["explained_weighted_energy"],
            **summary,
            "gate0_pass": all(check["pass"] for check in checks.values()),
        }
        summary_items.append(item)
        spectrum_rows.append(item)

    result = {
        "stage": "APDR-v0.4B derived low-field basis Gate 0 and coefficient predictability",
        "tag": args.tag,
        "sigma": args.sigma,
        "correctability_tau": tau,
        "num_images": len(meta_rows),
        "active_count": len(active_indices),
        "low_sizes": low_sizes,
        "k_values": k_values,
        "feature_names": feature_names,
        "summaries": [
            {
                **item,
                "gate0_checks": gate0_checks(summaries[(item["low_size"], item["K"])]),
            }
            for item in summary_items
        ],
        "best_by_recovery": max(summary_items, key=lambda row: row["oracle_recovery"]) if summary_items else None,
        "args": vars(args),
    }

    (output_dir / f"basis_projection_oracle_{label}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / f"derived_basis_spectrum_{label}.csv", spectrum_rows)
    write_csv(output_dir / f"coeff_predictability_cv_{label}.csv", coeff_rows)
    write_csv(output_dir / "basis_residual_error_groups.csv", residual_rows)
    write_csv(output_dir / "router_overfit32_coeff_vs_field.csv", router_rows)
    write_csv(output_dir / f"basis_projection_meta_rows_{label}.csv", meta_rows)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output_dir / f'basis_projection_oracle_{label}.json'}")
    print(f"wrote {output_dir / f'derived_basis_spectrum_{label}.csv'}")
    print(f"wrote {output_dir / f'coeff_predictability_cv_{label}.csv'}")
    print(f"wrote {output_dir / 'basis_residual_error_groups.csv'}")
    print(f"wrote {output_dir / 'router_overfit32_coeff_vs_field.csv'}")


if __name__ == "__main__":
    main()
