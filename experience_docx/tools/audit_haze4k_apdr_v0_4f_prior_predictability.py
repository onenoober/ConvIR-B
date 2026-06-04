import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from audit_haze4k_apdr_v0_4b_derived_basis import (  # noqa: E402
    collect_low_targets,
    corr_flat,
    feature_matrix,
    parse_int_list,
    pca_bases,
    safe_mean,
    sigma_label,
    weighted_project,
    write_csv,
)
from audit_haze4k_apdr_v0_4b_mapping_triage import (  # noqa: E402
    fit_pls,
    fit_ridge,
    knn_predict,
    parse_float_list,
    predict_pls,
    predict_ridge,
    standardizer,
)
from audit_haze4k_apdr_v0_4d_spatial_coeff_probe import (  # noqa: E402
    collect_spatial_feature_matrices,
)
from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
)
from overfit_haze4k_apdr_v0_4a_lowfield import read_correctability  # noqa: E402


EPS = 1e-8


def write_csv_union(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_stats(prefix, tensor):
    flat = tensor.detach().float().flatten()
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    quantiles = torch.quantile(flat.cpu(), torch.tensor([0.10, 0.50, 0.90]))
    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_std": flat.std(unbiased=False).item(),
        f"{prefix}_min": flat.min().item(),
        f"{prefix}_max": flat.max().item(),
        f"{prefix}_p10": quantiles[0].item(),
        f"{prefix}_p50": quantiles[1].item(),
        f"{prefix}_p90": quantiles[2].item(),
    }


def robust_normalize_np(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    sample = values[finite]
    lo, hi = np.percentile(sample, [2.0, 98.0])
    if hi - lo <= EPS:
        lo, hi = float(np.min(sample)), float(np.max(sample))
    if hi - lo <= EPS:
        out = np.zeros_like(values, dtype=np.float32)
    else:
        out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~finite] = 0.0
    return out


def depth_cache_path(cache_dir, split, name):
    return Path(cache_dir) / split / (name.replace("/", "__") + ".npy")


def load_depth_tensor(cache_dir, split, name, size):
    path = depth_cache_path(cache_dir, split, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing depth cache for {name}: {path}")
    depth = robust_normalize_np(np.load(path).astype(np.float32))
    depth = torch.from_numpy(depth).view(1, 1, depth.shape[0], depth.shape[1])
    if tuple(depth.shape[-2:]) != tuple(size):
        depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
    return depth.float(), str(path)


def gradient_magnitude_1ch(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def safe_corr_torch(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    if denom.item() <= 1e-12:
        return 0.0
    return (a * b).sum().div(denom).item()


def dark_channel(x, patch=15):
    dark = x.min(dim=1, keepdim=True).values
    pad = patch // 2
    return -F.max_pool2d(-dark, kernel_size=patch, stride=1, padding=pad)


def local_mean(x, patch=31):
    return F.avg_pool2d(x, kernel_size=patch, stride=1, padding=patch // 2)


def estimate_airlight(x, dark, top_frac=0.01):
    flat_dark = dark.flatten()
    flat_rgb = x.permute(0, 2, 3, 1).reshape(-1, 3)
    count = max(1, int(flat_dark.numel() * top_frac))
    idx = torch.topk(flat_dark, k=count, largest=True).indices
    return flat_rgb[idx].mean(dim=0).view(1, 3, 1, 1).clamp(0, 1)


def fft_amplitude_stats(prefix, x, bins=6):
    gray = luma(x).detach().float()
    amp = torch.fft.fftshift(torch.fft.fft2(gray, dim=(-2, -1)), dim=(-2, -1)).abs()
    h, w = amp.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, h, dtype=amp.dtype, device=amp.device).view(h, 1)
    xx = torch.linspace(-1.0, 1.0, w, dtype=amp.dtype, device=amp.device).view(1, w)
    radius = torch.sqrt(xx * xx + yy * yy).view(1, 1, h, w)
    out = {}
    total = amp.sum().item()
    out.update(tensor_stats(f"{prefix}_amp", torch.log1p(amp)))
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (radius >= lo) & (radius < hi if idx < bins - 1 else radius <= hi)
        masked = amp[mask.expand_as(amp)]
        value = masked.mean().item() if masked.numel() else 0.0
        energy = masked.sum().item() / max(total, 1e-12) if masked.numel() else 0.0
        out[f"{prefix}_radial_bin{idx}_mean"] = value
        out[f"{prefix}_radial_bin{idx}_energy_frac"] = energy
    return out


def spatial_pool_features(tensors, grid):
    parts = []
    for tensor in tensors:
        parts.append(F.adaptive_avg_pool2d(tensor.detach().float().cpu(), (grid, grid)).flatten())
    return torch.cat(parts, dim=0)


def collect_prior_feature_rows(apdr_model, loader, device, args, tau, scores):
    rows = []
    depth_spatial = []
    physics_spatial = []
    frequency_spatial = []
    depth_cache_hits = 0
    depth_cache_paths = []

    for index, (input_img, label_img, name) in enumerate(loader):
        image_name = name[0]
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import frozen_apdr_tensors

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
        depth, depth_path = load_depth_tensor(args.depth_cache_dir, args.depth_split, image_name, input_img.shape[-2:])
        depth = depth.to(device)
        depth_cache_hits += 1
        if len(depth_cache_paths) < 5:
            depth_cache_paths.append(depth_path)

        gray = luma(input_img)
        grad = gradient_magnitude_1ch(gray)
        depth_grad = gradient_magnitude_1ch(depth)
        min_rgb = input_img.min(dim=1, keepdim=True).values
        max_rgb = input_img.max(dim=1, keepdim=True).values
        saturation = max_rgb - min_rgb
        dark = dark_channel(input_img, args.dark_patch)
        bright = max_rgb
        airlight = estimate_airlight(input_img, dark, args.airlight_top_frac)
        air_scalar = airlight.mean(dim=1, keepdim=True)
        transmission = ((input_img - airlight).abs().mean(dim=1, keepdim=True) / air_scalar.clamp_min(0.05)).clamp(0, 1)
        inv_transmission = 1.0 - transmission
        local_contrast = (gray - local_mean(gray, args.local_patch)).abs()
        residual_proxy = (input_img - anchor).abs().mean(dim=1, keepdim=True)
        low_abs = low_target.abs().mean(dim=1, keepdim=True)

        row = {
            "index": index,
            "name": image_name,
            "P_benefit": float(p_benefit),
            "proxy_score": float(score),
            "M_safe_mean": m_safe.mean().item(),
            "M_safe_nonzero_frac": (m_safe > 1e-6).float().mean().item(),
            "depth_luma_corr": safe_corr_torch(depth, gray),
            "depth_dark_corr": safe_corr_torch(depth, dark),
            "depth_m_safe_corr": safe_corr_torch(depth, m_safe),
            "depth_target_abs_corr": safe_corr_torch(depth, low_abs),
            "trans_depth_corr": safe_corr_torch(transmission, depth),
            "trans_dark_corr": safe_corr_torch(transmission, dark),
        }
        for prefix, tensor in (
            ("depth", depth),
            ("depth_grad", depth_grad),
            ("input_luma", gray),
            ("dark_channel", dark),
            ("bright_channel", bright),
            ("saturation", saturation),
            ("local_contrast", local_contrast),
            ("transmission_proxy", transmission),
            ("inv_transmission_proxy", inv_transmission),
            ("airlight_rgb", airlight.expand_as(input_img)),
            ("anchor_residual_proxy", residual_proxy),
            ("m_safe", m_safe),
        ):
            row.update(tensor_stats(prefix, tensor))
        row.update(fft_amplitude_stats("input_fft", input_img, args.freq_bins))
        row.update(fft_amplitude_stats("anchor_fft", anchor, args.freq_bins))
        rows.append(row)

        depth_spatial.append(
            spatial_pool_features([depth, depth_grad, depth * gray, depth * dark, depth * m_safe], args.spatial_grid)
        )
        physics_spatial.append(
            spatial_pool_features(
                [
                    dark,
                    bright,
                    saturation,
                    local_contrast,
                    transmission,
                    inv_transmission,
                    airlight.expand_as(input_img),
                    residual_proxy,
                ],
                args.spatial_grid,
            )
        )
        frequency_spatial.append(
            spatial_pool_features(
                [
                    torch.log1p(gradient_magnitude_1ch(gray)),
                    torch.log1p(gradient_magnitude_1ch(luma(anchor))),
                    torch.log1p((input_img - anchor).abs().mean(dim=1, keepdim=True)),
                ],
                args.spatial_grid,
            )
        )
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"prior_collected={index + 1}", flush=True)

    return {
        "rows": rows,
        "depth_spatial": torch.stack(depth_spatial, dim=0),
        "physics_spatial": torch.stack(physics_spatial, dim=0),
        "frequency_spatial": torch.stack(frequency_spatial, dim=0),
        "depth_cache_hits": depth_cache_hits,
        "depth_cache_examples": depth_cache_paths,
    }


def normalize_by_train(features, train_indices, valid_indices):
    mean, std = standardizer(features[train_indices])
    return (features[train_indices] - mean) / std, (features[valid_indices] - mean) / std


def split_folds(indices, folds, seed):
    shuffled = list(indices)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    fold_count = max(2, min(int(folds), len(shuffled)))
    return [shuffled[idx::fold_count] for idx in range(fold_count)]


def feature_predictability_rows(feature_sets, projections, low_data, active_indices, args):
    rows = []
    folds = split_folds(active_indices, args.folds, args.seed)
    active_set = set(active_indices)
    for feature_name, feature_x in feature_sets.items():
        for (low_size, k_dim), projection in projections.items():
            coeffs = projection["coeffs"]
            targets = low_data[low_size]["flat_targets"]
            weights = low_data[low_size]["flat_weights"]
            bases = projection["bases"]
            mean = projection["mean"]
            for method in args.methods:
                fold_rows = []
                for fold_idx, valid_indices in enumerate(folds):
                    valid_indices = [idx for idx in valid_indices if idx in active_set]
                    train_indices = [idx for idx in active_indices if idx not in set(valid_indices)]
                    if not valid_indices or not train_indices:
                        continue
                    x_train, x_valid = normalize_by_train(feature_x, train_indices, valid_indices)
                    train_y = coeffs[train_indices]
                    if method.startswith("ridge_"):
                        ridge = float(method.split("_", 1)[1])
                        model = fit_ridge(x_train, train_y, ridge)
                        pred = predict_ridge(x_valid, model)
                    elif method.startswith("pls_"):
                        comp = int(method.split("_", 1)[1])
                        if comp > min(x_train.shape[0] - 1, x_train.shape[1], train_y.shape[1]):
                            continue
                        model = fit_pls(x_train, train_y, comp)
                        pred = predict_pls(x_valid, model)
                    elif method.startswith("kernel_knn_"):
                        knn_k = int(method.rsplit("_", 1)[1])
                        pred, _extra = knn_predict(x_train, train_y, x_valid, knn_k, kernel=True)
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    truth = coeffs[valid_indices]
                    mse = (pred - truth).square().mean().item()
                    mae = (pred - truth).abs().mean().item()
                    var = (truth - truth.mean(dim=0, keepdim=True)).square().mean().item()
                    field_l1_num = 0.0
                    field_l1_den = 0.0
                    for row_pos, image_index in enumerate(valid_indices):
                        recon = mean + pred[row_pos] @ bases
                        weight = weights[image_index]
                        field_l1_num += (weight * (recon - targets[image_index]).abs()).sum().item()
                        field_l1_den += weight.sum().item()
                    fold_row = {
                        "feature_set": feature_name,
                        "feature_dim": int(feature_x.shape[1]),
                        "low_size": low_size,
                        "K": k_dim,
                        "method": method,
                        "fold": fold_idx,
                        "valid_count": len(valid_indices),
                        "coeff_mse": mse,
                        "coeff_mae": mae,
                        "coeff_r2": 1.0 - mse / max(var, 1e-12),
                        "coeff_corr": corr_flat(pred, truth),
                        "pred_std": pred.std(unbiased=False).item(),
                        "truth_std": truth.std(unbiased=False).item(),
                        "pred_true_norm_ratio": pred.norm(dim=1).mean().item()
                        / max(truth.norm(dim=1).mean().item(), 1e-12),
                        "lowspace_weighted_field_l1": field_l1_num / max(field_l1_den, 1e-12),
                    }
                    fold_rows.append(fold_row)
                    rows.append(fold_row)
                if fold_rows:
                    rows.append(
                        {
                            "feature_set": feature_name,
                            "feature_dim": int(feature_x.shape[1]),
                            "low_size": low_size,
                            "K": k_dim,
                            "method": method,
                            "fold": "mean",
                            "valid_count": sum(row["valid_count"] for row in fold_rows),
                            "coeff_mse": safe_mean([row["coeff_mse"] for row in fold_rows]),
                            "coeff_mae": safe_mean([row["coeff_mae"] for row in fold_rows]),
                            "coeff_r2": safe_mean([row["coeff_r2"] for row in fold_rows]),
                            "coeff_corr": safe_mean([row["coeff_corr"] for row in fold_rows]),
                            "pred_std": safe_mean([row["pred_std"] for row in fold_rows]),
                            "truth_std": safe_mean([row["truth_std"] for row in fold_rows]),
                            "pred_true_norm_ratio": safe_mean(
                                [row["pred_true_norm_ratio"] for row in fold_rows]
                            ),
                            "lowspace_weighted_field_l1": safe_mean(
                                [row["lowspace_weighted_field_l1"] for row in fold_rows]
                            ),
                        }
                    )
    return rows


def best_mean_rows(rows):
    mean_rows = [row for row in rows if row.get("fold") == "mean"]
    best_by_feature = {}
    for row in mean_rows:
        key = row["feature_set"]
        old = best_by_feature.get(key)
        if old is None or (row.get("coeff_corr") or -1e9) > (old.get("coeff_corr") or -1e9):
            best_by_feature[key] = row
    best_overall = None
    for row in mean_rows:
        if best_overall is None or (row.get("coeff_corr") or -1e9) > (best_overall.get("coeff_corr") or -1e9):
            best_overall = row
    return {"best_overall": best_overall, "best_by_feature_set": best_by_feature}


def select_columns(names, matrix, prefixes=(), exact=()):
    exact = set(exact)
    indices = [
        idx
        for idx, name in enumerate(names)
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not indices:
        raise RuntimeError(f"No feature columns selected for prefixes={prefixes} exact={sorted(exact)}")
    return [names[idx] for idx in indices], matrix[:, indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4f_prior_predictability_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--eval_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="16")
    parser.add_argument("--projection_ridge", type=float, default=1e-5)
    parser.add_argument("--ridge_values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--pls_components", default="4,8,16")
    parser.add_argument("--knn_values", default="5,9")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca_oversample", type=int, default=8)
    parser.add_argument("--pca_niter", type=int, default=4)
    parser.add_argument("--spatial_grid", type=int, default=4)
    parser.add_argument("--spatial_proj_channels", type=int, default=8)
    parser.add_argument("--hook_paths", default="Encoder.0,Encoder.1,Encoder.2,Decoder.0,Decoder.1,Decoder.2,Convs.0,Convs.1,APDR_1.context")
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--dark_patch", type=int, default=15)
    parser.add_argument("--local_patch", type=int, default=31)
    parser.add_argument("--airlight_top_frac", type=float, default=0.01)
    parser.add_argument("--freq_bins", type=int, default=6)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.pca_device == "cuda" and not torch.cuda.is_available():
        args.pca_device = "cpu"
    args.ridge_values = parse_float_list(args.ridge_values)
    args.pls_components = parse_int_list(args.pls_components)
    args.knn_values = parse_int_list(args.knn_values)
    methods = [f"ridge_{value:g}" for value in args.ridge_values]
    methods += [f"pls_{value}" for value in args.pls_components]
    methods += [f"kernel_knn_{value}" for value in args.knn_values]
    args.methods = methods
    k_values = parse_int_list(args.k_values)
    label = sigma_label(args.sigma)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau, scores = read_correctability(args.correctability_json, args.correctability_train_csv)
    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    collect_loader = build_loader(args.data_dir, args.basis_num_images, args.num_workers, shuffle=False)
    low_targets, low_weights, feature_rows, meta_rows = collect_low_targets(
        apdr_model,
        collect_loader,
        device,
        args,
        tau,
        scores,
        [args.low_size],
    )
    global_names, global_features = feature_matrix(feature_rows)

    eval_loader = build_loader(args.data_dir, args.eval_count, args.num_workers, shuffle=False)
    spatial = collect_spatial_feature_matrices(apdr_model, eval_loader, device, args, tau, scores)
    eval_loader = build_loader(args.data_dir, args.eval_count, args.num_workers, shuffle=False)
    prior = collect_prior_feature_rows(apdr_model, eval_loader, device, args, tau, scores)
    prior_names, prior_global = feature_matrix(prior["rows"])
    depth_names, depth_global = select_columns(
        prior_names,
        prior_global,
        prefixes=("depth",),
        exact=("depth_luma_corr", "depth_dark_corr", "depth_m_safe_corr", "depth_target_abs_corr"),
    )
    physics_names, physics_global = select_columns(
        prior_names,
        prior_global,
        prefixes=(
            "dark_channel",
            "bright_channel",
            "saturation",
            "local_contrast",
            "transmission_proxy",
            "inv_transmission_proxy",
            "airlight_rgb",
            "anchor_residual_proxy",
            "trans_",
        ),
    )
    frequency_names, frequency_global = select_columns(
        prior_names,
        prior_global,
        prefixes=("input_fft", "anchor_fft"),
    )

    flat_targets = low_targets[args.low_size].flatten(1)
    flat_weights = low_weights[args.low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    low_data = {
        args.low_size: {
            "flat_targets": flat_targets,
            "flat_weights": flat_weights,
            "pca": pca,
        }
    }
    projections = {}
    for k_dim in k_values:
        if k_dim > pca["bases"].shape[0]:
            continue
        bases = pca["bases"][:k_dim]
        coeffs, recons = weighted_project(
            flat_targets,
            flat_weights,
            pca["mean"],
            bases,
            args.projection_ridge,
        )
        projections[(args.low_size, k_dim)] = {
            "low_size": args.low_size,
            "K": k_dim,
            "coeffs": coeffs,
            "recons": recons,
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k_dim),
        }

    eval_count = args.eval_count
    active_indices = [
        row["index"]
        for row in meta_rows
        if row["index"] < eval_count and row["low_weight_sum"] > 1e-8
    ]
    if len(active_indices) < args.folds:
        raise RuntimeError(f"Need at least {args.folds} active/open samples; found {len(active_indices)}")

    feature_sets = {
        "current_global_stats": global_features[:eval_count],
        "convir_spatial_features": spatial["conv_spatial"],
        "current_spatial_priors": spatial["spatial_priors"],
        "depth_features": torch.cat(
            [depth_global[:eval_count], prior["depth_spatial"][:eval_count]],
            dim=1,
        ),
        "physics_proxy_features": torch.cat(
            [physics_global[:eval_count], prior["physics_spatial"][:eval_count]],
            dim=1,
        ),
        "frequency_amplitude_features": torch.cat(
            [frequency_global[:eval_count], prior["frequency_spatial"][:eval_count]],
            dim=1,
        ),
    }
    feature_sets["depth_physics_frequency"] = torch.cat(
        [
            depth_global[:eval_count],
            physics_global[:eval_count],
            frequency_global[:eval_count],
            prior["depth_spatial"][:eval_count],
            prior["physics_spatial"][:eval_count],
            prior["frequency_spatial"][:eval_count],
        ],
        dim=1,
    )
    feature_sets["global_plus_convir_plus_priors"] = torch.cat(
        [
            feature_sets["current_global_stats"],
            feature_sets["convir_spatial_features"],
            feature_sets["current_spatial_priors"],
            depth_global[:eval_count],
            physics_global[:eval_count],
            frequency_global[:eval_count],
            prior["depth_spatial"][:eval_count],
            prior["physics_spatial"][:eval_count],
            prior["frequency_spatial"][:eval_count],
        ],
        dim=1,
    )

    rows = feature_predictability_rows(feature_sets, projections, low_data, active_indices, args)
    summary = {
        "stage": "APDR-v0.4F residual coefficient predictability with depth/physics/frequency priors",
        "tag": args.tag,
        "sigma": args.sigma,
        "correctability_tau": tau,
        "basis_num_images": len(meta_rows),
        "eval_count": eval_count,
        "active_open_count": len(active_indices),
        "low_size": args.low_size,
        "k_values": [projection["K"] for projection in projections.values()],
        "methods": methods,
        "feature_dims": {name: int(value.shape[1]) for name, value in feature_sets.items()},
        "global_feature_names": global_names,
        "prior_global_feature_names": prior_names,
        "depth_feature_names": depth_names,
        "physics_feature_names": physics_names,
        "frequency_feature_names": frequency_names,
        "depth_cache_hits": prior["depth_cache_hits"],
        "depth_cache_examples": prior["depth_cache_examples"],
        "best_rows": best_mean_rows(rows),
        "args": vars(args),
    }
    summary_path = output_dir / f"prior_predictability_summary_{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv_union(output_dir / f"prior_predictability_coeff_cv_{label}.csv", rows)
    write_csv(output_dir / f"prior_predictability_feature_rows_{label}.csv", prior["rows"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
