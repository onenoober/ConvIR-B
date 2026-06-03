import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from audit_haze4k_apdr_v0_4b_derived_basis import (  # noqa: E402
    collect_low_targets,
    feature_matrix,
    parse_int_list,
    pca_bases,
    sigma_label,
    weighted_project,
    write_csv,
)
from audit_haze4k_apdr_v0_4b_mapping_triage import (  # noqa: E402
    aggregate_mapper_rows,
    evaluate_predictions,
    fit_pls,
    fit_ridge,
    knn_predict,
    open_easy_failure_rows,
    pairwise_dist,
    parse_float_list,
    predict_pls,
    predict_ridge,
    standardizer,
    write_csv_union,
)
from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
    frozen_apdr_tensors,
)
from overfit_haze4k_apdr_v0_4a_lowfield import (  # noqa: E402
    gradient_magnitude,
    read_correctability,
)


def get_submodule(root, path):
    module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def stable_seed(text, seed):
    value = int(seed) & 0xFFFFFFFF
    for byte in text.encode("utf-8"):
        value = ((value * 16777619) ^ byte) & 0xFFFFFFFF
    return value


def projection_matrix(name, channels, out_channels, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(name, seed))
    weight = torch.randn(out_channels, channels, generator=generator, dtype=torch.float32)
    return weight / math.sqrt(max(1, channels))


def pooled_projected_features(name, tensor, grid, proj_channels, seed):
    tensor = tensor.detach().float().cpu()
    pooled = F.adaptive_avg_pool2d(tensor, (grid, grid)).squeeze(0)
    channels = pooled.shape[0]
    weight = projection_matrix(name, channels, proj_channels, seed)
    projected = torch.einsum("pc,chw->phw", weight, pooled)
    channel_mean = pooled.mean(dim=0, keepdim=True)
    channel_std = pooled.std(dim=0, keepdim=True, unbiased=False)
    return torch.cat([projected, channel_mean, channel_std], dim=0).flatten()


def low_grid(tensor, grid):
    return F.adaptive_avg_pool2d(tensor.detach().float().cpu(), (grid, grid)).flatten()


def spatial_prior_features(input_img, anchor, m_safe, p_benefit, grid):
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    diff = (input_img - anchor).abs()
    grad = gradient_magnitude(input_img)
    p_map = torch.full_like(m_safe, float(p_benefit))
    tensors = [input_img, anchor, diff, min_rgb, max_rgb, saturation, grad, m_safe, p_map]
    return torch.cat([low_grid(tensor, grid) for tensor in tensors], dim=0)


def register_hooks(model, hook_paths):
    latest = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if torch.is_tensor(output) and output.dim() == 4:
                latest[name] = output.detach()

        return hook

    for path in hook_paths:
        handles.append(get_submodule(model, path).register_forward_hook(make_hook(path)))
    return latest, handles


def collect_spatial_feature_matrices(apdr_model, loader, device, args, tau, scores):
    hook_paths = [item.strip() for item in args.hook_paths.split(",") if item.strip()]
    latest, handles = register_hooks(apdr_model, hook_paths)
    conv_rows = []
    prior_rows = []
    meta_rows = []
    try:
        for index, (input_img, label_img, name) in enumerate(loader):
            image_name = name[0]
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            latest.clear()
            anchor, m_safe, _delta_star, _low_target, _color_target = frozen_apdr_tensors(
                apdr_model,
                input_img,
                label_img,
                args,
            )
            missing = [path for path in hook_paths if path not in latest]
            if missing:
                raise RuntimeError(f"Missing hook outputs for {missing}")
            score = scores.get(image_name)
            if score is None:
                raise KeyError(f"Missing correctability score for {image_name}")
            p_benefit = 1.0 if score >= tau else 0.0
            conv_parts = [
                pooled_projected_features(
                    path,
                    latest[path],
                    args.spatial_grid,
                    args.spatial_proj_channels,
                    args.seed,
                )
                for path in hook_paths
            ]
            conv_rows.append(torch.cat(conv_parts, dim=0))
            prior_rows.append(spatial_prior_features(input_img, anchor, m_safe, p_benefit, args.spatial_grid))
            meta_rows.append({"index": index, "name": image_name, "P_benefit": p_benefit})
            if args.progress_freq and (index + 1) % args.progress_freq == 0:
                print(f"spatial_collected={index + 1}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "conv_spatial": torch.stack(conv_rows, dim=0),
        "spatial_priors": torch.stack(prior_rows, dim=0),
        "meta": meta_rows,
        "hook_paths": hook_paths,
    }


def normalize_by_train(features, train_indices, eval_indices):
    mean, std = standardizer(features[train_indices])
    return (features - mean) / std, mean, std


def make_probe_predictions(features, projections, train_indices, eval_indices, args, prefix):
    predictions = {}
    features, _mean, _std = normalize_by_train(features, train_indices, eval_indices)
    x_train = features[train_indices]
    x_eval = features[eval_indices]
    nn_distance = pairwise_dist(x_eval, x_train).min(dim=1).values
    base_extra = {
        int(eval_indices[pos]): {
            "nn_distance": nn_distance[pos].item(),
            "confidence_proxy": 1.0 / (1.0 + nn_distance[pos].item()),
            "feature_set": prefix,
        }
        for pos in range(len(eval_indices))
    }
    for key, projection in projections.items():
        coeffs = projection["coeffs"]
        train_y = coeffs[train_indices]
        k_dim = projection["K"]
        predictions[(key, f"{prefix}_zero_field")] = {
            "pred_coeffs": torch.zeros(len(eval_indices), k_dim, dtype=torch.float32),
            "zero_field": True,
            "extras": base_extra,
            "family": f"{prefix}_zero",
        }
        predictions[(key, f"{prefix}_mean_coeff")] = {
            "pred_coeffs": train_y.mean(dim=0, keepdim=True).repeat(len(eval_indices), 1),
            "zero_field": False,
            "extras": base_extra,
            "family": f"{prefix}_mean",
        }
        for ridge in args.ridge_values:
            weights = fit_ridge(x_train, train_y, ridge)
            predictions[(key, f"{prefix}_ridge_{ridge:g}")] = {
                "pred_coeffs": predict_ridge(x_eval, weights),
                "zero_field": False,
                "extras": base_extra,
                "family": f"{prefix}_ridge",
                "ridge": ridge,
            }
        for comp in args.pls_components:
            if comp > min(x_train.shape[0] - 1, x_train.shape[1], train_y.shape[1]):
                continue
            model = fit_pls(x_train, train_y, comp)
            predictions[(key, f"{prefix}_pls_{model['components']}")] = {
                "pred_coeffs": predict_pls(x_eval, model),
                "zero_field": False,
                "extras": base_extra,
                "family": f"{prefix}_pls",
                "components": model["components"],
            }
        for knn_k in args.knn_values:
            pred, extra = knn_predict(x_train, train_y, x_eval, knn_k, kernel=True)
            merged = {}
            for pos, image_index in enumerate(eval_indices):
                merged[int(image_index)] = {
                    **base_extra[int(image_index)],
                    "knn_mean_distance": extra["knn_mean_distance"][pos].item(),
                    "kernel_confidence": extra["kernel_confidence"][pos].item(),
                }
            predictions[(key, f"{prefix}_kernel_knn_{knn_k}")] = {
                "pred_coeffs": pred,
                "zero_field": False,
                "extras": merged,
                "family": f"{prefix}_kernel_knn",
                "knn_k": knn_k,
            }
    return predictions


def best_rows(rows, split="mini_val"):
    def value(row, key, default=-1e9):
        item = row.get(key)
        if item in (None, ""):
            return default
        return float(item)

    candidates = [row for row in rows if row.get("split") == split]
    safe = [
        row
        for row in candidates
        if value(row, "severe_regressions", 999) == 0
        and value(row, "strong_reference_regressions", 999) <= 1
        and value(row, "easy_top25_output_gain", -999) >= -0.02
    ]
    return {
        "best_l1": max(candidates, key=lambda row: value(row, "weighted_delta_l1_drop")) if candidates else None,
        "best_corr": max(candidates, key=lambda row: value(row, "pred_target_corr")) if candidates else None,
        "best_safe_l1": max(safe, key=lambda row: value(row, "weighted_delta_l1_drop")) if safe else None,
        "safe_count": len(safe),
        "candidate_count": len(candidates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4d_spatial_coeff_probe_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=128)
    parser.add_argument("--eval_count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="16,32")
    parser.add_argument("--projection_ridge", type=float, default=1e-5)
    parser.add_argument("--ridge_values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--pls_components", default="4,8,16")
    parser.add_argument("--knn_values", default="5,9")
    parser.add_argument("--pca_oversample", type=int, default=8)
    parser.add_argument("--pca_niter", type=int, default=4)
    parser.add_argument("--spatial_grid", type=int, default=4)
    parser.add_argument("--spatial_proj_channels", type=int, default=8)
    parser.add_argument(
        "--hook_paths",
        default="Encoder.0,Encoder.1,Encoder.2,Decoder.0,Decoder.1,Decoder.2,Convs.0,Convs.1,APDR_1.context",
    )
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    if args.train_count <= 0 or args.eval_count <= args.train_count:
        raise ValueError("--eval_count must be greater than --train_count.")
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

    train_indices = [
        row["index"]
        for row in meta_rows
        if row["index"] < args.train_count and row["low_weight_sum"] > 1e-8
    ]
    mini_indices = [
        row["index"]
        for row in meta_rows
        if args.train_count <= row["index"] < args.eval_count and row["low_weight_sum"] > 1e-8
    ]
    eval_indices = list(range(args.eval_count))
    if not train_indices or not mini_indices:
        raise RuntimeError("Need open train and mini-val samples for spatial probe.")

    flat_targets = low_targets[args.low_size].flatten(1)
    flat_weights = low_weights[args.low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    projections = {}
    for k in k_values:
        if k > pca["bases"].shape[0]:
            continue
        bases = pca["bases"][:k]
        coeffs, recons = weighted_project(
            flat_targets,
            flat_weights,
            pca["mean"],
            bases,
            args.projection_ridge,
        )
        projections[(args.low_size, k)] = {
            "low_size": args.low_size,
            "K": k,
            "coeffs": coeffs,
            "recons": recons,
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k),
        }

    feature_sets = {
        "global": global_features[: args.eval_count],
        "spatial_priors": spatial["spatial_priors"],
        "convir_spatial": spatial["conv_spatial"],
    }
    feature_sets["global_plus_spatial"] = torch.cat(
        [feature_sets["global"], feature_sets["spatial_priors"], feature_sets["convir_spatial"]],
        dim=1,
    )

    predictions = {}
    feature_dims = {}
    for name, matrix in feature_sets.items():
        feature_dims[name] = int(matrix.shape[1])
        predictions.update(
            make_probe_predictions(matrix.float(), projections, train_indices, eval_indices, args, name)
        )

    eval_loader = build_loader(args.data_dir, args.eval_count, args.num_workers, shuffle=False)
    per_image = evaluate_predictions(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        projections,
        predictions,
        eval_indices,
    )
    mapper_rows, coeff_rows, group_rows = aggregate_mapper_rows(
        per_image,
        projections,
        predictions,
        eval_indices,
    )
    open_easy_rows = open_easy_failure_rows(per_image)
    summary = {
        "stage": "APDR-v0.4D frozen ConvIR spatial coefficient probe",
        "tag": args.tag,
        "sigma": args.sigma,
        "correctability_tau": tau,
        "basis_num_images": len(meta_rows),
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "train_open_count": len(train_indices),
        "mini_val_open_count": len(mini_indices),
        "low_size": args.low_size,
        "k_values": [projection["K"] for projection in projections.values()],
        "feature_dims": feature_dims,
        "hook_paths": spatial["hook_paths"],
        "best_rows": best_rows(mapper_rows),
        "global_feature_names": global_names,
        "args": vars(args),
    }

    summary_path = output_dir / f"spatial_coeff_probe_summary_{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / f"spatial_coeff_probe_mapper_family_{label}.csv", mapper_rows)
    write_csv(output_dir / f"spatial_coeff_probe_coeff_error_by_split_{label}.csv", coeff_rows)
    write_csv(output_dir / f"spatial_coeff_probe_groups_{label}.csv", group_rows)
    write_csv_union(output_dir / f"spatial_coeff_probe_per_image_{label}.csv", per_image)
    write_csv(output_dir / f"spatial_coeff_probe_open_easy_failure_{label}.csv", open_easy_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
