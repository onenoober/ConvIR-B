import argparse
import csv
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
from audit_haze4k_apdr_v0_4b_mapping_triage import parse_float_list, write_csv_union  # noqa: E402
from audit_haze4k_apdr_v0_4d_spatial_coeff_probe import (  # noqa: E402
    collect_spatial_feature_matrices,
    make_probe_predictions,
)
from audit_haze4k_apdr_v0_4e_oof_calibration import (  # noqa: E402
    build_fold_projections,
    stratified_folds,
)
from audit_haze4k_apdr_v0_4f_prior_predictability import (  # noqa: E402
    dark_channel,
    estimate_airlight,
    load_depth_tensor,
    luma,
)
from overfit_haze4k_apdr_v0_4_freeparam_lowcolor import (  # noqa: E402
    build_apdr_model,
    build_loader,
    correlation,
    frozen_apdr_tensors,
    gaussian_lowpass,
    percentile,
    psnr,
)
from overfit_haze4k_apdr_v0_4a_lowfield import read_correctability  # noqa: E402


def parse_mapper_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def slug(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def projection_dict_from_full_pca(targets, weights, k_values, args):
    flat_targets = targets.flatten(1)
    flat_weights = weights.repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases(targets, weights, k_values, args)
    projections = {}
    for k_dim in k_values:
        if k_dim > pca["bases"].shape[0]:
            continue
        bases = pca["bases"][:k_dim]
        coeffs, _recons = weighted_project(
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
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k_dim),
        }
    return projections


def build_feature_sets(global_features, spatial):
    feature_sets = {
        "global": global_features,
        "spatial_priors": spatial["spatial_priors"],
        "convir_spatial": spatial["conv_spatial"],
    }
    feature_sets["global_plus_spatial"] = torch.cat(
        [feature_sets["global"], feature_sets["spatial_priors"], feature_sets["convir_spatial"]],
        dim=1,
    )
    return feature_sets


def prepare_v04d(args, apdr_model, device, tau, scores):
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
    eval_count = min(args.eval_count, len(meta_rows))
    eval_loader = build_loader(args.data_dir, eval_count, args.num_workers, shuffle=False)
    spatial = collect_spatial_feature_matrices(apdr_model, eval_loader, device, args, tau, scores)
    k_values = parse_int_list(args.k_values)
    projections = projection_dict_from_full_pca(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    train_indices = [
        row["index"]
        for row in meta_rows
        if row["index"] < args.train_count and row["low_weight_sum"] > 1e-8
    ]
    eval_indices = list(range(eval_count))
    if not train_indices:
        raise RuntimeError("Need at least one open v0.4D train sample.")
    feature_sets = build_feature_sets(global_features[:eval_count], spatial)
    predictions = {}
    for name, matrix in feature_sets.items():
        predictions.update(make_probe_predictions(matrix.float(), projections, train_indices, eval_indices, args, name))
    return {
        "mode": "v04d",
        "data_count": eval_count,
        "global_feature_names": global_names,
        "meta_rows": meta_rows[:eval_count],
        "contexts": {
            0: {
                "eval_indices": eval_indices,
                "index_pos": {int(index): pos for pos, index in enumerate(eval_indices)},
                "predictions": predictions,
                "projections": projections,
            }
        },
        "fold_by_index": {int(index): 0 for index in eval_indices},
    }


def prepare_v04e(args, apdr_model, device, tau, scores):
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
    data_count = len(meta_rows)
    global_names, global_features = feature_matrix(feature_rows)
    folds, fold_by_index = stratified_folds(meta_rows, args.fold_count, args.seed)
    spatial_loader = build_loader(args.data_dir, data_count, args.num_workers, shuffle=False)
    spatial = collect_spatial_feature_matrices(apdr_model, spatial_loader, device, args, tau, scores)
    feature_sets = build_feature_sets(global_features, spatial)
    k_values = parse_int_list(args.k_values)
    active_indices = [row["index"] for row in meta_rows if row["low_weight_sum"] > 1e-8]
    contexts = {}
    for fold_id, eval_indices in enumerate(folds):
        eval_set = set(eval_indices)
        train_indices = [index for index in active_indices if index not in eval_set]
        projections = build_fold_projections(
            low_targets[args.low_size],
            low_weights[args.low_size],
            train_indices,
            k_values,
            args,
        )
        predictions = {}
        for name, matrix in feature_sets.items():
            predictions.update(make_probe_predictions(matrix.float(), projections, train_indices, eval_indices, args, name))
        contexts[fold_id] = {
            "eval_indices": eval_indices,
            "index_pos": {int(index): pos for pos, index in enumerate(eval_indices)},
            "predictions": predictions,
            "projections": projections,
        }
        print(
            f"visual_fold_prepared={fold_id} train_open={len(train_indices)} eval={len(eval_indices)}",
            flush=True,
        )
    return {
        "mode": "v04e",
        "data_count": data_count,
        "global_feature_names": global_names,
        "meta_rows": meta_rows,
        "contexts": contexts,
        "fold_by_index": fold_by_index,
    }


def candidate_pred(context, index, mapper, k_dim, low_size):
    key = (low_size, k_dim)
    item = context["predictions"].get((key, mapper))
    if item is None:
        return None, None
    pos = context["index_pos"].get(int(index))
    if pos is None:
        return None, None
    projection = context["projections"][key]
    pred_coeff = item["pred_coeffs"][pos : pos + 1]
    if item.get("zero_field"):
        pred_low = torch.zeros(1, 3, low_size, low_size)
    else:
        pred_low = (projection["mean"] + pred_coeff[0] @ projection["bases"]).view(1, 3, low_size, low_size)
    return pred_low, item


def tensor_to_image(tensor, tile_size, mode="rgb", value_range=None):
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if mode == "signed_rgb":
        lo, hi = value_range
        tensor = (tensor - lo) / max(hi - lo, 1e-12)
        tensor = tensor.clamp(0, 1)
    elif tensor.shape[0] == 1:
        values = tensor[0]
        if value_range is None:
            lo = float(values.min().item())
            hi = float(values.max().item())
        else:
            lo, hi = value_range
        values = ((values - lo) / max(hi - lo, 1e-12)).clamp(0, 1)
        tensor = values.unsqueeze(0).repeat(3, 1, 1)
    else:
        tensor = tensor[:3].clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    return image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)


def labeled_tile(image, title, tile_size, label_height):
    out = Image.new("RGB", (tile_size, tile_size + label_height), (245, 245, 245))
    out.paste(image, (0, label_height))
    draw = ImageDraw.Draw(out)
    text = str(title)
    draw.rectangle([0, 0, tile_size, label_height], fill=(245, 245, 245))
    draw.text((6, 6), text[:42], fill=(20, 20, 20))
    return out


def make_grid(panels, output_path, tile_size=256, label_height=34, columns=5):
    rows = math.ceil(len(panels) / columns)
    grid = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), (255, 255, 255))
    for idx, (title, image) in enumerate(panels):
        x = (idx % columns) * tile_size
        y = (idx // columns) * (tile_size + label_height)
        grid.paste(labeled_tile(image, title, tile_size, label_height), (x, y))
    grid.save(output_path)


def prior_maps(input_img, depth_cache_dir, depth_split, name, args):
    depth, depth_path = load_depth_tensor(depth_cache_dir, depth_split, name, input_img.shape[-2:])
    depth = depth.to(input_img.device)
    dark = dark_channel(input_img, args.dark_patch)
    bright = input_img.max(dim=1, keepdim=True).values
    airlight = estimate_airlight(input_img, dark, args.airlight_top_frac)
    air_scalar = airlight.mean(dim=1, keepdim=True)
    transmission = ((input_img - airlight).abs().mean(dim=1, keepdim=True) / air_scalar.clamp_min(0.05)).clamp(0, 1)
    return {
        "depth": depth,
        "depth_path": depth_path,
        "transmission": transmission,
        "dark": dark,
        "bright": bright,
        "luma": luma(input_img),
    }


def row_for_candidate(
    mode,
    context_bundle,
    fold_id,
    index,
    image_name,
    anchor,
    label_img,
    m_safe,
    low_target,
    p_benefit,
    proxy_score,
    mapper,
    k_dim,
    scale,
    args,
):
    context = context_bundle["contexts"][fold_id]
    pred_low, item = candidate_pred(context, index, mapper, k_dim, args.low_size)
    if pred_low is None:
        return None, None
    pred = F.interpolate(pred_low.to(anchor.device), size=anchor.shape[-2:], mode="bilinear", align_corners=False)
    pred = gaussian_lowpass(pred, args.kernel_size, args.sigma)
    scaled_pred = float(scale) * pred
    weight = m_safe * float(p_benefit)
    output = (anchor + weight * scaled_pred).clamp(0, 1)
    anchor_psnr = psnr(anchor, label_img)
    output_psnr = psnr(output, label_img)
    oracle = (anchor + weight * low_target).clamp(0, 1)
    expanded = weight.expand_as(low_target)
    corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
    row = {
        "mode": mode,
        "split": "oof" if mode == "v04e" else ("train" if index < args.train_count else "mini_val"),
        "fold": fold_id,
        "index": index,
        "name": image_name,
        "mapper": mapper,
        "K": k_dim,
        "candidate_scale": float(scale),
        "family": item.get("family"),
        "anchor_psnr": anchor_psnr,
        "output_psnr": output_psnr,
        "oracle_psnr": psnr(oracle, label_img),
        "output_gain": output_psnr - anchor_psnr,
        "oracle_gain": psnr(oracle, label_img) - anchor_psnr,
        "corr": corr,
        "P_benefit": float(p_benefit),
        "proxy_score": float(proxy_score),
        "M_safe_mean": m_safe.mean().item(),
        "M_safe_nonzero_frac": (m_safe > 1e-6).float().mean().item(),
        "pred_abs_mean": pred.abs().mean().item(),
        "scaled_pred_abs_mean": scaled_pred.abs().mean().item(),
        "target_abs_mean": low_target.abs().mean().item(),
    }
    tensors = {
        "pred": pred,
        "scaled_pred": scaled_pred,
        "output": output,
    }
    return row, tensors


def collect_candidate_rows(apdr_model, loader, device, args, tau, scores, context_bundle):
    rows = []
    mapper_names = parse_mapper_list(args.candidate_mappers)
    k_values = parse_int_list(args.candidate_k_values)
    scales = parse_float_list(args.scales)
    missing = set()
    for index, (input_img, label_img, name) in enumerate(loader):
        if index >= context_bundle["data_count"]:
            break
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
        fold_id = context_bundle["fold_by_index"].get(index)
        if fold_id is None:
            continue
        for mapper in mapper_names:
            for k_dim in k_values:
                context = context_bundle["contexts"][fold_id]
                if ((args.low_size, k_dim), mapper) not in context["predictions"]:
                    missing.add((mapper, k_dim, fold_id))
                    continue
                for scale in scales:
                    row, _tensors = row_for_candidate(
                        context_bundle["mode"],
                        context_bundle,
                        fold_id,
                        index,
                        image_name,
                        anchor,
                        label_img,
                        m_safe,
                        low_target,
                        p_benefit,
                        score,
                        mapper,
                        k_dim,
                        scale,
                        args,
                    )
                    if row is not None:
                        rows.append(row)
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"visual_candidate_evaluated={index + 1}", flush=True)
    if missing:
        print(
            "missing_candidates="
            + ",".join(f"{mapper}/K{k}/fold{fold}" for mapper, k, fold in sorted(missing)[:20]),
            flush=True,
        )
    return rows


def select_rows(rows, args):
    if args.mode == "v04d" and args.v04d_select_split:
        rows = [row for row in rows if row["split"] == args.v04d_select_split]
    anchor_by_index = {}
    for row in rows:
        anchor_by_index.setdefault(int(row["index"]), float(row["anchor_psnr"]))
    strong_cut = percentile(list(anchor_by_index.values()), args.strong_anchor_percentile)
    severe = [row for row in rows if float(row["output_gain"]) <= args.severe_delta]
    strong = [
        row
        for row in rows
        if strong_cut is not None
        and float(row["anchor_psnr"]) >= strong_cut
        and float(row["output_gain"]) <= args.strong_delta
    ]
    selected = []
    seen = set()
    for group, members in (("severe", severe), ("strong", strong)):
        members = sorted(members, key=lambda row: (float(row["output_gain"]), -float(row["anchor_psnr"])))
        count = 0
        for row in members:
            key = (
                int(row["index"]),
                row["mapper"],
                int(row["K"]),
                float(row["candidate_scale"]),
                group,
            )
            if key in seen:
                continue
            item = dict(row)
            item["visual_group"] = group
            item["strong_anchor_cut_psnr"] = strong_cut
            selected.append(item)
            seen.add(key)
            count += 1
            if count >= args.max_per_group:
                break
    return selected


def render_selected(apdr_model, loader, device, args, tau, scores, context_bundle, selected_rows, output_dir):
    by_index = {}
    for row in selected_rows:
        by_index.setdefault(int(row["index"]), []).append(row)
    rendered = []
    for index, (input_img, label_img, name) in enumerate(loader):
        if index not in by_index:
            continue
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
        priors = prior_maps(input_img, args.depth_cache_dir, args.depth_split, image_name, args)
        for row in by_index[index]:
            fold_id = int(row["fold"])
            metric, tensors = row_for_candidate(
                context_bundle["mode"],
                context_bundle,
                fold_id,
                index,
                image_name,
                anchor,
                label_img,
                m_safe,
                low_target,
                p_benefit,
                score,
                row["mapper"],
                int(row["K"]),
                float(row["candidate_scale"]),
                args,
            )
            if tensors is None:
                continue
            file_name = (
                f"{args.mode}_{row['visual_group']}_idx{index:04d}_"
                f"{slug(image_name)}_{slug(row['mapper'])}_K{int(row['K'])}_"
                f"s{float(row['candidate_scale']):.2f}.png"
            )
            output_path = output_dir / file_name
            panels = [
                ("input", tensor_to_image(input_img, args.tile_size)),
                (f"anchor {metric['anchor_psnr']:.2f}", tensor_to_image(anchor, args.tile_size)),
                (
                    f"candidate d={metric['output_gain']:+.2f}",
                    tensor_to_image(tensors["output"], args.tile_size),
                ),
                ("GT", tensor_to_image(label_img, args.tile_size)),
                ("M_safe", tensor_to_image(m_safe, args.tile_size, value_range=(0.0, 1.0))),
                (
                    "pred residual",
                    tensor_to_image(
                        tensors["scaled_pred"],
                        args.tile_size,
                        mode="signed_rgb",
                        value_range=(-args.residual_max, args.residual_max),
                    ),
                ),
                ("depth", tensor_to_image(priors["depth"], args.tile_size, value_range=(0.0, 1.0))),
                (
                    "transmission proxy",
                    tensor_to_image(priors["transmission"], args.tile_size, value_range=(0.0, 1.0)),
                ),
                ("dark", tensor_to_image(priors["dark"], args.tile_size, value_range=(0.0, 1.0))),
                ("bright", tensor_to_image(priors["bright"], args.tile_size, value_range=(0.0, 1.0))),
            ]
            make_grid(panels, output_path, args.tile_size, args.label_height, args.grid_columns)
            rendered.append({**row, **metric, "grid_path": str(output_path), "depth_path": priors["depth_path"]})
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["v04d", "v04e"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4_visual_regression_grids_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=128)
    parser.add_argument("--eval_count", type=int, default=256)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="16")
    parser.add_argument("--candidate_k_values", default="16")
    parser.add_argument(
        "--candidate_mappers",
        default=(
            "global_plus_spatial_kernel_knn_9,"
            "convir_spatial_kernel_knn_9,"
            "spatial_priors_kernel_knn_9,"
            "spatial_priors_ridge_10,"
            "global_mean_coeff"
        ),
    )
    parser.add_argument("--scales", default="1.00")
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
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--dark_patch", type=int, default=15)
    parser.add_argument("--airlight_top_frac", type=float, default=0.01)
    parser.add_argument("--severe_delta", type=float, default=-0.20)
    parser.add_argument("--strong_delta", type=float, default=-0.05)
    parser.add_argument("--strong_anchor_percentile", type=float, default=75.0)
    parser.add_argument("--max_per_group", type=int, default=12)
    parser.add_argument("--v04d_select_split", default="mini_val", choices=["", "train", "mini_val"])
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--label_height", type=int, default=34)
    parser.add_argument("--grid_columns", type=int, default=5)
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

    label = sigma_label(args.sigma)
    output_dir = Path(args.output_dir)
    grid_dir = output_dir / f"{args.mode}_grids_{label}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    tau, scores = read_correctability(args.correctability_json, args.correctability_train_csv)
    apdr_model = build_apdr_model(args.selector_checkpoint, args.residual_max, device)
    if args.mode == "v04d":
        context_bundle = prepare_v04d(args, apdr_model, device, tau, scores)
    else:
        context_bundle = prepare_v04e(args, apdr_model, device, tau, scores)

    eval_count = context_bundle["data_count"]
    loader = build_loader(args.data_dir, eval_count, args.num_workers, shuffle=False)
    candidate_rows = collect_candidate_rows(apdr_model, loader, device, args, tau, scores, context_bundle)
    selected_rows = select_rows(candidate_rows, args)
    render_loader = build_loader(args.data_dir, eval_count, args.num_workers, shuffle=False)
    rendered_rows = render_selected(
        apdr_model,
        render_loader,
        device,
        args,
        tau,
        scores,
        context_bundle,
        selected_rows,
        grid_dir,
    )

    candidate_path = output_dir / f"{args.mode}_visual_candidate_rows_{label}.csv"
    selected_path = output_dir / f"{args.mode}_visual_selected_rows_{label}.csv"
    rendered_path = output_dir / f"{args.mode}_visual_rendered_rows_{label}.csv"
    write_csv_union(candidate_path, candidate_rows)
    write_csv_union(selected_path, selected_rows)
    write_csv_union(rendered_path, rendered_rows)
    summary = {
        "stage": "APDR-v0.4 severe/strong regression visual grids",
        "tag": args.tag,
        "mode": args.mode,
        "data_count": eval_count,
        "candidate_row_count": len(candidate_rows),
        "selected_row_count": len(selected_rows),
        "rendered_row_count": len(rendered_rows),
        "severe_delta": args.severe_delta,
        "strong_delta": args.strong_delta,
        "strong_anchor_percentile": args.strong_anchor_percentile,
        "outputs": {
            "candidate_rows": str(candidate_path),
            "selected_rows": str(selected_path),
            "rendered_rows": str(rendered_path),
            "grid_dir": str(grid_dir),
        },
        "args": vars(args),
        "global_feature_names": context_bundle["global_feature_names"],
    }
    summary_path = output_dir / f"{args.mode}_visual_regression_grid_summary_{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
