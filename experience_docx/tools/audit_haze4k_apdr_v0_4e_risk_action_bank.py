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
    parse_float_list,
    standardizer,
    write_csv_union,
)
from audit_haze4k_apdr_v0_4d_spatial_coeff_probe import (  # noqa: E402
    collect_spatial_feature_matrices,
    make_probe_predictions,
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


DEFAULT_RULES = [
    {
        "name": "RuleA_global_plus_spatial_kenel_knn_9_K16_pred_abs_mean_high",
        "mapper": "global_plus_spatial_kenel_knn_9",
        "K": 16,
        "scale": 1.0,
        "confidence_key": "pred_abs_mean",
        "direction": "high",
        "threshold": 0.010107760690152645,
    },
    {
        "name": "RuleB_spatial_priors_ridge_10_K16_pred_abs_mean_high",
        "mapper": "spatial_priors_ridge_10",
        "K": 16,
        "scale": 1.0,
        "confidence_key": "pred_abs_mean",
        "direction": "high",
        "threshold": 0.01394791239872575,
    },
]


def safe_float(value, default=None):
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values):
    values = [safe_float(value) for value in values]
    values = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(values) if values else None


def parse_mapper_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_rules(value):
    if not value:
        return DEFAULT_RULES
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def mapper_name_for_output(mapper_name, wanted_mappers):
    if mapper_name in wanted_mappers:
        return mapper_name
    aliases = []
    if "_kernel_" in mapper_name:
        aliases.append(mapper_name.replace("_kernel_", "_kenel_"))
    if "_kenel_" in mapper_name:
        aliases.append(mapper_name.replace("_kenel_", "_kernel_"))
    for alias in aliases:
        if alias in wanted_mappers:
            return alias
    return None


def normalize_extra_aliases(extra):
    out = dict(extra)
    if "kernel_confidence" in out and "kenel_confidence" not in out:
        out["kenel_confidence"] = out["kernel_confidence"]
    if "kenel_confidence" in out and "kernel_confidence" not in out:
        out["kernel_confidence"] = out["kenel_confidence"]
    return out


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rank_group(anchor_psnr, hard_cut, easy_cut):
    if anchor_psnr <= hard_cut:
        return "hard_bottom25"
    if anchor_psnr >= easy_cut:
        return "easy_top25"
    return "middle50"


def summarize_policy(rows, keep_mask):
    adjusted = []
    for row, keep in zip(rows, keep_mask):
        output_gain = row["output_gain"] if keep else 0.0
        final_l1_num = row["final_l1_num"] if keep else row["initial_l1_num"]
        adjusted.append({**row, "policy_output_gain": output_gain, "policy_final_l1_num": final_l1_num, "kept": keep})
    den = sum(row["weighted_den"] for row in adjusted)
    initial = sum(row["initial_l1_num"] for row in adjusted) / max(den, 1e-12)
    final = sum(row["policy_final_l1_num"] for row in adjusted) / max(den, 1e-12)
    ordered = sorted(adjusted, key=lambda row: row["anchor_psnr"])
    hard = ordered[: max(1, len(ordered) // 4)]
    easy = ordered[3 * len(ordered) // 4 :]
    strong_cut = percentile([row["anchor_psnr"] for row in adjusted], 75)
    strong = [row for row in adjusted if row["anchor_psnr"] >= strong_cut]
    positive = [row for row in adjusted if row["oracle_gain"] > 1e-6]
    oracle_sum = sum(row["oracle_gain"] for row in positive)
    return {
        "count": len(adjusted),
        "open_count": sum(row["P_benefit"] >= 0.5 for row in adjusted),
        "keep_count": sum(1 for keep in keep_mask if keep),
        "open_keep_count": sum(row["P_benefit"] >= 0.5 and keep for row, keep in zip(adjusted, keep_mask)),
        "initial_weighted_delta_l1": initial,
        "projection_weighted_delta_l1": final,
        "weighted_delta_l1_drop": (initial - final) / max(initial, 1e-12),
        "mean_gain": safe_mean([row["policy_output_gain"] for row in adjusted]),
        "mean_oracle_gain": safe_mean([row["oracle_gain"] for row in adjusted]),
        "oracle_recovery": sum(row["policy_output_gain"] for row in positive) / max(oracle_sum, 1e-12),
        "hard_bottom25_gain": safe_mean([row["policy_output_gain"] for row in hard]),
        "hard_bottom25_oracle_gain": safe_mean([row["oracle_gain"] for row in hard]),
        "easy_top25_gain": safe_mean([row["policy_output_gain"] for row in easy]),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regressions": sum(row["policy_output_gain"] <= -0.05 for row in strong),
        "severe_regressions": sum(row["policy_output_gain"] <= -0.20 for row in adjusted),
    }


def gate_pass(summary, min_keep):
    return (
        summary["severe_regressions"] == 0
        and summary["strong_regressions"] <= 1
        and summary["easy_top25_gain"] >= -0.02
        and summary["mean_gain"] >= 0.05
        and summary["hard_bottom25_gain"] >= 0.25
        and summary["keep_count"] >= min_keep
    )


def auc_score(values, labels):
    paired = [(float(value), bool(label)) for value, label in zip(values, labels) if value is not None]
    pos = [item for item in paired if item[1]]
    neg = [item for item in paired if not item[1]]
    if not pos or not neg:
        return None
    ordered = sorted(paired, key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][0] == ordered[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            if ordered[j][1]:
                rank_sum += avg_rank
        idx = end
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def bad_labels(rows):
    strong_cut = percentile([row["anchor_psnr"] for row in rows], 75)
    labels = []
    for row in rows:
        strong_bad = row["anchor_psnr"] >= strong_cut and row["output_gain"] <= -0.05
        severe_bad = row["output_gain"] <= -0.20
        small_bad = row["output_gain"] < -0.02
        labels.append(strong_bad or severe_bad or small_bad)
    return labels


def risk_feature_auc_rows(rows, feature_keys, split_name):
    out = []
    scopes = [("overall", rows)]
    for mapper, k_dim, scale in sorted({(row["mapper"], row["K"], row["candidate_scale"]) for row in rows}):
        scopes.append(
            (
                f"{mapper}|K{k_dim}|scale{scale:g}",
                [
                    row
                    for row in rows
                    if row["mapper"] == mapper and row["K"] == k_dim and row["candidate_scale"] == scale
                ],
            )
        )
    for scope, members in scopes:
        if len(members) < 4:
            continue
        labels = bad_labels(members)
        bad_count = sum(labels)
        for key in feature_keys:
            values = [safe_float(row.get(key)) for row in members]
            auc = auc_score(values, labels)
            if auc is None:
                continue
            out.append(
                {
                    "split": split_name,
                    "scope": scope,
                    "feature": key,
                    "count": len(members),
                    "bad_count": bad_count,
                    "auc_high_is_bad": auc,
                    "best_auc": max(auc, 1.0 - auc),
                    "best_direction_for_bad": "high" if auc >= 0.5 else "low",
                }
            )
    return out


def calibration_curve(rows, score_key, split_name, bins):
    scored = [row for row in rows if safe_float(row.get(score_key)) is not None]
    if not scored:
        return []
    scored = sorted(scored, key=lambda row: safe_float(row[score_key]), reverse=True)
    labels = bad_labels(scored)
    out = []
    for bin_idx in range(bins):
        start = math.floor(len(scored) * bin_idx / bins)
        end = math.floor(len(scored) * (bin_idx + 1) / bins)
        members = scored[start:end]
        member_labels = labels[start:end]
        if not members:
            continue
        out.append(
            {
                "split": split_name,
                "score_key": score_key,
                "bin": bin_idx + 1,
                "order": "descending_confidence",
                "count": len(members),
                "coverage_start_frac": start / len(scored),
                "coverage_end_frac": end / len(scored),
                "score_min": min(safe_float(row[score_key]) for row in members),
                "score_max": max(safe_float(row[score_key]) for row in members),
                "bad_rate": sum(member_labels) / len(member_labels),
                "mean_gain": safe_mean([row["output_gain"] for row in members]),
                "hard_bottom25_frac": sum(row["rank_group"] == "hard_bottom25" for row in members) / len(members),
                "easy_top25_frac": sum(row["rank_group"] == "easy_top25" for row in members) / len(members),
            }
        )
    return out


def accepted_rejected_group_rows(rule_rows, split_rows):
    out = []
    for rule_row in rule_rows:
        kept = {row["index"] for row in rule_row["kept_rows"]}
        for status in ("accepted", "rejected"):
            members = [
                row
                for row in split_rows
                if (row["index"] in kept) == (status == "accepted")
                and row["mapper"] == rule_row["mapper"]
                and row["K"] == rule_row["K"]
                and abs(row["candidate_scale"] - rule_row["scale"]) < 1e-12
            ]
            for group in ("hard_bottom25", "middle50", "easy_top25"):
                group_members = [row for row in members if row["rank_group"] == group]
                if not group_members:
                    continue
                out.append(
                    {
                        "rule": rule_row["name"],
                        "mapper": rule_row["mapper"],
                        "K": rule_row["K"],
                        "scale": rule_row["scale"],
                        "status": status,
                        "rank_group": group,
                        "count": len(group_members),
                        "open_count": sum(row["P_benefit"] >= 0.5 for row in group_members),
                        "mean_gain": safe_mean([row["output_gain"] for row in group_members]) if status == "accepted" else 0.0,
                        "mean_candidate_gain": safe_mean([row["output_gain"] for row in group_members]),
                        "mean_oracle_gain": safe_mean([row["oracle_gain"] for row in group_members]),
                        "mean_pred_abs_mean": safe_mean([row["pred_abs_mean"] for row in group_members]),
                        "mean_nn_distance": safe_mean([row.get("nn_distance") for row in group_members]),
                    }
                )
    return out


def failure_signature_rows(rows, feature_keys, split_name):
    out = []
    for mapper, k_dim, scale in sorted({(row["mapper"], row["K"], row["candidate_scale"]) for row in rows}):
        members = [
            row
            for row in rows
            if row["mapper"] == mapper and row["K"] == k_dim and row["candidate_scale"] == scale
        ]
        labels = bad_labels(members)
        bad = [row for row, label in zip(members, labels) if label]
        good = [row for row, label in zip(members, labels) if not label]
        if not bad:
            continue
        row = {
            "split": split_name,
            "mapper": mapper,
            "K": k_dim,
            "scale": scale,
            "count": len(members),
            "bad_count": len(bad),
            "bad_rate": len(bad) / len(members),
            "bad_mean_gain": safe_mean([item["output_gain"] for item in bad]),
            "good_mean_gain": safe_mean([item["output_gain"] for item in good]),
            "worst_name": min(bad, key=lambda item: item["output_gain"])["name"],
            "worst_gain": min(item["output_gain"] for item in bad),
        }
        for key in feature_keys:
            row[f"bad_{key}_mean"] = safe_mean([item.get(key) for item in bad])
            row[f"good_{key}_mean"] = safe_mean([item.get(key) for item in good])
        out.append(row)
    return out


def candidate_table_rows(rows):
    out = []
    for split_name in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split_name]
        for mapper, k_dim, scale in sorted(
            {
                (row["mapper"], row["K"], row["candidate_scale"])
                for row in split_rows
            }
        ):
            group_rows = [
                row
                for row in split_rows
                if row["mapper"] == mapper and row["K"] == k_dim and row["candidate_scale"] == scale
            ]
            summary = summarize_policy(group_rows, [True] * len(group_rows))
            out.append(
                {
                    "split": split_name,
                    "mapper": mapper,
                    "K": k_dim,
                    "scale": scale,
                    "policy": "apply_all_candidate",
                    **summary,
                }
            )
    return out


def evaluate_action_bank(apdr_model, loader, device, args, tau, scores, projections, predictions, prediction_indices):
    rows = []
    pos_by_index = {int(image_index): pos for pos, image_index in enumerate(prediction_indices)}
    wanted_mappers = set(args.candidate_mappers)
    wanted_k = set(args.candidate_k_values)
    predict_set = set(prediction_indices)
    fit_range = set(range(args.train_start, args.train_start + args.train_count))
    confirm_range = set(range(args.eval_start, args.eval_start + args.eval_count))
    for index, (input_img, label_img, name) in enumerate(loader):
        if index not in predict_set:
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
        weight = m_safe * p_benefit
        expanded = weight.expand_as(low_target)
        oracle = (anchor + weight * low_target).clamp(0, 1)
        anchor_psnr = psnr(anchor, label_img)
        oracle_psnr = psnr(oracle, label_img)
        if index in fit_range:
            split = args.fit_split_name
        elif index in confirm_range:
            split = args.eval_split_name
        else:
            split = "other"
        for (key, mapper_name), item in predictions.items():
            low_size, k_dim = key
            output_mapper_name = mapper_name_for_output(mapper_name, wanted_mappers)
            if output_mapper_name is None or k_dim not in wanted_k:
                continue
            projection = projections[key]
            pred_pos = pos_by_index[index]
            pred_coeff = item["pred_coeffs"][pred_pos : pred_pos + 1]
            if item.get("zero_field"):
                pred_low = torch.zeros(1, 3, low_size, low_size)
            else:
                pred_low = (projection["mean"] + pred_coeff[0] @ projection["bases"]).view(1, 3, low_size, low_size)
            pred = F.interpolate(pred_low.to(device), size=anchor.shape[-2:], mode="bilinear", align_corners=False)
            pred = gaussian_lowpass(pred, args.kernel_size, args.sigma)
            corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
            initial_num = (expanded * low_target.abs()).sum().item()
            weighted_den = expanded.sum().item()
            pred_abs = pred.abs()
            pred_flat = pred.flatten()
            positive_frac = (pred_flat > 0).float().mean().item()
            negative_frac = (pred_flat < 0).float().mean().item()
            pred_coeff_norm = pred_coeff.norm(dim=1).item()
            pred_coeff_l1 = pred_coeff.abs().mean().item()
            extra = normalize_extra_aliases(item.get("extras", {}).get(index, {}))
            for scale in args.scales:
                scaled_pred = float(scale) * pred
                output = (anchor + weight * scaled_pred).clamp(0, 1)
                output_psnr = psnr(output, label_img)
                final_num = (expanded * (scaled_pred - low_target).abs()).sum().item()
                rows.append(
                    {
                        "low_size": low_size,
                        "K": k_dim,
                        "mapper": output_mapper_name,
                        "family": item.get("family"),
                        "candidate_scale": float(scale),
                        "split": split,
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
                        "M_safe_nonzero_frac": (m_safe > 1e-6).float().mean().item(),
                        "target_abs_mean": low_target.abs().mean().item(),
                        "pred_abs_mean": pred_abs.mean().item(),
                        "pred_abs_max": pred_abs.max().item(),
                        "pred_coeff_norm": pred_coeff_norm,
                        "pred_coeff_l1": pred_coeff_l1,
                        "pred_low_energy": pred.square().mean().item(),
                        "scaled_pred_abs_mean": scaled_pred.abs().mean().item(),
                        "weighted_residual_norm": (expanded * scaled_pred.abs()).sum().item() / max(weighted_den, 1e-12),
                        "residual_mean": pred.mean().item(),
                        "residual_std": pred.std(unbiased=False).item(),
                        "residual_positive_frac": positive_frac,
                        "residual_negative_frac": negative_frac,
                        "weighted_den": weighted_den,
                        "initial_l1_num": initial_num,
                        "final_l1_num": final_num,
                        "weighted_field_l1": final_num / max(weighted_den, 1e-12),
                        **extra,
                    }
                )
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"v04e_evaluated={index + 1}", flush=True)
    return rows


def attach_rank_groups(rows, eval_split_name):
    for split_name in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split_name]
        anchor_values = [row["anchor_psnr"] for row in split_rows]
        hard_cut = percentile(anchor_values, 25)
        easy_cut = percentile(anchor_values, 75)
        for row in split_rows:
            row["rank_group"] = rank_group(row["anchor_psnr"], hard_cut, easy_cut)
            row["open_rank_group"] = ("open_" if row["P_benefit"] >= 0.5 else "closed_") + row["rank_group"]
            row["eval_target_split"] = row["split"] == eval_split_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4e_risk_action_bank_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_count", type=int, default=128)
    parser.add_argument("--eval_start", type=int, default=256)
    parser.add_argument("--eval_count", type=int, default=128)
    parser.add_argument("--fit_split_name", default="fit")
    parser.add_argument("--eval_split_name", default="confirm")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kernel_size", "--kenel_size", dest="kernel_size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--k_values", default="16")
    parser.add_argument("--candidate_k_values", default="16")
    parser.add_argument(
        "--candidate_mappers",
        default=(
            "global_plus_spatial_kenel_knn_9,"
            "convir_spatial_kenel_knn_9,"
            "spatial_priors_kenel_knn_9,"
            "spatial_priors_ridge_10,"
            "global_mean_coeff"
        ),
    )
    parser.add_argument("--scales", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--locked_rules", default="")
    parser.add_argument("--locked_min_keep", type=int, default=15)
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
    parser.add_argument(
        "--risk_feature_keys",
        default=(
            "pred_abs_mean,pred_abs_max,pred_coeff_norm,pred_low_energy,weighted_residual_norm,"
            "nn_distance,confidence_proxy,kenel_confidence,proxy_score,M_safe_mean,M_safe_nonzero_frac"
        ),
    )
    parser.add_argument("--calibration_score_key", default="pred_abs_mean")
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--progress_freq", type=int, default=100)
    args = parser.parse_args()

    if args.train_count <= 0 or args.eval_count <= 0:
        raise ValueError("train_count and eval_count must be positive.")
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
    args.scales = parse_float_list(args.scales)
    args.candidate_mappers = parse_mapper_list(args.candidate_mappers)
    args.candidate_k_values = parse_int_list(args.candidate_k_values)
    k_values = parse_int_list(args.k_values)
    locked_rules = parse_rules(args.locked_rules)
    risk_feature_keys = parse_mapper_list(args.risk_feature_keys)
    label = sigma_label(args.sigma)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_indices_all = list(range(args.train_start, args.train_start + args.train_count))
    confirm_indices_all = list(range(args.eval_start, args.eval_start + args.eval_count))
    prediction_indices = sorted(set(fit_indices_all + confirm_indices_all))
    spatial_collect_count = max(prediction_indices) + 1

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
    spatial_loader = build_loader(args.data_dir, spatial_collect_count, args.num_workers, shuffle=False)
    spatial = collect_spatial_feature_matrices(apdr_model, spatial_loader, device, args, tau, scores)

    train_indices = [
        index
        for index in fit_indices_all
        if index < len(meta_rows) and meta_rows[index]["low_weight_sum"] > 1e-8
    ]
    if not train_indices:
        raise RuntimeError("Need open fit samples for v0.4E action-bank mappers.")

    flat_targets = low_targets[args.low_size].flatten(1)
    flat_weights = low_weights[args.low_size].repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases(low_targets[args.low_size], low_weights[args.low_size], k_values, args)
    projections = {}
    for k_dim in k_values:
        if k_dim > pca["bases"].shape[0]:
            continue
        bases = pca["bases"][:k_dim]
        coeffs, _recons = weighted_project(flat_targets, flat_weights, pca["mean"], bases, args.projection_ridge)
        projections[(args.low_size, k_dim)] = {
            "low_size": args.low_size,
            "K": k_dim,
            "coeffs": coeffs,
            "bases": bases,
            "mean": pca["mean"],
            "explained_weighted_energy": pca["explained"].get(k_dim),
        }

    feature_sets = {
        "global": global_features[:spatial_collect_count],
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
        predictions.update(make_probe_predictions(matrix.float(), projections, train_indices, prediction_indices, args, name))

    eval_loader = build_loader(args.data_dir, spatial_collect_count, args.num_workers, shuffle=False)
    action_rows = evaluate_action_bank(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        projections,
        predictions,
        prediction_indices,
    )
    attach_rank_groups(action_rows, args.eval_split_name)
    confirm_rows = [row for row in action_rows if row["split"] == args.eval_split_name]
    candidate_rows = candidate_table_rows(action_rows)

    locked_rows = []
    locked_group_rule_rows = []
    for rule in locked_rules:
        members = [
            row
            for row in confirm_rows
            if row["mapper"] == rule["mapper"]
            and int(row["K"]) == int(rule["K"])
            and abs(float(row["candidate_scale"]) - float(rule.get("scale", 1.0))) < 1e-12
        ]
        if not members:
            locked_rows.append({**rule, "status": "missing_candidate", "gate_pass": False})
            continue
        key = rule["confidence_key"]
        if rule["direction"] == "low":
            keep_mask = [safe_float(row.get(key), float("inf")) <= float(rule["threshold"]) for row in members]
        else:
            keep_mask = [safe_float(row.get(key), -float("inf")) >= float(rule["threshold"]) for row in members]
        summary = summarize_policy(members, keep_mask)
        passed = gate_pass(summary, args.locked_min_keep)
        locked_rows.append(
            {
                **rule,
                "split": args.eval_split_name,
                "gate_pass": passed,
                "gate": "severe=0,strong<=1,easy>=-0.02,mean>=0.05,hard>=0.25,keep>=min_keep",
                **summary,
            }
        )
        locked_group_rule_rows.append({**rule, "kept_rows": [row for row, keep in zip(members, keep_mask) if keep]})

    auc_rows = risk_feature_auc_rows(confirm_rows, risk_feature_keys, args.eval_split_name)
    curve_rows = calibration_curve(confirm_rows, args.calibration_score_key, args.eval_split_name, args.calibration_bins)
    group_rows = accepted_rejected_group_rows(locked_group_rule_rows, confirm_rows)
    failure_rows = failure_signature_rows(confirm_rows, risk_feature_keys, args.eval_split_name)

    action_path = output_dir / f"v04e_candidate_action_per_image_{label}.csv"
    write_csv_union(action_path, action_rows)
    write_csv_union(output_dir / "v04e_candidate_action_table.csv", candidate_rows)
    write_csv_union(output_dir / "v04e_risk_feature_auc.csv", auc_rows)
    write_csv_union(output_dir / "v04e_oof_calibration_curve.csv", curve_rows)
    write_csv_union(output_dir / "v04e_accepted_vs_rejected_groups.csv", group_rows)
    write_csv_union(output_dir / "v04e_strong_failure_signature.csv", failure_rows)

    summary = {
        "stage": "APDR-v0.4E Risk-Calibrated Selective Action Bank intermediate audit",
        "tag": args.tag,
        "status": "locked threshold confirmation plus candidate action intermediate tables; no training",
        "sigma": args.sigma,
        "correctability_tau": tau,
        "basis_num_images": len(meta_rows),
        "train_start": args.train_start,
        "train_count": args.train_count,
        "eval_start": args.eval_start,
        "eval_count": args.eval_count,
        "fit_open_count": len(train_indices),
        "confirm_count": len(confirm_indices_all),
        "confirm_open_count": sum(
            meta_rows[index]["low_weight_sum"] > 1e-8 for index in confirm_indices_all if index < len(meta_rows)
        ),
        "low_size": args.low_size,
        "k_values": [projection["K"] for projection in projections.values()],
        "candidate_mappers": args.candidate_mappers,
        "candidate_k_values": args.candidate_k_values,
        "candidate_scales": args.scales,
        "locked_rules": locked_rows,
        "all_locked_rules_pass": all(row.get("gate_pass") for row in locked_rows),
        "feature_dims": feature_dims,
        "hook_paths": spatial["hook_paths"],
        "outputs": {
            "candidate_action_per_image": str(action_path),
            "candidate_action_table": str(output_dir / "v04e_candidate_action_table.csv"),
            "risk_feature_auc": str(output_dir / "v04e_risk_feature_auc.csv"),
            "oof_calibration_curve": str(output_dir / "v04e_oof_calibration_curve.csv"),
            "accepted_vs_rejected_groups": str(output_dir / "v04e_accepted_vs_rejected_groups.csv"),
            "strong_failure_signature": str(output_dir / "v04e_strong_failure_signature.csv"),
        },
        "args": vars(args),
        "global_feature_names": global_names,
    }
    summary_path = output_dir / "v04e_locked_threshold_confirm_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
