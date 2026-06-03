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
from audit_haze4k_apdr_v0_4b_mapping_triage import parse_float_list, write_csv_union  # noqa: E402
from audit_haze4k_apdr_v0_4d_spatial_coeff_probe import (  # noqa: E402
    collect_spatial_feature_matrices,
    make_probe_predictions,
)
from audit_haze4k_apdr_v0_4e_risk_action_bank import (  # noqa: E402
    accepted_rejected_group_rows,
    attach_rank_groups,
    bad_labels,
    calibration_curve,
    candidate_table_rows,
    failure_signature_rows,
    parse_mapper_list,
    parse_rules,
    risk_feature_auc_rows,
    safe_float,
    safe_mean,
    summarize_policy,
    write_json,
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


def bucket(value, cuts):
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return idx
    return len(cuts)


def stratified_folds(meta_rows, fold_count, seed):
    anchor_values = [row["anchor_psnr"] for row in meta_rows]
    msafe_values = [row["M_safe_mean"] for row in meta_rows]
    anchor_cuts = [percentile(anchor_values, pct) for pct in (20, 40, 60, 80)]
    msafe_cuts = [percentile(msafe_values, pct) for pct in (20, 40, 60, 80)]
    buckets = {}
    count = len(meta_rows)
    for row in meta_rows:
        index = int(row["index"])
        key = (
            bucket(row["anchor_psnr"], anchor_cuts),
            int(row["P_benefit"] >= 0.5),
            bucket(row["M_safe_mean"], msafe_cuts),
            min(fold_count - 1, int(index * fold_count / max(count, 1))),
        )
        buckets.setdefault(key, []).append(index)
    ng = random.Random(seed)
    folds = [[] for _ in range(fold_count)]
    for key in sorted(buckets):
        members = buckets[key]
        ng.shuffle(members)
        for pos, index in enumerate(members):
            folds[pos % fold_count].append(index)
    for fold in folds:
        fold.sort()
    fold_by_index = {}
    for fold_id, fold in enumerate(folds):
        for index in fold:
            fold_by_index[index] = fold_id
    return folds, fold_by_index


def pca_bases_from_indices(targets, weights, train_indices, k_values, args):
    train_indices = torch.tensor(train_indices, dtype=torch.long)
    train_targets = targets[train_indices]
    train_weights = weights[train_indices]
    return pca_bases(train_targets, train_weights, k_values, args)


def build_fold_projections(targets, weights, train_indices, k_values, args):
    flat_targets = targets.flatten(1)
    flat_weights = weights.repeat(1, 3, 1, 1).flatten(1).clamp_min(0.0)
    pca = pca_bases_from_indices(targets, weights, train_indices, k_values, args)
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
    return projections


def fold_assignment_rows(meta_rows, folds):
    fold_by_index = {}
    for fold_id, fold in enumerate(folds):
        for index in fold:
            fold_by_index[index] = fold_id
    rows = []
    for row in meta_rows:
        rows.append(
            {
                "index": row["index"],
                "name": row["name"],
                "fold": fold_by_index[row["index"]],
                "P_benefit": row["P_benefit"],
                "proxy_score": row["proxy_score"],
                "anchor_psnr": row["anchor_psnr"],
                "M_safe_mean": row["M_safe_mean"],
                "low_weight_sum": row["low_weight_sum"],
            }
        )
    return rows


def evaluate_oof_action_bank(apdr_model, loader, device, args, tau, scores, fold_by_index, fold_predictions, fold_projections):
    rows = []
    wanted_mappers = set(args.candidate_mappers)
    wanted_k = set(args.candidate_k_values)
    pos_maps = {
        fold_id: {int(image_index): pos for pos, image_index in enumerate(item["eval_indices"])}
        for fold_id, item in fold_predictions.items()
    }
    for index, (input_img, label_img, name) in enumerate(loader):
        fold_id = fold_by_index.get(index)
        if fold_id is None:
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
        predictions = fold_predictions[fold_id]["predictions"]
        projections = fold_projections[fold_id]
        pred_pos = pos_maps[fold_id][index]
        for (key, mapper_name), item in predictions.items():
            low_size, k_dim = key
            if mapper_name not in wanted_mappers or k_dim not in wanted_k:
                continue
            projection = projections[key]
            pred_coeff = item["pred_coeffs"][pred_pos : pred_pos + 1]
            if item.get("zero_field"):
                pred_low = torch.zeros(1, 3, low_size, low_size)
            else:
                pred_low = (projection["mean"] + pred_coeff[0] @ projection["bases"]).view(1, 3, low_size, low_size)
            pred = F.interpolate(pred_low.to(device), size=anchor.shape[-2:], mode="bilinear", align_coners=False)
            pred = gaussian_lowpass(pred, args.kenel_size, args.sigma)
            corr = correlation(pred.cpu(), low_target.cpu(), expanded.cpu())
            initial_num = (expanded * low_target.abs()).sum().item()
            weighted_den = expanded.sum().item()
            pred_abs = pred.abs()
            pred_flat = pred.flatten()
            pred_coeff_norm = pred_coeff.norm(dim=1).item()
            pred_coeff_l1 = pred_coeff.abs().mean().item()
            extra = item.get("extras", {}).get(index, {})
            for scale in args.scales:
                scaled_pred = float(scale) * pred
                output = (anchor + weight * scaled_pred).clamp(0, 1)
                output_psnr = psnr(output, label_img)
                final_num = (expanded * (scaled_pred - low_target).abs()).sum().item()
                rows.append(
                    {
                        "fold": fold_id,
                        "low_size": low_size,
                        "K": k_dim,
                        "mapper": mapper_name,
                        "family": item.get("family"),
                        "candidate_scale": float(scale),
                        "split": "oof",
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
                        "residual_positive_frac": (pred_flat > 0).float().mean().item(),
                        "residual_negative_frac": (pred_flat < 0).float().mean().item(),
                        "weighted_den": weighted_den,
                        "initial_l1_num": initial_num,
                        "final_l1_num": final_num,
                        "weighted_field_l1": final_num / max(weighted_den, 1e-12),
                        **extra,
                    }
                )
        if args.progress_freq and (index + 1) % args.progress_freq == 0:
            print(f"oof_evaluated={index + 1}", flush=True)
    return rows


def e1_gate_pass(summary, count):
    coverage = summary["keep_count"] / max(count, 1)
    strong_rate = summary["strong_regressions"] / max(count, 1)
    return (
        summary["severe_regressions"] == 0
        and strong_rate <= 0.01
        and summary["easy_top25_gain"] >= -0.02
        and summary["hard_bottom25_gain"] >= 0.25
        and summary["mean_gain"] > 0
        and coverage >= 0.10
        and summary["oracle_recovery"] >= 0.15
    )


def locked_rule_rows(rules, action_rows):
    out = []
    group_rules = []
    for rule in rules:
        members = [
            row
            for row in action_rows
            if row["mapper"] == rule["mapper"]
            and int(row["K"]) == int(rule["K"])
            and abs(float(row["candidate_scale"]) - float(rule.get("scale", 1.0))) < 1e-12
        ]
        if not members:
            out.append({**rule, "status": "missing_candidate", "gate_pass": False})
            continue
        key = rule["confidence_key"]
        if rule["direction"] == "low":
            keep_mask = [safe_float(row.get(key), float("inf")) <= float(rule["threshold"]) for row in members]
        else:
            keep_mask = [safe_float(row.get(key), -float("inf")) >= float(rule["threshold"]) for row in members]
        summary = summarize_policy(members, keep_mask)
        summary["coverage"] = summary["keep_count"] / max(summary["count"], 1)
        summary["strong_rate"] = summary["strong_regressions"] / max(summary["count"], 1)
        passed = e1_gate_pass(summary, summary["count"])
        out.append(
            {
                **rule,
                "split": "oof",
                "gate_pass": passed,
                "gate": "severe=0,strong_rate<=1%,easy>=-0.02,hard>=0.25,mean>0,coverage>=10%,oracle_recovery>=0.15",
                **summary,
            }
        )
        group_rules.append({**rule, "kept_rows": [row for row, keep in zip(members, keep_mask) if keep]})
    return out, group_rules


def fold_summary_rows(action_rows, rules):
    out = []
    for fold_id in sorted({row["fold"] for row in action_rows}):
        fold_rows = [row for row in action_rows if row["fold"] == fold_id]
        locked_rows, _group_rules = locked_rule_rows(rules, fold_rows)
        for row in locked_rows:
            out.append({"fold": fold_id, **row})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--correctability_json", required=True)
    parser.add_argument("--correctability_train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4e_oof_calibration_sigma3_seed3407")
    parser.add_argument("--basis_num_images", type=int, default=0)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca_device", default="cpu")
    parser.add_argument("--residual_max", type=float, default=0.04)
    parser.add_argument("--kenel_size", type=int, default=31)
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
    data_count = len(meta_rows)
    global_names, global_features = feature_matrix(feature_rows)
    folds, fold_by_index = stratified_folds(meta_rows, args.fold_count, args.seed)
    write_csv(output_dir / "v04e_oof_fold_assignments.csv", fold_assignment_rows(meta_rows, folds))

    spatial_loader = build_loader(args.data_dir, data_count, args.num_workers, shuffle=False)
    spatial = collect_spatial_feature_matrices(apdr_model, spatial_loader, device, args, tau, scores)
    feature_sets = {
        "global": global_features,
        "spatial_priors": spatial["spatial_priors"],
        "convir_spatial": spatial["conv_spatial"],
    }
    feature_sets["global_plus_spatial"] = torch.cat(
        [feature_sets["global"], feature_sets["spatial_priors"], feature_sets["convir_spatial"]],
        dim=1,
    )
    feature_dims = {name: int(matrix.shape[1]) for name, matrix in feature_sets.items()}

    fold_predictions = {}
    fold_projections = {}
    active_indices = [row["index"] for row in meta_rows if row["low_weight_sum"] > 1e-8]
    active_set = set(active_indices)
    for fold_id, eval_indices in enumerate(folds):
        eval_set = set(eval_indices)
        train_indices = [index for index in active_indices if index not in eval_set]
        fold_projections[fold_id] = build_fold_projections(
            low_targets[args.low_size],
            low_weights[args.low_size],
            train_indices,
            k_values,
            args,
        )
        predictions = {}
        for name, matrix in feature_sets.items():
            predictions.update(
                make_probe_predictions(
                    matrix.float(),
                    fold_projections[fold_id],
                    train_indices,
                    eval_indices,
                    args,
                    name,
                )
            )
        fold_predictions[fold_id] = {
            "eval_indices": eval_indices,
            "train_open_count": len(train_indices),
            "eval_count": len(eval_indices),
            "eval_open_count": sum(index in active_set for index in eval_indices),
            "predictions": predictions,
        }
        print(
            f"fold_prepared={fold_id} train_open={len(train_indices)} eval={len(eval_indices)} eval_open={fold_predictions[fold_id]['eval_open_count']}",
            flush=True,
        )

    eval_loader = build_loader(args.data_dir, data_count, args.num_workers, shuffle=False)
    action_rows = evaluate_oof_action_bank(
        apdr_model,
        eval_loader,
        device,
        args,
        tau,
        scores,
        fold_by_index,
        fold_predictions,
        fold_projections,
    )
    attach_rank_groups(action_rows, "oof")
    candidate_rows = candidate_table_rows(action_rows)
    locked_rows, group_rule_rows = locked_rule_rows(locked_rules, action_rows)
    fold_locked_rows = fold_summary_rows(action_rows, locked_rules)
    auc_rows = risk_feature_auc_rows(action_rows, risk_feature_keys, "oof")
    curve_rows = calibration_curve(action_rows, args.calibration_score_key, "oof", args.calibration_bins)
    group_rows = accepted_rejected_group_rows(group_rule_rows, action_rows)
    failure_rows = failure_signature_rows(action_rows, risk_feature_keys, "oof")

    action_path = output_dir / f"v04e_oof_candidate_action_per_image_{label}.csv"
    write_csv_union(action_path, action_rows)
    write_csv(output_dir / "v04e_oof_candidate_action_table.csv", candidate_rows)
    write_csv(output_dir / "v04e_oof_locked_threshold_by_fold.csv", fold_locked_rows)
    write_csv(output_dir / "v04e_oof_risk_feature_auc.csv", auc_rows)
    write_csv(output_dir / "v04e_oof_calibration_curve.csv", curve_rows)
    write_csv(output_dir / "v04e_oof_accepted_vs_rejected_groups.csv", group_rows)
    write_csv(output_dir / "v04e_oof_strong_failure_signature.csv", failure_rows)

    bad_count = sum(bad_labels(action_rows))
    summary = {
        "stage": "APDR-v0.4E 5-fold OOF candidate-action risk calibration",
        "tag": args.tag,
        "status": "OOF calibration intermediate tables; no training",
        "sigma": args.sigma,
        "correctability_tau": tau,
        "data_count": data_count,
        "active_open_count": len(active_indices),
        "fold_count": args.fold_count,
        "folds": [
            {
                "fold": fold_id,
                "eval_count": len(folds[fold_id]),
                **{
                    key: value
                    for key, value in fold_predictions[fold_id].items()
                    if key != "predictions" and key != "eval_indices"
                },
            }
            for fold_id in range(args.fold_count)
        ],
        "low_size": args.low_size,
        "k_values": k_values,
        "candidate_mappers": args.candidate_mappers,
        "candidate_k_values": args.candidate_k_values,
        "candidate_scales": args.scales,
        "locked_rules": locked_rows,
        "all_locked_rules_pass": all(row.get("gate_pass") for row in locked_rows),
        "action_row_count": len(action_rows),
        "bad_label_count": bad_count,
        "feature_dims": feature_dims,
        "hook_paths": spatial["hook_paths"],
        "outputs": {
            "fold_assignments": str(output_dir / "v04e_oof_fold_assignments.csv"),
            "candidate_action_per_image": str(action_path),
            "candidate_action_table": str(output_dir / "v04e_oof_candidate_action_table.csv"),
            "locked_threshold_by_fold": str(output_dir / "v04e_oof_locked_threshold_by_fold.csv"),
            "risk_feature_auc": str(output_dir / "v04e_oof_risk_feature_auc.csv"),
            "oof_calibration_curve": str(output_dir / "v04e_oof_calibration_curve.csv"),
            "accepted_vs_rejected_groups": str(output_dir / "v04e_oof_accepted_vs_rejected_groups.csv"),
            "strong_failure_signature": str(output_dir / "v04e_oof_strong_failure_signature.csv"),
        },
        "args": vars(args),
        "global_feature_names": global_names,
    }
    summary_path = output_dir / "v04e_oof_locked_threshold_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
