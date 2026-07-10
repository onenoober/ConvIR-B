#!/usr/bin/env python3
"""v3i-C counterfactual response OOF probe and policy replay.

This audit asks whether extra inference-time counterfactual response maps carry
the open action-value signal that single-forward FAM2 operator state failed to
expose in v3i-B. It trains only tiny diagnostic heads; A0, W_U, D7c, and the
ConvIR backbone stay frozen. Probe weights and raw tensors are not saved.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chd_rm_v3i_b_full_context_probe import (
    D7C_THRESHOLD,
    bootstrap_delta,
    build_gate_producer,
    build_model,
    extract_sample_patches,
    fold_assignments,
    forward_final,
    forward_full,
    full_context_maps,
    gradient_at_alpha,
    load_names,
    load_pair,
    load_v3i_a_controls,
    metric_pair,
    pad_to_factor,
    percentile,
    random_top_fraction_alpha,
    ranks01,
    residual_abs_top25,
    score_map_for_probe,
    sha256_file,
    summarize_policy,
    top_fraction_alpha,
    train_probe,
    write_csv,
    write_json,
)


def site_map(tensor, shape):
    return F.interpolate(tensor, size=shape, mode="area")


def add_rgb(features, names, prefix, tensor):
    features.append(tensor)
    names.extend([f"{prefix}_r", f"{prefix}_g", f"{prefix}_b"])


def counterfactual_response_feature_map(base, action_model, padded, hard_gate, include_context=True):
    shape = hard_gate.shape[-2:]
    zeros = torch.zeros_like(hard_gate)
    half = hard_gate * 0.5
    one = hard_gate
    with torch.no_grad():
        y0 = forward_full(action_model, padded, d7c_gate=zeros)
        y05 = forward_full(action_model, padded, d7c_gate=half)
        y1 = forward_full(action_model, padded, d7c_gate=one)
        base_full = forward_full(base, padded)
        response01 = site_map(y1 - y0, shape)
        response05 = site_map(y05 - y0, shape)
        curvature = site_map(y1 - 2.0 * y05 + y0, shape)
        features = []
        names = []
        add_rgb(features, names, "resp01", response01)
        add_rgb(features, names, "resp05", response05)
        add_rgb(features, names, "curvature", curvature)
        add_rgb(features, names, "abs_resp01", response01.abs())
        add_rgb(features, names, "abs_curvature", curvature.abs())
        if include_context:
            add_rgb(features, names, "input", site_map(padded, shape))
            add_rgb(features, names, "a0_output", site_map(base_full, shape))
            add_rgb(features, names, "alpha0_output", site_map(y0, shape))
            add_rgb(features, names, "a0_minus_input", site_map(base_full - padded, shape))
            add_rgb(features, names, "alpha0_minus_input", site_map(y0 - padded, shape))
        feature = torch.cat(features, dim=1).contiguous()
        response_mag = response01.abs().mean(dim=1, keepdim=True)
        curvature_mag = curvature.abs().mean(dim=1, keepdim=True)
    return feature, response_mag, curvature_mag, names


def replay_policy(action_model, padded, label, height, width, base_psnr, hard_gate, alpha, fold, index, name, policy):
    pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alpha)
    _, psnr = metric_pair(pred, label)
    return {"fold": fold, "index": index, "name": name, "policy": policy, "psnr_delta": psnr - base_psnr}


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    names = load_names(args.split_json, args.split_key, args.max_samples)
    folds_by_image, id_to_fold = fold_assignments(names, args.fold_count)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)

    patches = []
    centers = []
    coords = []
    q_values = []
    sample_image_indices = []
    sample_folds = []
    sample_rank = []
    feature_names = None

    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        _, coord, hard_gate, _ = full_context_maps(action_model, gate_producer, padded)
        grad_open_raw = gradient_at_alpha(action_model, padded, label, hard_gate, 0.0, height, width)
        open_score = -grad_open_raw * hard_gate
        fmap, _, _, names_for_features = counterfactual_response_feature_map(
            base, action_model, padded, hard_gate, include_context=not args.no_context_channels
        )
        feature_names = names_for_features
        sampled = extract_sample_patches(fmap, coord, hard_gate, open_score, args.sample_per_image, rng)
        if sampled is not None:
            patch_np, center_np, coord_np, q_np = sampled
            patches.append(patch_np)
            centers.append(center_np)
            coords.append(coord_np)
            q_values.append(q_np)
            sample_image_indices.append(np.full(len(q_np), index, dtype=np.int32))
            sample_folds.append(np.full(len(q_np), folds_by_image[index], dtype=np.int16))
            sample_rank.append(ranks01(q_np))
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3i_c_extract_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, coord, hard_gate, grad_open_raw, open_score, fmap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arrays = {
        "patch": np.concatenate(patches, axis=0),
        "center": np.concatenate(centers, axis=0),
        "coord": np.concatenate(coords, axis=0),
        "q": np.concatenate(q_values, axis=0),
        "image_index": np.concatenate(sample_image_indices, axis=0),
        "fold": np.concatenate(sample_folds, axis=0),
        "rank": np.concatenate(sample_rank, axis=0),
    }
    feature_manifest = {
        "sample_count": int(arrays["q"].shape[0]),
        "sample_per_image": args.sample_per_image,
        "feature_channels": int(arrays["center"].shape[1]),
        "feature_names": feature_names or [],
        "fold_count": args.fold_count,
        "clean_reference_count": len(id_to_fold),
        "raw_feature_tensors_saved": False,
        "counterfactual_forwards": ["alpha0", "alpha0p5", "alpha1"],
        "includes_context_channels": not args.no_context_channels,
    }

    histories = []
    replay_rows = []
    control_rows = []
    v3i_a_controls = load_v3i_a_controls(args.v3i_a_policy_per_image)
    hard_by_name = {}
    for name in names:
        hard_delta = v3i_a_controls.get((name, "HARD_D7C_ALPHA1"))
        oracle_delta = v3i_a_controls.get((name, "OPEN_TOP_0.5"))
        if hard_delta is not None:
            hard_by_name[name] = hard_delta
            control_rows.append({"name": name, "policy": "HARD_D7C_ALPHA1", "psnr_delta": hard_delta})
        if oracle_delta is not None:
            control_rows.append({"name": name, "policy": "GT_OPEN_TOP50_ORACLE", "psnr_delta": oracle_delta})
        control_rows.append({"name": name, "policy": "A0", "psnr_delta": 0.0})

    for fold in range(args.fold_count):
        train_mask = arrays["fold"] != fold
        fold_models = {}
        for kind in ("linear", "dw3x3"):
            model, mean, std, history = train_probe(kind, arrays, train_mask, args, device, fold)
            fold_models[kind] = (model, mean, std)
            histories.extend(history)
        for index, name in enumerate(names):
            if folds_by_image[index] != fold:
                continue
            input_img, label = load_pair(args.data_dir, args.source_split, name)
            input_img = input_img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            padded, height, width = pad_to_factor(input_img)
            label = label[:, :, :height, :width]
            with torch.no_grad():
                base_full = forward_full(base, padded)
                base_pred = base_full[:, :, :height, :width]
                _, base_psnr = metric_pair(base_pred, label)
                _, coord, hard_gate, _ = full_context_maps(action_model, gate_producer, padded)
                fmap, response_mag, curvature_mag, _ = counterfactual_response_feature_map(
                    base, action_model, padded, hard_gate, include_context=not args.no_context_channels
                )
                controls = [
                    ("RANDOM_TOP50", random_top_fraction_alpha(hard_gate, 0.5, rng)),
                    ("V3H_RESIDUAL_ABS_TOP25", residual_abs_top25(base_full, padded, hard_gate)),
                    ("CF_RESPONSE_MAG_TOP50", top_fraction_alpha(response_mag, hard_gate, 0.5, largest=True)),
                    ("CF_CURVATURE_MAG_TOP50", top_fraction_alpha(curvature_mag, hard_gate, 0.5, largest=True)),
                ]
                for policy, alpha in controls:
                    replay_rows.append(
                        replay_policy(action_model, padded, label, height, width, base_psnr, hard_gate, alpha, fold, index, name, policy)
                    )
                for kind, (model, mean, std) in fold_models.items():
                    score = score_map_for_probe(kind, model, fmap, coord, mean, std)
                    alpha = top_fraction_alpha(score, hard_gate, 0.5, largest=True)
                    replay_rows.append(
                        replay_policy(
                            action_model,
                            padded,
                            label,
                            height,
                            width,
                            base_psnr,
                            hard_gate,
                            alpha,
                            fold,
                            index,
                            name,
                            f"OOF_CF_RESPONSE_{kind.upper()}_TOP50",
                        )
                    )
            del input_img, label, padded, base_full, base_pred, coord, hard_gate, fmap, response_mag, curvature_mag
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"v3i_c_fold_done {fold}", flush=True)

    all_rows = control_rows + replay_rows
    policy_summary = summarize_policy(all_rows)
    summary_by_policy = {row["policy"]: row for row in policy_summary}
    hard = summary_by_policy["HARD_D7C_ALPHA1"]
    oracle = summary_by_policy["GT_OPEN_TOP50_ORACLE"]
    gate_line = hard["mean_psnr_delta"] + args.min_oracle_gap_retention * (
        oracle["mean_psnr_delta"] - hard["mean_psnr_delta"]
    )
    rng_boot = np.random.default_rng(args.seed + 2027)
    bootstrap_rows = []
    replay_by_policy = defaultdict(list)
    for row in replay_rows:
        replay_by_policy[row["policy"]].append(row)
    for policy, rows in sorted(replay_by_policy.items()):
        boot = bootstrap_delta(rows, hard_by_name, rng_boot, args.bootstrap_draws)
        boot["policy"] = policy
        bootstrap_rows.append(boot)

    candidate_policies = ["OOF_CF_RESPONSE_LINEAR_TOP50", "OOF_CF_RESPONSE_DW3X3_TOP50"]
    pass_policies = []
    strong_pass_policies = []
    information_gain_policies = []
    for policy in candidate_policies:
        row = summary_by_policy.get(policy)
        boot = next((b for b in bootstrap_rows if b["policy"] == policy), None)
        if not row or not boot:
            continue
        min_pass = (
            boot["ci95_low"] > 0
            and row["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
            and row["p10_psnr_delta"] >= hard["p10_psnr_delta"]
        )
        info_gain = min_pass and row["mean_psnr_delta"] > args.v3i_b_best_mean_line
        strong_pass = info_gain and row["mean_psnr_delta"] > args.ungated_mean_line and row["mean_psnr_delta"] >= gate_line
        if min_pass:
            pass_policies.append(policy)
        if info_gain:
            information_gain_policies.append(policy)
        if strong_pass:
            strong_pass_policies.append(policy)

    if strong_pass_policies:
        decision = "V3I_C_COUNTERFACTUAL_RESPONSE_REPLAY_STRONG_PASS_AUTHORIZE_RESPONSE_CONDITIONED_ROUTE"
        next_action = "Plan response-conditioned controller/self-evaluation route; no locked test and no single-forward canary."
    elif information_gain_policies:
        decision = "V3I_C_COUNTERFACTUAL_RESPONSE_INFORMATION_GAIN_REQUIRE_CONFIRM_OR_DISAGREEMENT_AUDIT"
        next_action = "Do not canary; confirm response-conditioned signal or audit disagreement features."
    elif pass_policies:
        decision = "V3I_C_COUNTERFACTUAL_RESPONSE_MIN_PASS_WEAK_SIGNAL_REQUIRE_DISAGREEMENT_AUDIT"
        next_action = "Treat as weak signal only; run disagreement/consistency audit before any route design."
    else:
        decision = "V3I_C_COUNTERFACTUAL_RESPONSE_FAIL_AUTHORIZE_DISAGREEMENT_AUDIT_ONLY"
        next_action = "Counterfactual RGB response is insufficient; audit checkpoint/transform disagreement only, then stop FAM2 router if it also fails."

    summary = {
        "phase": "v3i-C counterfactual response OOF probe replay",
        "decision": decision,
        "next_action": next_action,
        "diagnostic_training_only": True,
        "controller_canary_authorized": False,
        "single_forward_controller_authorized": False,
        "locked_test_touched": False,
        "sample_count": len(names),
        "feature_manifest": feature_manifest,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "metric_contract": {
            "split": "5-fold clean-reference OOF over internal val_inner 600",
            "target": "sampled within-image open_score rank from alpha=0",
            "replay": "D7c hard veto times predicted top-50 active action sites",
            "counterfactual_inputs": "signed RGB response maps from alpha 0, 0.5, and 1 plus optional input/output context",
            "minimum_pass": "paired mean gain over hard D7c CI95 low > 0, severe regressions <= hard, p10 >= hard",
            "information_gain": "minimum pass plus mean above v3i-B best candidate line",
            "strong_pass": "information gain plus mean > ungated line and >= 25% oracle gap retention",
            "v3i_b_best_mean_line": args.v3i_b_best_mean_line,
            "ungated_mean_line": args.ungated_mean_line,
            "oracle_gap_gate_line": gate_line,
        },
        "hard_policy": hard,
        "oracle_policy": oracle,
        "policy_summary": policy_summary,
        "bootstrap_vs_hard": bootstrap_rows,
        "minimum_pass_policies": pass_policies,
        "information_gain_policies": information_gain_policies,
        "strong_pass_policies": strong_pass_policies,
    }
    write_csv(output_dir / "v3i_c_probe_training_history.csv", histories)
    write_csv(output_dir / "v3i_c_policy_replay_per_image.csv", all_rows)
    write_csv(output_dir / "v3i_c_policy_replay_summary.csv", policy_summary)
    write_csv(output_dir / "v3i_c_bootstrap_vs_hard.csv", bootstrap_rows)
    write_json(output_dir / "v3i_c_counterfactual_response_summary.json", summary)
    write_json(output_dir / "v3i_c_counterfactual_feature_manifest.json", feature_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_key", default="val_inner")
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--v3i_a_policy_per_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--sample_per_image", type=int, default=192)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--linear_steps", type=int, default=320)
    parser.add_argument("--dw_steps", type=int, default=420)
    parser.add_argument("--coord_steps", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--value_loss_weight", type=float, default=0.25)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--min_oracle_gap_retention", type=float, default=0.25)
    parser.add_argument("--ungated_mean_line", type=float, default=0.03306524052036514)
    parser.add_argument("--v3i_b_best_mean_line", type=float, default=0.01662677374336075)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--no_context_channels", action="store_true")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
