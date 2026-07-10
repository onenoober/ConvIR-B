#!/usr/bin/env python3
"""v3i-D checkpoint/transform disagreement OOF probe and policy replay.

This is the final v3i information audit after single-forward full context and
counterfactual RGB response both failed. It uses only inference-time
disagreement signals from an e1/e5 FAM2 checkpoint pair and horizontal-flip
consistency. It does not train or authorize a deployable controller.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from chd_rm_v3i_b_full_context_probe import (
    D7C_THRESHOLD,
    bootstrap_delta,
    build_gate_producer,
    build_model,
    extract_sample_patches,
    fold_assignments,
    forward_full,
    full_context_maps,
    gradient_at_alpha,
    load_names,
    load_pair,
    load_v3i_a_controls,
    metric_pair,
    pad_to_factor,
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
from chd_rm_v3i_c_counterfactual_response_probe import add_rgb, replay_policy, site_map


def add_scalar(features, names, prefix, tensor):
    features.append(tensor)
    names.append(prefix)


def flip_w(tensor):
    return torch.flip(tensor, dims=[-1])


def response_pair(model, padded, hard_gate):
    zeros = torch.zeros_like(hard_gate)
    with torch.no_grad():
        y0 = forward_full(model, padded, d7c_gate=zeros)
        y1 = forward_full(model, padded, d7c_gate=hard_gate)
    return y0, y1, y1 - y0


def disagreement_feature_map(primary_model, peer_model, gate_producer, padded, hard_gate, score):
    shape = hard_gate.shape[-2:]
    with torch.no_grad():
        primary_y0, primary_y1, primary_resp = response_pair(primary_model, padded, hard_gate)
        peer_y0, peer_y1, peer_resp = response_pair(peer_model, padded, hard_gate)

        flipped_padded = flip_w(padded)
        _, _, flip_gate_raw, flip_score_raw = full_context_maps(primary_model, gate_producer, flipped_padded)
        flip_y0_raw, flip_y1_raw, flip_resp_raw = response_pair(primary_model, flipped_padded, flip_gate_raw)
        flip_gate = flip_w(flip_gate_raw)
        flip_score = flip_w(flip_score_raw)
        flip_y0 = flip_w(flip_y0_raw)
        flip_y1 = flip_w(flip_y1_raw)
        flip_resp = flip_w(flip_resp_raw)

        resp_site = site_map(primary_resp, shape)
        ckpt_resp_diff = site_map(primary_resp - peer_resp, shape)
        ckpt_y0_diff = site_map(primary_y0 - peer_y0, shape)
        ckpt_y1_diff = site_map(primary_y1 - peer_y1, shape)
        flip_resp_diff = site_map(primary_resp - flip_resp, shape)
        flip_y0_diff = site_map(primary_y0 - flip_y0, shape)
        flip_y1_diff = site_map(primary_y1 - flip_y1, shape)

        features = []
        names = []
        add_rgb(features, names, "primary_resp", resp_site)
        add_rgb(features, names, "abs_primary_resp", resp_site.abs())
        add_rgb(features, names, "ckpt_resp_diff", ckpt_resp_diff)
        add_rgb(features, names, "abs_ckpt_resp_diff", ckpt_resp_diff.abs())
        add_rgb(features, names, "ckpt_y0_diff", ckpt_y0_diff)
        add_rgb(features, names, "abs_ckpt_y0_diff", ckpt_y0_diff.abs())
        add_rgb(features, names, "ckpt_y1_diff", ckpt_y1_diff)
        add_rgb(features, names, "abs_ckpt_y1_diff", ckpt_y1_diff.abs())
        add_rgb(features, names, "flip_resp_diff", flip_resp_diff)
        add_rgb(features, names, "abs_flip_resp_diff", flip_resp_diff.abs())
        add_rgb(features, names, "flip_y0_diff", flip_y0_diff)
        add_rgb(features, names, "abs_flip_y0_diff", flip_y0_diff.abs())
        add_rgb(features, names, "flip_y1_diff", flip_y1_diff)
        add_rgb(features, names, "abs_flip_y1_diff", flip_y1_diff.abs())
        add_scalar(features, names, "flip_gate_diff", hard_gate - flip_gate)
        add_scalar(features, names, "abs_flip_gate_diff", (hard_gate - flip_gate).abs())
        add_scalar(features, names, "flip_score_diff", score - flip_score)
        add_scalar(features, names, "abs_flip_score_diff", (score - flip_score).abs())
        feature = torch.cat(features, dim=1).contiguous()
        ckpt_resp_mag = ckpt_resp_diff.abs().mean(dim=1, keepdim=True)
        flip_resp_mag = flip_resp_diff.abs().mean(dim=1, keepdim=True)
        flip_gate_mag = (hard_gate - flip_gate).abs()
    return feature, ckpt_resp_mag, flip_resp_mag, flip_gate_mag, names


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
    primary_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    peer_model = build_model("fam2_d7c_noop", args.peer_control_checkpoint, device)
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
        _, coord, hard_gate, score = full_context_maps(primary_model, gate_producer, padded)
        grad_open_raw = gradient_at_alpha(primary_model, padded, label, hard_gate, 0.0, height, width)
        open_score = -grad_open_raw * hard_gate
        fmap, _, _, _, names_for_features = disagreement_feature_map(
            primary_model, peer_model, gate_producer, padded, hard_gate, score
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
            print(f"v3i_d_extract_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, coord, hard_gate, score, grad_open_raw, open_score, fmap
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
        "primary_checkpoint": args.control_checkpoint,
        "peer_checkpoint": args.peer_control_checkpoint,
        "disagreement_sources": ["e5_vs_e1_checkpoint", "horizontal_flip_consistency"],
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
                _, coord, hard_gate, score = full_context_maps(primary_model, gate_producer, padded)
                fmap, ckpt_resp_mag, flip_resp_mag, flip_gate_mag, _ = disagreement_feature_map(
                    primary_model, peer_model, gate_producer, padded, hard_gate, score
                )
                controls = [
                    ("RANDOM_TOP50", random_top_fraction_alpha(hard_gate, 0.5, rng)),
                    ("V3H_RESIDUAL_ABS_TOP25", residual_abs_top25(base_full, padded, hard_gate)),
                    ("DISAGREE_CKPT_RESP_MAG_TOP50", top_fraction_alpha(ckpt_resp_mag, hard_gate, 0.5, largest=True)),
                    ("DISAGREE_FLIP_RESP_MAG_TOP50", top_fraction_alpha(flip_resp_mag, hard_gate, 0.5, largest=True)),
                    ("DISAGREE_FLIP_GATE_MAG_TOP50", top_fraction_alpha(flip_gate_mag, hard_gate, 0.5, largest=True)),
                ]
                for policy, alpha in controls:
                    replay_rows.append(
                        replay_policy(primary_model, padded, label, height, width, base_psnr, hard_gate, alpha, fold, index, name, policy)
                    )
                for kind, (model, mean, std) in fold_models.items():
                    score_map = score_map_for_probe(kind, model, fmap, coord, mean, std)
                    alpha = top_fraction_alpha(score_map, hard_gate, 0.5, largest=True)
                    replay_rows.append(
                        replay_policy(
                            primary_model,
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
                            f"OOF_DISAGREE_{kind.upper()}_TOP50",
                        )
                    )
            del input_img, label, padded, base_full, base_pred, coord, hard_gate, score, fmap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"v3i_d_fold_done {fold}", flush=True)

    all_rows = control_rows + replay_rows
    policy_summary = summarize_policy(all_rows)
    summary_by_policy = {row["policy"]: row for row in policy_summary}
    hard = summary_by_policy["HARD_D7C_ALPHA1"]
    oracle = summary_by_policy["GT_OPEN_TOP50_ORACLE"]
    gate_line = hard["mean_psnr_delta"] + args.min_oracle_gap_retention * (
        oracle["mean_psnr_delta"] - hard["mean_psnr_delta"]
    )
    rng_boot = np.random.default_rng(args.seed + 2028)
    bootstrap_rows = []
    replay_by_policy = defaultdict(list)
    for row in replay_rows:
        replay_by_policy[row["policy"]].append(row)
    for policy, rows in sorted(replay_by_policy.items()):
        boot = bootstrap_delta(rows, hard_by_name, rng_boot, args.bootstrap_draws)
        boot["policy"] = policy
        bootstrap_rows.append(boot)

    candidate_policies = ["OOF_DISAGREE_LINEAR_TOP50", "OOF_DISAGREE_DW3X3_TOP50"]
    pass_policies = []
    information_gain_policies = []
    strong_pass_policies = []
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
        decision = "V3I_D_DISAGREEMENT_REPLAY_STRONG_PASS_AUTHORIZE_DISAGREEMENT_CONDITIONED_ROUTE"
        next_action = "Plan disagreement-conditioned route only; no locked test and no single-forward canary."
    elif information_gain_policies:
        decision = "V3I_D_DISAGREEMENT_INFORMATION_GAIN_REQUIRE_CONFIRM"
        next_action = "Do not canary; run one confirm before any disagreement-conditioned design."
    elif pass_policies:
        decision = "V3I_D_DISAGREEMENT_MIN_PASS_WEAK_SIGNAL_STOP_CONTROLLER_CANARY"
        next_action = "Weak signal only; stop controller canary and consider redesign."
    else:
        decision = "V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE"
        next_action = "Stop FAM2 router/distillation; next route should redesign correction as joint correction-confidence or bounded experts."

    summary = {
        "phase": "v3i-D checkpoint and transform disagreement OOF probe replay",
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
        "peer_control_checkpoint": args.peer_control_checkpoint,
        "peer_control_checkpoint_sha256": sha256_file(args.peer_control_checkpoint),
        "metric_contract": {
            "split": "5-fold clean-reference OOF over internal val_inner 600",
            "target": "sampled within-image open_score rank from alpha=0 on primary e5 checkpoint",
            "replay": "D7c hard veto times predicted top-50 active action sites on primary e5 checkpoint",
            "disagreement_inputs": "e5-vs-e1 response/output disagreement plus horizontal-flip gate/score/response consistency",
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
    write_csv(output_dir / "v3i_d_probe_training_history.csv", histories)
    write_csv(output_dir / "v3i_d_policy_replay_per_image.csv", all_rows)
    write_csv(output_dir / "v3i_d_policy_replay_summary.csv", policy_summary)
    write_csv(output_dir / "v3i_d_bootstrap_vs_hard.csv", bootstrap_rows)
    write_json(output_dir / "v3i_d_disagreement_summary.json", summary)
    write_json(output_dir / "v3i_d_disagreement_feature_manifest.json", feature_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--peer_control_checkpoint", required=True)
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
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
