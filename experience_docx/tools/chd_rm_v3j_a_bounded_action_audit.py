#!/usr/bin/env python3
"""v3j-A no-training bounded safe-correction action-space audit.

This audit changes the actuator question from selecting whether to open the old
FAM2 correction to asking whether a bounded output residual branch could still
realize a privileged safe teacher. It performs no model training and writes only
compact text evidence.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chd_rm_v3i_a_teacher_compressibility_audit import (  # noqa: E402
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    forward_final,
    gradient_at_alpha,
    load_pair,
    make_policy_alphas,
    metric_pair,
    pad_to_factor,
    percentile,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)

ROUTE_ID = "haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711"
FIXED_UNGATED_MEAN_LINE = 0.03306524052036514
FIXED_25PCT_ORACLE_GAP_LINE = 0.11251170518611245
FIXED_V3I_HARD_P10_LINE = -0.12151199691896011


def clean_id(name):
    return Path(name).stem.split("_")[0]


def create_fresh_split(args):
    out_path = Path(args.fresh_split_manifest)
    if out_path.is_file() and not args.regenerate_split_manifest:
        return read_json(out_path)
    base = read_json(args.split_json)
    source_names = sorted(base["splits"][args.fresh_split_source_key])
    groups = defaultdict(list)
    for name in source_names:
        groups[clean_id(name)].append(name)
    ids = sorted(groups)
    rng = random.Random(args.split_seed)
    rng.shuffle(ids)
    targets = [
        ("v3j_controller_train", args.controller_train_count),
        ("v3j_controller_calib", args.controller_calib_count),
        ("v3j_route_confirm", args.route_confirm_count),
    ]
    splits = {key: [] for key, _ in targets}
    key_idx = 0
    for cid in ids:
        if key_idx >= len(targets):
            break
        key, target = targets[key_idx]
        if len(splits[key]) >= target:
            key_idx += 1
            if key_idx >= len(targets):
                break
            key, target = targets[key_idx]
        splits[key].extend(sorted(groups[cid]))
    for key, target in targets:
        splits[key] = sorted(splits[key])
        if len(splits[key]) < target:
            raise RuntimeError(f"Split {key} has {len(splits[key])}, expected at least {target}")
    manifest = {
        "route_id": ROUTE_ID,
        "source_split_json": args.split_json,
        "source_key": args.fresh_split_source_key,
        "split_seed": args.split_seed,
        "clean_reference_grouping": "Path(name).stem.split('_')[0]",
        "locked_test_touched": False,
        "splits": splits,
        "counts": {key: len(value) for key, value in splits.items()},
        "policy": "train_inner only; route_confirm is train-derived and locked before teacher/bound/gate metrics",
    }
    write_json(out_path, manifest)
    return manifest


def names_from_manifest(manifest, key, max_samples):
    names = sorted(manifest["splits"][key])
    if max_samples > 0:
        names = names[:max_samples]
    if not names:
        raise ValueError(f"No names for split key {key}")
    return names


def summarize_policy(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(float(row["psnr_delta"]))
    out = []
    for policy in sorted(grouped):
        vals = np.asarray(grouped[policy], dtype=np.float64)
        out.append(
            {
                "policy": policy,
                "n": int(vals.size),
                "mean_psnr_delta": float(np.mean(vals)),
                "median_psnr_delta": percentile(vals, 50),
                "p10_psnr_delta": percentile(vals, 10),
                "p05_psnr_delta": percentile(vals, 5),
                "worst_psnr_delta": float(np.min(vals)),
                "best_psnr_delta": float(np.max(vals)),
                "positive_ratio": float(np.mean(vals > 0)),
                "regression_le_0p2_count": int(np.sum(vals <= -0.2)),
                "regression_le_0p5_count": int(np.sum(vals <= -0.5)),
            }
        )
    return out


def bootstrap_delta(candidate_rows, hard_by_name, seed, draws):
    diffs = np.asarray(
        [float(row["psnr_delta"]) - hard_by_name[row["name"]] for row in candidate_rows if row["name"] in hard_by_name],
        dtype=np.float64,
    )
    if diffs.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        boot.append(float(np.mean(diffs[idx])))
    return {
        "n": int(diffs.size),
        "mean": float(np.mean(diffs)),
        "ci95_low": percentile(boot, 2.5),
        "ci95_high": percentile(boot, 97.5),
    }


def output_gate_from_action_gate(hard_gate, output_shape):
    return F.interpolate((hard_gate > 0.5).to(hard_gate.dtype), size=output_shape, mode="nearest")


def smooth3(x):
    if x.shape[-2] < 2 or x.shape[-1] < 2:
        return x
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)


def project_residual(residual, projection):
    height, width = residual.shape[-2:]
    if projection == "full_clip":
        return residual
    if projection == "half_bilinear":
        low = F.interpolate(residual, size=(max(1, height // 2), max(1, width // 2)), mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    if projection == "quarter_bilinear":
        low = F.interpolate(residual, size=(max(1, height // 4), max(1, width // 4)), mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    if projection == "half_smooth3":
        low = F.interpolate(residual, size=(max(1, height // 2), max(1, width // 2)), mode="bilinear", align_corners=False)
        low = smooth3(low)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    raise ValueError(f"Unknown projection {projection}")


def apply_bounded_projection(base_pred, teacher_pred, hard_gate, bound, projection):
    residual = teacher_pred - base_pred
    projected = project_residual(residual, projection)
    bound_t = torch.as_tensor(bound, device=projected.device, dtype=projected.dtype).view(1, 3, 1, 1)
    clipped = torch.maximum(torch.minimum(projected, bound_t), -bound_t)
    output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
    return torch.clamp(base_pred + output_gate * clipped, 0, 1)


def teacher_policy_outputs(action_model, padded, label, hard_gate, height, width, args):
    grad_open_raw, _, _ = gradient_at_alpha(action_model, padded, label, hard_gate, 0.0, height, width)
    grad_close_raw, _, _ = gradient_at_alpha(action_model, padded, label, hard_gate, 1.0, height, width)
    open_score = -grad_open_raw * hard_gate
    q1 = -grad_close_raw * hard_gate
    alphas, teacher_stats = make_policy_alphas(open_score, q1, hard_gate, args)
    return alphas, teacher_stats


def calibrate_bounds(args, manifest, base, action_model, gate_producer, device):
    names = names_from_manifest(manifest, args.bound_calib_key, args.max_bound_calib_samples)
    channel_samples = [[], [], []]
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
        alphas, _ = teacher_policy_outputs(action_model, padded, label, hard_gate, height, width, args)
        if args.bound_teacher_policy not in alphas:
            raise KeyError(f"Bound teacher policy missing: {args.bound_teacher_policy}")
        with torch.no_grad():
            teacher_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alphas[args.bound_teacher_policy])
            output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
            residual_abs = ((teacher_pred - base_pred) * output_gate).abs()
            for ch in range(3):
                flat = residual_abs[0, ch].reshape(-1)[:: args.bound_sample_stride].detach().cpu().numpy().astype(np.float32)
                channel_samples[ch].append(flat)
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3j_a_bound_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_gate, alphas
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bounds = []
    sample_counts = []
    for ch in range(3):
        arr = np.concatenate(channel_samples[ch]) if channel_samples[ch] else np.asarray([0.0], dtype=np.float32)
        sample_counts.append(int(arr.size))
        bounds.append(max(float(np.percentile(arr, args.bound_percentile)), args.bound_min))
    return {
        "bound_teacher_policy": args.bound_teacher_policy,
        "bound_percentile": args.bound_percentile,
        "bound_sample_stride": args.bound_sample_stride,
        "bound_calib_key": args.bound_calib_key,
        "bound_calib_count": len(names),
        "channel_abs_sample_counts": sample_counts,
        "channel_bounds_rgb": bounds,
    }


def evaluate(args, manifest, bounds, base, action_model, gate_producer, device):
    names = names_from_manifest(manifest, args.eval_key, args.max_samples)
    policy_rows = []
    target_rows = []
    teacher_keys = [args.primary_teacher_policy, args.ceiling_teacher_policy]
    bound = bounds["channel_bounds_rgb"]
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            base_mse, base_psnr = metric_pair(base_pred, label)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
        alphas, teacher_stats = teacher_policy_outputs(action_model, padded, label, hard_gate, height, width, args)
        alphas["HARD_D7C_ALPHA1"] = (hard_gate > 0.5).to(hard_gate.dtype)
        with torch.no_grad():
            hard_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alphas["HARD_D7C_ALPHA1"])
            hard_mse, hard_psnr = metric_pair(hard_pred, label)
            policy_rows.append({"index": index, "name": name, "policy": "A0", "mse": base_mse, "psnr": base_psnr, "psnr_delta": 0.0})
            policy_rows.append({"index": index, "name": name, "policy": "HARD_D7C_ALPHA1", "mse": hard_mse, "psnr": hard_psnr, "psnr_delta": hard_psnr - base_psnr})
            for teacher_key in teacher_keys:
                if teacher_key not in alphas:
                    raise KeyError(f"Teacher policy missing: {teacher_key}")
                teacher_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate * alphas[teacher_key])
                raw_mse, raw_psnr = metric_pair(teacher_pred, label)
                prefix = "PRIMARY" if teacher_key == args.primary_teacher_policy else "CEILING"
                raw_policy = f"{prefix}_{teacher_key}_RAW"
                policy_rows.append({"index": index, "name": name, "policy": raw_policy, "mse": raw_mse, "psnr": raw_psnr, "psnr_delta": raw_psnr - base_psnr})
                for projection in args.projections:
                    pred = apply_bounded_projection(base_pred, teacher_pred, hard_gate, bound, projection)
                    mse, psnr = metric_pair(pred, label)
                    policy_rows.append(
                        {
                            "index": index,
                            "name": name,
                            "policy": f"{prefix}_{projection.upper()}_P{args.bound_percentile:g}_D7C",
                            "teacher_policy": teacher_key,
                            "projection": projection,
                            "mse": mse,
                            "psnr": psnr,
                            "psnr_delta": psnr - base_psnr,
                        }
                    )
        active = int((hard_gate > 0.5).sum().item())
        target_rows.append(
            {
                "index": index,
                "name": name,
                "action_active_count": active,
                "action_coverage": float((hard_gate > 0.5).float().mean().item()),
                "secant_ignored_count": teacher_stats.get("secant_ignored_count", 0),
                "secant_near_zero_count": teacher_stats.get("secant_near_zero_count", 0),
            }
        )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3j_a_eval_progress {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_pred, hard_gate, alphas
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return policy_rows, target_rows


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = create_fresh_split(args)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    bounds = calibrate_bounds(args, manifest, base, action_model, gate_producer, device)
    policy_rows, target_rows = evaluate(args, manifest, bounds, base, action_model, gate_producer, device)
    policy_summary = summarize_policy(policy_rows)
    summary_by_policy = {row["policy"]: row for row in policy_summary}
    hard = summary_by_policy["HARD_D7C_ALPHA1"]
    hard_by_name = {row["name"]: float(row["psnr_delta"]) for row in policy_rows if row["policy"] == "HARD_D7C_ALPHA1"}
    by_policy_rows = defaultdict(list)
    for row in policy_rows:
        by_policy_rows[row["policy"]].append(row)
    bootstrap_rows = []
    minimum_pass = []
    strong_pass = []
    for policy, rows in sorted(by_policy_rows.items()):
        if policy in {"A0", "HARD_D7C_ALPHA1"} or policy.endswith("_RAW"):
            continue
        boot = bootstrap_delta(rows, hard_by_name, args.seed + 20260711, args.bootstrap_draws)
        boot["policy"] = policy
        bootstrap_rows.append(boot)
        row = summary_by_policy[policy]
        min_pass = (
            boot["ci95_low"] > 0
            and row["regression_le_0p2_count"] <= hard["regression_le_0p2_count"]
            and row["p10_psnr_delta"] >= hard["p10_psnr_delta"]
            and row["p10_psnr_delta"] >= FIXED_V3I_HARD_P10_LINE
        )
        strong = (
            min_pass
            and row["mean_psnr_delta"] > FIXED_UNGATED_MEAN_LINE
            and row["regression_le_0p2_count"] < hard["regression_le_0p2_count"]
            and row["mean_psnr_delta"] >= FIXED_25PCT_ORACLE_GAP_LINE
        )
        if min_pass:
            minimum_pass.append(policy)
        if strong:
            strong_pass.append(policy)
    primary_minimum = [p for p in minimum_pass if p.startswith("PRIMARY_")]
    primary_strong = [p for p in strong_pass if p.startswith("PRIMARY_")]
    decision = (
        "V3J_BOUNDED_ACTION_SPACE_PASS_AUTHORIZE_DIRECT_CORRECTION_OOF_ONLY"
        if primary_minimum
        else "V3J_BOUNDED_ACTION_SPACE_FAIL_STOP_CORRECTION_REDESIGN"
    )
    next_action = (
        "Run v3j-B direct bounded residual OOF diagnostic only; no backbone/adapter training and no locked test."
        if primary_minimum
        else "Do not train a correction head; redesign actuator or require new information before any training."
    )
    summary = {
        "route_id": ROUTE_ID,
        "phase": "v3j-A bounded safe-output correction action-space audit",
        "decision": decision,
        "next_action": next_action,
        "locked_test_touched": False,
        "training_authorized": False,
        "direct_correction_oof_authorized": bool(primary_minimum),
        "sample_count": len(names_from_manifest(manifest, args.eval_key, args.max_samples)),
        "eval_key": args.eval_key,
        "a0_checkpoint": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "metric_contract": {
            "baseline": "A0 PSNR on fresh train-derived v3j_route_confirm split",
            "teacher_primary": args.primary_teacher_policy,
            "teacher_ceiling": args.ceiling_teacher_policy,
            "bound_source": "per-channel p99 absolute primary-teacher output residual on v3j_controller_calib, sampled by fixed stride",
            "projection_variants": args.projections,
            "minimum_pass": "paired mean delta vs same-split hard D7c CI95 low > 0, p10 >= same-split hard and fixed v3i hard p10 line, severe regressions <= hard",
            "strong_pass": "minimum pass plus mean > fixed ungated line and >= fixed 25% oracle-gap line with fewer severe regressions than hard",
            "fixed_ungated_mean_line": FIXED_UNGATED_MEAN_LINE,
            "fixed_25pct_oracle_gap_line": FIXED_25PCT_ORACLE_GAP_LINE,
            "fixed_v3i_hard_p10_line": FIXED_V3I_HARD_P10_LINE,
        },
        "fresh_split_manifest": args.fresh_split_manifest,
        "split_counts": manifest["counts"],
        "bounds": bounds,
        "hard_policy": hard,
        "policy_summary": policy_summary,
        "bootstrap_vs_hard": bootstrap_rows,
        "minimum_pass_policies": minimum_pass,
        "strong_pass_policies": strong_pass,
        "primary_minimum_pass_policies": primary_minimum,
        "primary_strong_pass_policies": primary_strong,
    }
    write_csv(output_dir / "bounded_action_space_policy_replay.csv", policy_rows)
    write_csv(output_dir / "bounded_action_space_replay_summary.csv", policy_summary)
    write_csv(output_dir / "bounded_action_space_bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "bounded_action_space_target_stats.csv", target_rows)
    write_json(output_dir / "bounded_action_space_bounds.json", bounds)
    write_json(output_dir / "v3j_a_bounded_action_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--fresh_split_source_key", default="train_inner")
    parser.add_argument("--bound_calib_key", default="v3j_controller_calib")
    parser.add_argument("--eval_key", default="v3j_route_confirm")
    parser.add_argument("--controller_train_count", type=int, default=1200)
    parser.add_argument("--controller_calib_count", type=int, default=600)
    parser.add_argument("--route_confirm_count", type=int, default=600)
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--max_bound_calib_samples", type=int, default=600)
    parser.add_argument("--split_seed", type=int, default=3407)
    parser.add_argument("--regenerate_split_manifest", action="store_true")
    parser.add_argument("--primary_teacher_policy", default="CC_MIN4_FROM_OPEN_TOP_0.5")
    parser.add_argument("--ceiling_teacher_policy", default="ALPHA_SECANT_Q3")
    parser.add_argument("--bound_teacher_policy", default="CC_MIN4_FROM_OPEN_TOP_0.5")
    parser.add_argument("--bound_percentile", type=float, default=99.0)
    parser.add_argument("--bound_sample_stride", type=int, default=8)
    parser.add_argument("--bound_min", type=float, default=1e-6)
    parser.add_argument("--projections", nargs="+", default=["full_clip", "half_bilinear", "quarter_bilinear", "half_smooth3"])
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--block_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--smooth_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--min_component_sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--interior_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--uniform_alphas", type=float, nargs="*", default=[])
    parser.add_argument("--score_eps", type=float, default=0.0)
    parser.add_argument("--denom_eps", type=float, default=1e-12)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
