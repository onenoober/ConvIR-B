#!/usr/bin/env python3
"""Generate and audit the v3q A0b inference-time candidate-pair feature table."""

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v3q_active_signed_value_20260712"
OPERATORS = ("D_ref", "D_rep")
ACTION_PAIR = (0.125, 0.25)
FEATURE_COLUMNS = (
    "direct_step_energy",
    "d7c_score_mean",
    "delta_l1_mean",
    "delta_r_mean", "delta_g_mean", "delta_b_mean",
    "delta_r_std", "delta_g_std", "delta_b_std",
    "midpoint_r_mean", "midpoint_g_mean", "midpoint_b_mean",
    "hazy_minus_midpoint_dot_delta",
    "gradient_alignment",
    "cov_hazy_minus_midpoint_delta_r",
    "cov_hazy_minus_midpoint_delta_g",
    "cov_hazy_minus_midpoint_delta_b",
    "hazy_luminance_mean",
    "hazy_luminance_std",
    "hazy_saturation_mean",
    "clip_fraction_0p125",
    "clip_fraction_0p25",
    "signed_distance_to_clip_0p125",
    "signed_distance_to_clip_0p25",
)
COMPUTED_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS
    if column not in {
        "direct_step_energy",
        "d7c_score_mean",
        "clip_fraction_0p125",
        "clip_fraction_0p25",
    }
)
META_COLUMNS = (
    "operator_label", "seed", "fold", "name", "clean_reference_group", "block_y", "block_x",
)
TARGET_COLUMNS = ("g1", "g1_epsilon", "g1_state")
EXPECTED_FORMAL_ACTIVE = {
    "D_ref": {"active": 503995, "beneficial": 293415, "harmful": 210558, "gray": 22},
    "D_rep": {"active": 503995, "beneficial": 293232, "harmful": 210755, "gray": 8},
}
EXPECTED_SMOKE_ACTIVE = {
    "D_ref": {"active": 14151, "beneficial": 9177, "harmful": 4974, "gray": 0},
    "D_rep": {"active": 14151, "beneficial": 9173, "harmful": 4978, "gray": 0},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_head(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def import_producer(v3p_tools_dir):
    tools = str(Path(v3p_tools_dir).resolve())
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("chd_rm_v3p_a0_canonical_signed_gain")


def computed_feature_vector(hazy, raw_low, raw_high, candidate_low, candidate_high, y0, x0, y1, x1):
    sl = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
    i_block = hazy[sl]
    raw_low_block = raw_low[sl]
    raw_high_block = raw_high[sl]
    low_block = candidate_low[sl]
    high_block = candidate_high[sl]
    midpoint = 0.5 * (low_block + high_block)
    delta = high_block - low_block
    residual = i_block - midpoint
    delta_means = delta.mean(dim=(2, 3)).reshape(-1)
    delta_stds = delta.std(dim=(2, 3), unbiased=False).reshape(-1)
    midpoint_mean = midpoint.mean(dim=(2, 3)).reshape(-1)
    covariances = []
    for channel in range(3):
        lhs = residual[:, channel:channel + 1]
        rhs = delta[:, channel:channel + 1]
        covariances.append(((lhs - lhs.mean()) * (rhs - rhs.mean())).mean())
    grad_residual_y = residual[:, :, 1:, :] - residual[:, :, :-1, :]
    grad_delta_y = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    grad_residual_x = residual[:, :, :, 1:] - residual[:, :, :, :-1]
    grad_delta_x = delta[:, :, :, 1:] - delta[:, :, :, :-1]
    gradient_alignment = (grad_residual_y * grad_delta_y).mean() + (grad_residual_x * grad_delta_x).mean()
    luminance = 0.299 * i_block[:, 0:1] + 0.587 * i_block[:, 1:2] + 0.114 * i_block[:, 2:3]
    maximum = i_block.max(dim=1, keepdim=True).values
    minimum = i_block.min(dim=1, keepdim=True).values
    values = {
        "delta_l1_mean": delta.abs().mean(),
        "delta_r_mean": delta_means[0],
        "delta_g_mean": delta_means[1],
        "delta_b_mean": delta_means[2],
        "delta_r_std": delta_stds[0],
        "delta_g_std": delta_stds[1],
        "delta_b_std": delta_stds[2],
        "midpoint_r_mean": midpoint_mean[0],
        "midpoint_g_mean": midpoint_mean[1],
        "midpoint_b_mean": midpoint_mean[2],
        "hazy_minus_midpoint_dot_delta": (residual * delta).mean(),
        "gradient_alignment": gradient_alignment,
        "cov_hazy_minus_midpoint_delta_r": covariances[0],
        "cov_hazy_minus_midpoint_delta_g": covariances[1],
        "cov_hazy_minus_midpoint_delta_b": covariances[2],
        "hazy_luminance_mean": luminance.mean(),
        "hazy_luminance_std": luminance.std(unbiased=False),
        "hazy_saturation_mean": ((maximum - minimum) / (maximum + 1e-6)).mean(),
        "signed_distance_to_clip_0p125": torch.minimum(raw_low_block, 1.0 - raw_low_block).mean(),
        "signed_distance_to_clip_0p25": torch.minimum(raw_high_block, 1.0 - raw_high_block).mean(),
    }
    return torch.stack([values[column] for column in COMPUTED_FEATURE_COLUMNS])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3p-repo", required=True)
    parser.add_argument("--expected-v3p-source-commit", required=True)
    parser.add_argument("--canonical-blocks", required=True)
    parser.add_argument("--expected-canonical-blocks-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fresh-split-manifest", required=True)
    parser.add_argument("--v3j-a-bounds", required=True)
    parser.add_argument("--a0-closeout", required=True)
    parser.add_argument("--operator-artifact-manifest", required=True)
    parser.add_argument("--density-artifact", required=True)
    parser.add_argument("--d7c-artifact", required=True)
    parser.add_argument("--reference-oof-rows", required=True)
    parser.add_argument("--v3m-a0-source-manifest", required=True)
    parser.add_argument("--numerical-contract", required=True)
    parser.add_argument("--expected-fresh-split-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-a0-closeout-sha256", required=True)
    parser.add_argument("--expected-parent-operator-manifest-sha256", required=True)
    parser.add_argument("--expected-reference-oof-rows-sha256", required=True)
    parser.add_argument("--expected-v3m-a0-source-manifest-sha256", required=True)
    parser.add_argument("--expected-density-artifact-sha256", required=True)
    parser.add_argument("--expected-d7c-artifact-sha256", required=True)
    parser.add_argument("--expected-a0-checkpoint-sha256", required=True)
    parser.add_argument("--expected-control-checkpoint-sha256", required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--train-key", default="v3j_controller_train")
    parser.add_argument("--confirm-key", default="v3j_route_confirm")
    parser.add_argument("--smoke-sample-count", type=int, default=32)
    parser.add_argument("--formal-sample-count", type=int, default=1200)
    parser.add_argument("--operator-labels", nargs="+", default=list(OPERATORS))
    parser.add_argument("--common-alphas", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--proj-channels", type=int, default=24)
    parser.add_argument("--d7c-threshold", type=float, default=None)
    parser.add_argument("--replay-tolerance-db", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--route-commit", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if tuple(args.operator_labels) != OPERATORS:
        raise ValueError("A0b requires exactly D_ref and D_rep in canonical order")
    if args.source_split != "train" or args.block_size != 16:
        raise ValueError("A0b is fixed to the v3p train-derived block16 contract")
    if tuple(args.common_alphas) != (0.0, 0.125, 0.25, 0.5, 1.0):
        raise ValueError("A0b requires the frozen v3p action ladder")
    expected_images = args.smoke_sample_count if args.run_mode == "smoke" else args.formal_sample_count
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    v3p_repo = Path(args.v3p_repo)
    if git_head(v3p_repo) != args.expected_v3p_source_commit:
        raise RuntimeError("v3p producer commit mismatch")
    canonical_blocks = Path(args.canonical_blocks)
    canonical_hash = sha256_file(canonical_blocks)
    if canonical_hash != args.expected_canonical_blocks_sha256:
        raise RuntimeError("canonical block source hash mismatch")
    producer = import_producer(v3p_repo / "experience_docx/tools")
    if args.d7c_threshold is None:
        args.d7c_threshold = producer.D7C_THRESHOLD
    producer.load_numerical_contract(args)
    input_hashes = producer.v3m_a1.verify_input_contract(args)
    _, artifacts = producer.v3l_a1.validate_authorization(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = producer.read_json(args.fresh_split_manifest)
    full_names = producer.names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    if len(full_names) != args.formal_sample_count:
        raise RuntimeError("frozen OOF manifest count mismatch")
    names = full_names[:expected_images]
    folds, _ = producer.v3l_a1.v3j_b.fold_assignments(full_names, args.fold_count)
    fold_by_name = dict(zip(full_names, folds.tolist()))
    reference = producer.v3m_a1.load_fixed_reference(args.reference_oof_rows)
    expected_reference = {(operator, name) for operator in OPERATORS for name in names}
    if not expected_reference.issubset(reference):
        raise RuntimeError("fixed-alpha reference does not cover A0b names")

    features_path = output_dir / f"{args.run_tag}_active_features_cloud_only.csv"
    fields = list(META_COLUMNS + TARGET_COLUMNS + FEATURE_COLUMNS)
    feature_counts = {operator: Counter() for operator in OPERATORS}
    device = torch.device(args.device)
    base = producer.build_model("original", args.a0_checkpoint, device)
    action_model = producer.build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = producer.build_gate_producer(args, device)
    bound = producer.read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    with features_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for artifact_info in artifacts:
            operator = artifact_info["operator_label"]
            if operator not in OPERATORS:
                continue
            artifact = torch.load(artifact_info["artifact_path"], map_location=device)
            if int(artifact.get("seed")) != int(artifact_info["seed"]):
                raise RuntimeError(f"operator artifact seed mismatch: {operator}")
            cache = producer.v3l_a0.build_model_cache(artifact, args, device)
            for index, name in enumerate(names):
                input_img, label = producer.load_pair(args.data_dir, args.source_split, name)
                input_img = input_img.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                padded, height, width = producer.pad_to_factor(input_img)
                label = label[:, :, :height, :width]
                hazy = padded[:, :, :height, :width]
                fold = int(fold_by_name[name])
                with torch.no_grad():
                    base_pred = producer.v3l_a1.forward_final(base, padded, height, width)
                    gate_full, score_full, _ = gate_producer(padded)
                    action_shape = producer.action_shape_for_input(padded)
                    hard_gate = producer.action_gate_from_full(gate_full, action_shape).to(device)
                    fmap, _, _, _ = producer.full_context_maps(action_model, gate_producer, padded)
                    model, mean, std = producer.v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                    pred_low = producer.v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound)
                    output_gate = producer.output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                    output_step = output_gate * F.interpolate(
                        pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False
                    )
                    blocks, _, _, _ = producer.candidate_block_rows(
                        base_pred, output_step, score_full, label, args.block_size, args
                    )
                    raw_low = base_pred + ACTION_PAIR[0] * output_step
                    raw_high = base_pred + ACTION_PAIR[1] * output_step
                    candidate_low = torch.clamp(raw_low, 0.0, 1.0)
                    candidate_high = torch.clamp(raw_high, 0.0, 1.0)
                    layout, _ = producer.block_layout(height, width, args.block_size)
                active_blocks = []
                computed_vectors = []
                for block_index, canonical in enumerate(blocks):
                    if float(canonical["direct_step_energy"]) <= 0.0:
                        continue
                    y0, y1, x0, x1 = layout[block_index]
                    active_blocks.append(canonical)
                    computed_vectors.append(computed_feature_vector(
                        hazy, raw_low, raw_high, candidate_low, candidate_high, y0, x0, y1, x1
                    ))
                computed_rows = torch.stack(computed_vectors).detach().cpu().tolist() if computed_vectors else []
                for canonical, computed_values in zip(active_blocks, computed_rows):
                    record = {
                        "operator_label": operator,
                        "seed": int(artifact_info["seed"]),
                        "fold": fold,
                        "name": name,
                        "clean_reference_group": producer.clean_reference_group(name),
                        "block_y": int(canonical["block_y"]),
                        "block_x": int(canonical["block_x"]),
                        "g1": float(canonical["gain_0125_to_025"]),
                        "g1_epsilon": float(canonical["gain_0125_to_025_epsilon"]),
                        "g1_state": canonical["gain_0125_to_025_state"],
                        "direct_step_energy": float(canonical["direct_step_energy"]),
                        "d7c_score_mean": float(canonical["d7c_score_mean"]),
                        "clip_fraction_0p125": float(canonical["clip_fraction_0p125"]),
                        "clip_fraction_0p25": float(canonical["clip_fraction_0p25"]),
                    }
                    record.update({
                        column: float(value)
                        for column, value in zip(COMPUTED_FEATURE_COLUMNS, computed_values)
                    })
                    writer.writerow(record)
                    feature_counts[operator]["active"] += 1
                    feature_counts[operator][record["g1_state"]] += 1
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
                del input_img, label, padded, hazy, base_pred, gate_full, score_full, hard_gate, fmap, pred_low, output_step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    verification = verify_against_canonical(canonical_blocks, features_path, set(names))
    expected_counts = EXPECTED_SMOKE_ACTIVE if args.run_mode == "smoke" else EXPECTED_FORMAL_ACTIVE
    compact_rows = []
    for operator in OPERATORS:
        observed = {
            "active": feature_counts[operator]["active"],
            "beneficial": feature_counts[operator]["beneficial"],
            "harmful": feature_counts[operator]["harmful"],
            "gray": feature_counts[operator]["abstain"],
        }
        if observed != expected_counts[operator]:
            raise RuntimeError(f"A0b active count mismatch for {operator}: {observed}")
        compact_rows.append({"operator_label": operator, **observed, **verification[operator]})

    schema = {
        "metadata_columns": list(META_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "model_feature_columns": list(FEATURE_COLUMNS),
        "forbidden_model_features": ["name", "seed", "fold", "clean_reference_group", "operator_label", "g1", "g1_epsilon", "g1_state"],
        "feature_schema_sha256": sha256_text(json.dumps(list(FEATURE_COLUMNS), separators=(",", ":"))),
    }
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "route_commit": args.route_commit,
        "v3p_repo": str(v3p_repo),
        "v3p_source_commit": args.expected_v3p_source_commit,
        "v3p_producer_script_sha256": sha256_file(v3p_repo / "experience_docx/tools/chd_rm_v3p_a0_canonical_signed_gain.py"),
        "canonical_blocks": str(canonical_blocks),
        "canonical_blocks_sha256": canonical_hash,
        "input_sha256": input_hashes,
        "feature_table_cloud_only": str(features_path),
        "schema": schema,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    decision = (
        "V3Q_A0B_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY"
        if args.run_mode == "smoke"
        else "V3Q_A0B_FORMAL_PASS_AUTHORIZE_A1_SIGNED_LINEAR_PROBE_ONLY"
    )
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3q-A0b-candidate-pair-feature-contract",
        "state": "COMPLETED_GATE_PASS",
        "gate_type": "structural_integrity",
        "decision": decision,
        "metric_contract": "v3q route card A0b candidate-pair feature contract",
        "authorizes": "v3q-A0b-formal" if args.run_mode == "smoke" else "v3q-A1 signed-linear probe only",
        "reason": "pinned producer/source assets, canonical active-key verification, and inference-only schema passed",
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(output_dir / f"{args.run_tag}_schema.json", schema)
    write_json(output_dir / f"{args.run_tag}_source_manifest.json", source_manifest)
    write_json(output_dir / f"{args.run_tag}_summary.json", {"operators": compact_rows, "verification": verification, "schema": schema})
    write_json(output_dir / f"{args.run_tag}_closeout.json", closeout)
    write_rows(output_dir / f"{args.run_tag}_by_operator.csv", compact_rows)
    print(decision)


def verify_against_canonical(canonical_blocks, features_path, selected_names):
    results = {operator: {"verified_rows": 0, "max_abs_direct_energy_diff": 0.0, "max_abs_g1_diff": 0.0} for operator in OPERATORS}
    with Path(canonical_blocks).open(newline="", encoding="utf-8") as source_handle, Path(features_path).open(newline="", encoding="utf-8") as feature_handle:
        source = csv.DictReader(source_handle)
        feature = csv.DictReader(feature_handle)
        feature_row_value = next(feature, None)
        for source_row in source:
            if source_row["name"] not in selected_names:
                continue
            if float(source_row["direct_step_energy"]) <= 0.0:
                continue
            if feature_row_value is None:
                raise RuntimeError("feature table ended before canonical active rows")
            key_source = (source_row["operator_label"], source_row["name"], source_row["block_y"], source_row["block_x"])
            key_feature = (feature_row_value["operator_label"], feature_row_value["name"], feature_row_value["block_y"], feature_row_value["block_x"])
            if key_source != key_feature:
                raise RuntimeError(f"canonical/feature key order mismatch: source={key_source} feature={key_feature}")
            operator = source_row["operator_label"]
            energy_diff = abs(float(source_row["direct_step_energy"]) - float(feature_row_value["direct_step_energy"]))
            g1_diff = abs(float(source_row["gain_0125_to_025"]) - float(feature_row_value["g1"]))
            if energy_diff > 1e-12:
                raise RuntimeError(f"direct-step-energy drift for {key_source}: {energy_diff}")
            if g1_diff > float(source_row["gain_0125_to_025_epsilon"]):
                raise RuntimeError(f"canonical G1 drift outside gray-zone tolerance for {key_source}: {g1_diff}")
            if source_row["gain_0125_to_025_state"] != feature_row_value["g1_state"]:
                raise RuntimeError(f"canonical state drift for {key_source}")
            results[operator]["verified_rows"] += 1
            results[operator]["max_abs_direct_energy_diff"] = max(results[operator]["max_abs_direct_energy_diff"], energy_diff)
            results[operator]["max_abs_g1_diff"] = max(results[operator]["max_abs_g1_diff"], g1_diff)
            feature_row_value = next(feature, None)
        if feature_row_value is not None:
            raise RuntimeError("feature table has rows beyond canonical active source")
    return results


if __name__ == "__main__":
    main()
