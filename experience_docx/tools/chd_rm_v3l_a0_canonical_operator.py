#!/usr/bin/env python3
"""v3l-A0 canonical direct-operator freeze.

This diagnostic performs one-time context direct-head fitting for fixed seeds,
saves cloud-only head artifacts, then verifies that loading the saved artifact
twice gives deterministic replay. It does not search hyperparameters, train a
backbone, touch locked test, or use route-confirm data for strategy selection.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import chd_rm_v3j_b_direct_correction_oof as v3j_b  # noqa: E402
from chd_rm_v3i_a_teacher_compressibility_audit import (  # noqa: E402
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    forward_final,
    load_pair,
    metric_pair,
    pad_to_factor,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from chd_rm_v3i_b_full_context_probe import full_context_maps  # noqa: E402
from chd_rm_v3j_a_bounded_action_audit import (  # noqa: E402
    names_from_manifest,
    summarize_policy,
)

ROUTE_ID = "haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711"
V3J_A_DECISION = "V3J_BOUNDED_ACTION_SPACE_PASS_AUTHORIZE_DIRECT_CORRECTION_OOF_ONLY"
V3J_B_DECISION = "V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER"
SEVERE_DB = -0.2


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_determinism(seed, strict):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except Exception as exc:  # pragma: no cover - environment-specific.
        if strict:
            raise
        print(f"v3l_a0_determinism_warn {exc}", flush=True)


def row_key(row):
    return (row["split"], int(row["fold"]), int(row["index"]), row["name"], row["policy"])


def compare_rows(rows_a, rows_b, tensor_rows, tolerance, tensor_tolerance, label):
    mismatches = []
    max_abs_psnr = 0.0
    if len(rows_a) != len(rows_b):
        mismatches.append({"kind": "row_count", "pass1": len(rows_a), "pass2": len(rows_b)})
    for idx, (a, b) in enumerate(zip(rows_a, rows_b)):
        ka = row_key(a)
        kb = row_key(b)
        if ka != kb:
            mismatches.append({"kind": "row_identity", "row": idx, "pass1": list(ka), "pass2": list(kb)})
            if len(mismatches) >= 20:
                break
            continue
        diff = abs(float(a["psnr_delta"]) - float(b["psnr_delta"]))
        max_abs_psnr = max(max_abs_psnr, diff)
        if diff > tolerance:
            mismatches.append(
                {
                    "kind": "psnr_delta",
                    "row": idx,
                    "key": list(ka),
                    "pass1": float(a["psnr_delta"]),
                    "pass2": float(b["psnr_delta"]),
                    "abs_diff": diff,
                }
            )
            if len(mismatches) >= 20:
                break
    severe_a = {
        row_key(row)
        for row in rows_a
        if row["policy"].endswith("DIRECT_CONTEXT") and float(row["psnr_delta"]) <= SEVERE_DB
    }
    severe_b = {
        row_key(row)
        for row in rows_b
        if row["policy"].endswith("DIRECT_CONTEXT") and float(row["psnr_delta"]) <= SEVERE_DB
    }
    severe_exact = severe_a == severe_b
    if not severe_exact:
        mismatches.append(
            {
                "kind": "severe_set",
                "pass1_only": len(severe_a - severe_b),
                "pass2_only": len(severe_b - severe_a),
            }
        )
    tensor_max = max((float(row["max_abs_direct_tensor_diff"]) for row in tensor_rows), default=0.0)
    tensor_pass = tensor_max <= tensor_tolerance
    if not tensor_pass:
        mismatches.append(
            {
                "kind": "tensor_replay",
                "max_abs_direct_tensor_diff": tensor_max,
                "tensor_tolerance": tensor_tolerance,
            }
        )
    return {
        "label": label,
        "row_count_pass1": len(rows_a),
        "row_count_pass2": len(rows_b),
        "max_abs_psnr_delta_diff": max_abs_psnr,
        "psnr_tolerance": tolerance,
        "max_abs_direct_tensor_diff": tensor_max,
        "tensor_tolerance": tensor_tolerance,
        "severe_threshold_db": SEVERE_DB,
        "severe_set_exact": severe_exact,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def cpu_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_context_operator(args, seed, train_names, train_folds, id_to_fold, device):
    previous_seed = getattr(args, "seed", None)
    args.seed = int(seed)
    set_determinism(seed, args.strict_determinism)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    arrays = v3j_b.make_arrays(args, train_names, train_folds, base, action_model, gate_producer, device)
    histories = []
    fold_models = {}
    for fold in range(args.fold_count):
        train_mask = arrays["fold"] != fold
        model, mean, std, history = v3j_b.train_head("context", arrays, train_mask, args, device, fold)
        fold_models[fold] = (model, mean, std)
        histories.extend(history)
        print(f"v3l_a0_seed_{seed}_fold_done {fold}", flush=True)
    all_mask = np.ones(arrays["target"].shape[0], dtype=bool)
    final_model, final_mean, final_std, history = v3j_b.train_head("context", arrays, all_mask, args, device, 99)
    histories.extend(history)
    artifact = {
        "route_id": ROUTE_ID,
        "artifact_kind": "canonical_context_direct_operator",
        "seed": int(seed),
        "operator_label": "D_ref" if int(seed) == args.ref_seed else "D_rep",
        "head_kind": "context",
        "source": "one-time diagnostic fitting; no model search; cloud-only artifact",
        "config": {
            "teacher_policy": args.teacher_policy,
            "fold_count": args.fold_count,
            "active_sample_per_image": args.active_sample_per_image,
            "inactive_sample_per_image": args.inactive_sample_per_image,
            "batch_size": args.batch_size,
            "context_steps": args.context_steps,
            "proj_channels": args.proj_channels,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
            "lambda_noop": args.lambda_noop,
            "d7c_threshold": args.d7c_threshold,
        },
        "train_names": list(train_names),
        "train_folds": [int(v) for v in train_folds.tolist()],
        "clean_id_to_fold": {str(k): int(v) for k, v in id_to_fold.items()},
        "models": {
            "folds": {
                str(fold): {
                    "state_dict": cpu_state_dict(model),
                    "mean": np.asarray(mean, dtype=np.float32),
                    "std": np.asarray(std, dtype=np.float32),
                }
                for fold, (model, mean, std) in sorted(fold_models.items())
            },
            "final": {
                "state_dict": cpu_state_dict(final_model),
                "mean": np.asarray(final_mean, dtype=np.float32),
                "std": np.asarray(final_std, dtype=np.float32),
            },
        },
        "sample_counts": {
            "controller_train_images": len(train_names),
            "sampled_training_sites": int(arrays["target"].shape[0]),
            "active_sample_fraction": float(np.mean(arrays["active"] > 0.5)),
            "clean_reference_count": len(id_to_fold),
        },
        "rng_state_after_fit": {
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda_all": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        },
    }
    if previous_seed is None:
        delattr(args, "seed")
    else:
        args.seed = previous_seed
    return artifact, histories


def build_context_model_from_pack(pack, args, device):
    first_weight = pack["state_dict"]["proj.weight"]
    channels = int(first_weight.shape[1])
    model = v3j_b.ContextResidualHead(channels, args.proj_channels).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()
    mean = np.asarray(pack["mean"], dtype=np.float32)
    std = np.asarray(pack["std"], dtype=np.float32)
    return model, mean, std


def build_model_cache(artifact, args, device):
    cache = {
        ("FINAL", 0): build_context_model_from_pack(artifact["models"]["final"], args, device)
    }
    for fold, pack in artifact["models"]["folds"].items():
        cache[("OOF", int(fold))] = build_context_model_from_pack(pack, args, device)
    return cache


def model_pack_from_cache(cache, prefix, fold):
    if prefix == "OOF":
        return cache[("OOF", int(fold))]
    return cache[("FINAL", 0)]


def replay_equivalence(args, artifact_path, names, prefix, folds_by_image, device):
    artifact_a = torch.load(artifact_path, map_location=device)
    artifact_b = torch.load(artifact_path, map_location=device)
    cache_a = build_model_cache(artifact_a, args, device)
    cache_b = build_model_cache(artifact_b, args, device)
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    rows_a = []
    rows_b = []
    tensor_rows = []
    for index, name in enumerate(names):
        input_img, label = load_pair(args.data_dir, args.source_split, name)
        input_img = input_img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        padded, height, width = pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        fold = int(folds_by_image[index])
        action_shape = action_shape_for_input(padded)
        with torch.no_grad():
            base_pred = forward_final(base, padded, height, width)
            _, base_psnr = metric_pair(base_pred, label)
            gate_full, _, _ = gate_producer(padded)
            hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
            hard_pred = forward_final(action_model, padded, height, width, d7c_gate=hard_gate)
            _, hard_psnr = metric_pair(hard_pred, label)
            fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
            model_a, mean_a, std_a = model_pack_from_cache(cache_a, prefix, fold)
            pred_low_a = v3j_b.score_map("context", model_a, fmap, mean_a, std_a, bound)
            pred_a = v3j_b.replay_prediction(base_pred, hard_gate, pred_low_a)
            _, direct_psnr_a = metric_pair(pred_a, label)
            model_b, mean_b, std_b = model_pack_from_cache(cache_b, prefix, fold)
            pred_low_b = v3j_b.score_map("context", model_b, fmap, mean_b, std_b, bound)
            pred_b = v3j_b.replay_prediction(base_pred, hard_gate, pred_low_b)
            _, direct_psnr_b = metric_pair(pred_b, label)
            tensor_diff = float(torch.max(torch.abs(pred_a - pred_b)).item())
        base_row = {
            "split": prefix,
            "fold": fold,
            "index": index,
            "name": name,
            "policy": f"{prefix}_A0",
            "psnr_delta": 0.0,
        }
        hard_row = {
            "split": prefix,
            "fold": fold,
            "index": index,
            "name": name,
            "policy": f"{prefix}_HARD_D7C_ALPHA1",
            "psnr_delta": hard_psnr - base_psnr,
        }
        direct_a = {
            "split": prefix,
            "fold": fold,
            "index": index,
            "name": name,
            "policy": f"{prefix}_DIRECT_CONTEXT",
            "psnr_delta": direct_psnr_a - base_psnr,
        }
        direct_b = {
            "split": prefix,
            "fold": fold,
            "index": index,
            "name": name,
            "policy": f"{prefix}_DIRECT_CONTEXT",
            "psnr_delta": direct_psnr_b - base_psnr,
        }
        rows_a.extend([base_row, hard_row, direct_a])
        rows_b.extend([dict(base_row), dict(hard_row), direct_b])
        tensor_rows.append(
            {
                "split": prefix,
                "fold": fold,
                "index": index,
                "name": name,
                "policy": f"{prefix}_DIRECT_CONTEXT",
                "max_abs_direct_tensor_diff": tensor_diff,
            }
        )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"v3l_a0_replay_{Path(artifact_path).stem}_{prefix.lower()} {index + 1}/{len(names)}", flush=True)
        del input_img, label, padded, base_pred, hard_gate, fmap, pred_a, pred_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows_a, rows_b, tensor_rows


def validate_authorization(args):
    if args.source_split.lower() != "train":
        raise ValueError(f"v3l-A0 is train-derived only; got source_split={args.source_split!r}")
    for key in (args.train_key, args.confirm_key):
        if "test" in key.lower():
            raise ValueError(f"locked-test-like split key is forbidden for v3l-A0: {key!r}")
    v3j_a = read_json(args.v3j_a_summary)
    v3j_b = read_json(args.v3j_b_summary)
    if v3j_a.get("decision") != V3J_A_DECISION:
        raise RuntimeError(f"unexpected v3j-A decision: {v3j_a.get('decision')}")
    if v3j_b.get("decision") != V3J_B_DECISION:
        raise RuntimeError(f"unexpected v3j-B decision: {v3j_b.get('decision')}")
    return v3j_a, v3j_b


def write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    v3j_a, v3j_b_summary = validate_authorization(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_json(args.fresh_split_manifest)
    train_names = names_from_manifest(manifest, args.train_key, args.max_train_samples)
    confirm_names = names_from_manifest(manifest, args.confirm_key, args.max_confirm_samples)
    train_folds, id_to_fold = v3j_b.fold_assignments(train_names, args.fold_count)
    confirm_folds = np.zeros(len(confirm_names), dtype=np.int64)
    all_results = []
    manifest_rows = []
    all_history = []

    for seed in args.seeds:
        label = "D_ref" if int(seed) == args.ref_seed else "D_rep"
        artifact_path = artifact_dir / f"v3l_a0_{label}_context_seed{int(seed)}.pt"
        if artifact_path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {artifact_path}")
        artifact, history = train_context_operator(args, int(seed), train_names, train_folds, id_to_fold, device)
        all_history.extend({"seed": int(seed), **row} for row in history)
        torch.save(artifact, artifact_path)
        artifact_sha1 = sha256_path(artifact_path)
        artifact_sha2 = sha256_path(artifact_path)

        seed_results = {
            "seed": int(seed),
            "operator_label": label,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha1,
            "artifact_sha256_repeat": artifact_sha2,
            "artifact_sha_stable": artifact_sha1 == artifact_sha2,
            "sample_counts": artifact["sample_counts"],
            "replay": [],
        }
        for prefix, names, folds in (
            ("OOF", train_names, train_folds),
            ("CONFIRM_AUDIT_ONLY", confirm_names, confirm_folds),
        ):
            rows_a, rows_b, tensor_rows = replay_equivalence(args, artifact_path, names, prefix, folds, device)
            replay_cmp = compare_rows(
                rows_a,
                rows_b,
                tensor_rows,
                args.replay_psnr_tolerance,
                args.replay_tensor_tolerance,
                f"{label}_{prefix}",
            )
            replay_cmp["summary_pass1"] = summarize_policy(rows_a)
            replay_cmp["route_confirm_used_for_strategy_selection"] = False
            seed_results["replay"].append(replay_cmp)
            stem = f"v3l_a0_{label}_seed{int(seed)}_{prefix.lower()}"
            write_csv(output_dir / f"{stem}_replay_pass1.csv", rows_a)
            write_csv(output_dir / f"{stem}_replay_pass2.csv", rows_b)
            write_csv(output_dir / f"{stem}_tensor_equivalence.csv", tensor_rows)
        seed_results["passed"] = seed_results["artifact_sha_stable"] and all(item["passed"] for item in seed_results["replay"])
        all_results.append(seed_results)
        manifest_rows.append(
            {
                "operator_label": label,
                "seed": int(seed),
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_sha1,
                "artifact_size_bytes": artifact_path.stat().st_size,
                "cloud_only_not_for_github": True,
                "passed": seed_results["passed"],
            }
        )

    decision = (
        "V3L_A0_CANONICAL_OPERATOR_REPLAY_PASS_AUTHORIZE_A1_ORACLE_GRANULARITY_AUDIT"
        if all(item["passed"] for item in all_results)
        else "V3L_A0_CANONICAL_OPERATOR_REPLAY_FAIL_STOP"
    )
    closeout = {
        "route_id": ROUTE_ID,
        "phase": "v3l-A0 canonical context direct-operator freeze",
        "decision": decision,
        "locked_test_touched": False,
        "canary_authorized": False,
        "new_model_search_authorized": False,
        "operator_artifacts_saved_cloud_only": True,
        "raw_feature_tensors_saved": False,
        "route_confirm_role": "deterministic replay audit only; not strategy selection",
        "route_confirm_used_for_strategy_selection": False,
        "next_stage_authorized": "v3l-A1 oracle granularity audit only" if decision.endswith("AUDIT") else "none",
        "metric_contract": {
            "baseline": "A0 PSNR on train-derived clean-reference grouped OOF",
            "operator": "context direct bounded residual head under D7c output veto",
            "replay_gate": {
                "row_identity_order": "exact",
                "max_abs_psnr_delta_diff": args.replay_psnr_tolerance,
                "max_abs_direct_tensor_diff": args.replay_tensor_tolerance,
                "severe_set": "exact at <= -0.2 dB",
                "artifact_sha": "stable repeat hash",
            },
        },
        "source_assets": {
            "a0_checkpoint": args.a0_checkpoint,
            "a0_checkpoint_sha256": sha256_file(args.a0_checkpoint),
            "control_checkpoint": args.control_checkpoint,
            "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
            "data_dir": args.data_dir,
            "fresh_split_manifest": args.fresh_split_manifest,
            "v3j_a_summary": args.v3j_a_summary,
            "v3j_a_decision": v3j_a.get("decision"),
            "v3j_b_summary": args.v3j_b_summary,
            "v3j_b_decision": v3j_b_summary.get("decision"),
        },
        "results": all_results,
    }
    write_csv(output_dir / "v3l_a0_probe_training_history.csv", all_history)
    write_csv(output_dir / "v3l_a0_canonical_operator_artifact_manifest.csv", manifest_rows)
    write_json(output_dir / "v3l_a0_canonical_operator_artifact_manifest.json", manifest_rows)
    write_json(output_dir / "v3l_a0_canonical_operator_closeout.json", closeout)
    write_text(
        output_dir / "v3l_a0_cloud_only_artifact_notice.md",
        "# v3l-A0 Cloud-Only Artifact Notice\n\n"
        "Direct-head state_dict artifacts are saved under `cloud_only_artifacts/` on convir-4090 only. "
        "They are not intended for GitHub sync; GitHub should receive only SHA/path manifests and compact replay summaries.\n",
    )
    print(json.dumps(closeout, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_summary", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--v3j_b_summary", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_confirm_samples", type=int, default=600)
    parser.add_argument("--teacher_policy", default="CC_MIN4_FROM_OPEN_TOP_0.5")
    parser.add_argument("--top_fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--block_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--smooth_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--min_component_sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--interior_kernels", type=int, nargs="*", default=[])
    parser.add_argument("--uniform_alphas", type=float, nargs="*", default=[])
    parser.add_argument("--score_eps", type=float, default=0.0)
    parser.add_argument("--denom_eps", type=float, default=1e-12)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--active_sample_per_image", type=int, default=96)
    parser.add_argument("--inactive_sample_per_image", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--context_steps", type=int, default=480)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--lambda_noop", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408])
    parser.add_argument("--ref_seed", type=int, default=3407)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=120)
    parser.add_argument("--replay_psnr_tolerance", type=float, default=1e-6)
    parser.add_argument("--replay_tensor_tolerance", type=float, default=1e-7)
    parser.add_argument("--strict_determinism", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
