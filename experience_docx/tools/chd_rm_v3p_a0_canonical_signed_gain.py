#!/usr/bin/env python3
"""v3p-A0 canonical candidate-SSE reconstruction for frozen v3l operators.

Candidate construction remains the frozen v3o deployment calculation.  Only
the loss contract changes: both full-image and block losses are reduced from
the same pre-square float64 error tensor and are cross-checked against an
independent CPU longdouble implementation.
"""

import argparse
import csv
import hashlib
import json
import math
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

import chd_rm_v3m_a1_local_actuation_audit as v3m_a1
import chd_rm_v3l_a0_canonical_operator as v3l_a0
import chd_rm_v3l_a1_oracle_granularity_audit as v3l_a1
from chd_rm_v3i_a_teacher_compressibility_audit import (
    D7C_THRESHOLD,
    action_gate_from_full,
    action_shape_for_input,
    build_gate_producer,
    build_model,
    load_pair,
    metric_pair,
    pad_to_factor,
    read_json,
)
from chd_rm_v3i_b_full_context_probe import full_context_maps
from chd_rm_v3j_a_bounded_action_audit import names_from_manifest, output_gate_from_action_gate


ROUTE_ID = "haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712"
OPERATORS = ("D_ref", "D_rep")
ACTION_LADDER = (0.0, 0.125, 0.25, 0.5, 1.0)
FIXED_ALPHA = 0.125
NUMERICAL_CONTRACT_VERSION = "v3p-a0-canonical-sse-v1"
MIN_ATOL_SSE = 1e-12
MIN_RTOL = 1e-12
ENVELOPE_MULTIPLIER = 8.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def alpha_key(alpha):
    return f"{alpha:g}".replace(".", "p")


def clean_reference_group(name):
    stem = Path(name).stem
    return stem.split("_", 1)[0]


def block_layout(height, width, block_size):
    """Return an exact non-overlapping block partition and its coverage audit."""
    coverage = np.zeros((height, width), dtype=np.uint8)
    layout = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            coverage[y0:y1, x0:x1] += 1
            layout.append((y0, y1, x0, x1))
    if coverage.min() != 1 or coverage.max() != 1 or int(coverage.sum()) != height * width:
        raise RuntimeError("block layout does not cover every spatial pixel exactly once")
    return layout, {
        "height": height,
        "width": width,
        "block_count": len(layout),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_sum": int(coverage.sum()),
    }


def longdouble_sum(tensor):
    """Independent CPU high-precision reduction of an already canonical error tensor."""
    values = tensor.detach().cpu().contiguous().numpy()
    return float(np.sum(values, dtype=np.longdouble))


def tolerance_sse(reference_sse, args):
    return float(args.atol_sse + args.rtol * abs(reference_sse))


def normalized_error(observed, reference, args):
    return abs(observed - reference) / tolerance_sse(reference, args)


def sign_state(gain, epsilon):
    if gain > epsilon:
        return "beneficial"
    if gain < -epsilon:
        return "harmful"
    return "abstain"


def candidate_block_rows(base_pred, output_step, score_full, label, block_size, args):
    """Compute all loss views from each candidate's same float64 error tensor."""
    height, width = base_pred.shape[-2:]
    score_map = F.interpolate(score_full, size=(height, width), mode="bilinear", align_corners=False)
    raw_candidates = {alpha: base_pred + alpha * output_step for alpha in ACTION_LADDER}
    predictions = {alpha: torch.clamp(raw, 0.0, 1.0) for alpha, raw in raw_candidates.items()}
    clip_maps = {
        alpha: ((raw < 0.0) | (raw > 1.0)).any(dim=1, keepdim=True).to(base_pred.dtype)
        for alpha, raw in raw_candidates.items()
    }
    layout, coverage = block_layout(height, width, block_size)
    errors = {
        alpha: (predictions[alpha].to(torch.float64) - label.to(torch.float64)).square()
        for alpha in ACTION_LADDER
    }
    block_sse = {alpha: [] for alpha in ACTION_LADDER}
    gpu_block_sse = {alpha: [] for alpha in ACTION_LADDER}
    direct = {}
    for alpha in ACTION_LADDER:
        direct_reference = longdouble_sum(errors[alpha])
        direct_gpu = float(torch.sum(errors[alpha], dtype=torch.float64).item())
        for y0, y1, x0, x1 in layout:
            sl = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
            block_sse[alpha].append(longdouble_sum(errors[alpha][sl]))
            gpu_block_sse[alpha].append(float(torch.sum(errors[alpha][sl], dtype=torch.float64).item()))
        direct[alpha] = {
            "reference_sse": direct_reference,
            "gpu_sse": direct_gpu,
            "blocks_sse": math.fsum(block_sse[alpha]),
            "gpu_blocks_sse": math.fsum(gpu_block_sse[alpha]),
            "block_e_max": max(
                normalized_error(value, ref, args)
                for value, ref in zip(gpu_block_sse[alpha], block_sse[alpha])
            ),
        }
    rows = []
    for block_index, (y0, y1, x0, x1) in enumerate(layout):
        sl = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
        candidate_sse = {alpha: block_sse[alpha][block_index] for alpha in ACTION_LADDER}
        best_alpha = min(ACTION_LADDER, key=lambda alpha: candidate_sse[alpha])
        step_block = output_step[sl]
        row = {
                "block_y": y0 // block_size,
                "block_x": x0 // block_size,
                "candidate_pixel_count": int(label[sl].numel()),
                "oracle_alpha": best_alpha,
                "direct_step_energy": float(torch.mean(step_block * step_block).item()),
                "d7c_score_mean": float(torch.mean(score_map[:, :, y0:y1, x0:x1]).item()),
        }
        for alpha in ACTION_LADDER:
            key = alpha_key(alpha)
            row[f"candidate_sse_{key}"] = candidate_sse[alpha]
            row[f"candidate_sse_e_{key}"] = tolerance_sse(candidate_sse[alpha], args)
            row[f"clip_fraction_{key}"] = float(torch.mean(clip_maps[alpha][:, :, y0:y1, x0:x1]).item())
        row["gain_0125_to_025"] = candidate_sse[0.125] - candidate_sse[0.25]
        row["gain_025_to_05"] = candidate_sse[0.25] - candidate_sse[0.5]
        row["gain_05_to_10"] = candidate_sse[0.5] - candidate_sse[1.0]
        row["gain_0125_to_025_epsilon"] = row["candidate_sse_e_0p125"] + row["candidate_sse_e_0p25"]
        row["gain_0125_to_025_state"] = sign_state(
            row["gain_0125_to_025"], row["gain_0125_to_025_epsilon"]
        )
        rows.append(row)
    return rows, direct, coverage, gpu_block_sse


def add_gain(accumulator, row):
    for name in ("gain_0125_to_025", "gain_025_to_05", "gain_05_to_10"):
        value = float(row[name])
        accumulator[name]["count"] += 1
        accumulator[name]["total"] += value
        accumulator[name]["beneficial_sse"] += max(value, 0.0)
        accumulator[name]["harmful_sse"] += max(-value, 0.0)
        accumulator[name]["beneficial_count"] += int(value > 0.0)
        accumulator[name]["harmful_count"] += int(value < 0.0)


def gain_row(operator, fold, values):
    row = {"operator_label": operator, "fold": fold}
    for name, stats in values.items():
        beneficial = stats["beneficial_sse"]
        harmful = stats["harmful_sse"]
        row.update(
            {
                f"{name}_block_count": stats["count"],
                f"{name}_beneficial_block_count": stats["beneficial_count"],
                f"{name}_harmful_block_count": stats["harmful_count"],
                f"{name}_net_sse": stats["total"],
                f"{name}_beneficial_sse": beneficial,
                f"{name}_harmful_sse": harmful,
                f"{name}_harmful_to_beneficial": harmful / beneficial if beneficial > 0.0 else float("nan"),
            }
        )
    return row


def new_gain_accumulator():
    return {
        name: {
            "count": 0,
            "total": 0.0,
            "beneficial_sse": 0.0,
            "harmful_sse": 0.0,
            "beneficial_count": 0,
            "harmful_count": 0,
        }
        for name in ("gain_0125_to_025", "gain_025_to_05", "gain_05_to_10")
    }


def correlation(rows, field):
    paired = defaultdict(dict)
    for row in rows:
        paired[row["name"]][row["operator_label"]] = float(row[field])
    values = [item for item in paired.values() if set(item) == set(OPERATORS)]
    if len(values) < 2:
        return float("nan"), len(values)
    ref = np.asarray([item["D_ref"] for item in values], dtype=np.float64)
    rep = np.asarray([item["D_rep"] for item in values], dtype=np.float64)
    return float(np.corrcoef(ref, rep)[0, 1]), len(values)


def run_legacy(args):
    if args.source_split.lower() != "train":
        raise ValueError("v3o-A0 is train-derived only")
    if args.block_size != 16:
        raise ValueError("v3o-A0 is fixed to block16")
    if tuple(args.common_alphas) != ACTION_LADDER:
        raise ValueError("v3o-A0 requires the fixed [0, .125, .25, .5, 1] ladder")
    if args.run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be smoke or formal")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "blocks": output_dir / f"{args.run_tag}_block_candidate_losses_cloud_only.csv",
        "images": output_dir / f"{args.run_tag}_image_candidate_replay_cloud_only.csv",
        "integrity": output_dir / f"{args.run_tag}_replay_integrity.json",
        "gains": output_dir / f"{args.run_tag}_adjacent_gain_summary.csv",
        "gain_folds": output_dir / f"{args.run_tag}_adjacent_gain_by_fold_operator.csv",
        "agreement": output_dir / f"{args.run_tag}_cross_operator_gain_agreement.csv",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite v3o-A0 outputs: " + ", ".join(existing))

    input_hashes = v3m_a1.verify_input_contract(args)
    _, artifacts = v3l_a1.validate_authorization(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = read_json(args.fresh_split_manifest)
    full_names = names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    if len(full_names) != args.formal_sample_count:
        raise ValueError(f"expected {args.formal_sample_count} OOF names, got {len(full_names)}")
    names = full_names[: args.max_train_samples]
    expected_count = args.smoke_sample_count if args.run_mode == "smoke" else args.formal_sample_count
    if len(names) != expected_count:
        raise ValueError(f"{args.run_mode} run requires {expected_count} names, got {len(names)}")
    full_folds, _ = v3l_a1.v3j_b.fold_assignments(full_names, args.fold_count)
    fold_by_name = dict(zip(full_names, full_folds.tolist()))
    reference = v3m_a1.load_fixed_reference(args.reference_oof_rows)
    expected_reference = {(operator, name) for operator in args.operator_labels for name in names}
    if not expected_reference.issubset(reference):
        raise ValueError("fixed-alpha reference table does not cover requested OOF rows")

    device = torch.device(args.device)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    image_rows = []
    by_operator = defaultdict(new_gain_accumulator)
    by_fold = defaultdict(new_gain_accumulator)
    integrity_rows = []
    raw_fields = [
        "operator_label", "seed", "fold", "index", "name", "clean_reference_group",
        "block_y", "block_x", "candidate_pixel_count", "oracle_alpha", "direct_step_energy", "d7c_score_mean",
    ]
    for alpha in ACTION_LADDER:
        key = alpha_key(alpha)
        raw_fields.extend((f"candidate_sse_{key}", f"clip_fraction_{key}"))
    raw_fields.extend(("gain_0125_to_025", "gain_025_to_05", "gain_05_to_10"))

    with outputs["blocks"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, lineterminator="\n")
        writer.writeheader()
        for artifact_info in artifacts:
            operator = artifact_info["operator_label"]
            if operator not in set(args.operator_labels):
                continue
            artifact = torch.load(artifact_info["artifact_path"], map_location=device)
            if int(artifact.get("seed")) != int(artifact_info["seed"]):
                raise RuntimeError(f"operator artifact seed mismatch for {operator}")
            cache = v3l_a0.build_model_cache(artifact, args, device)
            max_fixed_replay_diff = 0.0
            max_mse_aggregation_diff = 0.0
            block_count = 0
            for index, name in enumerate(names):
                input_img, label = load_pair(args.data_dir, args.source_split, name)
                input_img = input_img.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                padded, height, width = pad_to_factor(input_img)
                label = label[:, :, :height, :width]
                fold = int(fold_by_name[name])
                action_shape = action_shape_for_input(padded)
                with torch.no_grad():
                    base_pred = v3l_a1.forward_final(base, padded, height, width)
                    _, base_psnr = metric_pair(base_pred, label)
                    gate_full, score_full, _ = gate_producer(padded)
                    hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
                    fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
                    model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                    pred_low = v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound)
                    output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                    output_step = output_gate * F.interpolate(
                        pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False
                    )
                    blocks, direct = candidate_block_rows(base_pred, output_step, score_full, label, args.block_size)
                    total_pixels = sum(int(row["candidate_pixel_count"]) for row in blocks)
                    image_row = {
                        "operator_label": operator,
                        "seed": int(artifact_info["seed"]),
                        "fold": fold,
                        "index": index,
                        "name": name,
                        "clean_reference_group": clean_reference_group(name),
                        "base_psnr": float(base_psnr),
                        "candidate_pixel_count": total_pixels,
                    }
                    for alpha in ACTION_LADDER:
                        key = alpha_key(alpha)
                        block_sse = sum(float(row[f"candidate_sse_{key}"]) for row in blocks)
                        block_mse = block_sse / total_pixels
                        image_row[f"candidate_sse_{key}"] = direct[alpha]["sse"]
                        image_row[f"candidate_mse_direct_{key}"] = direct[alpha]["mse"]
                        image_row[f"candidate_mse_from_blocks_{key}"] = block_mse
                        image_row[f"candidate_mse_abs_diff_{key}"] = abs(direct[alpha]["mse"] - block_mse)
                        max_mse_aggregation_diff = max(max_mse_aggregation_diff, image_row[f"candidate_mse_abs_diff_{key}"])
                    fixed_mse = direct[FIXED_ALPHA]["mse"]
                    fixed_psnr = 10.0 * math.log10(1.0 / max(fixed_mse, 1e-30))
                    fixed_delta = fixed_psnr - float(base_psnr)
                    reference_delta = reference[(operator, name)]
                    fixed_diff = abs(fixed_delta - reference_delta)
                    max_fixed_replay_diff = max(max_fixed_replay_diff, fixed_diff)
                    if fixed_diff > args.replay_tolerance_db:
                        raise RuntimeError(
                            f"fixed-alpha replay mismatch for {operator}/{name}: {fixed_delta} vs {reference_delta}"
                        )
                    image_row["fixed_psnr_delta"] = fixed_delta
                    image_row["reference_fixed_psnr_delta"] = reference_delta
                    image_row["fixed_replay_abs_diff_db"] = fixed_diff
                for row in blocks:
                    record = {
                        "operator_label": operator,
                        "seed": int(artifact_info["seed"]),
                        "fold": fold,
                        "index": index,
                        "name": name,
                        "clean_reference_group": clean_reference_group(name),
                        **row,
                    }
                    writer.writerow(record)
                    add_gain(by_operator[operator], row)
                    add_gain(by_fold[(operator, fold)], row)
                    block_count += 1
                image_rows.append(image_row)
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
                del input_img, label, padded, base_pred, hard_gate, fmap, output_step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            integrity_rows.append(
                {
                    "operator_label": operator,
                    "seed": int(artifact_info["seed"]),
                    "image_count": len(names),
                    "block_count": block_count,
                    "fixed_alpha_max_abs_psnr_delta_diff_db": max_fixed_replay_diff,
                    "candidate_mse_aggregation_max_abs_diff": max_mse_aggregation_diff,
                    "fixed_replay_pass": max_fixed_replay_diff <= args.replay_tolerance_db,
                    "candidate_sse_aggregation_pass": max_mse_aggregation_diff <= args.mse_aggregation_tolerance,
                }
            )

    gain_rows = [gain_row(operator, "ALL", by_operator[operator]) for operator in args.operator_labels]
    fold_rows = [gain_row(operator, fold, by_fold[(operator, fold)]) for operator in args.operator_labels for fold in range(args.fold_count)]
    agreement_rows = []
    for field in ("candidate_sse_0p125", "candidate_sse_0p25", "candidate_sse_0p5", "candidate_sse_1", "fixed_psnr_delta"):
        value, n = correlation(image_rows, field)
        agreement_rows.append({"field": field, "paired_image_count": n, "pearson_ref_rep": value})
    all_pass = all(row["fixed_replay_pass"] and row["candidate_sse_aggregation_pass"] for row in integrity_rows)
    if args.run_mode == "smoke":
        decision = (
            "V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY"
            if all_pass
            else "V3O_A0_SMOKE_REPLAY_INTEGRITY_FAIL_STOP"
        )
        next_stage = "v3o-A0 formal 1200-image OOF candidate-SSE audit only" if all_pass else "none"
    else:
        decision = (
            "V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_PASS_AUTHORIZE_A1_SCORE_SUFFICIENCY_AUDIT_ONLY"
            if all_pass
            else "V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_FAIL_STOP"
        )
        next_stage = "v3o-A1 direct-step-energy sufficiency audit only" if all_pass else "none"
    integrity = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "decision": decision,
        "next_stage_authorized": next_stage,
        "integrity_rows": integrity_rows,
        "fixed_alpha_replay_tolerance_db": args.replay_tolerance_db,
        "candidate_mse_aggregation_tolerance": args.mse_aggregation_tolerance,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "parent_evidence_main_commit": args.parent_evidence_main_commit,
        "runnable_source_commit": args.runnable_source_commit,
        "input_sha256": input_hashes,
        "script_sha256": sha256_file(__file__),
        "operator_labels": list(args.operator_labels),
        "action_ladder": list(ACTION_LADDER),
        "fixed_alpha": FIXED_ALPHA,
        "raw_block_candidate_losses_cloud_only": str(outputs["blocks"]),
        "raw_image_candidate_replay_cloud_only": str(outputs["images"]),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    summary = {
        **integrity,
        "phase": "v3o-A0 frozen candidate-SSE and signed-adjacent-gain audit",
        "metric_contract": {
            "source_split": "train-derived clean-reference grouped OOF only",
            "block_size": args.block_size,
            "action_ladder": list(ACTION_LADDER),
            "baseline": "fixed alpha=0.125 on the same operator/name rows",
            "primary_loss": "additive RGB block SSE",
            "transitions": ["0.125_to_0.25", "0.25_to_0.5", "0.5_to_1.0"],
        },
        "image_count_per_operator": len(names),
        "gain_rows": gain_rows,
        "agreement_rows": agreement_rows,
        "source_manifest": source_manifest,
    }
    write_rows(outputs["images"], image_rows)
    write_json(outputs["integrity"], integrity)
    write_rows(outputs["gains"], gain_rows)
    write_rows(outputs["gain_folds"], fold_rows)
    write_rows(outputs["agreement"], agreement_rows)
    write_json(outputs["summary"], summary)
    write_json(outputs["source"], source_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def numerical_preflight(args):
    """Freeze a hardware-specific numeric envelope without reading formal OOF data."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.run_tag}_numerical_reference_contract.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite numerical preflight: {output}")
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise RuntimeError("numpy longdouble is not higher precision than float64 on this host")
    device = torch.device(args.device)
    shapes = ((1, 3, 400, 400), (1, 3, 480, 640))
    patterns = ("constant", "alternating", "deterministic_uniform")
    rows = []
    max_abs = 0.0
    max_relative = 0.0
    for shape in shapes:
        count = int(np.prod(shape))
        index = torch.arange(count, dtype=torch.float64, device=device)
        for pattern in patterns:
            if pattern == "constant":
                error = torch.full(shape, 1.0 / 9.0, dtype=torch.float64, device=device)
            elif pattern == "alternating":
                flat = torch.where(index.remainder(2) == 0, torch.ones_like(index), torch.full_like(index, 2.0 ** -40))
                error = flat.reshape(shape)
            else:
                flat = torch.remainder(index * 0.6180339887498949, 1.0).square()
                error = flat.reshape(shape)
            layout, coverage = block_layout(shape[-2], shape[-1], args.block_size)
            reference = longdouble_sum(error)
            gpu_direct = float(torch.sum(error, dtype=torch.float64).item())
            reference_blocks = []
            gpu_blocks = []
            for y0, y1, x0, x1 in layout:
                sl = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
                reference_blocks.append(longdouble_sum(error[sl]))
                gpu_blocks.append(float(torch.sum(error[sl], dtype=torch.float64).item()))
            values = {
                "gpu_direct": gpu_direct,
                "reference_blocks_fsum": math.fsum(reference_blocks),
                "gpu_blocks_fsum": math.fsum(gpu_blocks),
            }
            errors = {name: abs(value - reference) for name, value in values.items()}
            max_abs = max(max_abs, *errors.values())
            max_relative = max(max_relative, *(value / max(abs(reference), 1.0) for value in errors.values()))
            rows.append({
                "shape": list(shape),
                "pattern": pattern,
                "reference_sse": reference,
                "coverage": coverage,
                "absolute_errors": errors,
            })
            del error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    contract = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3p-A0-numerical-preflight",
        "state": "COMPLETED_GATE_PASS",
        "contract_version": NUMERICAL_CONTRACT_VERSION,
        "gate_type": "numerical_equivalence",
        "reference": "CPU numpy.longdouble full SSE; CPU longdouble blocks plus math.fsum",
        "deployment_semantics": "candidate dtype, clamp, and frozen operator semantics are unchanged; only loss reduction is canonicalized",
        "stress_shapes": [list(shape) for shape in shapes],
        "stress_patterns": list(patterns),
        "stress_rows": rows,
        "max_stress_abs_sse_error": max_abs,
        "max_stress_relative_sse_error": max_relative,
        "atol_sse": max(MIN_ATOL_SSE, ENVELOPE_MULTIPLIER * max_abs),
        "rtol": max(MIN_RTOL, ENVELOPE_MULTIPLIER * max_relative),
        "envelope_multiplier": ENVELOPE_MULTIPLIER,
        "gray_zone": "epsilon_G = tol(L_0.125) + tol(L_0.25); gray blocks abstain",
        "decision": "V3P_A0_NUMERICAL_PREFLIGHT_PASS_AUTHORIZE_SMOKE_ONLY",
        "authorizes": "v3p-A0 smoke only",
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


def load_numerical_contract(args):
    if not args.numerical_contract:
        raise ValueError("smoke/formal requires --numerical_contract from v3p numerical preflight")
    contract = read_json(args.numerical_contract)
    if contract.get("route_id") != ROUTE_ID or contract.get("contract_version") != NUMERICAL_CONTRACT_VERSION:
        raise ValueError("numerical contract route identity mismatch")
    if contract.get("decision") != "V3P_A0_NUMERICAL_PREFLIGHT_PASS_AUTHORIZE_SMOKE_ONLY":
        raise ValueError("numerical preflight did not authorize v3p-A0 smoke/formal")
    args.atol_sse = float(contract["atol_sse"])
    args.rtol = float(contract["rtol"])
    return contract


def run(args):
    if args.run_mode == "numerical_preflight":
        numerical_preflight(args)
        return
    if args.run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be numerical_preflight, smoke, or formal")
    if args.source_split.lower() != "train" or args.block_size != 16:
        raise ValueError("v3p-A0 is fixed to train-derived block16 OOF reconstruction")
    if tuple(args.common_alphas) != ACTION_LADDER:
        raise ValueError("v3p-A0 requires the frozen [0, .125, .25, .5, 1] ladder")
    contract = load_numerical_contract(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "blocks": output_dir / f"{args.run_tag}_block_candidate_losses_cloud_only.csv",
        "images": output_dir / f"{args.run_tag}_image_candidate_replay_cloud_only.csv",
        "integrity": output_dir / f"{args.run_tag}_closeout.json",
        "gains": output_dir / f"{args.run_tag}_adjacent_gain_summary.csv",
        "gain_folds": output_dir / f"{args.run_tag}_adjacent_gain_by_fold_operator.csv",
        "agreement": output_dir / f"{args.run_tag}_cross_operator_gain_agreement.csv",
        "summary": output_dir / f"{args.run_tag}_summary.json",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite v3p-A0 outputs: " + ", ".join(existing))

    input_hashes = v3m_a1.verify_input_contract(args)
    _, artifacts = v3l_a1.validate_authorization(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = read_json(args.fresh_split_manifest)
    full_names = names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    names = full_names[:args.max_train_samples]
    expected_count = args.smoke_sample_count if args.run_mode == "smoke" else args.formal_sample_count
    if len(full_names) != args.formal_sample_count or len(names) != expected_count:
        raise ValueError("OOF name count does not match the frozen v3p-A0 contract")
    full_folds, _ = v3l_a1.v3j_b.fold_assignments(full_names, args.fold_count)
    fold_by_name = dict(zip(full_names, full_folds.tolist()))
    reference = v3m_a1.load_fixed_reference(args.reference_oof_rows)
    expected_reference = {(operator, name) for operator in args.operator_labels for name in names}
    if not expected_reference.issubset(reference):
        raise ValueError("fixed-alpha reference table does not cover requested OOF rows")

    device = torch.device(args.device)
    base = build_model("original", args.a0_checkpoint, device)
    action_model = build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = build_gate_producer(args, device)
    bound = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    image_rows = []
    by_operator = defaultdict(new_gain_accumulator)
    by_fold = defaultdict(new_gain_accumulator)
    integrity_rows = []
    raw_fields = ["operator_label", "seed", "fold", "index", "name", "clean_reference_group", "block_y", "block_x", "candidate_pixel_count", "oracle_alpha", "direct_step_energy", "d7c_score_mean"]
    for alpha in ACTION_LADDER:
        key = alpha_key(alpha)
        raw_fields.extend((f"candidate_sse_{key}", f"candidate_sse_e_{key}", f"clip_fraction_{key}"))
    raw_fields.extend(("gain_0125_to_025", "gain_025_to_05", "gain_05_to_10", "gain_0125_to_025_epsilon", "gain_0125_to_025_state"))

    with outputs["blocks"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, lineterminator="\n")
        writer.writeheader()
        for artifact_info in artifacts:
            operator = artifact_info["operator_label"]
            if operator not in args.operator_labels:
                continue
            artifact = torch.load(artifact_info["artifact_path"], map_location=device)
            if int(artifact.get("seed")) != int(artifact_info["seed"]):
                raise RuntimeError(f"operator artifact seed mismatch for {operator}")
            cache = v3l_a0.build_model_cache(artifact, args, device)
            max_fixed_replay_diff = 0.0
            max_normalized_error = 0.0
            non_gray_sign_flips = 0
            block_count = 0
            coverage_shapes = {}
            for index, name in enumerate(names):
                input_img, label = load_pair(args.data_dir, args.source_split, name)
                input_img = input_img.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                padded, height, width = pad_to_factor(input_img)
                label = label[:, :, :height, :width]
                fold = int(fold_by_name[name])
                action_shape = action_shape_for_input(padded)
                with torch.no_grad():
                    base_pred = v3l_a1.forward_final(base, padded, height, width)
                    _, base_psnr = metric_pair(base_pred, label)
                    gate_full, score_full, _ = gate_producer(padded)
                    hard_gate = action_gate_from_full(gate_full, action_shape).to(device)
                    fmap, _, _, _ = full_context_maps(action_model, gate_producer, padded)
                    model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                    pred_low = v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound)
                    output_gate = output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                    output_step = output_gate * F.interpolate(pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False)
                    blocks, direct, coverage, gpu_block_sse = candidate_block_rows(base_pred, output_step, score_full, label, args.block_size, args)
                    total_pixels = sum(int(row["candidate_pixel_count"]) for row in blocks)
                    if total_pixels != int(label.numel()):
                        raise RuntimeError("block RGB element count does not match image label")
                    coverage_shapes[f"{coverage['height']}x{coverage['width']}"] = coverage
                    image_row = {"operator_label": operator, "seed": int(artifact_info["seed"]), "fold": fold, "index": index, "name": name, "clean_reference_group": clean_reference_group(name), "base_psnr": float(base_psnr), "candidate_pixel_count": total_pixels}
                    for alpha in ACTION_LADDER:
                        key = alpha_key(alpha)
                        canonical_sse = direct[alpha]["blocks_sse"]
                        reference_sse = direct[alpha]["reference_sse"]
                        gpu_direct_sse = direct[alpha]["gpu_sse"]
                        gpu_blocks_total = direct[alpha]["gpu_blocks_sse"]
                        e_values = {
                            "canonical_blocks": normalized_error(canonical_sse, reference_sse, args),
                            "gpu_direct": normalized_error(gpu_direct_sse, reference_sse, args),
                            "gpu_blocks": normalized_error(gpu_blocks_total, reference_sse, args),
                            "gpu_block_max": direct[alpha]["block_e_max"],
                        }
                        max_normalized_error = max(max_normalized_error, *e_values.values())
                        image_row.update({
                            f"candidate_sse_{key}": canonical_sse,
                            f"candidate_sse_reference_{key}": reference_sse,
                            f"candidate_sse_gpu_direct_{key}": gpu_direct_sse,
                            f"candidate_sse_gpu_blocks_{key}": gpu_blocks_total,
                            f"candidate_e_max_{key}": max(e_values.values()),
                            f"candidate_mse_{key}": canonical_sse / total_pixels,
                        })
                    fixed_mse = image_row["candidate_mse_0p125"]
                    fixed_psnr = 10.0 * math.log10(1.0 / max(fixed_mse, 1e-30))
                    fixed_delta = fixed_psnr - float(base_psnr)
                    reference_delta = reference[(operator, name)]
                    fixed_diff = abs(fixed_delta - reference_delta)
                    max_fixed_replay_diff = max(max_fixed_replay_diff, fixed_diff)
                    if fixed_diff > args.replay_tolerance_db:
                        raise RuntimeError(f"fixed-alpha replay mismatch for {operator}/{name}")
                    image_row["fixed_psnr_delta"] = fixed_delta
                    image_row["reference_fixed_psnr_delta"] = reference_delta
                    image_row["fixed_replay_abs_diff_db"] = fixed_diff
                for block_index, row in enumerate(blocks):
                    gpu_gain = gpu_block_sse[0.125][block_index] - gpu_block_sse[0.25][block_index]
                    gpu_state = sign_state(gpu_gain, row["gain_0125_to_025_epsilon"])
                    if row["gain_0125_to_025_state"] != "abstain" and gpu_state != "abstain" and gpu_state != row["gain_0125_to_025_state"]:
                        non_gray_sign_flips += 1
                    writer.writerow({"operator_label": operator, "seed": int(artifact_info["seed"]), "fold": fold, "index": index, "name": name, "clean_reference_group": clean_reference_group(name), **row})
                    add_gain(by_operator[operator], row)
                    add_gain(by_fold[(operator, fold)], row)
                    block_count += 1
                image_rows.append(image_row)
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
                del input_img, label, padded, base_pred, hard_gate, fmap, output_step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            integrity_rows.append({
                "operator_label": operator,
                "seed": int(artifact_info["seed"]),
                "image_count": len(names),
                "block_count": block_count,
                "coverage_shapes": coverage_shapes,
                "fixed_alpha_max_abs_psnr_delta_diff_db": max_fixed_replay_diff,
                "max_normalized_numeric_error": max_normalized_error,
                "non_gray_signed_gain_flips": non_gray_sign_flips,
                "fixed_replay_pass": max_fixed_replay_diff <= args.replay_tolerance_db,
                "numerical_envelope_pass": max_normalized_error <= 1.0,
                "signed_gain_stability_pass": non_gray_sign_flips == 0,
            })

    gain_rows = [gain_row(operator, "ALL", by_operator[operator]) for operator in args.operator_labels]
    fold_rows = [gain_row(operator, fold, by_fold[(operator, fold)]) for operator in args.operator_labels for fold in range(args.fold_count)]
    agreement_rows = []
    for field in ("candidate_sse_0p125", "candidate_sse_0p25", "candidate_sse_0p5", "candidate_sse_1", "fixed_psnr_delta"):
        value, count = correlation(image_rows, field)
        agreement_rows.append({"field": field, "paired_image_count": count, "pearson_ref_rep": value})
    hard_fail = any(not row["fixed_replay_pass"] or not row["signed_gain_stability_pass"] for row in integrity_rows)
    numerical_pass = all(row["numerical_envelope_pass"] for row in integrity_rows)
    if hard_fail:
        decision = "V3P_A0_CANONICAL_NUMERICAL_HARD_FAIL_STOP"
        state = "COMPLETED_GATE_FAIL"
        authorizes = "none"
    elif not numerical_pass:
        decision = "V3P_A0_CANONICAL_NUMERICAL_INCONCLUSIVE_REPAIR_ONLY"
        state = "COMPLETED_INCONCLUSIVE"
        authorizes = "same-stage numerical repair only"
    elif args.run_mode == "smoke":
        decision = "V3P_A0_SMOKE_CANONICAL_NUMERICAL_PASS_AUTHORIZE_FORMAL_OOF_ONLY"
        state = "COMPLETED_GATE_PASS"
        authorizes = "v3p-A0 formal 1200-image OOF canonical reconstruction only"
    else:
        decision = "V3P_A0_CANONICAL_NUMERICAL_PASS_AUTHORIZE_A1_RECONSTRUCTION_ONLY"
        state = "COMPLETED_GATE_PASS"
        authorizes = "v3p-A1 frozen reconstruction and G1 decomposition only"
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "parent_evidence_main_commit": args.parent_evidence_main_commit,
        "runnable_source_commit": args.runnable_source_commit,
        "input_sha256": input_hashes,
        "script_sha256": sha256_file(__file__),
        "numerical_contract_sha256": sha256_file(args.numerical_contract),
        "raw_block_candidate_losses_cloud_only": str(outputs["blocks"]),
        "raw_image_candidate_replay_cloud_only": str(outputs["images"]),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": f"v3p-A0-{args.run_mode}",
        "state": state,
        "gate_type": "numerical_equivalence",
        "decision": decision,
        "metric_contract": "route card canonical SSE section",
        "authorizes": authorizes,
        "reason": "canonical block SSE, full-image high-precision reference, GPU float64 cross-check, coverage audit, and non-gray signed-gain stability",
        "integrity_rows": integrity_rows,
        "atol_sse": args.atol_sse,
        "rtol": args.rtol,
        "fixed_alpha_replay_tolerance_db": args.replay_tolerance_db,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    summary = {**closeout, "phase": "v3p-A0 canonical signed-gain numerical reconstruction", "image_count_per_operator": len(names), "gain_rows": gain_rows, "agreement_rows": agreement_rows, "source_manifest": source_manifest}
    write_rows(outputs["images"], image_rows)
    write_rows(outputs["gains"], gain_rows)
    write_rows(outputs["gain_folds"], fold_rows)
    write_rows(outputs["agreement"], agreement_rows)
    write_json(outputs["integrity"], closeout)
    write_json(outputs["summary"], summary)
    write_json(outputs["source"], source_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--a0_closeout", required=True)
    parser.add_argument("--operator_artifact_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--reference_oof_rows", required=True)
    parser.add_argument("--v3m_a0_source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--run_mode", required=True)
    parser.add_argument("--expected_fresh_split_manifest_sha256", required=True)
    parser.add_argument("--expected_parent_a0_closeout_sha256", required=True)
    parser.add_argument("--expected_parent_operator_manifest_sha256", required=True)
    parser.add_argument("--expected_reference_oof_rows_sha256", required=True)
    parser.add_argument("--expected_v3m_a0_source_manifest_sha256", required=True)
    parser.add_argument("--expected_density_artifact_sha256", required=True)
    parser.add_argument("--expected_d7c_artifact_sha256", required=True)
    parser.add_argument("--expected_a0_checkpoint_sha256", required=True)
    parser.add_argument("--expected_control_checkpoint_sha256", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--confirm_key", default="v3j_route_confirm")
    parser.add_argument("--max_train_samples", type=int, required=True)
    parser.add_argument("--smoke_sample_count", type=int, default=32)
    parser.add_argument("--formal_sample_count", type=int, default=1200)
    parser.add_argument("--operator_labels", nargs="+", default=list(OPERATORS))
    parser.add_argument("--common_alphas", type=float, nargs="+", default=list(ACTION_LADDER))
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, default=D7C_THRESHOLD)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--numerical_contract")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent_evidence_main_commit", required=True)
    parser.add_argument("--runnable_source_commit", required=True)
    args = parser.parse_args()
    if sorted(args.operator_labels) != sorted(OPERATORS):
        raise ValueError("v3p-A0 requires exactly D_ref and D_rep")
    run(args)


if __name__ == "__main__":
    main()
