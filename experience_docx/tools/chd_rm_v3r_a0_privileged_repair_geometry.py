#!/usr/bin/env python3
"""v3r-A0 privileged, rendered first-step repair geometry audit.

The audit reconstructs the frozen v3p operators and compares only fixed,
privileged residual repair families.  It does not fit a model or replay a
deployable policy.  All candidate losses include the real clamp operation.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v3r_signed_margin_operator_repair_20260712"
OPERATORS = ("D_ref", "D_rep")
FIXED_ALPHAS = (0.125, 0.25, 0.5)
BLOCK_SIZE = 16
GRID_STEPS = 64
SESOI_DB = 0.005
SEVERE_DB = -0.2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def psnr_from_sse(sse, element_count):
    mse = sse / max(float(element_count), 1.0)
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def clean_reference_group(name):
    return Path(name).stem.split("_", 1)[0]


def import_parent_modules(v3p_source_root):
    tools_dir = Path(v3p_source_root) / "experience_docx" / "tools"
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"missing pinned v3p tool directory: {tools_dir}")
    sys.path.insert(0, str(tools_dir))
    return {
        "v3i_a": importlib.import_module("chd_rm_v3i_a_teacher_compressibility_audit"),
        "v3i_b": importlib.import_module("chd_rm_v3i_b_full_context_probe"),
        "v3j_a": importlib.import_module("chd_rm_v3j_a_bounded_action_audit"),
        "v3l_a0": importlib.import_module("chd_rm_v3l_a0_canonical_operator"),
        "v3l_a1": importlib.import_module("chd_rm_v3l_a1_oracle_granularity_audit"),
        "v3m_a1": importlib.import_module("chd_rm_v3m_a1_local_actuation_audit"),
    }


def block_sum(values, block_size):
    """Return exact RGB SSE sums as [1, block_y, block_x] for full blocks."""
    _, _, height, width = values.shape
    if height % block_size or width % block_size:
        raise RuntimeError(f"v3r requires full block16 geometry, got {height}x{width}")
    unfolded = F.unfold(values, kernel_size=block_size, stride=block_size)
    return unfolded.sum(dim=1).reshape(1, height // block_size, width // block_size)


def block_channel_sum(values, block_size):
    """Return per-channel sums as [1, 3, block_y, block_x]."""
    _, channels, height, width = values.shape
    if channels != 3 or height % block_size or width % block_size:
        raise RuntimeError("v3r channel block geometry mismatch")
    unfolded = F.unfold(values, kernel_size=block_size, stride=block_size)
    unfolded = unfolded.reshape(1, channels, block_size * block_size, -1)
    return unfolded.sum(dim=2).reshape(1, channels, height // block_size, width // block_size)


def expand_block_map(values, height, width, block_size):
    return values.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)[..., :height, :width]


def losses_for_step(y0, residual, label, block_size):
    result = {}
    for alpha in FIXED_ALPHAS:
        raw = y0 + alpha * residual
        prediction = torch.clamp(raw, 0.0, 1.0)
        error = (prediction.to(torch.float64) - label.to(torch.float64)).square()
        result[alpha] = {
            "raw": raw,
            "prediction": prediction,
            "block": block_sum(error, block_size),
            "channel_block": block_channel_sum(error, block_size),
            "total": float(torch.sum(error, dtype=torch.float64).item()),
            "clip_pixel": ((raw < 0.0) | (raw > 1.0)).any(dim=1, keepdim=True).to(torch.float64),
        }
    return result


def epsilon_sse(loss, atol, rtol):
    return atol + rtol * loss.abs()


def step_gain(losses):
    gain = losses[0.125]["block"] - losses[0.25]["block"]
    epsilon = epsilon_sse(losses[0.125]["block"], losses["atol"], losses["rtol"])
    epsilon = epsilon + epsilon_sse(losses[0.25]["block"], losses["atol"], losses["rtol"])
    return gain, epsilon


def attach_tolerance(losses, atol, rtol):
    losses["atol"] = atol
    losses["rtol"] = rtol
    return losses


def select_scale(y0, residual, label, old_losses, active, block_size, grid):
    old_l125 = old_losses[0.125]["block"]
    best_gain = torch.full_like(old_l125, float("-inf"))
    best_value = torch.ones_like(old_l125)
    first_value = torch.full_like(old_l125, float("nan"))
    first_seen = torch.zeros_like(active, dtype=torch.bool)
    for value in grid:
        candidate = residual * value
        losses = attach_tolerance(losses_for_step(y0, candidate, label, block_size), old_losses["atol"], old_losses["rtol"])
        gain, epsilon = step_gain(losses)
        feasible = active & (losses[0.125]["block"] <= old_l125 + epsilon) & (gain > epsilon)
        update = feasible & (gain > best_gain)
        best_gain = torch.where(update, gain, best_gain)
        best_value = torch.where(update, torch.full_like(best_value, float(value)), best_value)
        first_update = feasible & ~first_seen
        first_value = torch.where(first_update, torch.full_like(first_value, float(value)), first_value)
        first_seen |= feasible
    height, width = residual.shape[-2:]
    selected = expand_block_map(best_value, height, width, block_size).to(residual.dtype)
    repaired = residual * selected
    repaired = torch.where(expand_block_map(first_seen.to(residual.dtype), height, width, block_size) > 0, repaired, residual)
    return {
        "residual": repaired,
        "repairable": first_seen,
        "best_parameter": best_value,
        "minimum_parameter": first_value,
    }


def select_channel_scale(y0, residual, label, old_losses, active, block_size, grid):
    old_l125 = old_losses[0.125]["channel_block"]
    best_gain = torch.full_like(old_l125, float("-inf"))
    best_value = torch.ones_like(old_l125)
    for value in grid:
        candidate = residual * value
        losses = attach_tolerance(losses_for_step(y0, candidate, label, block_size), old_losses["atol"], old_losses["rtol"])
        gain = losses[0.125]["channel_block"] - losses[0.25]["channel_block"]
        epsilon = epsilon_sse(losses[0.125]["channel_block"], old_losses["atol"], old_losses["rtol"])
        epsilon = epsilon + epsilon_sse(losses[0.25]["channel_block"], old_losses["atol"], old_losses["rtol"])
        feasible = active.unsqueeze(1) & (losses[0.125]["channel_block"] <= old_l125 + epsilon)
        update = feasible & (gain > best_gain)
        best_gain = torch.where(update, gain, best_gain)
        best_value = torch.where(update, torch.full_like(best_value, float(value)), best_value)
    height, width = residual.shape[-2:]
    selected = expand_block_map(best_value, height, width, block_size).to(residual.dtype)
    candidate = residual * selected
    losses = attach_tolerance(losses_for_step(y0, candidate, label, block_size), old_losses["atol"], old_losses["rtol"])
    gain, epsilon = step_gain(losses)
    repairable = active & (losses[0.125]["block"] <= old_losses[0.125]["block"] + epsilon) & (gain > epsilon)
    repaired = torch.where(expand_block_map(repairable.to(residual.dtype), height, width, block_size) > 0, candidate, residual)
    return {
        "residual": repaired,
        "repairable": repairable,
        "best_parameter": best_value,
        "minimum_parameter": torch.full_like(gain, float("nan")),
    }


def select_direction_line(y0, residual, target_residual, label, old_losses, active, block_size, grid):
    old_l125 = old_losses[0.125]["block"]
    best_gain = torch.full_like(old_l125, float("-inf"))
    best_value = torch.ones_like(old_l125)
    first_value = torch.full_like(old_l125, float("nan"))
    first_seen = torch.zeros_like(active, dtype=torch.bool)
    for value in grid:
        candidate = (1.0 - value) * residual + value * target_residual
        losses = attach_tolerance(losses_for_step(y0, candidate, label, block_size), old_losses["atol"], old_losses["rtol"])
        gain, epsilon = step_gain(losses)
        feasible = active & (losses[0.125]["block"] <= old_l125 + epsilon) & (gain > epsilon)
        update = feasible & (gain > best_gain)
        best_gain = torch.where(update, gain, best_gain)
        best_value = torch.where(update, torch.full_like(best_value, float(value)), best_value)
        first_update = feasible & ~first_seen
        first_value = torch.where(first_update, torch.full_like(first_value, float(value)), first_value)
        first_seen |= feasible
    height, width = residual.shape[-2:]
    selected = expand_block_map(best_value, height, width, block_size).to(residual.dtype)
    candidate = (1.0 - selected) * residual + selected * target_residual
    repaired = torch.where(expand_block_map(first_seen.to(residual.dtype), height, width, block_size) > 0, candidate, residual)
    return {
        "residual": repaired,
        "repairable": first_seen,
        "best_parameter": best_value,
        "minimum_parameter": first_value,
    }


def select_direct_clean(y0, residual, target_residual, label, old_losses, active, block_size):
    candidate_losses = attach_tolerance(losses_for_step(y0, target_residual, label, block_size), old_losses["atol"], old_losses["rtol"])
    gain, epsilon = step_gain(candidate_losses)
    feasible = active & (candidate_losses[0.125]["block"] <= old_losses[0.125]["block"] + epsilon) & (gain > epsilon)
    height, width = residual.shape[-2:]
    repaired = torch.where(expand_block_map(feasible.to(residual.dtype), height, width, block_size) > 0, target_residual, residual)
    return {
        "residual": repaired,
        "repairable": feasible,
        "best_parameter": torch.where(feasible, torch.ones_like(gain), torch.zeros_like(gain)),
        "minimum_parameter": torch.where(feasible, torch.ones_like(gain), torch.full_like(gain, float("nan"))),
    }


def quantile_summary(values):
    if not values:
        return {"count": 0, "median": float("nan"), "p90": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9))}


def bootstrap_lcb95(rows, value_key, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["clean_reference_group"]].append(float(row[value_key]))
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if values.size < 2:
        return float("nan"), int(values.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(4000, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), int(values.size)


def cvar5(values):
    array = np.sort(np.asarray(values, dtype=np.float64))
    if not array.size:
        return float("nan")
    return float(array[: max(1, int(math.ceil(0.05 * array.size)))].mean())


def block_categories(old_losses, residual, label, active):
    gain1, epsilon1 = step_gain(old_losses)
    gain2 = old_losses[0.25]["block"] - old_losses[0.5]["block"]
    epsilon2 = epsilon_sse(old_losses[0.25]["block"], old_losses["atol"], old_losses["rtol"])
    epsilon2 = epsilon2 + epsilon_sse(old_losses[0.5]["block"], old_losses["atol"], old_losses["rtol"])
    dot = block_sum((label - old_losses["y0"]) * residual, BLOCK_SIZE)
    harmful = active & (gain1 < -epsilon1)
    wrong = harmful & (dot <= 0.0)
    overshoot = harmful & ~wrong
    oversized = active & (gain1 > epsilon1) & (gain2 < -epsilon2)
    conservative = ~(wrong | overshoot | oversized)
    return {
        "wrong_direction": wrong,
        "harmful_overshoot": overshoot,
        "beneficial_but_oversized": oversized,
        "conservative_or_ok": conservative,
    }


def source_match(source_row, operator, name, block_y, block_x, old_losses, old_gain, old_epsilon, index):
    required = {
        "operator_label": operator,
        "index": str(index),
        "name": name,
        "block_y": str(block_y),
        "block_x": str(block_x),
    }
    for key, expected in required.items():
        if source_row.get(key) != expected:
            raise RuntimeError(f"source identity mismatch key={key} expected={expected} observed={source_row.get(key)}")
    source_gain = float(source_row["gain_0125_to_025"])
    source_epsilon = float(source_row["gain_0125_to_025_epsilon"])
    observed = float(old_gain)
    if abs(source_gain - observed) > max(source_epsilon, float(old_epsilon)):
        raise RuntimeError(f"source gain mismatch {operator}/{name}/{block_y}/{block_x}")
    source_state = source_row["gain_0125_to_025_state"]
    state = "beneficial" if observed > float(old_epsilon) else "harmful" if observed < -float(old_epsilon) else "abstain"
    if source_state != state:
        raise RuntimeError(f"source signed state mismatch {operator}/{name}/{block_y}/{block_x}")


def evaluate_type(name, selection, y0, label, old_losses, block_size, bound, active):
    repaired_losses = attach_tolerance(losses_for_step(y0, selection["residual"], label, block_size), old_losses["atol"], old_losses["rtol"])
    gain, epsilon = step_gain(repaired_losses)
    delta = selection["residual"] - old_losses["residual"]
    old_norm = torch.sqrt(block_sum(old_losses["residual"].square(), block_size).clamp_min(1e-30))
    new_norm = torch.sqrt(block_sum(selection["residual"].square(), block_size).clamp_min(1e-30))
    dot = block_sum(old_losses["residual"] * selection["residual"], block_size)
    cosine = torch.clamp(dot / (old_norm * new_norm).clamp_min(1e-30), -1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    relative_norm = torch.sqrt(block_sum(delta.square(), block_size)) / old_norm
    changed = active & (relative_norm > 1e-12)
    active_pixels = expand_block_map(active.to(torch.float64), y0.shape[-2], y0.shape[-1], block_size)
    bound_view = bound.to(y0.device, y0.dtype).view(1, 3, 1, 1)
    saturation = ((selection["residual"].abs() >= 0.999 * bound_view).to(torch.float64) * active_pixels).sum()
    saturation_denominator = (active_pixels.sum() * 3.0).clamp_min(1.0)
    clip = (repaired_losses[0.25]["clip_pixel"] * active_pixels).sum() / active_pixels.sum().clamp_min(1.0)
    return {
        "name": name,
        "losses": repaired_losses,
        "gain": gain,
        "epsilon": epsilon,
        "angle": angle,
        "relative_norm": relative_norm,
        "changed": changed,
        "bound_saturation": float((saturation / saturation_denominator).item()),
        "clip_fraction": float(clip.item()),
        "repairable": selection["repairable"],
        "best_parameter": selection["best_parameter"],
        "minimum_parameter": selection["minimum_parameter"],
    }


def summarize_image(result, old_losses, element_count):
    repaired = result["losses"]
    old_125 = psnr_from_sse(old_losses[0.125]["total"], element_count)
    old_25 = psnr_from_sse(old_losses[0.25]["total"], element_count)
    new_125 = psnr_from_sse(repaired[0.125]["total"], element_count)
    new_25 = psnr_from_sse(repaired[0.25]["total"], element_count)
    gain = result["gain"]
    active = result["repairable"].new_zeros(result["repairable"].shape, dtype=torch.bool)
    active[:] = True
    positive = torch.clamp(gain, min=0.0).sum().item()
    harmful = torch.clamp(-gain, min=0.0).sum().item()
    return {
        "old_psnr_0125": old_125,
        "old_psnr_025": old_25,
        "new_psnr_0125": new_125,
        "new_psnr_025": new_25,
        "delta_025_vs_old_0125": new_25 - old_125,
        "delta_025_vs_old_025": new_25 - old_25,
        "delta_0125_vs_old_0125": new_125 - old_125,
        "beneficial_sse": float(positive),
        "harmful_sse": float(harmful),
        "harmful_to_beneficial": float(harmful / positive) if positive > 0 else float("nan"),
    }


def run(args):
    is_smoke = args.run_mode.startswith("smoke")
    if not is_smoke and args.run_mode != "formal":
        raise ValueError("run_mode must be smoke* or formal")
    if args.block_size != BLOCK_SIZE or args.grid_steps != GRID_STEPS:
        raise ValueError("v3r-A0 requires fixed block16 and 65-point grids")
    if sorted(args.operator_labels) != sorted(OPERATORS):
        raise ValueError("v3r-A0 requires exactly D_ref and D_rep")
    if sha256_file(args.canonical_blocks) != args.expected_canonical_blocks_sha256:
        raise RuntimeError("canonical source SHA-256 mismatch")
    if sha256_file(args.v3p_source_root + "/experience_docx/tools/chd_rm_v3p_a0_canonical_signed_gain.py") != args.expected_v3p_a0_script_sha256:
        raise RuntimeError("pinned v3p candidate producer script SHA-256 mismatch")
    if not Path(args.v3p_source_root).is_dir():
        raise FileNotFoundError(args.v3p_source_root)
    source_head = subprocess.check_output(
        ["git", "-C", args.v3p_source_root, "rev-parse", "HEAD"], text=True
    ).strip()
    if source_head != args.expected_v3p_source_commit:
        raise RuntimeError(f"pinned v3p source commit mismatch: {source_head}")

    expected_images = args.smoke_sample_count if is_smoke else args.formal_sample_count
    if args.max_train_samples != expected_images:
        raise ValueError("run mode and image count are inconsistent")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_blocks": output_dir / f"{args.run_tag}_repair_rows_cloud_only.csv",
        "raw_images": output_dir / f"{args.run_tag}_repair_images_cloud_only.csv",
        "summary": output_dir / f"{args.run_tag}_repair_type_summary.csv",
        "categories": output_dir / f"{args.run_tag}_category_summary.csv",
        "source": output_dir / f"{args.run_tag}_source_manifest.json",
        "closeout": output_dir / f"{args.run_tag}_closeout.json",
        "decision": output_dir / f"{args.run_tag}_dual_operator_decision.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite v3r-A0 outputs: " + ", ".join(existing))

    modules = import_parent_modules(args.v3p_source_root)
    v3i_a = modules["v3i_a"]
    v3i_b = modules["v3i_b"]
    v3j_a = modules["v3j_a"]
    v3l_a0 = modules["v3l_a0"]
    v3l_a1 = modules["v3l_a1"]
    v3m_a1 = modules["v3m_a1"]

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
    full_names = v3j_a.names_from_manifest(manifest, args.train_key, args.formal_sample_count)
    names = full_names[:args.max_train_samples]
    if len(full_names) != args.formal_sample_count or len(names) != expected_images:
        raise RuntimeError("frozen OOF names do not match v3r-A0 contract")
    folds, _ = v3l_a1.v3j_b.fold_assignments(full_names, args.fold_count)
    fold_by_name = dict(zip(full_names, folds.tolist()))
    reference = v3m_a1.load_fixed_reference(args.reference_oof_rows)
    if not {(operator, name) for operator in OPERATORS for name in names}.issubset(reference):
        raise RuntimeError("fixed reference does not cover v3r rows")

    device = torch.device(args.device)
    base = v3i_a.build_model("original", args.a0_checkpoint, device)
    action_model = v3i_a.build_model("fam2_d7c_noop", args.control_checkpoint, device)
    gate_producer = v3i_a.build_gate_producer(args, device)
    bound_values = read_json(args.v3j_a_bounds)["channel_bounds_rgb"]
    bound = torch.tensor(bound_values, dtype=torch.float32, device=device)
    grid = torch.linspace(0.0, 1.0, args.grid_steps + 1, dtype=torch.float32, device=device)
    numerical = read_json(args.v3p_numerical_contract)
    atol = float(numerical["atol_sse"])
    rtol = float(numerical["rtol"])
    image_rows = []
    category_rows = []
    rotation_values = defaultdict(list)
    source_count = 0
    source_reader = None
    source_by_key = None
    if is_smoke:
        selected_names = set(names)
        source_by_key = {}
        with Path(args.canonical_blocks).open(newline="", encoding="utf-8") as source_handle:
            for row in csv.DictReader(source_handle):
                if row["name"] in selected_names:
                    key = (row["operator_label"], row["name"], int(row["block_y"]), int(row["block_x"]))
                    if key in source_by_key:
                        raise RuntimeError(f"duplicate smoke source key: {key}")
                    source_by_key[key] = row
        if len(source_by_key) != args.expected_source_rows:
            raise RuntimeError("smoke source key count does not match the frozen contract")
    else:
        source_reader = csv.DictReader(Path(args.canonical_blocks).open(newline="", encoding="utf-8"))

    raw_block_fields = [
        "operator_label", "fold", "index", "name", "clean_reference_group", "block_y", "block_x", "category",
        "old_gain", "old_epsilon", "scale_repairable", "scale_lambda", "channel_repairable",
        "channel_lambda_r", "channel_lambda_g", "channel_lambda_b", "direction_repairable",
        "direction_gamma_best", "direction_gamma_minimum", "direct_clean_repairable",
    ]
    with outputs["raw_blocks"].open("w", newline="", encoding="utf-8") as block_handle:
        block_writer = csv.DictWriter(block_handle, fieldnames=raw_block_fields, lineterminator="\n")
        block_writer.writeheader()
        artifact_by_operator = {item["operator_label"]: item for item in artifacts}
        for operator in OPERATORS:
            if operator not in artifact_by_operator:
                raise RuntimeError(f"missing pinned operator artifact: {operator}")
            artifact_info = artifact_by_operator[operator]
            artifact = torch.load(artifact_info["artifact_path"], map_location=device)
            cache = v3l_a0.build_model_cache(artifact, args, device)
            for index, name in enumerate(names):
                input_img, label = v3i_a.load_pair(args.data_dir, args.source_split, name)
                input_img = input_img.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                padded, height, width = v3i_a.pad_to_factor(input_img)
                label = label[:, :, :height, :width]
                fold = int(fold_by_name[name])
                with torch.no_grad():
                    base_pred = v3l_a1.forward_final(base, padded, height, width)
                    _, base_psnr = v3i_a.metric_pair(base_pred, label)
                    gate_full, score_full, _ = gate_producer(padded)
                    action_shape = v3i_a.action_shape_for_input(padded)
                    hard_gate = v3i_a.action_gate_from_full(gate_full, action_shape).to(device)
                    fmap, _, _, _ = v3i_b.full_context_maps(action_model, gate_producer, padded)
                    model, mean, std = v3l_a0.model_pack_from_cache(cache, "OOF", fold)
                    pred_low = v3l_a1.v3j_b.score_map("context", model, fmap, mean, std, bound_values)
                    output_gate = v3j_a.output_gate_from_action_gate(hard_gate, base_pred.shape[-2:])
                    residual = output_gate * F.interpolate(pred_low, size=base_pred.shape[-2:], mode="bilinear", align_corners=False)
                    y0 = base_pred
                    old_losses = attach_tolerance(losses_for_step(y0, residual, label, args.block_size), atol, rtol)
                    old_losses["residual"] = residual
                    old_losses["y0"] = y0
                    old_gain, old_epsilon = step_gain(old_losses)
                    active = block_sum(residual.square(), args.block_size) > 0.0
                    support = expand_block_map(active.to(residual.dtype), height, width, args.block_size)
                    bound_view = bound.view(1, 3, 1, 1)
                    target_residual = torch.maximum(
                        torch.minimum(4.0 * (label - y0), bound_view), -bound_view
                    ) * support
                    selections = {
                        "scale": select_scale(y0, residual, label, old_losses, active, args.block_size, grid),
                        "channel_scale": select_channel_scale(y0, residual, label, old_losses, active, args.block_size, grid),
                        "direction_line": select_direction_line(y0, residual, target_residual, label, old_losses, active, args.block_size, grid),
                        "direct_clean": select_direct_clean(y0, residual, target_residual, label, old_losses, active, args.block_size),
                    }
                    evaluated = {
                        key: evaluate_type(key, value, y0, label, old_losses, args.block_size, bound, active)
                        for key, value in selections.items()
                    }
                    categories = block_categories(old_losses, residual, label, active)
                    element_count = int(label.numel())
                    fixed_delta = psnr_from_sse(old_losses[0.125]["total"], element_count) - float(base_psnr)
                    if abs(fixed_delta - float(reference[(operator, name)])) > args.replay_tolerance_db:
                        raise RuntimeError(f"fixed-alpha replay mismatch for {operator}/{name}")
                    for block_y in range(old_gain.shape[-2]):
                        for block_x in range(old_gain.shape[-1]):
                            if source_by_key is not None:
                                key = (operator, name, block_y, block_x)
                                if key not in source_by_key:
                                    raise RuntimeError(f"smoke source key missing: {key}")
                                source_row = source_by_key[key]
                            else:
                                try:
                                    source_row = next(source_reader)
                                except StopIteration as exc:
                                    raise RuntimeError("canonical source ended early") from exc
                            source_match(
                                source_row, operator, name, block_y, block_x, old_losses,
                                old_gain[0, block_y, block_x], old_epsilon[0, block_y, block_x], index,
                            )
                            source_count += 1
                            category = next(key for key, mask in categories.items() if bool(mask[0, block_y, block_x]))
                            if bool(active[0, block_y, block_x]):
                                channel_value = evaluated["channel_scale"]["best_parameter"][0, :, block_y, block_x]
                                block_writer.writerow({
                                    "operator_label": operator,
                                    "fold": fold,
                                    "index": index,
                                    "name": name,
                                    "clean_reference_group": clean_reference_group(name),
                                    "block_y": block_y,
                                    "block_x": block_x,
                                    "category": category,
                                    "old_gain": float(old_gain[0, block_y, block_x]),
                                    "old_epsilon": float(old_epsilon[0, block_y, block_x]),
                                    "scale_repairable": int(evaluated["scale"]["repairable"][0, block_y, block_x]),
                                    "scale_lambda": float(evaluated["scale"]["best_parameter"][0, block_y, block_x]),
                                    "channel_repairable": int(evaluated["channel_scale"]["repairable"][0, block_y, block_x]),
                                    "channel_lambda_r": float(channel_value[0]),
                                    "channel_lambda_g": float(channel_value[1]),
                                    "channel_lambda_b": float(channel_value[2]),
                                    "direction_repairable": int(evaluated["direction_line"]["repairable"][0, block_y, block_x]),
                                    "direction_gamma_best": float(evaluated["direction_line"]["best_parameter"][0, block_y, block_x]),
                                    "direction_gamma_minimum": float(evaluated["direction_line"]["minimum_parameter"][0, block_y, block_x]),
                                    "direct_clean_repairable": int(evaluated["direct_clean"]["repairable"][0, block_y, block_x]),
                                })
                    for repair_name, result in evaluated.items():
                        record = {
                            "operator_label": operator,
                            "repair_type": repair_name,
                            "fold": fold,
                            "index": index,
                            "name": name,
                            "clean_reference_group": clean_reference_group(name),
                            **summarize_image(result, old_losses, element_count),
                            "repairable_active_blocks": int((result["repairable"] & active).sum().item()),
                            "active_blocks": int(active.sum().item()),
                            "bound_saturation": result["bound_saturation"],
                            "clip_fraction": result["clip_fraction"],
                        }
                        image_rows.append(record)
                        changed = result["changed"]
                        rotation_values[(operator, repair_name, "angle")].extend(result["angle"][changed].detach().cpu().tolist())
                        rotation_values[(operator, repair_name, "relative_norm")].extend(result["relative_norm"][changed].detach().cpu().tolist())
                        for category, mask in categories.items():
                            eligible = mask & active
                            category_rows.append({
                                "operator_label": operator,
                                "repair_type": repair_name,
                                "category": category,
                                "active_blocks": int(eligible.sum().item()),
                                "repairable_blocks": int((result["repairable"] & eligible).sum().item()),
                            })
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    print(f"{args.run_tag}_{operator}_{index + 1}/{len(names)}", flush=True)
                del input_img, label, padded, base_pred, gate_full, score_full, hard_gate, fmap, residual
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if source_reader is not None:
        try:
            next(source_reader)
            raise RuntimeError("canonical source contains unconsumed rows")
        except StopIteration:
            pass
    if source_count != args.expected_source_rows:
        raise RuntimeError(f"source row count mismatch: {source_count} vs {args.expected_source_rows}")

    with outputs["raw_images"].open("w", newline="", encoding="utf-8") as image_handle:
        writer = csv.DictWriter(image_handle, fieldnames=list(image_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(image_rows)

    summary_rows = []
    for operator in OPERATORS:
        for repair_name in ("scale", "channel_scale", "direction_line", "direct_clean"):
            rows = [row for row in image_rows if row["operator_label"] == operator and row["repair_type"] == repair_name]
            delta_25 = [row["delta_025_vs_old_025"] for row in rows]
            delta_125 = [row["delta_025_vs_old_0125"] for row in rows]
            lcb, groups = bootstrap_lcb95(rows, "delta_025_vs_old_025", args.seed)
            beneficial = sum(row["beneficial_sse"] for row in rows)
            harmful = sum(row["harmful_sse"] for row in rows)
            active_blocks = sum(row["active_blocks"] for row in rows)
            repairable = sum(row["repairable_active_blocks"] for row in rows)
            summary_rows.append({
                "operator_label": operator,
                "repair_type": repair_name,
                "image_count": len(rows),
                "group_count": groups,
                "mean_lift_vs_old_025_db": float(np.mean(delta_25)),
                "lcb95_lift_vs_old_025_db": lcb,
                "mean_lift_vs_old_0125_db": float(np.mean(delta_125)),
                "p10_vs_old_0125_db": float(np.quantile(delta_125, 0.1)),
                "p05_vs_old_0125_db": float(np.quantile(delta_125, 0.05)),
                "cvar5_vs_old_0125_db": cvar5(delta_125),
                "severe_vs_old_0125_count": int(sum(value <= SEVERE_DB for value in delta_125)),
                "active_block_count": active_blocks,
                "repairable_active_block_count": repairable,
                "repairable_active_fraction": repairable / active_blocks if active_blocks else float("nan"),
                "beneficial_sse": beneficial,
                "harmful_sse": harmful,
                "harmful_to_beneficial": harmful / beneficial if beneficial else float("nan"),
                "mean_bound_saturation": float(np.mean([row["bound_saturation"] for row in rows])),
                "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in rows])),
                "angle": quantile_summary(rotation_values[(operator, repair_name, "angle")]),
                "relative_norm": quantile_summary(rotation_values[(operator, repair_name, "relative_norm")]),
            })
    write_rows(outputs["summary"], summary_rows)

    grouped_categories = defaultdict(lambda: {"active_blocks": 0, "repairable_blocks": 0})
    for row in category_rows:
        key = (row["operator_label"], row["repair_type"], row["category"])
        grouped_categories[key]["active_blocks"] += row["active_blocks"]
        grouped_categories[key]["repairable_blocks"] += row["repairable_blocks"]
    compact_categories = []
    for (operator, repair_name, category), value in sorted(grouped_categories.items()):
        compact_categories.append({
            "operator_label": operator,
            "repair_type": repair_name,
            "category": category,
            **value,
            "repairable_fraction": value["repairable_blocks"] / value["active_blocks"] if value["active_blocks"] else float("nan"),
        })
    write_rows(outputs["categories"], compact_categories)

    by_type = {name: [row for row in summary_rows if row["repair_type"] == name] for name in ("scale", "channel_scale", "direction_line", "direct_clean")}
    dual = {}
    for name, rows in by_type.items():
        lcbs = {row["operator_label"]: row["lcb95_lift_vs_old_025_db"] for row in rows}
        dual[name] = {
            "lcb95_by_operator_db": lcbs,
            "worst_operator_lcb95_db": min(lcbs.values()),
            "passes_sesoi": all(value >= SESOI_DB for value in lcbs.values()),
        }
    if dual["scale"]["passes_sesoi"]:
        decision = "V3R_A0_SCALE_REPAIR_CEILING_PASS_AUTHORIZE_SCALE_ROUTE_DESIGN_ONLY"
        authorizes = "new scale-repair representation/training contract design only"
    elif dual["channel_scale"]["passes_sesoi"]:
        decision = "V3R_A0_CHANNEL_REPAIR_CEILING_PASS_AUTHORIZE_CHANNEL_REPAIR_ROUTE_DESIGN_ONLY"
        authorizes = "new channel-repair representation/training contract design only"
    elif dual["direction_line"]["passes_sesoi"]:
        decision = "V3R_A0_DIRECTION_REPAIR_CEILING_PASS_AUTHORIZE_DIRECTION_REPAIR_ROUTE_DESIGN_ONLY"
        authorizes = "new direction-repair representation/training contract design only"
    elif dual["direct_clean"]["passes_sesoi"]:
        decision = "V3R_A0_LOW_CAPACITY_REPAIR_FAIL_DIRECT_CLEAN_HEADROOM_REQUIRE_NEW_DIRECTION_REPRESENTATION"
        authorizes = "new residual-direction representation design only"
    else:
        decision = "V3R_A0_REPAIR_CEILING_FAIL_REDESIGN_ACTION_PARAMETERIZATION"
        authorizes = "none"
    state = "COMPLETED_GATE_PASS" if any(item["passes_sesoi"] for item in dual.values()) else "COMPLETED_GATE_FAIL"
    source_manifest = {
        "route_id": ROUTE_ID,
        "run_tag": args.run_tag,
        "run_mode": args.run_mode,
        "parent_evidence_main_commit": args.parent_evidence_main_commit,
        "runnable_source_commit": args.runnable_source_commit,
        "v3p_source_commit": args.expected_v3p_source_commit,
        "canonical_blocks": args.canonical_blocks,
        "canonical_blocks_sha256": sha256_file(args.canonical_blocks),
        "v3p_a0_script_sha256": sha256_file(args.v3p_source_root + "/experience_docx/tools/chd_rm_v3p_a0_canonical_signed_gain.py"),
        "input_sha256": input_hashes,
        "script_sha256": sha256_file(__file__),
        "source_rows_consumed": source_count,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": f"v3r-A0-{args.run_mode}",
        "state": state if not is_smoke else "COMPLETED_GATE_PASS",
        "gate_type": "structural_integrity" if is_smoke else "scientific_utility",
        "decision": "V3R_A0_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY" if is_smoke else decision,
        "authorizes": "v3r-A0 formal only" if is_smoke else authorizes,
        "metric_contract": "v3r route card A0 privileged rendered repair geometry",
        "reason": "pinned v3p reconstruction with anchor-preserving fixed-grid scale, channel, direction-line, and direct-clean ceilings",
        "source_rows_consumed": source_count,
        "expected_source_rows": args.expected_source_rows,
        "sesoi_db": SESOI_DB,
        "dual_operator": dual,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(outputs["source"], source_manifest)
    write_json(outputs["decision"], {"route_id": ROUTE_ID, "decision": closeout["decision"], "authorizes": closeout["authorizes"], "dual_operator": dual})
    write_json(outputs["closeout"], closeout)
    print(json.dumps(closeout, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3p_source_root", required=True)
    parser.add_argument("--canonical_blocks", required=True)
    parser.add_argument("--v3p_numerical_contract", required=True)
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
    parser.add_argument("--expected_canonical_blocks_sha256", required=True)
    parser.add_argument("--expected_v3p_a0_script_sha256", required=True)
    parser.add_argument("--expected_v3p_source_commit", required=True)
    parser.add_argument("--expected_source_rows", type=int, required=True)
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
    parser.add_argument("--common_alphas", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--grid_steps", type=int, default=GRID_STEPS)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, required=True)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent_evidence_main_commit", required=True)
    parser.add_argument("--runnable_source_commit", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
