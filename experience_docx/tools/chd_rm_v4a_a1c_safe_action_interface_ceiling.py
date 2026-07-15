#!/usr/bin/env python3
"""Privileged safe-action interface ceiling on the fixed A1R fresh512 screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715"
PARENT_REVIEW = {
    "route_id": "haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714",
    "state": "COMPLETED_R3_REVIEW",
    "decision": "V4A_A1R_PRIMARY_REPRESENTATION_SUFFICIENCY_FAIL_DIRECTIONAL_SIGNAL_BELOW_MATERIAL_UTILITY",
    "authorizes": "R3_AMENDMENT_DESIGN_ONLY",
}
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 3407
INTERFACE_FLOOR_DB = 0.080
RETENTION_FLOOR = 0.50
CLOSE_UCB_DB = 0.020
PARENT: Any = None
SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_parent_review(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("A1R R3 review hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in PARENT_REVIEW.items():
        if value.get(key) != expected:
            raise RuntimeError(f"A1R R3 review tuple mismatch: {key}={value.get(key)!r}")
    return value


def interface_endpoint(target: torch.Tensor, support: torch.Tensor, bound: torch.Tensor, interface: str) -> torch.Tensor:
    if interface == "full":
        value = target
    else:
        height, width = target.shape[-2:]
        low_size = ((height + 1) // 2, (width + 1) // 2)
        if interface == "exact_half":
            low = F.interpolate(target, size=low_size, mode="bilinear", align_corners=False)
        elif interface == "half_aa":
            low = F.interpolate(target, size=low_size, mode="bilinear", align_corners=False, antialias=True)
        else:
            raise ValueError(f"unknown interface: {interface}")
        value = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    return support * PARENT.clamp_channelwise(value, bound)


def select_direction(
    sample: dict[str, torch.Tensor], step: torch.Tensor, current: torch.Tensor, endpoint: torch.Tensor,
    shrink: dict[str, Any], old_low: float, old_high: float, family: str,
) -> tuple[dict[str, Any], int]:
    candidates = PARENT.grid_metrics(sample, step, current, endpoint, PARENT.GRID)
    for row in candidates:
        row.update({"family": family, "family_rank": 1})
    return PARENT.select_safe([dict(shrink), *candidates], old_low, old_high)


def canonical_shrink(sample: dict[str, torch.Tensor], step: torch.Tensor, current: torch.Tensor, old_low: float, old_high: float) -> tuple[dict[str, Any], int, float]:
    candidates = PARENT.grid_metrics(sample, step, torch.zeros_like(current), current, PARENT.GRID)
    zero = [row for row in candidates if row["grid_value"] == 0.0]
    if len(zero) != 1:
        raise RuntimeError("A1C shrink grid lacks a unique zero action")
    zero_drift = max(abs(float(zero[0]["low_mse"]) - old_low), abs(float(zero[0]["high_mse"]) - old_high))
    zero[0].update({"low_mse": old_low, "high_mse": old_high, "delta_abs": 0.0})
    for row in candidates:
        row.update({"family": "shrink", "family_rank": 0})
    selected, safe_count = PARENT.select_safe(candidates, old_low, old_high)
    return selected, safe_count, zero_drift


def selected_delta(current: torch.Tensor, endpoint: torch.Tensor, choice: dict[str, Any]) -> torch.Tensor:
    if choice["family"] == "shrink":
        return float(choice["grid_value"]) * current
    return current + float(choice["grid_value"]) * (endpoint - current)


def bootstrap(rows: list[dict[str, Any]], interface: str) -> dict[str, Any]:
    assert SOURCE is not None
    names = sorted({str(row["name"]) for row in rows})
    operators = tuple(SOURCE.V3W.OPERATORS)
    if len({(row["name"], row["operator"]) for row in rows}) != len(names) * len(operators):
        raise RuntimeError("A1C bootstrap rows are incomplete")
    keyed = {(str(row["name"]), str(row["operator"])): row for row in rows}
    fields = ("full_vs_shrink_db", "interface_vs_shrink_db")
    values = {
        operator: {
            field: np.asarray([float(keyed[(name, operator)][field]) for name in names], dtype=np.float64)
            for field in fields
        }
        for operator in operators
    }
    grouped = defaultdict(list)
    for name in names:
        grouped[str(keyed[(name, operators[0])]["native_shape"])].append(name)
    if len(grouped) < 2:
        raise RuntimeError("A1C requires both frozen native-size groups")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = {field: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for field in fields}
    ratios = {shape: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for shape in sorted(grouped)}
    index_by_name = {name: index for index, name in enumerate(names)}
    for draw in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(names), size=len(names), endpoint=False)
        for field in fields:
            draws[field][draw] = min(float(np.mean(values[op][field][indices])) for op in operators)
        for shape, shape_names in grouped.items():
            shape_indices = np.asarray([index_by_name[name] for name in shape_names], dtype=np.int64)
            sampled = generator.choice(shape_indices, size=len(shape_indices), replace=True)
            ratios[shape][draw] = min(
                float(np.mean(values[op]["interface_vs_shrink_db"][sampled])) / float(np.mean(values[op]["full_vs_shrink_db"][sampled]))
                if float(np.mean(values[op]["full_vs_shrink_db"][sampled])) > 0.0 else -math.inf
                for op in operators
            )
    family_count = 2 + len(ratios)
    lower_index = max(0, math.ceil(BOOTSTRAP_REPLICATES * (0.05 / (2 * family_count))) - 1)
    upper_index = min(BOOTSTRAP_REPLICATES - 1, math.floor(BOOTSTRAP_REPLICATES * (1.0 - 0.05 / (2 * family_count))) - 1)
    result: dict[str, Any] = {
        "schema_version": 1, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "interface": interface, "operator_pairing_retained": True, "worst_operator_within_each_draw": True,
        "simultaneous_family_count": family_count, "simultaneous_lower_index": lower_index, "simultaneous_upper_index": upper_index,
    }
    for field, array in draws.items():
        ordered = np.sort(array)
        result[f"worst_operator_{field}"] = min(float(np.mean(values[op][field])) for op in operators)
        result[f"worst_operator_{field}_lcb95"] = float(ordered[lower_index])
        result[f"worst_operator_{field}_ucb95"] = float(ordered[upper_index])
    result["native_size_ratio"] = {}
    for shape, array in sorted(ratios.items()):
        ordered = np.sort(array)
        result["native_size_ratio"][shape] = {"lcb95": float(ordered[lower_index]), "ucb95": float(ordered[upper_index])}
    result["worst_native_size_ratio_lcb95"] = min(item["lcb95"] for item in result["native_size_ratio"].values())
    result["worst_native_size_ratio_ucb95"] = min(item["ucb95"] for item in result["native_size_ratio"].values())
    return result


def run_a1c(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert PARENT is not None and SOURCE is not None and AUDIT is not None
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    PARENT.a0p.validate_frozen_args(args)
    audit = AUDIT
    review = validate_parent_review(Path(audit.a1r_review), audit.expected_a1r_review_sha256)
    payload, trace, state_path = PARENT.load_final_state(Path(audit.a0r_trace_dir))
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    fresh = list(all_names[256:768])
    if len(fresh) != 512 or len(set(fresh)) != 512:
        raise RuntimeError("A1C requires the immutable A1R fresh512 name range")
    selected_names = fresh if audit.a1c_stage == "formal" else fresh[:audit.smoke_count]
    if audit.a1c_stage == "smoke" and len(selected_names) != audit.smoke_count:
        raise RuntimeError("A1C smoke name count mismatch")
    _, model, optimizer, parameters = PARENT.a0p.load_cell(payload, args, v3s, legacy, frozen, list(all_names[:128]), folds, device)
    del optimizer, parameters
    model.eval()
    output = Path(output_dir)
    source_path = output / f"{args.run_tag}_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update({
        "route_id": ROUTE_ID, "route_commit": audit.expected_route_commit, "route_card_sha256": audit.expected_route_card_sha256,
        "stage": audit.a1c_stage, "interface": audit.interface, "a1r_review": str(Path(audit.a1r_review)),
        "a1r_review_sha256": sha256_file(Path(audit.a1r_review)), "a0r_trace_manifest_sha256": sha256_file(Path(audit.a0r_trace_dir) / "trace_manifest.json"),
        "final_state": str(state_path), "final_state_sha256": sha256_file(state_path), "a1f_module_sha256": sha256_file(Path(audit.a1f_module)),
        "fresh_names_sha256": hashlib.sha256("\n".join(fresh).encode("utf-8")).hexdigest(), "locked_test_touched": False, "canary_touched": False,
    })
    PARENT.write_json(source_path, source)
    PARENT.write_json(output / "v4a_a1c_source_manifest.json", source)
    rows: list[dict[str, Any]] = []
    zero_tensor_max = bound_excess_max = support_excess_max = grid_zero_mse_max = 0.0
    shape_names: set[str] = set()
    with torch.no_grad():
        for index, name in enumerate(selected_names):
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            support = sample["support"]
            shape = f"{sample['base'].shape[-2]}x{sample['base'].shape[-1]}"
            shape_names.add(shape)
            bound = sample["base"].new_tensor(frozen["bound"]).view(1, 3, 1, 1)
            delta_bound = sample["base"].new_tensor(args.delta_bound).view(1, 3, 1, 1)
            target_step = support * PARENT.clamp_channelwise(4.0 * (sample["label"] - sample["base"]), bound)
            for operator in SOURCE.V3W.OPERATORS:
                step = sample["steps"][operator]
                current = SOURCE.V3W.delta_for("output", model, sample, operator)
                target = support * PARENT.clamp_channelwise(target_step - step, delta_bound)
                endpoint = interface_endpoint(target, support, delta_bound, audit.interface)
                full_endpoint = interface_endpoint(target, support, delta_bound, "full")
                bound_excess_max = max(bound_excess_max, float(torch.clamp(endpoint.abs() - delta_bound, min=0.0).max().item()))
                inactive = support <= 0.0
                if bool(inactive.any()):
                    support_excess_max = max(support_excess_max, float(endpoint.masked_select(inactive.expand_as(endpoint)).abs().max().item()))
                old_low, old_high, current_low, current_high = v3s.candidate_predictions(sample["base"], step, current)
                old_low_mse = PARENT.a0p.scalar(v3s.per_image_mse(old_low, sample["label"]))
                old_high_mse = PARENT.a0p.scalar(v3s.per_image_mse(old_high, sample["label"]))
                zero_low, zero_high = v3s.candidate_predictions(sample["base"], step, torch.zeros_like(current))[2:]
                zero_tensor_max = max(zero_tensor_max, float((zero_low - old_low).abs().max().item()), float((zero_high - old_high).abs().max().item()))
                if zero_tensor_max != 0.0:
                    raise RuntimeError("A1C zero action does not reproduce predecessor tensors")
                shrink, shrink_safe, zero_drift = canonical_shrink(sample, step, current, old_low_mse, old_high_mse)
                grid_zero_mse_max = max(grid_zero_mse_max, zero_drift)
                selected, safe_count = select_direction(sample, step, current, endpoint, shrink, old_low_mse, old_high_mse, audit.interface)
                full, full_safe = select_direction(sample, step, current, full_endpoint, shrink, old_low_mse, old_high_mse, "full")
                selected_high = PARENT.metric_psnr(float(selected["high_mse"]))
                full_high = PARENT.metric_psnr(float(full["high_mse"]))
                shrink_high = PARENT.metric_psnr(float(shrink["high_mse"]))
                selected_action = selected_delta(current, endpoint, selected)
                tolerance = PARENT.numerical_tolerance(old_low_mse, old_high_mse, float(selected["low_mse"]), float(selected["high_mse"]))
                rows.append({
                    "schema_version": 1, "name": name, "operator": operator, "native_shape": shape, "interface": audit.interface,
                    "old_low_mse": old_low_mse, "old_high_mse": old_high_mse, "current_high_mse": PARENT.a0p.scalar(v3s.per_image_mse(current_high, sample["label"])),
                    "shrink_high_mse": float(shrink["high_mse"]), "interface_low_mse": float(selected["low_mse"]), "interface_high_mse": float(selected["high_mse"]),
                    "full_high_mse": float(full["high_mse"]), "interface_vs_shrink_db": selected_high - shrink_high, "full_vs_shrink_db": full_high - shrink_high,
                    "interface_vs_old25_db": selected_high - PARENT.metric_psnr(old_high_mse), "interface_grid_value": float(selected["grid_value"]),
                    "full_grid_value": float(full["grid_value"]), "interface_safe_count": safe_count, "full_safe_count": full_safe, "shrink_safe_count": shrink_safe,
                    "anchor_nonworse": float(selected["low_mse"]) <= old_low_mse + tolerance, "predecessor_nonworse": float(selected["high_mse"]) <= old_high_mse + tolerance,
                    "severe_vs_old25": selected_high - PARENT.metric_psnr(old_high_mse) <= -0.2, "hard_vs_old25": selected_high - PARENT.metric_psnr(old_high_mse) <= -0.5,
                    "interface_bound_saturation_fraction": PARENT.active_saturation_fraction(selected_action, delta_bound, support),
                })
            PARENT.emit_progress(Path(audit.status_file), {"V4A_A1C_PROGRESS": {"stage": audit.a1c_stage, "interface": audit.interface, "completed_images": index + 1, "total_images": len(selected_names)}})
    if audit.a1c_stage == "smoke" and len(shape_names) < 2:
        raise RuntimeError("A1C smoke did not cover both native-size groups")
    raw_path = output / "v4a_a1c_rows_cloud_only.csv"
    PARENT.write_rows(raw_path, rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["interface"], row["operator"])].append(row)
    summary = [{"interface": interface, "operator": operator, "image_count": len(values), "mean_interface_vs_shrink_db": float(np.mean([row["interface_vs_shrink_db"] for row in values])), "mean_full_vs_shrink_db": float(np.mean([row["full_vs_shrink_db"] for row in values])), "severe_count": sum(bool(row["severe_vs_old25"]) for row in values), "hard_count": sum(bool(row["hard_vs_old25"]) for row in values)} for (interface, operator), values in sorted(grouped.items())]
    PARENT.write_rows(output / "v4a_a1c_operator_summary.csv", summary)
    structural_valid = (len(rows) == len(selected_names) * len(SOURCE.V3W.OPERATORS) and zero_tensor_max == 0.0 and bound_excess_max <= 1e-7 and support_excess_max == 0.0 and grid_zero_mse_max <= 1e-9 and all(row["anchor_nonworse"] and row["predecessor_nonworse"] and not row["severe_vs_old25"] and not row["hard_vs_old25"] for row in rows))
    result = bootstrap(rows, audit.interface) if audit.a1c_stage == "formal" and structural_valid else {"schema_version": 1, "status": "NOT_RUN_SMOKE"}
    PARENT.write_json(output / "v4a_a1c_bootstrap_summary.json", result)
    if audit.a1c_stage == "smoke":
        state, decision, authorizes = ("COMPLETED_GATE_PASS", "V4A_A1C_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY", "A1C_FORMAL_EXACT_HALF_ONLY") if structural_valid else ("COMPLETED_GATE_FAIL", "V4A_A1C_S0_ALIGNMENT_FAIL_STOP", "NONE")
    else:
        interface_lcb = float(result["worst_operator_interface_vs_shrink_db_lcb95"])
        ratio_lcb = float(result["worst_native_size_ratio_lcb95"])
        full_lcb = float(result["worst_operator_full_vs_shrink_db_lcb95"])
        passed = interface_lcb >= INTERFACE_FLOOR_DB and ratio_lcb >= RETENTION_FLOOR and full_lcb >= INTERFACE_FLOOR_DB
        if passed:
            state, decision, authorizes = "COMPLETED_GATE_PASS", f"V4A_A1C_{audit.interface.upper()}_PASS", "R3_REVIEW_ONLY"
        else:
            state, decision, authorizes = "COMPLETED_GATE_FAIL", f"V4A_A1C_{audit.interface.upper()}_FAIL_R3_HANDOFF", "R3_REVIEW_ONLY"
    closeout = {"route_id": ROUTE_ID, "run_id": args.run_tag, "route_commit": audit.expected_route_commit, "runner_sha256": audit.runner_sha256, "stage": f"v4a-A1C-{audit.a1c_stage}", "state": state, "decision": decision, "authorizes": authorizes, "interface": audit.interface, "structural_valid": structural_valid, "evidence_role": "engineering_debug" if audit.a1c_stage == "smoke" else "development_screening", "gate_type": "structural_integrity" if audit.a1c_stage == "smoke" else "scientific_utility", "row_count": len(rows), "bound_excess_max": bound_excess_max, "support_excess_max": support_excess_max, "zero_tensor_max_abs": zero_tensor_max, "grid_zero_mse_max_abs": grid_zero_mse_max, "bootstrap_summary": str(output / "v4a_a1c_bootstrap_summary.json"), "raw_rows_cloud_only": str(raw_path), "operator_summary": str(output / "v4a_a1c_operator_summary.csv"), "source_manifest": str(output / "v4a_a1c_source_manifest.json"), "parent_review_decision": review["decision"], "locked_test_touched": False, "canary_touched": False, "training_occurred": False, "candidate_selected": False, "wall_seconds": time.perf_counter() - started, "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0}
    PARENT.write_json(output / "v4a_a1c_closeout.json", closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--a1f-module", required=True)
    parser.add_argument("--a0r-trace-dir", required=True)
    parser.add_argument("--a1r-review", required=True)
    parser.add_argument("--expected-a1r-review-sha256", required=True)
    parser.add_argument("--expected-route-commit", required=True)
    parser.add_argument("--expected-route-card-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--a1c-stage", required=True, choices=("smoke", "formal"))
    parser.add_argument("--interface", required=True, choices=("exact_half", "half_aa", "full"))
    parser.add_argument("--smoke-count", type=int, default=16)
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("frozen v3z arguments are required after A1C arguments")
    if args.smoke_count < 2 or args.smoke_count > 512:
        raise ValueError("smoke-count must be in [2, 512]")
    global PARENT, SOURCE, AUDIT
    PARENT = load_module(Path(args.a1f_module).resolve(), "a1f_parent")
    SOURCE = PARENT.a0p.load_source(Path(args.v3z_root).resolve())
    PARENT.SOURCE = SOURCE
    PARENT.a0p.SOURCE = SOURCE
    PARENT.a0p.AUDIT = args
    AUDIT = args
    SOURCE.run_projected = run_a1c
    original = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *v3z_args]
        SOURCE.main()
    finally:
        sys.argv = original


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a1c_safe_action_interface_ceiling.py audit ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
