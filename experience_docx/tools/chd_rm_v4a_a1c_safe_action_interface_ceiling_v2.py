#!/usr/bin/env python3
"""Frozen A1C full/exact-half/antialiased-half safe-action ceiling audit.

This program is intentionally non-deployable: labels choose privileged actions and
the only variable is the fixed endpoint transport declared by the A1C card.
"""
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
CELLS = ("full", "exact_half", "antialiased_half")
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 4000, 3407
FULL_GAIN_FLOOR, HALF_GAIN_FLOOR, RETENTION_FLOOR, REPAIR_FLOOR = 0.080, 0.080, 0.50, 0.20
CLOSE_GAIN = 0.020
PARENT: Any = None
SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def endpoint(target: torch.Tensor, current: torch.Tensor, support: torch.Tensor, bound: torch.Tensor, cell: str) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Reproduce the observed A1R correction transport, not an assumed half size."""
    if cell == "full":
        return target, None
    height, width = current.shape[-2:]
    expected = {(400, 400): (208, 208), (480, 640): (240, 320)}
    if (height, width) not in expected:
        raise RuntimeError(f"unsealed A1R transport shape {(height, width)}")
    low_size = expected[(height, width)]
    if cell not in ("exact_half", "antialiased_half"):
        raise ValueError(cell)
    if cell == "exact_half":
        low_target = F.interpolate(target, size=low_size, mode="bilinear", align_corners=False, antialias=False)
        low_current = F.interpolate(current, size=low_size, mode="bilinear", align_corners=False, antialias=False)
    else:
        low_target = F.interpolate(target, size=low_size, mode="bilinear", align_corners=False, antialias=True)
        low_current = F.interpolate(current, size=low_size, mode="bilinear", align_corners=False, antialias=True)
    low_difference = low_target - low_current
    if float(torch.clamp(low_difference.abs() - 2.0 * bound, min=0).max().item()) > 1e-6:
        raise RuntimeError("half correction exceeds the theoretical +/-2B range")
    replay = F.interpolate(low_difference, size=(height, width), mode="bilinear", align_corners=False)
    # A1R clips only the reconstructed endpoint; support is applied after replay.
    return support * PARENT.clamp_channelwise(current + support * replay, bound), low_size


def canonical_shrink(sample: dict[str, torch.Tensor], step: torch.Tensor, current: torch.Tensor, old_low: float, old_high: float) -> tuple[dict[str, Any], int, float]:
    candidates = PARENT.grid_metrics(sample, step, torch.zeros_like(current), current, PARENT.GRID)
    zero = [item for item in candidates if item["grid_value"] == 0.0]
    if len(zero) != 1:
        raise RuntimeError("canonical shrink zero action is not unique")
    drift = max(abs(float(zero[0]["low_mse"]) - old_low), abs(float(zero[0]["high_mse"]) - old_high))
    zero[0].update({"low_mse": old_low, "high_mse": old_high, "delta_abs": 0.0})
    for item in candidates:
        item.update({"family": "shrink", "family_rank": 0})
    selected, count = PARENT.select_safe(candidates, old_low, old_high)
    return selected, count, drift


def select(sample: dict[str, torch.Tensor], step: torch.Tensor, current: torch.Tensor, value: torch.Tensor, shrink: dict[str, Any], old_low: float, old_high: float, cell: str) -> tuple[dict[str, Any], int]:
    rows = PARENT.grid_metrics(sample, step, current, value, PARENT.GRID)
    for item in rows:
        item.update({"family": cell, "family_rank": 1})
    return PARENT.select_safe([dict(shrink), *rows], old_low, old_high)


def selected_delta(current: torch.Tensor, value: torch.Tensor, choice: dict[str, Any]) -> torch.Tensor:
    return float(choice["grid_value"]) * current if choice["family"] == "shrink" else current + float(choice["grid_value"]) * (value - current)


def _bound(values: np.ndarray, alpha: float) -> tuple[float, float]:
    ordered = np.sort(values)
    lower = max(0, math.ceil(len(ordered) * alpha) - 1)
    upper = min(len(ordered) - 1, math.floor(len(ordered) * (1.0 - alpha)) - 1)
    return float(ordered[lower]), float(ordered[upper])


def paired_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resample images once per draw; retain all three cells and both operators."""
    assert SOURCE is not None
    names = sorted({str(row["name"]) for row in rows})
    operators = tuple(SOURCE.V3W.OPERATORS)
    keyed = {(str(row["name"]), str(row["operator"]), str(row["cell"])): row for row in rows}
    if len(keyed) != len(names) * len(operators) * len(CELLS):
        raise RuntimeError("incomplete paired A1C cells")
    shapes = sorted({str(row["native_shape"]) for row in rows})
    if set(shapes) != {"400x400", "480x640"}:
        raise RuntimeError(f"A1C shape blocks drifted: {shapes}")
    index = {name: i for i, name in enumerate(names)}
    shape_indices = {shape: np.asarray([index[name] for name in names if keyed[(name, operators[0], "full")]["native_shape"] == shape]) for shape in shapes}
    fields = ("gain", "repairable")
    values = {cell: {op: {field: np.asarray([float(keyed[(name, op, cell)][field]) for name in names]) for field in fields} for op in operators} for cell in CELLS}
    gains = {cell: np.empty(BOOTSTRAP_DRAWS) for cell in CELLS}
    repairs = {cell: np.empty(BOOTSTRAP_DRAWS) for cell in CELLS}
    retention = {cell: {shape: np.empty(BOOTSTRAP_DRAWS) for shape in shapes} for cell in CELLS if cell != "full"}
    aa_minus_exact = np.empty(BOOTSTRAP_DRAWS)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(names), len(names))
        for cell in CELLS:
            gains[cell][draw] = min(float(values[cell][op]["gain"][sampled].mean()) for op in operators)
            repairs[cell][draw] = min(float(values[cell][op]["repairable"][sampled].mean()) for op in operators)
        aa_minus_exact[draw] = gains["antialiased_half"][draw] - gains["exact_half"][draw]
        for cell in ("exact_half", "antialiased_half"):
            for shape, indices in shape_indices.items():
                retention[cell][shape][draw] = min(
                    float(values[cell][op]["gain"][indices].mean()) / float(values["full"][op]["gain"][indices].mean())
                    if float(values["full"][op]["gain"][indices].mean()) > 0 else -math.inf
                    for op in operators
                )
    result: dict[str, Any] = {"schema_version": 2, "bootstrap_replicates": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "paired_image_resampling": True, "operators_worst_within_draw": True, "full_alpha": 0.05, "half_family_alpha": 0.025, "cells": {}}
    for cell in CELLS:
        alpha = 0.05 if cell == "full" else 0.025
        gain_lcb, gain_ucb = _bound(gains[cell], alpha)
        repair_lcb, repair_ucb = _bound(repairs[cell], alpha)
        entry: dict[str, Any] = {"worst_operator_gain_db": float(gains[cell].mean()), "gain_lcb": gain_lcb, "gain_ucb": gain_ucb, "repairable_lcb": repair_lcb, "repairable_ucb": repair_ucb}
        if cell != "full":
            entry["retention_by_native_shape"] = {shape: dict(zip(("lcb", "ucb"), _bound(retention[cell][shape], alpha))) for shape in shapes}
            entry["worst_native_size_retention_lcb"] = min(item["lcb"] for item in entry["retention_by_native_shape"].values())
            entry["worst_native_size_retention_ucb"] = min(item["ucb"] for item in entry["retention_by_native_shape"].values())
        result["cells"][cell] = entry
    result["antialiased_minus_exact_gain_lcb"] = _bound(aa_minus_exact, 0.05)[0]
    return result


def classify(result: dict[str, Any], structural_valid: bool) -> tuple[str, str, str]:
    if not structural_valid:
        return "COMPLETED_GATE_FAIL", "V4A_A1C_FULL_REFERENCE_OR_INTEGRITY_FAIL_R3_HANDOFF", "R3_REVIEW_ONLY"
    full = result["cells"]["full"]
    full_pass = full["gain_lcb"] >= FULL_GAIN_FLOOR and full["repairable_lcb"] >= REPAIR_FLOOR
    full_fail = full["gain_ucb"] < CLOSE_GAIN or full["repairable_ucb"] < REPAIR_FLOOR
    if full_fail:
        return "COMPLETED_GATE_FAIL", "V4A_A1C_FULL_REFERENCE_OR_INTEGRITY_FAIL_R3_HANDOFF", "R3_REVIEW_ONLY"
    adequate, failed = {}, {}
    for cell in ("exact_half", "antialiased_half"):
        item = result["cells"][cell]
        adequate[cell] = item["gain_lcb"] >= HALF_GAIN_FLOOR and item["worst_native_size_retention_lcb"] >= RETENTION_FLOOR and item["repairable_lcb"] >= REPAIR_FLOOR
        failed[cell] = item["gain_ucb"] < CLOSE_GAIN or item["worst_native_size_retention_ucb"] < RETENTION_FLOOR or item["repairable_ucb"] < REPAIR_FLOOR
    if full_pass and adequate["exact_half"]:
        return "COMPLETED_GATE_PASS", "V4A_A1C_EXACT_HALF_INTERFACE_ADEQUACY_PASS_R3_HANDOFF", "R3_REVIEW_ONLY"
    if full_pass and adequate["antialiased_half"]:
        return "COMPLETED_GATE_PASS", "V4A_A1C_ANTIALIASED_HALF_INTERFACE_ADEQUACY_ONLY_PASS_R3_HANDOFF", "R3_REVIEW_ONLY"
    if full_pass and all(failed.values()):
        return "COMPLETED_GATE_FAIL", "V4A_A1C_HALF_INTERFACE_ADEQUACY_FAIL_R3_HANDOFF", "R3_REVIEW_ONLY"
    return "COMPLETED_GATE_INCONCLUSIVE", "V4A_A1C_INTERFACE_RESULT_INCONCLUSIVE_R3_HANDOFF", "R3_REVIEW_ONLY"


def run_a1c(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert PARENT is not None and SOURCE is not None and AUDIT is not None
    started, audit = time.perf_counter(), AUDIT
    PARENT.a0p.validate_frozen_args(args)
    payload, _, state_path = PARENT.load_final_state(Path(audit.a0r_trace_dir))
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    fresh = list(all_names[256:768])
    selected = fresh[:32] if audit.mode == "s0" else fresh
    if len(fresh) != 512 or len(set(fresh)) != 512 or len(selected) != (32 if audit.mode == "s0" else 512):
        raise RuntimeError("A1C frozen fresh512/S0 name contract failed")
    _, model, optimizer, parameters = PARENT.a0p.load_cell(payload, args, v3s, legacy, frozen, list(all_names[:128]), folds, device)
    del optimizer, parameters
    model.eval()
    out = Path(output_dir)
    source_path = out / f"{args.run_tag}_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update({"route_id": ROUTE_ID, "route_commit": audit.expected_route_commit, "stage": audit.mode, "cells": list(CELLS), "a1r_transport_pairs": {"400x400": "208x208", "480x640": "240x320"}, "fresh_names_sha256": hashlib.sha256("\n".join(fresh).encode()).hexdigest(), "final_state_sha256": sha256(state_path), "locked_test_touched": False, "canary_touched": False, "confirmation_touched": False})
    PARENT.write_json(out / "v4a_a1c_source_manifest.json", source)
    rows: list[dict[str, Any]] = []
    limits = {"zero": 0.0, "endpoint": 0.0, "support": 0.0, "grid_zero": 0.0}
    with torch.no_grad():
        for number, name in enumerate(selected, 1):
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            support, shape = sample["support"], f"{sample['base'].shape[-2]}x{sample['base'].shape[-1]}"
            bound = sample["base"].new_tensor(frozen["bound"]).view(1, 3, 1, 1)
            delta_bound = sample["base"].new_tensor(args.delta_bound).view(1, 3, 1, 1)
            target_step = support * PARENT.clamp_channelwise(4.0 * (sample["label"] - sample["base"]), bound)
            for operator in SOURCE.V3W.OPERATORS:
                step, current = sample["steps"][operator], SOURCE.V3W.delta_for("output", model, sample, operator)
                target = support * PARENT.clamp_channelwise(target_step - step, delta_bound)
                old_low, old_high, _, current_high = v3s.candidate_predictions(sample["base"], step, current)
                old_low_mse, old_high_mse = PARENT.a0p.scalar(v3s.per_image_mse(old_low, sample["label"])), PARENT.a0p.scalar(v3s.per_image_mse(old_high, sample["label"]))
                zero_low, zero_high = v3s.candidate_predictions(sample["base"], step, torch.zeros_like(current))[2:]
                limits["zero"] = max(limits["zero"], float((zero_low - old_low).abs().max()), float((zero_high - old_high).abs().max()))
                shrink, shrink_count, drift = canonical_shrink(sample, step, current, old_low_mse, old_high_mse)
                limits["grid_zero"] = max(limits["grid_zero"], drift)
                for cell in CELLS:
                    value, low_size = endpoint(target, current, support, delta_bound, cell)
                    limits["endpoint"] = max(limits["endpoint"], float(torch.clamp(value.abs() - delta_bound, min=0).max()))
                    inactive = support <= 0
                    limits["support"] = max(limits["support"], float(value.masked_select(inactive.expand_as(value)).abs().max()) if bool(inactive.any()) else 0.0)
                    chosen, safe_count = select(sample, step, current, value, shrink, old_low_mse, old_high_mse, cell)
                    high_psnr, shrink_psnr, old_psnr = (PARENT.metric_psnr(float(chosen["high_mse"])), PARENT.metric_psnr(float(shrink["high_mse"])), PARENT.metric_psnr(old_high_mse))
                    tolerance = PARENT.numerical_tolerance(old_low_mse, old_high_mse, float(chosen["low_mse"]), float(chosen["high_mse"]))
                    action = selected_delta(current, value, chosen)
                    rows.append({"name": name, "operator": operator, "cell": cell, "native_shape": shape, "low_shape": "full" if low_size is None else f"{low_size[0]}x{low_size[1]}", "gain": high_psnr - shrink_psnr, "repairable": float(high_psnr > old_psnr), "anchor_nonworse": float(chosen["low_mse"]) <= old_low_mse + tolerance, "predecessor_nonworse": float(chosen["high_mse"]) <= old_high_mse + tolerance, "severe": high_psnr - old_psnr <= -0.2, "hard": high_psnr - old_psnr <= -0.5, "safe_count": safe_count, "saturation": PARENT.active_saturation_fraction(action, delta_bound, support)})
            PARENT.emit_progress(Path(audit.status_file), {"V4A_A1C_PROGRESS": {"stage": audit.mode, "completed_images": number, "total_images": len(selected), "cells": list(CELLS)}})
    structural = len(rows) == len(selected) * len(SOURCE.V3W.OPERATORS) * len(CELLS) and limits["zero"] == 0.0 and limits["endpoint"] <= 1e-7 and limits["support"] == 0.0 and limits["grid_zero"] <= 1e-9 and all(row["anchor_nonworse"] and row["predecessor_nonworse"] and not row["severe"] and not row["hard"] for row in rows)
    PARENT.write_rows(out / "v4a_a1c_rows_cloud_only.csv", rows)
    summary = [{"cell": cell, "operator": op, "count": len(group), "mean_gain_db": float(np.mean([row["gain"] for row in group]))} for (cell, op), group in sorted(((key, value) for key, value in defaultdict(list, {key: [row for row in rows if (row["cell"], row["operator"]) == key] for key in {(row["cell"], row["operator"]) for row in rows}}).items()))]
    PARENT.write_rows(out / "v4a_a1c_cell_operator_summary.csv", summary)
    bootstrap = paired_bootstrap(rows) if audit.mode == "formal" and structural else {"schema_version": 2, "status": "NOT_RUN_S0"}
    PARENT.write_json(out / "v4a_a1c_bootstrap_summary.json", bootstrap)
    state, decision, authorizes = ("COMPLETED_GATE_PASS", "V4A_A1C_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY", "A1C_FORMAL_INTERFACE_CEILING_ONLY") if audit.mode == "s0" and structural else (("COMPLETED_GATE_FAIL", "V4A_A1C_S0_ALIGNMENT_FAIL_STOP", "NONE") if audit.mode == "s0" else classify(bootstrap, structural))
    closeout = {"schema_version": 2, "route_id": ROUTE_ID, "run_id": args.run_tag, "route_commit": audit.expected_route_commit, "runner_sha256": audit.runner_sha256, "stage": audit.mode, "state": state, "decision": decision, "authorizes": authorizes, "cells": list(CELLS), "structural_valid": structural, "evidence_role": "engineering_debug" if audit.mode == "s0" else "development_screening", "gate_type": "structural_integrity" if audit.mode == "s0" else "scientific_utility", "limits": limits, "source_manifest": str(out / "v4a_a1c_source_manifest.json"), "bootstrap_summary": str(out / "v4a_a1c_bootstrap_summary.json"), "raw_rows_cloud_only": str(out / "v4a_a1c_rows_cloud_only.csv"), "locked_test_touched": False, "canary_touched": False, "confirmation_touched": False, "training_occurred": False, "candidate_selected": False, "wall_seconds": time.perf_counter() - started}
    PARENT.write_json(out / "v4a_a1c_closeout.json", closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("v3z-root", "a1f-module", "a0r-trace-dir", "expected-route-commit", "expected-route-card-sha256", "runner-sha256", "status-file"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--mode", required=True, choices=("s0", "formal"))
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("frozen v3z arguments are required")
    global PARENT, SOURCE, AUDIT
    PARENT = load_module(Path(args.a1f_module).resolve(), "a1f_parent")
    SOURCE = PARENT.a0p.load_source(Path(args.v3z_root).resolve())
    PARENT.SOURCE = SOURCE
    PARENT.a0p.SOURCE = SOURCE
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
        raise SystemExit("usage: chd_rm_v4a_a1c_safe_action_interface_ceiling_v2.py audit ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
