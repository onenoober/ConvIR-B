#!/usr/bin/env python3
"""Privileged bounded Delta-u action feasibility at the exact v3z final state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import chd_rm_v4a_a0p_audit as a0p


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714"
PARENT_ROUTE_ID = "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714"
EXPECTED_R3_REVIEW = {
    "route_id": PARENT_ROUTE_ID,
    "state": "COMPLETED_R3_REVIEW",
    "decision": "V4A_A0P_NO_LOCAL_CORRECTION_AUTHORIZE_A1F_METRIC_ALIGNED_FEASIBILITY_ONLY",
    "authorizes": "A1F_ROUTE_DESIGN_AND_IMPLEMENTATION_ONLY",
}
GRID = tuple(index / 64.0 for index in range(65))
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 3407
UTILITY_SESOI_DB = 0.005
REPAIRABLE_FRACTION_FLOOR = 0.20
MSE_REPLAY_TOLERANCE = 1e-12
PSNR_REPLAY_TOLERANCE_DB = 1e-9
SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty rows: {path}")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def numerical_tolerance(*values: float) -> float:
    return 2.0 * (1e-12 + 1e-12 * max(abs(value) for value in values))


def clamp_channelwise(value: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(value, bound), -bound)


def validate_review(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("v4a A0P R3 review hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_R3_REVIEW.items():
        if value.get(key) != expected:
            raise RuntimeError(f"v4a A0P R3 review tuple mismatch: {key}={value.get(key)!r}")
    return value


def load_final_state(trace_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    assert SOURCE is not None
    trace_path = trace_dir / "trace_manifest.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("route_id") != PARENT_ROUTE_ID or trace.get("replicate_id") != "r1":
        raise RuntimeError("A1F requires the canonical v4a A0R r1 trace")
    if sha256_file(Path(SOURCE.__file__)) != trace.get("v3z_source_sha256"):
        raise RuntimeError("A1F immutable v3z source differs from A0R")
    final_rows = [row for row in trace.get("states", []) if row.get("state_kind") == "final"]
    if len(final_rows) != 1:
        raise RuntimeError(f"A1F requires one final state, found {len(final_rows)}")
    final = final_rows[0]
    if int(final.get("epoch", -1)) != 16 or int(final.get("update", -1)) != 512:
        raise RuntimeError("A1F final state is not epoch16/update512")
    state_path = trace_dir.parent / Path(str(final["relative_path"]))
    if not state_path.is_file() or sha256_file(state_path) != final.get("sha256"):
        raise RuntimeError("A1F final state file is missing or hash-mismatched")
    payload = torch.load(state_path, map_location="cpu")
    if payload.get("replicate_id") != "r1" or payload.get("state_kind") != "final":
        raise RuntimeError("A1F final state payload identity mismatch")
    return payload, trace, state_path


def load_a0d_rows(path: Path, expected_sha256: str) -> dict[tuple[str, str, str], dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("A0D raw-row hash mismatch")
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != 512:
        raise RuntimeError(f"A1F requires 512 A0D rows, found {len(rows)}")
    keyed = {(row["split"], row["name"], row["operator"]): row for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("A0D row keys are not unique")
    return keyed


def grid_metrics(
    sample: dict[str, torch.Tensor],
    step: torch.Tensor,
    start_delta: torch.Tensor,
    end_delta: torch.Tensor,
    values: tuple[float, ...],
    chunk_size: int = 8,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    base = sample["base"]
    label = sample["label"]
    with torch.no_grad():
        for offset in range(0, len(values), chunk_size):
            chunk = values[offset:offset + chunk_size]
            coefficient = base.new_tensor(chunk).view(-1, 1, 1, 1)
            delta = start_delta + coefficient * (end_delta - start_delta)
            low = torch.clamp(base + 0.125 * (step + delta), 0.0, 1.0)
            high = torch.clamp(base + 0.25 * (step + delta), 0.0, 1.0)
            low_mse = (low - label).square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            high_mse = (high - label).square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            delta_abs = delta.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for index, value in enumerate(chunk):
                results.append({
                    "grid_value": float(value),
                    "low_mse": float(low_mse[index]),
                    "high_mse": float(high_mse[index]),
                    "delta_abs": float(delta_abs[index]),
                })
    return results


def select_safe(
    candidates: list[dict[str, Any]], old_low_mse: float, old_high_mse: float
) -> tuple[dict[str, Any], int]:
    safe: list[dict[str, Any]] = []
    for candidate in candidates:
        tolerance = numerical_tolerance(old_low_mse, old_high_mse, candidate["low_mse"], candidate["high_mse"])
        if candidate["low_mse"] <= old_low_mse + tolerance and candidate["high_mse"] <= old_high_mse + tolerance:
            safe.append(candidate)
    if not safe:
        raise RuntimeError("an action family has no safe candidate")
    selected = min(
        safe,
        key=lambda row: (float(row["high_mse"]), int(row["family_rank"]), float(row["grid_value"])),
    )
    return selected, len(safe)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["operator"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (split, operator), values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "schema_version": 1,
            "split": split,
            "operator": operator,
            "image_count": len(values),
            "direction_selected_count": sum(bool(row["direction_selected"]) for row in values),
            "direction_selected_fraction": float(np.mean([float(row["direction_selected"]) for row in values])),
            "severe_vs_old25_count": sum(bool(row["direction_severe_vs_old25"]) for row in values),
            "hard_vs_old25_count": sum(bool(row["direction_hard_vs_old25"]) for row in values),
            "max_anchor_excess": max(float(row["direction_low_mse"] - row["old_low_mse"]) for row in values),
            "max_predecessor_excess": max(float(row["direction_high_mse"] - row["old_high_mse"]) for row in values),
        }
        for key in ("direction_vs_shrink_db", "direction_vs_current_db", "direction_vs_old25_db"):
            array = np.asarray([float(row[key]) for row in values], dtype=np.float64)
            result[f"mean_{key}"] = float(np.mean(array))
            result[f"p05_{key}"] = float(np.quantile(array, 0.05, method="linear"))
            result[f"p10_{key}"] = float(np.quantile(array, 0.10, method="linear"))
        summaries.append(result)
    return summaries


def bootstrap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = [row for row in rows if row["split"] == "heldout128"]
    names = sorted({str(row["name"]) for row in heldout})
    operators = tuple(SOURCE.V3W.OPERATORS)
    if len(names) != 128 or len(heldout) != 128 * len(operators):
        raise RuntimeError("formal A1F bootstrap requires heldout128 paired operators")
    keyed = {(str(row["name"]), str(row["operator"])): row for row in heldout}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for operator in operators:
        arrays[operator] = {
            key: np.asarray([float(keyed[(name, operator)][key]) for name in names], dtype=np.float64)
            for key in ("direction_vs_shrink_db", "direction_vs_current_db", "direction_vs_old25_db", "direction_selected")
        }
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = {key: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for key in arrays[operators[0]]}
    for draw_index in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(names), size=len(names), endpoint=False)
        for key in draws:
            draws[key][draw_index] = min(float(np.mean(arrays[operator][key][indices])) for operator in operators)
    result: dict[str, Any] = {
        "schema_version": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "operator_pairing_retained": True,
        "worst_operator_within_each_draw": True,
        "operator_points": {
            operator: {key: float(np.mean(values)) for key, values in arrays[operator].items()}
            for operator in operators
        },
    }
    for key, values in draws.items():
        ordered = np.sort(values)
        result[f"worst_operator_{key}"] = min(float(np.mean(arrays[operator][key])) for operator in operators)
        result[f"worst_operator_{key}_lcb95"] = float(ordered[199])
        result[f"worst_operator_{key}_ucb95"] = float(ordered[3799])
    return result


def run_a1f(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert SOURCE is not None and AUDIT is not None
    a0p.validate_frozen_args(args)
    audit = AUDIT
    review = validate_review(Path(audit.r3_review), audit.expected_r3_review_sha256)
    a0d = load_a0d_rows(Path(audit.a0d_rows), audit.expected_a0d_rows_sha256)
    payload, trace, state_path = load_final_state(Path(audit.a0r_trace_dir))
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    update_names = list(all_names[:128])
    heldout_names = list(all_names[128:256])
    if list(names) != update_names or len(set(update_names) & set(heldout_names)) != 0:
        raise RuntimeError("A1F update128/heldout128 identity mismatch")
    limit = 8 if audit.a1f_stage == "smoke" else 128
    selected_names = {"update128": update_names[:limit], "heldout128": heldout_names[:limit]}
    label, model, optimizer, parameters = a0p.load_cell(payload, args, v3s, legacy, frozen, update_names, folds, device)
    del optimizer, parameters
    model.eval()
    output = Path(output_dir)
    source_path = output / f"{args.run_tag}_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update({
        "route_id": ROUTE_ID,
        "route_commit": audit.expected_route_commit,
        "route_card_sha256": audit.expected_route_card_sha256,
        "stage": audit.a1f_stage,
        "parent_route_id": PARENT_ROUTE_ID,
        "r3_review": str(Path(audit.r3_review)),
        "r3_review_sha256": sha256_file(Path(audit.r3_review)),
        "a0d_rows": str(Path(audit.a0d_rows)),
        "a0d_rows_sha256": sha256_file(Path(audit.a0d_rows)),
        "a0r_trace_manifest_sha256": sha256_file(Path(audit.a0r_trace_dir) / "trace_manifest.json"),
        "final_state": str(state_path),
        "final_state_sha256": sha256_file(state_path),
        "grid": list(GRID),
        "action_families": ["shrink", "direction_union_shrink"],
        "sample_counts": {split: len(values) for split, values in selected_names.items()},
        "locked_test_touched": False,
        "canary_touched": False,
    })
    manifest_path = output / "v4a_a1f_source_manifest.json"
    write_json(source_path, source)
    write_json(manifest_path, source)
    rows: list[dict[str, Any]] = []
    replay_mse_max = 0.0
    replay_psnr_max = 0.0
    bound_excess_max = 0.0
    support_excess_max = 0.0
    with torch.no_grad():
        for split, split_names in selected_names.items():
            for name_index, name in enumerate(split_names):
                sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
                support = sample["support"]
                bound = sample["base"].new_tensor(frozen["bound"]).view(1, 3, 1, 1)
                delta_bound = sample["base"].new_tensor(args.delta_bound).view(1, 3, 1, 1)
                target_step = support * clamp_channelwise(4.0 * (sample["label"] - sample["base"]), bound)
                for operator in SOURCE.V3W.OPERATORS:
                    step = sample["steps"][operator]
                    current_delta = SOURCE.V3W.delta_for(label, model, sample, operator)
                    target_delta = support * clamp_channelwise(target_step - step, delta_bound)
                    bound_excess_max = max(
                        bound_excess_max,
                        float(torch.clamp(current_delta.abs() - delta_bound, min=0.0).max().item()),
                        float(torch.clamp(target_delta.abs() - delta_bound, min=0.0).max().item()),
                    )
                    inactive = support <= 0.0
                    if bool(inactive.any()):
                        support_excess_max = max(
                            support_excess_max,
                            float(current_delta.masked_select(inactive.expand_as(current_delta)).abs().max().item()),
                            float(target_delta.masked_select(inactive.expand_as(target_delta)).abs().max().item()),
                        )
                    old_low, old_high, current_low, current_high = v3s.candidate_predictions(
                        sample["base"], step, current_delta
                    )
                    old_low_mse = a0p.scalar(v3s.per_image_mse(old_low, sample["label"]))
                    old_high_mse = a0p.scalar(v3s.per_image_mse(old_high, sample["label"]))
                    current_low_mse = a0p.scalar(v3s.per_image_mse(current_low, sample["label"]))
                    current_high_mse = a0p.scalar(v3s.per_image_mse(current_high, sample["label"]))
                    reference = a0d[(split, name, operator)]
                    replay_values = (
                        (old_low_mse, float(reference["old_125_mse"])),
                        (old_high_mse, float(reference["old_250_mse"])),
                        (current_high_mse, float(reference["new_250_mse"])),
                    )
                    replay_mse_max = max(replay_mse_max, *(abs(left - right) for left, right in replay_values))
                    current_psnr = metric_psnr(current_high_mse)
                    old_high_psnr = metric_psnr(old_high_mse)
                    replay_psnr_max = max(
                        replay_psnr_max,
                        abs(current_psnr - float(reference["new_250_psnr"])),
                        abs(old_high_psnr - float(reference["old_250_psnr"])),
                        abs((current_psnr - old_high_psnr) - float(reference["delta_psnr"])),
                    )
                    shrink_candidates = grid_metrics(sample, step, torch.zeros_like(current_delta), current_delta, GRID)
                    for candidate in shrink_candidates:
                        candidate.update({"family": "shrink", "family_rank": 0})
                    shrink, shrink_safe_count = select_safe(shrink_candidates, old_low_mse, old_high_mse)
                    direction_candidates = grid_metrics(sample, step, current_delta, target_delta, GRID)
                    for candidate in direction_candidates:
                        candidate.update({"family": "direction", "family_rank": 1})
                    direction, direction_safe_count = select_safe(
                        [dict(shrink), *direction_candidates], old_low_mse, old_high_mse
                    )
                    shrink_psnr = metric_psnr(float(shrink["high_mse"]))
                    direction_psnr = metric_psnr(float(direction["high_mse"]))
                    direction_vs_shrink = direction_psnr - shrink_psnr
                    tolerance = numerical_tolerance(float(shrink["high_mse"]), float(direction["high_mse"]))
                    direction_selected = bool(
                        direction["family"] == "direction"
                        and float(direction["high_mse"]) < float(shrink["high_mse"]) - tolerance
                    )
                    inherited_harm = max(old_high_mse - old_low_mse, 0.0)
                    direction_total_harm = max(float(direction["high_mse"]) - old_low_mse, 0.0)
                    row = {
                        "schema_version": 1,
                        "split": split,
                        "name": name,
                        "operator": operator,
                        "old_low_mse": old_low_mse,
                        "old_high_mse": old_high_mse,
                        "current_low_mse": current_low_mse,
                        "current_high_mse": current_high_mse,
                        "shrink_low_mse": float(shrink["low_mse"]),
                        "shrink_high_mse": float(shrink["high_mse"]),
                        "shrink_grid_value": float(shrink["grid_value"]),
                        "shrink_safe_count": shrink_safe_count,
                        "direction_low_mse": float(direction["low_mse"]),
                        "direction_high_mse": float(direction["high_mse"]),
                        "direction_family": str(direction["family"]),
                        "direction_grid_value": float(direction["grid_value"]),
                        "direction_safe_count": direction_safe_count,
                        "direction_selected": direction_selected,
                        "direction_vs_shrink_db": direction_vs_shrink,
                        "direction_vs_current_db": direction_psnr - current_psnr,
                        "direction_vs_old25_db": direction_psnr - old_high_psnr,
                        "direction_severe_vs_old25": direction_psnr - old_high_psnr <= -0.2,
                        "direction_hard_vs_old25": direction_psnr - old_high_psnr <= -0.5,
                        "inherited_harm": inherited_harm,
                        "direction_total_harm": direction_total_harm,
                        "direction_added_harm": direction_total_harm - inherited_harm,
                        "current_delta_abs": float(current_delta.abs().mean().item()),
                        "target_delta_abs": float(target_delta.abs().mean().item()),
                    }
                    if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"split", "name", "operator", "direction_family"}):
                        raise FloatingPointError("non-finite A1F row")
                    rows.append(row)
                print(json.dumps({"V4A_A1F_PROGRESS": {"stage": audit.a1f_stage, "split": split, "completed_images": name_index + 1, "total_images": len(split_names)}}, sort_keys=True), flush=True)
    raw_path = output / "v4a_a1f_rows_cloud_only.csv"
    write_rows(raw_path, rows)
    summaries = summarize_rows(rows)
    summary_path = output / "v4a_a1f_operator_summary.csv"
    write_rows(summary_path, summaries)
    structural_valid = (
        replay_mse_max <= MSE_REPLAY_TOLERANCE
        and replay_psnr_max <= PSNR_REPLAY_TOLERANCE_DB
        and bound_excess_max <= 1e-7
        and support_excess_max == 0.0
        and len(rows) == 2 * limit * len(SOURCE.V3W.OPERATORS)
        and all(not row["direction_severe_vs_old25"] and not row["direction_hard_vs_old25"] for row in rows)
    )
    bootstrap: dict[str, Any] | None = None
    formal_pass = False
    if audit.a1f_stage == "formal" and structural_valid:
        bootstrap = bootstrap_summary(rows)
        formal_pass = (
            float(bootstrap["worst_operator_direction_vs_shrink_db_lcb95"]) >= UTILITY_SESOI_DB
            and float(bootstrap["worst_operator_direction_selected_lcb95"]) >= REPAIRABLE_FRACTION_FLOOR
            and float(bootstrap["worst_operator_direction_vs_old25_db_lcb95"]) >= 0.0
        )
    bootstrap_path = output / "v4a_a1f_bootstrap_summary.json"
    write_json(bootstrap_path, bootstrap or {"status": "NOT_RUN_SMOKE", "schema_version": 1})
    if audit.a1f_stage == "smoke":
        state = "COMPLETED_GATE_PASS" if structural_valid else "COMPLETED_GATE_FAIL"
        decision = "V4A_A1F_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY" if structural_valid else "V4A_A1F_S0_ALIGNMENT_FAIL_STOP"
        authorizes = "A1F_FORMAL_ONLY" if structural_valid else "NONE"
        reason = (
            "exact A0D replay and the frozen action-integrity checks passed"
            if structural_valid
            else "exact A0D replay or a frozen action-integrity check failed"
        )
    elif formal_pass:
        state = "COMPLETED_GATE_PASS"
        decision = "V4A_A1F_SAFE_DIRECTION_HEADROOM_PASS_R3_HANDOFF"
        authorizes = "R3_REVIEW_ONLY"
        reason = "heldout worst-operator direction-over-shrink, repairable-fraction, and predecessor-safety gates passed"
    else:
        state = "COMPLETED_GATE_FAIL"
        worst_current = float(bootstrap["worst_operator_direction_vs_current_db_lcb95"]) if bootstrap else -math.inf
        decision = (
            "V4A_A1F_DIRECTION_NOT_ABOVE_SHRINK_STOP_R3_HANDOFF"
            if structural_valid and worst_current > 0.0
            else "V4A_A1F_NO_SAFE_BOUNDED_HEADROOM_STOP_R3_HANDOFF"
        )
        authorizes = "R3_REVIEW_ONLY"
        reason = (
            "safe direction lift versus current exists but does not clear the direction-over-shrink formal gate"
            if structural_valid and worst_current > 0.0
            else "structural validity or safe bounded direction headroom is insufficient for continuation"
        )
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "route_commit": audit.expected_route_commit,
        "stage": f"v4a-A1F-{audit.a1f_stage}",
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "reason": reason,
        "evidence_role": "engineering_debug" if audit.a1f_stage == "smoke" else "development_screening",
        "gate_type": "structural_integrity" if audit.a1f_stage == "smoke" else "scientific_utility",
        "metric_contract": "2026-07-14 v4a-A1F card: safe direction-union versus privileged shrink oracle",
        "structural_valid": structural_valid,
        "formal_pass": formal_pass,
        "sample_counts": {split: len(values) for split, values in selected_names.items()},
        "row_count": len(rows),
        "replay_mse_max_abs": replay_mse_max,
        "replay_psnr_max_abs_db": replay_psnr_max,
        "bound_excess_max": bound_excess_max,
        "support_excess_max": support_excess_max,
        "r3_review_sha256": sha256_file(Path(audit.r3_review)),
        "a0d_rows_sha256": sha256_file(Path(audit.a0d_rows)),
        "a0r_trace_manifest_sha256": sha256_file(Path(audit.a0r_trace_dir) / "trace_manifest.json"),
        "final_state_sha256": sha256_file(state_path),
        "raw_rows_cloud_only": str(raw_path),
        "operator_summary": str(summary_path),
        "bootstrap_summary": str(bootstrap_path),
        "source_manifest": str(manifest_path),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
        "candidate_selected": False,
        "parent_review_decision": review["decision"],
    }
    closeout["contract_id"] = audit.expected_route_card_sha256
    closeout_path = output / "v4a_a1f_closeout.json"
    write_json(closeout_path, closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--a0r-trace-dir", required=True)
    parser.add_argument("--a0d-rows", required=True)
    parser.add_argument("--expected-a0d-rows-sha256", required=True)
    parser.add_argument("--r3-review", required=True)
    parser.add_argument("--expected-r3-review-sha256", required=True)
    parser.add_argument("--expected-route-commit", required=True)
    parser.add_argument("--expected-route-card-sha256", required=True)
    parser.add_argument("--a1f-stage", required=True, choices=("smoke", "formal"))
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("frozen v3z arguments are required after A1F arguments")
    global SOURCE, AUDIT
    AUDIT = args
    SOURCE = a0p.load_source(Path(args.v3z_root).resolve())
    a0p.SOURCE = SOURCE
    a0p.AUDIT = args
    SOURCE.run_projected = run_a1f
    original = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *v3z_args]
        SOURCE.main()
    finally:
        sys.argv = original


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a1f_action_feasibility.py audit --v3z-root ... --a1f-stage smoke|formal ... projected ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
