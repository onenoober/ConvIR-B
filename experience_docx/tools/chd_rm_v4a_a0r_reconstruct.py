#!/usr/bin/env python3
"""Instrumented, math-preserving reconstruction of the closed v3z contract.

This tool deliberately imports the exact historical v3z source instead of
copying its model or loss code.  The only additions are cloud-only state,
per-image, and projection traces needed to decide whether v3z is
reconstructable before any new route changes the scientific contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUTE_ID = "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714"
SCHEMA_VERSION = 1
SOURCE: Any = None
TRACE_DIR: Path | None = None
REPLICATE_ID = ""
TRACE: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        value["torch_cuda_all"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return value


def load_source(v3z_root: Path) -> Any:
    source_path = v3z_root / "experience_docx" / "tools" / "chd_rm_v3x_projected_safety_constraint.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"missing exact v3z source: {source_path}")
    spec = importlib.util.spec_from_file_location("v4a_v3z_source", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import v3z source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_trace_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Any,
    epoch: int,
    update: int,
    state_kind: str,
    phase: str,
    window_names: list[str],
) -> dict[str, Any]:
    if TRACE_DIR is None or TRACE is None:
        raise RuntimeError("trace was not initialized")
    states_dir = TRACE_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    filename = f"epoch{epoch:02d}_update{update:03d}_{state_kind}.pt"
    path = states_dir / filename
    payload = {
        "schema_version": SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "replicate_id": REPLICATE_ID,
        "epoch": epoch,
        "update": update,
        "state_kind": state_kind,
        "phase": phase,
        "window_names": list(window_names),
        "model_state": cpu_copy(model.state_dict()),
        "optimizer_state": cpu_copy(optimizer.state_dict()),
        "rng_state": rng_state(),
        "config": {key: value for key, value in vars(args).items() if key not in {"device"}},
    }
    torch.save(payload, path)
    row = {
        "epoch": epoch,
        "update": update,
        "state_kind": state_kind,
        "phase": phase,
        "relative_path": str(path.relative_to(TRACE_DIR.parent)),
        "sha256": sha256_file(path),
        "window_names": list(window_names),
    }
    TRACE["states"].append(row)
    write_json(TRACE_DIR / "trace_manifest.json", TRACE)
    return row


def per_image_rows(
    *,
    args: Any,
    v3s: Any,
    legacy: Any,
    frozen: Any,
    names: list[str],
    folds: dict[str, int],
    kind: str,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    point: str,
) -> list[dict[str, Any]]:
    assert SOURCE is not None
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for name in names:
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            sample["delta_bound"] = args.delta_bound
            pieces = Path(name).stem.split("_")
            image_id = pieces[0]
            haze_1 = float(pieces[1]) if len(pieces) > 2 else math.nan
            haze_2 = float(pieces[2]) if len(pieces) > 2 else math.nan
            for operator in SOURCE.V3W.OPERATORS:
                delta = SOURCE.V3W.delta_for(kind, model, sample, operator)
                old_low, old_high, new_low, new_high = v3s.candidate_predictions(
                    sample["base"], sample["steps"][operator], delta
                )
                old_low_mse = float(v3s.per_image_mse(old_low, sample["label"]).item())
                old_high_mse = float(v3s.per_image_mse(old_high, sample["label"]).item())
                new_low_mse = float(v3s.per_image_mse(new_low, sample["label"]).item())
                new_high_mse = float(v3s.per_image_mse(new_high, sample["label"]).item())
                metrics = SOURCE.V3W.candidate_metrics(v3s, sample, delta, operator)
                anchor_signed = new_low_mse - old_low_mse
                added_signed = new_high_mse - old_high_mse
                total_signed = new_high_mse - old_low_mse
                step = sample["steps"][operator]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "split": split,
                        "point": point,
                        "name": name,
                        "clean_reference_group": image_id,
                        "haze_param_1": haze_1,
                        "haze_param_2": haze_2,
                        "operator": operator,
                        "old_125_mse": old_low_mse,
                        "old_250_mse": old_high_mse,
                        "new_125_mse": new_low_mse,
                        "new_250_mse": new_high_mse,
                        "old_125_psnr": 10.0 * math.log10(1.0 / max(old_low_mse, 1e-30)),
                        "old_250_psnr": 10.0 * math.log10(1.0 / max(old_high_mse, 1e-30)),
                        "new_125_psnr": 10.0 * math.log10(1.0 / max(new_low_mse, 1e-30)),
                        "new_250_psnr": 10.0 * math.log10(1.0 / max(new_high_mse, 1e-30)),
                        "utility_vs_old_250_mse": old_high_mse - new_high_mse,
                        "utility_vs_old_250_psnr": 10.0 * math.log10(max(old_high_mse, 1e-30) / max(new_high_mse, 1e-30)),
                        "anchor_signed": anchor_signed,
                        "anchor_burden": max(anchor_signed, 0.0),
                        "inherited_harm": max(old_high_mse - old_low_mse, 0.0),
                        "total_harm_signed": total_signed,
                        "total_harm": max(total_signed, 0.0),
                        "added_harm_signed": added_signed,
                        "added_harm": max(added_signed, 0.0),
                        "anchor_relative": anchor_signed / max(old_low_mse, 1e-30),
                        "total_harm_relative": max(total_signed, 0.0) / max(old_low_mse, 1e-30),
                        "support_fraction": float(sample["support"].mean().item()),
                        "old_step_l1": float(step.abs().mean().item()),
                        "old_step_l2": float(torch.sqrt(torch.mean(step.square())).item()),
                        "old_step_p95": float(torch.quantile(step.abs().flatten(), 0.95).item()),
                        "delta_l1": float(delta.abs().mean().item()),
                        "delta_l2": float(torch.sqrt(torch.mean(delta.square())).item()),
                        "delta_p95": float(torch.quantile(delta.abs().flatten(), 0.95).item()),
                        "repair": float(metrics["repair"].item()),
                        "margin": float(metrics["margin"].item()),
                    }
                )
    return rows


def instrumented_train_projected(args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label):
    assert SOURCE is not None
    if objective != "safety_curriculum":
        raise ValueError(f"unexpected objective: {objective}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = SOURCE.V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    append_trace_state(
        model=model,
        optimizer=optimizer,
        args=args,
        epoch=0,
        update=0,
        state_kind="initial",
        phase="initial",
        window_names=[],
    )
    midpoint = None
    midpoint_per_image: list[dict[str, Any]] = []
    history = []
    trace_rows = []
    global_update = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        updates = 0
        projected_updates = 0
        for start in range(0, len(names), args.risk_window):
            window_names = list(names[start:start + args.risk_window])
            optimizer.zero_grad(set_to_none=True)
            phase = "render_warmup" if epoch <= args.warmup_epochs else "projected_safety"
            if epoch > args.warmup_epochs:
                append_trace_state(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    update=global_update,
                    state_kind="pre",
                    phase=phase,
                    window_names=window_names,
                )
            terms = defaultdict(list)
            for name in window_names:
                sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
                sample["delta_bound"] = args.delta_bound
                for operator in SOURCE.V3W.OPERATORS:
                    values = SOURCE.V3W.candidate_metrics(
                        v3s, sample, SOURCE.V3W.delta_for(kind, model, sample, operator), operator
                    )
                    for key, value in values.items():
                        terms[key].append(value)
            means = {key: torch.stack(value).mean() for key, value in terms.items()}
            harms = torch.stack(terms["harm"])
            cvar_count = max(1, int(math.ceil(args.cvar_fraction * harms.numel())))
            cvar = torch.topk(harms, cvar_count).values.mean()
            render_grad = SOURCE.grad_of(means["render"], parameters, retain_graph=epoch > args.warmup_epochs)
            constraint_grads: list[tuple[str, list[torch.Tensor]]] = []
            if epoch <= args.warmup_epochs:
                direction, projections = render_grad, []
            else:
                constraint_grads = [
                    ("anchor", SOURCE.grad_of(means["anchor"], parameters, retain_graph=True)),
                    ("harm", SOURCE.grad_of(means["harm"], parameters, retain_graph=True)),
                    ("margin", SOURCE.grad_of(means["margin"], parameters, retain_graph=True)),
                    ("cvar", SOURCE.grad_of(cvar, parameters, retain_graph=False)),
                ]
                direction, projections = SOURCE.projected_grad(render_grad, constraint_grads)
            if any(row["dot_before"] < 0.0 for row in projections):
                projected_updates += 1
            for parameter, value in zip(parameters, direction):
                parameter.grad = value
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm).item())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite projected gradient for {label}")
            parameters_before = flatten_parameters(parameters)
            optimizer.step()
            parameters_after = flatten_parameters(parameters)
            applied_delta = parameters_after - parameters_before
            trace_row: dict[str, Any] = {
                "epoch": epoch,
                "update": global_update,
                "phase": phase,
                "window_names": window_names,
                "window_size": len(window_names),
                "render": float(means["render"].detach().item()),
                "anchor": float(means["anchor"].detach().item()),
                "harm": float(means["harm"].detach().item()),
                "margin": float(means["margin"].detach().item()),
                "cvar": float(cvar.detach().item()),
                "render_direction_norm": float(torch.sqrt(SOURCE.dot(render_grad, render_grad)).item()),
                "projected_direction_norm": float(torch.sqrt(SOURCE.dot(direction, direction)).item()),
                "gradient_norm_before_clip": gradient_norm,
                "applied_delta_norm": float(torch.linalg.vector_norm(applied_delta).item()),
                "projection_order": [row["constraint"] for row in projections],
            }
            for projection in projections:
                name = projection["constraint"]
                trace_row[f"{name}_dot_before"] = projection["dot_before"]
                trace_row[f"{name}_dot_after"] = projection["dot_after"]
            for name, gradient in constraint_grads:
                flat_gradient = torch.cat([value.reshape(-1) for value in gradient])
                trace_row[f"{name}_actual_linear_change"] = float(torch.dot(flat_gradient, applied_delta).item())
            trace_rows.append(trace_row)
            if epoch > args.warmup_epochs:
                append_trace_state(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    update=global_update,
                    state_kind="post",
                    phase=phase,
                    window_names=window_names,
                )
            for key, value in means.items():
                totals[key] += float(value.detach().item())
            totals["render_direction_norm"] += float(torch.sqrt(SOURCE.dot(render_grad, render_grad)).item())
            totals["projected_direction_norm"] += float(torch.sqrt(SOURCE.dot(direction, direction)).item())
            totals["cvar"] += float(cvar.detach().item())
            totals["gradient_norm"] += gradient_norm
            updates += 1
            global_update += 1
        row = {
            "cell": label,
            "epoch": epoch,
            "phase": "render_warmup" if epoch <= args.warmup_epochs else "projected_safety",
            "updates": updates,
            "projected_update_ratio": projected_updates / updates,
            **{key: value / updates for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps({"V4A_A0R_PROGRESS": row}, sort_keys=True), flush=True)
        if epoch == args.warmup_epochs:
            midpoint = SOURCE.V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
            midpoint_per_image = per_image_rows(
                args=args,
                v3s=v3s,
                legacy=legacy,
                frozen=frozen,
                names=names,
                folds=folds,
                kind=kind,
                model=model,
                device=device,
                split="update",
                point="warmup_end",
            )
            append_trace_state(
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                update=global_update,
                state_kind="warmup_end",
                phase="render_warmup",
                window_names=[],
            )
    final = SOURCE.V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    append_trace_state(
        model=model,
        optimizer=optimizer,
        args=args,
        epoch=args.epochs,
        update=global_update,
        state_kind="final",
        phase="projected_safety",
        window_names=[],
    )
    return initial, midpoint, final, history, trace_rows, midpoint_per_image


def instrumented_run_projected(args, v3s, legacy, frozen, names, folds, device, output_dir):
    assert SOURCE is not None and TRACE_DIR is not None and TRACE is not None
    source_path = Path(output_dir) / f"{args.run_tag}_source_manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest["mode"] = "projected"
    source_manifest["objective"] = {
        "warmup_epochs": args.warmup_epochs,
        "render_objective": "MSE",
        "constraints": ["anchor", "harm", "margin", "CVaR25"],
    }
    source_manifest["v4a_instrumentation"] = {
        "schema_version": SCHEMA_VERSION,
        "replicate_id": REPLICATE_ID,
        "trace_dir": str(TRACE_DIR),
        "source_file_sha256": sha256_file(Path(SOURCE.__file__)),
    }
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    heldout_names = all_names[args.sample_count:args.sample_count * 2]
    if len(heldout_names) != args.sample_count or set(heldout_names) & set(names):
        raise RuntimeError("v3z requires a disjoint fixed heldout sample")
    source_manifest["heldout_names"] = list(heldout_names)
    SOURCE.V3W.write_json(source_path, source_manifest)
    TRACE.update(
        {
            "route_id": ROUTE_ID,
            "schema_version": SCHEMA_VERSION,
            "replicate_id": REPLICATE_ID,
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "config_sha256": canonical_json_hash(vars(args)),
            "v3z_source": str(Path(SOURCE.__file__)),
            "v3z_source_sha256": sha256_file(Path(SOURCE.__file__)),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "states": [],
        }
    )
    models = SOURCE.V3W.import_v3w_models()
    first = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = SOURCE.V3W.build_cells(models, first, args, device)
    label, (kind, objective, model) = next(iter(cells.items()))
    heldout_initial = SOURCE.V3W.evaluate_cell(args, v3s, legacy, frozen, heldout_names, folds, kind, model, device)
    initial_rows = per_image_rows(
        args=args, v3s=v3s, legacy=legacy, frozen=frozen, names=names, folds=folds, kind=kind,
        model=model, device=device, split="update", point="initial"
    )
    heldout_initial_rows = per_image_rows(
        args=args, v3s=v3s, legacy=legacy, frozen=frozen, names=heldout_names, folds=folds, kind=kind,
        model=model, device=device, split="heldout", point="initial"
    )
    initial, midpoint, final, history, trace_rows, midpoint_rows = instrumented_train_projected(
        args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label
    )
    final_rows = per_image_rows(
        args=args, v3s=v3s, legacy=legacy, frozen=frozen, names=names, folds=folds, kind=kind,
        model=model, device=device, split="update", point="final"
    )
    heldout_final = SOURCE.V3W.evaluate_cell(args, v3s, legacy, frozen, heldout_names, folds, kind, model, device)
    heldout_final_rows = per_image_rows(
        args=args, v3s=v3s, legacy=legacy, frozen=frozen, names=heldout_names, folds=folds, kind=kind,
        model=model, device=device, split="heldout", point="final"
    )
    reduction = (initial["render"] - final["render"]) / max(initial["render"], 1e-30)
    midpoint_reduction = (initial["render"] - midpoint["render"]) / max(initial["render"], 1e-30)
    heldout_reduction = (heldout_initial["render"] - heldout_final["render"]) / max(heldout_initial["render"], 1e-30)
    historical_contract = {
        "cells": {
            label: {
                "initial": initial,
                "midpoint": midpoint,
                "final": final,
                "relative_render_reduction": reduction,
                "midpoint_relative_render_reduction": midpoint_reduction,
                "heldout_initial": heldout_initial,
                "heldout_final": heldout_final,
                "heldout_relative_render_reduction": heldout_reduction,
                "parameter_count": sum(value.numel() for value in model.parameters()),
            }
        }
    }
    SOURCE.V3W.write_rows(Path(output_dir) / f"{args.run_tag}_history.csv", history)
    SOURCE.V3W.write_rows(Path(output_dir) / f"{args.run_tag}_per_image.csv", initial_rows + heldout_initial_rows + midpoint_rows + final_rows + heldout_final_rows)
    SOURCE.V3W.write_rows(Path(output_dir) / f"{args.run_tag}_projection_trace.csv", trace_rows)
    TRACE["state_count"] = len(TRACE["states"])
    TRACE["projection_trace_rows"] = len(trace_rows)
    TRACE["per_image_rows"] = len(initial_rows + heldout_initial_rows + midpoint_rows + final_rows + heldout_final_rows)
    trace_path = TRACE_DIR / "trace_manifest.json"
    write_json(trace_path, TRACE)
    reconstruction = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v4a-A0R-instrumented-reconstruction-replicate",
        "evidence_role": "engineering_debug",
        "state": "COMPLETED_INCONCLUSIVE",
        "gate_type": "numerical_equivalence_pending_paired_comparison",
        "decision": "A0R_REPLICATE_COMPLETE_AWAITING_PAIRED_COMPARISON",
        "authorizes": "v4a-A0R paired comparison only",
        "metric_contract": "historical v3z aggregate contract plus v4a state and per-image retention",
        "historical_contract": historical_contract,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": True,
        "trace_manifest": str(trace_path),
        "trace_manifest_sha256": sha256_file(trace_path),
    }
    SOURCE.V3W.write_json(Path(output_dir) / f"{args.run_tag}_reconstruction.json", reconstruction)
    return reconstruction


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(numeric_leaves(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append({key: float(value) for key, value in row.items() if value not in {None, ""} and key not in {"cell", "phase"}})
    return parsed


def max_pair_difference(left: dict[str, float], right: dict[str, float]) -> tuple[float, str]:
    keys = sorted(set(left) | set(right))
    maximum = -1.0
    maximum_key = ""
    for key in keys:
        if key not in left or key not in right:
            return math.inf, f"missing:{key}"
        difference = abs(left[key] - right[key])
        if difference > maximum:
            maximum = difference
            maximum_key = key
    return maximum, maximum_key


def summarize(args: argparse.Namespace) -> None:
    r1 = Path(args.r1_dir)
    r2 = Path(args.r2_dir)
    historical_closeout = json.loads(Path(args.historical_closeout).read_text(encoding="utf-8"))
    r1_reconstruction = json.loads(next(r1.glob("*_reconstruction.json")).read_text(encoding="utf-8"))
    r2_reconstruction = json.loads(next(r2.glob("*_reconstruction.json")).read_text(encoding="utf-8"))
    historical_values = numeric_leaves({"cells": historical_closeout["cells"]})
    r1_values = numeric_leaves(r1_reconstruction["historical_contract"])
    r2_values = numeric_leaves(r2_reconstruction["historical_contract"])
    historical_r1, historical_r1_key = max_pair_difference(historical_values, r1_values)
    historical_r2, historical_r2_key = max_pair_difference(historical_values, r2_values)
    r1_r2, r1_r2_key = max_pair_difference(r1_values, r2_values)
    historical_history = read_history(Path(args.historical_history))
    r1_history = read_history(next(r1.glob("*_history.csv")))
    r2_history = read_history(next(r2.glob("*_history.csv")))
    if len(historical_history) != len(r1_history) or len(r1_history) != len(r2_history):
        raise RuntimeError("history row count mismatch")
    history_diffs = []
    for index, (historical_row, r1_row, r2_row) in enumerate(zip(historical_history, r1_history, r2_history), start=1):
        history_diffs.append({
            "epoch": index,
            "historical_r1": max_pair_difference(historical_row, r1_row)[0],
            "historical_r2": max_pair_difference(historical_row, r2_row)[0],
            "r1_r2": max_pair_difference(r1_row, r2_row)[0],
        })
    max_history_historical_r1 = max(row["historical_r1"] for row in history_diffs)
    max_history_historical_r2 = max(row["historical_r2"] for row in history_diffs)
    max_history_r1_r2 = max(row["r1_r2"] for row in history_diffs)
    tolerance = max(1e-9, 5.0 * max(r1_r2, max_history_r1_r2))
    noop = json.loads(Path(args.noop_closeout).read_text(encoding="utf-8"))
    noop_exact = all(float(noop[key]) == 0.0 for key in ("max_abs_delta", "max_abs_prediction_diff", "max_abs_reference_psnr_delta_diff_db"))
    trace_counts = []
    for replicate_dir in (r1, r2):
        trace = json.loads(next((replicate_dir / "trace").glob("trace_manifest.json")).read_text(encoding="utf-8"))
        trace_counts.append({
            "replicate": trace["replicate_id"],
            "state_count": trace["state_count"],
            "projection_trace_rows": trace["projection_trace_rows"],
            "per_image_rows": trace["per_image_rows"],
            "trace_sha256": sha256_file(replicate_dir / "trace" / "trace_manifest.json"),
        })
    trace_complete = all(item["state_count"] == 515 and item["projection_trace_rows"] == 512 and item["per_image_rows"] == 1280 for item in trace_counts)
    reproduction_pass = (
        noop_exact
        and trace_complete
        and historical_r1 <= tolerance
        and historical_r2 <= tolerance
        and max_history_historical_r1 <= tolerance
        and max_history_historical_r2 <= tolerance
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "route_id": ROUTE_ID,
        "stage": "v4a-A0R-instrumented-reconstruction",
        "evidence_role": "engineering_debug",
        "historical_closeout": str(args.historical_closeout),
        "historical_history": str(args.historical_history),
        "tolerance": tolerance,
        "noop_exact": noop_exact,
        "contract_max_abs": {
            "historical_r1": historical_r1,
            "historical_r1_key": historical_r1_key,
            "historical_r2": historical_r2,
            "historical_r2_key": historical_r2_key,
            "r1_r2": r1_r2,
            "r1_r2_key": r1_r2_key,
        },
        "history_max_abs": {
            "historical_r1": max_history_historical_r1,
            "historical_r2": max_history_historical_r2,
            "r1_r2": max_history_r1_r2,
        },
        "trace_counts": trace_counts,
        "reproduction_pass": reproduction_pass,
    }
    decision = "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P" if reproduction_pass else "V4A_A0R_REPRODUCTION_FAIL_STOP"
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": "v4a_a0r_reconstruction",
        "stage": "v4a-A0R-instrumented-reconstruction",
        "evidence_role": "engineering_debug",
        "contract_id": canonical_json_hash(summary),
        "state": "COMPLETED_GATE_PASS" if reproduction_pass else "COMPLETED_GATE_FAIL",
        "gate_type": "numerical_equivalence",
        "decision": decision,
        "metric_contract": "2026-07-14 v4a card A0R gate",
        "authorizes": "A0D_AND_A0P_ONLY" if reproduction_pass else "none",
        "reason": "two reconstructed trajectories, original observable aggregates, no-op fields, and retained-state counts were compared under the frozen numerical contract",
        "locked_test_touched": False,
        "canary_touched": False,
        "summary": summary,
    }
    write_json(output_dir / "v4a_a0r_reproduction_summary.json", summary)
    write_json(output_dir / "v4a_a0r_closeout.json", closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)


def reconstruct(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--replicate-id", required=True)
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("v3z replay arguments are required after v4a arguments")
    global SOURCE, TRACE_DIR, REPLICATE_ID, TRACE
    SOURCE = load_source(Path(args.v3z_root).resolve())
    TRACE_DIR = Path(args.trace_dir).resolve()
    REPLICATE_ID = args.replicate_id
    TRACE = {}
    SOURCE.train_projected = instrumented_train_projected
    SOURCE.run_projected = instrumented_run_projected
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *v3z_args]
        SOURCE.main()
    finally:
        sys.argv = original_argv


def parse_summary_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", required=True)
    parser.add_argument("--r2-dir", required=True)
    parser.add_argument("--noop-closeout", required=True)
    parser.add_argument("--historical-closeout", required=True)
    parser.add_argument("--historical-history", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: chd_rm_v4a_a0r_reconstruct.py reconstruct|summarize ...")
    command = sys.argv[1]
    if command == "reconstruct":
        reconstruct(sys.argv[2:])
    elif command == "summarize":
        summarize(parse_summary_args(sys.argv[2:]))
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
