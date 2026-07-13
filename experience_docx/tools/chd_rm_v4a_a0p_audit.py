#!/usr/bin/env python3
"""Frozen A0P paired shadow audit over the retained A0R r1 pre-update states.

The runner imports the immutable v3z implementation for every model, metric,
and historical projection call.  This file only restores A0R states, applies
the frozen A0P method/window factorial, retains cloud-only raw rows, and emits
the pre-registered compact bootstrap closeout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from chd_rm_v4a_a0p_contract import (
    BOOTSTRAP_REPLICATES,
    CONSTRAINT_ORDER,
    METHODS,
    UTILITY_MARGIN_DB,
    WINDOWS,
    actual_proposal_projection,
    bootstrap_bounds,
    bootstrap_indices,
    classify_a0p,
    clip_vector,
    exact_gradient_intersection,
    numerical_tolerance,
    prestratified32,
    shuffled16,
)


ROUTE_ID = "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714"
EXPECTED_A0R = {
    "route_id": ROUTE_ID,
    "state": "COMPLETED_GATE_PASS",
    "decision": "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P",
    "authorizes": "A0D_AND_A0P_ONLY",
}
EXPECTED_ARGS = {
    "sample_count": 128,
    "epochs": 16,
    "risk_window": 4,
    "warmup_epochs": 8,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "grad_clip_norm": 0.1,
    "cvar_fraction": 0.25,
}
METRIC_KEYS = ("anchor", "harm", "margin", "render_mse", "psnr", "severe", "hard")
ENDPOINTS = (
    "anchor",
    "harm",
    "margin",
    "cvar25",
    "cvar10",
    "severe_rate",
    "hard_rate",
    "render_mse",
    "mean_psnr",
    "p05_psnr",
)
SAFETY_ENDPOINTS = ENDPOINTS[:8]
UTILITY_ENDPOINTS = ENDPOINTS[8:]
RAW_FIELDS = (
    "schema_version",
    "record_kind",
    "state_index",
    "state_sha256",
    "epoch",
    "update",
    "method",
    "window",
    "name",
    "operator",
    "anchor",
    "harm",
    "margin",
    "render_mse",
    "psnr",
    "old_high_mse",
    "delta_psnr",
    "severe",
    "hard",
)

SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_source(v3z_root: Path) -> Any:
    source_path = v3z_root / "experience_docx" / "tools" / "chd_rm_v3x_projected_safety_constraint.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"missing immutable v3z source: {source_path}")
    spec = importlib.util.spec_from_file_location("v4a_a0p_v3z", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import immutable v3z source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_rng(payload: dict[str, Any]) -> None:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        raise RuntimeError("retained state has no RNG payload")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_states = state.get("torch_cuda_all")
        if not isinstance(cuda_states, list):
            raise RuntimeError("retained CUDA state is missing")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def flatten_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).to(dtype=torch.float64) for parameter in parameters])


def assign_parameters(parameters: list[torch.nn.Parameter], vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            value = vector[offset:offset + count].reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(value)
            offset += count
    if offset != vector.numel():
        raise RuntimeError("parameter vector length mismatch")


def set_gradients(parameters: list[torch.nn.Parameter], vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = vector[offset:offset + count].reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype).clone()
        offset += count
    if offset != vector.numel():
        raise RuntimeError("gradient vector length mismatch")


def flatten_gradient(gradient: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([item.detach().reshape(-1).to(dtype=torch.float64) for item in gradient])


def validate_frozen_args(args: Any) -> None:
    for name, expected in EXPECTED_ARGS.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0):
                raise RuntimeError(f"A0P frozen argument mismatch: {name}={actual!r}")
        elif actual != expected:
            raise RuntimeError(f"A0P frozen argument mismatch: {name}={actual!r}")


def validate_adamw(optimizer: torch.optim.Optimizer) -> None:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("A0P requires exactly one AdamW parameter group")
    group = optimizer.param_groups[0]
    expected = {"lr": 5e-4, "weight_decay": 1e-5, "eps": 1e-8}
    for name, value in expected.items():
        if not math.isclose(float(group[name]), value, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(f"retained AdamW {name} mismatch")
    if tuple(group["betas"]) != (0.9, 0.999) or bool(group["amsgrad"]):
        raise RuntimeError("retained AdamW beta or amsgrad mismatch")


def metric_psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scalar(value: torch.Tensor) -> float:
    result = float(value.detach().item())
    if not math.isfinite(result):
        raise FloatingPointError("non-finite rendered metric")
    return result


def cvar(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise FloatingPointError("non-finite or empty CVaR values")
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.mean(np.partition(flat, flat.size - count)[-count:]))


def endpoint_values(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "anchor": float(np.mean(metrics["anchor"])),
        "harm": float(np.mean(metrics["harm"])),
        "margin": float(np.mean(metrics["margin"])),
        "cvar25": cvar(metrics["harm"], 0.25),
        "cvar10": cvar(metrics["harm"], 0.10),
        "severe_rate": float(np.mean(metrics["severe"])),
        "hard_rate": float(np.mean(metrics["hard"])),
        "render_mse": float(np.mean(metrics["render_mse"])),
        "mean_psnr": float(np.mean(metrics["psnr"])),
        "p05_psnr": float(np.quantile(metrics["psnr"], 0.05, method="linear")),
    }


def load_cell(
    payload: dict[str, Any], args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device
) -> tuple[str, torch.nn.Module, torch.optim.Optimizer, list[torch.nn.Parameter]]:
    assert SOURCE is not None
    models = SOURCE.V3W.import_v3w_models()
    first = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = SOURCE.V3W.build_cells(models, first, args, device)
    _, (kind, objective, model) = next(iter(cells.items()))
    if kind != "output" or objective != "safety_curriculum":
        raise RuntimeError("A0P did not reconstruct the frozen output safety cell")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.load_state_dict(payload["optimizer_state"])
    validate_adamw(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("A0P restored no trainable parameters")
    restore_rng(payload)
    return kind, model, optimizer, parameters


def construction_terms(
    args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], kind: str,
    model: torch.nn.Module, device: torch.device, parameters: list[torch.nn.Parameter]
) -> tuple[dict[str, float], torch.Tensor, dict[str, torch.Tensor]]:
    assert SOURCE is not None
    model.train()
    terms: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name in names:
        sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
        sample["delta_bound"] = args.delta_bound
        for operator in SOURCE.V3W.OPERATORS:
            values = SOURCE.V3W.candidate_metrics(v3s, sample, SOURCE.V3W.delta_for(kind, model, sample, operator), operator)
            for key, value in values.items():
                terms[key].append(value)
    means = {key: torch.stack(values).mean() for key, values in terms.items()}
    harms = torch.stack(terms["harm"])
    count = max(1, int(math.ceil(args.cvar_fraction * harms.numel())))
    cvar25 = torch.topk(harms, count).values.mean()
    render_gradient = flatten_gradient(SOURCE.grad_of(means["render"], parameters, retain_graph=True))
    constraints = {
        "anchor": flatten_gradient(SOURCE.grad_of(means["anchor"], parameters, retain_graph=True)),
        "harm": flatten_gradient(SOURCE.grad_of(means["harm"], parameters, retain_graph=True)),
        "margin": flatten_gradient(SOURCE.grad_of(means["margin"], parameters, retain_graph=True)),
        "cvar": flatten_gradient(SOURCE.grad_of(cvar25, parameters, retain_graph=False)),
    }
    pre = {key: scalar(value) for key, value in means.items()}
    pre["cvar"] = scalar(cvar25)
    return pre, render_gradient, constraints


def construction_render_metrics(
    args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], kind: str,
    model: torch.nn.Module, device: torch.device
) -> dict[str, float]:
    assert SOURCE is not None
    model.eval()
    terms: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for name in names:
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            sample["delta_bound"] = args.delta_bound
            for operator in SOURCE.V3W.OPERATORS:
                values = SOURCE.V3W.candidate_metrics(v3s, sample, SOURCE.V3W.delta_for(kind, model, sample, operator), operator)
                for key, value in values.items():
                    terms[key].append(scalar(value))
    result = {key: float(np.mean(values)) for key, values in terms.items()}
    result["cvar"] = cvar(np.asarray(terms["harm"], dtype=np.float64), args.cvar_fraction)
    return result


def render_population(
    args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], kind: str,
    model: torch.nn.Module, device: torch.device
) -> dict[str, np.ndarray]:
    assert SOURCE is not None
    operator_count = len(SOURCE.V3W.OPERATORS)
    values = {key: np.full((len(names), operator_count), np.nan, dtype=np.float64) for key in (*METRIC_KEYS, "old_high_mse", "delta_psnr")}
    model.eval()
    with torch.no_grad():
        for image_index, name in enumerate(names):
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            sample["delta_bound"] = args.delta_bound
            for operator_index, operator in enumerate(SOURCE.V3W.OPERATORS):
                delta = SOURCE.V3W.delta_for(kind, model, sample, operator)
                base_metrics = SOURCE.V3W.candidate_metrics(v3s, sample, delta, operator)
                old_low, old_high, _, new_high = v3s.candidate_predictions(sample["base"], sample["steps"][operator], delta)
                old_high_mse = scalar(v3s.per_image_mse(old_high, sample["label"]))
                new_high_mse = scalar(v3s.per_image_mse(new_high, sample["label"]))
                values["anchor"][image_index, operator_index] = scalar(base_metrics["anchor"])
                values["harm"][image_index, operator_index] = scalar(base_metrics["harm"])
                values["margin"][image_index, operator_index] = scalar(base_metrics["margin"])
                values["render_mse"][image_index, operator_index] = scalar(base_metrics["render"])
                values["psnr"][image_index, operator_index] = metric_psnr(new_high_mse)
                values["old_high_mse"][image_index, operator_index] = old_high_mse
                values["delta_psnr"][image_index, operator_index] = metric_psnr(new_high_mse) - metric_psnr(old_high_mse)
                values["severe"][image_index, operator_index] = float(values["delta_psnr"][image_index, operator_index] <= -0.2)
                values["hard"][image_index, operator_index] = float(values["delta_psnr"][image_index, operator_index] <= -0.5)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("non-finite heldout A0P render")
    return values


def inherited_ratios(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device) -> dict[str, float]:
    assert SOURCE is not None
    result: dict[str, float] = {}
    with torch.no_grad():
        for name in names:
            sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
            zero = torch.zeros_like(sample["base"])
            ratios = []
            for operator in SOURCE.V3W.OPERATORS:
                old_low, old_high, _, _ = v3s.candidate_predictions(sample["base"], sample["steps"][operator], zero)
                old_low_mse = scalar(v3s.per_image_mse(old_low, sample["label"]))
                old_high_mse = scalar(v3s.per_image_mse(old_high, sample["label"]))
                ratios.append(max(old_high_mse - old_low_mse, 0.0) / max(old_low_mse, 1e-30))
            result[name] = max(ratios)
    if len(result) != 128 or not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("invalid pre-intervention inherited-harm ratios")
    return result


def restore_model_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, payload: dict[str, Any]) -> None:
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    validate_adamw(optimizer)
    restore_rng(payload)


def actual_adamw_proposal(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter], render_gradient: torch.Tensor
) -> tuple[torch.Tensor, float]:
    before = flatten_parameters(parameters)
    optimizer.zero_grad(set_to_none=True)
    set_gradients(parameters, render_gradient)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 0.1).item())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("non-finite raw AdamW proposal gradient")
    optimizer.step()
    return flatten_parameters(parameters) - before, gradient_norm


def nonlinear_nonworse(pre: dict[str, float], post: dict[str, float]) -> bool:
    return all(post[key] <= pre[key] + numerical_tolerance(pre[key], post[key]) for key in CONSTRAINT_ORDER)


def run_factor_cell(
    *, payload: dict[str, Any], args: Any, v3s: Any, legacy: Any, frozen: Any, all_update_names: list[str], heldout_names: list[str],
    folds: dict[str, int], device: torch.device, method: str, window: str, window_names: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    label, model, optimizer, parameters = load_cell(payload, args, v3s, legacy, frozen, all_update_names, folds, device)
    pre_construction, render_gradient, constraint_gradients = construction_terms(
        args, v3s, legacy, frozen, window_names, folds, label, model, device, parameters
    )
    ordered_constraints = [constraint_gradients[name] for name in CONSTRAINT_ORDER]
    theta = flatten_parameters(parameters)
    diagnostics: dict[str, Any] = {
        "method": method,
        "window": window,
        "window_size": len(window_names),
        "solver_valid": True,
        "active_set": "legacy_sequential",
        "solver_objective": 0.0,
        "solver_primal_residual": 0.0,
        "solver_dual_residual": 0.0,
        "gradient_norm_before_clip": math.nan,
        "beta": 1.0,
        "backtracking_null": False,
        "post_construction_checked": False,
    }
    if method == "historical_sequential_gradient":
        pairs = [
            (name, [item.to(dtype=parameters[0].dtype) for item in _split_vector(parameters, constraint_gradients[name])])
            for name in CONSTRAINT_ORDER
        ]
        historical_gradient, projections = SOURCE.projected_grad(
            [item.to(dtype=parameters[0].dtype) for item in _split_vector(parameters, render_gradient)], pairs
        )
        direction = flatten_gradient(historical_gradient)
        optimizer.zero_grad(set_to_none=True)
        set_gradients(parameters, direction)
        diagnostics["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_(parameters, 0.1).item())
        if not math.isfinite(diagnostics["gradient_norm_before_clip"]):
            raise FloatingPointError("non-finite historical projected gradient")
        optimizer.step()
        diagnostics["projection_order"] = ",".join(str(row["constraint"]) for row in projections)
    elif method == "exact_gradient_intersection":
        projection = exact_gradient_intersection(render_gradient, ordered_constraints)
        diagnostics.update({
            "solver_valid": projection.valid,
            "active_set": ",".join(str(value) for value in projection.active_set),
            "solver_objective": projection.objective,
            "solver_primal_residual": projection.primal_residual,
            "solver_dual_residual": projection.dual_residual,
        })
        if not projection.valid:
            return {}, diagnostics
        direction = clip_vector(projection.vector)
        optimizer.zero_grad(set_to_none=True)
        set_gradients(parameters, direction)
        diagnostics["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_(parameters, 0.1).item())
        if not math.isfinite(diagnostics["gradient_norm_before_clip"]):
            raise FloatingPointError("non-finite exact-intersection gradient")
        optimizer.step()
    elif method == "actual_proposal_projection_with_backtracking":
        proposal, raw_norm = actual_adamw_proposal(model, optimizer, parameters, render_gradient)
        diagnostics["gradient_norm_before_clip"] = raw_norm
        restore_model_optimizer(model, optimizer, payload)
        projection = actual_proposal_projection(proposal, ordered_constraints)
        diagnostics.update({
            "solver_valid": projection.valid,
            "active_set": ",".join(str(value) for value in projection.active_set),
            "solver_objective": projection.objective,
            "solver_primal_residual": projection.primal_residual,
            "solver_dual_residual": projection.dual_residual,
            "post_construction_checked": True,
            "proposal_norm": float(torch.linalg.vector_norm(proposal).item()),
        })
        if not projection.valid:
            return {}, diagnostics
        pre_render = construction_render_metrics(args, v3s, legacy, frozen, window_names, folds, label, model, device)
        chosen_beta: float | None = None
        for exponent in range(11):
            beta = 2.0 ** (-exponent)
            assign_parameters(parameters, theta + beta * projection.vector)
            post_render = construction_render_metrics(args, v3s, legacy, frozen, window_names, folds, label, model, device)
            if nonlinear_nonworse(pre_render, post_render):
                chosen_beta = beta
                diagnostics["post_construction"] = post_render
                break
        if chosen_beta is None:
            assign_parameters(parameters, theta)
            diagnostics["beta"] = 0.0
            diagnostics["backtracking_null"] = True
            diagnostics["post_construction"] = pre_render
        else:
            diagnostics["beta"] = chosen_beta
    else:
        raise RuntimeError(f"unknown frozen A0P method: {method}")
    applied = flatten_parameters(parameters) - theta
    diagnostics["applied_delta_norm"] = float(torch.linalg.vector_norm(applied).item())
    diagnostics["constraint_dots_applied"] = {
        name: float(torch.dot(constraint_gradients[name], applied).item()) for name in CONSTRAINT_ORDER
    }
    if not all(math.isfinite(value) for value in diagnostics["constraint_dots_applied"].values()):
        raise FloatingPointError("non-finite applied constraint dot")
    return render_population(args, v3s, legacy, frozen, heldout_names, folds, label, model, device), diagnostics


def _split_vector(parameters: list[torch.nn.Parameter], vector: torch.Tensor) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        values.append(vector[offset:offset + count].reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype))
        offset += count
    if offset != vector.numel():
        raise RuntimeError("historical gradient vector length mismatch")
    return values


def validate_a0r(trace_dir: Path, closeout_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_A0R.items():
        if closeout.get(key) != expected:
            raise RuntimeError(f"A0R typed tuple mismatch: {key}={closeout.get(key)!r}")
    trace_path = trace_dir / "trace_manifest.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("replicate_id") != "r1" or trace.get("route_id") != ROUTE_ID:
        raise RuntimeError("A0P requires canonical A0R r1 trace")
    source_manifest_path = Path(str(trace["source_manifest"]))
    if not source_manifest_path.is_file() or sha256_file(source_manifest_path) != trace.get("source_manifest_sha256"):
        raise RuntimeError("A0R source manifest is missing or hash-mismatched")
    if sha256_file(Path(SOURCE.__file__)) != trace.get("v3z_source_sha256"):
        raise RuntimeError("immutable v3z source hash differs from A0R")
    pre_states = [row for row in trace.get("states", []) if row.get("state_kind") == "pre"]
    pre_states.sort(key=lambda row: (int(row["epoch"]), int(row["update"])))
    if len(pre_states) != 256:
        raise RuntimeError(f"A0P requires 256 retained pre states, found {len(pre_states)}")
    expected_epochs = set(range(9, 17))
    if {int(row["epoch"]) for row in pre_states} != expected_epochs:
        raise RuntimeError("A0P retained pre-state epochs are not exactly 9-16")
    if len({(int(row["epoch"]), int(row["update"])) for row in pre_states}) != 256:
        raise RuntimeError("A0P retained pre-state identity is not unique")
    for row in pre_states:
        relative = Path(str(row["relative_path"]))
        path = trace_dir.parent / relative
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"A0R retained state hash mismatch: {path}")
        if len(row.get("window_names", [])) != 4:
            raise RuntimeError("A0R fixed4 window is missing or has the wrong cardinality")
        row["path"] = str(path)
    return pre_states, trace, json.loads(source_manifest_path.read_text(encoding="utf-8"))


def verify_source_identity(a0r_source: dict[str, Any], current_source: dict[str, Any], heldout_names: list[str]) -> None:
    for key in ("v3s_commit", "v3p_commit", "assets", "frozen_operator_artifacts", "delta_bound", "names"):
        if a0r_source.get(key) != current_source.get(key):
            raise RuntimeError(f"A0R/current source identity mismatch: {key}")
    if a0r_source.get("heldout_names") != heldout_names:
        raise RuntimeError("A0R heldout source identity mismatch")


def write_raw_rows(
    writer: csv.DictWriter[str], *, record_kind: str, state_index: int, state: dict[str, Any], method: str, window: str,
    names: list[str], population: dict[str, np.ndarray]
) -> int:
    assert SOURCE is not None
    count = 0
    for image_index, name in enumerate(names):
        for operator_index, operator in enumerate(SOURCE.V3W.OPERATORS):
            writer.writerow({
                "schema_version": 1,
                "record_kind": record_kind,
                "state_index": state_index,
                "state_sha256": state["sha256"],
                "epoch": state["epoch"],
                "update": state["update"],
                "method": method,
                "window": window,
                "name": name,
                "operator": operator,
                **{key: float(population[key][image_index, operator_index]) for key in METRIC_KEYS},
                "old_high_mse": float(population["old_high_mse"][image_index, operator_index]),
                "delta_psnr": float(population["delta_psnr"][image_index, operator_index]),
            })
            count += 1
    return count


def effect_specs() -> list[dict[str, Any]]:
    assert SOURCE is not None
    specs: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        for window_index, window in enumerate(WINDOWS):
            for operator in SOURCE.V3W.OPERATORS:
                for endpoint in ENDPOINTS:
                    specs.append({"kind": "absolute", "method_index": method_index, "window_index": window_index, "method": method, "window": window, "operator": operator, "endpoint": endpoint})
    for method_index, method in enumerate(METHODS[1:], start=1):
        for window_index, window in enumerate(WINDOWS):
            for operator in SOURCE.V3W.OPERATORS:
                for endpoint in ENDPOINTS:
                    specs.append({"kind": "vs_historical", "method_index": method_index, "window_index": window_index, "method": method, "window": window, "operator": operator, "endpoint": endpoint})
    for method_index, method in enumerate(METHODS[1:], start=1):
        for window_index, window in enumerate(WINDOWS[1:], start=1):
            for operator in SOURCE.V3W.OPERATORS:
                for endpoint in ENDPOINTS:
                    specs.append({"kind": "method_window_did", "method_index": method_index, "window_index": window_index, "method": method, "window": window, "operator": operator, "endpoint": endpoint})
    for index, spec in enumerate(specs):
        spec["key"] = f"{index:04d}:{spec['kind']}:{spec['method']}:{spec['window']}:{spec['operator']}:{spec['endpoint']}"
    return specs


def sampled_statistics(
    post: dict[str, np.ndarray], pre: dict[str, np.ndarray], state_indices: np.ndarray, image_indices: np.ndarray
) -> tuple[dict[tuple[int, int, int], dict[str, float]], dict[int, dict[str, float]]]:
    assert SOURCE is not None
    cells: dict[tuple[int, int, int], dict[str, float]] = {}
    baseline: dict[int, dict[str, float]] = {}
    state_grid = state_indices[:, None]
    image_grid = image_indices[None, :]
    for operator_index in range(len(SOURCE.V3W.OPERATORS)):
        baseline[operator_index] = endpoint_values({
            key: pre[key][state_grid, image_grid, operator_index] for key in METRIC_KEYS
        })
        for method_index in range(len(METHODS)):
            for window_index in range(len(WINDOWS)):
                cells[(method_index, window_index, operator_index)] = endpoint_values({
                    key: post[key][state_grid, method_index, window_index, image_grid, operator_index] for key in METRIC_KEYS
                })
    return cells, baseline


def spec_value(spec: dict[str, Any], cells: dict[tuple[int, int, int], dict[str, float]], baseline: dict[int, dict[str, float]]) -> float:
    method = int(spec["method_index"])
    window = int(spec["window_index"])
    operator = list(SOURCE.V3W.OPERATORS).index(spec["operator"])
    endpoint = str(spec["endpoint"])
    value = cells[(method, window, operator)][endpoint]
    if spec["kind"] == "absolute":
        return value - baseline[operator][endpoint]
    historical = cells[(0, window, operator)][endpoint]
    if spec["kind"] == "vs_historical":
        return value - historical
    fixed = cells[(method, 0, operator)][endpoint] - cells[(0, 0, operator)][endpoint]
    return (value - historical) - fixed


def bootstrap_summary(post: dict[str, np.ndarray], pre: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    specs = effect_specs()
    state_count, _, _, image_count, _ = post["harm"].shape
    all_states = np.arange(state_count, dtype=np.int64)
    all_images = np.arange(image_count, dtype=np.int64)
    point_cells, point_baseline = sampled_statistics(post, pre, all_states, all_images)
    point = np.asarray([spec_value(spec, point_cells, point_baseline) for spec in specs], dtype=np.float64)
    draws = np.empty((len(specs), BOOTSTRAP_REPLICATES), dtype=np.float64)
    for draw_index, (state_indices, image_indices) in enumerate(bootstrap_indices(state_count, image_count)):
        cells, baseline = sampled_statistics(post, pre, state_indices, image_indices)
        draws[:, draw_index] = [spec_value(spec, cells, baseline) for spec in specs]
    if not np.isfinite(draws).all():
        raise FloatingPointError("non-finite paired bootstrap draw")
    standard_error = np.std(draws, axis=1, ddof=1)
    scale = np.where(standard_error > 1e-30, standard_error, 1.0)
    max_abs = np.max(np.abs((draws - point[:, None]) / scale[:, None]), axis=0)
    critical = bootstrap_bounds(max_abs)[1]
    rows: list[dict[str, Any]] = []
    intervals: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(specs):
        half_width = 0.0 if standard_error[index] <= 1e-30 else float(critical * standard_error[index])
        interval = {"estimate": float(point[index]), "lower": float(point[index] - half_width), "upper": float(point[index] + half_width)}
        intervals[spec["key"]] = interval
        rows.append({
            "schema_version": 1,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": 3407,
            "simultaneous_max_statistic": True,
            "max_statistic_critical": float(critical),
            "standard_error": float(standard_error[index]),
            **spec,
            **interval,
        })
    return rows, intervals


def lookup_interval(intervals: dict[str, dict[str, float]], kind: str, method: str, window: str, operator: str, endpoint: str) -> dict[str, float]:
    matches = [value for key, value in intervals.items() if key.split(":", 2)[1] == kind and key.endswith(f":{method}:{window}:{operator}:{endpoint}")]
    if len(matches) != 1:
        raise RuntimeError(f"missing bootstrap interval: {kind}/{method}/{window}/{operator}/{endpoint}")
    return matches[0]


def cell_positive(method: str, window: str, intervals: dict[str, dict[str, float]]) -> bool:
    assert SOURCE is not None
    for operator in SOURCE.V3W.OPERATORS:
        for endpoint in SAFETY_ENDPOINTS:
            absolute = lookup_interval(intervals, "absolute", method, window, operator, endpoint)
            comparative = lookup_interval(intervals, "vs_historical", method, window, operator, endpoint)
            if absolute["upper"] > numerical_tolerance(0.0, absolute["estimate"]):
                return False
            if comparative["upper"] > numerical_tolerance(0.0, comparative["estimate"]):
                return False
        for endpoint in UTILITY_ENDPOINTS:
            absolute = lookup_interval(intervals, "absolute", method, window, operator, endpoint)
            comparative = lookup_interval(intervals, "vs_historical", method, window, operator, endpoint)
            if absolute["lower"] < UTILITY_MARGIN_DB or comparative["lower"] < UTILITY_MARGIN_DB:
                return False
    return any(
        lookup_interval(intervals, "vs_historical", method, window, operator, endpoint)["upper"]
        < -numerical_tolerance(0.0, lookup_interval(intervals, "vs_historical", method, window, operator, endpoint)["estimate"])
        for operator in SOURCE.V3W.OPERATORS for endpoint in ("harm", "cvar25")
    )


def interaction_reversal(intervals: dict[str, dict[str, float]]) -> bool:
    assert SOURCE is not None
    for method in METHODS[1:]:
        for operator in SOURCE.V3W.OPERATORS:
            for endpoint in ("harm", "cvar25"):
                directions = []
                for window in WINDOWS:
                    interval = lookup_interval(intervals, "vs_historical", method, window, operator, endpoint)
                    tolerance = numerical_tolerance(0.0, interval["estimate"])
                    if interval["upper"] < -tolerance:
                        directions.append(-1)
                    elif interval["lower"] > tolerance:
                        directions.append(1)
                    else:
                        directions.append(0)
                if -1 in directions and 1 in directions:
                    return True
    return False


def factor_summary(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        grouped[(row["method"], row["window"])].append(row)
    rows: list[dict[str, Any]] = []
    for method, window in ((method, window) for method in METHODS for window in WINDOWS):
        values = grouped[(method, window)]
        valid = [row for row in values if row.get("solver_valid")]
        applied = [float(row["applied_delta_norm"]) for row in valid if math.isfinite(float(row.get("applied_delta_norm", math.nan)))]
        beta = [float(row["beta"]) for row in valid if math.isfinite(float(row.get("beta", math.nan)))]
        row = {
            "schema_version": 1,
            "method": method,
            "window": window,
            "state_count": len(values),
            "valid_solver_count": len(valid),
            "invalid_solver_count": len(values) - len(valid),
            "backtracking_null_count": sum(bool(row.get("backtracking_null")) for row in values),
            "mean_applied_delta_norm": float(np.mean(applied)) if applied else math.nan,
            "mean_beta": float(np.mean(beta)) if beta else math.nan,
        }
        for constraint in CONSTRAINT_ORDER:
            dots = [
                float(item["constraint_dots_applied"][constraint])
                for item in valid
                if constraint in item.get("constraint_dots_applied", {})
            ]
            row[f"mean_{constraint}_dot_applied"] = float(np.mean(dots)) if dots else math.nan
        rows.append(row)
    return rows


def run_a0p(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert SOURCE is not None and AUDIT is not None
    validate_frozen_args(args)
    audit = AUDIT
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    heldout_names = list(all_names[args.sample_count:args.sample_count * 2])
    if list(names) != list(all_names[:args.sample_count]) or len(heldout_names) != 128 or set(names) & set(heldout_names):
        raise RuntimeError("A0P update128/heldout128 population identity mismatch")
    states, trace, a0r_source = validate_a0r(Path(audit.a0r_trace_dir), Path(audit.a0r_closeout))
    output = Path(output_dir)
    current_source_path = output / f"{args.run_tag}_source_manifest.json"
    current_source = json.loads(current_source_path.read_text(encoding="utf-8"))
    current_source["heldout_names"] = heldout_names
    current_source["v4a_a0p"] = {"methods": list(METHODS), "windows": list(WINDOWS), "state_count": len(states)}
    write_json(current_source_path, current_source)
    verify_source_identity(a0r_source, current_source, heldout_names)
    ratios = inherited_ratios(args, v3s, legacy, frozen, list(names), folds, device)
    assignments: dict[str, dict[str, list[str]]] = {}
    for state in states:
        state_hash = str(state["sha256"])
        fixed = [str(name) for name in state["window_names"]]
        if len(fixed) != 4 or len(set(fixed)) != 4 or not set(fixed).issubset(set(names)):
            raise RuntimeError("invalid historical fixed4 assignment")
        assignments[state_hash] = {
            "fixed4": fixed,
            "shuffled16": shuffled16(names, state_hash),
            "prestratified32": prestratified32(names, ratios, state_hash),
        }
    assignments_path = output / "v4a_a0p_window_assignments_cloud_only.json"
    write_json(assignments_path, assignments)
    assignment_hash = sha256_file(assignments_path)
    assignment_manifest = {
        "schema_version": 1,
        "state_count": len(assignments),
        "assignment_sha256": assignment_hash,
        "window_cardinality": {"fixed4": 4, "shuffled16": 16, "prestratified32": 32},
        "raw_assignments_cloud_only": str(assignments_path),
    }
    write_json(output / "v4a_a0p_window_assignment_manifest.json", assignment_manifest)
    operator_count = len(SOURCE.V3W.OPERATORS)
    shape = (len(states), len(METHODS), len(WINDOWS), len(heldout_names), operator_count)
    post = {key: np.full(shape, np.nan, dtype=np.float64) for key in METRIC_KEYS}
    pre = {key: np.full((len(states), len(heldout_names), operator_count), np.nan, dtype=np.float64) for key in METRIC_KEYS}
    raw_path = output / "v4a_a0p_rows_cloud_only.csv"
    diagnostic_path = output / "v4a_a0p_projection_diagnostics_cloud_only.csv"
    diagnostics: list[dict[str, Any]] = []
    raw_count = 0
    with raw_path.open("w", encoding="utf-8", newline="") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS, lineterminator="\n")
        raw_writer.writeheader()
        for state_index, state in enumerate(states):
            payload = torch.load(state["path"], map_location="cpu")
            if payload.get("replicate_id") != "r1" or payload.get("state_kind") != "pre" or int(payload.get("epoch", -1)) != int(state["epoch"]):
                raise RuntimeError("retained state payload identity mismatch")
            label, model, optimizer, parameters = load_cell(payload, args, v3s, legacy, frozen, list(names), folds, device)
            baseline = render_population(args, v3s, legacy, frozen, heldout_names, folds, label, model, device)
            for key in METRIC_KEYS:
                pre[key][state_index] = baseline[key]
            raw_count += write_raw_rows(raw_writer, record_kind="pre_state", state_index=state_index, state=state, method="pre_state", window="pre_state", names=heldout_names, population=baseline)
            del model, optimizer, parameters
            release_cuda_memory()
            for method_index, method in enumerate(METHODS):
                for window_index, window in enumerate(WINDOWS):
                    try:
                        population, diagnostic = run_factor_cell(
                            payload=payload, args=args, v3s=v3s, legacy=legacy, frozen=frozen, all_update_names=list(names),
                            heldout_names=heldout_names, folds=folds, device=device, method=method, window=window,
                            window_names=assignments[str(state["sha256"])][window],
                        )
                    except (FloatingPointError, ValueError) as error:
                        population = {}
                        diagnostic = {"method": method, "window": window, "solver_valid": False, "failure": str(error), "backtracking_null": False}
                    diagnostic.update({
                        "schema_version": 1,
                        "state_index": state_index,
                        "state_sha256": state["sha256"],
                        "epoch": state["epoch"],
                        "update": state["update"],
                        "window_assignment_sha256": canonical_hash(assignments[str(state["sha256"])][window]),
                    })
                    diagnostics.append(diagnostic)
                    if population:
                        for key in METRIC_KEYS:
                            post[key][state_index, method_index, window_index] = population[key]
                        raw_count += write_raw_rows(raw_writer, record_kind="post_step", state_index=state_index, state=state, method=method, window=window, names=heldout_names, population=population)
                    del population
                    release_cuda_memory()
            print(json.dumps({"V4A_A0P_PROGRESS": {"completed_states": state_index + 1, "total_states": len(states), "raw_rows": raw_count}}, sort_keys=True), flush=True)
    write_rows(diagnostic_path, diagnostics)
    complete = all(np.isfinite(array).all() for array in (*pre.values(), *post.values()))
    structural_valid = complete and all(bool(row.get("solver_valid")) for row in diagnostics) and len(diagnostics) == len(states) * len(METHODS) * len(WINDOWS)
    factor_rows = factor_summary(diagnostics)
    factor_path = output / "v4a_a0p_step_summary.csv"
    write_rows(factor_path, factor_rows)
    if structural_valid:
        bootstrap_rows, intervals = bootstrap_summary(post, pre)
        exact_positive = [window for window in WINDOWS if cell_positive(METHODS[1], window, intervals)]
        proposal_positive = [window for window in WINDOWS if cell_positive(METHODS[2], window, intervals)]
        exact_only = bool(exact_positive) and not bool(proposal_positive)
        reversal = interaction_reversal(intervals)
    else:
        bootstrap_rows = [{"schema_version": 1, "status": "NOT_RUN_FAIL_CLOSED", "reason": "incomplete_or_invalid_factor_family"}]
        intervals = {}
        exact_positive = []
        proposal_positive = []
        exact_only = False
        reversal = False
    bootstrap_path = output / "v4a_a0p_poststep_summary.csv"
    write_rows(bootstrap_path, bootstrap_rows)
    state, decision, authorizes = classify_a0p(
        complete=complete,
        structural_valid=structural_valid,
        proposal_positive_windows=proposal_positive,
        exact_only_positive=exact_only,
        interaction_reversal=reversal,
    )
    closeout = {
        "route_id": ROUTE_ID,
        "stage": "v4a-A0P-paired-applied-update-audit",
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "evidence_role": "development_screening",
        "metric_contract": "2026-07-14 v4a A0P amendment: complete 3x3 paired actual-render factorial with simultaneous two-way bootstrap",
        "a0r_closeout_sha256": sha256_file(Path(audit.a0r_closeout)),
        "a0r_trace_manifest_sha256": sha256_file(Path(audit.a0r_trace_dir) / "trace_manifest.json"),
        "v3z_source_sha256": sha256_file(Path(SOURCE.__file__)),
        "state_count": len(states),
        "heldout_image_count": len(heldout_names),
        "operator_count": operator_count,
        "factor_cells": len(states) * len(METHODS) * len(WINDOWS),
        "complete_family": complete,
        "structural_valid": structural_valid,
        "failure_class": "FAIL_CLOSED" if not structural_valid else None,
        "proposal_positive_windows": proposal_positive,
        "exact_positive_windows": exact_positive,
        "exact_only_positive": exact_only,
        "interaction_reversal": reversal,
        "assignment_manifest": str(output / "v4a_a0p_window_assignment_manifest.json"),
        "assignment_sha256": assignment_hash,
        "raw_rows_cloud_only": str(raw_path),
        "raw_rows_count": raw_count,
        "projection_diagnostics_cloud_only": str(diagnostic_path),
        "factor_summary": str(factor_path),
        "bootstrap_summary": str(bootstrap_path),
        "locked_test_touched": False,
        "canary_touched": False,
        "candidate_selected": False,
    }
    closeout["contract_id"] = canonical_hash(closeout)
    closeout_path = output / "v4a_a0p_closeout.json"
    write_json(closeout_path, closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--a0r-closeout", required=True)
    parser.add_argument("--a0r-trace-dir", required=True)
    args, v3z_args = parser.parse_known_args(argv)
    if not v3z_args:
        raise ValueError("frozen v3z arguments are required after A0P arguments")
    global SOURCE, AUDIT
    AUDIT = args
    SOURCE = load_source(Path(args.v3z_root).resolve())
    SOURCE.run_projected = run_a0p
    original = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *v3z_args]
        SOURCE.main()
    finally:
        sys.argv = original


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a0p_audit.py audit --v3z-root ... --a0r-closeout ... --a0r-trace-dir ... projected ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
