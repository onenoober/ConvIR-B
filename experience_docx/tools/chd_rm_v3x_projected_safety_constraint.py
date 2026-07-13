#!/usr/bin/env python3
"""Projected first-order safety constraints for the activated Delta-u head."""

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROUTE_ID = "haze4k_v5_chd_rm_v3y_cross_sample_safety_20260713"
V3W = None


def load_legacy():
    path = Path(__file__).with_name("chd_rm_v3w_gradual_safety_ramp.py")
    spec = importlib.util.spec_from_file_location("v3w_legacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grad_of(loss, parameters, retain_graph):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=False)
    return [value.detach() for value in values]


def dot(left, right):
    return sum(torch.sum(a * b) for a, b in zip(left, right))


def projected_grad(render_grad, constraints):
    """Remove components that would increase any direct safety loss to first order."""
    direction = [value.clone() for value in render_grad]
    projections = []
    for name, constraint_grad in constraints:
        numerator = dot(constraint_grad, direction)
        denominator = dot(constraint_grad, constraint_grad).clamp_min(1e-30)
        before = float(numerator.item())
        if before < 0.0:
            scale = numerator / denominator
            direction = [value - scale * normal for value, normal in zip(direction, constraint_grad)]
        projections.append({"constraint": name, "dot_before": before, "dot_after": float(dot(constraint_grad, direction).item())})
    return direction, projections


def train_projected(args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label):
    if objective != "safety_curriculum":
        raise ValueError(f"unexpected objective: {objective}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    midpoint = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        updates = 0
        projected_updates = 0
        for start in range(0, len(names), args.risk_window):
            optimizer.zero_grad(set_to_none=True)
            terms = defaultdict(list)
            for name in names[start:start + args.risk_window]:
                sample = V3W.frozen_output_sample(args, v3s, legacy, frozen, name, folds[name], device)
                sample["delta_bound"] = args.delta_bound
                for operator in V3W.OPERATORS:
                    values = V3W.candidate_metrics(v3s, sample, V3W.delta_for(kind, model, sample, operator), operator)
                    for key, value in values.items():
                        terms[key].append(value)
            means = {key: torch.stack(value).mean() for key, value in terms.items()}
            harms = torch.stack(terms["harm"])
            cvar_count = max(1, int(math.ceil(args.cvar_fraction * harms.numel())))
            cvar = torch.topk(harms, cvar_count).values.mean()
            render_grad = grad_of(means["render"], parameters, retain_graph=epoch > args.warmup_epochs)
            if epoch <= args.warmup_epochs:
                direction, projections = render_grad, []
            else:
                constraints = [
                    ("anchor", grad_of(means["anchor"], parameters, retain_graph=True)),
                    ("harm", grad_of(means["harm"], parameters, retain_graph=True)),
                    ("margin", grad_of(means["margin"], parameters, retain_graph=True)),
                    ("cvar", grad_of(cvar, parameters, retain_graph=False)),
                ]
                direction, projections = projected_grad(render_grad, constraints)
            if any(row["dot_before"] < 0.0 for row in projections):
                projected_updates += 1
            for parameter, value in zip(parameters, direction):
                parameter.grad = value
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm).item())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite projected gradient for {label}")
            optimizer.step()
            for key, value in means.items():
                totals[key] += float(value.detach().item())
            totals["render_direction_norm"] += float(torch.sqrt(dot(render_grad, render_grad)).item())
            totals["projected_direction_norm"] += float(torch.sqrt(dot(direction, direction)).item())
            totals["cvar"] += float(cvar.detach().item())
            totals["gradient_norm"] += gradient_norm
            updates += 1
        row = {"cell": label, "epoch": epoch, "phase": "render_warmup" if epoch <= args.warmup_epochs else "projected_safety", "updates": updates,
               "projected_update_ratio": projected_updates / updates,
               **{key: value / updates for key, value in totals.items()}}
        history.append(row)
        print(json.dumps({"PROJECTED_PROGRESS": row}, sort_keys=True), flush=True)
        if epoch == args.warmup_epochs:
            midpoint = V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    final = V3W.evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    return initial, midpoint, final, history


def run_projected(args, v3s, legacy, frozen, names, folds, device, output_dir):
    source_path = Path(output_dir) / f"{args.run_tag}_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["mode"] = "projected"
    source["objective"] = {"warmup_epochs": args.warmup_epochs, "render_objective": "MSE", "constraints": ["anchor", "harm", "margin", "CVaR25"]}
    all_names, _ = v3s.load_names_and_folds(args, legacy)
    heldout_names = all_names[args.sample_count:args.sample_count * 2]
    if len(heldout_names) != args.sample_count or set(heldout_names) & set(names):
        raise RuntimeError("v3y requires a disjoint fixed 32-name holdout")
    source["heldout_names"] = list(heldout_names)
    V3W.write_json(source_path, source)
    models = V3W.import_v3w_models()
    first = V3W.frozen_output_sample(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = V3W.build_cells(models, first, args, device)
    label, (kind, objective, model) = next(iter(cells.items()))
    heldout_initial = V3W.evaluate_cell(args, v3s, legacy, frozen, heldout_names, folds, kind, model, device)
    initial, midpoint, final, history = train_projected(args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label)
    heldout_final = V3W.evaluate_cell(args, v3s, legacy, frozen, heldout_names, folds, kind, model, device)
    reduction = (initial["render"] - final["render"]) / max(initial["render"], 1e-30)
    midpoint_reduction = (initial["render"] - midpoint["render"]) / max(initial["render"], 1e-30)
    midpoint_pass = midpoint["delta_abs"] >= args.activity_delta_abs and midpoint_reduction >= args.min_relative_render_reduction
    final_pass = final["delta_abs"] >= args.activity_delta_abs and reduction >= args.min_relative_render_reduction
    safety_pass = all(final[key] <= V3W.V3U_FINAL_SAFETY[key] for key in V3W.V3U_FINAL_SAFETY)
    heldout_reduction = (heldout_initial["render"] - heldout_final["render"]) / max(heldout_initial["render"], 1e-30)
    heldout_pass = (heldout_final["delta_abs"] >= args.activity_delta_abs and heldout_reduction >= 0.0
                    and all(heldout_final[key] <= V3W.V3U_FINAL_SAFETY[key] for key in V3W.V3U_FINAL_SAFETY))
    decision = "V3Y_S1_CROSS_SAMPLE_SAFETY_PASS_AUTHORIZE_SEALED_INTERNAL_CONFIRMATION_ONLY" if midpoint_pass and final_pass and safety_pass and heldout_pass else "V3Y_S1_CROSS_SAMPLE_SAFETY_FAIL_STOP"
    closeout = {"route_id": ROUTE_ID, "run_id": args.run_tag, "stage": "v3y-S1-output-side-cross-sample-direct-safety",
                "state": "COMPLETED_GATE_PASS" if decision.startswith("V3Y_S1_CROSS_SAMPLE_SAFETY_PASS") else "COMPLETED_GATE_FAIL",
                "gate_type": "mechanism_direct_safety", "decision": decision,
                "authorizes": "sealed internal safety confirmation only" if decision.endswith("ONLY") else "none; cross-sample safety contract stopped",
                "metric_contract": "train32 output Delta-u with projected direct safety and disjoint heldout32 activity, nonnegative render, and v3u-reference safety gate",
                "cells": {label: {"initial": initial, "midpoint": midpoint, "final": final, "relative_render_reduction": reduction,
                                   "midpoint_relative_render_reduction": midpoint_reduction, "midpoint_activity_pass": midpoint_pass,
                                   "final_activity_pass": final_pass, "safety_nonworse_vs_v3u": safety_pass,
                                   "parameter_count": sum(value.numel() for value in model.parameters()),
                                   "heldout_initial": heldout_initial, "heldout_final": heldout_final,
                                   "heldout_relative_render_reduction": heldout_reduction, "heldout_pass": heldout_pass}},
                "locked_test_touched": False, "canary_touched": False, "training_occurred": True}
    V3W.write_rows(Path(output_dir) / f"{args.run_tag}_history.csv", history)
    V3W.write_json(Path(output_dir) / f"{args.run_tag}_summary.json", closeout)
    V3W.write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def run_noop_v3x(original, args, v3s, legacy, frozen, names, folds, device, output_dir):
    closeout = original(args, v3s, legacy, frozen, names, folds, device, output_dir)
    passed = closeout["state"] == "COMPLETED_GATE_PASS"
    closeout.update({
        "stage": "v3y-S0-output-form-exact-noop",
        "decision": "V3Y_S0_NOOP_PASS_AUTHORIZE_CROSS_SAMPLE_SAFETY_ONLY" if passed else "V3Y_S0_NOOP_FAIL_STOP",
        "authorizes": "v3y-S1 cross-sample direct-safety fixed32 diagnostic only" if passed else "none",
    })
    legacy.write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def main():
    global V3W
    V3W = load_legacy()
    V3W.ROUTE_ID = ROUTE_ID
    original_noop = V3W.run_noop
    V3W.run_noop = lambda *values: run_noop_v3x(original_noop, *values)
    V3W.run_curriculum = run_projected
    original = sys.argv[:]
    sys.argv = ["v3x"] + ["ramp" if value == "projected" else value for value in sys.argv[1:]]
    V3W.main()
    sys.argv = original


if __name__ == "__main__":
    main()
