#!/usr/bin/env python3
"""Four-cell real-render diagnostic for the v3s Delta-u zero-lock failure."""

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


ROUTE_ID = "haze4k_v5_chd_rm_v3t_zero_lock_context_diagnostic_20260713"
OPERATORS = ("D_ref", "D_rep")
ALPHA_LOW = 0.125
ALPHA_HIGH = 0.25


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def import_v3t_models():
    model_dir = Path(__file__).resolve().parents[2] / "Dehazing" / "ITS" / "models"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    return importlib.import_module("direction_repair_context")


def import_v3s(v3s_root):
    v3s_root = Path(v3s_root).resolve()
    tools_dir = v3s_root / "experience_docx" / "tools"
    its_dir = v3s_root / "Dehazing" / "ITS"
    if not tools_dir.is_dir() or not its_dir.is_dir():
        raise FileNotFoundError(f"invalid frozen v3s source: {v3s_root}")
    for path in (str(its_dir), str(tools_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("chd_rm_v3s_delta_u_direction_repair")


def frozen_sample_with_context(args, v3s, legacy, frozen, name, fold, device):
    input_img, label = legacy.load_pair(args.data_dir, args.source_split, name)
    input_img = input_img.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    with torch.no_grad():
        padded, height, width = legacy.pad_to_factor(input_img)
        label = label[:, :, :height, :width]
        hazy = padded[:, :, :height, :width]
        base = legacy.v3l_a0.forward_final(frozen["base"], padded, height, width)
        gate_full, _, _ = frozen["gate_producer"](padded)
        hard_gate = legacy.action_gate_from_full(gate_full, legacy.action_shape_for_input(padded)).to(device)
        support = legacy.output_gate_from_action_gate(hard_gate, base.shape[-2:])
        context, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
        steps = {}
        for operator in OPERATORS:
            head, mean, std = legacy.v3l_a0.model_pack_from_cache(
                frozen["caches"][operator], "OOF", fold
            )
            pred_low = legacy.v3l_a0.v3j_b.score_map(
                "context", head, context, mean, std, frozen["bound"]
            )
            steps[operator] = support * torch.nn.functional.interpolate(
                pred_low,
                size=base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    return {
        "hazy": hazy.detach(),
        "label": label.detach(),
        "base": base.detach(),
        "support": support.detach(),
        "context": context.detach(),
        "steps": {key: value.detach() for key, value in steps.items()},
    }


def build_cells(models, sample, args, device):
    context_channels = int(sample["context"].shape[1])
    return {
        "output_safe": ("output", "safe", models.DIRT_OutputDeltaU(args.channels, args.delta_bound).to(device)),
        "output_utility": ("output", "utility", models.DIRT_OutputDeltaU(args.channels, args.delta_bound).to(device)),
        "context_safe": (
            "context",
            "safe",
            models.DIRT_ContextDeltaU(context_channels, args.channels, args.delta_bound).to(device),
        ),
        "context_utility": (
            "context",
            "utility",
            models.DIRT_ContextDeltaU(context_channels, args.channels, args.delta_bound).to(device),
        ),
    }


def delta_for(kind, model, sample, operator):
    if kind == "output":
        return model(sample["hazy"], sample["base"], sample["steps"][operator], sample["support"])
    return model(
        sample["context"],
        sample["hazy"],
        sample["base"],
        sample["steps"][operator],
        sample["support"],
    )


def candidate_metrics(v3s, sample, delta, operator):
    old_low, old_high, new_low, new_high = v3s.candidate_predictions(
        sample["base"], sample["steps"][operator], delta
    )
    old_low_mse = v3s.per_image_mse(old_low, sample["label"])
    new_low_mse = v3s.per_image_mse(new_low, sample["label"])
    new_high_mse = v3s.per_image_mse(new_high, sample["label"])
    margin = v3s.active_block_negative_margin_loss(new_low, new_high, sample["label"], sample["support"])
    repair = torch.mean(torch.abs(delta) / delta.new_tensor(sample["delta_bound"]).view(1, 3, 1, 1))
    return {
        "render": new_high_mse.mean(),
        "anchor": torch.relu(new_low_mse - old_low_mse).mean(),
        "harm": torch.relu(new_high_mse - old_low_mse).mean(),
        "margin": margin,
        "repair": repair,
        "delta_abs": torch.mean(torch.abs(delta)),
    }


def evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device):
    model.eval()
    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for name in names:
            sample = frozen_sample_with_context(args, v3s, legacy, frozen, name, folds[name], device)
            sample["delta_bound"] = args.delta_bound
            for operator in OPERATORS:
                values = candidate_metrics(v3s, sample, delta_for(kind, model, sample, operator), operator)
                for key, value in values.items():
                    totals[key] += float(value.item())
                count += 1
    return {key: value / count for key, value in totals.items()}


def train_cell(args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial = evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        updates = 0
        for start in range(0, len(names), args.risk_window):
            optimizer.zero_grad(set_to_none=True)
            terms = defaultdict(list)
            for name in names[start:start + args.risk_window]:
                sample = frozen_sample_with_context(args, v3s, legacy, frozen, name, folds[name], device)
                sample["delta_bound"] = args.delta_bound
                for operator in OPERATORS:
                    values = candidate_metrics(v3s, sample, delta_for(kind, model, sample, operator), operator)
                    for key, value in values.items():
                        terms[key].append(value)
            means = {key: torch.stack(value).mean() for key, value in terms.items()}
            if objective == "utility":
                total = means["render"] + args.repair_weight * means["repair"]
                cvar = means["harm"].new_zeros(())
            else:
                harms = torch.stack(terms["harm"])
                cvar_count = max(1, int(math.ceil(args.cvar_fraction * harms.numel())))
                cvar = torch.topk(harms, cvar_count).values.mean()
                total = (
                    means["render"]
                    + args.anchor_weight * means["anchor"]
                    + args.margin_weight * means["margin"]
                    + args.harm_weight * means["harm"]
                    + args.cvar_weight * cvar
                    + args.repair_weight * means["repair"]
                )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite {label} loss")
            total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm).item())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite {label} gradient")
            optimizer.step()
            for key, value in means.items():
                totals[key] += float(value.detach().item())
            totals["total"] += float(total.detach().item())
            totals["cvar"] += float(cvar.detach().item())
            totals["gradient_norm"] += gradient_norm
            updates += 1
        history.append({
            "cell": label,
            "epoch": epoch,
            "updates": updates,
            **{key: value / updates for key, value in totals.items()},
        })
        print(json.dumps({"FACTORIAL_PROGRESS": history[-1]}, sort_keys=True), flush=True)
    final = evaluate_cell(args, v3s, legacy, frozen, names, folds, kind, model, device)
    return initial, final, history


def run_noop(args, v3s, legacy, frozen, names, folds, device, output_dir):
    models = import_v3t_models()
    first = frozen_sample_with_context(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = build_cells(models, first, args, device)
    reference = legacy.v3m_a1.load_fixed_reference(args.reference_oof_rows)
    maxima = defaultdict(float)
    counts = defaultdict(int)
    with torch.no_grad():
        for name in names:
            sample = frozen_sample_with_context(args, v3s, legacy, frozen, name, folds[name], device)
            _, base_psnr = legacy.metric_pair(sample["base"], sample["label"])
            for cell, (kind, _, model) in cells.items():
                for operator in OPERATORS:
                    delta = delta_for(kind, model, sample, operator)
                    old_low, _, new_low, _ = v3s.candidate_predictions(sample["base"], sample["steps"][operator], delta)
                    _, new_psnr = legacy.metric_pair(new_low, sample["label"])
                    maxima["delta"] = max(maxima["delta"], float(torch.max(torch.abs(delta)).item()))
                    maxima["prediction"] = max(maxima["prediction"], float(torch.max(torch.abs(new_low - old_low)).item()))
                    maxima["reference"] = max(
                        maxima["reference"],
                        abs((new_psnr - base_psnr) - float(reference[(operator, name)])),
                    )
                    counts[cell] += 1
    passed = (
        all(counts[cell] == len(names) * len(OPERATORS) for cell in cells)
        and maxima["delta"] == 0.0
        and maxima["prediction"] == 0.0
        and maxima["reference"] <= args.replay_tolerance_db
    )
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3t-S0-dual-form-exact-noop",
        "state": "COMPLETED_GATE_PASS" if passed else "COMPLETED_GATE_FAIL",
        "gate_type": "structural_integrity",
        "decision": "V3T_S0_NOOP_PASS_AUTHORIZE_FACTORIAL_ONLY" if passed else "V3T_S0_NOOP_FAIL_STOP",
        "authorizes": "v3t-S1 four-cell fixed32 diagnostic only" if passed else "none",
        "max_abs_delta": maxima["delta"],
        "max_abs_prediction_diff": maxima["prediction"],
        "max_abs_reference_psnr_delta_diff_db": maxima["reference"],
        "counts": dict(counts),
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": False,
    }
    write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def classify(summary):
    output_safe = summary["output_safe"]["activity_pass"]
    output_utility = summary["output_utility"]["activity_pass"]
    context_safe = summary["context_safe"]["activity_pass"]
    context_utility = summary["context_utility"]["activity_pass"]
    if not output_utility and not context_utility:
        return "V3T_S1_ALL_UTILITY_CELLS_INACTIVE_REQUIRE_OPTIMIZATION_REDESIGN"
    if context_utility and not output_utility and not context_safe:
        return "V3T_S1_CONTEXT_SUFFICIENCY_AND_SAFE_ZERO_LOCK_DIAGNOSIS_ONLY"
    if (output_utility and not output_safe) or (context_utility and not context_safe):
        return "V3T_S1_SAFE_OBJECTIVE_ZERO_LOCK_DIAGNOSIS_ONLY"
    if context_safe:
        return "V3T_S1_CONTEXT_SAFE_ACTIVITY_PASS_AUTHORIZE_NEW_TRAINING_CONTRACT_DESIGN_ONLY"
    return "V3T_S1_OUTPUT_UTILITY_ACTIVITY_ONLY_AUTHORIZE_REPRESENTATION_DESIGN_ONLY"


def run_factorial(args, v3s, legacy, frozen, names, folds, device, output_dir):
    models = import_v3t_models()
    first = frozen_sample_with_context(args, v3s, legacy, frozen, names[0], folds[names[0]], device)
    cells = build_cells(models, first, args, device)
    summaries = {}
    histories = []
    for label, (kind, objective, model) in cells.items():
        initial, final, history = train_cell(
            args, v3s, legacy, frozen, names, folds, kind, objective, model, device, label
        )
        relative_reduction = (initial["render"] - final["render"]) / max(initial["render"], 1e-30)
        activity = final["delta_abs"] >= args.activity_delta_abs and relative_reduction >= args.min_relative_render_reduction
        summaries[label] = {
            "input_form": kind,
            "objective": objective,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initial": initial,
            "final": final,
            "relative_render_reduction": relative_reduction,
            "activity_pass": activity,
        }
        histories.extend(history)
    decision = classify(summaries)
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "stage": "v3t-S1-four-cell-zero-lock-diagnostic",
        "state": "COMPLETED_GATE_PASS",
        "gate_type": "mechanism_diagnostic",
        "decision": decision,
        "authorizes": "new route design only; no formal training, policy, canary, or locked test",
        "metric_contract": "32 fixed OOF images, real rendered alpha=.25 objective, factorial frozen-context and safety-objective controls",
        "activity_delta_abs": args.activity_delta_abs,
        "min_relative_render_reduction": args.min_relative_render_reduction,
        "cells": summaries,
        "locked_test_touched": False,
        "canary_touched": False,
        "training_occurred": True,
    }
    write_rows(Path(output_dir) / f"{args.run_tag}_history.csv", histories)
    write_json(Path(output_dir) / f"{args.run_tag}_summary.json", closeout)
    write_json(Path(output_dir) / f"{args.run_tag}_closeout.json", closeout)
    return closeout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("noop", "factorial"))
    parser.add_argument("--v3s_root", required=True)
    parser.add_argument("--expected_v3s_commit", required=True)
    parser.add_argument("--v3p_root", required=True)
    parser.add_argument("--expected_v3p_commit", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fresh_split_manifest", required=True)
    parser.add_argument("--v3j_a_bounds", required=True)
    parser.add_argument("--operator_artifact_manifest", required=True)
    parser.add_argument("--density_artifact", required=True)
    parser.add_argument("--d7c_artifact", required=True)
    parser.add_argument("--reference_oof_rows", required=True)
    parser.add_argument("--expected_a0_checkpoint_sha256", required=True)
    parser.add_argument("--expected_control_checkpoint_sha256", required=True)
    parser.add_argument("--expected_density_artifact_sha256", required=True)
    parser.add_argument("--expected_d7c_artifact_sha256", required=True)
    parser.add_argument("--expected_fresh_split_manifest_sha256", required=True)
    parser.add_argument("--expected_operator_manifest_sha256", required=True)
    parser.add_argument("--expected_reference_oof_rows_sha256", required=True)
    parser.add_argument("--expected_v3j_a_bounds_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--train_key", default="v3j_controller_train")
    parser.add_argument("--formal_sample_count", type=int, default=1200)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--proj_channels", type=int, default=24)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--d7c_threshold", type=float, default=0.5773006677627563)
    parser.add_argument("--delta_bound_multiplier", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--risk_window", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=30.0)
    parser.add_argument("--margin_weight", type=float, default=5.0)
    parser.add_argument("--harm_weight", type=float, default=20.0)
    parser.add_argument("--cvar_weight", type=float, default=40.0)
    parser.add_argument("--repair_weight", type=float, default=0.02)
    parser.add_argument("--cvar_fraction", type=float, default=0.25)
    parser.add_argument("--activity_delta_abs", type=float, default=0.000001)
    parser.add_argument("--min_relative_render_reduction", type=float, default=0.001)
    parser.add_argument("--replay_tolerance_db", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_split.lower() != "train" or args.sample_count != 32:
        raise ValueError("v3t is fixed to the first 32 train-derived OOF names")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v3t requires the authorized CUDA runtime")
    if args.epochs <= 0 or args.risk_window <= 0 or not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("invalid fixed factorial budget")
    v3s_root = Path(args.v3s_root)
    if git_output(v3s_root, "rev-parse", "HEAD") != args.expected_v3s_commit:
        raise RuntimeError("frozen v3s source commit mismatch")
    if git_output(v3s_root, "status", "--porcelain"):
        raise RuntimeError("frozen v3s source checkout must be clean")
    v3p_root = Path(args.v3p_root)
    if git_output(v3p_root, "rev-parse", "HEAD") != args.expected_v3p_commit:
        raise RuntimeError("frozen v3p source commit mismatch")
    if git_output(v3p_root, "status", "--porcelain"):
        raise RuntimeError("frozen v3p source checkout must be clean")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    set_determinism(args.seed)
    v3s = import_v3s(v3s_root)
    legacy = v3s.import_legacy_modules(v3p_root)
    assets = v3s.asset_manifest(args)
    device = torch.device(args.device)
    frozen = v3s.build_frozen_operator(args, legacy, device)
    args.delta_bound = tuple(args.delta_bound_multiplier * value for value in frozen["bound"])
    names, folds = v3s.load_names_and_folds(args, legacy)
    names = names[:args.sample_count]
    source = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "mode": args.mode,
        "v3s_root": str(v3s_root),
        "v3s_commit": args.expected_v3s_commit,
        "v3p_root": str(v3p_root),
        "v3p_commit": args.expected_v3p_commit,
        "assets": assets,
        "frozen_operator_artifacts": frozen["artifacts"],
        "delta_bound": list(args.delta_bound),
        "names": list(names),
        "locked_test_touched": False,
        "canary_touched": False,
    }
    write_json(output_dir / f"{args.run_tag}_source_manifest.json", source)
    if args.mode == "noop":
        closeout = run_noop(args, v3s, legacy, frozen, names, folds, device, output_dir)
    else:
        closeout = run_factorial(args, v3s, legacy, frozen, names, folds, device, output_dir)
    print(json.dumps(closeout, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
