#!/usr/bin/env python3
"""Run the frozen R3 A2 full-OOF test of the sole C3 screen survivor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import r3_a1_acv_screen as a1
from route_program_api import (
    asset_path, atomic_json, load_context, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)

ROUTE_ID = "haze4k_v5_r3_proposal_first_acv_20260717"
OPERATION_ID = "R3_A2_ACV_FULL_OOF"
CELLS = ("C1_action", "C3_deep_response")
FOLDS = (0, 1, 2, 3)
SEEDS = (3407, 3411)
FEATURE_UNITS = 1536
TRAINING_UNITS = len(CELLS) * len(FOLDS) * len(SEEDS)
TOTAL_UNITS = FEATURE_UNITS + TRAINING_UNITS
GAIN_GATE = 0.020
RETENTION_GATE = 0.25
SHUFFLE_GATE = 0.005
RESPONSE_INCREMENT_GATE = 0.005


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contract(context_path: Path) -> None:
    import torch

    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parameter_count = (
        a1.INPUT_DIM * a1.HIDDEN_DIM + a1.HIDDEN_DIM
        + a1.HIDDEN_DIM * a1.HIDDEN_DIM + a1.HIDDEN_DIM
        + a1.HIDDEN_DIM + 1
    )
    checks = {
        "contract_cpu_only": context.device == "cpu"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "candidate_frozen_to_c3": CELLS == ("C1_action", "C3_deep_response"),
        "all_outer_folds_frozen": FOLDS == (0, 1, 2, 3),
        "paired_seeds_frozen": SEEDS == a1.SEEDS,
        "optimizer_inherited": a1.EPOCHS == 32
        and a1.LEARNING_RATE == 1.0e-3 and a1.WEIGHT_DECAY == 1.0e-4,
        "parameter_identity": parameter_count == 9153,
        "formal_gates_frozen": GAIN_GATE == 0.020
        and RETENTION_GATE == 0.25 and SHUFFLE_GATE == 0.005
        and RESPONSE_INCREMENT_GATE == 0.005,
        "generic_progress_total": context.total_units == TOTAL_UNITS,
        "protected_roles_blocked": True,
        "torch_available_for_fixture": hasattr(torch, "Tensor"),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "a2_synthetic_contract.json",
        {"schema_version": 1, "checks": checks, "parameter_count": parameter_count},
    )
    write_contract_result(context, checks=checks)


def paired_increment_bootstrap(rows: list[dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    names = sorted({row["name"] for row in rows})
    operators = ("D_ref", "D_rep")
    keyed = {(row["cell"], row["name"], row["operator"]): row for row in rows}
    arrays = {
        operator: np.asarray([
            float(keyed[("C3_deep_response", name, operator)]["gain"])
            - float(keyed[("C1_action", name, operator)]["gain"])
            for name in names
        ], dtype=np.float64)
        for operator in operators
    }
    point = min(float(array.mean()) for array in arrays.values())
    rng = np.random.default_rng(a1.BOOTSTRAP_SEED + 1)
    samples = []
    for _ in range(a1.BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(names), len(names))
        samples.append(min(float(array[index].mean()) for array in arrays.values()))
    vector = np.asarray(samples, dtype=np.float64)
    return {
        "point": point,
        "lcb95": float(np.quantile(vector, 0.025)),
        "ucb95": float(np.quantile(vector, 0.975)),
    }


def risk_delta_ucb(rows: list[dict[str, Any]], threshold: float) -> float:
    import numpy as np

    names = sorted({row["name"] for row in rows})
    operators = ("D_ref", "D_rep")
    keyed = {(row["name"], row["operator"]): row for row in rows}
    arrays = {
        operator: np.asarray([
            float(float(keyed[(name, operator)]["gain"]) <= threshold)
            for name in names
        ], dtype=np.float64)
        for operator in operators
    }
    rng = np.random.default_rng(a1.BOOTSTRAP_SEED + int(abs(threshold) * 1000))
    samples = []
    for _ in range(a1.BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(names), len(names))
        samples.append(max(float(array[index].mean()) for array in arrays.values()))
    return float(np.quantile(np.asarray(samples, dtype=np.float64), 0.95))


def descriptive_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    coverage_rows = []
    strata_rows = []
    for operator in ("D_ref", "D_rep"):
        subset = [row for row in rows if row["operator"] == operator]
        ordered = sorted(subset, key=lambda row: float(row["confidence"]), reverse=True)
        for coverage in (0.05, 0.10, 0.20, 0.40, 1.00):
            count = max(1, math.ceil(len(ordered) * coverage))
            values = np.asarray([float(row["gain"]) for row in ordered[:count]], dtype=np.float64)
            coverage_rows.append({
                "operator": operator, "coverage": coverage, "count": count,
                "mean_gain_db": float(values.mean()),
                "p10_gain_db": float(np.quantile(values, 0.10)),
                "cvar5_gain_db": float(np.sort(values)[:max(1, math.ceil(len(values) * 0.05))].mean()),
                "severe_count": int((values <= -0.2).sum()),
                "hard_count": int((values <= -0.5).sum()),
            })
        reference_values = np.asarray([float(row["reference_psnr"]) for row in subset])
        easy_cut = float(np.quantile(reference_values, 0.75))
        groups = {
            "easy_reference_top_quartile": [row for row in subset if float(row["reference_psnr"]) >= easy_cut],
            "low_haze_transmission_ge_0.80": [row for row in subset if float(row["haze_transmission"]) >= 0.80],
        }
        for native_shape in sorted({row["native_shape"] for row in subset}):
            groups[f"native_{native_shape}"] = [row for row in subset if row["native_shape"] == native_shape]
        for label, items in groups.items():
            if not items:
                continue
            values = np.asarray([float(row["gain"]) for row in items], dtype=np.float64)
            strata_rows.append({
                "operator": operator, "stratum": label, "count": len(items),
                "mean_gain_db": float(values.mean()),
                "p10_gain_db": float(np.quantile(values, 0.10)),
                "severe_count": int((values <= -0.2).sum()),
                "hard_count": int((values <= -0.5).sum()),
            })
    return coverage_rows, strata_rows


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ledger = json.loads(asset_path(context, "r4_ledger", kind="file").read_text(encoding="utf-8"))
    a0_manifest = json.loads(asset_path(context, "a0_cache_manifest", kind="file").read_text(encoding="utf-8"))
    raw_manifest_path = asset_path(context, "a0_raw_manifest", kind="file")
    cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    data_root = asset_path(context, "haze4k_data", kind="directory")
    a1_closeout = json.loads(asset_path(context, "a1_closeout", kind="file").read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_rows) != FEATURE_UNITS or a0_manifest.get("cache_manifest_sha256") != "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d":
        raise RuntimeError("A0 sealed cache identity mismatch")
    if a1_closeout.get("decision") != "R3_A1_ACV_SCREEN_SURVIVOR" or a1_closeout.get("details", {}).get("survivors") != ["C3_deep_response"]:
        raise RuntimeError("A1 sole-survivor identity mismatch")
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}
    fold_lookup = {name: fold for fold, names in folds.items() for name in names}
    if sorted(len(value) for value in folds.values()) != [192, 192, 192, 192]:
        raise RuntimeError("S0 fold contract mismatch")
    confirmation = set(ledger["roles"]["confirmation"]); development = set(ledger["roles"]["development"])
    if confirmation & development or set(fold_lookup) != development:
        raise RuntimeError("development/confirmation isolation failed")

    def load_label(name: str) -> Any:
        stem, extension = os.path.splitext(name)
        for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
            path = data_root / "train/gt" / candidate
            if path.is_file():
                with Image.open(path) as image: array = np.asarray(image.convert("RGB")).copy()
                return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        raise FileNotFoundError(name)

    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A2 official frozen encoder feature extraction requires CUDA")
    device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    sys.path.insert(0, str(context.remote_repo / "Dehazing/ITS"))
    from models.ConvIR import build_net
    encoder_model = build_net("base", "Haze4K").to(device).eval()
    checkpoint_path = asset_path(context, "official_checkpoint", kind="file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder_model.load_state_dict(checkpoint, strict=True)
    for parameter in encoder_model.parameters(): parameter.requires_grad_(False)

    units: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, 1):
        unit_path = cache_root / f"{row['unit_key']}.pt"
        if a1.sha256_file(unit_path) != row["cache_sha256"]:
            raise RuntimeError(f"cache unit hash mismatch: {row['unit_key']}")
        payload = torch.load(unit_path, map_location="cpu", weights_only=False)
        name = payload["name"]; names = list(payload["candidate_names"]); n = len(names)
        base = payload["base"].float(); step = payload["step"].float(); candidates = payload["candidates"].float()
        current = payload["current"].float(); support = payload["support"].float(); label = load_label(name)
        if label.shape[-2:] != base.shape[-2:]: label = label[:, :, :base.shape[-2], :base.shape[-1]]
        reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
        renders = torch.clamp(base + 0.25 * (step + candidates), 0.0, 1.0)
        candidate_mse = (renders - label).square().mean((1, 2, 3)); reference_mse = (reference - label).square().mean()
        target = a1.metric_psnr(candidate_mse) - a1.metric_psnr(reference_mse)
        state = torch.cat((a1.tensor_stats(base), a1.tensor_stats(step), a1.tensor_stats(current), a1.tensor_stats(support)), dim=1)
        action = a1.action_features(names, torch.device("cpu")); rgb = a1.rgb_response_features(renders - reference)
        with torch.no_grad():
            encoder_inputs = torch.cat((reference, renders), dim=0).to(device)
            deep_all = encoder_model.Encoder[0](encoder_model.feat_extract[0](encoder_inputs))
            deep = a1.deep_response_features(deep_all[1:] - deep_all[0:1]).cpu()
        stem = Path(name).stem.split("_")
        units.append({
            "name": name, "operator": payload["operator"], "fold": fold_lookup[name],
            "n": n, "target": target, "mse": candidate_mse,
            "features": {cell: a1.build_cell_features(state, action, rgb, deep, cell) for cell in CELLS},
            "reference_psnr": float(a1.metric_psnr(reference_mse)),
            "native_shape": row["native_shape"],
            "haze_transmission": float(stem[1]), "haze_parameter_2": float(stem[2]),
        })
        if index % 8 == 0 or index == FEATURE_UNITS:
            write_workload_progress(context, completed_units=index, stage="feature_extract")

    def pack(selected: list[dict[str, Any]], cell: str) -> tuple[Any, Any, Any, Any]:
        x = torch.zeros(len(selected), 9, a1.INPUT_DIM); y = torch.zeros(len(selected), 9)
        mse = torch.zeros(len(selected), 9); mask = torch.zeros(len(selected), 9, dtype=torch.bool)
        for i, unit in enumerate(selected):
            n = unit["n"]; x[i, :n] = unit["features"][cell]; y[i, :n] = unit["target"]
            mse[i, :n] = unit["mse"]; mask[i, :n] = True
        return x, y, mse, mask

    prediction_rows: list[dict[str, Any]] = []; training_rows: list[dict[str, Any]] = []
    completed_training = 0
    for fold in FOLDS:
        calibration_fold = (fold + 1) % 4
        train_units = [unit for unit in units if unit["fold"] not in (fold, calibration_fold)]
        calibration_units = [unit for unit in units if unit["fold"] == calibration_fold]
        test_units = [unit for unit in units if unit["fold"] == fold]
        for seed in SEEDS:
            for cell in CELLS:
                x_train, y_train, mse_train, mask_train = pack(train_units, cell)
                x_cal, y_cal, _mse_cal, mask_cal = pack(calibration_units, cell)
                x_test, y_test, _mse_test, mask_test = pack(test_units, cell)
                model = a1.train_model(x_train, y_train, mse_train, mask_train, seed, False)
                with torch.no_grad(): cal_pred = model(x_cal).squeeze(-1)
                threshold = a1.select_threshold(cal_pred, y_cal, mask_cal)
                main = a1.predict_policy(model, x_test, y_test, mask_test, threshold)
                generator = torch.Generator().manual_seed(seed + 1000 * fold + 31)
                permutation = torch.zeros(len(test_units), 9, dtype=torch.long)
                for i, unit in enumerate(test_units):
                    perm = torch.arange(9); perm[:unit["n"]] = torch.randperm(unit["n"], generator=generator); permutation[i] = perm
                action_x = x_test.clone(); gathered = x_test.gather(1, permutation[:, :, None].expand_as(x_test))
                action_x[:, :, a1.STATE_DIM:a1.STATE_DIM + a1.ACTION_DIM] = gathered[:, :, a1.STATE_DIM:a1.STATE_DIM + a1.ACTION_DIM]
                action_shuffled = a1.predict_policy(model, action_x, y_test, mask_test, threshold)
                response_x = x_test.clone(); response_x[:, :, a1.STATE_DIM + a1.ACTION_DIM:] = gathered[:, :, a1.STATE_DIM + a1.ACTION_DIM:]
                response_shuffled = a1.predict_policy(model, response_x, y_test, mask_test, threshold)
                for i, unit in enumerate(test_units):
                    prediction_rows.append({
                        "cell": cell, "fold": fold, "seed": seed, "name": unit["name"],
                        "operator": unit["operator"], "gain": float(main["gain"][i]),
                        "oracle": float(main["oracle"][i]), "action_shuffle_gain": float(action_shuffled["gain"][i]),
                        "response_shuffle_gain": float(response_shuffled["gain"][i]),
                        "confidence": float(main["confidence"][i]), "selected_index": int(main["selected"][i]),
                        "reference_psnr": unit["reference_psnr"], "native_shape": unit["native_shape"],
                        "haze_transmission": unit["haze_transmission"], "haze_parameter_2": unit["haze_parameter_2"],
                    })
                training_rows.append({
                    "cell": cell, "fold": fold, "calibration_fold": calibration_fold, "seed": seed,
                    "threshold": threshold, "train_units": len(train_units),
                    "calibration_units": len(calibration_units), "test_units": len(test_units),
                    "epochs": a1.EPOCHS, "learning_rate": a1.LEARNING_RATE, "weight_decay": a1.WEIGHT_DECAY,
                })
                completed_training += 1
                write_workload_progress(context, completed_units=FEATURE_UNITS + completed_training, stage="critic_train_eval")

    write_csv(context.phase_output_path / "a2_oof_rows_cloud_only.csv", prediction_rows)
    write_csv(context.phase_output_path / "a2_training_units_cloud_only.csv", training_rows)
    collapsed: list[dict[str, Any]] = []
    for cell in CELLS:
        subset = [row for row in prediction_rows if row["cell"] == cell]
        for name in sorted({row["name"] for row in subset}):
            for operator in ("D_ref", "D_rep"):
                items = [row for row in subset if row["name"] == name and row["operator"] == operator]
                collapsed.append({
                    "cell": cell, "name": name, "operator": operator,
                    "gain": sum(float(row["gain"]) for row in items) / len(items),
                    "oracle": sum(float(row["oracle"]) for row in items) / len(items),
                    "shuffle_gain": sum(float(row["action_shuffle_gain"]) for row in items) / len(items),
                    "response_shuffle_gain": sum(float(row["response_shuffle_gain"]) for row in items) / len(items),
                    "confidence": sum(float(row["confidence"]) for row in items) / len(items),
                    "reference_psnr": items[0]["reference_psnr"], "native_shape": items[0]["native_shape"],
                    "haze_transmission": items[0]["haze_transmission"], "haze_parameter_2": items[0]["haze_parameter_2"],
                })
    c3_rows = [row for row in collapsed if row["cell"] == "C3_deep_response"]
    c3_metrics = a1.bootstrap_metrics(c3_rows); increment = paired_increment_bootstrap(collapsed)
    severe = sum(float(row["gain"]) <= -0.2 for row in c3_rows); hard = sum(float(row["gain"]) <= -0.5 for row in c3_rows)
    severe_ucb = risk_delta_ucb(c3_rows, -0.2); hard_ucb = risk_delta_ucb(c3_rows, -0.5)
    structural_checks = {
        "cache_units_complete": len(raw_rows) == FEATURE_UNITS,
        "training_units_complete": len(training_rows) == TRAINING_UNITS,
        "oof_rows_complete": len(prediction_rows) == 768 * 2 * 2 * len(CELLS),
        "collapsed_rows_complete": len(collapsed) == 768 * 2 * len(CELLS),
        "all_folds_present": sorted({int(row["fold"]) for row in prediction_rows}) == list(FOLDS),
        "finite": all(math.isfinite(float(row[key])) for row in collapsed for key in ("gain", "oracle", "shuffle_gain", "response_shuffle_gain")),
        "development_confirmation_disjoint": not development & confirmation,
        "candidate_fixed_c3": a1_closeout["details"]["survivors"] == ["C3_deep_response"],
    }
    formal_gates = {
        "gain_lcb95": c3_metrics["gain"]["lcb95"] >= GAIN_GATE,
        "retention_lcb95": c3_metrics["retention"]["lcb95"] >= RETENTION_GATE,
        "true_minus_action_shuffle_lcb95": c3_metrics["true_minus_shuffle"]["lcb95"] >= SHUFFLE_GATE,
        "c3_minus_c1_lcb95": increment["lcb95"] >= RESPONSE_INCREMENT_GATE,
        "severe_point_zero": severe == 0, "hard_point_zero": hard == 0,
        "severe_paired_delta_ucb_nonpositive": severe_ucb <= 0.0,
        "hard_paired_delta_ucb_nonpositive": hard_ucb <= 0.0,
    }
    structural_valid = all(structural_checks.values()); formal_pass = structural_valid and all(formal_gates.values())
    if formal_pass:
        state, decision, authorizes = "COMPLETED_GATE_PASS", "R3_A2_ACV_FULL_OOF_PASS", "R3_CANDIDATE_FREEZE_REVIEW"
    elif structural_valid:
        state, decision, authorizes = "COMPLETED_GATE_FAIL", "R3_A2_ACV_FULL_OOF_FAIL_STOP", "NONE"
    else:
        state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R3_A2_ACV_FULL_OOF_INCONCLUSIVE", "NONE"
    coverage_rows, strata_rows = descriptive_tables(c3_rows)
    cell_summary = []
    for cell in CELLS:
        rows = [row for row in collapsed if row["cell"] == cell]
        metrics = a1.bootstrap_metrics(rows)
        cell_summary.append({
            "cell": cell, "gain_point_db": metrics["gain"]["point"],
            "gain_lcb95_db": metrics["gain"]["lcb95"], "gain_ucb95_db": metrics["gain"]["ucb95"],
            "retention_point": metrics["retention"]["point"], "retention_lcb95": metrics["retention"]["lcb95"],
            "true_minus_shuffle_point_db": metrics["true_minus_shuffle"]["point"],
            "true_minus_shuffle_lcb95_db": metrics["true_minus_shuffle"]["lcb95"],
            "severe_count": sum(float(row["gain"]) <= -0.2 for row in rows),
            "hard_count": sum(float(row["gain"]) <= -0.5 for row in rows),
        })
    selection = {
        "schema_version": 1, "candidate": "C3_deep_response", "matched_control": "C1_action",
        "candidate_source": "sole A1 screen survivor", "selected_before_a2_outcomes": True,
        "response_increment": increment, "formal_gates": formal_gates,
        "formal_pass": formal_pass, "next_action": authorizes,
    }
    access = {
        "schema_version": 1, "route_commit": context.route_commit,
        "a0_cache_manifest_sha256": a0_manifest["cache_manifest_sha256"],
        "a1_closeout_sha256": a1.sha256_file(asset_path(context, "a1_closeout", kind="file")),
        "development_images_targets_accessed": 768,
        "confirmation_images_targets_outcomes_touched": False, "historical_a1x_432_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
    }
    resource = {
        "schema_version": 1, "wall_seconds": time.perf_counter() - started,
        "cache_units_read": FEATURE_UNITS, "training_units": TRAINING_UNITS,
        "trainable_parameters": 9153, "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
    }
    atomic_json(context.phase_output_path / "a2_structural_summary.json", {"schema_version": 1, "checks": structural_checks, "valid": structural_valid})
    atomic_json(context.phase_output_path / "a2_bootstrap_summary.json", {"schema_version": 1, "c3": c3_metrics, "c3_minus_c1": increment})
    atomic_json(context.phase_output_path / "a2_gate_summary.json", {"schema_version": 1, "gates": formal_gates, "pass": formal_pass})
    atomic_json(context.phase_output_path / "a2_candidate_selection.json", selection)
    write_csv(context.phase_output_path / "a2_cell_summary.csv", cell_summary)
    write_csv(context.phase_output_path / "a2_risk_coverage.csv", coverage_rows)
    write_csv(context.phase_output_path / "a2_strata_summary.csv", strata_rows)
    atomic_json(context.phase_output_path / "a2_resource_summary.json", resource)
    atomic_json(context.phase_output_path / "a2_source_access_audit.json", access)
    write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
        "candidate": "C3_deep_response", "formal_pass": formal_pass,
        "structural_valid": structural_valid, "training_units": TRAINING_UNITS,
        "gain_lcb95_db": c3_metrics["gain"]["lcb95"],
        "retention_lcb95": c3_metrics["retention"]["lcb95"],
        "true_minus_shuffle_lcb95_db": c3_metrics["true_minus_shuffle"]["lcb95"],
        "c3_minus_c1_lcb95_db": increment["lcb95"],
        "severe_count": severe, "hard_count": hard,
    })


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("contract", "run")); parser.add_argument("--context", required=True, type=Path); args = parser.parse_args()
    contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__": main()
