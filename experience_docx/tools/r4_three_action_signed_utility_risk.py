#!/usr/bin/env python3
"""Frozen R4 development test of a three-action signed utility-risk head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path, atomic_json, load_context, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)

ROUTE_ID = "haze4k_v5_r4_three_action_signed_utility_risk_20260718"
OPERATION_ID = "R4_D0_THREE_ACTION_SIGNED_UTILITY_RISK_DEV"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
CELLS = ("joint_utility_risk", "independent_scalar", "risk_disabled")
FOLDS = (0, 1, 2, 3)
SEEDS = (3407, 3411)
EPOCHS = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BATCH_SIZE = 64
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
FEATURE_UNITS = 1536
TRAINING_UNITS = len(FOLDS) * len(SEEDS) * len(CELLS)
TOTAL_UNITS = FEATURE_UNITS + TRAINING_UNITS
STATE_DIM = 40
ACTION_DIM = 5
RGB_DIM = 12
DEEP_DIM = 24
INPUT_DIM = STATE_DIM + ACTION_DIM + RGB_DIM + DEEP_DIM
HIDDEN_DIM = 64
GAIN_GATE = 0.020
RETENTION_GATE = 0.25
SHUFFLE_GATE = 0.005
INCREMENT_GATE = 0.005
COVERAGE_GATE = 0.10
SEVERE_RATE_GATE = 0.005
NATIVE_SHAPE_FLOOR = -0.020
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_psnr(mse: Any) -> Any:
    import torch
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-30))


def tensor_stats(value: Any) -> Any:
    import torch
    flat = value.float().flatten(2)
    return torch.cat(
        (flat.mean(2), flat.std(2, unbiased=False), flat.abs().mean(2), flat.abs().amax(2)),
        dim=1,
    )


def rgb_response_features(response: Any) -> Any:
    import torch.nn.functional as functional
    return tensor_stats(functional.adaptive_avg_pool2d(response.float(), (32, 32)))


def deep_response_features(response: Any) -> Any:
    import torch
    if response.shape[1] != 32:
        raise RuntimeError(f"expected 32 official encoder channels, got {response.shape[1]}")
    grouped = response.float().reshape(response.shape[0], 8, 4, *response.shape[-2:])
    flat = grouped.flatten(2)
    return torch.cat((flat.mean(2), flat.std(2, unbiased=False), flat.abs().amax(2)), dim=1)


def action_features(device: Any) -> Any:
    import torch
    return torch.tensor(
        ((1.0, 0.0, 0.0, 0.0, 0.0),
         (0.0, 1.0, 0.0, 1.0, 1.0),
         (0.0, 0.0, 1.0, -1.0, 1.0)),
        dtype=torch.float32, device=device,
    )


def build_head(outputs: int) -> Any:
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, HIDDEN_DIM), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_DIM, outputs),
    )


def head_loss(model: Any, features: Any, targets: Any, cell: str) -> Any:
    import torch
    import torch.nn.functional as functional
    raw = model(features)
    utility = raw[..., 0]
    utility_loss = functional.smooth_l1_loss(utility, targets)
    true_gap = targets.unsqueeze(2) - targets.unsqueeze(1)
    pred_gap = utility.unsqueeze(2) - utility.unsqueeze(1)
    pair_mask = true_gap.abs() > 1.0e-8
    pair = functional.softplus(-torch.sign(true_gap) * pred_gap)
    pair_loss = (pair * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    loss = utility_loss + 0.25 * pair_loss
    if cell == "joint_utility_risk":
        harm = (targets < 0.0).float()
        severe = (targets <= SEVERE_GAIN).float()
        loss = loss + 0.5 * functional.binary_cross_entropy_with_logits(raw[..., 1], harm)
        loss = loss + 0.5 * functional.binary_cross_entropy_with_logits(raw[..., 2], severe)
    return loss


def train_head(features: Any, targets: Any, seed: int, cell: str) -> Any:
    import torch
    torch.manual_seed(seed)
    outputs = 1 if cell == "independent_scalar" else 3
    model = build_head(outputs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 17)
    for _epoch in range(EPOCHS):
        for indices in torch.randperm(len(features), generator=generator).split(BATCH_SIZE):
            loss = head_loss(model, features[indices], targets[indices], cell)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def head_outputs(model: Any, features: Any, cell: str) -> tuple[Any, Any, Any]:
    import torch
    with torch.no_grad():
        raw = model(features)
    utility = raw[..., 0]
    if cell == "joint_utility_risk":
        return utility, raw[..., 1].sigmoid(), raw[..., 2].sigmoid()
    return utility, torch.zeros_like(utility), torch.zeros_like(utility)


def apply_policy(utility: Any, harm: Any, severe: Any, truth: Any,
                 threshold: dict[str, float], risk_enabled: bool) -> dict[str, Any]:
    import torch
    active = utility[:, 1:]
    active_index = active.argmax(1) + 1
    best_utility = utility.gather(1, active_index[:, None]).squeeze(1)
    margin = best_utility - utility[:, 0]
    best_harm = harm.gather(1, active_index[:, None]).squeeze(1)
    best_severe = severe.gather(1, active_index[:, None]).squeeze(1)
    allowed = margin >= threshold["margin"]
    if risk_enabled:
        allowed &= best_harm <= threshold["harm"]
        allowed &= best_severe <= threshold["severe"]
    selected = torch.where(allowed, active_index, torch.zeros_like(active_index))
    gain = truth.gather(1, selected[:, None]).squeeze(1)
    oracle = truth.max(1).values.clamp(min=0.0)
    negative_oracle = (truth[:, 2] > truth[:, 1]) & (truth[:, 2] > 0.0)
    return {
        "selected": selected, "gain": gain, "oracle": oracle,
        "margin": margin, "best_harm": best_harm, "best_severe": best_severe,
        "negative_oracle": negative_oracle,
    }


def calibrate(model: Any, features: Any, truth: Any, cell: str) -> dict[str, float]:
    import torch
    utility, harm, severe = head_outputs(model, features, cell)
    active_index = utility[:, 1:].argmax(1) + 1
    margin = utility.gather(1, active_index[:, None]).squeeze(1) - utility[:, 0]
    margin_grid = sorted(set(float(torch.quantile(margin, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.90)))
    risk_enabled = cell == "joint_utility_risk"
    harm_grid = (0.05, 0.10, 0.20, 0.50, 1.0) if risk_enabled else (1.0,)
    severe_grid = (0.01, 0.05, 0.10, 0.50, 1.0) if risk_enabled else (1.0,)
    feasible: list[tuple[float, float, float, float, dict[str, float]]] = []
    for margin_value in margin_grid:
        for harm_value in harm_grid:
            for severe_value in severe_grid:
                threshold = {"margin": margin_value, "harm": harm_value, "severe": severe_value}
                result = apply_policy(utility, harm, severe, truth, threshold, risk_enabled)
                gain = result["gain"]
                coverage = float((result["selected"] != 0).float().mean())
                severe_count = int((gain <= SEVERE_GAIN).sum())
                hard_count = int((gain <= HARD_GAIN).sum())
                if coverage >= COVERAGE_GATE and severe_count == 0 and hard_count == 0:
                    feasible.append((float(gain.mean()), -coverage, -margin_value, -harm_value, threshold))
    if not feasible:
        return {"margin": math.inf, "harm": 0.0, "severe": 0.0}
    return max(feasible, key=lambda item: item[:4])[-1]


def contract(context_path: Path) -> None:
    import torch
    from route_engineering_fixture import (
        assert_finite_tensors, assert_loss_decreased, assert_nonzero_gradients,
        assert_trainable_scope,
    )
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    torch.manual_seed(SEEDS[0])
    features = torch.randn(8, 3, INPUT_DIM)
    targets = torch.randn(8, 3) * 0.1
    targets[:, 0] = 0.0
    joint = build_head(3)
    scalar = build_head(1)
    before = float(head_loss(joint, features, targets, "joint_utility_risk"))
    optimizer = torch.optim.AdamW(joint.parameters(), lr=0.01)
    for _ in range(8):
        loss = head_loss(joint, features, targets, "joint_utility_risk")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    after = float(head_loss(joint, features, targets, "joint_utility_risk"))
    scope = assert_trainable_scope(joint, allowed_prefixes=("0", "2", "4"), required_prefixes=("4",))
    gradients = assert_nonzero_gradients(joint, required_prefixes=("0", "2", "4"))
    finite = assert_finite_tensors((("joint_output", joint(features)), ("scalar_output", scalar(features))))
    microfit = assert_loss_decreased(before, after)
    joint_count = sum(parameter.numel() for parameter in joint.parameters())
    scalar_count = sum(parameter.numel() for parameter in scalar.parameters())
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "three_actions_frozen": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
        "cells_frozen": CELLS == ("joint_utility_risk", "independent_scalar", "risk_disabled"),
        "folds_and_seeds_frozen": FOLDS == (0, 1, 2, 3) and SEEDS == (3407, 3411),
        "optimizer_frozen": EPOCHS == 32 and LEARNING_RATE == 1.0e-3 and WEIGHT_DECAY == 1.0e-4,
        "matched_parameter_budget": abs(joint_count - scalar_count) / joint_count <= 0.015,
        "formal_gates_frozen": GAIN_GATE == 0.020 and RETENTION_GATE == 0.25
        and SHUFFLE_GATE == 0.005 and INCREMENT_GATE == 0.005
        and COVERAGE_GATE == 0.10 and SEVERE_RATE_GATE == 0.005,
        "fixture_scope_valid": scope["trainable_parameter_count"] == joint_count,
        "fixture_gradients_valid": gradients["gradient_max_abs"] > 0.0,
        "fixture_finite": finite["finite_tensor_count"] == 2,
        "fixture_microfit": microfit["microfit_loss_after"] < microfit["microfit_loss_before"],
        "generic_progress_total": context.total_units == TOTAL_UNITS,
        "protected_roles_blocked": True,
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(context.phase_output_path / "r4_synthetic_contract.json", {
        "schema_version": 1, "checks": checks, "joint_parameters": joint_count,
        "scalar_parameters": scalar_count, "scope": scope, "gradients": gradients,
        "finite": finite, "microfit": microfit,
    })
    write_contract_result(context, checks=checks)


def bootstrap_metrics(rows: list[dict[str, Any]], independent: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    names = sorted({row["name"] for row in rows})
    operators = ("D_ref", "D_rep")
    keyed = {(row["name"], row["operator"]): row for row in rows}
    control = {(row["name"], row["operator"]): row for row in independent}
    fields = ("gain", "oracle", "shuffle_gain", "selected_rate", "negative_oracle")
    arrays = {
        operator: {field: np.asarray([float(keyed[(name, operator)][field]) for name in names]) for field in fields}
        for operator in operators
    }
    control_gain = {
        operator: np.asarray([float(control[(name, operator)]["gain"]) for name in names])
        for operator in operators
    }

    def evaluate(index: Any) -> dict[str, float]:
        gains = [float(arrays[op]["gain"][index].mean()) for op in operators]
        oracles = [float(arrays[op]["oracle"][index].mean()) for op in operators]
        shuffles = [float(arrays[op]["shuffle_gain"][index].mean()) for op in operators]
        coverages = [float(arrays[op]["selected_rate"][index].mean()) for op in operators]
        increments = [float((arrays[op]["gain"][index] - control_gain[op][index]).mean()) for op in operators]
        negative = []
        for op in operators:
            mask = arrays[op]["negative_oracle"][index] > 0.5
            negative.append(float(arrays[op]["gain"][index][mask].mean()) if mask.any() else math.nan)
        worst = min(range(2), key=lambda item: gains[item])
        return {
            "gain": gains[worst],
            "retention": gains[worst] / max(oracles[worst], 1.0e-12),
            "true_minus_shuffle": min(gain - shuffle for gain, shuffle in zip(gains, shuffles)),
            "joint_minus_independent": min(increments),
            "negative_oracle_gain": min(negative),
            "coverage": min(coverages),
        }

    full = np.arange(len(names))
    point = evaluate(full)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {key: [] for key in point}
    severe_rates = []
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(names), len(names))
        result = evaluate(index)
        for key, value in result.items():
            samples[key].append(value)
        severe_rates.append(max(float((arrays[op]["gain"][index] <= SEVERE_GAIN).mean()) for op in operators))
    output = {}
    for key, values in samples.items():
        vector = np.asarray(values, dtype=np.float64)
        output[key] = {
            "point": point[key], "lcb95": float(np.quantile(vector, 0.025)),
            "ucb95": float(np.quantile(vector, 0.975)),
        }
    output["severe_rate"] = {
        "point": max(float((arrays[op]["gain"] <= SEVERE_GAIN).mean()) for op in operators),
        "ucb95_one_sided": float(np.quantile(np.asarray(severe_rates), 0.95)),
    }
    return output


def simple_cell_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    operators = ("D_ref", "D_rep")
    gain = [sum(float(row["gain"]) for row in rows if row["operator"] == op) /
            sum(row["operator"] == op for row in rows) for op in operators]
    oracle = [sum(float(row["oracle"]) for row in rows if row["operator"] == op) /
              sum(row["operator"] == op for row in rows) for op in operators]
    coverage = [sum(float(row["selected_rate"]) for row in rows if row["operator"] == op) /
                sum(row["operator"] == op for row in rows) for op in operators]
    worst = min(range(2), key=lambda item: gain[item])
    return {"gain_point_db": gain[worst], "retention_point": gain[worst] / max(oracle[worst], 1.0e-12),
            "coverage_point": min(coverage)}


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ledger_path = asset_path(context, "r4_ledger", kind="file")
    cache_manifest_path = asset_path(context, "a0_cache_manifest", kind="file")
    raw_manifest_path = asset_path(context, "a0_raw_manifest", kind="file")
    cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    data_root = asset_path(context, "haze4k_data", kind="directory")
    checkpoint_path = asset_path(context, "official_checkpoint", kind="file")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_rows) != FEATURE_UNITS or cache_manifest.get("cache_manifest_sha256") != "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d":
        raise RuntimeError("sealed A0 cache identity mismatch")
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}
    fold_lookup = {name: fold for fold, names in folds.items() for name in names}
    development = set(ledger["roles"]["development"])
    confirmation = set(ledger["roles"]["confirmation"])
    if sorted(len(value) for value in folds.values()) != [192, 192, 192, 192] or set(fold_lookup) != development or development & confirmation:
        raise RuntimeError("S0 development role/fold contract mismatch")

    def load_label(name: str) -> Any:
        stem, extension = os.path.splitext(name)
        for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
            path = data_root / "train/gt" / candidate
            if path.is_file():
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB")).copy()
                return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        raise FileNotFoundError(name)

    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R4 frozen official encoder feature extraction requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    sys.path.insert(0, str(context.remote_repo / "Dehazing/ITS"))
    from models.ConvIR import build_net
    encoder = build_net("base", "Haze4K").to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder.load_state_dict(checkpoint, strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    units: list[dict[str, Any]] = []
    fixed_action = action_features(torch.device("cpu"))
    for index, row in enumerate(raw_rows, 1):
        unit_path = cache_root / f"{row['unit_key']}.pt"
        if sha256_file(unit_path) != row["cache_sha256"]:
            raise RuntimeError(f"cache unit hash mismatch: {row['unit_key']}")
        payload = torch.load(unit_path, map_location="cpu", weights_only=False)
        names = list(payload["candidate_names"])
        selected_indices = [names.index(action) for action in ACTIONS]
        name = payload["name"]
        base = payload["base"].float()
        step = payload["step"].float()
        candidate_delta = payload["candidates"].float()[selected_indices]
        current = payload["current"].float()
        support = payload["support"].float()
        label = load_label(name)
        if label.shape[-2:] != base.shape[-2:]:
            label = label[:, :, :base.shape[-2], :base.shape[-1]]
        reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
        renders = torch.clamp(base + 0.25 * (step + candidate_delta), 0.0, 1.0)
        target = metric_psnr((renders - label).square().mean((1, 2, 3))) - metric_psnr((reference - label).square().mean())
        state = torch.cat((tensor_stats(base), tensor_stats(step), tensor_stats(current), tensor_stats(support)), dim=1)
        if state.shape[1] != STATE_DIM:
            raise RuntimeError(f"state feature identity mismatch: {state.shape}")
        rgb = rgb_response_features(renders - reference)
        with torch.no_grad():
            encoder_inputs = torch.cat((reference, renders), dim=0).to(device)
            deep_all = encoder.Encoder[0](encoder.feat_extract[0](encoder_inputs))
            deep = deep_response_features(deep_all[1:] - deep_all[0:1]).cpu()
        features = torch.cat((state.expand(3, -1), fixed_action, rgb, deep), dim=1)
        if features.shape != (3, INPUT_DIM) or not bool(torch.isfinite(features).all()):
            raise RuntimeError("candidate-conditioned feature contract failed")
        units.append({
            "name": name, "operator": payload["operator"], "fold": fold_lookup[name],
            "features": features, "target": target, "native_shape": row["native_shape"],
        })
        if index % 8 == 0 or index == FEATURE_UNITS:
            write_workload_progress(context, completed_units=index, stage="feature_extract")

    prediction_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    completed_training = 0
    for fold in FOLDS:
        calibration_fold = (fold + 1) % 4
        train_units = [unit for unit in units if unit["fold"] not in (fold, calibration_fold)]
        calibration_units = [unit for unit in units if unit["fold"] == calibration_fold]
        test_units = [unit for unit in units if unit["fold"] == fold]
        x_train = torch.stack([unit["features"] for unit in train_units])
        y_train = torch.stack([unit["target"] for unit in train_units])
        x_cal = torch.stack([unit["features"] for unit in calibration_units])
        y_cal = torch.stack([unit["target"] for unit in calibration_units])
        x_test = torch.stack([unit["features"] for unit in test_units])
        y_test = torch.stack([unit["target"] for unit in test_units])
        x_shuffle = x_test.clone()
        x_shuffle[:, 1] = x_test[:, 2]
        x_shuffle[:, 2] = x_test[:, 1]
        for seed in SEEDS:
            for cell in CELLS:
                model = train_head(x_train, y_train, seed, cell)
                threshold = calibrate(model, x_cal, y_cal, cell)
                utility, harm, severe = head_outputs(model, x_test, cell)
                main = apply_policy(utility, harm, severe, y_test, threshold, cell == "joint_utility_risk")
                shuffle_utility, shuffle_harm, shuffle_severe = head_outputs(model, x_shuffle, cell)
                shuffled = apply_policy(shuffle_utility, shuffle_harm, shuffle_severe, y_test, threshold, cell == "joint_utility_risk")
                for item, unit in enumerate(test_units):
                    prediction_rows.append({
                        "cell": cell, "fold": fold, "seed": seed, "name": unit["name"],
                        "operator": unit["operator"], "gain": float(main["gain"][item]),
                        "oracle": float(main["oracle"][item]), "shuffle_gain": float(shuffled["gain"][item]),
                        "selected": int(main["selected"][item]), "margin": float(main["margin"][item]),
                        "negative_oracle": bool(main["negative_oracle"][item]),
                        "native_shape": unit["native_shape"],
                    })
                training_rows.append({
                    "cell": cell, "fold": fold, "calibration_fold": calibration_fold, "seed": seed,
                    "margin_threshold": threshold["margin"], "harm_threshold": threshold["harm"],
                    "severe_threshold": threshold["severe"], "train_units": len(train_units),
                    "calibration_units": len(calibration_units), "test_units": len(test_units),
                    "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                })
                completed_training += 1
                write_workload_progress(context, completed_units=FEATURE_UNITS + completed_training, stage="head_train_eval")

    write_csv(context.phase_output_path / "r4_oof_rows_cloud_only.csv", prediction_rows)
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
                    "shuffle_gain": sum(float(row["shuffle_gain"]) for row in items) / len(items),
                    "selected_rate": sum(int(row["selected"] != 0) for row in items) / len(items),
                    "negative_rate": sum(int(row["selected"] == 2) for row in items) / len(items),
                    "negative_oracle": bool(items[0]["negative_oracle"]),
                    "native_shape": items[0]["native_shape"],
                })
    joint_rows = [row for row in collapsed if row["cell"] == "joint_utility_risk"]
    independent_rows = [row for row in collapsed if row["cell"] == "independent_scalar"]
    bootstrap = bootstrap_metrics(joint_rows, independent_rows)
    severe_count = sum(float(row["gain"]) <= SEVERE_GAIN for row in joint_rows)
    hard_count = sum(float(row["gain"]) <= HARD_GAIN for row in joint_rows)

    strata_rows: list[dict[str, Any]] = []
    native_shape_gate = True
    for operator in ("D_ref", "D_rep"):
        for shape in sorted({row["native_shape"] for row in joint_rows if row["operator"] == operator}):
            items = [row for row in joint_rows if row["operator"] == operator and row["native_shape"] == shape]
            mean_gain = sum(float(row["gain"]) for row in items) / len(items)
            passes = mean_gain >= NATIVE_SHAPE_FLOOR
            native_shape_gate &= passes
            strata_rows.append({
                "operator": operator, "native_shape": shape, "count": len(items),
                "mean_gain_db": mean_gain, "severe_count": sum(float(row["gain"]) <= SEVERE_GAIN for row in items),
                "hard_count": sum(float(row["gain"]) <= HARD_GAIN for row in items), "passes_floor": passes,
            })
    structural_checks = {
        "feature_units_complete": len(units) == FEATURE_UNITS,
        "training_units_complete": len(training_rows) == TRAINING_UNITS,
        "oof_rows_complete": len(prediction_rows) == 768 * 2 * len(SEEDS) * len(CELLS),
        "collapsed_rows_complete": len(collapsed) == 768 * 2 * len(CELLS),
        "all_folds_present": sorted({int(row["fold"]) for row in prediction_rows}) == list(FOLDS),
        "all_cells_present": sorted({row["cell"] for row in prediction_rows}) == sorted(CELLS),
        "finite": all(math.isfinite(float(row[key])) for row in collapsed for key in ("gain", "oracle", "shuffle_gain", "selected_rate")),
        "development_confirmation_disjoint": not development & confirmation,
        "actions_exactly_frozen": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
    }
    formal_gates = {
        "gain_lcb95": bootstrap["gain"]["lcb95"] >= GAIN_GATE,
        "retention_lcb95": bootstrap["retention"]["lcb95"] >= RETENTION_GATE,
        "true_minus_sign_shuffle_lcb95": bootstrap["true_minus_shuffle"]["lcb95"] >= SHUFFLE_GATE,
        "joint_minus_independent_lcb95": bootstrap["joint_minus_independent"]["lcb95"] >= INCREMENT_GATE,
        "negative_oracle_subset_gain_lcb95": bootstrap["negative_oracle_gain"]["lcb95"] > 0.0,
        "coverage_lcb95": bootstrap["coverage"]["lcb95"] >= COVERAGE_GATE,
        "severe_point_zero": severe_count == 0,
        "hard_point_zero": hard_count == 0,
        "severe_rate_ucb95": bootstrap["severe_rate"]["ucb95_one_sided"] <= SEVERE_RATE_GATE,
        "native_shape_floor": native_shape_gate,
    }
    structural_valid = all(structural_checks.values())
    formal_pass = structural_valid and all(formal_gates.values())
    if formal_pass:
        state, decision, authorizes = "COMPLETED_GATE_PASS", "R4_D0_THREE_ACTION_DEV_PASS", "R4_CONFIRMATION_REVIEW_ONLY"
    elif structural_valid:
        state, decision, authorizes = "COMPLETED_GATE_FAIL", "R4_D0_THREE_ACTION_DEV_FAIL_STOP", "NONE"
    else:
        state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R4_D0_THREE_ACTION_DEV_INCONCLUSIVE", "NONE"

    cell_summary = []
    risk_coverage = []
    for cell in CELLS:
        rows = [row for row in collapsed if row["cell"] == cell]
        metrics = simple_cell_metrics(rows)
        cell_summary.append({
            "cell": cell, **metrics,
            "severe_count": sum(float(row["gain"]) <= SEVERE_GAIN for row in rows),
            "hard_count": sum(float(row["gain"]) <= HARD_GAIN for row in rows),
            "negative_selection_rate": sum(float(row["negative_rate"]) for row in rows) / len(rows),
        })
        for operator in ("D_ref", "D_rep"):
            items = [row for row in rows if row["operator"] == operator]
            risk_coverage.append({
                "cell": cell, "operator": operator,
                "coverage": sum(float(row["selected_rate"]) for row in items) / len(items),
                "mean_gain_db": sum(float(row["gain"]) for row in items) / len(items),
                "severe_count": sum(float(row["gain"]) <= SEVERE_GAIN for row in items),
                "hard_count": sum(float(row["gain"]) <= HARD_GAIN for row in items),
            })
    contract_summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "question": "Can a joint candidate-conditioned utility/harm/severe head safely recover signed positive/negative action utility over exact no-op?",
        "population": "frozen R3 S0 768 development images", "analysis_unit": "clean-reference image/group; D_ref and D_rep paired",
        "actions": list(ACTIONS), "cells": list(CELLS), "folds": list(FOLDS), "seeds": list(SEEDS),
        "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gates": {"gain_lcb95_db": GAIN_GATE, "retention_lcb95": RETENTION_GATE,
                  "true_minus_sign_shuffle_lcb95_db": SHUFFLE_GATE, "joint_minus_independent_lcb95_db": INCREMENT_GATE,
                  "negative_oracle_subset_gain_lcb95_db": 0.0, "coverage_lcb95": COVERAGE_GATE,
                  "severe_count": 0, "hard_count": 0, "severe_rate_ucb95": SEVERE_RATE_GATE,
                  "native_shape_mean_floor_db": NATIVE_SHAPE_FLOOR},
        "protected_data": {"confirmation": "prohibited", "canary": "prohibited", "locked_test": "prohibited"},
    }
    access = {
        "schema_version": 1, "route_commit": context.route_commit,
        "ledger_sha256": sha256_file(ledger_path), "a0_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "official_checkpoint_sha256": sha256_file(checkpoint_path),
        "development_images_targets_accessed": 768,
        "confirmation_images_targets_outcomes_touched": False,
        "historical_a1x_432_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
    }
    resource = {
        "schema_version": 1, "wall_seconds": time.perf_counter() - started,
        "feature_units": FEATURE_UNITS, "training_units": TRAINING_UNITS,
        "joint_trainable_parameters": sum(parameter.numel() for parameter in build_head(3).parameters()),
        "scalar_trainable_parameters": sum(parameter.numel() for parameter in build_head(1).parameters()),
        "conv_ir_trainable_parameters": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
    }
    atomic_json(context.phase_output_path / "r4_contract_summary.json", contract_summary)
    atomic_json(context.phase_output_path / "r4_structural_summary.json", {"schema_version": 1, "checks": structural_checks, "valid": structural_valid})
    atomic_json(context.phase_output_path / "r4_bootstrap_summary.json", {"schema_version": 1, "joint": bootstrap})
    atomic_json(context.phase_output_path / "r4_gate_summary.json", {"schema_version": 1, "gates": formal_gates, "pass": formal_pass})
    write_csv(context.phase_output_path / "r4_cell_summary.csv", cell_summary)
    write_csv(context.phase_output_path / "r4_risk_coverage.csv", risk_coverage)
    write_csv(context.phase_output_path / "r4_strata_summary.csv", strata_rows)
    write_csv(context.phase_output_path / "r4_training_summary.csv", training_rows)
    atomic_json(context.phase_output_path / "r4_resource_summary.json", resource)
    atomic_json(context.phase_output_path / "r4_source_access_audit.json", access)
    write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
        "formal_pass": formal_pass, "structural_valid": structural_valid,
        "gain_lcb95_db": bootstrap["gain"]["lcb95"],
        "retention_lcb95": bootstrap["retention"]["lcb95"],
        "true_minus_sign_shuffle_lcb95_db": bootstrap["true_minus_shuffle"]["lcb95"],
        "joint_minus_independent_lcb95_db": bootstrap["joint_minus_independent"]["lcb95"],
        "negative_oracle_subset_gain_lcb95_db": bootstrap["negative_oracle_gain"]["lcb95"],
        "coverage_lcb95": bootstrap["coverage"]["lcb95"],
        "severe_count": severe_count, "hard_count": hard_count,
        "severe_rate_ucb95": bootstrap["severe_rate"]["ucb95_one_sided"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__":
    main()
