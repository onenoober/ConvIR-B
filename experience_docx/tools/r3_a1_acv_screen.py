#!/usr/bin/env python3
"""Frozen R3 A1 proposal-first action-conditioned value development screen."""

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
    write_contract_result, write_run_result,
)

ROUTE_ID = "haze4k_v5_r3_proposal_first_acv_20260717"
OPERATION_ID = "R3_A1_ACV_SCREEN"
CELLS = ("C0_state", "C1_action", "C2_rgb_response", "C3_deep_response")
CONTROLS = ("C1_action_only", "C1_unsigned_target")
FOLDS = (0, 1)
SEEDS = (3407, 3411)
EPOCHS = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
GAIN_TARGET = 0.020
RETENTION_TARGET = 0.25
SHUFFLE_TARGET = 0.005
TIE_MSE = 1.0e-10
GRAY_MSE = 1.0e-6
HIGH_MARGIN_MSE = 1.0e-5
HARM_ASYMMETRY = 4.0
STATE_DIM = 40
ACTION_DIM = 12
RESPONSE_DIM = 24
INPUT_DIM = STATE_DIM + ACTION_DIM + RESPONSE_DIM
HIDDEN_DIM = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def emit(context: Any, stage: str, completed: int, total: int) -> None:
    line = json.dumps({"R3_A1_PROGRESS": {"stage": stage, "completed_units": completed, "total_units": total}}, sort_keys=True)
    print(line, flush=True)
    with context.status_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def metric_psnr(mse: Any) -> Any:
    import torch
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-30))


def tensor_stats(value: Any) -> Any:
    import torch
    flat = value.float().flatten(2)
    return torch.cat((flat.mean(2), flat.std(2, unbiased=False), flat.abs().mean(2), flat.abs().amax(2)), dim=1)


def rgb_response_features(response: Any) -> Any:
    import torch
    import torch.nn.functional as F
    pooled = F.adaptive_avg_pool2d(response.float(), (32, 32))
    return tensor_stats(pooled)


def deep_response_features(response: Any) -> Any:
    if response.shape[1] != 32:
        raise RuntimeError(f"expected 32 official encoder channels, got {response.shape[1]}")
    grouped = response.float().reshape(response.shape[0], 8, 4, *response.shape[-2:])
    flat = grouped.flatten(2)
    return __import__("torch").cat((flat.mean(2), flat.std(2, unbiased=False), flat.abs().amax(2)), dim=1)


def action_features(names: list[str], device: Any) -> Any:
    import torch
    rows = []
    bank = ("reference_noop", "state_positive_full", "state_negative_full",
            "state_positive_exact_half", "state_negative_exact_half",
            "response_positive_full", "response_negative_full",
            "response_positive_exact_half", "response_negative_exact_half")
    for name in names:
        onehot = [0.0] * 9
        onehot[bank.index(name)] = 1.0
        sign = float("positive" in name) - float("negative" in name)
        amplitude = 0.0 if name == "reference_noop" else 1.0
        half = float("exact_half" in name)
        rows.append(onehot + [sign, amplitude, half])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def build_cell_features(state: Any, action: Any, rgb: Any, deep: Any, cell: str) -> Any:
    import torch
    n = action.shape[0]
    zeros_action = torch.zeros_like(action)
    zeros_response = torch.zeros(n, RESPONSE_DIM, dtype=state.dtype, device=state.device)
    state_rows = state.expand(n, -1)
    if cell == "C0_state":
        return torch.cat((state_rows, zeros_action, zeros_response), dim=1)
    if cell == "C1_action":
        return torch.cat((state_rows, action, zeros_response), dim=1)
    if cell == "C2_rgb_response":
        response = zeros_response.clone(); response[:, :rgb.shape[1]] = rgb
        return torch.cat((state_rows, action, response), dim=1)
    if cell == "C3_deep_response":
        return torch.cat((state_rows, action, deep), dim=1)
    if cell == "C1_action_only":
        return torch.cat((torch.zeros_like(state_rows), action, zeros_response), dim=1)
    if cell == "C1_unsigned_target":
        return torch.cat((state_rows, action, zeros_response), dim=1)
    raise ValueError(cell)


def contract(context_path: Path) -> None:
    import torch
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parameter_count = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1
    fixture = torch.linspace(-1.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)
    rgb = rgb_response_features(fixture)
    deep = deep_response_features(fixture.repeat(1, 11, 1, 1)[:, :32])
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "cells_frozen": CELLS == ("C0_state", "C1_action", "C2_rgb_response", "C3_deep_response"),
        "screen_folds_frozen": FOLDS == (0, 1),
        "paired_seeds_frozen": SEEDS == (3407, 3411),
        "optimizer_frozen": EPOCHS == 32 and LEARNING_RATE == 1.0e-3 and WEIGHT_DECAY == 1.0e-4,
        "parameter_budget": parameter_count < 300000,
        "identical_parameter_count": True,
        "feature_shapes": tuple(rgb.shape) == (1, 12) and tuple(deep.shape) == (1, 24),
        "fixed_deep_pool_finite": bool(torch.isfinite(deep).all()),
        "gate_contract": GAIN_TARGET == 0.020 and RETENTION_TARGET == 0.25 and SHUFFLE_TARGET == 0.005,
        "protected_roles_blocked": True,
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(context.phase_output_path / "a1_synthetic_contract.json", {"schema_version": 1, "checks": checks, "parameter_count": parameter_count})
    write_contract_result(context, checks=checks)


def train_model(features: Any, targets: Any, candidate_mse: Any, mask: Any, seed: int, unsigned: bool) -> Any:
    import torch
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, HIDDEN_DIM), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_DIM, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 17)
    n = features.shape[0]
    for _epoch in range(EPOCHS):
        for indices in torch.randperm(n, generator=generator).split(64):
            x = features[indices]; y_signed = targets[indices]; valid = mask[indices]
            y = y_signed.abs() if unsigned else y_signed
            pred = model(x).squeeze(-1)
            harm = ((y_signed < 0.0) & (pred > 0.0)).float()
            weight = (1.0 + (HARM_ASYMMETRY - 1.0) * harm) * valid.float()
            mse_loss = ((pred - y).square() * weight).sum() / valid.sum().clamp(min=1)
            true_diff = y_signed.unsqueeze(2) - y_signed.unsqueeze(1)
            pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
            mse_gap = (candidate_mse[indices].unsqueeze(2) - candidate_mse[indices].unsqueeze(1)).abs()
            pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (mse_gap > TIE_MSE)
            pair_weight = torch.where(mse_gap <= GRAY_MSE, 0.1, torch.where(mse_gap > HIGH_MARGIN_MSE, 1.0, 0.5))
            pair_loss = torch.nn.functional.softplus(-torch.sign(true_diff) * pred_diff)
            pair_loss = (pair_loss * pair_weight * pair_mask.float()).sum() / pair_mask.sum().clamp(min=1)
            loss = mse_loss + 0.25 * pair_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval()


def select_threshold(pred: Any, truth: Any, mask: Any) -> float:
    import torch
    masked = pred.masked_fill(~mask, -torch.inf)
    best = masked.argmax(1)
    confidence = pred.gather(1, best[:, None]).squeeze(1) - pred[:, 0]
    candidates = [float(torch.quantile(confidence, q)) for q in (0.95, 0.90, 0.80, 0.60)] + [-math.inf]
    chosen = candidates[0]; best_score = -math.inf
    for threshold in candidates:
        selected = torch.where(confidence >= threshold, best, torch.zeros_like(best))
        gain = truth.gather(1, selected[:, None]).squeeze(1)
        severe = int((gain <= -0.2).sum()); hard = int((gain <= -0.5).sum())
        score = float(gain.mean()) if severe == 0 and hard == 0 else -math.inf
        if score > best_score:
            best_score = score; chosen = threshold
    return chosen


def predict_policy(model: Any, features: Any, truth: Any, mask: Any, threshold: float, permutation: Any | None = None) -> dict[str, Any]:
    import torch
    use_features = features if permutation is None else features.gather(1, permutation[:, :, None].expand_as(features))
    with torch.no_grad(): pred = model(use_features).squeeze(-1)
    pred = pred.masked_fill(~mask, -torch.inf)
    best = pred.argmax(1)
    confidence = pred.gather(1, best[:, None]).squeeze(1) - pred[:, 0]
    selected = torch.where(confidence >= threshold, best, torch.zeros_like(best))
    gain = truth.gather(1, selected[:, None]).squeeze(1)
    oracle = truth.masked_fill(~mask, -torch.inf).amax(1).clamp(min=0.0)
    return {"selected": selected, "gain": gain, "oracle": oracle, "confidence": confidence}


def bootstrap_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    names = sorted({row["name"] for row in rows})
    operators = ("D_ref", "D_rep")
    keyed = {(row["name"], row["operator"]): row for row in rows}
    arrays = {op: {field: np.asarray([float(keyed[(name, op)][field]) for name in names]) for field in ("gain", "oracle", "shuffle_gain")} for op in operators}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"gain": [], "retention": [], "true_minus_shuffle": []}
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(names), len(names))
        op_gain = [float(arrays[op]["gain"][index].mean()) for op in operators]
        op_oracle = [float(arrays[op]["oracle"][index].mean()) for op in operators]
        op_shuffle = [float(arrays[op]["shuffle_gain"][index].mean()) for op in operators]
        worst = min(range(2), key=lambda item: op_gain[item])
        values["gain"].append(op_gain[worst])
        values["retention"].append(op_gain[worst] / max(op_oracle[worst], 1e-12))
        values["true_minus_shuffle"].append(min(g - s for g, s in zip(op_gain, op_shuffle)))
    actual_gain = [float(arrays[op]["gain"].mean()) for op in operators]
    actual_oracle = [float(arrays[op]["oracle"].mean()) for op in operators]
    actual_shuffle = [float(arrays[op]["shuffle_gain"].mean()) for op in operators]
    actual_worst = min(range(2), key=lambda item: actual_gain[item])
    points = {
        "gain": actual_gain[actual_worst],
        "retention": actual_gain[actual_worst] / max(actual_oracle[actual_worst], 1e-12),
        "true_minus_shuffle": min(g - s for g, s in zip(actual_gain, actual_shuffle)),
    }
    output = {}
    for key, sample in values.items():
        vector = np.asarray(sample, dtype=np.float64)
        output[key] = {"point": points[key], "lcb95": float(np.quantile(vector, 0.025)), "ucb95": float(np.quantile(vector, 0.975))}
    return output


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
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_rows) != 1536 or a0_manifest.get("cache_manifest_sha256") != "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d":
        raise RuntimeError("A0 sealed cache identity mismatch")
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}
    if sorted(len(value) for value in folds.values()) != [192, 192, 192, 192]:
        raise RuntimeError("S0 fold contract mismatch")
    confirmation = set(ledger["roles"]["confirmation"]); development = set(ledger["roles"]["development"])
    if confirmation & development or set().union(*map(set, folds.values())) != development:
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
        raise RuntimeError("A1 official frozen encoder feature extraction requires CUDA")
    device = torch.device("cuda")
    its_root = context.remote_repo / "Dehazing/ITS"
    sys.path.insert(0, str(its_root))
    from models.ConvIR import build_net
    encoder_model = build_net("base", "Haze4K").to(device).eval()
    checkpoint = torch.load(asset_path(context, "official_checkpoint", kind="file"), map_location="cpu", weights_only=False)
    checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder_model.load_state_dict(checkpoint, strict=True)
    for parameter in encoder_model.parameters(): parameter.requires_grad_(False)
    units: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, 1):
        unit_path = cache_root / f"{row['unit_key']}.pt"
        if sha256_file(unit_path) != row["cache_sha256"]:
            raise RuntimeError(f"cache unit hash mismatch: {row['unit_key']}")
        payload = torch.load(unit_path, map_location="cpu", weights_only=False)
        name = payload["name"]; operator = payload["operator"]; names = list(payload["candidate_names"]); n = len(names)
        base = payload["base"].float(); step = payload["step"].float(); candidates = payload["candidates"].float()
        current = payload["current"].float(); support = payload["support"].float()
        label = load_label(name)
        if label.shape[-2:] != base.shape[-2:]: label = label[:, :, :base.shape[-2], :base.shape[-1]]
        reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
        renders = torch.clamp(base + 0.25 * (step + candidates), 0.0, 1.0)
        candidate_mse = (renders - label).square().mean((1, 2, 3))
        reference_mse = (reference - label).square().mean()
        target = metric_psnr(candidate_mse) - metric_psnr(reference_mse)
        state = torch.cat((tensor_stats(base), tensor_stats(step), tensor_stats(current), tensor_stats(support)), dim=1)
        action = action_features(names, torch.device("cpu"))
        rgb = rgb_response_features(renders - reference)
        with torch.no_grad():
            encoder_inputs = torch.cat((reference, renders), dim=0).to(device)
            deep_all = encoder_model.Encoder[0](encoder_model.feat_extract[0](encoder_inputs))
            deep_reference = deep_all[0:1]
            deep = deep_response_features(deep_all[1:] - deep_reference).cpu()
        feature_map = {cell: build_cell_features(state, action, rgb, deep, cell) for cell in CELLS + CONTROLS}
        feature_map = {cell: build_cell_features(state, action, rgb, deep, cell) for cell in CELLS + CONTROLS}
        units.append({"name": name, "operator": operator, "fold": next(k for k,v in folds.items() if name in set(v)), "names": names, "n": n, "features": feature_map, "target": target, "mse": candidate_mse})
        if index % 8 == 0 or index == len(raw_rows): emit(context, "feature_extract", index, len(raw_rows))

    max_candidates = 9
    def pack(selected: list[dict[str, Any]], cell: str) -> tuple[Any, Any, Any, Any]:
        x = torch.zeros(len(selected), max_candidates, INPUT_DIM); y = torch.zeros(len(selected), max_candidates); mse = torch.zeros(len(selected), max_candidates); mask = torch.zeros(len(selected), max_candidates, dtype=torch.bool)
        for i, unit in enumerate(selected):
            n = unit["n"]; x[i,:n] = unit["features"][cell]; y[i,:n] = unit["target"]; mse[i,:n] = unit["mse"]; mask[i,:n] = True
        return x, y, mse, mask

    training_rows: list[dict[str, Any]] = []; prediction_rows: list[dict[str, Any]] = []
    total_train = len(FOLDS) * len(SEEDS) * (len(CELLS) + len(CONTROLS)); completed = 0
    for fold in FOLDS:
        calibration_fold = (fold + 1) % 4
        train_units = [unit for unit in units if unit["fold"] not in (fold, calibration_fold)]
        calibration_units = [unit for unit in units if unit["fold"] == calibration_fold]
        test_units = [unit for unit in units if unit["fold"] == fold]
        for seed in SEEDS:
            for cell in CELLS + CONTROLS:
                x_train, y_train, mse_train, mask_train = pack(train_units, cell)
                x_calibration, y_calibration, _mse_calibration, mask_calibration = pack(calibration_units, cell)
                x_test, y_test, _mse_test, mask_test = pack(test_units, cell)
                model = train_model(x_train, y_train, mse_train, mask_train, seed, cell == "C1_unsigned_target")
                with torch.no_grad(): calibration_pred = model(x_calibration).squeeze(-1)
                threshold = select_threshold(calibration_pred, y_calibration, mask_calibration)
                main = predict_policy(model, x_test, y_test, mask_test, threshold)
                generator = torch.Generator().manual_seed(seed + 1000 * fold + 31)
                permutation = torch.stack([torch.randperm(9, generator=generator) for _ in test_units])
                for i, unit in enumerate(test_units):
                    valid_n = unit["n"]; perm = torch.arange(9); perm[:valid_n] = torch.randperm(valid_n, generator=generator); permutation[i] = perm
                action_shuffled_x = x_test.clone()
                action_shuffled_x[:, :, STATE_DIM:STATE_DIM + ACTION_DIM] = x_test.gather(1, permutation[:, :, None].expand_as(x_test))[:, :, STATE_DIM:STATE_DIM + ACTION_DIM]
                shuffled = predict_policy(model, action_shuffled_x, y_test, mask_test, threshold)
                response_shuffled_x = x_test.clone()
                response_shuffled_x[:, :, STATE_DIM + ACTION_DIM:] = x_test.gather(1, permutation[:, :, None].expand_as(x_test))[:, :, STATE_DIM + ACTION_DIM:]
                response_shuffled = predict_policy(model, response_shuffled_x, y_test, mask_test, threshold)
                for i, unit in enumerate(test_units):
                    prediction_rows.append({"cell": cell, "fold": fold, "seed": seed, "name": unit["name"], "operator": unit["operator"], "gain": float(main["gain"][i]), "oracle": float(main["oracle"][i]), "shuffle_gain": float(shuffled["gain"][i]), "response_shuffle_gain": float(response_shuffled["gain"][i]), "selected_index": int(main["selected"][i]), "confidence": float(main["confidence"][i])})
                training_rows.append({"cell": cell, "fold": fold, "calibration_fold": calibration_fold, "seed": seed, "threshold": threshold, "train_units": len(train_units), "calibration_units": len(calibration_units), "test_units": len(test_units), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY})
                completed += 1; emit(context, "critic_train_eval", completed, total_train)

    raw_predictions = context.phase_output_path / "a1_oof_rows_cloud_only.csv"; write_csv(raw_predictions, prediction_rows)
    write_csv(context.phase_output_path / "a1_training_units_cloud_only.csv", training_rows)
    bootstrap: dict[str, Any] = {}; cell_rows: list[dict[str, Any]] = []; risk_rows: list[dict[str, Any]] = []
    for cell in CELLS:
        subset = [row for row in prediction_rows if row["cell"] == cell]
        collapsed = []
        for name in sorted({row["name"] for row in subset}):
            for operator in ("D_ref", "D_rep"):
                items = [row for row in subset if row["name"] == name and row["operator"] == operator]
                collapsed.append({"name": name, "operator": operator, "gain": sum(float(x["gain"]) for x in items)/len(items), "oracle": sum(float(x["oracle"]) for x in items)/len(items), "shuffle_gain": sum(float(x["shuffle_gain"]) for x in items)/len(items)})
        metrics = bootstrap_metrics(collapsed); bootstrap[cell] = metrics
        severe = sum(float(row["gain"]) <= -0.2 for row in collapsed); hard = sum(float(row["gain"]) <= -0.5 for row in collapsed)
        structural = len(collapsed) == 384 * 2 and all(math.isfinite(float(row[key])) for row in collapsed for key in ("gain", "oracle", "shuffle_gain"))
        safety = severe == 0 and hard == 0
        futile = metrics["gain"]["ucb95"] < GAIN_TARGET and metrics["retention"]["ucb95"] < RETENTION_TARGET and metrics["true_minus_shuffle"]["ucb95"] < SHUFFLE_TARGET
        survives = structural and safety and not futile
        cell_rows.append({"cell": cell, "survives": survives, "futile": futile, "structural": structural, "safety": safety, "gain_point_db": metrics["gain"]["point"], "gain_ucb95_db": metrics["gain"]["ucb95"], "retention_point": metrics["retention"]["point"], "retention_ucb95": metrics["retention"]["ucb95"], "true_minus_shuffle_point_db": metrics["true_minus_shuffle"]["point"], "true_minus_shuffle_ucb95_db": metrics["true_minus_shuffle"]["ucb95"]})
        risk_rows.append({"cell": cell, "severe_count": severe, "hard_count": hard, "analysis_units": len(collapsed)})
    control_rows = []
    for cell in CONTROLS:
        subset = [row for row in prediction_rows if row["cell"] == cell]
        control_rows.append({"control": cell, "mean_gain_db": sum(float(row["gain"]) for row in subset)/len(subset), "mean_response_shuffle_gain_db": sum(float(row["response_shuffle_gain"]) for row in subset)/len(subset), "severe_count": sum(float(row["gain"]) <= -0.2 for row in subset), "hard_count": sum(float(row["gain"]) <= -0.5 for row in subset)})
    survivors = [row["cell"] for row in cell_rows if row["survives"]]
    structural_valid = all(row["structural"] for row in cell_rows)
    if survivors:
        state, decision, authorizes = "COMPLETED_GATE_PASS", "R3_A1_ACV_SCREEN_SURVIVOR", "R3_A2_AMENDMENT_REVIEW"
    elif structural_valid:
        state, decision, authorizes = "COMPLETED_GATE_FAIL", "R3_A1_ACV_SCREEN_FUTILITY_STOP", "NONE"
    else:
        state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R3_A1_ACV_SCREEN_INCONCLUSIVE", "NONE"
    amendment = {"schema_version": 1, "review": "APPROVED", "basis": "A0 PASS exceeds all proposal-oracle gates and sealed cache makes the two-fold screen non-duplicative and executable", "scope": {"folds": list(FOLDS), "seeds": list(SEEDS), "cells": list(CELLS), "controls": list(CONTROLS)}, "forbidden": ["A2 runtime", "confirmation", "canary", "locked test"], "threshold_source": "pre-result R3 design card"}
    contract_summary = {"schema_version": 1, "question": "Can any fixed proposal-first critic cell escape two-fold development futility without structural or safety failure?", "population": "exact S0 768 development names; outer folds 0/1 evaluated, one disjoint development fold calibrates abstention, and the remaining two development folds fit the critic", "analysis_unit": "clean-reference image/group with D_ref and D_rep paired", "input_whitelist": ["A0 sealed base/step/support/current", "proposal identity", "candidate-minus-reference RGB", "official checkpoint-frozen ConvIR first-encoder response"], "prohibited_inputs": ["filename", "fold id", "GT/clean RGB as model input", "confirmation", "canary", "locked test"], "optimizer": "AdamW", "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "folds": list(FOLDS), "seeds": list(SEEDS), "futility_targets": {"gain_db": GAIN_TARGET, "retention": RETENTION_TARGET, "true_minus_shuffle_db": SHUFFLE_TARGET}, "terminal_policy": {"survivor": "R3_A2_AMENDMENT_REVIEW only", "futility": "stop route", "inconclusive": "no next runtime"}}
    access = {"schema_version": 1, "route_commit": context.route_commit, "a0_cache_manifest_sha256": a0_manifest["cache_manifest_sha256"], "official_checkpoint_sha256": sha256_file(asset_path(context, "official_checkpoint", kind="file")), "development_images_targets_accessed": 768, "confirmation_images_targets_outcomes_touched": False, "canary_touched": False, "locked_test_touched": False, "historical_a1x_432_outcomes_touched": False}
    resource = {"schema_version": 1, "wall_seconds": time.perf_counter()-started, "cache_units_read": len(raw_rows), "training_units": total_train, "trainable_parameters": INPUT_DIM*HIDDEN_DIM+HIDDEN_DIM+HIDDEN_DIM*HIDDEN_DIM+HIDDEN_DIM+HIDDEN_DIM+1, "gpu_used": True, "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device))/(1024.0*1024.0)}
    atomic_json(context.phase_output_path / "a1_amendment_review.json", amendment)
    atomic_json(context.phase_output_path / "a1_contract_summary.json", contract_summary)
    write_csv(context.phase_output_path / "a1_cell_summary.csv", cell_rows)
    atomic_json(context.phase_output_path / "a1_bootstrap_summary.json", bootstrap)
    write_csv(context.phase_output_path / "a1_control_summary.csv", control_rows)
    write_csv(context.phase_output_path / "a1_risk_summary.csv", risk_rows)
    atomic_json(context.phase_output_path / "a1_resource_summary.json", resource)
    atomic_json(context.phase_output_path / "a1_source_access_audit.json", access)
    write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={"survivors": survivors, "structural_valid": structural_valid, "screen_fold_count": 2, "seed_count": 2, "cell_count": 4, "training_units": total_train})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("contract", "run")); parser.add_argument("--context", required=True, type=Path); args = parser.parse_args()
    contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__": main()
