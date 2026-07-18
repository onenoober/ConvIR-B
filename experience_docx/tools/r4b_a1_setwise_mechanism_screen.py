#!/usr/bin/env python3
"""Frozen R4B A1 two-fold screen of a genuine three-action set-wise risk model."""

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

ROUTE_ID = "haze4k_v5_r4b_three_action_setwise_utility_risk_20260718"
OPERATION_ID = "R4B_A1_SETWISE_MECHANISM_SCREEN"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
CELLS = ("S_utility_risk", "I_utility_risk", "S_mean_only", "S_action_only",
         "S_state_only", "S_unsigned", "S_risk_label_shuffle")
RISK_CELLS = {"S_utility_risk", "I_utility_risk", "S_action_only", "S_state_only", "S_risk_label_shuffle"}
NUISANCE_CELLS = ("S_action_only", "S_state_only", "S_unsigned")
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
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
SET_DIM = 48
INDEPENDENT_HIDDEN = 115
GAIN_TARGET = 0.020
RETENTION_TARGET = 0.25
SPECIFICITY_TARGET = 0.005
INCREMENT_TARGET = 0.005
MIN_COVERAGE = 0.10
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
PERMUTATION_TOLERANCE = 1.0e-6


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
    return torch.cat((flat.mean(2), flat.std(2, unbiased=False),
                      flat.abs().mean(2), flat.abs().amax(2)), dim=1)


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
    return torch.tensor(((1.0, 0.0, 0.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0, 1.0, 1.0),
                         (0.0, 0.0, 1.0, -1.0, 1.0)), dtype=torch.float32, device=device)


def build_model(cell: str) -> Any:
    import torch

    class SetwiseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Linear(INPUT_DIM, SET_DIM)
            self.attention = torch.nn.MultiheadAttention(SET_DIM, 4, batch_first=True)
            self.norm1 = torch.nn.LayerNorm(SET_DIM)
            self.ffn = torch.nn.Sequential(torch.nn.Linear(SET_DIM, 2 * SET_DIM), torch.nn.Tanh(),
                                           torch.nn.Linear(2 * SET_DIM, SET_DIM))
            self.norm2 = torch.nn.LayerNorm(SET_DIM)
            self.output = torch.nn.Linear(SET_DIM, 4)

        def forward(self, value: Any) -> Any:
            embedded = torch.tanh(self.embed(value))
            attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
            hidden = self.norm1(embedded + attended)
            hidden = self.norm2(hidden + self.ffn(hidden))
            return self.output(hidden)

    if cell == "I_utility_risk":
        return torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, INDEPENDENT_HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(INDEPENDENT_HIDDEN, INDEPENDENT_HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(INDEPENDENT_HIDDEN, 4),
        )
    return SetwiseModel()


def mask_features(features: Any, cell: str) -> Any:
    output = features.clone()
    if cell == "S_action_only":
        output[..., :STATE_DIM] = 0.0
        output[..., STATE_DIM + ACTION_DIM:] = 0.0
    elif cell == "S_state_only":
        output[..., STATE_DIM:] = 0.0
    return output


def model_loss(model: Any, features: Any, targets: Any, cell: str,
               risk_targets: Any | None = None) -> Any:
    import torch
    import torch.nn.functional as functional
    raw = model(mask_features(features, cell))
    utility_target = targets.abs() if cell == "S_unsigned" else targets
    mean = raw[..., 0]
    mean_loss = functional.smooth_l1_loss(mean, utility_target)
    true_gap = utility_target.unsqueeze(2) - utility_target.unsqueeze(1)
    pred_gap = mean.unsqueeze(2) - mean.unsqueeze(1)
    pair_mask = true_gap.abs() > 1.0e-8
    pair = functional.softplus(-torch.sign(true_gap) * pred_gap)
    pair_loss = (pair * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    loss = mean_loss + 0.25 * pair_loss
    if cell in RISK_CELLS:
        error = targets - raw[..., 1]
        q05_loss = torch.maximum(0.05 * error, -0.95 * error).mean()
        label_target = targets if risk_targets is None else risk_targets
        harm = (label_target < 0.0).float()
        severe = (label_target <= SEVERE_GAIN).float()
        loss = loss + 0.25 * q05_loss
        loss = loss + 0.5 * functional.binary_cross_entropy_with_logits(raw[..., 2], harm)
        loss = loss + 0.5 * functional.binary_cross_entropy_with_logits(raw[..., 3], severe)
    return loss


def train_model(features: Any, targets: Any, seed: int, cell: str) -> Any:
    import torch
    torch.manual_seed(seed)
    model = build_model(cell)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 17)
    risk_targets = None
    if cell == "S_risk_label_shuffle":
        risk_targets = targets[torch.randperm(len(targets), generator=torch.Generator().manual_seed(seed + 991))]
    for _epoch in range(EPOCHS):
        for indices in torch.randperm(len(features), generator=generator).split(BATCH_SIZE):
            batch_risk = None if risk_targets is None else risk_targets[indices]
            loss = model_loss(model, features[indices], targets[indices], cell, batch_risk)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval()


def predict_raw(model: Any, features: Any, cell: str) -> Any:
    import torch
    with torch.no_grad():
        raw = model(mask_features(features, cell))
    output = raw.clone()
    output[..., 2:] = output[..., 2:].sigmoid()
    if cell not in RISK_CELLS:
        output[..., 1] = output[..., 0]
        output[..., 2:] = 0.0
    return output


def ensemble_raw(models: list[Any], features: Any, cell: str) -> Any:
    import torch
    return torch.stack([predict_raw(model, features, cell) for model in models]).mean(0)


def apply_policy(raw: Any, truth: Any, threshold: dict[str, float], risk_enabled: bool) -> dict[str, Any]:
    import torch
    values = raw.clone()
    values[:, 0, :] = 0.0
    active_index = values[:, 1:, 0].argmax(1) + 1
    mean = values.gather(1, active_index[:, None, None].expand(-1, 1, 4)).squeeze(1)
    margin = mean[:, 0]
    allowed = margin >= threshold["margin"]
    if risk_enabled:
        allowed &= mean[:, 1] >= threshold["q_floor"]
        allowed &= mean[:, 2] <= threshold["harm"]
        allowed &= mean[:, 3] <= threshold["severe"]
    selected = torch.where(allowed, active_index, torch.zeros_like(active_index))
    gain = truth.gather(1, selected[:, None]).squeeze(1)
    oracle = truth.max(1).values.clamp(min=0.0)
    return {"selected": selected, "gain": gain, "oracle": oracle, "margin": margin,
            "q05": mean[:, 1], "harm": mean[:, 2], "severe": mean[:, 3],
            "negative_oracle": (truth[:, 2] > truth[:, 1]) & (truth[:, 2] > 0.0)}


def grouped_policy_summary(result: dict[str, Any], units: list[dict[str, Any]]) -> dict[str, float | int]:
    rows = [{"name": unit["name"], "operator": unit["operator"],
             "gain": float(result["gain"][index]), "selected": int(result["selected"][index])}
            for index, unit in enumerate(units)]
    gains = [sum(row["gain"] for row in rows if row["operator"] == op) /
             sum(row["operator"] == op for row in rows) for op in OPERATORS]
    coverage = [sum(row["selected"] != 0 for row in rows if row["operator"] == op) /
                sum(row["operator"] == op for row in rows) for op in OPERATORS]
    by_group = {(name, op): next(row for row in rows if row["name"] == name and row["operator"] == op)
                for name in {row["name"] for row in rows} for op in OPERATORS}
    severe = sum(any(by_group[(name, op)]["gain"] <= SEVERE_GAIN for op in OPERATORS)
                 for name in {row["name"] for row in rows})
    hard = sum(any(by_group[(name, op)]["gain"] <= HARD_GAIN for op in OPERATORS)
               for name in {row["name"] for row in rows})
    return {"gain": min(gains), "coverage": min(coverage), "severe_groups": severe, "hard_groups": hard}


def calibrate(raw: Any, truth: Any, units: list[dict[str, Any]], cell: str) -> dict[str, float]:
    import torch
    values = raw.clone(); values[:, 0, :] = 0.0
    active = values[:, 1:, 0].max(1).values
    margins = sorted(set(float(torch.quantile(active, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.90)))
    risk_enabled = cell in RISK_CELLS
    q_floors = (-0.20, -0.10, -0.05, 0.0) if risk_enabled else (-math.inf,)
    harm_caps = (0.10, 0.20, 0.35, 0.50) if risk_enabled else (1.0,)
    severe_caps = (0.05, 0.10, 0.20, 0.40) if risk_enabled else (1.0,)
    feasible = []
    for margin in margins:
        for q_floor in q_floors:
            for harm in harm_caps:
                for severe in severe_caps:
                    threshold = {"margin": margin, "q_floor": q_floor, "harm": harm, "severe": severe}
                    result = apply_policy(raw, truth, threshold, risk_enabled)
                    summary = grouped_policy_summary(result, units)
                    if summary["coverage"] >= MIN_COVERAGE and summary["severe_groups"] == 0 and summary["hard_groups"] == 0:
                        feasible.append((float(summary["gain"]), float(summary["coverage"]), -margin, -harm, threshold))
    if not feasible:
        return {"margin": math.inf, "q_floor": 0.0, "harm": 0.0, "severe": 0.0}
    return max(feasible, key=lambda item: item[:4])[-1]


def auc_ap(labels: Any, scores: Any) -> dict[str, float]:
    import numpy as np
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum()); negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": positives / max(len(labels), 1)}
    ranks = rankdata(scores, method="average")
    auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="mergesort"); ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    ap = float(precision[ordered == 1].mean())
    return {"auroc": float(auc), "auprc": ap, "prevalence": positives / len(labels)}


def calibration_rows(labels: Any, scores: Any, bins: int = 10) -> tuple[list[dict[str, Any]], float, float]:
    import numpy as np
    labels = np.asarray(labels, dtype=np.float64); scores = np.asarray(scores, dtype=np.float64)
    rows = []; ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & ((scores < high) if index + 1 < bins else (scores <= high))
        if not mask.any():
            continue
        confidence = float(scores[mask].mean()); rate = float(labels[mask].mean()); weight = float(mask.mean())
        ece += weight * abs(confidence - rate)
        rows.append({"bin": index, "low": low, "high": high, "count": int(mask.sum()),
                     "mean_probability": confidence, "event_rate": rate})
    brier = float(np.mean((scores - labels) ** 2))
    return rows, ece, brier


def policy_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    names = sorted({row["name"] for row in rows}); keyed = {(row["cell"], row["name"], row["operator"]): row for row in rows}
    arrays = {cell: {op: {field: np.asarray([float(keyed[(cell, name, op)][field]) for name in names])
                               for field in ("gain", "oracle", "shuffle_gain")}
                     for op in OPERATORS} for cell in CELLS}

    def evaluate(index: Any) -> dict[str, float]:
        cell_gain = {cell: [float(arrays[cell][op]["gain"][index].mean()) for op in OPERATORS] for cell in CELLS}
        primary = cell_gain["S_utility_risk"]
        oracles = [float(arrays["S_utility_risk"][op]["oracle"][index].mean()) for op in OPERATORS]
        shuffles = [float(arrays["S_utility_risk"][op]["shuffle_gain"][index].mean()) for op in OPERATORS]
        worst = min(range(2), key=lambda item: primary[item])
        return {
            "gain": primary[worst], "retention": primary[worst] / max(oracles[worst], 1.0e-12),
            "true_minus_shuffle": min(primary[i] - shuffles[i] for i in range(2)),
            "setwise_minus_independent": min(primary[i] - cell_gain["I_utility_risk"][i] for i in range(2)),
            "utility_risk_minus_mean_only": min(primary[i] - cell_gain["S_mean_only"][i] for i in range(2)),
            "primary_minus_best_nuisance": min(primary[i] - max(cell_gain[cell][i] for cell in NUISANCE_CELLS) for i in range(2)),
        }

    point = evaluate(np.arange(len(names))); samples = {key: [] for key in point}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        value = evaluate(rng.integers(0, len(names), len(names)))
        for key in samples: samples[key].append(value[key])
    return {key: {"point": point[key], "lcb95": float(np.quantile(values, 0.025)),
                  "ucb95": float(np.quantile(values, 0.975))} for key, values in samples.items()}


def risk_bootstrap(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    names = sorted({row["name"] for row in group_rows})
    by_cell = {cell: {(row["name"], row["action"]): row for row in group_rows if row["cell"] == cell}
               for cell in ("S_utility_risk", "S_risk_label_shuffle")}

    def metric(cell: str, chosen: list[str]) -> dict[str, float]:
        rows = [by_cell[cell][(name, action)] for name in chosen for action in (1, 2)]
        return auc_ap([row["severe_label"] for row in rows], [row["severe_score"] for row in rows])

    point_primary = metric("S_utility_risk", names); point_shuffle = metric("S_risk_label_shuffle", names)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 41); samples = {"auroc": [], "auprc_lift": [], "auprc_minus_shuffle": []}
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [names[index] for index in rng.integers(0, len(names), len(names))]
        primary = metric("S_utility_risk", chosen); shuffled = metric("S_risk_label_shuffle", chosen)
        samples["auroc"].append(primary["auroc"])
        samples["auprc_lift"].append(primary["auprc"] - primary["prevalence"])
        samples["auprc_minus_shuffle"].append(primary["auprc"] - shuffled["auprc"])
    points = {"auroc": point_primary["auroc"], "auprc_lift": point_primary["auprc"] - point_primary["prevalence"],
              "auprc_minus_shuffle": point_primary["auprc"] - point_shuffle["auprc"]}
    output = {key: {"point": points[key], "lcb95": float(np.nanquantile(values, 0.025)),
                    "ucb95": float(np.nanquantile(values, 0.975))} for key, values in samples.items()}
    output["primary"] = point_primary; output["risk_label_shuffle"] = point_shuffle
    return output


def contract(context_path: Path) -> None:
    import torch
    context = load_context(context_path, "contract"); prepare_phase_output(context)
    torch.manual_seed(SEEDS[0]); features = torch.randn(8, 3, INPUT_DIM); targets = torch.randn(8, 3) * 0.1; targets[:, 0] = 0.0
    primary = build_model("S_utility_risk"); independent = build_model("I_utility_risk")
    permutation = torch.tensor((2, 0, 1)); inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = primary(features); permuted = primary(features[:, permutation])[:, inverse]
        permutation_error = float((original - permuted).abs().max())
        altered = features.clone(); altered[:, 1:] += 7.0
        independent_error = float((independent(features)[:, 0] - independent(altered)[:, 0]).abs().max())
    loss = model_loss(primary, features, targets, "S_utility_risk"); loss.backward()
    set_parameters = sum(parameter.numel() for parameter in primary.parameters())
    independent_parameters = sum(parameter.numel() for parameter in independent.parameters())
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "cells_exact": CELLS == ("S_utility_risk", "I_utility_risk", "S_mean_only", "S_action_only", "S_state_only", "S_unsigned", "S_risk_label_shuffle"),
        "permutation_equivariant": permutation_error <= PERMUTATION_TOLERANCE,
        "independent_candidate_locality": independent_error == 0.0,
        "matched_parameter_budget": abs(set_parameters - independent_parameters) / set_parameters <= 0.01,
        "primary_gradients_finite_nonzero": all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in primary.parameters())
        and max(float(parameter.grad.abs().max()) for parameter in primary.parameters()) > 0.0,
        "fold_seed_epoch_contract": FOLDS == (0, 1) and SEEDS == (3407, 3411) and EPOCHS == 32,
        "generic_progress_total": context.total_units == TOTAL_UNITS,
        "protected_roles_blocked": True,
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(context.phase_output_path / "r4b_a1_synthetic_contract.json", {
        "schema_version": 1, "checks": checks, "permutation_max_abs": permutation_error,
        "independent_locality_max_abs": independent_error, "set_parameters": set_parameters,
        "independent_parameters": independent_parameters,
    })
    write_contract_result(context, checks=checks)


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run"); prepare_phase_output(context); started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ledger_path = asset_path(context, "r4_ledger", kind="file"); cache_manifest_path = asset_path(context, "a0_cache_manifest", kind="file")
    raw_manifest_path = asset_path(context, "a0_raw_manifest", kind="file"); cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    a0_closeout_path = asset_path(context, "a0_closeout", kind="file"); data_root = asset_path(context, "haze4k_data", kind="directory")
    checkpoint_path = asset_path(context, "official_checkpoint", kind="file")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")); cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    a0_closeout = json.loads(a0_closeout_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_rows) != FEATURE_UNITS or cache_manifest.get("cache_manifest_sha256") != "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d":
        raise RuntimeError("sealed cache identity mismatch")
    if (a0_closeout.get("state"), a0_closeout.get("decision"), a0_closeout.get("authorizes")) != (
            "COMPLETED_GATE_PASS", "R4B_A0_RISK_FEASIBILITY_PASS", OPERATION_ID):
        raise RuntimeError("A0 authorization mismatch")
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}; fold_lookup = {name: fold for fold, names in folds.items() for name in names}
    development = set(ledger["roles"]["development"]); confirmation = set(ledger["roles"]["confirmation"])
    if set(fold_lookup) != development or development & confirmation: raise RuntimeError("data-role isolation failed")

    def load_label(name: str) -> Any:
        stem, extension = os.path.splitext(name)
        for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
            path = data_root / "train/gt" / candidate
            if path.is_file():
                with Image.open(path) as image: array = np.asarray(image.convert("RGB")).copy()
                return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        raise FileNotFoundError(name)

    if context.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("frozen encoder extraction requires CUDA")
    device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device); sys.path.insert(0, str(context.remote_repo / "Dehazing/ITS"))
    from models.ConvIR import build_net
    encoder = build_net("base", "Haze4K").to(device).eval(); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder.load_state_dict(checkpoint, strict=True)
    for parameter in encoder.parameters(): parameter.requires_grad_(False)

    units = []; fixed_action = action_features(torch.device("cpu"))
    for index, row in enumerate(raw_rows, 1):
        unit_path = cache_root / f"{row['unit_key']}.pt"
        if sha256_file(unit_path) != row["cache_sha256"]: raise RuntimeError(f"cache hash mismatch: {row['unit_key']}")
        payload = torch.load(unit_path, map_location="cpu", weights_only=False); candidate_names = list(payload["candidate_names"])
        selected_indices = [candidate_names.index(action) for action in ACTIONS]; name = payload["name"]
        base = payload["base"].float(); step = payload["step"].float(); current = payload["current"].float(); support = payload["support"].float()
        candidate_delta = payload["candidates"].float()[selected_indices]; label = load_label(name)
        if label.shape[-2:] != base.shape[-2:]: label = label[:, :, :base.shape[-2], :base.shape[-1]]
        reference = torch.clamp(base + 0.25 * step, 0.0, 1.0); renders = torch.clamp(base + 0.25 * (step + candidate_delta), 0.0, 1.0)
        target = metric_psnr((renders - label).square().mean((1, 2, 3))) - metric_psnr((reference - label).square().mean()); target[0] = 0.0
        state = torch.cat((tensor_stats(base), tensor_stats(step), tensor_stats(current), tensor_stats(support)), dim=1)
        if state.shape[1] != STATE_DIM: raise RuntimeError(f"state identity mismatch: {state.shape}")
        rgb = rgb_response_features(renders - reference)
        with torch.no_grad():
            encoder_inputs = torch.cat((reference, renders), dim=0).to(device)
            deep_all = encoder.Encoder[0](encoder.feat_extract[0](encoder_inputs)); deep = deep_response_features(deep_all[1:] - deep_all[0:1]).cpu()
        features = torch.cat((state.expand(3, -1), fixed_action, rgb, deep), dim=1)
        if features.shape != (3, INPUT_DIM) or not bool(torch.isfinite(features).all()): raise RuntimeError("feature contract failed")
        units.append({"name": name, "operator": payload["operator"], "fold": fold_lookup[name], "features": features, "target": target})
        if index % 8 == 0 or index == FEATURE_UNITS: write_workload_progress(context, completed_units=index, stage="feature_extract")

    prediction_rows = []; risk_rows = []; training_rows = []; score_rows = []; permutation_errors = []; completed_training = 0
    for fold in FOLDS:
        calibration_fold = (fold + 1) % 4
        train_units = [unit for unit in units if unit["fold"] not in (fold, calibration_fold)]
        cal_units = [unit for unit in units if unit["fold"] == calibration_fold]; test_units = [unit for unit in units if unit["fold"] == fold]
        x_train = torch.stack([unit["features"] for unit in train_units]); y_train = torch.stack([unit["target"] for unit in train_units])
        x_cal = torch.stack([unit["features"] for unit in cal_units]); y_cal = torch.stack([unit["target"] for unit in cal_units])
        x_test = torch.stack([unit["features"] for unit in test_units]); y_test = torch.stack([unit["target"] for unit in test_units])
        x_shuffle = x_test.clone(); x_shuffle[:, 1] = x_test[:, 2]; x_shuffle[:, 2] = x_test[:, 1]
        for cell in CELLS:
            models = []
            for seed in SEEDS:
                model = train_model(x_train, y_train, seed, cell); models.append(model); completed_training += 1
                write_workload_progress(context, completed_units=FEATURE_UNITS + completed_training, stage="set_model_train")
                if cell == "S_utility_risk":
                    permutation = torch.tensor((2, 0, 1)); inverse = torch.argsort(permutation)
                    with torch.no_grad(): error = float((model(x_test[:, permutation])[:, inverse] - model(x_test)).abs().max())
                    permutation_errors.append({"fold": fold, "seed": seed, "max_abs": error})
            cal_raw = ensemble_raw(models, x_cal, cell); threshold = calibrate(cal_raw, y_cal, cal_units, cell)
            test_raw = ensemble_raw(models, x_test, cell); shuffled_raw = ensemble_raw(models, x_shuffle, cell)
            main = apply_policy(test_raw, y_test, threshold, cell in RISK_CELLS)
            shuffled = apply_policy(shuffled_raw, y_test, threshold, cell in RISK_CELLS)
            training_rows.append({"cell": cell, "fold": fold, "calibration_fold": calibration_fold,
                                  "ensemble_seeds": "3407,3411", "margin_threshold": threshold["margin"],
                                  "q_floor": threshold["q_floor"], "harm_threshold": threshold["harm"],
                                  "severe_threshold": threshold["severe"], "train_units": len(train_units),
                                  "calibration_units": len(cal_units), "test_units": len(test_units), "epochs": EPOCHS})
            for item, unit in enumerate(test_units):
                prediction_rows.append({"cell": cell, "fold": fold, "name": unit["name"], "operator": unit["operator"],
                                        "gain": float(main["gain"][item]), "oracle": float(main["oracle"][item]),
                                        "shuffle_gain": float(shuffled["gain"][item]), "selected": int(main["selected"][item]),
                                        "negative_oracle": bool(main["negative_oracle"][item])})
                for action in (1, 2):
                    risk_rows.append({"cell": cell, "name": unit["name"], "operator": unit["operator"], "action": action,
                                      "harm_label": int(y_test[item, action] < 0.0), "severe_label": int(y_test[item, action] <= SEVERE_GAIN),
                                      "harm_score": float(test_raw[item, action, 2]), "severe_score": float(test_raw[item, action, 3])})
                if cell == "S_utility_risk":
                    values = test_raw[item].clone(); values[0] = 0.0; active = int(values[1:, 0].argmax()) + 1
                    score_rows.append({"name": unit["name"], "operator": unit["operator"], "fold": fold, "action": active,
                                       "gain": float(y_test[item, active]), "confidence": float(min(values[active, 0], values[active, 1] + 0.2,
                                                                                                  0.5 - values[active, 2], 0.2 - values[active, 3]))})

    write_csv(context.phase_output_path / "r4b_a1_oof_rows_cloud_only.csv", prediction_rows)
    write_csv(context.phase_output_path / "r4b_a1_candidate_risk_rows_cloud_only.csv", risk_rows)
    policy_metrics = policy_bootstrap(prediction_rows)
    group_risk_rows = []
    for cell in CELLS:
        subset = [row for row in risk_rows if row["cell"] == cell]
        for name in sorted({row["name"] for row in subset}):
            for action in (1, 2):
                items = [row for row in subset if row["name"] == name and row["action"] == action]
                group_risk_rows.append({"cell": cell, "name": name, "action": action,
                                        "harm_label": max(row["harm_label"] for row in items),
                                        "severe_label": max(row["severe_label"] for row in items),
                                        "harm_score": max(row["harm_score"] for row in items),
                                        "severe_score": max(row["severe_score"] for row in items)})
    risk_metrics = risk_bootstrap(group_risk_rows)
    primary_group_risk = [row for row in group_risk_rows if row["cell"] == "S_utility_risk"]
    calibration, severe_ece, severe_brier = calibration_rows([row["severe_label"] for row in primary_group_risk],
                                                              [row["severe_score"] for row in primary_group_risk])
    for row in calibration: row["target"] = "severe"; row["cell"] = "S_utility_risk"

    primary_rows = [row for row in prediction_rows if row["cell"] == "S_utility_risk"]
    primary_keyed = {(row["name"], row["operator"]): row for row in primary_rows}; evaluated_names = sorted({row["name"] for row in primary_rows})
    severe_groups = sum(any(float(primary_keyed[(name, op)]["gain"]) <= SEVERE_GAIN for op in OPERATORS) for name in evaluated_names)
    hard_groups = sum(any(float(primary_keyed[(name, op)]["gain"]) <= HARD_GAIN for op in OPERATORS) for name in evaluated_names)
    risk_coverage = []
    score_keyed = {(row["name"], row["operator"]): row for row in score_rows}
    ordered_names = sorted(evaluated_names, key=lambda name: min(score_keyed[(name, op)]["confidence"] for op in OPERATORS), reverse=True)
    for coverage in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00):
        count = max(1, math.ceil(coverage * len(ordered_names))); selected_names = ordered_names[:count]
        group_severe = sum(any(score_keyed[(name, op)]["gain"] <= SEVERE_GAIN for op in OPERATORS) for name in selected_names)
        group_hard = sum(any(score_keyed[(name, op)]["gain"] <= HARD_GAIN for op in OPERATORS) for name in selected_names)
        from scipy.stats import beta
        ucb = 1.0 if group_severe == count else float(beta.ppf(0.95, group_severe + 1, count - group_severe))
        for operator in OPERATORS:
            gains = np.asarray([score_keyed[(name, operator)]["gain"] for name in selected_names], dtype=np.float64)
            risk_coverage.append({"coverage_target": coverage, "selected_groups": count, "operator": operator,
                                  "mean_gain_db": float(gains.mean()), "cvar5_gain_db": float(np.sort(gains)[:max(1, math.ceil(0.05 * len(gains)))].mean()),
                                  "group_severe_count": group_severe, "group_hard_count": group_hard,
                                  "group_severe_ucb95_exact": ucb})

    cell_summary = []; control_summary = []
    for cell in CELLS:
        rows = [row for row in prediction_rows if row["cell"] == cell]
        gains = [sum(float(row["gain"]) for row in rows if row["operator"] == op) / sum(row["operator"] == op for row in rows) for op in OPERATORS]
        coverage = [sum(row["selected"] != 0 for row in rows if row["operator"] == op) / sum(row["operator"] == op for row in rows) for op in OPERATORS]
        cell_summary.append({"cell": cell, "gain_point_db": min(gains), "coverage_point": min(coverage),
                             "severe_operator_images": sum(float(row["gain"]) <= SEVERE_GAIN for row in rows),
                             "hard_operator_images": sum(float(row["gain"]) <= HARD_GAIN for row in rows),
                             "negative_selection_rate": sum(row["selected"] == 2 for row in rows) / len(rows)})
        if cell != "S_utility_risk": control_summary.append({"control": cell, "gain_point_db": min(gains), "coverage_point": min(coverage)})

    max_permutation_error = max(row["max_abs"] for row in permutation_errors)
    structural_checks = {"feature_units_complete": len(units) == FEATURE_UNITS,
                         "training_units_complete": len(training_rows) == len(FOLDS) * len(CELLS),
                         "seed_model_units_complete": completed_training == TRAINING_UNITS,
                         "policy_rows_complete": len(prediction_rows) == 384 * 2 * len(CELLS),
                         "risk_group_rows_complete": len(group_risk_rows) == 384 * 2 * len(CELLS),
                         "all_cells_present": sorted({row["cell"] for row in prediction_rows}) == sorted(CELLS),
                         "all_folds_present": sorted({row["fold"] for row in prediction_rows}) == list(FOLDS),
                         "finite": all(math.isfinite(float(row[key])) for row in prediction_rows for key in ("gain", "oracle", "shuffle_gain")),
                         "development_confirmation_disjoint": not development & confirmation}
    gates = {"gain_ucb95": policy_metrics["gain"]["ucb95"] >= GAIN_TARGET,
             "retention_ucb95": policy_metrics["retention"]["ucb95"] >= RETENTION_TARGET,
             "true_minus_shuffle_ucb95": policy_metrics["true_minus_shuffle"]["ucb95"] >= SPECIFICITY_TARGET,
             "setwise_minus_independent_ucb95": policy_metrics["setwise_minus_independent"]["ucb95"] >= INCREMENT_TARGET,
             "utility_risk_minus_mean_only_ucb95": policy_metrics["utility_risk_minus_mean_only"]["ucb95"] >= INCREMENT_TARGET,
             "primary_minus_best_nuisance_ucb95": policy_metrics["primary_minus_best_nuisance"]["ucb95"] >= INCREMENT_TARGET,
             "severe_auroc_lcb95": risk_metrics["auroc"]["lcb95"] > 0.5,
             "severe_auprc_lift_lcb95": risk_metrics["auprc_lift"]["lcb95"] > 0.0,
             "severe_auprc_minus_shuffle_lcb95": risk_metrics["auprc_minus_shuffle"]["lcb95"] > 0.0,
             "selected_severe_groups_zero": severe_groups == 0, "selected_hard_groups_zero": hard_groups == 0,
             "permutation_equivariance": max_permutation_error <= PERMUTATION_TOLERANCE}
    structural_valid = all(structural_checks.values()); survives = structural_valid and all(gates.values())
    if survives: state, decision, authorizes = "COMPLETED_GATE_PASS", "R4B_A1_SETWISE_MECHANISM_SURVIVES", "R4B_A2_FULL_OOF"
    elif structural_valid: state, decision, authorizes = "COMPLETED_GATE_FAIL", "R4B_A1_SETWISE_MECHANISM_FUTILITY_STOP", "NONE"
    else: state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R4B_A1_SETWISE_MECHANISM_INCONCLUSIVE", "NONE"

    contract_summary = {"schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
                        "cells": list(CELLS), "actions": list(ACTIONS), "folds": list(FOLDS), "ensemble_seeds": list(SEEDS),
                        "epochs": EPOCHS, "optimizer": "AdamW", "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "risk_outputs": ["mean_utility", "q05_utility", "harm_probability", "severe_probability"]}
    risk_metrics["severe_ece"] = severe_ece; risk_metrics["severe_brier"] = severe_brier
    access = {"schema_version": 1, "route_commit": context.route_commit, "a0_closeout_sha256": sha256_file(a0_closeout_path),
              "official_checkpoint_sha256": sha256_file(checkpoint_path), "development_images_targets_accessed": 768,
              "evaluated_development_images": 384, "confirmation_images_targets_outcomes_touched": False,
              "historical_a1x_432_outcomes_touched": False, "canary_touched": False, "locked_test_touched": False}
    resource = {"schema_version": 1, "wall_seconds": time.perf_counter() - started, "feature_units": FEATURE_UNITS,
                "seed_model_training_units": TRAINING_UNITS, "set_parameters": sum(p.numel() for p in build_model("S_utility_risk").parameters()),
                "independent_parameters": sum(p.numel() for p in build_model("I_utility_risk").parameters()),
                "conv_ir_trainable_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)}
    atomic_json(context.phase_output_path / "r4b_a1_contract_summary.json", contract_summary)
    atomic_json(context.phase_output_path / "r4b_a1_structural_summary.json", {"schema_version": 1, "checks": structural_checks, "valid": structural_valid})
    atomic_json(context.phase_output_path / "r4b_a1_bootstrap_summary.json", {"schema_version": 1, **policy_metrics})
    atomic_json(context.phase_output_path / "r4b_a1_gate_summary.json", {"schema_version": 1, "gates": gates, "survives": survives})
    write_csv(context.phase_output_path / "r4b_a1_cell_summary.csv", cell_summary)
    write_csv(context.phase_output_path / "r4b_a1_control_summary.csv", control_summary)
    atomic_json(context.phase_output_path / "r4b_a1_risk_discrimination.json", {"schema_version": 1, **risk_metrics})
    write_csv(context.phase_output_path / "r4b_a1_calibration_summary.csv", calibration)
    atomic_json(context.phase_output_path / "r4b_a1_permutation_audit.json", {"schema_version": 1, "rows": permutation_errors, "max_abs": max_permutation_error, "tolerance": PERMUTATION_TOLERANCE})
    write_csv(context.phase_output_path / "r4b_a1_risk_coverage.csv", risk_coverage)
    write_csv(context.phase_output_path / "r4b_a1_training_summary.csv", training_rows)
    atomic_json(context.phase_output_path / "r4b_a1_source_access_audit.json", access)
    atomic_json(context.phase_output_path / "r4b_a1_resource_summary.json", resource)
    write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
        "survives": survives, "structural_valid": structural_valid, "gain_ucb95_db": policy_metrics["gain"]["ucb95"],
        "retention_ucb95": policy_metrics["retention"]["ucb95"], "true_minus_shuffle_ucb95_db": policy_metrics["true_minus_shuffle"]["ucb95"],
        "setwise_minus_independent_ucb95_db": policy_metrics["setwise_minus_independent"]["ucb95"],
        "utility_risk_minus_mean_only_ucb95_db": policy_metrics["utility_risk_minus_mean_only"]["ucb95"],
        "severe_auroc_lcb95": risk_metrics["auroc"]["lcb95"], "severe_auprc_lift_lcb95": risk_metrics["auprc_lift"]["lcb95"],
        "severe_auprc_minus_shuffle_lcb95": risk_metrics["auprc_minus_shuffle"]["lcb95"],
        "selected_severe_groups": severe_groups, "selected_hard_groups": hard_groups, "permutation_max_abs": max_permutation_error})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("contract", "run")); parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args(); contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__": main()
