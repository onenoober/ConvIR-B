#!/usr/bin/env python3
"""Run the frozen R3 A0 GT-free proposal-bank oracle in two sealed passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
)


ROUTE_ID = "haze4k_v5_r3_proposal_first_acv_20260717"
OPERATION_ID = "R3_A0_GT_FREE_PROPOSAL_ORACLE"
OPERATORS = ("D_ref", "D_rep")
BANK = (
    "reference_noop",
    "state_positive_full",
    "state_negative_full",
    "state_positive_exact_half",
    "state_negative_exact_half",
    "response_positive_full",
    "response_negative_full",
    "response_positive_exact_half",
    "response_negative_exact_half",
)
HALF_SIZES = {(400, 400): (208, 208), (480, 640): (240, 320)}
GRID = tuple(index / 64.0 for index in range(65))
GAIN_PASS = 0.080
GAIN_INCONCLUSIVE = 0.050
RETENTION_FLOOR = 0.50
REPAIRABLE_FLOOR = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    parent = str(path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def emit(context: Any, stage: str, completed: int, total: int) -> None:
    value = {
        "R3_A0_PROGRESS": {
            "stage": stage,
            "completed_units": completed,
            "total_units": total,
        }
    }
    line = json.dumps(value, sort_keys=True)
    print(line, flush=True)
    with context.status_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def metric_psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def numerical_tolerance(*values: float) -> float:
    return 2.0 * (1e-12 + 1e-12 * max(abs(value) for value in values))


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    synthetic = [
        ("reference_noop", (0, 0, 0)),
        ("state_positive_full", (1, 2, 3)),
        ("state_negative_full", (-1, -2, -3)),
        ("state_positive_exact_half", (1, 1, 2)),
        ("state_negative_exact_half", (-1, -1, -2)),
        ("response_positive_full", (2, 3, 4)),
        ("response_negative_full", (-2, -3, -4)),
        ("response_positive_exact_half", (2, 2, 3)),
        ("response_negative_exact_half", (-2, -2, -3)),
    ]
    deduplicated = []
    seen = set()
    for name, value in synthetic:
        if value not in seen:
            seen.add(value)
            deduplicated.append(name)
    checks = {
        "contract_cpu_only": context.device == "cpu"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "primary_bank_exact": tuple(name for name, _ in synthetic) == BANK,
        "primary_bank_cap": len(deduplicated) <= 9,
        "dedup_first_identity": deduplicated[0] == "reference_noop",
        "operators_paired": OPERATORS == ("D_ref", "D_rep"),
        "bootstrap_frozen": int(os.environ["CONVIR_ROUTE_BOOTSTRAP_DRAWS"]) == 4000
        and int(os.environ["CONVIR_ROUTE_BOOTSTRAP_SEED"]) == 3407,
        "development_count_frozen": int(os.environ["CONVIR_ROUTE_DEVELOPMENT_COUNT"]) == 768,
        "half_transport_frozen": HALF_SIZES[(400, 400)] == (208, 208)
        and HALF_SIZES[(480, 640)] == (240, 320),
        "gate_order_valid": GAIN_PASS > GAIN_INCONCLUSIVE > 0.0,
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "synthetic_bank_contract.json",
        {"schema_version": 1, "bank": list(BANK), "checks": checks},
    )
    write_contract_result(context, checks=checks)


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A0 requires the authorized CUDA runtime")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    ledger_path = asset_path(context, "r4_ledger", kind="file")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    development = list(ledger["roles"]["development"])
    confirmation = set(ledger["roles"]["confirmation"])
    if len(development) != 768 or len(set(development)) != 768:
        raise RuntimeError("S0 development identity is not exactly 768 unique names")
    if set(development) & confirmation or len(confirmation) != 432:
        raise RuntimeError("S0 development/confirmation separation failed")
    if canonical_hash(ledger) != "bf09dd05e2fd53c26158b31351554102f10fc6574b7dbe4e0d0b8b95b1cbd02a":
        raise RuntimeError("S0 ledger content hash mismatch")

    data_root = asset_path(context, "haze4k_data", kind="directory")
    v3z_root = asset_path(context, "v3z_checkout", kind="git_checkout")
    v3s_root = asset_path(context, "v3s_checkout", kind="git_checkout")
    v3p_root = asset_path(context, "v3p_checkout", kind="git_checkout")
    a1f_root = asset_path(context, "a1f_checkout", kind="git_checkout")
    a0p = load_module(
        a1f_root / "experience_docx/tools/chd_rm_v4a_a0p_audit.py",
        "r3_a0_a0p",
    )
    source = a0p.load_source(v3z_root)
    source.V3W = source.load_legacy()
    a0p.SOURCE = source
    v3s = source.V3W.import_v3s(v3s_root)
    legacy = v3s.import_legacy_modules(v3p_root)

    args = SimpleNamespace(
        a0_checkpoint=str(asset_path(context, "official_checkpoint", kind="file")),
        control_checkpoint=str(asset_path(context, "control_checkpoint", kind="file")),
        data_dir=str(data_root),
        fresh_split_manifest=str(asset_path(context, "fresh_split_manifest", kind="file")),
        v3j_a_bounds=str(asset_path(context, "bounds", kind="file")),
        operator_artifact_manifest=str(asset_path(context, "operator_manifest", kind="file")),
        density_artifact=str(asset_path(context, "density_artifact", kind="file")),
        d7c_artifact=str(asset_path(context, "d7c_artifact", kind="file")),
        reference_oof_rows=str(asset_path(context, "reference_oof_rows", kind="file")),
        expected_a0_checkpoint_sha256="6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088",
        expected_control_checkpoint_sha256="08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2",
        expected_density_artifact_sha256="1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f",
        expected_d7c_artifact_sha256="09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361",
        expected_fresh_split_manifest_sha256="c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7",
        expected_operator_manifest_sha256="1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84",
        expected_reference_oof_rows_sha256="b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1",
        expected_v3j_a_bounds_sha256="485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21",
        source_split="train",
        train_key="v3j_controller_train",
        formal_sample_count=1200,
        fold_count=5,
        proj_channels=24,
        channels=24,
        d7c_threshold=0.5773006677627563,
        delta_bound_multiplier=2.0,
        sample_count=128,
        epochs=16,
        risk_window=4,
        warmup_epochs=8,
        learning_rate=0.0005,
        weight_decay=0.00001,
        grad_clip_norm=0.1,
        cvar_fraction=0.25,
    )
    a0p.validate_frozen_args(args)
    v3s.asset_manifest(args)
    frozen = v3s.build_frozen_operator(args, legacy, device)
    args.delta_bound = tuple(args.delta_bound_multiplier * value for value in frozen["bound"])
    trace = json.loads(asset_path(context, "trace_manifest", kind="file").read_text(encoding="utf-8"))
    if trace.get("route_id") != "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714" \
            or trace.get("replicate_id") != "r1":
        raise RuntimeError("v4a trace identity mismatch")
    if sha256_file(Path(source.__file__)) != trace.get("v3z_source_sha256"):
        raise RuntimeError("v3z source differs from the v4a final-state source")
    payload = torch.load(asset_path(context, "final_state", kind="file"), map_location="cpu")
    if payload.get("state_kind") != "final" or payload.get("replicate_id") != "r1":
        raise RuntimeError("v4a final-state payload identity mismatch")
    models = source.V3W.import_v3w_models()
    cells = source.V3W.build_cells(models, None, args, device)
    _, (kind, objective, model) = next(iter(cells.items()))
    if kind != "output" or objective != "safety_curriculum":
        raise RuntimeError("unexpected v4a model cell")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    cache_root = context.phase_output_path / "candidate_cache_cloud_only"
    cache_root.mkdir()
    raw_cache_manifest = context.phase_output_path / "a0_cache_units_cloud_only.jsonl"
    cache_rows: list[dict[str, Any]] = []
    bound_excess = 0.0
    support_excess = 0.0
    noop_excess = 0.0
    max_unique = 0
    candidate_counts: dict[str, int] = {name: 0 for name in BANK}

    def load_hazy(name: str) -> torch.Tensor:
        path = data_root / "train/haze" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0)

    def exact_half(value: torch.Tensor, support: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        if (height, width) not in HALF_SIZES:
            raise RuntimeError(f"unregistered exact-half shape: {(height, width)}")
        low = F.interpolate(value, size=HALF_SIZES[(height, width)], mode="bilinear", align_corners=False, antialias=False)
        replay = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
        return support * a0p.clamp_channelwise(replay, bound)

    total_units = len(development) * len(OPERATORS)
    completed = 0
    with torch.no_grad():
        for name in development:
            input_img = load_hazy(name).unsqueeze(0).to(device)
            padded, height, width = legacy.pad_to_factor(input_img)
            hazy = padded[:, :, :height, :width]
            base = legacy.v3l_a0.forward_final(frozen["base"], padded, height, width)
            gate_full, _, _ = frozen["gate_producer"](padded)
            hard_gate = legacy.action_gate_from_full(gate_full, legacy.action_shape_for_input(padded)).to(device)
            support = legacy.output_gate_from_action_gate(hard_gate, base.shape[-2:])
            context_map, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
            delta_bound = base.new_tensor(args.delta_bound).view(1, 3, 1, 1)
            for operator in OPERATORS:
                head, mean, std = legacy.v3l_a0.model_pack_from_cache(
                    frozen["caches"][operator], "FINAL", 0
                )
                pred_low = legacy.v3l_a0.v3j_b.score_map(
                    "context", head, context_map, mean, std, frozen["bound"]
                )
                step = support * F.interpolate(pred_low, size=base.shape[-2:], mode="bilinear", align_corners=False)
                current = source.V3W.delta_for(kind, model, {"hazy": hazy, "base": base, "steps": {operator: step}, "support": support}, operator)
                reference_render = torch.clamp(base + 0.25 * step, 0.0, 1.0)
                current_render = torch.clamp(base + 0.25 * (step + current), 0.0, 1.0)
                response = support * a0p.clamp_channelwise(4.0 * (current_render - reference_render), delta_bound)
                candidates = [
                    (BANK[0], torch.zeros_like(current)),
                    (BANK[1], support * a0p.clamp_channelwise(current, delta_bound)),
                    (BANK[2], support * a0p.clamp_channelwise(-current, delta_bound)),
                    (BANK[3], exact_half(current, support, delta_bound)),
                    (BANK[4], exact_half(-current, support, delta_bound)),
                    (BANK[5], response),
                    (BANK[6], support * a0p.clamp_channelwise(-response, delta_bound)),
                    (BANK[7], exact_half(response, support, delta_bound)),
                    (BANK[8], exact_half(-response, support, delta_bound)),
                ]
                unique_names: list[str] = []
                unique_values: list[torch.Tensor] = []
                for candidate_name, candidate in candidates:
                    candidate = candidate.to(dtype=torch.float32)
                    if not any(torch.equal(candidate, prior) for prior in unique_values):
                        unique_names.append(candidate_name)
                        unique_values.append(candidate)
                        candidate_counts[candidate_name] += 1
                max_unique = max(max_unique, len(unique_values))
                bound_excess = max(
                    bound_excess,
                    max(float(torch.clamp(value.abs() - delta_bound, min=0.0).max()) for value in unique_values),
                )
                inactive = support <= 0.0
                if bool(inactive.any()):
                    support_excess = max(
                        support_excess,
                        max(float(value.masked_select(inactive.expand_as(value)).abs().max()) for value in unique_values),
                    )
                noop_excess = max(noop_excess, float(unique_values[0].abs().max()))
                unit_key = hashlib.sha256(f"{name}\0{operator}".encode()).hexdigest()[:32]
                unit_path = cache_root / f"{unit_key}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "name": name,
                        "operator": operator,
                        "operator_pack": "FINAL",
                        "candidate_names": unique_names,
                        "base": base.detach().cpu().to(torch.float32),
                        "step": step.detach().cpu().to(torch.float16),
                        "support": support.detach().cpu().to(torch.float16),
                        "current": current.detach().cpu().to(torch.float16),
                        "candidates": torch.cat(unique_values, dim=0).detach().cpu().to(torch.float16),
                    },
                    unit_path,
                )
                row = {
                    "unit_key": unit_key,
                    "name": name,
                    "operator": operator,
                    "operator_pack": "FINAL",
                    "candidate_count": len(unique_values),
                    "candidate_names": unique_names,
                    "native_shape": f"{height}x{width}",
                    "cache_bytes": unit_path.stat().st_size,
                    "cache_sha256": sha256_file(unit_path),
                }
                append_jsonl(raw_cache_manifest, row)
                cache_rows.append(row)
                completed += 1
                if completed % 16 == 0 or completed == total_units:
                    emit(context, "cache_gt_free", completed, total_units)
            del input_img, padded, hazy, base, support, context_map

    cache_identity = [
        {key: row[key] for key in ("unit_key", "operator", "operator_pack", "candidate_count", "candidate_names", "native_shape", "cache_bytes", "cache_sha256")}
        for row in cache_rows
    ]
    cache_manifest_hash = canonical_hash(cache_identity)
    cache_manifest = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "unit_count": len(cache_rows),
        "image_count": len(development),
        "operators": list(OPERATORS),
        "cache_manifest_sha256": cache_manifest_hash,
        "cache_total_bytes": sum(row["cache_bytes"] for row in cache_rows),
        "raw_unit_manifest_cloud_only": True,
        "candidate_cache_cloud_only": True,
        "sealed_before_gt_access": True,
    }
    atomic_json(context.phase_output_path / "a0_cache_manifest.json", cache_manifest)
    cache_manifest_sealed_at = time.time()

    def load_label(name: str) -> torch.Tensor:
        stem, extension = os.path.splitext(name)
        candidates = (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png")
        for candidate in candidates:
            path = data_root / "train/gt" / candidate
            if path.is_file():
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB")).copy()
                return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0)
        raise FileNotFoundError(f"GT not found for development name {name}")

    score_rows: list[dict[str, Any]] = []
    gt_first_access_at: float | None = None
    for index, row in enumerate(cache_rows, 1):
        if gt_first_access_at is None:
            gt_first_access_at = time.time()
        label = load_label(row["name"]).unsqueeze(0).to(device)
        payload_cache = torch.load(cache_root / f"{row['unit_key']}.pt", map_location=device)
        base = payload_cache["base"].float()
        step = payload_cache["step"].float()
        support = payload_cache["support"].float()
        current = payload_cache["current"].float()
        candidates = payload_cache["candidates"].float()
        names = list(payload_cache["candidate_names"])
        if label.shape[-2:] != base.shape[-2:]:
            label = label[:, :, :base.shape[-2], :base.shape[-1]]
        old_low = torch.clamp(base + 0.125 * step, 0.0, 1.0)
        old_high = torch.clamp(base + 0.25 * step, 0.0, 1.0)
        old_low_mse = float((old_low - label).square().mean())
        old_high_mse = float((old_high - label).square().mean())
        low = torch.clamp(base + 0.125 * (step + candidates), 0.0, 1.0)
        high = torch.clamp(base + 0.25 * (step + candidates), 0.0, 1.0)
        low_mse = (low - label).square().mean(dim=(1, 2, 3))
        high_mse = (high - label).square().mean(dim=(1, 2, 3))
        safe = []
        for candidate_index in range(len(names)):
            low_value = float(low_mse[candidate_index])
            high_value = float(high_mse[candidate_index])
            tolerance = numerical_tolerance(old_low_mse, old_high_mse, low_value, high_value)
            if low_value <= old_low_mse + tolerance and high_value <= old_high_mse + tolerance:
                safe.append(candidate_index)
        if not safe or names[0] != "reference_noop":
            raise RuntimeError("proposal bank lost its safe no-op")
        selected_index = min(safe, key=lambda item: (float(high_mse[item]), item))
        selected_mse = float(high_mse[selected_index])
        proposal_gain = metric_psnr(selected_mse) - metric_psnr(old_high_mse)

        delta_bound = base.new_tensor(args.delta_bound).view(1, 3, 1, 1)
        target_step = support * a0p.clamp_channelwise(4.0 * (label - base), base.new_tensor(frozen["bound"]).view(1, 3, 1, 1))
        target_delta = support * a0p.clamp_channelwise(target_step - step, delta_bound)
        best_privileged_mse = old_high_mse
        for offset in range(0, len(GRID), 8):
            values = base.new_tensor(GRID[offset:offset + 8]).view(-1, 1, 1, 1)
            shrink = values * current
            direction = current + values * (target_delta - current)
            for family in (shrink, direction):
                family_low = torch.clamp(base + 0.125 * (step + family), 0.0, 1.0)
                family_high = torch.clamp(base + 0.25 * (step + family), 0.0, 1.0)
                family_low_mse = (family_low - label).square().mean(dim=(1, 2, 3))
                family_high_mse = (family_high - label).square().mean(dim=(1, 2, 3))
                for item in range(family.shape[0]):
                    low_value = float(family_low_mse[item])
                    high_value = float(family_high_mse[item])
                    tolerance = numerical_tolerance(old_low_mse, old_high_mse, low_value, high_value)
                    if low_value <= old_low_mse + tolerance and high_value <= old_high_mse + tolerance:
                        best_privileged_mse = min(best_privileged_mse, high_value)
        privileged_gain = metric_psnr(best_privileged_mse) - metric_psnr(old_high_mse)
        score_rows.append(
            {
                "name": row["name"],
                "operator": row["operator"],
                "candidate_count": len(names),
                "selected_candidate": names[selected_index],
                "proposal_gain_db": proposal_gain,
                "privileged_gain_db": privileged_gain,
                "repairable": float(proposal_gain > 1e-12),
                "anchor_excess_mse": float(low_mse[selected_index]) - old_low_mse,
                "predecessor_excess_mse": selected_mse - old_high_mse,
                "severe": float(proposal_gain <= -0.2),
                "hard": float(proposal_gain <= -0.5),
            }
        )
        if index % 16 == 0 or index == len(cache_rows):
            emit(context, "score_after_cache_seal", index, len(cache_rows))

    raw_scores = context.phase_output_path / "a0_scores_cloud_only.csv"
    write_csv(raw_scores, score_rows)
    keyed = {(row["name"], row["operator"]): row for row in score_rows}
    if len(keyed) != len(development) * len(OPERATORS):
        raise RuntimeError("paired score table is incomplete")
    arrays = {
        operator: {
            field: np.asarray([float(keyed[(name, operator)][field]) for name in development], dtype=np.float64)
            for field in ("proposal_gain_db", "privileged_gain_db", "repairable")
        }
        for operator in OPERATORS
    }
    draws = int(os.environ["CONVIR_ROUTE_BOOTSTRAP_DRAWS"])
    rng = np.random.Generator(np.random.PCG64(int(os.environ["CONVIR_ROUTE_BOOTSTRAP_SEED"])))
    gain_draws = np.empty(draws)
    retention_draws = np.empty(draws)
    repair_draws = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(0, len(development), len(development))
        gains = {operator: float(arrays[operator]["proposal_gain_db"][indices].mean()) for operator in OPERATORS}
        privileged = {operator: float(arrays[operator]["privileged_gain_db"][indices].mean()) for operator in OPERATORS}
        gain_draws[draw] = min(gains.values())
        retention_draws[draw] = min(
            gains[operator] / privileged[operator] if privileged[operator] > 0.0 else -math.inf
            for operator in OPERATORS
        )
        repair_draws[draw] = min(float(arrays[operator]["repairable"][indices].mean()) for operator in OPERATORS)

    def interval(values: np.ndarray) -> dict[str, float]:
        ordered = np.sort(values)
        return {
            "point": float(values.mean()),
            "lcb95": float(ordered[199]),
            "ucb95": float(ordered[3799]),
        }

    bootstrap = {
        "schema_version": 1,
        "draws": draws,
        "seed": 3407,
        "paired_image_resampling": True,
        "worse_operator_within_draw": True,
        "proposal_gain_db": interval(gain_draws),
        "privileged_retention": interval(retention_draws),
        "repairable_fraction": interval(repair_draws),
    }
    operator_rows = []
    for operator in OPERATORS:
        gains = arrays[operator]["proposal_gain_db"]
        privileged = arrays[operator]["privileged_gain_db"]
        operator_rows.append(
            {
                "operator": operator,
                "image_count": len(development),
                "mean_proposal_gain_db": float(gains.mean()),
                "mean_privileged_gain_db": float(privileged.mean()),
                "retention_ratio": float(gains.mean() / privileged.mean()),
                "repairable_fraction": float(arrays[operator]["repairable"].mean()),
                "p10_gain_db": float(np.quantile(gains, 0.10)),
                "cvar5_gain_db": float(np.mean(np.sort(gains)[:max(1, math.ceil(0.05 * len(gains)))])),
            }
        )
    structural_checks = {
        "development_count_768": len(development) == 768,
        "confirmation_count_432": len(confirmation) == 432,
        "role_overlap_zero": not set(development) & confirmation,
        "cache_units_complete": len(cache_rows) == 768 * 2,
        "score_units_complete": len(score_rows) == 768 * 2,
        "paired_operators_complete": len(keyed) == 768 * 2,
        "bank_cap_respected": max_unique <= 9,
        "finite_scores": all(
            math.isfinite(float(value))
            for row in score_rows
            for key, value in row.items()
            if key not in {"name", "operator", "selected_candidate"}
        ),
        "exact_noop": noop_excess == 0.0,
        "bound_integrity": bound_excess <= 1e-7,
        "support_integrity": support_excess == 0.0,
        "cache_sealed_before_gt": gt_first_access_at is not None and cache_manifest_sealed_at < gt_first_access_at,
        "anchor_safety": all(float(row["anchor_excess_mse"]) <= 1e-8 for row in score_rows),
        "predecessor_safety": all(float(row["predecessor_excess_mse"]) <= 1e-8 for row in score_rows),
        "severe_zero": not any(float(row["severe"]) for row in score_rows),
        "hard_zero": not any(float(row["hard"]) for row in score_rows),
    }
    structural_valid = all(structural_checks.values())
    gain_lcb = bootstrap["proposal_gain_db"]["lcb95"]
    retention_lcb = bootstrap["privileged_retention"]["lcb95"]
    repair_lcb = bootstrap["repairable_fraction"]["lcb95"]
    secondary_pass = retention_lcb >= RETENTION_FLOOR and repair_lcb >= REPAIRABLE_FLOOR
    if structural_valid and secondary_pass and gain_lcb >= GAIN_PASS:
        state = "COMPLETED_GATE_PASS"
        decision = "R3_A0_GT_FREE_PROPOSAL_ORACLE_PASS"
        authorizes = "R3_A1_AMENDMENT_REVIEW"
    elif structural_valid and secondary_pass and gain_lcb >= GAIN_INCONCLUSIVE:
        state = "COMPLETED_GATE_INCONCLUSIVE"
        decision = "R3_A0_GT_FREE_PROPOSAL_ORACLE_INCONCLUSIVE"
        authorizes = "R3_A1_STATE_ACTION_AMENDMENT_REVIEW"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "R3_A0_GT_FREE_PROPOSAL_ORACLE_FAIL_STOP"
        authorizes = "NONE"

    bank_identity = {
        "schema_version": 1,
        "primary_bank": list(BANK),
        "primary_bank_cap": 9,
        "candidate_sources": ["v4a_final_state_delta", "actual_high_render_response_backprojection"],
        "amplitudes": [0.0, 1.0],
        "half_sizes": {f"{h}x{w}": f"{lh}x{lw}" for (h, w), (lh, lw) in HALF_SIZES.items()},
        "deduplication": "exact_float32_tensor_first_identity",
        "observed_candidate_presence": candidate_counts,
        "maximum_unique_candidates": max_unique,
    }
    risk = {
        "schema_version": 1,
        "new_severe_count": sum(int(row["severe"]) for row in score_rows),
        "new_hard_count": sum(int(row["hard"]) for row in score_rows),
        "maximum_anchor_excess_mse": max(float(row["anchor_excess_mse"]) for row in score_rows),
        "maximum_predecessor_excess_mse": max(float(row["predecessor_excess_mse"]) for row in score_rows),
        "operators": {row["operator"]: {"p10_gain_db": row["p10_gain_db"], "cvar5_gain_db": row["cvar5_gain_db"]} for row in operator_rows},
    }
    resource = {
        "schema_version": 1,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "cache_total_bytes": cache_manifest["cache_total_bytes"],
        "cache_unit_count": len(cache_rows),
        "model_forward_images": len(development),
        "candidate_operator_units": len(cache_rows),
        "training_occurred": False,
    }
    access_audit = {
        "schema_version": 1,
        "route_commit": context.route_commit,
        "ledger_sha256": canonical_hash(ledger),
        "development_names_sha256": canonical_hash(sorted(development)),
        "confirmation_names_sha256": canonical_hash(sorted(confirmation)),
        "hazy_development_images_accessed": 768,
        "gt_development_images_accessed": 768,
        "gt_first_access_after_cache_manifest_seal": cache_manifest_sealed_at < float(gt_first_access_at),
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "historical_a1x_432_outcomes_touched": False,
        "proposal_generation_used_gt": False,
        "training_occurred": False,
    }
    atomic_json(context.phase_output_path / "a0_bank_identity.json", bank_identity)
    atomic_json(context.phase_output_path / "a0_structural_summary.json", {"schema_version": 1, "structural_valid": structural_valid, "checks": structural_checks})
    write_csv(context.phase_output_path / "a0_operator_aggregate.csv", operator_rows)
    atomic_json(context.phase_output_path / "a0_bootstrap_summary.json", bootstrap)
    atomic_json(context.phase_output_path / "a0_risk_summary.json", risk)
    atomic_json(context.phase_output_path / "a0_resource_summary.json", resource)
    atomic_json(context.phase_output_path / "a0_source_access_audit.json", access_audit)
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "structural_valid": structural_valid,
            "proposal_gain_lcb95_db": gain_lcb,
            "privileged_retention_lcb95": retention_lcb,
            "repairable_fraction_lcb95": repair_lcb,
            "cache_manifest_sha256": cache_manifest_hash,
            "development_count": len(development),
            "operator_count": len(OPERATORS),
            "training_occurred": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
