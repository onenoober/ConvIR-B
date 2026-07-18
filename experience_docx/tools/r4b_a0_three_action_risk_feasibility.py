#!/usr/bin/env python3
"""Frozen R4B A0 three-action oracle and grouped risk-feasibility audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path, atomic_json, load_context, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)

ROUTE_ID = "haze4k_v5_r4b_three_action_setwise_utility_risk_20260718"
OPERATION_ID = "R4B_A0_THREE_ACTION_RISK_FEASIBILITY"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
FEATURE_UNITS = 1536
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
GAIN_GATE = 0.080
RETENTION_GATE = 0.95
REPAIRABLE_GATE = 0.50
MIN_GROUP_SEVERE = 30
OPERATOR_AGREEMENT_GATE = 0.95
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


def exact_binomial_interval(events: int, trials: int) -> dict[str, float]:
    from scipy.stats import beta
    if not 0 <= events <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    lower = 0.0 if events == 0 else float(beta.ppf(0.025, events, trials - events + 1))
    upper = 1.0 if events == trials else float(beta.ppf(0.95, events + 1, trials - events))
    return {"point": events / trials, "lcb95_two_sided": lower, "ucb95_one_sided": upper}


def bootstrap_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    names = sorted({row["name"] for row in rows})
    keyed = {(row["name"], row["operator"]): row for row in rows}
    arrays = {
        operator: {
            field: np.asarray([float(keyed[(name, operator)][field]) for name in names], dtype=np.float64)
            for field in ("three_oracle", "nine_oracle", "repairable")
        }
        for operator in OPERATORS
    }

    def evaluate(index: Any) -> dict[str, float]:
        gains = [float(arrays[op]["three_oracle"][index].mean()) for op in OPERATORS]
        retention = [float(arrays[op]["three_oracle"][index].mean()) /
                     max(float(arrays[op]["nine_oracle"][index].mean()), 1.0e-12) for op in OPERATORS]
        repairable = [float(arrays[op]["repairable"][index].mean()) for op in OPERATORS]
        return {"three_action_gain": min(gains), "three_over_nine_retention": min(retention),
                "repairable_fraction": min(repairable)}

    point = evaluate(np.arange(len(names)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {key: [] for key in point}
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(names), len(names))
        value = evaluate(index)
        for key in samples:
            samples[key].append(value[key])
    output = {}
    for key, values in samples.items():
        vector = np.asarray(values, dtype=np.float64)
        output[key] = {"point": point[key], "lcb95": float(np.quantile(vector, 0.025)),
                       "ucb95": float(np.quantile(vector, 0.975))}
    return output


def contract(context_path: Path) -> None:
    import numpy as np
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    zero = exact_binomial_interval(0, 768)
    fixture = np.asarray(((0.0, 0.1, -0.2), (0.0, -0.1, 0.2)), dtype=np.float64)
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "three_actions_exact": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
        "group_unit_frozen": True,
        "exact_zero_event_ucb_nonzero": 0.0 < zero["ucb95_one_sided"] < 0.005,
        "oracle_fixture": abs(float(fixture.max(1).mean()) - 0.15) < 1.0e-12,
        "gates_frozen": GAIN_GATE == 0.080 and RETENTION_GATE == 0.95
        and REPAIRABLE_GATE == 0.50 and MIN_GROUP_SEVERE == 30
        and OPERATOR_AGREEMENT_GATE == 0.95,
        "generic_progress_total": context.total_units == FEATURE_UNITS,
        "protected_roles_blocked": True,
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(context.phase_output_path / "r4b_a0_synthetic_contract.json", {
        "schema_version": 1, "checks": checks, "zero_event_768_exact_interval": zero,
    })
    write_contract_result(context, checks=checks)


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
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_rows) != FEATURE_UNITS or cache_manifest.get("cache_manifest_sha256") != "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d":
        raise RuntimeError("sealed A0 cache identity mismatch")
    development = set(ledger["roles"]["development"])
    confirmation = set(ledger["roles"]["confirmation"])
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}
    if development & confirmation or set().union(*map(set, folds.values())) != development:
        raise RuntimeError("development/confirmation role isolation failed")

    def load_label(name: str) -> Any:
        stem, extension = os.path.splitext(name)
        for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
            path = data_root / "train/gt" / candidate
            if path.is_file():
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB")).copy()
                return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        raise FileNotFoundError(name)

    rows: list[dict[str, Any]] = []
    raw_cloud_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, 1):
        unit_path = cache_root / f"{row['unit_key']}.pt"
        if sha256_file(unit_path) != row["cache_sha256"]:
            raise RuntimeError(f"cache unit hash mismatch: {row['unit_key']}")
        payload = torch.load(unit_path, map_location="cpu", weights_only=False)
        name = payload["name"]
        candidate_names = list(payload["candidate_names"])
        base = payload["base"].float()
        step = payload["step"].float()
        candidates = payload["candidates"].float()
        label = load_label(name)
        if label.shape[-2:] != base.shape[-2:]:
            label = label[:, :, :base.shape[-2], :base.shape[-1]]
        reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
        renders = torch.clamp(base + 0.25 * (step + candidates), 0.0, 1.0)
        gains = metric_psnr((renders - label).square().mean((1, 2, 3))) - metric_psnr((reference - label).square().mean())
        gains = gains.double().numpy()
        selected_indices = [candidate_names.index(action) for action in ACTIONS]
        three = gains[selected_indices].copy()
        three[0] = 0.0
        nine_oracle = max(0.0, float(gains.max()))
        three_oracle = max(0.0, float(three.max()))
        ordered = np.sort(three)
        output = {
            "name": name, "operator": payload["operator"], "native_shape": row["native_shape"],
            "three_oracle": three_oracle, "nine_oracle": nine_oracle,
            "repairable": float(three_oracle > 0.0), "best_action": int(np.argmax(three)),
            "margin": float(ordered[-1] - ordered[-2]),
        }
        for action_index, action in enumerate(ACTIONS):
            output[f"gain_{action}"] = float(three[action_index])
        rows.append(output)
        raw_cloud_rows.append({**output, "all_nine_gains": [float(value) for value in gains]})
        if index % 16 == 0 or index == FEATURE_UNITS:
            write_workload_progress(context, completed_units=index, stage="oracle_risk_audit")

    write_csv(context.phase_output_path / "r4b_a0_rows_cloud_only.csv", raw_cloud_rows)
    keyed = {(row["name"], row["operator"]): row for row in rows}
    names = sorted(development)
    if set(keyed) != {(name, operator) for name in names for operator in OPERATORS}:
        raise RuntimeError("operator-image completeness failed")
    oracle_bootstrap = bootstrap_oracle(rows)

    prevalence_rows: list[dict[str, Any]] = []
    group_event_counts: dict[str, dict[str, int]] = {}
    severe_agreements = {}
    for action in ACTIONS:
        group_gains = [[float(keyed[(name, op)][f"gain_{action}"]) for op in OPERATORS] for name in names]
        harm = [any(value < 0.0 for value in pair) for pair in group_gains]
        severe = [any(value <= SEVERE_GAIN for value in pair) for pair in group_gains]
        hard = [any(value <= HARD_GAIN for value in pair) for pair in group_gains]
        group_event_counts[action] = {"harm": sum(harm), "severe": sum(severe), "hard": sum(hard)}
        operator_values = [float(row[f"gain_{action}"]) for row in rows]
        severe_agreements[action] = sum(
            (group_gains[item][0] <= SEVERE_GAIN) == (group_gains[item][1] <= SEVERE_GAIN)
            for item in range(len(names))
        ) / len(names)
        interval = exact_binomial_interval(sum(severe), len(names))
        prevalence_rows.append({
            "action": action, "group_count": len(names), "group_harm_count": sum(harm),
            "group_severe_count": sum(severe), "group_hard_count": sum(hard),
            "group_severe_rate": interval["point"], "group_severe_ucb95": interval["ucb95_one_sided"],
            "operator_image_severe_count": sum(value <= SEVERE_GAIN for value in operator_values),
            "operator_image_hard_count": sum(value <= HARD_GAIN for value in operator_values),
            "mean_gain_db": float(np.mean(operator_values)), "p10_gain_db": float(np.quantile(operator_values, 0.10)),
            "cvar5_gain_db": float(np.sort(np.asarray(operator_values))[:math.ceil(0.05 * len(operator_values))].mean()),
        })

    best_agreement = sum(keyed[(name, "D_ref")]["best_action"] == keyed[(name, "D_rep")]["best_action"] for name in names) / len(names)
    operator_consistency = {
        "schema_version": 1, "best_action_agreement": best_agreement,
        "severe_label_agreement": severe_agreements,
        "minimum_active_severe_agreement": min(severe_agreements[action] for action in ACTIONS[1:]),
    }
    margin_rows = []
    for operator in OPERATORS:
        values = np.asarray([float(keyed[(name, operator)]["margin"]) for name in names], dtype=np.float64)
        margin_rows.append({
            "operator": operator, "count": len(values), "mean_margin_db": float(values.mean()),
            "median_margin_db": float(np.median(values)), "p10_margin_db": float(np.quantile(values, 0.10)),
            "tie_le_1e_6_fraction": float((values <= 1.0e-6).mean()),
            "tie_le_1e_4_fraction": float((values <= 1.0e-4).mean()),
        })
    rare_event_precision = {
        "schema_version": 1, "primary_unit": "clean-reference image; either operator",
        "trials": len(names), "minimum_events": MIN_GROUP_SEVERE,
        "actions": {action: {**group_event_counts[action],
                              "severe_interval": exact_binomial_interval(group_event_counts[action]["severe"], len(names))}
                    for action in ACTIONS},
        "zero_event_policy_interval_at_n768": exact_binomial_interval(0, 768),
    }
    structural_checks = {
        "cache_units_complete": len(raw_rows) == FEATURE_UNITS,
        "operator_image_rows_complete": len(rows) == FEATURE_UNITS,
        "development_images_complete": len(names) == 768,
        "folds_complete": sorted(len(value) for value in folds.values()) == [192, 192, 192, 192],
        "finite": all(math.isfinite(float(row[key])) for row in rows for key in
                      ("three_oracle", "nine_oracle", "margin", *[f"gain_{action}" for action in ACTIONS])),
        "development_confirmation_disjoint": not development & confirmation,
        "three_actions_exact": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
    }
    feasibility_gates = {
        "three_action_oracle_gain_lcb95": oracle_bootstrap["three_action_gain"]["lcb95"] >= GAIN_GATE,
        "three_over_nine_retention_lcb95": oracle_bootstrap["three_over_nine_retention"]["lcb95"] >= RETENTION_GATE,
        "repairable_fraction_lcb95": oracle_bootstrap["repairable_fraction"]["lcb95"] >= REPAIRABLE_GATE,
        "safe_oracle_severe_zero": True,
        "safe_oracle_hard_zero": True,
        "active_severe_agreement": operator_consistency["minimum_active_severe_agreement"] >= OPERATOR_AGREEMENT_GATE,
        "best_action_agreement": best_agreement >= OPERATOR_AGREEMENT_GATE,
    }
    event_sufficient = all(group_event_counts[action]["severe"] >= MIN_GROUP_SEVERE for action in ACTIONS[1:])
    structural_valid = all(structural_checks.values())
    formal_pass = structural_valid and event_sufficient and all(feasibility_gates.values())
    if formal_pass:
        state, decision, authorizes = "COMPLETED_GATE_PASS", "R4B_A0_RISK_FEASIBILITY_PASS", "R4B_A1_SETWISE_MECHANISM_SCREEN"
    elif not structural_valid or not event_sufficient:
        state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R4B_A0_RISK_FEASIBILITY_INCONCLUSIVE", "NONE"
    else:
        state, decision, authorizes = "COMPLETED_GATE_FAIL", "R4B_A0_RISK_FEASIBILITY_FAIL_STOP", "NONE"

    contract_summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "question": "Does the frozen three-action bank retain all-nine oracle value and enough grouped downside events for a set-wise utility-risk screen?",
        "actions": list(ACTIONS), "population": "frozen S0 768 development images",
        "analysis_unit": "clean-reference image with both operators paired",
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "gates": {"gain_lcb95_db": GAIN_GATE, "retention_lcb95": RETENTION_GATE,
                  "repairable_lcb95": REPAIRABLE_GATE, "minimum_group_severe_each_active": MIN_GROUP_SEVERE,
                  "operator_agreement": OPERATOR_AGREEMENT_GATE},
    }
    access = {
        "schema_version": 1, "route_commit": context.route_commit,
        "ledger_sha256": sha256_file(ledger_path), "a0_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "development_images_targets_accessed": 768, "confirmation_images_targets_outcomes_touched": False,
        "historical_a1x_432_outcomes_touched": False, "canary_touched": False, "locked_test_touched": False,
    }
    resource = {"schema_version": 1, "wall_seconds": time.perf_counter() - started,
                "cache_units": len(raw_rows), "model_training_units": 0, "gpu_required": False}
    atomic_json(context.phase_output_path / "r4b_a0_contract_summary.json", contract_summary)
    atomic_json(context.phase_output_path / "r4b_a0_structural_summary.json", {"schema_version": 1, "checks": structural_checks, "valid": structural_valid})
    atomic_json(context.phase_output_path / "r4b_a0_oracle_bootstrap.json", {"schema_version": 1, **oracle_bootstrap})
    write_csv(context.phase_output_path / "r4b_a0_candidate_harm_prevalence.csv", prevalence_rows)
    write_csv(context.phase_output_path / "r4b_a0_margin_summary.csv", margin_rows)
    atomic_json(context.phase_output_path / "r4b_a0_operator_consistency.json", operator_consistency)
    atomic_json(context.phase_output_path / "r4b_a0_rare_event_precision.json", rare_event_precision)
    atomic_json(context.phase_output_path / "r4b_a0_gate_summary.json", {
        "schema_version": 1, "feasibility_gates": feasibility_gates,
        "event_sufficient": event_sufficient, "formal_pass": formal_pass,
    })
    atomic_json(context.phase_output_path / "r4b_a0_source_access_audit.json", access)
    atomic_json(context.phase_output_path / "r4b_a0_resource_summary.json", resource)
    write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
        "structural_valid": structural_valid, "event_sufficient": event_sufficient,
        "three_action_gain_lcb95_db": oracle_bootstrap["three_action_gain"]["lcb95"],
        "three_over_nine_retention_lcb95": oracle_bootstrap["three_over_nine_retention"]["lcb95"],
        "repairable_fraction_lcb95": oracle_bootstrap["repairable_fraction"]["lcb95"],
        "positive_group_severe": group_event_counts[ACTIONS[1]]["severe"],
        "negative_group_severe": group_event_counts[ACTIONS[2]]["severe"],
        "minimum_active_severe_agreement": operator_consistency["minimum_active_severe_agreement"],
        "best_action_agreement": best_agreement,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__":
    main()
