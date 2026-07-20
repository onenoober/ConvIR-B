#!/usr/bin/env python3
"""Full-population same-action real-development precision qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)

import r15_s0a_measurement_qualification as s0a


ROUTE_ID = "haze4k_v5_r15_s0e_same_action_population_precision_20260720"
OPERATION_ID = "R15_S0E_SAME_ACTION_POPULATION_PRECISION"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
EXPECTED_S0A = {
    "state": "COMPLETED_GATE_INCONCLUSIVE",
    "decision": "R15_S0A_MEASUREMENT_INPUT_INCONCLUSIVE_STOP",
    "authorizes": "R15_S0A_EVIDENCE_COMPLETION_ONLY",
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tensor_hash(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "hazy_file", "gt_file", "width", "height", "pixels"}
    if len(rows) != 55 or not rows or not required.issubset(rows[0]):
        raise RuntimeError("NH-HAZE pair manifest contract failed")
    ids = [row["image_id"] for row in rows]
    if sorted(ids) != [f"{index:02d}" for index in range(1, 56)] or len(set(ids)) != 55:
        raise RuntimeError("NH-HAZE pair ids are incomplete or duplicated")
    return sorted(rows, key=lambda row: row["image_id"])


def strata_for(ids: list[str]) -> dict[str, int]:
    ordered = sorted(
        ids,
        key=lambda image_id: (hashlib.sha256(f"{ROUTE_ID}:{image_id}".encode()).hexdigest(), image_id),
    )
    return {image_id: int(index >= 27) for index, image_id in enumerate(ordered)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV refused: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contract(context_path: Path) -> None:
    import numpy as np

    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    ids = [f"{index:02d}" for index in range(1, 56)]
    strata = strata_for(ids)
    rng1 = np.random.Generator(np.random.PCG64(3407))
    rng2 = np.random.Generator(np.random.PCG64(3407))
    fixture = np.linspace(-0.05, 0.15, 55, dtype=np.float64)
    draws1 = np.asarray([fixture[rng1.integers(0, 55, 55)].mean() for _ in range(4000)])
    draws2 = np.asarray([fixture[rng2.integers(0, 55, 55)].mean() for _ in range(4000)])
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "actions_exact": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
        "operators_exact": OPERATORS == ("D_ref", "D_rep"),
        "environment_exact": (
            int(os.environ["CONVIR_ROUTE_BOOTSTRAP_DRAWS"]) == 4000
            and int(os.environ["CONVIR_ROUTE_BOOTSTRAP_SEED"]) == 3407
            and int(os.environ["CONVIR_ROUTE_GRID"]) == 8
            and int(os.environ["CONVIR_ROUTE_SHUFFLE_REPLICATES"]) == 16
            and float(os.environ["CONVIR_ROUTE_ACTION_SCALE"]) == 0.25
        ),
        "strata_exact_27_28": list(strata.values()).count(0) == 27 and list(strata.values()).count(1) == 28,
        "strata_order_invariant": strata == strata_for(list(reversed(ids))),
        "bootstrap_deterministic": np.array_equal(draws1, draws2),
        "bootstrap_finite": bool(np.isfinite(draws1).all()),
        "gate_order_valid": (
            float(os.environ["CONVIR_ROUTE_ABSOLUTE_GAIN_DB"])
            > float(os.environ["CONVIR_ROUTE_INCREMENT_GAIN_DB"]) > 0.0
        ),
        "tail_order_valid": (
            float(os.environ["CONVIR_ROUTE_HARD_GAIN_DB"])
            < float(os.environ["CONVIR_ROUTE_SEVERE_GAIN_DB"])
            < float(os.environ["CONVIR_ROUTE_TAIL_MARGIN_DB"])
        ),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "synthetic_population_precision_contract.json",
        {
            "schema_version": 1,
            "checks": checks,
            "stratum_counts": {"0": 27, "1": 28},
            "bootstrap_fixture_sha256": hashlib.sha256(draws1.tobytes()).hexdigest(),
        },
    )
    write_contract_result(context, checks=checks)


def generate_actions(
    input_tensor: Any, source: Any, legacy: Any, frozen: Any, model: Any,
    kind: str, args: Any, r3: Any, device: Any,
) -> dict[str, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        padded, height, width = legacy.pad_to_factor(input_tensor)
        hazy = padded[:, :, :height, :width]
        base = legacy.v3l_a0.forward_final(frozen["base"], padded, height, width)
        gate_full, _, _ = frozen["gate_producer"](padded)
        hard_gate = legacy.action_gate_from_full(gate_full, legacy.action_shape_for_input(padded)).to(device)
        support = legacy.output_gate_from_action_gate(hard_gate, base.shape[-2:])
        context_map, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
        delta_bound = base.new_tensor(args.delta_bound).view(1, 3, 1, 1)
        result: dict[str, dict[str, Any]] = {}
        for operator in OPERATORS:
            head, mean, std = legacy.v3l_a0.model_pack_from_cache(frozen["caches"][operator], "FINAL", 0)
            pred_low = legacy.v3l_a0.v3j_b.score_map("context", head, context_map, mean, std, frozen["bound"])
            step = support * functional.interpolate(pred_low, size=base.shape[-2:], mode="bilinear", align_corners=False)
            current = source.V3W.delta_for(
                kind, model, {"hazy": hazy, "base": base, "steps": {operator: step}, "support": support}, operator,
            )
            positive = support * r3.clamp_channelwise(current, delta_bound)
            negative = support * r3.clamp_channelwise(-current, delta_bound)
            reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
            positive_render = torch.clamp(base + 0.25 * (step + positive), 0.0, 1.0)
            negative_render = torch.clamp(base + 0.25 * (step + negative), 0.0, 1.0)
            result[operator] = {
                "base": base, "step": step, "support": support,
                "positive": positive, "negative": negative,
                "reference": reference, "positive_render": positive_render,
                "negative_render": negative_render,
            }
    return result


def action_identity(values: dict[str, Any]) -> dict[str, Any]:
    names = (
        "base", "step", "support", "positive", "negative",
        "reference", "positive_render", "negative_render",
    )
    return {
        "hashes": {name: tensor_hash(values[name]) for name in names},
        "shape": list(values["reference"].shape),
        "support_fraction": float((values["support"] > 0.0).float().mean()),
        "maximum_action_abs": float(values["positive"].abs().max()),
        "maximum_render_delta_abs": max(
            float((values["positive_render"] - values["reference"]).abs().max()),
            float((values["negative_render"] - values["reference"]).abs().max()),
        ),
        "sign_symmetry_max_abs": float((values["positive"] + values["negative"]).abs().max()),
    }


def interval_half_width(value: dict[str, float]) -> float:
    return 0.5 * (float(value["ucb95"]) - float(value["lcb95"]))


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S0E requires authorized CUDA runtime")
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    prior = json.loads(asset_path(context, "s0a_closeout", kind="file").read_text(encoding="utf-8"))
    if any(prior.get(key) != value for key, value in EXPECTED_S0A.items()):
        raise RuntimeError("S0A terminal tuple does not authorize evidence completion")
    pairs = read_pairs(asset_path(context, "v27_pairs", kind="file"))
    strata = strata_for([row["image_id"] for row in pairs])
    data_root = asset_path(context, "nhhaze_data", kind="directory")
    r3, source, legacy, frozen, model, kind, args = s0a._build_frozen_objects(context, device)
    r10_path = asset_path(context, "r10_source_checkout", kind="git_checkout") / "experience_docx/tools/r10_a0_fixed_region_action_feasibility.py"
    if s0a.sha256_file(r10_path) != s0a.EXPECTED_R10_SOURCE_SHA256:
        raise RuntimeError("R10 source identity mismatch")
    r10 = s0a.load_module(r10_path, "r15_s0e_frozen_r10")
    if tuple(r10.ACTIONS) != ACTIONS or tuple(r10.OPERATORS) != OPERATORS:
        raise RuntimeError("R10 action/operator identity mismatch")

    def load_hazy(row: dict[str, str]) -> Any:
        with Image.open(data_root / row["hazy_file"]) as image:
            array = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(device)

    pass1: dict[str, Any] = {}
    for index, row in enumerate(pairs, 1):
        values = generate_actions(load_hazy(row), source, legacy, frozen, model, kind, args, r3, device)
        pass1[row["image_id"]] = {operator: action_identity(values[operator]) for operator in OPERATORS}
        del values
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=index, stage="gt_free_action_seal")
    manifest = {
        "schema_version": 1, "route_id": ROUTE_ID, "image_count": 55,
        "actions": list(ACTIONS), "operators": list(OPERATORS),
        "gt_opened_during_pass1": False, "pass1": pass1,
    }
    manifest_path = context.phase_output_path / "r15_s0e_pass1_action_manifest_cloud_only.json"
    atomic_json(manifest_path, manifest)
    manifest_sealed_at = time.time()

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    replay_failures = []
    gt_first_open_at = None
    model_support = []
    max_actions = []
    max_render_deltas = []
    per_image_rows = []
    for index, row in enumerate(pairs, 1):
        image_id = row["image_id"]
        values = generate_actions(load_hazy(row), source, legacy, frozen, model, kind, args, r3, device)
        current_identity = {operator: action_identity(values[operator]) for operator in OPERATORS}
        if current_identity != pass1[image_id]:
            replay_failures.append(image_id)
        if gt_first_open_at is None:
            gt_first_open_at = time.time()
        with Image.open(data_root / row["gt_file"]) as image:
            gt_array = np.asarray(image.convert("RGB")).copy()
        gt = torch.from_numpy(gt_array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        operator_units = {}
        image_summary: dict[str, Any] = {"image_id": image_id, "stratum": strata[image_id]}
        for operator in OPERATORS:
            item = values[operator]
            renders = torch.cat((item["reference"], item["positive_render"], item["negative_render"]), dim=0).cpu()
            errors = renders - gt
            sse, pixel_counts = r10.tile_sse(errors)
            operator_units[operator] = {
                "sse": sse, "pixel_counts": pixel_counts,
                "fold": strata[image_id], "shape": "1200x1600",
            }
            identity = current_identity[operator]
            model_support.append(identity["support_fraction"])
            max_actions.append(identity["maximum_action_abs"])
            max_render_deltas.append(identity["maximum_render_delta_abs"])
            image_summary[f"support_{operator}"] = identity["support_fraction"]
            image_summary[f"max_action_{operator}"] = identity["maximum_action_abs"]
            image_summary[f"max_render_delta_{operator}"] = identity["maximum_render_delta_abs"]
        groups[image_id] = operator_units
        per_image_rows.append(image_summary)
        del values, gt
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=55 + index, stage="replay_and_score")

    if replay_failures:
        raise RuntimeError(f"two-pass action identity replay failed: {replay_failures}")
    if gt_first_open_at is None or gt_first_open_at <= manifest_sealed_at:
        raise RuntimeError("GT access preceded complete action-manifest seal")
    rows, integrity = r10.analyze_groups(groups)
    boot = r10.bootstrap(rows, 4000, 3407)
    write_workload_progress(context, completed_units=4110, stage="primary_bootstrap_complete")

    mixed = r10.binomial_interval(sum(row["mixed_noop_active"] for row in rows), len(rows))
    bidirectional = r10.binomial_interval(sum(row["bidirectional"] for row in rows), len(rows))
    severe = sum(any(row[f"region_{operator}"] <= -0.2 for operator in OPERATORS) for row in rows)
    hard = sum(any(row[f"region_{operator}"] <= -0.5 for operator in OPERATORS) for row in rows)
    strata_rows = {
        str(stratum): [row for row in rows if row["fold"] == stratum]
        for stratum in (0, 1)
    }
    strata_bootstrap = {}
    for stratum in (0, 1):
        strata_bootstrap[str(stratum)] = r10.bootstrap(
            strata_rows[str(stratum)], 4000, 3407 + stratum,
        )
        write_workload_progress(
            context, completed_units=8110 + 4000 * stratum,
            stage=f"stratum_{stratum}_bootstrap_complete",
        )
    strata_metrics = {
        stratum: {key: value["point"] for key, value in values.items()}
        for stratum, values in strata_bootstrap.items()
    }
    operator_means = {
        operator: float(np.mean([row[f"region_{operator}"] for row in rows]))
        for operator in OPERATORS
    }
    stratum_material = {
        stratum: (
            metrics["region_gain"] >= 0.020
            and metrics["region_minus_global"] >= 0.005
            and metrics["region_minus_shuffle"] >= 0.005
        )
        for stratum, metrics in strata_metrics.items()
    }
    stratum_decisive_fail = {
        stratum: (
            values["region_gain"]["ucb95"] < 0.020
            or values["region_minus_global"]["ucb95"] < 0.005
            or values["region_minus_shuffle"]["ucb95"] < 0.005
        )
        for stratum, values in strata_bootstrap.items()
    }
    region_half_width = interval_half_width(boot["region_gain"])
    gates = {
        "region_gain_lcb95": boot["region_gain"]["lcb95"] >= 0.020,
        "region_minus_global_lcb95": boot["region_minus_global"]["lcb95"] >= 0.005,
        "region_minus_shuffle_lcb95": boot["region_minus_shuffle"]["lcb95"] >= 0.005,
        "region_minus_global_cvar5_lcb95": boot["region_minus_global_cvar5"]["lcb95"] >= -0.005,
        "zero_severe": severe == 0, "zero_hard": hard == 0,
        "mixed_fraction_lcb95": mixed["lcb95"] >= 0.25,
        "bidirectional_fraction_lcb95": bidirectional["lcb95"] >= 0.10,
        "both_strata_material": all(stratum_material.values()),
        "both_operators_positive": all(value > 0.0 for value in operator_means.values()),
        "region_gain_ci_half_width": region_half_width <= 0.020,
    }
    structural = {
        "prior_terminal_authorized": all(prior.get(key) == value for key, value in EXPECTED_S0A.items()),
        "complete_population": len(rows) == len(groups) == 55,
        "paired_operators": all(set(group) == set(OPERATORS) for group in groups.values()),
        "two_pass_replay_exact": not replay_failures,
        "gt_after_complete_manifest": gt_first_open_at > manifest_sealed_at,
        "r10_action_tuple_exact": tuple(r10.ACTIONS) == ACTIONS,
        "local_safety": integrity["local_safety_violations"] == 0,
        "local_materiality": integrity["local_materiality_violations"] == 0,
        "shuffle_histograms_exact": integrity["shuffle_histogram_violations"] == 0,
        "shuffle_pixel_area_exact": integrity["shuffle_area_violations"] == 0,
        "protected_roles_untouched": not any(context.protected_data_permissions.values()),
    }
    if not all(structural.values()):
        raise RuntimeError(f"S0E structural integrity failure: {[key for key, value in structural.items() if not value]}")

    decisive_fail = (
        boot["region_gain"]["ucb95"] < 0.020
        or boot["region_minus_global"]["ucb95"] < 0.005
        or boot["region_minus_shuffle"]["ucb95"] < 0.005
        or boot["region_minus_global_cvar5"]["ucb95"] < -0.005
        or mixed["ucb95"] < 0.25 or bidirectional["ucb95"] < 0.10
        or severe > 0 or hard > 0 or any(stratum_decisive_fail.values())
    )
    if all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "R15_S0E_SAME_ACTION_POPULATION_PRECISION_PASS"
        authorizes = "R15_S0B_IDENTIFIABILITY_MEASUREMENT_CONTRACT_REVIEW_ONLY"
    elif decisive_fail:
        state = "COMPLETED_GATE_FAIL"
        decision = "R15_S0E_REAL_ACTION_HEADROOM_FAIL_STRATEGIC_RESET"
        authorizes = "R15_S3_REFORMULATION_ONLY"
    else:
        state = "COMPLETED_GATE_INCONCLUSIVE"
        decision = "R15_S0E_SAME_ACTION_OR_PRECISION_INCONCLUSIVE_STOP"
        authorizes = "R15_REAL_DEVELOPMENT_EVIDENCE_COMPLETION_ONLY"

    policy_rows = []
    for policy, prefix in (("region_oracle", "region"), ("safe_global_oracle", "global"), ("spatial_shuffle_control", "shuffle")):
        for operator in OPERATORS:
            values = np.asarray([row[f"{prefix}_{operator}"] for row in rows], dtype=np.float64)
            policy_rows.append({
                "policy": policy, "operator": operator, "image_count": len(rows),
                "mean_gain_db": float(values.mean()), "cvar5_gain_db": r10.cvar(values),
                "severe_count": int(np.sum(values <= -0.2)), "hard_count": int(np.sum(values <= -0.5)),
            })
    active_area = np.asarray([row["active_area_fraction"] for row in rows], dtype=np.float64)
    support_summary = {
        "schema_version": 1,
        "model_support_fraction": {
            "mean": float(np.mean(model_support)), "median": float(np.median(model_support)),
            "p10": float(np.quantile(model_support, 0.10)), "p90": float(np.quantile(model_support, 0.90)),
            "zero_units": int(np.sum(np.asarray(model_support) == 0.0)),
        },
        "regional_oracle_active_area": {
            "mean": float(active_area.mean()), "median": float(np.median(active_area)),
            "p10": float(np.quantile(active_area, 0.10)), "p90": float(np.quantile(active_area, 0.90)),
        },
        "maximum_action_abs": {"mean": float(np.mean(max_actions)), "max": float(np.max(max_actions))},
        "maximum_render_delta_abs": {"mean": float(np.mean(max_render_deltas)), "max": float(np.max(max_render_deltas))},
        "mixed_noop_active": mixed, "bidirectional_positive_negative": bidirectional,
        "action_tile_counts": {
            "noop": int(sum(row["noop_tiles"] for row in rows)),
            "positive": int(sum(row["positive_tiles"] for row in rows)),
            "negative": int(sum(row["negative_tiles"] for row in rows)),
        },
    }
    identity_access = {
        "schema_version": 1, "route_commit": context.route_commit,
        "s0a_terminal_sha256": s0a.sha256_file(asset_path(context, "s0a_closeout", kind="file")),
        "pass1_manifest_sha256": s0a.sha256_file(manifest_path),
        "pass1_image_count": len(pass1), "two_pass_replay_failures": replay_failures,
        "gt_first_open_after_complete_pass1_seal": gt_first_open_at > manifest_sealed_at,
        "data_role": "development_screening_previously_used_by_v2_7",
        "eligible_as_final_external_validation": False,
        "confirmation_images_targets_outcomes_touched": False, "canary_touched": False,
        "locked_test_touched": False, "training_occurred": False,
        "checkpoint_selected": False, "threshold_selected": False, "sample_excluded": False,
    }
    strata_operator = {
        "schema_version": 1, "stratum_counts": {"0": 27, "1": 28},
        "strata_metrics": strata_metrics, "strata_bootstrap": strata_bootstrap,
        "stratum_materiality": stratum_material,
        "stratum_decisive_fail": stratum_decisive_fail,
        "operator_region_means_db": operator_means,
    }
    gate_summary = {
        "schema_version": 1, "structural_checks": structural, "gates": gates,
        "passes": all(gates.values()), "decisive_fail": decisive_fail,
        "region_gain_ci_half_width_db": region_half_width,
        "region_severe_images": severe, "region_hard_images": hard,
        "state": state, "decision": decision, "authorizes": authorizes,
    }
    resource_summary = {
        "schema_version": 1, "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "maximum_resident_set_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "model_forward_images": 110, "operator_action_units": 220,
        "primary_bootstrap_draws": 4000, "per_stratum_bootstrap_draws": 4000,
        "total_bootstrap_draws": 12000, "training_occurred": False,
    }
    failed_gates = [key for key, value in gates.items() if not value]
    conclusion = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": context.run_id, "decision": decision, "authorizes": authorizes,
        "primary_result": (
            f"Same-action real-development qualification ended {state}: region gain "
            f"{boot['region_gain']['point']:.6f} dB (LCB95 {boot['region_gain']['lcb95']:.6f}, "
            f"UCB95 {boot['region_gain']['ucb95']:.6f}), CI half-width {region_half_width:.6f} dB, "
            f"with {severe} severe and {hard} hard images."
        ),
        "gate_reasons": [
            "All 55 image actions were generated GT-free, sealed, exactly regenerated and verified before GT scoring.",
            f"Failed frozen gates: {failed_gates if failed_gates else 'none'}.",
            f"Region-minus-global and region-minus-shuffle LCB95 were {boot['region_minus_global']['lcb95']:.6f} and {boot['region_minus_shuffle']['lcb95']:.6f} dB.",
            f"Mixed and bidirectional LCB95 were {mixed['lcb95']:.6f} and {bidirectional['lcb95']:.6f}.",
            f"Audit-stratum materiality was {stratum_material} and decisive failure was {stratum_decisive_fail}; operator region means were {operator_means}.",
        ],
        "competing_explanation": "A failure can arise from domain/action/region mismatch rather than the absence of all useful inference information; a pass establishes only F00 measurement qualification, not deployment or target alignment.",
        "limitations": [
            "NH-HAZE was previously used and is not an independent external validation population.",
            "This is a privileged paired-target feasibility audit, not a deployable policy.",
            "PSNR and fixed 8x8 regions remain the audited F00 definition; semantic protection and naturalness are not measured.",
            "No training, human annotation, confirmation, canary or locked-test access occurred.",
        ],
    }
    atomic_json(context.phase_output_path / "r15_s0e_identity_and_access.json", identity_access)
    atomic_json(context.phase_output_path / "r15_s0e_action_support_summary.json", support_summary)
    write_csv(context.phase_output_path / "r15_s0e_policy_summary.csv", policy_rows)
    atomic_json(context.phase_output_path / "r15_s0e_bootstrap_summary.json", {"schema_version": 1, **boot})
    atomic_json(context.phase_output_path / "r15_s0e_strata_operator_summary.json", strata_operator)
    atomic_json(context.phase_output_path / "r15_s0e_gate_summary.json", gate_summary)
    atomic_json(context.phase_output_path / "r15_s0e_resource_summary.json", resource_summary)
    atomic_json(context.phase_output_path / "r15_s0e_scientific_conclusion.json", conclusion)
    write_csv(context.phase_output_path / "r15_s0e_per_image_rows_cloud_only.csv", rows)
    write_csv(context.phase_output_path / "r15_s0e_action_magnitude_rows_cloud_only.csv", per_image_rows)
    write_run_result(
        context, state=state, decision=decision, authorizes=authorizes,
        details={
            "region_gain_db": boot["region_gain"]["point"],
            "region_gain_lcb95_db": boot["region_gain"]["lcb95"],
            "region_gain_ci_half_width_db": region_half_width,
            "region_minus_global_lcb95_db": boot["region_minus_global"]["lcb95"],
            "region_minus_shuffle_lcb95_db": boot["region_minus_shuffle"]["lcb95"],
            "severe_images": severe, "hard_images": hard,
            "training_occurred": False, "old_terminals_changed": False,
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
