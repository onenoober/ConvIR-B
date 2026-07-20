#!/usr/bin/env python3
"""Qualify real-target data, frozen R10 actions and image-level precision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
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
    write_workload_progress,
)


ROUTE_ID = "haze4k_v5_r15_real_target_region_action_identifiability_20260720"
OPERATION_ID = "R15_S0A_MEASUREMENT_QUALIFICATION"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
EXPECTED_R3_SOURCE_SHA256 = "698baf415c12f33feeb05b327c5c680fab2e2bfd9fb3c7fa5a7d4d39c7f72b0d"
EXPECTED_R10_SOURCE_SHA256 = "a44a822750dd99ddbde259e1e45632316effa357ddcf36506c050b2371c4c372"
EXPECTED_R10_DECISION = "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_PASS"
EXPECTED_R14_ROUTE_STATUS = "\u5df2\u8017\u5c3d\uff0c\u5e94\u5173\u95ed\uff1b\u603b\u76ee\u6807\u9700\u8981\u6218\u7565\u91cd\u6784"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial counts")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return {"point": probability, "lcb95": center - half, "ucb95": center + half, "half_width": half}


def zero_event_ucb95(total: int) -> float:
    return 1.0 - 0.05 ** (1.0 / total)


def deterministic_probe_id(image_ids: list[str]) -> str:
    return min(
        image_ids,
        key=lambda image_id: (hashlib.sha256(f"{ROUTE_ID}:{image_id}".encode()).hexdigest(), image_id),
    )


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    ids = [f"{index:02d}" for index in range(1, 56)]
    first = deterministic_probe_id(ids)
    second = deterministic_probe_id(list(reversed(ids)))
    target = float(os.environ["CONVIR_ROUTE_CONCORDANCE_TARGET"])
    max_half = float(os.environ["CONVIR_ROUTE_CONCORDANCE_HALF_WIDTH_MAX"])
    qualifying_counts = [
        successes for successes in range(56)
        if wilson_interval(successes, 55)["lcb95"] >= target
        and wilson_interval(successes, 55)["half_width"] <= max_half
    ]
    fixture = [0.0, 0.25, -0.25]
    checks = {
        "contract_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_has_no_assets": not context.assets,
        "action_tuple_exact": ACTIONS == ("reference_noop", "state_positive_full", "state_negative_full"),
        "operator_tuple_exact": OPERATORS == ("D_ref", "D_rep"),
        "deterministic_probe_order_invariant": first == second,
        "pair_count_frozen": int(os.environ["CONVIR_ROUTE_EXPECTED_PAIRS"]) == 55,
        "native_shape_frozen": (int(os.environ["CONVIR_ROUTE_EXPECTED_WIDTH"]), int(os.environ["CONVIR_ROUTE_EXPECTED_HEIGHT"])) == (1600, 1200),
        "action_scale_frozen": float(os.environ["CONVIR_ROUTE_ACTION_SCALE"]) == 0.25,
        "sign_symmetry_fixture": fixture[1] == -fixture[2] and fixture[0] == 0.0,
        "concordance_gate_arithmetically_reachable": bool(qualifying_counts),
        "zero_event_gate_reachable": zero_event_ucb95(55) <= float(os.environ["CONVIR_ROUTE_ZERO_EVENT_UCB_MAX"]),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "synthetic_qualification_contract.json",
        {
            "schema_version": 1,
            "checks": checks,
            "probe_id": first,
            "minimum_qualifying_concordant_images": min(qualifying_counts),
            "minimum_qualifying_fraction": min(qualifying_counts) / 55.0,
            "zero_event_ucb95": zero_event_ucb95(55),
        },
    )
    write_contract_result(context, checks=checks)


def _read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "hazy_file", "gt_file", "width", "height", "pixels"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("pair manifest field contract failed")
    return rows


def _read_v27_variability(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 55 or any("alpha_a0p375_dPSNR" not in row for row in rows):
        raise RuntimeError("v2.7 planning proxy contract failed")
    return [float(row["alpha_a0p375_dPSNR"]) for row in rows]


def _build_frozen_objects(context: Any, device: Any) -> tuple[Any, Any, Any, Any, Any, str, Any]:
    import torch

    r3_root = asset_path(context, "r3_source_checkout", kind="git_checkout")
    r3_path = r3_root / "experience_docx/tools/r3_a0_gt_free_proposal_oracle.py"
    if sha256_file(r3_path) != EXPECTED_R3_SOURCE_SHA256:
        raise RuntimeError("R3 action generator source identity mismatch")
    r10_root = asset_path(context, "r10_source_checkout", kind="git_checkout")
    r10_path = r10_root / "experience_docx/tools/r10_a0_fixed_region_action_feasibility.py"
    if sha256_file(r10_path) != EXPECTED_R10_SOURCE_SHA256:
        raise RuntimeError("R10 action source identity mismatch")
    r3 = load_module(r3_path, "r15_frozen_r3")
    data_root = asset_path(context, "nhhaze_data", kind="directory")
    v3z_root = asset_path(context, "v3z_checkout", kind="git_checkout")
    v3s_root = asset_path(context, "v3s_checkout", kind="git_checkout")
    v3p_root = asset_path(context, "v3p_checkout", kind="git_checkout")
    a1f_root = asset_path(context, "a1f_checkout", kind="git_checkout")
    a0p = r3.load_module(a1f_root / "experience_docx/tools/chd_rm_v4a_a0p_audit.py", "r15_a0p")
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
        source_split="train", train_key="v3j_controller_train", formal_sample_count=1200,
        fold_count=5, proj_channels=24, channels=24, d7c_threshold=0.5773006677627563,
        delta_bound_multiplier=2.0, sample_count=128, epochs=16, risk_window=4,
        warmup_epochs=8, learning_rate=0.0005, weight_decay=0.00001,
        grad_clip_norm=0.1, cvar_fraction=0.25,
    )
    a0p.validate_frozen_args(args)
    v3s.asset_manifest(args)
    frozen = v3s.build_frozen_operator(args, legacy, device)
    args.delta_bound = tuple(args.delta_bound_multiplier * value for value in frozen["bound"])
    trace = json.loads(asset_path(context, "trace_manifest", kind="file").read_text(encoding="utf-8"))
    if trace.get("route_id") != "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714" or trace.get("replicate_id") != "r1":
        raise RuntimeError("v4a trace identity mismatch")
    if sha256_file(Path(source.__file__)) != trace.get("v3z_source_sha256"):
        raise RuntimeError("v3z source differs from v4a final-state source")
    payload = torch.load(asset_path(context, "final_state", kind="file"), map_location="cpu")
    if payload.get("state_kind") != "final" or payload.get("replicate_id") != "r1":
        raise RuntimeError("v4a final-state identity mismatch")
    models = source.V3W.import_v3w_models()
    cells = source.V3W.build_cells(models, None, args, device)
    _, (kind, objective, model) = next(iter(cells.items()))
    if kind != "output" or objective != "safety_curriculum":
        raise RuntimeError("unexpected v4a model cell")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return r3, source, legacy, frozen, model, kind, args


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R15 S0A requires authorized CUDA runtime")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    pairs = _read_pairs(asset_path(context, "v27_pairs", kind="file"))
    image_ids = [row["image_id"] for row in pairs]
    probe_id = deterministic_probe_id(image_ids)
    preflight = json.loads(asset_path(context, "v27_preflight", kind="file").read_text(encoding="utf-8"))
    r10_closeout = json.loads(asset_path(context, "r10_closeout", kind="file").read_text(encoding="utf-8"))
    r10_distribution = json.loads(asset_path(context, "r10_action_distribution", kind="file").read_text(encoding="utf-8"))
    r14 = json.loads(asset_path(context, "r14_conclusion", kind="file").read_text(encoding="utf-8"))
    nhhaze_root = asset_path(context, "nhhaze_data", kind="directory")
    pair_by_id = {row["image_id"]: row for row in pairs}
    if len(pair_by_id) != len(pairs) or probe_id not in pair_by_id:
        raise RuntimeError("pair identity duplication or probe failure")

    write_workload_progress(context, completed_units=1, stage="identity_loaded")
    probe = pair_by_id[probe_id]
    hazy_path = nhhaze_root / probe["hazy_file"]
    with Image.open(hazy_path) as image:
        hazy_array = np.asarray(image.convert("RGB")).copy()
    input_tensor = torch.from_numpy(hazy_array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(device)
    r3, source, legacy, frozen, model, kind, args = _build_frozen_objects(context, device)
    delta_bound = input_tensor.new_tensor(args.delta_bound).view(1, 3, 1, 1)
    action_payload: dict[str, dict[str, Any]] = {}
    raw_payload: dict[str, Any] = {"schema_version": 1, "probe_id": probe_id, "operators": {}}
    with torch.no_grad():
        padded, height, width = legacy.pad_to_factor(input_tensor)
        hazy = padded[:, :, :height, :width]
        base = legacy.v3l_a0.forward_final(frozen["base"], padded, height, width)
        gate_full, _, _ = frozen["gate_producer"](padded)
        hard_gate = legacy.action_gate_from_full(gate_full, legacy.action_shape_for_input(padded)).to(device)
        support = legacy.output_gate_from_action_gate(hard_gate, base.shape[-2:])
        context_map, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
        for operator in OPERATORS:
            head, mean, std = legacy.v3l_a0.model_pack_from_cache(frozen["caches"][operator], "FINAL", 0)
            pred_low = legacy.v3l_a0.v3j_b.score_map("context", head, context_map, mean, std, frozen["bound"])
            step = support * functional.interpolate(pred_low, size=base.shape[-2:], mode="bilinear", align_corners=False)
            current = source.V3W.delta_for(kind, model, {"hazy": hazy, "base": base, "steps": {operator: step}, "support": support}, operator)
            positive = support * r3.clamp_channelwise(current, delta_bound)
            negative = support * r3.clamp_channelwise(-current, delta_bound)
            reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
            positive_render = torch.clamp(base + 0.25 * (step + positive), 0.0, 1.0)
            negative_render = torch.clamp(base + 0.25 * (step + negative), 0.0, 1.0)
            sign_error = float((negative + positive).abs().max())
            noop_error = float((reference - torch.clamp(base + 0.25 * (step + torch.zeros_like(current)), 0.0, 1.0)).abs().max())
            max_action = float(positive.abs().max())
            render_delta = max(float((positive_render - reference).abs().max()), float((negative_render - reference).abs().max()))
            finite = all(bool(torch.isfinite(value).all()) for value in (reference, positive_render, negative_render, positive, negative))
            native_shape = list(reference.shape[-2:]) == [1200, 1600]
            action_payload[operator] = {
                "finite": finite, "native_shape": native_shape, "render_shape": list(reference.shape),
                "sign_symmetry_max_abs": sign_error, "noop_identity_max_abs": noop_error,
                "maximum_action_abs": max_action, "maximum_render_delta_abs": render_delta,
                "support_fraction": float((support > 0.0).float().mean()),
            }
            raw_payload["operators"][operator] = {
                "step": step.detach().cpu().to(torch.float16),
                "support": support.detach().cpu().to(torch.float16),
                "positive": positive.detach().cpu().to(torch.float16),
                "negative": negative.detach().cpu().to(torch.float16),
                "reference": reference.detach().cpu().to(torch.float16),
                "positive_render": positive_render.detach().cpu().to(torch.float16),
                "negative_render": negative_render.detach().cpu().to(torch.float16),
            }
    raw_path = context.phase_output_path / "r15_s0a_probe_actions_cloud_only.pt"
    torch.save(raw_payload, raw_path)
    action_manifest = {
        "schema_version": 1, "probe_id": probe_id, "probe_selection": "lowest_sha256(route_id:image_id)",
        "actions": list(ACTIONS), "operators": list(OPERATORS), "action_scale": 0.25,
        "raw_payload_sha256": sha256_file(raw_path), "raw_payload_bytes": raw_path.stat().st_size,
        "gt_opened_before_seal": False, "operator_checks": action_payload,
    }
    action_manifest_path = context.phase_output_path / "r15_s0a_probe_action_manifest_cloud_only.json"
    atomic_json(action_manifest_path, action_manifest)
    action_sealed_at = time.time()
    write_workload_progress(context, completed_units=3, stage="action_manifest_sealed")

    pair_rows = []
    gt_first_open_at = None
    for index, row in enumerate(pairs, 1):
        hazy = nhhaze_root / row["hazy_file"]
        gt = nhhaze_root / row["gt_file"]
        if gt_first_open_at is None:
            gt_first_open_at = time.time()
        with Image.open(hazy) as hazy_image, Image.open(gt) as gt_image:
            hazy_size = list(hazy_image.size)
            gt_size = list(gt_image.size)
            hazy_mode = hazy_image.convert("RGB").mode
            gt_mode = gt_image.convert("RGB").mode
        pair_rows.append({
            "image_id": row["image_id"], "hazy_file": row["hazy_file"], "gt_file": row["gt_file"],
            "hazy_sha256": sha256_file(hazy), "gt_sha256": sha256_file(gt),
            "hazy_size": hazy_size, "gt_size": gt_size, "hazy_mode": hazy_mode, "gt_mode": gt_mode,
        })
        write_workload_progress(context, completed_units=3 + index, stage="pair_identity")

    canonical_pair_hash = canonical_hash(pair_rows)
    sizes_ok = all(row["hazy_size"] == [1600, 1200] and row["gt_size"] == [1600, 1200] for row in pair_rows)
    modes_ok = all(row["hazy_mode"] == "RGB" and row["gt_mode"] == "RGB" for row in pair_rows)
    hashes_unique = len({row["hazy_sha256"] for row in pair_rows}) == 55 and len({row["gt_sha256"] for row in pair_rows}) == 55
    source_checks = {
        "r10_terminal_pass": r10_closeout.get("decision") == EXPECTED_R10_DECISION,
        "r10_actions_exact": tuple(r10_distribution.get("actions", [])) == ACTIONS,
        "r14_route_closed": r14.get("route_status") == EXPECTED_R14_ROUTE_STATUS,
        "r3_source_exact": sha256_file(asset_path(context, "r3_source_checkout", kind="git_checkout") / "experience_docx/tools/r3_a0_gt_free_proposal_oracle.py") == EXPECTED_R3_SOURCE_SHA256,
        "r10_source_exact": sha256_file(asset_path(context, "r10_source_checkout", kind="git_checkout") / "experience_docx/tools/r10_a0_fixed_region_action_feasibility.py") == EXPECTED_R10_SOURCE_SHA256,
    }
    data_checks = {
        "preflight_status_ok": preflight.get("status") == "OK",
        "pair_count_exact": len(pair_rows) == 55 and len(pair_by_id) == 55,
        "ids_exact": sorted(image_ids) == [f"{index:02d}" for index in range(1, 56)],
        "native_sizes_exact": sizes_ok, "rgb_modes": modes_ok, "within_role_hashes_unique": hashes_unique,
        "gt_opened_after_action_seal": gt_first_open_at is not None and gt_first_open_at > action_sealed_at,
    }
    transport_checks = {
        "operators_complete": set(action_payload) == set(OPERATORS),
        "finite_native_renders": all(item["finite"] and item["native_shape"] for item in action_payload.values()),
        "exact_sign_symmetry": all(item["sign_symmetry_max_abs"] == 0.0 for item in action_payload.values()),
        "exact_noop_identity": all(item["noop_identity_max_abs"] == 0.0 for item in action_payload.values()),
        "nonzero_action": all(item["maximum_action_abs"] > 1.0e-8 and item["maximum_render_delta_abs"] > 1.0e-8 for item in action_payload.values()),
    }

    proxy = np.asarray(_read_v27_variability(asset_path(context, "v27_per_image", kind="file")), dtype=np.float64)
    proxy_sd = float(proxy.std(ddof=1))
    mean_half = 1.959963984540054 * proxy_sd / math.sqrt(len(proxy))
    mean_target = float(os.environ["CONVIR_ROUTE_MEAN_GAIN_HALF_WIDTH_MAX_DB"])
    n_required = math.ceil((1.959963984540054 * proxy_sd / mean_target) ** 2) if proxy_sd > 0.0 else 1
    concordance_rows = [
        {"successes": successes, **wilson_interval(successes, 55)}
        for successes in range(56)
    ]
    qualifying = [
        row for row in concordance_rows
        if row["lcb95"] >= float(os.environ["CONVIR_ROUTE_CONCORDANCE_TARGET"])
        and row["half_width"] <= float(os.environ["CONVIR_ROUTE_CONCORDANCE_HALF_WIDTH_MAX"])
    ]
    zero_ucb = zero_event_ucb95(55)
    precision = {
        "schema_version": 1, "image_count": 55, "grouping_unit": "paired_image_id",
        "nested_units_not_independent": ["region", "action", "operator", "rater"],
        "minimum_concordant_images_for_gates": min(row["successes"] for row in qualifying),
        "minimum_concordance_fraction_for_gates": min(row["successes"] for row in qualifying) / 55.0,
        "zero_event_exact_ucb95": zero_ucb,
        "v27_different_action_family_proxy_sd_db": proxy_sd,
        "v27_proxy_mean_ci_half_width_db": mean_half,
        "proxy_required_image_count_for_0p020_db_half_width": n_required,
        "proxy_role": "planning_only_different_action_family_not_a_scientific_endpoint",
        "concordance_precision_arithmetically_reachable": bool(qualifying),
        "zero_event_precision_pass": zero_ucb <= float(os.environ["CONVIR_ROUTE_ZERO_EVENT_UCB_MAX"]),
        "mean_gain_precision_qualified_by_proxy": mean_half <= mean_target,
    }
    # A different-family variance proxy cannot prove same-action precision. It
    # can only positively qualify when already sufficiently narrow; otherwise
    # the correct S0A result is evidence-completion/inconclusive, not FAIL.
    precision_qualified = precision["concordance_precision_arithmetically_reachable"] and precision["zero_event_precision_pass"] and precision["mean_gain_precision_qualified_by_proxy"]
    source_valid = all(source_checks.values())
    data_valid = all(data_checks.values())
    transport_valid = all(transport_checks.values())
    if source_valid and data_valid and transport_valid and precision_qualified:
        state = "COMPLETED_GATE_PASS"
        decision = "R15_S0A_MEASUREMENT_QUALIFICATION_PASS"
        authorizes = "R15_S0B_IDENTIFIABILITY_MEASUREMENT_CONTRACT_REVIEW_ONLY"
    elif source_valid and data_valid and not transport_valid:
        state = "COMPLETED_GATE_FAIL"
        decision = "R15_S0A_ACTION_TRANSPORT_FAIL_STRATEGIC_RESET"
        authorizes = "R15_S3_REFORMULATION_ONLY"
    else:
        state = "COMPLETED_GATE_INCONCLUSIVE"
        decision = "R15_S0A_MEASUREMENT_INPUT_INCONCLUSIVE_STOP"
        authorizes = "R15_S0A_EVIDENCE_COMPLETION_ONLY"

    dataset_identity = {
        "schema_version": 1, "pair_count": len(pair_rows), "image_ids_sha256": canonical_hash(sorted(image_ids)),
        "pair_file_manifest_sha256": canonical_pair_hash, "probe_id": probe_id, "checks": data_checks,
        "cloud_only_pair_hash_rows": True, "role": "development_screening_previously_used_by_v2_7",
        "eligible_as_final_independent_external_validation": False,
    }
    action_transport = {
        "schema_version": 1, "actions": list(ACTIONS), "operators": list(OPERATORS),
        "source_checks": source_checks, "transport_checks": transport_checks, "operator_evidence": action_payload,
        "probe_id": probe_id, "utility_estimated": False, "gt_used_to_generate_action": False,
    }
    access = {
        "schema_version": 1, "route_commit": context.route_commit, "data_role": "development_screening",
        "nhhaze_previously_used_by_v2_7": True, "gt_first_open_after_action_manifest_seal": data_checks["gt_opened_after_action_seal"],
        "confirmation_images_targets_outcomes_touched": False, "canary_touched": False, "locked_test_touched": False,
        "training_occurred": False, "checkpoint_selected": False, "threshold_selected": False,
        "sample_excluded": False, "probe_selected_from_outcomes": False,
    }
    resource_summary = {
        "schema_version": 1, "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "maximum_resident_set_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "model_forward_images": 1, "operator_count": 2, "pair_files_hashed": 110, "training_occurred": False,
    }
    qualification = {
        "schema_version": 1, "state": state, "decision": decision, "authorizes": authorizes,
        "source_valid": source_valid, "data_valid": data_valid, "transport_valid": transport_valid,
        "precision_qualified": precision_qualified, "old_terminals_changed": False,
    }
    conclusion = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": context.run_id, "decision": decision, "authorizes": authorizes,
        "primary_result": qualification,
        "gate_reasons": {"source": source_checks, "data": data_checks, "transport": transport_checks, "precision": precision},
        "competing_explanation": "A valid action transport does not establish target alignment; insufficient image-level precision does not establish action failure.",
        "limitations": ["one pre-outcome runtime probe only", "NH-HAZE already used for development", "v2.7 variance proxy uses a different action family", "no semantic or human labels collected"],
    }
    atomic_json(context.phase_output_path / "r15_s0a_qualification_summary.json", qualification)
    atomic_json(context.phase_output_path / "r15_s0a_dataset_identity.json", dataset_identity)
    atomic_json(context.phase_output_path / "r15_s0a_action_transport.json", action_transport)
    atomic_json(context.phase_output_path / "r15_s0a_precision_feasibility.json", precision)
    atomic_json(context.phase_output_path / "r15_s0a_provenance_and_access.json", access)
    atomic_json(context.phase_output_path / "r15_s0a_resource_summary.json", resource_summary)
    atomic_json(context.phase_output_path / "r15_s0a_scientific_conclusion.json", conclusion)
    atomic_json(context.phase_output_path / "r15_s0a_pair_hash_rows_cloud_only.json", pair_rows)
    write_workload_progress(context, completed_units=60, stage="qualification_finalized")
    write_run_result(
        context, state=state, decision=decision, authorizes=authorizes,
        details={
            "source_valid": source_valid, "data_valid": data_valid, "transport_valid": transport_valid,
            "precision_qualified": precision_qualified, "probe_id": probe_id,
            "pair_manifest_sha256": canonical_pair_hash, "training_occurred": False,
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
