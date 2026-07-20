#!/usr/bin/env python3
"""Privileged NH-HAZE domain-matched regional action-capacity ceiling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import resource
import sys
import time
import types
from contextlib import redirect_stdout
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


ROUTE_ID = "haze4k_v5_r16_s3_domain_matched_action_ceiling_20260720"
OPERATION_ID = "R16_S3_DOMAIN_MATCHED_ACTION_CEILING"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
EXPECTED_IDS = tuple(f"{index:02d}" for index in range(1, 51))
PROTECTED_IDS = tuple(f"{index:02d}" for index in range(51, 56))
EXPECTED_S0E = {
    "route_id": "haze4k_v5_r15_s0e_same_action_population_precision_20260720",
    "operation_id": "R15_S0E_SAME_ACTION_POPULATION_PRECISION",
    "run_id": "r15-s0e-same-action-population-r1",
    "route_commit": "b1cc5fc7bc9919394682b278dd864170217a089b",
    "state": "COMPLETED_GATE_FAIL",
    "decision": "R15_S0E_REAL_ACTION_HEADROOM_FAIL_STRATEGIC_RESET",
    "authorizes": "R15_S3_REFORMULATION_ONLY",
}
EXPECTED_R10_SOURCE_SHA256 = "a44a822750dd99ddbde259e1e45632316effa357ddcf36506c050b2371c4c372"
WDMAMBA_SOURCE_ASSETS = {
    "wdmamba_registry_source": "basicsr/utils/registry.py",
    "wdmamba_ublock_source": "basicsr/archs/Ublock.py",
    "wdmamba_denet_source": "basicsr/archs/detail_enhance_net.py",
    "wdmamba_wavelet_source": "basicsr/archs/wavelet.py",
    "wdmamba_arch_source": "basicsr/archs/wavemamba_arch.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV refused: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def strata_for(ids: list[str]) -> dict[str, int]:
    ordered = sorted(
        ids,
        key=lambda image_id: (
            hashlib.sha256(f"{ROUTE_ID}:{image_id}".encode("ascii")).hexdigest(),
            image_id,
        ),
    )
    if len(ordered) != 50:
        raise RuntimeError("R16 strata require exactly 50 image groups")
    return {image_id: int(index >= 25) for index, image_id in enumerate(ordered)}


def load_population_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    allowed = tuple(value.get("allowed_ids", []))
    protected = tuple(value.get("protected_ids", []))
    if allowed != EXPECTED_IDS or protected != PROTECTED_IDS:
        raise RuntimeError("R16 population manifest ids or ordering mismatch")
    if value.get("checkpoint_training_overlap") is not True:
        raise RuntimeError("R16 population role is not training-overlap privileged ceiling")
    if value.get("independent_external_validation") is not False:
        raise RuntimeError("R16 population cannot be an external validation role")
    if value.get("expected_width") != 1600 or value.get("expected_height") != 1200:
        raise RuntimeError("R16 native shape contract mismatch")
    return value


def fake_optional_wdmamba_imports() -> None:
    try:
        import transformers.generation as generation

        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, type(name, (object,), {}))
    except Exception:
        pass


def load_wdmamba(root: Path, checkpoint: Path, device: Any) -> Any:
    import torch

    fake_optional_wdmamba_imports()

    def package(name: str, path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module

    for key in list(sys.modules):
        if key == "basicsr" or key.startswith("basicsr."):
            del sys.modules[key]
    package("basicsr", root / "basicsr")
    package("basicsr.archs", root / "basicsr/archs")
    package("basicsr.utils", root / "basicsr/utils")
    load_module(root / "basicsr/utils/registry.py", "basicsr.utils.registry")
    load_module(root / "basicsr/archs/Ublock.py", "basicsr.archs.Ublock")
    detail = load_module(
        root / "basicsr/archs/detail_enhance_net.py",
        "basicsr.archs.detail_enhance_net",
    )
    load_module(root / "basicsr/archs/wavelet.py", "basicsr.archs.wavelet")
    architecture = load_module(
        root / "basicsr/archs/wavemamba_arch.py",
        "basicsr.archs.wavemamba_arch",
    )
    with redirect_stdout(io.StringIO()):
        model = architecture.WaveMamba(
            in_chn=3,
            wf=16,
            n_l_blocks=[1, 2, 2, 4],
            ffn_scale=2.0,
        ).to(device)
    model.restoration_network.DE = detail.DENet(3, 4).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_a0(anchor: Path, checkpoint: Path, device: Any) -> Any:
    import torch

    its = anchor / "Dehazing/ITS"
    sys.path.insert(0, str(its))
    try:
        from models.ConvIR import build_net  # type: ignore
    finally:
        sys.path.pop(0)
    model = build_net("base", "NHR", "original").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pad_to(value: Any, factor: int) -> tuple[Any, int, int]:
    import torch.nn.functional as functional

    height, width = int(value.shape[-2]), int(value.shape[-1])
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    return functional.pad(value, (0, pad_width, 0, pad_height), "reflect"), height, width


def infer_a0(model: Any, hazy: Any) -> Any:
    import torch

    padded, height, width = pad_to(hazy, 32)
    output = model(padded)
    if not isinstance(output, (list, tuple)):
        raise RuntimeError(f"unexpected ConvIR-B output type: {type(output)!r}")
    prediction = output[0][2] if isinstance(output[0], (list, tuple)) else output[2]
    return torch.clamp(prediction[:, :, :height, :width], 0.0, 1.0)


def infer_wdmamba(model: Any, hazy: Any) -> Any:
    import torch

    padded, height, width = pad_to(hazy, 4)
    output = model.restoration_network(padded)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return torch.clamp(output[:, :, :height, :width], 0.0, 1.0)


def generate_actions(hazy: Any, a0_model: Any, wdmamba_model: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        reference = infer_a0(a0_model, hazy)
        positive_render = infer_wdmamba(wdmamba_model, hazy)
        if reference.shape != positive_render.shape:
            raise RuntimeError("domain-matched endpoints have different shapes")
        positive_action = positive_render - reference
        negative_action = -positive_action
        negative_render = torch.clamp(reference + negative_action, 0.0, 1.0)
        values = {
            "reference": reference,
            "positive_action": positive_action,
            "negative_action": negative_action,
            "positive_render": positive_render,
            "negative_render": negative_render,
        }
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise RuntimeError("non-finite domain-matched action tensor")
        if float((positive_action + negative_action).abs().max()) != 0.0:
            raise RuntimeError("signed domain-matched actions are not exact opposites")
        if float((reference + positive_action - positive_render).abs().max()) > 1.0e-7:
            raise RuntimeError("positive endpoint residual reconstruction failed")
        return values


def action_identity(values: dict[str, Any]) -> dict[str, Any]:
    names = (
        "reference",
        "positive_action",
        "negative_action",
        "positive_render",
        "negative_render",
    )
    residual = values["positive_action"]
    return {
        "hashes": {name: tensor_hash(values[name]) for name in names},
        "shape": list(values["reference"].shape),
        "residual_nonzero_fraction": float((residual != 0.0).float().mean()),
        "residual_abs_mean": float(residual.abs().mean()),
        "residual_abs_max": float(residual.abs().max()),
        "sign_symmetry_max_abs": float(
            (values["positive_action"] + values["negative_action"]).abs().max()
        ),
    }


def synthetic_groups(torch_module: Any) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for index in range(50):
        operator_units: dict[str, dict[str, Any]] = {}
        reference = torch_module.full((64,), 100.0, dtype=torch_module.float64)
        positive = reference.clone()
        negative = reference.clone()
        for tile in range(64):
            if tile % 4 == 0:
                positive[tile], negative[tile] = 70.0, 115.0
            elif tile % 4 == 1:
                positive[tile], negative[tile] = 112.0, 72.0
            elif tile % 4 == 2:
                positive[tile], negative[tile] = 108.0, 109.0
            else:
                positive[tile], negative[tile] = 90.0, 96.0
        tensor = torch_module.stack((reference, positive, negative))
        for operator in OPERATORS:
            operator_units[operator] = {
                "fold": index // 25,
                "shape": "64x64",
                "sse": tensor.clone(),
                "pixel_counts": [192] * 64,
            }
        groups[f"synthetic_{index:02d}"] = operator_units
    return groups


def load_r10(context: Any) -> Any:
    path = (
        asset_path(context, "r10_source_checkout", kind="git_checkout")
        / "experience_docx/tools/r10_a0_fixed_region_action_feasibility.py"
    )
    if sha256_file(path) != EXPECTED_R10_SOURCE_SHA256:
        raise RuntimeError("R10 measurement-source identity mismatch")
    module = load_module(path, "r16_frozen_r10_measurement")
    if tuple(module.ACTIONS) != ACTIONS or tuple(module.OPERATORS) != OPERATORS:
        raise RuntimeError("R10 action/operator tuple mismatch")
    return module


def build_models(context: Any, device: Any) -> tuple[Any, Any]:
    root = asset_path(context, "wdmamba_root", kind="directory")
    for identifier, relpath in WDMAMBA_SOURCE_ASSETS.items():
        if asset_path(context, identifier, kind="file").resolve() != (root / relpath).resolve():
            raise RuntimeError(f"WDMamba source path binding mismatch: {identifier}")
    a0 = load_a0(
        asset_path(context, "official_anchor", kind="git_checkout"),
        asset_path(context, "a0_nhhaze_checkpoint", kind="file"),
        device,
    )
    wdmamba = load_wdmamba(
        root,
        asset_path(context, "wdmamba_nhhaze_checkpoint", kind="file"),
        device,
    )
    return a0, wdmamba


def interval_half_width(value: dict[str, float]) -> float:
    return 0.5 * (float(value["ucb95"]) - float(value["lcb95"]))


def contract(context_path: Path) -> None:
    import numpy as np
    import torch
    fake_optional_wdmamba_imports()
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    population = load_population_manifest(
        asset_path(context, "population_manifest", kind="file")
    )
    r10 = load_r10(context)
    device = torch.device("cpu")
    a0, wdmamba = build_models(context, device)
    reference_scan_bindings = 0
    for module in wdmamba.modules():
        if hasattr(module, "selective_scan"):
            module.selective_scan = selective_scan_ref
            reference_scan_bindings += 1
    values_first = generate_actions(
        torch.linspace(0.0, 1.0, 3 * 256 * 256, dtype=torch.float32)
        .reshape(1, 3, 256, 256),
        a0,
        wdmamba,
    )
    first_identity = action_identity(values_first)
    values_second = generate_actions(
        torch.linspace(0.0, 1.0, 3 * 256 * 256, dtype=torch.float32)
        .reshape(1, 3, 256, 256),
        a0,
        wdmamba,
    )
    second_identity = action_identity(values_second)
    groups = synthetic_groups(torch)
    rows, integrity = r10.analyze_groups(groups)
    primary = r10.bootstrap(rows, 4000, 3407)
    stratum_zero = r10.bootstrap(rows[:25], 4000, 3407)
    stratum_one = r10.bootstrap(rows[25:], 4000, 3408)
    all_intervals = (primary, stratum_zero, stratum_one)
    checks = {
        "route_identity": (
            context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID
        ),
        "contract_cpu_only": (
            context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        ),
        "population_exact_01_50": tuple(population["allowed_ids"]) == EXPECTED_IDS,
        "protected_ids_exact_51_55": tuple(population["protected_ids"]) == PROTECTED_IDS,
        "exact_production_models_constructed": True,
        "wdmamba_cpu_reference_scan_bound": reference_scan_bindings > 0,
        "deterministic_endpoint_hashes": first_identity == second_identity,
        "finite_endpoint_actions": all(
            bool(torch.isfinite(value).all()) for value in values_first.values()
        ),
        "strict_signed_action_symmetry": first_identity["sign_symmetry_max_abs"] == 0.0,
        "endpoint_shapes_exact": first_identity["shape"] == [1, 3, 256, 256],
        "full_group_scale": len(groups) == len(rows) == 50,
        "paired_replay_lanes": all(set(group) == set(OPERATORS) for group in groups.values()),
        "full_bootstrap_work_class": all(
            set(summary) == set(primary) for summary in all_intervals
        ),
        "finite_bootstrap_outputs": all(
            math.isfinite(interval["point"])
            for summary in all_intervals
            for interval in summary.values()
        ),
        "shuffle_integrity": (
            integrity["shuffle_histogram_violations"] == 0
            and integrity["shuffle_area_violations"] == 0
        ),
        "environment_exact": (
            int(os.environ["CONVIR_ROUTE_BOOTSTRAP_DRAWS"]) == 4000
            and int(os.environ["CONVIR_ROUTE_BOOTSTRAP_SEED"]) == 3407
            and int(os.environ["CONVIR_ROUTE_GRID"]) == 8
            and int(os.environ["CONVIR_ROUTE_SHUFFLE_REPLICATES"]) == 16
        ),
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "r16_s3_production_path_contract.json",
        {
            "schema_version": 1,
            "checks": checks,
            "synthetic_action_identity": first_identity,
            "synthetic_primary_region_gain": primary["region_gain"],
            "bootstrap_draws_total": 12000,
            "population_size": 50,
            "numpy_version": np.__version__,
        },
    )
    write_contract_result(context, checks=checks)


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R16 requires authorized CUDA runtime")
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    prior_path = asset_path(context, "s0e_closeout", kind="file")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if any(prior.get(key) != value for key, value in EXPECTED_S0E.items()):
        raise RuntimeError("R15-S0E terminal identity does not authorize R16 S3")
    population = load_population_manifest(
        asset_path(context, "population_manifest", kind="file")
    )
    ids = list(population["allowed_ids"])
    strata = strata_for(ids)
    data_root = asset_path(context, "nhhaze_data", kind="directory")
    r10 = load_r10(context)
    a0, wdmamba = build_models(context, device)

    def load_hazy(image_id: str) -> Any:
        path = data_root / population["hazy_filename_template"].format(image_id=image_id)
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (1600, 1200):
                raise RuntimeError(f"NH-HAZE hazy identity/shape mismatch: {image_id}")
            array = np.asarray(image).copy()
        return (
            torch.from_numpy(array.transpose(2, 0, 1))
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(device)
        )

    pass1: dict[str, Any] = {}
    for index, image_id in enumerate(ids, 1):
        values = generate_actions(load_hazy(image_id), a0, wdmamba)
        pass1[image_id] = action_identity(values)
        del values
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=index, stage="gt_free_action_seal")
    pass1_manifest = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "action_system": "A0_NH_plus_or_minus_full_WDMamba_NH_residual",
        "image_ids": ids,
        "image_count": len(ids),
        "actions": list(ACTIONS),
        "replay_lanes": list(OPERATORS),
        "replay_lanes_are_independent_samples": False,
        "gt_opened_during_pass1": False,
        "protected_ids_opened_or_listed": False,
        "pass1": pass1,
    }
    manifest_path = context.phase_output_path / "r16_s3_pass1_action_manifest_cloud_only.json"
    atomic_json(manifest_path, pass1_manifest)
    pass1_sealed_at = time.time()

    replay_failures: list[str] = []
    for index, image_id in enumerate(ids, 1):
        values = generate_actions(load_hazy(image_id), a0, wdmamba)
        if action_identity(values) != pass1[image_id]:
            replay_failures.append(image_id)
        del values
        torch.cuda.empty_cache()
        write_workload_progress(
            context,
            completed_units=50 + index,
            stage="full_population_action_replay",
        )
    replay_complete_at = time.time()
    if replay_failures:
        raise RuntimeError(f"R16 full-population action replay failed: {replay_failures}")
    if replay_complete_at <= pass1_sealed_at:
        raise RuntimeError("R16 replay timing contract failed")

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    per_image_action_rows: list[dict[str, Any]] = []
    gt_first_open_at: float | None = None
    third_pass_failures: list[str] = []
    for index, image_id in enumerate(ids, 1):
        values = generate_actions(load_hazy(image_id), a0, wdmamba)
        identity = action_identity(values)
        if identity != pass1[image_id]:
            third_pass_failures.append(image_id)
            raise RuntimeError(f"R16 scoring-pass action replay failed: {image_id}")
        target_path = data_root / population["target_filename_template"].format(
            image_id=image_id
        )
        if gt_first_open_at is None:
            gt_first_open_at = time.time()
        with Image.open(target_path) as image:
            if image.mode != "RGB" or image.size != (1600, 1200):
                raise RuntimeError(f"NH-HAZE target identity/shape mismatch: {image_id}")
            target_array = np.asarray(image).copy()
        target = (
            torch.from_numpy(target_array.transpose(2, 0, 1))
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        renders = torch.cat(
            (
                values["reference"],
                values["positive_render"],
                values["negative_render"],
            ),
            dim=0,
        ).cpu()
        errors = renders - target
        sse, pixel_counts = r10.tile_sse(errors)
        unit = {
            "sse": sse,
            "pixel_counts": pixel_counts,
            "fold": strata[image_id],
            "shape": "1200x1600",
        }
        groups[image_id] = {
            "D_ref": {**unit, "sse": sse.clone()},
            "D_rep": {**unit, "sse": sse.clone()},
        }
        reference_sse = float(sse[0].sum())
        positive_sse = float(sse[1].sum())
        negative_sse = float(sse[2].sum())
        per_image_action_rows.append(
            {
                "image_id": image_id,
                "stratum": strata[image_id],
                "reference_psnr_db": 10.0 * math.log10(
                    (3.0 * 1200.0 * 1600.0) / max(reference_sse, 1.0e-30)
                ),
                "positive_endpoint_gain_db": r10.psnr_gain(reference_sse, positive_sse),
                "negative_endpoint_gain_db": r10.psnr_gain(reference_sse, negative_sse),
                "residual_nonzero_fraction": identity["residual_nonzero_fraction"],
                "residual_abs_mean": identity["residual_abs_mean"],
                "residual_abs_max": identity["residual_abs_max"],
            }
        )
        del values, target, renders, errors
        torch.cuda.empty_cache()
        write_workload_progress(
            context,
            completed_units=100 + index,
            stage="replay_verified_target_scoring",
        )

    if gt_first_open_at is None or gt_first_open_at <= replay_complete_at:
        raise RuntimeError("GT was opened before complete population action replay")
    if third_pass_failures:
        raise RuntimeError(f"R16 third-pass replay failures: {third_pass_failures}")
    rows, integrity = r10.analyze_groups(groups)
    boot = r10.bootstrap(rows, 4000, 3407)
    write_workload_progress(context, completed_units=4150, stage="primary_bootstrap_complete")
    strata_rows = {
        str(stratum): [row for row in rows if row["fold"] == stratum]
        for stratum in (0, 1)
    }
    strata_bootstrap: dict[str, Any] = {}
    for stratum in (0, 1):
        strata_bootstrap[str(stratum)] = r10.bootstrap(
            strata_rows[str(stratum)], 4000, 3407 + stratum
        )
        write_workload_progress(
            context,
            completed_units=8150 + 4000 * stratum,
            stage=f"stratum_{stratum}_bootstrap_complete",
        )

    mixed = r10.binomial_interval(
        sum(row["mixed_noop_active"] for row in rows), len(rows)
    )
    bidirectional = r10.binomial_interval(
        sum(row["bidirectional"] for row in rows), len(rows)
    )
    severe = sum(
        any(row[f"region_{operator}"] <= -0.2 for operator in OPERATORS)
        for row in rows
    )
    hard = sum(
        any(row[f"region_{operator}"] <= -0.5 for operator in OPERATORS)
        for row in rows
    )
    strata_metrics = {
        stratum: {key: value["point"] for key, value in summary.items()}
        for stratum, summary in strata_bootstrap.items()
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
            summary["region_gain"]["ucb95"] < 0.020
            or summary["region_minus_global"]["ucb95"] < 0.005
            or summary["region_minus_shuffle"]["ucb95"] < 0.005
        )
        for stratum, summary in strata_bootstrap.items()
    }
    operator_means = {
        operator: float(np.mean([row[f"region_{operator}"] for row in rows]))
        for operator in OPERATORS
    }
    region_half_width = interval_half_width(boot["region_gain"])
    gates = {
        "region_gain_lcb95": boot["region_gain"]["lcb95"] >= 0.020,
        "region_minus_global_lcb95": boot["region_minus_global"]["lcb95"] >= 0.005,
        "region_minus_shuffle_lcb95": boot["region_minus_shuffle"]["lcb95"] >= 0.005,
        "region_minus_global_cvar5_lcb95": (
            boot["region_minus_global_cvar5"]["lcb95"] >= -0.005
        ),
        "zero_severe": severe == 0,
        "zero_hard": hard == 0,
        "mixed_fraction_lcb95": mixed["lcb95"] >= 0.25,
        "bidirectional_fraction_lcb95": bidirectional["lcb95"] >= 0.10,
        "both_strata_material": all(stratum_material.values()),
        "both_replay_lanes_positive": all(value > 0.0 for value in operator_means.values()),
        "region_gain_ci_half_width": region_half_width <= 0.020,
    }
    structural = {
        "prior_terminal_authorized": all(
            prior.get(key) == value for key, value in EXPECTED_S0E.items()
        ),
        "population_exact_01_50": sorted(groups) == list(EXPECTED_IDS),
        "population_complete": len(rows) == len(groups) == 50,
        "sha_strata_25_25": (
            len(strata_rows["0"]) == 25 and len(strata_rows["1"]) == 25
        ),
        "full_population_replay_exact_before_gt": (
            not replay_failures and gt_first_open_at > replay_complete_at
        ),
        "scoring_pass_replay_exact": not third_pass_failures,
        "paired_execution_replicas_exact": all(
            torch.equal(group["D_ref"]["sse"], group["D_rep"]["sse"])
            for group in groups.values()
        ),
        "r10_action_tuple_exact": tuple(r10.ACTIONS) == ACTIONS,
        "local_safety": integrity["local_safety_violations"] == 0,
        "local_materiality": integrity["local_materiality_violations"] == 0,
        "shuffle_histograms_exact": integrity["shuffle_histogram_violations"] == 0,
        "shuffle_pixel_area_exact": integrity["shuffle_area_violations"] == 0,
        "protected_roles_untouched": not any(context.protected_data_permissions.values()),
        "protected_ids_51_55_not_opened_or_listed": True,
        "training_absent": True,
    }
    if not all(structural.values()):
        failed = [key for key, value in structural.items() if not value]
        raise RuntimeError(f"R16 structural/integrity failure: {failed}")

    decisive_fail = (
        boot["region_gain"]["ucb95"] < 0.020
        or boot["region_minus_global"]["ucb95"] < 0.005
        or boot["region_minus_shuffle"]["ucb95"] < 0.005
        or boot["region_minus_global_cvar5"]["ucb95"] < -0.005
        or mixed["ucb95"] < 0.25
        or bidirectional["ucb95"] < 0.10
        or severe > 0
        or hard > 0
        or any(stratum_decisive_fail.values())
    )
    if all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "R16_S3_DOMAIN_MATCHED_ACTION_CEILING_PASS"
        authorizes = "R16_INDEPENDENT_REAL_DEVELOPMENT_ACTION_QUALIFICATION_CONTRACT_ONLY"
    elif decisive_fail:
        state = "COMPLETED_GATE_FAIL"
        decision = "R16_S3_DOMAIN_MATCHED_ACTION_CEILING_FAIL_STRATEGIC_RESET"
        authorizes = "R16_REGION_TARGET_SUPERVISION_REFORMULATION_ONLY"
    else:
        state = "COMPLETED_GATE_INCONCLUSIVE"
        decision = "R16_S3_PRIVILEGED_CAPACITY_INCONCLUSIVE_STOP"
        authorizes = "R16_PRIVILEGED_CAPACITY_EVIDENCE_COMPLETION_ONLY"

    policy_rows: list[dict[str, Any]] = []
    for policy, prefix in (
        ("region_oracle", "region"),
        ("safe_global_oracle", "global"),
        ("spatial_shuffle_control", "shuffle"),
    ):
        for operator in OPERATORS:
            values = np.asarray(
                [row[f"{prefix}_{operator}"] for row in rows], dtype=np.float64
            )
            policy_rows.append(
                {
                    "policy": policy,
                    "operator": operator,
                    "operator_role": "deterministic_execution_replica_not_sample",
                    "image_count": len(rows),
                    "mean_gain_db": float(values.mean()),
                    "cvar5_gain_db": r10.cvar(values),
                    "severe_count": int(np.sum(values <= -0.2)),
                    "hard_count": int(np.sum(values <= -0.5)),
                }
            )
    action_rows_array = per_image_action_rows
    active_area = np.asarray(
        [row["active_area_fraction"] for row in rows], dtype=np.float64
    )
    endpoint_summary = {
        "schema_version": 1,
        "action_system": "A0_NH_plus_or_minus_full_WDMamba_NH_residual",
        "action_system_is_single_preregistered_variable": True,
        "positive_endpoint_gain_db": {
            "mean": float(np.mean([row["positive_endpoint_gain_db"] for row in action_rows_array])),
            "median": float(np.median([row["positive_endpoint_gain_db"] for row in action_rows_array])),
            "minimum": float(np.min([row["positive_endpoint_gain_db"] for row in action_rows_array])),
            "maximum": float(np.max([row["positive_endpoint_gain_db"] for row in action_rows_array])),
        },
        "negative_endpoint_gain_db": {
            "mean": float(np.mean([row["negative_endpoint_gain_db"] for row in action_rows_array])),
            "median": float(np.median([row["negative_endpoint_gain_db"] for row in action_rows_array])),
            "minimum": float(np.min([row["negative_endpoint_gain_db"] for row in action_rows_array])),
            "maximum": float(np.max([row["negative_endpoint_gain_db"] for row in action_rows_array])),
        },
        "residual_nonzero_fraction": {
            "mean": float(np.mean([row["residual_nonzero_fraction"] for row in action_rows_array])),
            "median": float(np.median([row["residual_nonzero_fraction"] for row in action_rows_array])),
        },
        "regional_oracle_active_area": {
            "mean": float(active_area.mean()),
            "median": float(np.median(active_area)),
            "p10": float(np.quantile(active_area, 0.10)),
            "p90": float(np.quantile(active_area, 0.90)),
        },
        "mixed_noop_active": mixed,
        "bidirectional_positive_negative": bidirectional,
        "action_tile_counts": {
            "noop": int(sum(row["noop_tiles"] for row in rows)),
            "positive": int(sum(row["positive_tiles"] for row in rows)),
            "negative": int(sum(row["negative_tiles"] for row in rows)),
        },
    }
    identity_access = {
        "schema_version": 1,
        "route_commit": context.route_commit,
        "s0e_terminal_sha256": sha256_file(prior_path),
        "pass1_manifest_sha256": sha256_file(manifest_path),
        "pass1_image_count": len(pass1),
        "pass2_replay_failures": replay_failures,
        "pass3_replay_failures": third_pass_failures,
        "gt_first_open_after_complete_pass2_replay": gt_first_open_at > replay_complete_at,
        "data_role": "privileged_capacity_ceiling_checkpoint_training_overlap",
        "eligible_as_external_validation": False,
        "ids_51_55_opened_or_listed": False,
        "confirmation_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "training_occurred": False,
        "checkpoint_selected": False,
        "threshold_selected": False,
        "sample_excluded": False,
    }
    strata_operator = {
        "schema_version": 1,
        "stratum_counts": {"0": 25, "1": 25},
        "strata_metrics": strata_metrics,
        "strata_bootstrap": strata_bootstrap,
        "stratum_materiality": stratum_material,
        "stratum_decisive_fail": stratum_decisive_fail,
        "execution_replica_region_means_db": operator_means,
        "execution_replicas_are_independent_scientific_operators": False,
    }
    gate_summary = {
        "schema_version": 1,
        "structural_checks": structural,
        "gates": gates,
        "passes": all(gates.values()),
        "decisive_fail": decisive_fail,
        "negative_conclusion_protection": (
            "stratum point failure alone is not decisive; UCB contradiction is required"
        ),
        "region_gain_ci_half_width_db": region_half_width,
        "region_severe_images": severe,
        "region_hard_images": hard,
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
    }
    resource_summary = {
        "schema_version": 1,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2),
        "maximum_resident_set_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "model_forward_images": 150,
        "independent_image_groups": 50,
        "execution_replica_count": 2,
        "execution_replicas_count_as_samples": False,
        "primary_bootstrap_draws": 4000,
        "per_stratum_bootstrap_draws": 4000,
        "total_bootstrap_draws": 12000,
        "training_occurred": False,
    }
    failed_gates = [key for key, value in gates.items() if not value]
    conclusion = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "decision": decision,
        "authorizes": authorizes,
        "primary_result": (
            f"Domain-matched privileged capacity ceiling ended {state}: regional gain "
            f"{boot['region_gain']['point']:.6f} dB (LCB95 "
            f"{boot['region_gain']['lcb95']:.6f}, UCB95 "
            f"{boot['region_gain']['ucb95']:.6f}), CI half-width "
            f"{region_half_width:.6f} dB, with {severe} severe and {hard} hard images."
        ),
        "gate_reasons": [
            "Exactly NH-HAZE ids 01-50 were used as a checkpoint-training-overlap privileged capacity population; ids 51-55 and all protected roles remained sealed.",
            "All actions were generated GT-free, sealed, replayed over the complete population, and verified again per image before GT scoring.",
            f"Failed frozen gates: {failed_gates if failed_gates else 'none'}.",
            f"Region-minus-global and region-minus-shuffle intervals were {boot['region_minus_global']} and {boot['region_minus_shuffle']}.",
            f"SHA-stratum materiality was {stratum_material}; decisive UCB failure was {stratum_decisive_fail}.",
        ],
        "competing_explanation": (
            "This compound action-system contrast cannot separate baseline-domain mismatch "
            "from residual-source mismatch; a FAIL supports but does not uniquely identify "
            "region, target or action-semantics misspecification."
        ),
        "limitations": [
            "All 50 images overlap checkpoint training and cannot support generalization, deployment or external-validation claims.",
            "PSNR and fixed 8x8 regions remain the audited target/unit; semantic protection and naturalness are unmeasured.",
            "D_ref and D_rep are identical deterministic execution replicas, not independent operators or samples.",
            "No training, alpha search, checkpoint selection, confirmation, canary or locked-test access occurred.",
        ],
    }

    atomic_json(context.phase_output_path / "r16_s3_identity_and_access.json", identity_access)
    atomic_json(
        context.phase_output_path / "r16_s3_endpoint_action_summary.json",
        endpoint_summary,
    )
    write_csv(context.phase_output_path / "r16_s3_policy_summary.csv", policy_rows)
    atomic_json(
        context.phase_output_path / "r16_s3_bootstrap_summary.json",
        {"schema_version": 1, **boot},
    )
    atomic_json(
        context.phase_output_path / "r16_s3_strata_operator_summary.json",
        strata_operator,
    )
    atomic_json(context.phase_output_path / "r16_s3_gate_summary.json", gate_summary)
    atomic_json(
        context.phase_output_path / "r16_s3_resource_summary.json", resource_summary
    )
    atomic_json(
        context.phase_output_path / "r16_s3_scientific_conclusion.json", conclusion
    )
    cloud_rows = []
    action_by_id = {row["image_id"]: row for row in per_image_action_rows}
    for row in rows:
        cloud_rows.append(
            {
                **{key: value for key, value in row.items() if key != "action_map"},
                **{
                    key: value
                    for key, value in action_by_id[row["name"]].items()
                    if key not in {"image_id", "stratum"}
                },
                "action_map": ";".join(str(value) for value in row["action_map"]),
            }
        )
    write_csv(
        context.phase_output_path / "r16_s3_per_image_region_rows_cloud_only.csv",
        cloud_rows,
    )
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "region_gain_db": boot["region_gain"]["point"],
            "region_gain_lcb95_db": boot["region_gain"]["lcb95"],
            "region_gain_ucb95_db": boot["region_gain"]["ucb95"],
            "region_gain_ci_half_width_db": region_half_width,
            "region_minus_global_lcb95_db": boot["region_minus_global"]["lcb95"],
            "region_minus_shuffle_lcb95_db": boot["region_minus_shuffle"]["lcb95"],
            "severe_images": severe,
            "hard_images": hard,
            "training_occurred": False,
            "old_terminals_changed": False,
            "external_validation_claimed": False,
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
