#!/usr/bin/env python3
"""CDP-RM S1: preregistered region/action/target identifiability audit."""

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


ROUTE_ID = "nhhaze_cdprm_s1_region_target_identifiability_20260720"
OPERATION_ID = "CDPRM_S1_REGION_TARGET_IDENTIFIABILITY"
EXPECTED_IDS = tuple(f"{index:02d}" for index in range(1, 51))
PROTECTED_IDS = tuple(f"{index:02d}" for index in range(51, 56))
REGION_COUNT = 64
GRID = 8
ACTION_ALPHA = 0.25
REGION_ITERATIONS = 5
REGION_HEIGHT = 75
REGION_WIDTH = 100
SPATIAL_WEIGHT = 0.20
DEMAND_GAIN_DB = 0.005
HARM_RATIO = 1.01
SHUFFLE_REPLICATES = 16
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
SEVERE_GAIN_DB = -0.2
HARD_GAIN_DB = -0.5
R16_EXPECTED = {
    "route_id": "haze4k_v5_r16_s3_domain_matched_action_ceiling_20260720",
    "operation_id": "R16_S3_DOMAIN_MATCHED_ACTION_CEILING",
    "run_id": "r16-s3-domain-matched-action-ceiling-r4",
    "route_commit": "7c6b1b51e430ca46ecb4ae277c9b14e71253904c",
    "state": "COMPLETED_GATE_FAIL",
    "decision": "R16_S3_DOMAIN_MATCHED_ACTION_CEILING_FAIL_STRATEGIC_RESET",
    "authorizes": "R16_REGION_TARGET_SUPERVISION_REFORMULATION_ONLY",
}
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


def load_population_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if tuple(value.get("allowed_ids", [])) != EXPECTED_IDS:
        raise RuntimeError("CDP-RM S1 population ids mismatch")
    if tuple(value.get("protected_ids", [])) != PROTECTED_IDS:
        raise RuntimeError("CDP-RM S1 protected ids mismatch")
    if value.get("checkpoint_training_overlap") is not True:
        raise RuntimeError("CDP-RM S1 population role mismatch")
    if value.get("independent_external_validation") is not False:
        raise RuntimeError("CDP-RM S1 cannot use an external-validation role")
    if value.get("expected_width") != 1600 or value.get("expected_height") != 1200:
        raise RuntimeError("CDP-RM S1 native shape mismatch")
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
            in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0,
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


def build_models(context: Any, device: Any) -> tuple[Any, Any]:
    root = asset_path(context, "wdmamba_root", kind="directory")
    for identifier, relpath in WDMAMBA_SOURCE_ASSETS.items():
        if asset_path(context, identifier, kind="file").resolve() != (root / relpath).resolve():
            raise RuntimeError(f"WDMamba source path mismatch: {identifier}")
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


def generate_action(hazy: Any, a0_model: Any, wdmamba_model: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        reference = infer_a0(a0_model, hazy)
        endpoint = infer_wdmamba(wdmamba_model, hazy)
        if reference.shape != endpoint.shape:
            raise RuntimeError("CDP-RM S1 endpoints have different shapes")
        candidate = torch.clamp(
            reference + ACTION_ALPHA * (endpoint - reference), 0.0, 1.0,
        )
        if not all(bool(torch.isfinite(value).all()) for value in (reference, endpoint, candidate)):
            raise RuntimeError("CDP-RM S1 action contains non-finite values")
        if float((candidate - reference).abs().max()) > ACTION_ALPHA + 1.0e-7:
            raise RuntimeError("CDP-RM S1 action violates the L-infinity bound")
        return {"reference": reference, "endpoint": endpoint, "candidate": candidate}


def action_identity(values: dict[str, Any]) -> dict[str, Any]:
    delta = values["candidate"] - values["reference"]
    return {
        "hashes": {key: tensor_hash(values[key]) for key in ("reference", "endpoint", "candidate")},
        "shape": list(values["reference"].shape),
        "delta_abs_mean": float(delta.abs().mean()),
        "delta_abs_max": float(delta.abs().max()),
        "delta_nonzero_fraction": float((delta != 0.0).float().mean()),
    }


def fixed_grid_labels(height: int, width: int, device: Any) -> Any:
    import torch

    yy = torch.arange(height, device=device)[:, None]
    xx = torch.arange(width, device=device)[None, :]
    return ((yy * GRID) // height) * GRID + ((xx * GRID) // width)


def adaptive_region_labels(hazy: Any, reference: Any, *, perturbed: bool) -> Any:
    import torch
    import torch.nn.functional as functional

    source = torch.cat((hazy, reference), dim=1)
    small = functional.interpolate(
        source, size=(REGION_HEIGHT, REGION_WIDTH), mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).contiguous()
    if perturbed:
        small = torch.round(small * 255.0) / 255.0
    device = small.device
    yy, xx = torch.meshgrid(
        torch.arange(REGION_HEIGHT, device=device, dtype=torch.float32),
        torch.arange(REGION_WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )
    step_y = REGION_HEIGHT / GRID
    step_x = REGION_WIDTH / GRID
    offset_y = 0.25 * step_y if perturbed else 0.0
    offset_x = -0.25 * step_x if perturbed else 0.0
    seed_y = []
    seed_x = []
    for row in range(GRID):
        for column in range(GRID):
            seed_y.append(min(REGION_HEIGHT - 1, max(0, int((row + 0.5) * step_y + offset_y))))
            seed_x.append(min(REGION_WIDTH - 1, max(0, int((column + 0.5) * step_x + offset_x))))
    center_y = torch.tensor(seed_y, device=device, dtype=torch.float32)
    center_x = torch.tensor(seed_x, device=device, dtype=torch.float32)
    center_f = small[torch.tensor(seed_y, device=device), torch.tensor(seed_x, device=device)]
    flat_f = small.reshape(-1, small.shape[-1])
    flat_y = yy.reshape(-1)
    flat_x = xx.reshape(-1)
    labels = torch.zeros(flat_y.numel(), device=device, dtype=torch.long)
    for _iteration in range(REGION_ITERATIONS):
        feature_distance = ((flat_f[:, None, :] - center_f[None, :, :]) ** 2).mean(2)
        spatial_distance = (
            ((flat_y[:, None] - center_y[None, :]) / step_y) ** 2
            + ((flat_x[:, None] - center_x[None, :]) / step_x) ** 2
        )
        labels = torch.argmin(feature_distance + SPATIAL_WEIGHT * spatial_distance, dim=1)
        counts = torch.bincount(labels, minlength=REGION_COUNT).float()
        for region in range(REGION_COUNT):
            if counts[region] > 0:
                mask = labels == region
                center_f[region] = flat_f[mask].mean(0)
                center_y[region] = flat_y[mask].mean()
                center_x[region] = flat_x[mask].mean()
    # Make the fixed 64-region budget explicit even for an empty terminal cluster.
    for region, (sy, sx) in enumerate(zip(seed_y, seed_x)):
        labels[sy * REGION_WIDTH + sx] = region
    labels = labels.reshape(REGION_HEIGHT, REGION_WIDTH)
    full = functional.interpolate(
        labels[None, None].float(),
        size=(int(hazy.shape[-2]), int(hazy.shape[-1])),
        mode="nearest",
    )[0, 0].long()
    if int(torch.unique(full).numel()) != REGION_COUNT:
        raise RuntimeError("content-adaptive partition does not contain 64 regions")
    return full


def region_identity(labels: Any) -> dict[str, Any]:
    counts = labels.flatten().bincount(minlength=REGION_COUNT).cpu()
    return {
        "hash": tensor_hash(labels),
        "region_count": int((counts > 0).sum()),
        "minimum_area_fraction": float(counts.min() / counts.sum()),
        "maximum_area_fraction": float(counts.max() / counts.sum()),
    }


def psnr_gain(reference_sse: float, candidate_sse: float) -> float:
    return 10.0 * math.log10(max(reference_sse, 1.0e-30) / max(candidate_sse, 1.0e-30))


def pixel_errors(image: Any, target: Any) -> tuple[Any, Any, Any]:
    import torch

    rgb = (image - target).double().square().sum(0)
    chroma_image = torch.stack((image[0] - image[1], 0.5 * (image[0] + image[1]) - image[2]))
    chroma_target = torch.stack((target[0] - target[1], 0.5 * (target[0] + target[1]) - target[2]))
    chroma = (chroma_image - chroma_target).double().square().sum(0)
    luma_image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    luma_target = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
    gradient = torch.zeros_like(luma_image, dtype=torch.float64)
    dx = (luma_image[:, 1:] - luma_image[:, :-1]) - (luma_target[:, 1:] - luma_target[:, :-1])
    dy = (luma_image[1:, :] - luma_image[:-1, :]) - (luma_target[1:, :] - luma_target[:-1, :])
    gradient[:, :-1] += dx.double().square()
    gradient[:-1, :] += dy.double().square()
    return rgb, chroma, gradient


def aggregate_regions(values: Any, labels: Any) -> Any:
    import torch
    return torch.bincount(
        labels.flatten(), weights=values.flatten(), minlength=REGION_COUNT,
    )


def label_masks(labels: Any, reference: Any, candidate: Any, target: Any) -> dict[str, Any]:
    import torch

    ref_rgb, ref_chroma, ref_gradient = pixel_errors(reference, target)
    cand_rgb, cand_chroma, cand_gradient = pixel_errors(candidate, target)
    ref_rgb_region = aggregate_regions(ref_rgb, labels)
    cand_rgb_region = aggregate_regions(cand_rgb, labels)
    demand = torch.tensor(
        [psnr_gain(float(first), float(second)) >= DEMAND_GAIN_DB
         for first, second in zip(ref_rgb_region, cand_rgb_region)],
        device=labels.device, dtype=torch.bool,
    )
    ref_chroma_region = aggregate_regions(ref_chroma, labels)
    cand_chroma_region = aggregate_regions(cand_chroma, labels)
    ref_gradient_region = aggregate_regions(ref_gradient, labels)
    cand_gradient_region = aggregate_regions(cand_gradient, labels)
    protection = (
        cand_chroma_region > HARM_RATIO * ref_chroma_region + 1.0e-12
    ) | (
        cand_gradient_region > HARM_RATIO * ref_gradient_region + 1.0e-12
    )
    safe = demand & ~protection
    conflict = demand & protection
    return {
        "demand_region": demand,
        "protection_region": protection,
        "safe_region": safe,
        "conflict_region": conflict,
        "demand_mask": demand[labels],
        "protection_mask": protection[labels],
        "safe_mask": safe[labels],
        "conflict_mask": conflict[labels],
    }


def render_gain(reference: Any, candidate: Any, target: Any, mask: Any) -> float:
    import torch
    rendered = torch.where(mask[None], candidate, reference)
    return psnr_gain(
        float((reference - target).double().square().sum()),
        float((rendered - target).double().square().sum()),
    )


def jaccard(first: Any, second: Any) -> float:
    union = int((first | second).sum())
    return 1.0 if union == 0 else float((first & second).sum() / union)


def deterministic_shifts(name: str, height: int, width: int) -> list[tuple[int, int]]:
    import numpy as np
    seed = int.from_bytes(hashlib.sha256(f"{ROUTE_ID}:{name}".encode()).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    shifts: list[tuple[int, int]] = []
    while len(shifts) < SHUFFLE_REPLICATES:
        dy = int(generator.integers(1, height))
        dx = int(generator.integers(1, width))
        if (dy, dx) not in shifts:
            shifts.append((dy, dx))
    return shifts


def evaluate_image(
    name: str, hazy: Any, reference: Any, candidate: Any, target: Any,
    nominal_labels: Any, perturbed_labels: Any,
) -> dict[str, Any]:
    import torch

    nominal = label_masks(nominal_labels, reference, candidate, target)
    perturbed = label_masks(perturbed_labels, reference, candidate, target)
    fixed_labels = fixed_grid_labels(int(reference.shape[-2]), int(reference.shape[-1]), reference.device)
    fixed = label_masks(fixed_labels, reference, candidate, target)
    adaptive_gain = render_gain(reference, candidate, target, nominal["safe_mask"])
    fixed_gain = render_gain(reference, candidate, target, fixed["safe_mask"])
    whole_labels = torch.zeros_like(nominal_labels)
    whole = label_masks(whole_labels, reference, candidate, target)
    global_gain = render_gain(reference, candidate, target, whole["safe_mask"])
    shuffle_gains = [
        render_gain(reference, candidate, target, torch.roll(nominal["safe_mask"], (dy, dx), (0, 1)))
        for dy, dx in deterministic_shifts(name, int(reference.shape[-2]), int(reference.shape[-1]))
    ]
    safe_area = float(nominal["safe_mask"].float().mean())
    return {
        "name": name,
        "adaptive_gain": adaptive_gain,
        "fixed_gain": fixed_gain,
        "global_gain": global_gain,
        "shuffle_gain": float(sum(shuffle_gains) / len(shuffle_gains)),
        "demand_jaccard": jaccard(nominal["demand_mask"], perturbed["demand_mask"]),
        "protection_jaccard": jaccard(nominal["protection_mask"], perturbed["protection_mask"]),
        "safe_jaccard": jaccard(nominal["safe_mask"], perturbed["safe_mask"]),
        "safe_area_fraction": safe_area,
        "demand_area_fraction": float(nominal["demand_mask"].float().mean()),
        "protection_area_fraction": float(nominal["protection_mask"].float().mean()),
        "conflict_area_fraction": float(nominal["conflict_mask"].float().mean()),
        "nonzero_safe_coverage": safe_area >= 0.01,
        "has_conflict_region": bool(nominal["conflict_region"].any()),
        "adaptive_safe_regions": int(nominal["safe_region"].sum()),
        "adaptive_demand_regions": int(nominal["demand_region"].sum()),
        "adaptive_protection_regions": int(nominal["protection_region"].sum()),
    }


def cvar5(values: Any) -> float:
    import numpy as np
    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[: max(1, math.ceil(0.05 * len(array)))].mean())


def evaluate_rows(rows: list[dict[str, Any]], indices: Any) -> dict[str, float]:
    import numpy as np

    def values(key: str) -> Any:
        return np.asarray([row[key] for row in rows], dtype=np.float64)[indices]

    adaptive = values("adaptive_gain")
    fixed = values("fixed_gain")
    global_gain = values("global_gain")
    shuffle = values("shuffle_gain")
    return {
        "adaptive_gain": float(adaptive.mean()),
        "adaptive_minus_fixed": float((adaptive - fixed).mean()),
        "adaptive_minus_global": float((adaptive - global_gain).mean()),
        "adaptive_minus_shuffle": float((adaptive - shuffle).mean()),
        "adaptive_cvar5": cvar5(adaptive),
        "demand_jaccard": float(values("demand_jaccard").mean()),
        "protection_jaccard": float(values("protection_jaccard").mean()),
        "safe_jaccard": float(values("safe_jaccard").mean()),
    }


def interval(point: float, samples: list[float]) -> dict[str, float]:
    import numpy as np
    array = np.asarray(samples, dtype=np.float64)
    if not math.isfinite(point) or not np.isfinite(array).all():
        raise RuntimeError("non-finite bootstrap statistic")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(array, 0.025)),
        "ucb95": float(np.quantile(array, 0.975)),
    }


def bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    import numpy as np
    count = len(rows)
    point = evaluate_rows(rows, np.arange(count))
    samples = {key: [] for key in point}
    generator = np.random.default_rng(seed)
    for _draw in range(draws):
        value = evaluate_rows(rows, generator.integers(0, count, count))
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(point[key], samples[key]) for key in point}


def binomial_interval(events: int, total: int) -> dict[str, float]:
    from scipy.stats import beta
    point = events / total
    low = 0.0 if events == 0 else float(beta.ppf(0.025, events, total - events + 1))
    high = 1.0 if events == total else float(beta.ppf(0.975, events + 1, total - events))
    return {"events": events, "total": total, "point": point, "lcb95": low, "ucb95": high}


def contract(context_path: Path) -> None:
    import numpy as np
    import torch
    fake_optional_wdmamba_imports()
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    population = load_population_manifest(asset_path(context, "population_manifest", kind="file"))
    prior = json.loads(asset_path(context, "r16_closeout", kind="file").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    a0, wdmamba = build_models(context, device)
    bindings = 0
    for module in wdmamba.modules():
        if hasattr(module, "selective_scan"):
            module.selective_scan = selective_scan_ref
            bindings += 1
    fixture = torch.linspace(0.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    first = generate_action(fixture, a0, wdmamba)
    second = generate_action(fixture, a0, wdmamba)
    nominal_first = adaptive_region_labels(fixture, first["reference"], perturbed=False)
    nominal_second = adaptive_region_labels(fixture, first["reference"], perturbed=False)
    perturbed = adaptive_region_labels(fixture, first["reference"], perturbed=True)
    # Same-asymptotic 50-group/4000-draw algorithm probe without scientific data.
    rows = []
    for index in range(50):
        rows.append({
            "adaptive_gain": 0.03 + 0.001 * (index % 5),
            "fixed_gain": 0.02, "global_gain": 0.01, "shuffle_gain": 0.0,
            "demand_jaccard": 0.8, "protection_jaccard": 0.8, "safe_jaccard": 0.8,
        })
    synthetic_boot = bootstrap(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "population_manifest_exact": tuple(population["allowed_ids"]) == EXPECTED_IDS,
        "r16_authorization_exact": all(prior.get(key) == value for key, value in R16_EXPECTED.items()),
        "cpu_reference_scan_bound": bindings > 0,
        "production_action_deterministic": action_identity(first) == action_identity(second),
        "action_bound_exact": action_identity(first)["delta_abs_max"] <= ACTION_ALPHA + 1.0e-7,
        "adaptive_partition_deterministic": torch.equal(nominal_first, nominal_second),
        "adaptive_partition_has_64_regions": int(torch.unique(nominal_first).numel()) == REGION_COUNT,
        "perturbed_partition_has_64_regions": int(torch.unique(perturbed).numel()) == REGION_COUNT,
        "perturbation_is_nontrivial": not torch.equal(nominal_first, perturbed),
        "same_asymptotic_group_count": len(rows) == 50,
        "same_asymptotic_bootstrap": set(synthetic_boot) == {
            "adaptive_gain", "adaptive_minus_fixed", "adaptive_minus_global",
            "adaptive_minus_shuffle", "adaptive_cvar5", "demand_jaccard",
            "protection_jaccard", "safe_jaccard",
        },
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(context.phase_output_path / "cdprm_s1_engineering_contract.json", {
        "schema_version": 1, "checks": checks,
        "action_identity": action_identity(first),
        "nominal_region_identity": region_identity(nominal_first),
        "perturbed_region_identity": region_identity(perturbed),
        "synthetic_bootstrap_keys": sorted(synthetic_boot),
        "numpy_version": np.__version__,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_reference_equivalent",
            "device": "cpu",
            "fixture": {"batch": 1, "channels": 3, "height": 256, "width": 256},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started = time.perf_counter()
    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CDP-RM S1 requires authorized CUDA runtime")
    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    population_path = asset_path(context, "population_manifest", kind="file")
    population = load_population_manifest(population_path)
    prior_path = asset_path(context, "r16_closeout", kind="file")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if any(prior.get(key) != value for key, value in R16_EXPECTED.items()):
        raise RuntimeError("R16 terminal does not authorize CDP-RM reformulation")
    data_root = asset_path(context, "nhhaze_data", kind="directory")
    a0, wdmamba = build_models(context, device)

    def load_image(image_id: str, template: str, *, target: bool) -> Any:
        path = data_root / template.format(image_id=image_id)
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (1600, 1200):
                raise RuntimeError(f"NH-HAZE image identity/shape mismatch: {image_id}")
            array = np.asarray(image).copy()
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0)
        return tensor if target else tensor.unsqueeze(0).to(device)

    pass1: dict[str, Any] = {}
    for index, image_id in enumerate(EXPECTED_IDS, 1):
        hazy = load_image(image_id, population["hazy_filename_template"], target=False)
        values = generate_action(hazy, a0, wdmamba)
        nominal = adaptive_region_labels(hazy, values["reference"], perturbed=False)
        perturbed = adaptive_region_labels(hazy, values["reference"], perturbed=True)
        pass1[image_id] = {
            "action": action_identity(values),
            "nominal_region": region_identity(nominal),
            "perturbed_region": region_identity(perturbed),
        }
        del hazy, values, nominal, perturbed
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=index, stage="gt_free_action_region_seal")
    manifest_path = context.phase_output_path / "cdprm_s1_pass1_manifest_cloud_only.json"
    atomic_json(manifest_path, {
        "schema_version": 1, "route_id": ROUTE_ID, "image_ids": list(EXPECTED_IDS),
        "gt_opened": False, "protected_ids_opened_or_listed": False, "pass1": pass1,
    })
    pass1_sealed_at = time.time()

    replay_failures = []
    for index, image_id in enumerate(EXPECTED_IDS, 1):
        hazy = load_image(image_id, population["hazy_filename_template"], target=False)
        values = generate_action(hazy, a0, wdmamba)
        value = {
            "action": action_identity(values),
            "nominal_region": region_identity(adaptive_region_labels(hazy, values["reference"], perturbed=False)),
            "perturbed_region": region_identity(adaptive_region_labels(hazy, values["reference"], perturbed=True)),
        }
        if value != pass1[image_id]:
            replay_failures.append(image_id)
        del hazy, values
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=50 + index, stage="full_population_replay")
    replay_complete_at = time.time()
    if replay_failures or replay_complete_at <= pass1_sealed_at:
        raise RuntimeError(f"CDP-RM S1 full-population replay failed: {replay_failures}")

    rows: list[dict[str, Any]] = []
    scoring_replay_failures = []
    gt_first_open_at: float | None = None
    for index, image_id in enumerate(EXPECTED_IDS, 1):
        hazy = load_image(image_id, population["hazy_filename_template"], target=False)
        values = generate_action(hazy, a0, wdmamba)
        nominal = adaptive_region_labels(hazy, values["reference"], perturbed=False)
        perturbed = adaptive_region_labels(hazy, values["reference"], perturbed=True)
        identity = {
            "action": action_identity(values),
            "nominal_region": region_identity(nominal),
            "perturbed_region": region_identity(perturbed),
        }
        if identity != pass1[image_id]:
            scoring_replay_failures.append(image_id)
            raise RuntimeError(f"CDP-RM S1 scoring replay failed: {image_id}")
        if gt_first_open_at is None:
            gt_first_open_at = time.time()
        target = load_image(image_id, population["target_filename_template"], target=True)
        row = evaluate_image(
            image_id, hazy[0].cpu(), values["reference"][0].cpu(),
            values["candidate"][0].cpu(), target, nominal.cpu(), perturbed.cpu(),
        )
        rows.append(row)
        del hazy, values, nominal, perturbed, target
        torch.cuda.empty_cache()
        write_workload_progress(context, completed_units=100 + index, stage="replay_verified_scoring")
    if gt_first_open_at is None or gt_first_open_at <= replay_complete_at:
        raise RuntimeError("GT opened before complete action-region replay")

    boot = bootstrap(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    write_workload_progress(context, completed_units=4150, stage="bootstrap_complete")
    nonzero = binomial_interval(sum(row["nonzero_safe_coverage"] for row in rows), len(rows))
    conflict = binomial_interval(sum(row["has_conflict_region"] for row in rows), len(rows))
    severe = sum(row["adaptive_gain"] <= SEVERE_GAIN_DB for row in rows)
    hard = sum(row["adaptive_gain"] <= HARD_GAIN_DB for row in rows)
    half_width = 0.5 * (boot["adaptive_gain"]["ucb95"] - boot["adaptive_gain"]["lcb95"])
    gates = {
        "adaptive_materiality": boot["adaptive_gain"]["lcb95"] >= 0.020,
        "spatial_specificity": boot["adaptive_minus_shuffle"]["lcb95"] >= 0.005,
        "regional_specificity": boot["adaptive_minus_global"]["lcb95"] >= 0.005,
        "fixed_grid_noninferiority": boot["adaptive_minus_fixed"]["lcb95"] >= -0.005,
        "demand_stability": boot["demand_jaccard"]["lcb95"] >= 0.70,
        "protection_stability": boot["protection_jaccard"]["lcb95"] >= 0.70,
        "safe_mask_stability": boot["safe_jaccard"]["lcb95"] >= 0.70,
        "nonzero_coverage": nonzero["lcb95"] >= 0.50,
        "dual_target_necessity": conflict["lcb95"] >= 0.10,
        "tail_safety": boot["adaptive_cvar5"]["lcb95"] >= -0.005,
        "zero_severe_hard": severe == 0 and hard == 0,
        "precision": half_width <= 0.070,
    }
    decisive_fail = (
        boot["adaptive_gain"]["ucb95"] < 0.020
        or boot["adaptive_minus_shuffle"]["ucb95"] < 0.005
        or boot["adaptive_minus_global"]["ucb95"] < 0.005
        or boot["adaptive_minus_fixed"]["ucb95"] < -0.005
        or boot["demand_jaccard"]["ucb95"] < 0.70
        or boot["protection_jaccard"]["ucb95"] < 0.70
        or boot["safe_jaccard"]["ucb95"] < 0.70
        or nonzero["ucb95"] < 0.50
        or conflict["ucb95"] < 0.10
        or boot["adaptive_cvar5"]["ucb95"] < -0.005
        or severe > 0 or hard > 0
    )
    structural = {
        "r16_authorization_exact": all(prior.get(key) == value for key, value in R16_EXPECTED.items()),
        "population_complete": len(rows) == 50,
        "population_exact_01_50": [row["name"] for row in rows] == list(EXPECTED_IDS),
        "full_replay_before_gt": not replay_failures and gt_first_open_at > replay_complete_at,
        "scoring_replay_exact": not scoring_replay_failures,
        "protected_ids_51_55_untouched": True,
        "protected_roles_untouched": not any(context.protected_data_permissions.values()),
        "training_absent": True,
    }
    if not all(structural.values()):
        raise RuntimeError(f"CDP-RM S1 structural failure: {[key for key, value in structural.items() if not value]}")
    if all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "CDPRM_S1_REGION_TARGET_IDENTIFIABILITY_PASS"
        authorizes = "CDPRM_S2_NOOP_ARCHITECTURE_CONTRACT_ONLY"
    elif decisive_fail:
        state = "COMPLETED_GATE_FAIL"
        decision = "CDPRM_S1_REGION_OR_TARGET_FAIL_STOP"
        authorizes = "CDPRM_S1_REFORMULATION_REVIEW_ONLY"
    else:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "CDPRM_S1_EVIDENCE_INCONCLUSIVE_STOP"
        authorizes = "CDPRM_S1_EVIDENCE_COMPLETION_ONLY"

    descriptive = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "minimum": float(np.min([row[key] for row in rows])),
            "maximum": float(np.max([row[key] for row in rows])),
        }
        for key in ("safe_area_fraction", "demand_area_fraction", "protection_area_fraction", "conflict_area_fraction")
    }
    identity_access = {
        "schema_version": 1, "route_commit": context.route_commit,
        "r16_closeout_sha256": sha256_file(prior_path),
        "population_manifest_sha256": sha256_file(population_path),
        "pass1_manifest_sha256": sha256_file(manifest_path),
        "pass1_image_count": len(pass1), "pass2_replay_failures": replay_failures,
        "pass3_replay_failures": scoring_replay_failures,
        "gt_first_open_after_complete_replay": gt_first_open_at > replay_complete_at,
        "data_role": "development_screening_checkpoint_training_overlap",
        "eligible_as_external_validation": False, "ids_51_55_opened_or_listed": False,
        "confirmation_touched": False, "canary_touched": False,
        "locked_test_touched": False, "training_occurred": False,
        "checkpoint_selected": False, "threshold_selected": False,
        "sample_excluded": False,
    }
    action_region = {
        "schema_version": 1, "action_alpha": ACTION_ALPHA,
        "action_semantics": "A0_NH plus 0.25 times frozen WDMamba_NH minus A0_NH residual",
        "action_linf_bound": ACTION_ALPHA, "region_count": REGION_COUNT,
        "nominal_region_iterations": REGION_ITERATIONS,
        "nominal_region_resolution": [REGION_HEIGHT, REGION_WIDTH],
        "spatial_weight": SPATIAL_WEIGHT, "demand_gain_db": DEMAND_GAIN_DB,
        "protection_harm_ratio": HARM_RATIO, "descriptive_area_fractions": descriptive,
        "nonzero_safe_coverage": nonzero, "demand_protection_conflict": conflict,
    }
    gate_summary = {
        "schema_version": 1, "structural_checks": structural, "gates": gates,
        "decisive_fail": decisive_fail, "primary_ci_half_width_db": half_width,
        "severe_images": severe, "hard_images": hard,
        "state": state, "decision": decision, "authorizes": authorizes,
    }
    conclusion = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": context.run_id, "decision": decision, "authorizes": authorizes,
        "primary_result": (
            f"CDP-RM S1 ended {state}: adaptive safe-region gain "
            f"{boot['adaptive_gain']['point']:.6f} dB (LCB95 {boot['adaptive_gain']['lcb95']:.6f}, "
            f"UCB95 {boot['adaptive_gain']['ucb95']:.6f}), with interval half-width {half_width:.6f} dB."
        ),
        "gate_reasons": [
            f"Failed frozen gates: {[key for key, value in gates.items() if not value]}.",
            f"Adaptive-minus-fixed interval: {boot['adaptive_minus_fixed']}.",
            f"Adaptive-minus-global and shuffle intervals: {boot['adaptive_minus_global']} and {boot['adaptive_minus_shuffle']}.",
            f"Demand/protection/safe stability intervals: {boot['demand_jaccard']}, {boot['protection_jaccard']}, {boot['safe_jaccard']}.",
        ],
        "competing_explanation": (
            "The result is a GT-privileged target-definition screen for one proxy action; "
            "it cannot establish learnability or that content-adaptive regions alone cause downstream improvement."
        ),
        "limitations": [
            "All 50 images overlap checkpoint training and are not external validation.",
            "The frozen action is one 0.25 endpoint interpolation and does not qualify other actions.",
            "Demand and protection use paired GT and are unavailable at inference.",
            "PASS authorizes only a separate zero-initialized architecture contract, not training.",
        ],
    }
    atomic_json(context.phase_output_path / "cdprm_s1_identity_and_access.json", identity_access)
    atomic_json(context.phase_output_path / "cdprm_s1_action_region_contract.json", action_region)
    atomic_json(context.phase_output_path / "cdprm_s1_bootstrap_summary.json", {"schema_version": 1, **boot})
    atomic_json(context.phase_output_path / "cdprm_s1_stability_summary.json", {
        "schema_version": 1, "demand_jaccard": boot["demand_jaccard"],
        "protection_jaccard": boot["protection_jaccard"],
        "safe_jaccard": boot["safe_jaccard"], "nonzero_coverage": nonzero,
        "demand_protection_conflict": conflict,
    })
    atomic_json(context.phase_output_path / "cdprm_s1_gate_summary.json", gate_summary)
    atomic_json(context.phase_output_path / "cdprm_s1_resource_summary.json", {
        "schema_version": 1, "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2),
        "maximum_resident_set_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "independent_image_groups": len(rows), "model_forward_images": 150,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "training_occurred": False,
    })
    atomic_json(context.phase_output_path / "cdprm_s1_scientific_conclusion.json", conclusion)
    write_csv(context.phase_output_path / "cdprm_s1_per_image_rows_cloud_only.csv", rows)
    write_run_result(
        context, state=state, decision=decision, authorizes=authorizes,
        details={
            "adaptive_gain_db": boot["adaptive_gain"]["point"],
            "adaptive_gain_lcb95_db": boot["adaptive_gain"]["lcb95"],
            "adaptive_minus_fixed_lcb95_db": boot["adaptive_minus_fixed"]["lcb95"],
            "adaptive_minus_shuffle_lcb95_db": boot["adaptive_minus_shuffle"]["lcb95"],
            "safe_jaccard_lcb95": boot["safe_jaccard"]["lcb95"],
            "severe_images": severe, "hard_images": hard,
            "training_occurred": False, "external_validation_claimed": False,
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
