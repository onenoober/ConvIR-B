#!/usr/bin/env python3
"""Protected-data-free S0 qualification for frozen CONVIR_ONLY_RDPCM_V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from route_engineering_fixture import (
    assert_finite_tensors,
    assert_noop,
    assert_trainable_scope,
)
from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


FIXTURE = {"batch": 1, "channels": 3, "height": 64, "width": 64}
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "0ba530b66c65a14695223215931ca6e4463ee3cd5d65ae64e0e6cb764d07f4e6"
REVIEW_SHA256 = "3d3346823e083290c958e0879ee0c8f139ef6a465ee8624919c001dd90fb90c0"
RDPCM_PREFIX = "RDPCM"
RDPCM_PARAMETER_COUNT = 41699
RDPCM_PARAMETER_LIMIT = 50000
MEMORY_INCREMENT_LIMIT_BYTES = 512 * 1024 * 1024
NOOP_ATOL = 0.0
NOOP_RTOL = 0.0
CONTROLLED_GATE_VALUE = 0.1
SEED = 20260721


def _load_checkpoint(torch, path: Path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "model" in value:
        value = value["model"]
    if not isinstance(value, dict) or not value:
        raise RuntimeError("official checkpoint is not a non-empty state dict")
    return value


def _strict_partial_load(base, candidate, checkpoint):
    base.load_state_dict(checkpoint, strict=True)
    candidate_state = candidate.state_dict()
    checkpoint_keys = set(checkpoint)
    candidate_keys = set(candidate_state)
    missing = sorted(candidate_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - candidate_keys)
    shape_mismatch = sorted(
        key for key in checkpoint_keys & candidate_keys
        if tuple(checkpoint[key].shape) != tuple(candidate_state[key].shape)
    )
    if unexpected or shape_mismatch or not missing:
        raise RuntimeError(
            f"strict partial-load identity failed: unexpected={unexpected[:8]} "
            f"shape_mismatch={shape_mismatch[:8]} missing_count={len(missing)}"
        )
    if any(not key.startswith(RDPCM_PREFIX + ".") for key in missing):
        raise RuntimeError(f"missing keys escape {RDPCM_PREFIX} prefix: {missing[:8]}")
    accepted = dict(candidate_state)
    for key, tensor in checkpoint.items():
        accepted[key] = tensor
    candidate.load_state_dict(accepted, strict=True)
    return {
        "checkpoint_key_count": len(checkpoint_keys),
        "loaded_official_key_count": len(checkpoint_keys),
        "allowed_missing_key_count": len(missing),
        "allowed_missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "final_strict_load": True,
    }


def _freeze_scope(candidate):
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(name.startswith(RDPCM_PREFIX + "."))
    return assert_trainable_scope(
        candidate, allowed_prefixes=(RDPCM_PREFIX,), required_prefixes=(RDPCM_PREFIX,),
    )


def _group_gradient_evidence(candidate):
    groups = {
        "shared": "RDPCM.shared",
        "demand": "RDPCM.demand",
        "protection": "RDPCM.protection",
        "residual": "RDPCM.residual",
        "output_gate": "RDPCM.output_gate",
    }
    result = {}
    for group, prefix in groups.items():
        maxima = []
        for name, parameter in candidate.named_parameters():
            if name == prefix or name.startswith(prefix + "."):
                if parameter.grad is None:
                    raise RuntimeError(f"missing controlled-activation gradient: {name}")
                gradient = parameter.grad.detach()
                if not bool(gradient.isfinite().all().item()):
                    raise RuntimeError(f"non-finite controlled-activation gradient: {name}")
                maxima.append(float(gradient.abs().max().item()))
        if not maxima or max(maxima) <= 0.0:
            raise RuntimeError(f"controlled activation did not reach RDPCM group: {group}")
        result[group] = {
            "tensor_count": len(maxima),
            "max_abs": max(maxima),
            "min_of_max_abs": min(maxima),
        }
    return result


def _forward_outputs(model, value):
    outputs = model(value)
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise RuntimeError("official three-scale output contract changed")
    return outputs


def _model_contract(context):
    import torch

    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S0 requires the frozen CUDA synthetic contract")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("S0 protected-data permissions must remain false")
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    anchor = context.assets["official_anchor"]
    source = context.assets["rdpcm_model_source"]
    review = context.assets["authoritative_review"]
    if anchor.commit != ANCHOR_COMMIT or source.sha256 != MODEL_SOURCE_SHA256 \
            or review.sha256 != REVIEW_SHA256 \
            or context.assets["official_checkpoint"].sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("identity-bound capability assets do not match frozen identities")

    sys.path.insert(0, str(context.remote_repo / "Dehazing" / "ITS"))
    from models.ConvIR import build_net

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    state = _load_checkpoint(torch, checkpoint)

    torch.cuda.empty_cache()
    base = build_net("base", "Haze4K", rdpcm_mode="off").to(device).eval()
    base.load_state_dict(state, strict=True)
    synthetic = torch.linspace(
        0.0, 1.0, steps=FIXTURE["channels"] * FIXTURE["height"] * FIXTURE["width"],
        dtype=torch.float32, device=device,
    ).reshape(FIXTURE["batch"], FIXTURE["channels"], FIXTURE["height"], FIXTURE["width"])
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        base_outputs = [item.detach().cpu() for item in _forward_outputs(base, synthetic)]
    torch.cuda.synchronize(device)
    base_peak = int(torch.cuda.max_memory_allocated(device))
    del base
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    candidate = build_net("base", "Haze4K", rdpcm_mode="v1").to(device).eval()
    load_evidence = _strict_partial_load(
        build_net("base", "Haze4K", rdpcm_mode="off"), candidate, state,
    )
    scope = _freeze_scope(candidate)
    if scope["trainable_parameter_count"] != RDPCM_PARAMETER_COUNT \
            or scope["trainable_parameter_count"] > RDPCM_PARAMETER_LIMIT:
        raise RuntimeError(f"RDPCM parameter contract failed: {scope}")

    torch.cuda.reset_peak_memory_stats(device)
    candidate.zero_grad(set_to_none=True)
    candidate_outputs = _forward_outputs(candidate, synthetic)
    torch.cuda.synchronize(device)
    candidate_peak = int(torch.cuda.max_memory_allocated(device))
    memory_increment = candidate_peak - base_peak
    if memory_increment > MEMORY_INCREMENT_LIMIT_BYTES:
        raise RuntimeError(
            f"RDPCM memory increment {memory_increment} exceeds {MEMORY_INCREMENT_LIMIT_BYTES}"
        )
    noop = [
        assert_noop(reference, observed.detach().cpu(), atol=NOOP_ATOL, rtol=NOOP_RTOL)
        for reference, observed in zip(base_outputs, candidate_outputs)
    ]
    finite = assert_finite_tensors(
        (f"output_{index}", value) for index, value in enumerate(candidate_outputs)
    )

    candidate.RDPCM.enabled = False
    with torch.no_grad():
        disabled = [item.detach().cpu() for item in _forward_outputs(candidate, synthetic)]
    disabled_noop = [
        assert_noop(reference, observed, atol=NOOP_ATOL, rtol=NOOP_RTOL)
        for reference, observed in zip(base_outputs, disabled)
    ]
    candidate.RDPCM.enabled = True

    candidate.zero_grad(set_to_none=True)
    initial_loss = sum(item.square().mean() for item in _forward_outputs(candidate, synthetic))
    initial_loss.backward()
    gate_gradient = candidate.RDPCM.output_gate.grad
    if gate_gradient is None or not bool(torch.isfinite(gate_gradient).all().item()) \
            or float(gate_gradient.abs().max().item()) <= 0.0:
        raise RuntimeError("exact-no-op initialization did not reach the output gate")
    initial_gradient = {
        "required_parameter": "RDPCM.output_gate",
        "max_abs": float(gate_gradient.abs().max().item()),
        "all_other_rdpcm_gradients_required_nonzero": False,
    }

    with torch.no_grad():
        candidate.RDPCM.output_gate.fill_(CONTROLLED_GATE_VALUE)
    candidate.zero_grad(set_to_none=True)
    controlled_loss = sum(item.square().mean() for item in _forward_outputs(candidate, synthetic))
    controlled_loss.backward()
    controlled_gradients = _group_gradient_evidence(candidate)
    with torch.no_grad():
        candidate.RDPCM.output_gate.zero_()
    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        restored = [item.detach().cpu() for item in _forward_outputs(candidate, synthetic)]
    restored_noop = [
        assert_noop(reference, observed, atol=NOOP_ATOL, rtol=NOOP_RTOL)
        for reference, observed in zip(base_outputs, restored)
    ]

    return {
        "module_definition_id": "CONVIR_ONLY_RDPCM_V1",
        "insertion_location": "Decoder[0]_output_before_feat_extract[3]_and_skip_fusion",
        "channel": 128,
        "hidden_width": 32,
        "branch_depth_convolutions": {"shared": 1, "demand": 1, "protection": 1, "residual": 1},
        "bounded_function": "0.25*tanh(output_gate)*sigmoid(demand)*sigmoid(-protection)*tanh(residual)",
        "modulation_bound": 0.25,
        "initial_output_gate": 0.0,
        "controlled_activation_gate": CONTROLLED_GATE_VALUE,
        "optimizer_steps": 0,
        "scientific_data_touched": False,
        "load": load_evidence,
        "scope": scope,
        "finite": finite,
        "initial_noop": noop,
        "disabled_noop": disabled_noop,
        "restored_noop": restored_noop,
        "initial_gradient": initial_gradient,
        "controlled_activation_gradients": controlled_gradients,
        "base_peak_memory_bytes": base_peak,
        "candidate_peak_memory_bytes": candidate_peak,
        "memory_increment_bytes": memory_increment,
        "memory_increment_limit_bytes": MEMORY_INCREMENT_LIMIT_BYTES,
    }


def contract(context_path):
    context = load_context(Path(context_path), "contract")
    prepare_phase_output(context)
    details = _model_contract(context)
    atomic_json(output_file(context, "rdpcm_s0_contract_details.json"), details)
    checks = {
        "asset_identity": True,
        "strict_partial_load": True,
        "initial_noop": True,
        "disabled_noop": True,
        "finite_forward": True,
        "trainable_scope": True,
        "initial_reachable_gradient": True,
        "controlled_full_gradient": True,
        "parameter_limit": True,
        "memory_limit": True,
        "protected_data_untouched": True,
    }
    write_contract_result(
        context, checks=checks, engineering={
            "mode": "gpu_synthetic_no_data",
            "device": "cuda",
            "fixture": FIXTURE,
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path):
    context = load_context(Path(context_path), "run")
    prepare_phase_output(context)
    contract_details = context.output_path / "contract" / "rdpcm_s0_contract_details.json"
    details = json.loads(contract_details.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "decision": "RDPCM_S0_ARCHITECTURE_QUALIFIED",
        "authorizes": "ONE_SMALL_HAZE4K_DEVELOPMENT_MECHANISM_DISCRIMINATION_CONTRACT_ONLY",
        "engineering_only": True,
        "training_occurred": False,
        "scientific_data_touched": False,
        "confirmation_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "contract_details": details,
    }
    atomic_json(output_file(context, "rdpcm_s0_summary.json"), summary)
    write_workload_progress(context, completed_units=1, stage="s0_qualified")
    write_run_result(
        context,
        state="COMPLETED_GATE_PASS",
        decision="RDPCM_S0_ARCHITECTURE_QUALIFIED",
        authorizes="ONE_SMALL_HAZE4K_DEVELOPMENT_MECHANISM_DISCRIMINATION_CONTRACT_ONLY",
        details={
            "engineering_only": True,
            "training_occurred": False,
            "scientific_data_touched": False,
            "module_parameter_count": details["scope"]["trainable_parameter_count"],
            "memory_increment_bytes": details["memory_increment_bytes"],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
