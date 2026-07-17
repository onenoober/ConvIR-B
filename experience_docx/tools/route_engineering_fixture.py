#!/usr/bin/env python3
"""Reusable assertions for protected-data-free real-structure fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


class FixtureError(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise FixtureError("PyTorch is required for model fixture assertions") from exc
    return torch


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def assert_trainable_scope(model: Any, *, allowed_prefixes: Iterable[str],
                           required_prefixes: Iterable[str] = ()) -> dict[str, Any]:
    allowed = tuple(dict.fromkeys(allowed_prefixes))
    required = tuple(dict.fromkeys(required_prefixes))
    if not allowed or any(not isinstance(item, str) or not item for item in allowed + required):
        raise FixtureError("trainable prefixes must be non-empty strings")
    named = list(model.named_parameters())
    if not named:
        raise FixtureError("model exposes no named parameters")
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable:
        raise FixtureError("model exposes no trainable parameters")
    unexpected = [name for name, _ in trainable if not _matches(name, allowed)]
    missing = [prefix for prefix in required if not any(
        name == prefix or name.startswith(prefix + ".") for name, _ in trainable
    )]
    if unexpected:
        raise FixtureError(f"unexpected trainable parameters: {unexpected[:16]}")
    if missing:
        raise FixtureError(f"required trainable prefixes are absent: {missing}")
    names = sorted(name for name, _ in trainable)
    digest = hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "parameter_count": sum(parameter.numel() for _, parameter in named),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensor_count": len(trainable),
        "trainable_name_sha256": digest,
    }


def assert_finite_tensors(named_tensors: Iterable[tuple[str, Any]]) -> dict[str, int]:
    torch = _torch()
    count = 0
    elements = 0
    for name, tensor in named_tensors:
        if not isinstance(name, str) or not name or not torch.is_tensor(tensor):
            raise FixtureError("finite checks require named tensors")
        value = tensor.detach()
        if not bool(torch.isfinite(value).all().item()):
            raise FixtureError(f"non-finite tensor: {name}")
        count += 1
        elements += value.numel()
    if count < 1:
        raise FixtureError("finite check received no tensors")
    return {"finite_tensor_count": count, "finite_element_count": elements}


def assert_noop(reference: Any, candidate: Any, *, atol: float = 0.0,
                rtol: float = 0.0) -> dict[str, float]:
    torch = _torch()
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        raise FixtureError("no-op check requires tensors")
    if reference.shape != candidate.shape:
        raise FixtureError("no-op tensors have different shapes")
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise FixtureError("no-op tolerances must be finite and non-negative")
    reference = reference.detach()
    candidate = candidate.detach()
    assert_finite_tensors((("reference", reference), ("candidate", candidate)))
    difference = (candidate - reference).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    try:
        torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise FixtureError(f"no-op equivalence failed: max_abs={max_abs}") from exc
    return {"noop_max_abs": max_abs, "noop_atol": atol, "noop_rtol": rtol}


def assert_nonzero_gradients(model: Any, *, required_prefixes: Iterable[str]) \
        -> dict[str, float | int]:
    torch = _torch()
    prefixes = tuple(dict.fromkeys(required_prefixes))
    if not prefixes:
        raise FixtureError("gradient check requires at least one prefix")
    matched = []
    missing_grad = []
    maxima = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not _matches(name, prefixes):
            continue
        matched.append(name)
        gradient = parameter.grad
        if gradient is None:
            missing_grad.append(name)
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise FixtureError(f"non-finite gradient: {name}")
        maxima.append(float(gradient.detach().abs().max().item()))
    missing_prefixes = [prefix for prefix in prefixes if not any(
        name == prefix or name.startswith(prefix + ".") for name in matched
    )]
    if missing_prefixes or missing_grad or not maxima or max(maxima) <= 0:
        raise FixtureError(
            "required gradients are missing or zero: "
            f"prefixes={missing_prefixes} missing={missing_grad[:16]}"
        )
    return {
        "gradient_tensor_count": len(maxima),
        "gradient_max_abs": max(maxima),
        "gradient_min_of_max_abs": min(maxima),
    }


def assert_loss_decreased(before: float, after: float, *,
                          minimum_relative_decrease: float = 0.0) -> dict[str, float]:
    before, after, minimum_relative_decrease = (
        float(before), float(after), float(minimum_relative_decrease),
    )
    if not all(math.isfinite(item) for item in (before, after, minimum_relative_decrease)) \
            or before <= 0 or not 0 <= minimum_relative_decrease < 1:
        raise FixtureError("microfit losses or threshold are invalid")
    relative = (before - after) / before
    if relative < minimum_relative_decrease:
        raise FixtureError(
            f"microfit loss did not decrease enough: {relative} <= {minimum_relative_decrease}"
        )
    return {
        "microfit_loss_before": float(before),
        "microfit_loss_after": float(after),
        "microfit_relative_decrease": relative,
    }
