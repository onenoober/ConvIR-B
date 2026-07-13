"""Frozen numerical contract for the v4a A0P paired shadow audit.

The module is intentionally free of dataset/model loading. The cloud runner is
responsible for supplying exact A0R states, rendered tensors, and trace hashes.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


METHODS = (
    "historical_sequential_gradient",
    "exact_gradient_intersection",
    "actual_proposal_projection_with_backtracking",
)
WINDOWS = ("fixed4", "shuffled16", "prestratified32")
CONSTRAINT_ORDER = ("anchor", "harm", "margin", "cvar")
ACTIVE_SET_TOL = 1e-10
PINV_RCOND = 1e-12
GRAD_CLIP_NORM = 0.1
BOOTSTRAP_SEED = 3407
BOOTSTRAP_REPLICATES = 1000
UTILITY_MARGIN_DB = -0.005


@dataclass(frozen=True)
class ProjectionResult:
    vector: torch.Tensor
    active_set: tuple[int, ...]
    objective: float
    primal_residual: float
    dual_residual: float
    valid: bool


def sha256_rank(prefix: str, *parts: str) -> str:
    return hashlib.sha256("|".join((prefix, *parts)).encode("utf-8")).hexdigest()


def clip_vector(vector: torch.Tensor, limit: float = GRAD_CLIP_NORM) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite vector norm")
    if float(norm.item()) <= limit:
        return vector.clone()
    return vector * (limit / norm)


def normalize_rows(rows: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    normalized: list[torch.Tensor] = []
    for row in rows:
        flat = row.detach().to(dtype=torch.float64).reshape(-1)
        norm = torch.linalg.vector_norm(flat)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite constraint norm")
        if float(norm.item()) > 1e-30:
            normalized.append(flat / norm)
    return normalized


def project_to_nonnegative_halfspaces(target: torch.Tensor, constraints: Iterable[torch.Tensor]) -> ProjectionResult:
    """Project target onto c_i^T x >= 0 using deterministic active sets."""
    target64 = target.detach().to(dtype=torch.float64).reshape(-1)
    rows = normalize_rows(constraints)
    if not torch.isfinite(target64).all():
        raise FloatingPointError("non-finite target")
    if not rows:
        return ProjectionResult(target64, (), 0.0, 0.0, 0.0, True)
    matrix = torch.stack(rows, dim=0)
    best: ProjectionResult | None = None
    for size in range(len(rows) + 1):
        for active in itertools.combinations(range(len(rows)), size):
            if active:
                active_matrix = matrix[list(active)]
                gram = active_matrix @ active_matrix.T
                lagrange = torch.linalg.pinv(gram, rcond=PINV_RCOND) @ (-(active_matrix @ target64))
                candidate = target64 + active_matrix.T @ lagrange
                dual_residual = float(torch.clamp(-lagrange, min=0.0).max().item())
                equality_residual = float(torch.abs(active_matrix @ candidate).max().item())
            else:
                candidate = target64.clone()
                dual_residual = 0.0
                equality_residual = 0.0
            products = matrix @ candidate
            primal_residual = float(torch.clamp(-products, min=0.0).max().item())
            if not torch.isfinite(candidate).all():
                continue
            if primal_residual > ACTIVE_SET_TOL or dual_residual > ACTIVE_SET_TOL or equality_residual > ACTIVE_SET_TOL:
                continue
            objective = float((0.5 * torch.sum((candidate - target64).square())).item())
            result = ProjectionResult(candidate, tuple(active), objective, primal_residual, dual_residual, True)
            if best is None or (result.objective, result.active_set) < (best.objective, best.active_set):
                best = result
    if best is None:
        return ProjectionResult(target64, (), math.inf, math.inf, math.inf, False)
    return best


def exact_gradient_intersection(render_gradient: torch.Tensor, constraint_gradients: Iterable[torch.Tensor]) -> ProjectionResult:
    return project_to_nonnegative_halfspaces(render_gradient, constraint_gradients)


def actual_proposal_projection(proposal: torch.Tensor, constraint_gradients: Iterable[torch.Tensor]) -> ProjectionResult:
    return project_to_nonnegative_halfspaces(proposal, (-gradient for gradient in constraint_gradients))


def shuffled16(names: Iterable[str], state_sha256: str) -> list[str]:
    ranked = sorted((str(name) for name in names), key=lambda name: sha256_rank("shuffled16", state_sha256, name))
    if len(ranked) != 128:
        raise ValueError("shuffled16 requires exactly update128 names")
    return ranked[:16]


def prestratified32(names: Iterable[str], inherited_ratios: dict[str, float], state_sha256: str) -> list[str]:
    ordered = sorted((str(name) for name in names), key=lambda name: (inherited_ratios[name], name))
    if len(ordered) != 128 or set(ordered) != set(inherited_ratios):
        raise ValueError("prestratified32 requires one inherited ratio for every update128 name")
    selected: list[str] = []
    for index in range(8):
        stratum = ordered[index * 16:(index + 1) * 16]
        selected.extend(sorted(stratum, key=lambda name: sha256_rank("prestratified32", state_sha256, str(index), name))[:4])
    if len(selected) != 32 or len(set(selected)) != 32:
        raise RuntimeError("prestratified32 cardinality failure")
    return selected


def numerical_tolerance(pre: float, post: float) -> float:
    return 2.0 * (1e-12 + 1e-12 * max(abs(pre), abs(post)))


def bootstrap_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return the frozen Q0.05/Q0.95 indexed bounds for 1,000 values."""
    if values.shape != (BOOTSTRAP_REPLICATES,) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be exactly 1,000 finite values")
    ordered = np.sort(values.astype(np.float64, copy=False))
    return float(ordered[49]), float(ordered[949])


def bootstrap_indices(state_count: int, image_count: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if state_count != 256 or image_count != 128:
        raise ValueError("A0P bootstrap requires exactly 256 states and heldout128 images")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for _ in range(BOOTSTRAP_REPLICATES):
        yield (
            generator.integers(0, state_count, size=state_count, endpoint=False),
            generator.integers(0, image_count, size=image_count, endpoint=False),
        )


def classify_a0p(*, complete: bool, structural_valid: bool, proposal_positive_windows: list[str], exact_only_positive: bool, interaction_reversal: bool) -> tuple[str, str, str]:
    if not complete or not structural_valid:
        return ("COMPLETED_GATE_FAIL", "A0P_INCONCLUSIVE_AMENDMENT_REQUIRED", "R3_REVIEW_ONLY")
    if proposal_positive_windows:
        return ("COMPLETED_GATE_PASS", "A0P_ACTUAL_PROPOSAL_POSITIVE_R3_HANDOFF", "R3_REVIEW_ONLY")
    if exact_only_positive or interaction_reversal:
        return ("COMPLETED_GATE_FAIL", "A0P_INCONCLUSIVE_AMENDMENT_REQUIRED", "R3_REVIEW_ONLY")
    return ("COMPLETED_GATE_PASS", "A0P_NO_LOCAL_CORRECTION_R3_HANDOFF", "R3_REVIEW_ONLY")
