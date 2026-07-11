# Haze4K CHD-RM v3j Bounded Safe-Correction Audit

Date: 2026-07-11
Branch: `codex/haze4k-v5-v3j-bounded-safe-correction-audit`
Evidence:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/`
Status: closed, no promotion

## Purpose

v3i closed the current post-hoc FAM2 router/distillation route: the privileged
open-value oracle is strong and compressible, but deployable signals failed OOF
replay. v3j changes the question from selecting an old fixed correction to
directly auditing a bounded safe output residual correction action space.

## Route Identity

This is a new diagnostic route. It is not a v3d continuation, not v3f-B ranker
training, not a larger router, and not a model-structure canary.

## Locked Test Policy

Locked Haze4K test remains sealed. v3j-A and v3j-B use train-derived splits
only.

## Stage Plan

v3j-A: no-training bounded action-space audit using primary teacher
`CC_MIN4_FROM_OPEN_TOP_0.5`, ceiling teacher `ALPHA_SECANT_Q3`, p99 residual
bounds from `v3j_controller_calib`, and replay on `v3j_route_confirm`.

v3j-B: direct residual OOF diagnostic only if v3j-A passes.

## Decision

v3j-A passed; v3j-B failed. The bounded output-residual actuator is viable
under the privileged teacher, but deployable tiny direct residual heads create
unsafe tails even with positive mean gains.

Final decision:
`V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER`.

No v3j canary, no no-op architecture equivalence, and no internal-router
continuation are authorized. A future route must add new tail-risk information
before any model-promotion experiment.
