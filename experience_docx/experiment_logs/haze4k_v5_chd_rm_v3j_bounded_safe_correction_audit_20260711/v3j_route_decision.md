# v3j Route Decision

Route: `haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711`
Branch: `codex/haze4k-v5-v3j-bounded-safe-correction-audit`
Date: 2026-07-11
Status: `CLOSED_NO_PROMOTION`

## Route Identity

This is a new diagnostic audit after v3i. It is not a v3d continuation, not
v3f-B ranker training, not a larger FAM2 router, and not a model-structure
canary.

## Source Of Truth

- GitHub `main`: v3i closeout and FAM family summary.
- Parent runnable branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
  at `8517614`.
- Cloud workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3j-bounded-safe-correction-audit`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Question

Can a privileged GT-derived safe teacher supervise a bounded output residual
correction action space, instead of supervising a post-hoc switch for the old
FAM2 correction?

## Stage Gates

v3j-A runs first and performs no training. It calibrates per-channel residual
bounds on fresh train-derived `v3j_controller_calib` and evaluates bounded
teacher projections on fresh train-derived `v3j_route_confirm`.

v3j-B is authorized only if a primary-teacher bounded projection beats hard D7c
with positive bootstrap lower bound, non-worse p10, and non-worse
severe-regression count.

v3j-B must pass on both OOF and route-confirm for the same direct residual head.
Mean improvement alone is insufficient; p10 and severe-regression gates are
blocking safety gates.

## Forbidden Flow

No locked Haze4K test, no v3d continuation, no v3f-B ranker, no current-signal
FAM2 router, no backbone/adapter/FAM producer training, no canary expansion
before v3j-B pass, and no changing teacher/bound/gate after seeing
route-confirm metrics.

## Closeout

v3j-A passed and changed the diagnosis: the bounded output-residual action
space itself is not the bottleneck. On `v3j_route_confirm`, the primary p99
D7c-bounded projection reached mean `+0.229641`, p10 `+0.002454`, worst
`-0.018076`, severe `0`, with bootstrap CI95 low vs hard `+0.199968`.

v3j-B failed the deployable direct-residual gate. Both tiny heads improved mean
PSNR but introduced unsafe tails:

- `CONFIRM_DIRECT_LINEAR`: mean `+0.057145`, p10 `-0.392219`, severe `121`.
- `CONFIRM_DIRECT_CONTEXT`: mean `+0.099199`, p10 `-0.444083`, severe `121`.
- hard D7c reference: mean `+0.008634`, p10 `-0.131343`, severe `23`.

Tail audit confirmed the failure is direct-only in most severe cases:
route-confirm direct-only severe counts are `108` for linear and `111` for
context. The median direct-minus-hard delta is negative for both heads despite
positive mean bootstrap deltas.

Decision:
`V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER`.

No v3j canary, no no-op architecture equivalence, and no internal-router
continuation are authorized. The next viable route must introduce new
tail-risk information rather than reuse the same deployable feature family.
