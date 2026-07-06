# Haze4K v2.37 Tail-Safe Same-Context WDMamba Eligibility and Preservation Audit

Date: 2026-07-06

Branch:
`codex/haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation`

Route identity: new tail-safe substrate audit. This is not a v2.36 generator or
bridge continuation.

Current bottleneck:
`NEXT_BOTTLENECK_TAIL_SAFE_TEACHER_ELIGIBILITY_AND_NOOP_LEARNABILITY`.

## Primary Question

Can the v2.36 full600 WDMamba same-context teacher distribution be converted
into a fold-stable, tail-safe teacher-positive plus A0-preservation substrate
before any bridge/generator work?

## Inherited Facts

- v2.35 full-image/full-image-slice same-context teacher and S4+S6 free-tensor
  projection passed on diagnostic canary.
- v2.36 unmasked alpha0.5 full600 substrate failed CVaR/severe/
  strong-reference gates.
- v2.36 P0B/P1/P2/P3/P4/canary80/locked test were not authorized.

## Not Allowed

- No bridge/generator training.
- No canary80.
- No locked test.
- No direct WDMamba-on-256-crop teacher.
- No 256 crop-input/full-image-slice target contract.
- No S5-only BILFCF continuation.
- No RGB output post-processing.
- No runtime teacher/expert.
- No relaxation of CVaR, severe, strong-reference, or fold gates.

## Stage Gates

P0 alpha/blend safety sweep tests alpha `0.125`, `0.25`, `0.375`, and `0.50`
from cached full-image A0 and WDMamba tensors. It must preserve the v2.36 P0
gate: image_count 600, cache coverage 100%, mean >= +0.30, hard >= +0.50,
easy >= -0.03, p05 >= -0.05, CVaR5 >= -0.10, severe_rate 0,
strong-reference regression <= 0.02, and fold pass 5/5.

P1 tail failure atlas is authorized only if no unmasked alpha passes. It
summarizes negative, severe, and strong-reference regressions with target-only
image/A0 proxies and offline teacher-minus-A0 diagnostic energy.

P2 teacher-positive plus A0-preservation mask sweep is authorized after P1. A
masked substrate may pass only if it preserves all negative/severe samples,
keeps eligible_count >= 300/600, hard_eligible_rate >= 0.80, masked mean >=
+1.00 dB, masked hard >= +2.00 dB, p05/CVaR5 >= -0.01 dB, severe 0,
strong-reference regression 0, and fold pass 5/5.

P3 fold-stable mask selection uses 4/5 folds to select a predeclared rule and
evaluates only on the heldout fold. It must pass 5/5 heldout folds.

P4 target-only no-op/unsafe separability uses only input/A0/full-context proxy
features at runtime. It cannot use GT, teacher output, teacher_delta, or locked
test as runtime features.

P5 masked same-contract free-tensor projection is not authorized until P3 passes
and P4 demonstrates target-only unsafe/no-op separability.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/`
