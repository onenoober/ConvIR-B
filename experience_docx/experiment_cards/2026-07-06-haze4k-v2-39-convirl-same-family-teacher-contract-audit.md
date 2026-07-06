# Haze4K v2.39 ConvIR-L Same-Family Teacher Contract Audit

Date: 2026-07-06

Branch:
`codex/haze4k-v2-39-convirl-same-family-teacher-contract-audit`

Route identity: same-family teacher contract audit after v2.38 micro-alpha and
v2.38B richer target-only separability both failed gates.

## Primary Question

Can ConvIR-L, a same-family teacher, provide a fold-stable, tail-safe full-image
same-context teacher alpha that avoids the WDMamba easy/strong-reference tail
failure?

## Not Allowed

- No bridge/generator training in this route.
- No canary80.
- No locked test.
- No direct WDMamba-on-256-crop teacher.
- No 256 crop-input/full-image-slice target.
- No S5-only continuation.
- No relaxation of strong-reference, severe, or fold gates.

## Metric Contract

P0 runs ConvIR-L full-image inference on the same 600 train-derived images used
by v2.37/v2.38. It compares full-image ConvIR-L and A0 in the same context and
sweeps:

`0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0`.

P0 passes only if at least one alpha satisfies the strict no-selector tail gate:
image_count 600, cache coverage 100%, mean >= +0.30, hard >= +0.50, easy >=
+0.05, p05 >= 0, CVaR5 >= -0.01, worst >= -0.05, severe 0, strong-reference
regressions 0, and fold pass 5/5.

P1 free-tensor projection is blocked unless P0 passes.

## Result

Decision: `P0_FAIL_CONVIRL_NO_SAFE_TEACHER_ALPHA`.

P0 completed the ConvIR-L full-image same-context alpha sweep on 600
train-derived images. No alpha passed the strict no-selector teacher-contract
gate. P1 free-tensor projection, bridge/generator training, canary80, and
locked test are not authorized.
