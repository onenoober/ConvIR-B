# DTA Depth-Transmission Adapter Family Summary

Date: 2026-06-11

Status: positive diagnostic family, not promotion-ready; DTA-v3 Phase A R0 gate failed.

## Scope

This family covers ConvIR-B Haze4K depth/transmission adapter routes, including
DTA-v2 CalGate diagnostics and the DTA-v3 DAPC fine-tune route. Use the route
cards and evidence roots for exact command and artifact details.

## Current Verdict

DTA-v2 CalGate showed a real train-derived OOF signal, but the mechanism was not
cleanly depth-attributed: zero/shuffle controls retained most of the gain, wrong
orientation remained competitive, SSIM was slightly negative, and tail
regressions were too high. It must not be promoted or evaluated on locked Haze4K
test.

DTA-v3 DAPC correctly moved to an official-anchor fine-tune route on
`convir-4090`, with deterministic eval shuffle and explicit R0/depth branch
separation. Stage 0 preflight passed, but Phase A `dta_r0_only` OOF20 fold0
failed the R0 safety baseline: mean dPSNR `-0.012119`, hard bottom-25
`-0.069025`, easy top-25 `+0.044643`, dSSIM `-0.00001218`, positive ratio
`0.45`, strong regressions `48/150`, and worst regressions `70/600`.

Decision: `COMPLETED_GATE_FAIL_PHASE_A_R0_NO_PHASE_B`. Conservative R0 scout variants also failed; do not launch Phase B
from the current R0 checkpoint.

## Route Table

| Route | Evidence | Decision |
| --- | --- | --- |
| DTA-v2 CalGate | Multi-seed OOF showed `invert` about `+0.0887 dB`, but zero/shuffle retained most of the improvement and tail/SSIM did not pass. | Positive diagnostic only; no locked test; use as motivation for attribution controls. |
| DTA-v3 DAPC fine-tune | `convir-4090` preflight passed; deterministic eval shuffle audit passed; Phase A R0 OOF20 fold0 failed gate. | Stop current Phase A, redesign R0 before Phase B. |

## Reopen Conditions

Reopen this family only with a predeclared DTA-v3 Phase A variant that first
produces a safe generic R0 baseline, or with an alternate depth-active route that
proves true-depth surplus over zero, deterministic shuffle, and wrong-orientation
controls while keeping SSIM and tail regressions no worse than the zero/R0
baseline.

Do not use locked Haze4K test for checkpoint, gate, depth mode, or loss
selection.
