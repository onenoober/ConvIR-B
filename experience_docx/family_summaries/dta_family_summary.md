# DTA Depth-Transmission Adapter Family Summary

Date: 2026-06-11

Status: positive diagnostic family, not promotion-ready; DTA-v3 depthDirect invert is mechanism-positive but tail/SSIM fail.

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

Decision: `COMPLETED_MECHANISM_POSITIVE_NO_PROMOTION_DEPTHDIRECT_INVERT_ONLY`. Conservative R0 scout variants failed, so do not launch Phase B
from the current R0 checkpoint. The zero-R0 depthDirect scout shows the first
clean mean true-vs-zero surplus for train=`invert` (`+0.032286 dB`), but it is
not promotion-ready because mean vs A0 is only `+0.013905 dB`, hard is
`-0.005602 dB`, dSSIM is `-0.00002676`, positive ratio is `0.5883`, and worst
regressions are `75/600` versus eval-zero `35/600`.

## Route Table

| Route | Evidence | Decision |
| --- | --- | --- |
| DTA-v2 CalGate | Multi-seed OOF showed `invert` about `+0.0887 dB`, but zero/shuffle retained most of the improvement and tail/SSIM did not pass. | Positive diagnostic only; no locked test; use as motivation for attribution controls. |
| DTA-v3 DAPC fine-tune | `convir-4090` preflight passed; deterministic eval shuffle audit passed; Phase A R0 and conservative R0 scouts failed. Zero-R0 depthDirect train=`invert` produced mean true-vs-zero surplus `+0.032286 dB`, but hard/SSIM/tail failed. | Mechanism-positive diagnostic only; continue with wide-gate tail/SSIM-guard scouts, no locked test. |

## Reopen Conditions

Reopen this family only with a predeclared DTA-v3 Phase A variant that first
produces a safe generic R0 baseline, or with an alternate depth-active route that
proves true-depth surplus over zero, deterministic shuffle, and wrong-orientation
controls while keeping SSIM and tail regressions no worse than the zero/R0
baseline.

Do not use locked Haze4K test for checkpoint, gate, depth mode, or loss
selection.


## 2026-06-11 DepthDirect Follow-Up

The next allowed queue is not a promotion run: keep `R0=0`, train only
`dta_depth_only` with train depth `invert`, make encoder gates wider, and add
stronger tail/SSIM/mask-budget protection. Continue only if true-vs-zero surplus
stays at least `+0.03 dB` while worst/strong regressions and SSIM improve over
the depthDirect baseline. Locked Haze4K test remains blocked.
