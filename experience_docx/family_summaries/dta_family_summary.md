# DTA Depth-Transmission Adapter Family Summary

Date: 2026-06-11

Status: positive diagnostic family, not promotion-ready; DTA-v3 tail-lite wide-gate is mechanism-positive but tail/SSIM fail.

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

Decision: `COMPLETED_MECHANISM_POSITIVE_TAIL_FAIL_TAILLITE_WIDE_GATE`. Conservative R0
scout variants failed, so do not launch Phase B from the current R0 checkpoint.
The zero-R0 depthDirect scout shows the first clean mean true-vs-zero surplus for
train=`invert` (`+0.032286 dB`), but it is not promotion-ready because mean vs
A0 is only `+0.013905 dB`, hard is `-0.005602 dB`, dSSIM is `-0.00002676`,
positive ratio is `0.5883`, and worst regressions are `75/600` versus eval-zero
`35/600`. The first wide-gate tailguard queue reduced worst regressions to
`37..46/600`, but all true means were negative and surplus fell below the
mechanism gate. The lighter `wg18_base_s008_b14` row is the current best
diagnostic: mean `+0.024404`, hard `+0.006360`, true-vs-zero `+0.036631`,
true-vs-shuffle `+0.032141`, and true-vs-normal `+0.033084`, but dSSIM remains
negative and worst regressions are `76/600`.

DTA-v3.1 WG18-RiskSelect-AConsistent completed the fold0 scout queue. Output
semantics were clean (`max_abs_noop_diff=0.0`, DTA refine input is residual),
but GT/oracle airlight did not rescue quality, the light hinge only moved mean
from `+0.024404` to `+0.025084` while leaving worst at `76/600`, and same-fold
risk selection fixed SSIM/tail only by reducing coverage to `0.25` and dropping
true-vs-zero surplus to about `+0.0198 dB`. No B0-B4 row passed the scout gate,
so 5-fold x 3-seed formal validation remains blocked.

## Route Table

| Route | Evidence | Decision |
| --- | --- | --- |
| DTA-v2 CalGate | Multi-seed OOF showed `invert` about `+0.0887 dB`, but zero/shuffle retained most of the improvement and tail/SSIM did not pass. | Positive diagnostic only; no locked test; use as motivation for attribution controls. |
| DTA-v3 DAPC fine-tune | `convir-4090` preflight passed; R0 scouts failed. Zero-R0 depthDirect train=`invert` proved surplus. Strong tailguard over-suppressed the branch. Tail-lite `wg18_base_s008_b14` improved mean/hard and passed true-vs-zero/shuffle/normal mechanism thresholds, but SSIM/tail still failed. DTA-v3.1 airlight/risk/light-hinge scout did not pass the fold0 gate. | Mechanism-positive diagnostic only; no 5-fold formal validation from B0-B4; no locked test. |

## Reopen Conditions

Reopen this family only with a predeclared DTA-v3 Phase A variant that first
produces a safe generic R0 baseline, or with an alternate depth-active route that
proves true-depth surplus over zero, deterministic shuffle, and wrong-orientation
controls while keeping SSIM and tail regressions no worse than the zero/R0
baseline.

Do not use locked Haze4K test for checkpoint, gate, depth mode, or loss
selection.


## 2026-06-12 DepthDirect Follow-Up

The DTA-v3.1 follow-up is closed as a scout-gate fail. Reopen only with a new
mechanism that improves tail/SSIM without sacrificing true-vs-zero/shuffle/normal
surplus; do not launch 5-fold x seeds `3407/3411/3413` or locked Haze4K test
from B0-B4.

## 2026-06-12 DTA-v3.2 Reopen Condition

DTA-v3.1 is closed as scout-gate fail, but the family is reopened for a
no/low-training CTDG-SafeMix audit queue only. The queue must first measure
oracle depth-action coverage, alpha action amplitude, GT-transmission/t-error
failure coupling, corrected selector metrics, and fold0 internal nested selector
overfit. No 5-fold x 3-seed formal validation or locked Haze4K test is allowed
until a fixed DTA-v3.2 scout row passes the written gate.

## 2026-06-12 DTA-v3.2 Audit Outcome

The CTDG-SafeMix no/low-training audit completed. Alpha-only shrink is ruled out:
it cannot keep true-vs-zero surplus while meeting SSIM/tail gates. The local
action oracle is strong at image/patch/pixel granularity, so the family remains
worth reopening for a learned SafeMix gate/residual. GT transmission error is a
risk feature rather than a single-cause explanation; low-transmission samples
carry many worst regressions. Same-fold threshold selection remains diagnostic
only after internal fold0 nested smoke. Locked test and 5-fold x 3-seed remain
blocked until a fixed SafeMix scout row passes.


## 2026-06-12 DTA-v3.2 SafeMix Scout Plan

SafeMix C1/C3 is the next fixed scout after the CTDG audit. It keeps
`wg18_base_s008_b14`, disables R0, preserves the train-derived fold0 protocol,
and adds only the new uncertainty/gate/residual heads as partial-load modules.
C1 trains a soft gate over clipped physical action; C3 trains the gate, learned
residual, transmission head, and uncertainty head. Locked Haze4K test and formal
5-fold x 3-seed validation remain blocked until a fixed fallback-A SafeMix row
passes the written fold0 scout gate.
