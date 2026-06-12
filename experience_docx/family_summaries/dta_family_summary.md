# DTA Depth-Transmission Adapter Family Summary

Date: 2026-06-11

Status: positive diagnostic family, not promotion-ready; DTA-v3.3 RouterFusion triage failed and locked Haze4K test remains blocked.

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

DTA-v3.2 SafeMix C3 improved the depth-attributed line but still failed the
fixed scout gate: fallback-A mean `+0.031636`, hard `+0.009309`,
true-vs-zero `+0.039813`, true-vs-shuffle `+0.034174`,
true-vs-normal `+0.035193`, and worst `48/600`, with dSSIM still negative and
positive ratio `0.6133`.

DTA-v3.3 RouterFusion-SafeMix++ triage completed on `convir-4090` across
fold0/fold1 x seeds `3407/3411`. D1/D2 improved mean/hard/depth surplus but
failed the worst-tail gate (`worst` about `80/600`). D3 RouterFusion was too
suppressive or misrouted: aggregate mean `+0.032786`, hard `+0.035370`, dSSIM
`-0.00000629`, positive ratio `0.5713`, worst `75.5/600`,
true-vs-zero `+0.029598`, true-vs-shuffle `+0.026012`, and true-vs-normal
`+0.026889`. No D-row passed triage, so formal 5-fold x 3-seed and locked test
remain blocked.

## Route Table

| Route | Evidence | Decision |
| --- | --- | --- |
| DTA-v2 CalGate | Multi-seed OOF showed `invert` about `+0.0887 dB`, but zero/shuffle retained most of the improvement and tail/SSIM did not pass. | Positive diagnostic only; no locked test; use as motivation for attribution controls. |
| DTA-v3 DAPC fine-tune | `convir-4090` preflight passed; R0 scouts failed. Zero-R0 depthDirect train=`invert` proved surplus. Strong tailguard over-suppressed the branch. Tail-lite `wg18_base_s008_b14` passed mechanism thresholds but failed SSIM/tail. DTA-v3.1 airlight/risk/light-hinge, DTA-v3.2 SafeMix, and DTA-v3.3 RouterFusion all failed their written scout/triage gates. | Mechanism-positive diagnostic only; no 5-fold formal validation from B0-B4/C/D rows; no locked test. |

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

## 2026-06-12 DTA-v3.2 SafeMix Scout Outcome

SafeMix C1/C3 completed and remains diagnostic only. C3 fallback-A is the best
row: mean `+0.031636`, hard `+0.009309`, true-vs-zero `+0.039813`,
true-vs-shuffle `+0.034174`, true-vs-normal `+0.035193`, and worst `48/600`.
This confirms SafeMix learned residual/gating is a better direction than C1
physical-action gate-only and better than B4 on tail count, but it still fails
hard, SSIM, and positive-ratio gates. Do not launch 5-fold x 3-seed or locked
test from this row.

## 2026-06-12 DTA-v3.3 RouterFusion-SafeMix++ Plan

DTA-v3.2 SafeMix C3 is the best current diagnostic but still fails hard, SSIM,
and positive-ratio gates, so formal 5-fold x 3-seed and locked test remain
blocked. The family is reopened only for a DTA-v3.3 RouterFusion-SafeMix++
triage queue: build on C3, keep R0 disabled, add image/patch/pixel routing,
low-phys/high-learned SafeMix, SSIM-CVaR/group-tail losses, and counterfactual
wrong-depth gate suppression. Continue only if a fixed D-row passes the
predeclared fold0/fold1 x seeds 3407/3411 triage gate while preserving
true-vs-zero/shuffle/normal surplus.

## 2026-06-12 DTA-v3.3 RouterFusion Triage Outcome

Decision: `TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED`.

DTA-v3.3 completed the low-cost triage on `convir-4090` from commit `bc28db8`.
No variant passed. D1/D2 show that stronger SSIM-CVaR/group-tail losses and
low-phys/high-learned action can create strong average and hard gains, but they
do not control the worst tail. D3 shows that the implemented image/patch/pixel
RouterFusion is not a reliable selector: it sacrifices positive ratio and
depth-control surplus while still leaving worst regressions high.

Reopen only with a material mechanism change. The two acceptable directions are
either a stricter tail-safe selector with explicit worst-regression constraints,
or a pivot toward UDP/DeHamer-style multi-scale feature-level depth fusion with
weak bounded late RGB correction. Do not run formal 5-fold x 3-seed or locked
Haze4K test from D1/D2/D3.

## 2026-06-12 DTA-v3.4 FDF-TSR Plan And Test Override

DTA-v3.3 RouterFusion as implemented is stopped. The next material mechanism
change is DTA-v3.4 FDF-TSR: move depth usage from late physical RGB correction
toward feature-level depth fusion, keep late physical action disabled or
near-zero, and optionally use a tiny bounded learned residual.

Repository rule update: future Haze4K model routes are fine-tuning routes by
default from the official Haze4K checkpoint unless a route card explicitly says
otherwise.

The user explicitly requested one Haze4K test experiment and result images with
wide gates. Record this as `USER_EXPLICIT_TEST_OVERRIDE_ONE_SHOT`; do not use the
result for repeated test-set selection.
