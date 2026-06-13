# DTA Depth-Transmission Adapter Family Summary

Date: 2026-06-13

Status: DTA-v3.7 Phase C1 real soft-blend oracle passed; continue with integrated T/A/U supervised candidate training and deployable U-TQS soft-mix policy, not hard-reject threshold search.

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

DTA-v3.4 FDF-TSR proved that feature-level depth fusion carries real signal but
also made the over-action failure explicit. The train-derived E1-E4 triage had
mean/hard gains and strong depth-control surplus, but positive ratio remained
around `0.579-0.590` and worst regressions stayed around `116-128/600`.

DTA-v3.7 U-TQS-Mix is now authorized as the family mainline. The route
keeps the ConvIR-B A0 anchor and conservative FDF/DTA action family, but changes
the strategy space from hard accept/reject to utility-aware soft action-bank /
shrink-mix. It explicitly requires transmission, airlight, quality, and
uncertainty gain-risk signals before any deployable policy claim. The v3.7
execution order is Phase A table-only soft-oracle diagnostics, then TQS
gain-risk predictor and real soft-blend/integrated-head validation in parallel
if Phase A passes. D1 and later multi-variant routes must start with a staged
screen (`folds 0,1 x seeds 3407,3411`) and only promote a fixed top
candidate/policy to full `5 folds x 3 seeds`. Do not continue v3.6 threshold
tuning as the main path, and do not start broad full-formal queues before
screen evidence exists.

DTA-v3.5 FDF-RCS-Lite completed the relaxed train-derived flow on
`convir-4090`. Conservative FDF moved the family in the intended direction:
L1/L3 positive ratio reached about `0.630`, L2 reduced worst to `60.5/600`, and
all non-L0 variants kept positive dSSIM and true-vs-zero surplus. Strict gates
still failed because all-image worst remained above `48/600`, and the nested
threshold selector only reached a low-coverage relaxed diagnostic. DTA-v3.6 HRCS
then confirmed on 5-fold x 3-seed formal train-derived validation that L1/L3
oracle high-coverage rows can strict-pass, but deployable selectors still lose
coverage and positive ratio. The bottleneck remains deployable risk-calibrated
selection rather than more residual/router capacity.

## Route Table

| Route | Evidence | Decision |
| --- | --- | --- |
| DTA-v2 CalGate | Multi-seed OOF showed `invert` about `+0.0887 dB`, but zero/shuffle retained most of the improvement and tail/SSIM did not pass. | Positive diagnostic only; no locked test; use as motivation for attribution controls. |
| DTA-v3 DAPC/FDF fine-tune | `convir-4090` preflight passed; R0 scouts failed. Zero-R0 depthDirect train=`invert` proved surplus. DTA-v3.1 airlight/risk/light-hinge, DTA-v3.2 SafeMix, DTA-v3.3 RouterFusion, and DTA-v3.4 FDF-TSR failed their written gates. DTA-v3.5 FDF-RCS-Lite fixed much of the over-action pattern but still failed strict all-image tail; DTA-v3.6 HRCS formal validation confirms strong oracle selector/action-bank headroom but deployable selectors still strict-fail. | Mechanism-positive diagnostic only; no promotion. A relaxed one-shot locked test may only use the fixed L3 logistic `deployable_all` target `0.93` policy under the 2026-06-13 user override, with no post-test tuning. |
| DTA-v3.7 U-TQS-Mix | Phase A soft-oracle and Phase C1 actual real-blend oracle both pass; Phase B/B2 deployable table policy still fails. | Mainline route. Do not resume hard-reject threshold search; proceed to integrated T/A/U supervised candidate training and deployable utility-aware soft-mix policy. |

## Reopen Conditions

Reopen this family only with a predeclared selector-first route that uses the
DTA-v3.5 OOF action table/oracle evidence to improve nested risk calibration, or
with a materially safer feature-action candidate that keeps true-depth surplus
over zero, deterministic shuffle, and wrong-orientation controls while bringing
all-image worst regressions inside the strict gate.

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

## 2026-06-12 DTA-v3.4 One-Shot Test Outcome

Decision: `COMPLETED_ONE_SHOT_TEST_FAIL_NO_FURTHER_TEST_SELECTION`.

The user-requested DTA-v3.4 FDF-TSR `E2=e2_tiny_residual` one-shot Haze4K test
completed on `convir-5090` after copying the Depth Anything cache from
`convir-4090`. True-depth attribution on test is strong
(`true-vs-zero=+0.093063`, `true-vs-shuffle=+0.106035`,
`true-vs-normal=+0.138891`) and dSSIM is positive (`+0.00004687`), but absolute
quality fails: mean dPSNR is `-0.014802`, positive ratio is `0.489`, and worst
regressions are `257/1000`. This is diagnostic only and not promotion-ready.
Do not run another Haze4K test variant from this result.

## 2026-06-12 DTA-v3.4 Train-Derived Triage Outcome

Decision: `TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED`.

The follow-up train-derived triage completed on `convir-5090` with
`RUN_TEST=0`, variants `E1-E4`, folds `0/1`, and seeds `3407/3411`. All variants
show strong depth attribution and positive internal mean/hard metrics, but all
fail the written safety gate because positive ratio remains around
`0.579-0.590` and worst regressions remain around `116-128/600` on average.
The best mean row is `e4_plus_film` (`+0.081446` mean, `+0.057209` hard), but
its worst count is still `123.00/600` and `max_run_worst=137`. Do not launch
formal 5-fold x 3-seed validation from DTA-v3.4, and keep locked Haze4K test
blocked.

## 2026-06-12 DTA-v3.5 FDF-RCS-Lite Plan

DTA-v3.4 is closed as `TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED`: FDF produced
strong depth attribution, but broad feature action and non-nested TSR/router
training failed positive-ratio and worst-tail gates. The family is reopened for
DTA-v3.5 FDF-RCS-Lite only.

DTA-v3.5 keeps the v3.4 feature-level depth fusion mechanism but makes action
conservative: first-grid feature fusion strength is `{0.02,0.04}`, gate limit is
`{0.15,0.25}`, and gate bias is `-2.0`. It adds explicit A0 MSE regression,
feature gate/action budget, optional tiny learned residual action budget, and a
post-run nested fold selector over train-derived OOF evidence.

The user requested relaxed continuation metrics so the full train-derived flow
can complete. This relaxation is diagnostic only: strict gates remain reported,
formal claims require the nested reports, and locked Haze4K test remains blocked
unless a fixed config is explicitly authorized later.

## 2026-06-12 DTA-v3.5 FDF-RCS-Lite Outcome

Decision: `COMPLETED_RELAXED_FLOW_PASS_STRICT_FAIL_SELECTOR_DIAGNOSTIC_LOCKED_TEST_BLOCKED`.

The DTA-v3.5 relaxed train-derived queue completed on `convir-4090`. The first
queue finished all non-L0 variants and only failed because the old L0 A0 sanity
eval path had an engineering bug. The L0 repair/postprocess queue ran from
commit `4c7589b` with two GPUs, within the user-requested five-GPU cap, and
produced the final summary, OOF action table, oracle risk-coverage curve, and
nested selector reports. Locked Haze4K test was not touched.

All non-L0 variants passed the relaxed diagnostic flow but failed strict triage:

| Variant | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | true-vs-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 conservative FDF | `+0.071183` | `+0.081919` | `+0.00001536` | `0.6308` | `82.50` | `+0.069518` |
| L2 smaller action | `+0.052706` | `+0.064818` | `+0.00001700` | `0.6250` | `60.50` | `+0.045710` |
| L3 lower gate | `+0.062959` | `+0.075288` | `+0.00001736` | `0.6304` | `69.50` | `+0.056608` |
| L4 tail losses | `+0.066923` | `+0.078181` | `+0.00001603` | `0.6288` | `78.50` | `+0.061267` |
| L5 residual 0.010 | `+0.066249` | `+0.078210` | `+0.00001698` | `0.6296` | `76.25` | `+0.055320` |
| L5 residual 0.015 | `+0.066245` | `+0.078211` | `+0.00001696` | `0.6296` | `76.25` | `+0.055353` |

The result validates the conservative-action diagnosis from v3.4: lowering
feature gate/strength improves positive ratio and tail counts while preserving
depth attribution. However, L4/L5 show that adding the tested tail losses and
tiny residual does not solve all-image tail safety.

The oracle risk-coverage curve is the main positive result. It shows that the
candidate action has safe-subset headroom, including positive oracle mean/hard
at `0.50` coverage with zero worst regressions for non-L0 variants. The current
nested threshold selector is not enough: the best relaxed selector diagnostic is
L4 at about `0.21` coverage with selected mean about `+0.0197 dB`, selected
positive ratio about `0.67`, and worst about `31.5/600`. Continue only with a
stronger selector/calibration route; do not increase router/FiLM/residual
capacity or run locked test from DTA-v3.5.

## 2026-06-13 DTA-v3.6 HRCS Phase A

Decision: `PHASE_A_COMPLETED_RELAXED_PASS_STRICT_FAIL_FORMAL_QUEUE_PENDING`.

DTA-v3.6 HRCS reopens the family only for selector/calibration, continuing from
DTA-v3.5 commit `1e1d87a` on branch `codex/haze4k-dta-v3-6-hrcs`. Phase A ran
on `convir-4090` from commit `754d62a` using the existing v3.5 OOF action table
and did not touch locked test.

The high-coverage oracle is strong enough for strict all-image gates: L3 oracle
at `0.95` coverage has mean `+0.088241`, positive ratio `0.6304`, and worst
`39.50/600`; L1 oracle at `0.93` coverage has mean `+0.106708`, positive ratio
`0.6308`, and worst `40.50/600`. This confirms that the conservative FDF action
family has deployable value if the risk boundary can be learned.

The deployable high-coverage selectors remain strict-fail. Best relaxed
deployable rows are L1 logistic `input_only` at coverage `0.9000`, mean
`+0.075882`, positive ratio `0.5846`, worst `63.25/600`; L2 logistic
`input_only` at coverage `0.9017`, mean `+0.055817`, positive ratio `0.5783`,
worst `44.50/600`; and L3 logistic `input_only` at coverage `0.9042`, mean
`+0.067108`, positive ratio `0.5862`, worst `52.25/600`. The selector-action
bank `{A0,L2,L3,L1}` is close on tail (`50.25/600`) but still loses too many
positive samples (`0.5833` positive ratio).

Interpretation: v3.6 strengthens the root-cause claim. Candidate capacity is not
the first bottleneck; deployable risk calibration/features are. Per the
2026-06-13 user instruction, the route continues into a relaxed 5-fold x 3-seed
train-derived queue, but this is exploratory and not promotion-grade unless
strict gates pass without post-test tuning.


## 2026-06-13 DTA-v3.6 HRCS Formal Outcome

Decision: `FORMAL_COMPLETED_RELAXED_PASS_STRICT_FAIL_FIXED_POLICY_READY_LOCKED_TEST_UNTOUCHED`.

The relaxed formal train-derived queue completed on `convir-4090` from commit
`6f5965e` with marker `DTA_V3_6_HRCS_FORMAL_QUEUE_OK`. It ran L1/L2/L3 across
folds `0..4` and seeds `3407/3411/2026`, producing `45` candidate train runs,
`180` depth-control evals, `45` aggregate jobs, and a `27000`-row formal OOF
action table. Locked Haze4K test was not touched.

Best deployable formal rows all relaxed-pass but strict-fail:

| Candidate | Selector | Feature group | Coverage | mean dPSNR | positive ratio | worst/600 | max outer worst/600 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| L1 `s004_g025` | logistic | `deployable_all` | `0.8961` | `+0.075380` | `0.5876` | `56.60` | `71.67` |
| L2 `s002_g025` | logistic | `deployable_all` | `0.8931` | `+0.054269` | `0.5788` | `39.47` | `50.33` |
| L3 `s004_g015` | logistic | `deployable_all` | `0.8887` | `+0.065404` | `0.5817` | `47.07` | `59.33` |

Oracle evidence remains the key positive sign: L1 oracle strict-passes at
`0.93-0.95` coverage, and L3 oracle strict-passes at `0.93-0.97` coverage. The
oracle action bank `{A0,L2,L3,L1}` reaches mean `+0.143298`, positive ratio
`0.6623`, and zero worst regressions, while the deployable selector bank reaches
mean `+0.065880`, positive ratio `0.5980`, and worst `48.87/600`.

Interpretation: v3.6 formal validation confirms the candidate/action family is
usable in principle, but the deployable selector still cannot preserve enough
positive samples at high coverage. The fixed relaxed one-shot policy, if the
user continues to locked test, is L3 `l3_fdf_lite_s004_g015_bm2` with logistic
`deployable_all` HRCS at coverage target `0.93`, falling back to A0 on reject.
This is exploratory only and not promotion-ready.

## 2026-06-13 DTA-v3.7 U-TQS-Mix Plan

Decision: `AUTHORIZED_PHASE_A_TABLE_ONLY_FIRST`.

The final route correction is now written into project memory: preserve ConvIR-B
A0, keep conservative FDF/DTA actions, abandon hard reject as the main strategy,
and move to utility-aware soft action-bank / shrink-mix with explicit
transmission, airlight, quality, and uncertainty supervision/features. Negative
samples are not banned; severe regressions are the hard constraint.

Phase A is mandatory but intentionally fast and no-training. It consumes the
v3.6 formal OOF action table and writes:

```text
v37_positive_loss_budget_report.csv
v37_soft_action_bank_oracle_grid.csv
v37_false_reject_false_accept_taxonomy.csv
v37_feature_ablation_auc_report.csv
v37_tA_quality_uncertainty_preflight.json
```

If Phase A soft oracle strict-passes, launch TQS gain-risk prediction and real
soft-blend/integrated-head validation in parallel on cloud GPUs. If Phase A
fails, do not train v3.7 policy; redesign the candidate action family first.
Locked Haze4K test remains blocked until a formal train-derived strict pass
seals one fixed policy.

## 2026-06-13 DTA-v3.7 Phase A Outcome

Decision: `PHASE_A_PASS_SOFT_ORACLE_HEADROOM`.

The Phase A table-only diagnostic completed on `convir-4090` from commit
`71d1f88` and consumed `27000` formal v3.6 OOF rows. It produced all required
artifacts and found `13/18` strict-passing soft-oracle rows. The best
`A0_L2_L3_L1_full` oracle row reaches mean `+0.143298`, hard `+0.121101`, dSSIM
`+0.00002551`, positive ratio `0.6623`, worst `0/600`, and max outer worst
`0/600`.

This confirms that the strategy-space change has enough headroom. It also
confirms why v3.6 hard reject must not remain the main path: L3 hard reject loses
`32.73/600` positives, about `8.77x` the strict L3 positive-loss budget. Current
deployable severe-risk AUC remains only about `0.608`, so Phase B must improve
T/A/Q/U separability and then verify real blended outputs on train-derived folds.
Locked Haze4K test remains blocked.


## 2026-06-13 DTA-v3.7 Phase B Outcome

Decision: `PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND`.

Nested table-only ridge TQS policies completed on `convir-4090` using current
v3.6 OOF features and soft action-bank candidates. No feature group strict-passed.
The best deployable group was `T_pred`: mean `+0.015792`, hard `+0.013137`,
dSSIM `-0.00000566`, positive ratio `0.6360`, worst `0.80/600`, and max outer
worst `2.33/600`. Tail control is easy, but gain collapses. This preserves the
main route judgment: the action family has headroom, but deployable T/A/Q/U
feature separability is still insufficient. The next step is feature enrichment
and real blended-output verification, not v3.6 hard-reject threshold tuning.


## 2026-06-13 DTA-v3.7 Phase B2 Outcome

Decision: `PHASE_B2_ENRICHED_TABLE_POLICY_STRICT_FAIL`.

B2 extracted deployable image quality/color/edge/dark-channel features for all
`3000` train hazy images and reran nested TQS with `v37_tqs_enriched_*` outputs.
Strict pass count remained `0`. The enriched all-deployable group improved mean
and dSSIM relative to the first table policy (`+0.021754` mean, `+0.024839` hard,
`+0.00000301` dSSIM), but positive ratio fell to `0.5128`. This confirms tail
control is not the hard part; preserving enough high-gain positive action is.
Continue to real soft-blend verification and integrated T/A/U supervised
candidate training. Do not reopen v3.6 hard-reject threshold search.


## 2026-06-13 DTA-v3.7 Phase C1 Outcome

Decision: `PHASE_C1_REAL_BLEND_ORACLE_PASS`.

Phase C1 rendered actual image-space blends for the v3.7 action bank on
train-root OOF fold validation images, across folds `0..4` and seeds
`3407/3411/2026`. This replaced Phase A's linear metric proxy with real tensor
outputs:

```text
blend = clamp(A0 + alpha * (candidate - A0), 0, 1)
```

Aggregate marker:

```text
DTA_V3_7_REAL_BLEND_AGGREGATE_OK rows=162000 grid=18 strict_pass=14 decision=PHASE_C1_REAL_BLEND_ORACLE_PASS
```

Best row: `A0_L2_L3_L1_micro_shrink / max_dpsnr` with mean `+0.143568`, hard
`+0.121118`, dSSIM `+0.00002579`, positive ratio `0.6977`, worst `0/600`, max
outer worst `0/600`, true-vs-zero `+0.106861`, true-vs-shuffle `+0.080749`, and
true-vs-normal `+0.088555`.

Interpretation: the soft action-bank/micro-shrink strategy is image-space valid,
not a table artifact. The family bottleneck is now deployable gain-risk
separability and supervised T/A/U policy training. Do not reopen v3.6 hard
accept/reject threshold tuning, and do not spend the next step on broad router
capacity before integrated T/A/U supervision.
