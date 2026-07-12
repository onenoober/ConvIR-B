# v3m Route Decision

Start decision:
`V3M_START_A0A_COMMON_ACTION_GRANULARITY_ONLY_NO_TRAINING_NO_CANARY_NO_LOCKED_TEST`.

v3l established deterministic frozen operators and large oracle headroom, but
block16 and pixel results used different action spaces. A0a isolates that
remaining granularity question with a common ladder. Route-confirm is emitted
only as an audit result and cannot select policies, thresholds, or gates.

Only a dual-operator A0a pass authorizes A0b dense-grid and continuous-pixel
mechanism audits. No other next stage is implied by A0a alone.

The initial frozen replay completed, but its bootstrap-summary implementation
failed before any gate decision. The only authorized recovery is a no-inference
summary rebuild from the existing verified cloud-only raw rows.

## A0a Decision

The constrained rebuild passed the common-action block16 gate for `D_ref` and
`D_rep`. A subsequent compact-only repair changed the operator-agreement reader
from the nonexistent `mean_selected_alpha_mean` field to the raw-table field
`selected_alpha_mean`. It backed up the prior JSON/CSV, read the same raw rows,
and preserved all raw SHA256 values and gate fields exactly.

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

A0b may only cross-audit the already computed dense-grid and continuous-pixel
frozen evidence. It cannot use route-confirm to select a policy and cannot
authorize A1 feasible local actuation, any physics/proxy work, training,
canary, or locked-test access.

## A0b Preflight Decision

`v3m_a0b_metric_contract.md` freezes a no-inference cross-audit. It requires
the v3l 33-level grid and continuous-pixel rows to have the pinned SHA256
values, exact fixed-alpha replay agreement with the v3m five-level rows, and
the same 1,200 OOF names. Its only scientific gate is whether the 95% upper
bound of the dense/continuous advantage over the five-level ladder is at most
`0.005 dB` for every policy/operator pair, with no nested-action monotonicity
violation. No result has been read under this contract yet.

## A0b Metric-Contract Correction

The initial read-only A0b command completed with all fixed-alpha replays exact
and all mean-gap upper bounds below `0.005 dB`, but it was not a valid gate.
The contract incorrectly required per-image dominance for the continuous pixel
policy. That policy solves alpha against the unclamped residual and clamps only
the final output, so it is not the same finite candidate minimization as the
five-level clamped grid. Tiny grid-only negatives also remain within the
pre-existing `1e-6 dB` replay precision.

Therefore record the initial output as `FAILED_METRIC_CONTRACT`, not as an A1
block. A0b-r1 retains all source hashes and paired means, uses `1e-6 dB` only
for grid numerical monotonicity, treats continuous pixel as a ceiling
diagnostic, and requires both policies to satisfy their existing p10/severe
tail checks against fixed `alpha=0.125`. No A1 result exists until A0b-r1
closes.

## A0b-r1 Decision

A0b-r1 retained the exact same pinned inputs and paired OOF names, then applied
the corrected semantics. Fixed `alpha=0.125` replay was exact for both
operators. All eight dense/continuous-vs-common pairs passed their `0.005 dB`
mean-gap upper-bound gate, all finite-grid comparisons met `1e-6 dB` numerical
monotonicity, and both policies in every pair preserved p10 and severe-count
tail safety against their own fixed-step reference.

Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

The next and only authorized stage is A1 feasible-local-actuation audit. It
must determine whether the frozen block16 oracle action can be actuated from
the already available deployable signal without controller training. Nothing
in A0b-r1 authorizes a learned controller, route-confirm selection, physics or
proxy policy work, canary, or locked-test access.

## A1 Preflight Decision

A1 may determine only whether any fixed, inference-time signal can rank the
block16 common-ladder oracle's `alpha > 0.125` action label without fitting a
controller. The candidate signals are frozen before launch: D7c score mean,
direct-step energy, their fixed product, and alpha1 clamp exposure. The gate is
grouped by image on OOF: one same-direction signal must have an AUROC CI95 low
of at least `0.56` for both frozen operators, with valid positive/negative
blocks in at least 80% of images. A 32-image fixed-alpha replay screen precedes
the 1,200-image formal audit.

No controller, calibration threshold, route-confirm choice, canary, physics or
proxy policy, training, or locked-test access is authorized by this preflight.

## A1 Smoke Decision

The corrected r2 screen used the full 1,200-name OOF fold map before selecting
its first 32 names. All 64 fixed-alpha replay comparisons were exact and its
cloud-only block table contained 40,000 records. The smoke signal summaries are
engineering diagnostics only and did not select a candidate.

Decision:
`V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`.

Only the exact 1,200-image formal A1 OOF audit is authorized. It must use the
same signal set, direction, target, bootstrap, and gate in the existing metric
contract; no other next stage is implied by this engineering screen.

## A1 Formal Decision

Both frozen operators reproduced all 1,200 fixed-alpha OOF rows exactly.
Direct-step energy passed the common local-observability gate with grouped
AUROC CI95 lows `0.8522` and `0.8516`; D7c score and its fixed product also
passed, while clip exposure failed. This is a strong deployable observability
result, not a trained controller.

Decision: `V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`.

A2 may only perform fold-separated OOF calibration of a fixed signal. No
learned controller, route-confirm threshold, canary, training, physics/proxy
policy, or locked-test access is authorized.

## A2 Decision

A2 used only the fixed A1 primary signal, `direct_step_energy`, with
fold-separated OOF calibration. The monotone calibration rule collapsed from
16 target bins to 9 actual bins because of duplicate calibration-fold score
quantiles, but passed the label-only gates by a wide margin for both frozen
operators. Ordinal MAE improvement CI95 lows were `0.6734292` and `0.6719839`;
escalation AUROC CI95 lows were `0.8518716` and `0.8514065`; AP-lift CI95 lows
were `0.3123022` and `0.3117141`.

Decision:
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`.

A2 is label-only and does not claim actual PSNR policy utility. Only A3 frozen
policy replay is authorized. No learned controller, route-confirm threshold,
canary, training, physics/proxy policy, or locked-test access is authorized.

## A3 Decision

A3 replayed the frozen A2 calibrated block16 policy on all 1,200 train-derived
OOF images for both frozen operators. Fixed `alpha=0.125` replay was exact. The
policy had positive mean lift over fixed alpha (`+0.0828431 dB` and
`+0.0826054 dB`) but failed the preregistered utility gate: retention versus
block16 oracle was only about `0.232`, paired lift p10 was about `-0.22` to
`-0.23 dB`, severe counts rose from `0` to `148`/`146`, and hard counts rose
from `0` to `39`/`41`.

Decision:
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.

No route-confirm audit, canary, locked-test access, controller training,
learned ranker, physics/proxy continuation, or policy deployment is authorized.

## A3 Failure-Decomposition Diagnostic Decision

A corrected r1 post-fail diagnostic read only the completed A3 replay rows and
A2 calibration bins. It did not train, tune thresholds, rerun inference, replay
a new policy, use route-confirm, touch canary, or touch locked test. The r0
diagnostic used paired lift instead of actual policy PSNR delta for severe/hard
counts and is retained only as a metric-mismatched operational deviation.

R1 confirms that the tail failure is stable across frozen operators: severe
overlap was `140` of union `154`, hard overlap was `38` of union `42`, policy
lift correlation was `0.9930474`, selected-alpha correlation was `0.9962972`,
and oracle-lift correlation was `0.9970668`. Severe images still had positive
block16-oracle headroom on average, while aggressive A2 calibration bins mixed
oracle labels heavily.

Decision:
`V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION`.

This sharpens the bottleneck to safe utility calibration / action semantics
under aggressive local escalation, but it authorizes no next stage.
