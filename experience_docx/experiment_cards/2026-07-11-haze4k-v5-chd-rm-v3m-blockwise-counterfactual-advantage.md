# Haze4K v5 CHD-RM v3m Blockwise Counterfactual Advantage

Date: 2026-07-11

Status: `A3_FAIL_STOP_NO_ROUTE_CONFIRM`

Branch: `codex/haze4k-v5-v3m-blockwise-counterfactual-advantage`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/`

## Route Identity

v3m is a new diagnostic audit continuation of v3l, not a repair of the failed
v3l-B transmission policy. Its first purpose is to determine whether block16
remains the best deployable control granularity when image, block, and pixel
oracles all use the same discrete action set.

Parent evidence is GitHub `main` commit `5acaaa54d7aca7c9764dc3dd757ff58cdf6d96fa`.
The runnable source parent is cloud commit `0031b66799ce44574c555fa7bfb879cd5394b991` plus
the frozen v3l artifacts and analysis source recorded in the v3m source manifest.

## A0a Objective

Compare image, block32, block16, and pixel-grid privileged oracles using only
the common ladder `{0, 0.125, 0.25, 0.5, 1.0}`. The primary comparison is
relative to fixed `alpha=0.125` on the same train-derived clean-reference
grouped OOF rows for both frozen operators.

## Forbidden

- no Haze4K locked-test access;
- no canary;
- no controller, router, threshold, backbone, or direct-head training;
- no action-set expansion, post-hoc denominator change, or route-confirm
  strategy selection;
- no use of dense-grid or continuous pixel results to rescue a common-action
  gate failure;
- no checkpoint, tensor, image, raw per-image, or raw block table GitHub sync.

## A0a Gate

For both `D_ref` and `D_rep`, block16 must satisfy all of:

- clean-reference-grouped paired mean-lift CI95 low greater than zero;
- common-action retention CI95 low at least `0.80` relative to pixel-grid lift
  beyond fixed `alpha=0.125`;
- p10 and worst PSNR delta no lower than fixed `alpha=0.125`;
- severe count at `<= -0.2 dB` no higher than fixed `alpha=0.125`.

Pass authorizes only A0b dense-grid and continuous-pixel mechanism audits. Fail
records `V3M_A0_BLOCK16_GRANULARITY_LOCK_FAIL_NO_BLOCK16_CONTROLLER` and
authorizes neither a block controller nor physics policy work.

## Source And Runtime Contract

- Cloud worktree:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711`
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Inputs: frozen v3l `D_ref`/`D_rep` artifacts, exact A0 closeout, v3j split
  manifest, official Haze4K base checkpoint, D7c and density artifacts.
- Dataset: Haze4K `train` only. The v3j route-confirm panel is audit-only.
- Output: cloud-only per-image rows under `cloud_only_raw_common_action/` and
  compact summary/gate artifacts in this evidence root.

## A0a Result

The common-action A0a gate passed for both frozen operators. The exact OOF
block16 results relative to fixed `alpha=0.125` were:

| Operator | Mean lift (dB) | Lift CI95 low (dB) | Retention | Retention CI95 low | Candidate / reference p10 (dB) | Severe count | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `D_ref` | `+0.3571825` | `+0.3381417` | `0.8481430` | `0.8403679` | `+0.0080694` / `-0.0274989` | `0` / `0` | yes |
| `D_rep` | `+0.3555132` | `+0.3374088` | `0.8499444` | `0.8423180` | `+0.0068382` / `-0.0269539` | `0` / `0` | yes |

The raw replay tables remained cloud-only. A constrained compact-artifact
repair corrected a diagnostic field-name error in operator agreement without
rerunning inference: all three raw-table hashes and both gate rows were exactly
unchanged, and the repaired block16 cross-operator Pearson correlation was
`0.9975927`. See `v3m_a0a_closeout.md` in the evidence root.

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

## Next Stage

Only A0b dense-grid and continuous-pixel mechanism cross-audit is authorized.
It must validate the corresponding frozen v3l artifacts and metric pairing
without using route-confirm for selection. A0a does not authorize local
actuation, physics, proxy, controller training, canary, or locked-test access.

The A0b contract is `v3m_a0b_metric_contract.md` in the evidence root. It
compares the 33-level v3l grid (and its continuous-pixel ceiling) against the
five-level A0a common ladder on exactly paired OOF names. The only possible
A0b pass authorization is A1 feasible-local-actuation audit; A0b itself cannot
authorize a controller, training, canary, physics/proxy route, or locked test.

The first A0b read-only audit exposed a metric-contract error before a valid
scientific decision: continuous pixel alpha is analytically optimized before
the final output clamp and therefore is not pointwise nested with the clamped
five-level pixel candidates. A0b-r1 corrects this provenance-preserving
semantic error; it is still a no-inference audit and remains the only
authorized phase.

## A0b-r1 Result

The corrected read-only cross-audit passed. Fixed `alpha=0.125` replay was
exact for all 1,200 OOF names of both operators. For every dense/continuous
comparison, the paired mean-gap 95% upper bound was at most `0.0013426 dB`,
well below the preregistered `0.005 dB`; all finite-grid numerical monotonicity
and both-policy p10/severe tail checks passed. The full compact closeout is
`v3m_a0b_r1_closeout.md` in the evidence root.

Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

This rules out action-ladder density as the material cause of the remaining
block16-to-pixel gap. It authorizes only A1 feasible-local-actuation audit;
controller training, canary, physics/proxy work, and locked-test access remain
blocked.

## A1 Preflight

`v3m_a1_metric_contract.md` defines the only authorized A1 work: frozen
block16 common-ladder oracle labels versus four predeclared, inference-time
signals. It uses no learned controller, no threshold selection, no
route-confirm, and no locked test. A deterministic 32-image OOF replay screen
must prove `alpha=0.125` reproduction before the 1,200-image OOF audit can
read its scientific gate.

## A1 Smoke Result

The corrected r2 32-image screen completed on the frozen OOF fold map. Both
operators reproduced every fixed `alpha=0.125` reference exactly (maximum
absolute PSNR-delta difference `0 dB`), and the cloud-only block table has
`40,001` lines including header. The predeclared signal scores were not used
for selection at this stage.

Decision:
`V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`.

The only promotion is the exact 1,200-image OOF A1 audit defined in
`v3m_a1_metric_contract.md`; all other route restrictions remain unchanged.

## A1 Formal Result

Both frozen operators replayed all 1,200 fixed-alpha rows exactly. The
predeclared direct-step-energy signal passed decisively: grouped image AUROC
CI95 lows were `0.8522` (`D_ref`) and `0.8516` (`D_rep`), with valid labels in
over 99.5% of images. D7c score also passed; alpha1 clipping did not. See
`v3m_a1_formal_closeout.md` for the compact result.

Decision: `V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`.
Only A2 OOF calibration audit is authorized. Training, learned controller,
route-confirm selection, canary, physics/proxy work, and locked test remain
blocked.

## A2 Plan

`v3m_a2_metric_contract.md` defines the only authorized A2 work: a
fold-separated OOF label-calibration audit using the fixed A1 primary signal
`direct_step_energy`. It uses train folds to build a fixed 16-bin monotone
calibration map and evaluates only the held-out fold. It does not train a
controller, use route-confirm for threshold selection, replay policy utility,
touch canary, or touch locked test.

A2 can only authorize A3 frozen-policy replay. A2 cannot claim actual PSNR
policy utility because the A1 block table does not contain every calibrated
candidate action's block MSE.

## A2 Result

The fold-separated OOF calibration audit passed for both frozen operators.
Using only `direct_step_energy`, the fixed 16-bin monotone calibration rule
beat the fixed `alpha=0.125` label baseline with image-grouped ordinal MAE
improvement CI95 lows `0.6734292` (`D_ref`) and `0.6719839` (`D_rep`).
Escalation AUROC CI95 lows were `0.8518716` and `0.8514065`; AP-lift CI95 lows
were `0.3123022` and `0.3117141`. Minimum fold Spearman was `0.9761905` for
both operators. See `v3m_a2_closeout.md`.

Decision:
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`.
Only A3 frozen-policy replay is authorized. Training, learned controllers,
route-confirm selection, canary, physics/proxy work, and locked test remain
blocked.

## A3 Plan

`v3m_a3_metric_contract.md` defines the only authorized A3 work: frozen replay
of the A2 calibrated block16 policy on the same train-derived OOF split. It
first runs a 32-image smoke replay using the full 1,200-image fold map, then
the formal 1,200-image replay only if smoke passes. The candidate policy is
fixed from `v3m_a2_calibration_bins.csv`; A3 may not recalibrate, train, tune,
use route-confirm for selection, touch canary, or touch locked test.

## A3 Result

A3 formal completed all 1,200 train-derived OOF images for both operators with
exact fixed-alpha replay, but failed the policy utility gate. Mean PSNR lift
over fixed `alpha=0.125` was positive (`+0.0828431 dB` for `D_ref`,
`+0.0826054 dB` for `D_rep`), but retention versus block16 oracle was only
about `0.232` and the tail was unsafe: paired lift p10 was about `-0.22` to
`-0.23 dB`, severe counts rose from `0` to `148`/`146`, and hard counts rose
from `0` to `39`/`41`. See `v3m_a3_closeout.md`.

Decision:
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.
No route-confirm audit, canary, locked test, training, learned controller,
ranker, physics/proxy continuation, or policy deployment is authorized.

## Engineering Deviation

The first A0a launcher completed all frozen replay segments and wrote its
cloud-only raw tables, then failed during bootstrap summary construction because
the percentile helper used a NumPy array as a boolean. The repaired summary
command may only read those verified raw tables. It must not load checkpoints,
rerun inference, replace rows, or alter the frozen action set or gates.
