# Haze4K v5 R5 Spatial Candidate-Response Sufficiency

Date: 2026-07-19

Status: `DRAFT_NOT_AUTHORIZED`

Runtime authorization: `NONE`

This document is a route-design contract only. It is not launch-ready, is not
listed in `route_operations.json`, and authorizes no local or cloud scientific
calculation.

## Identity

- Route id: `haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719`.
- Provisional first operation: `R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN`.
- Question: at fixed 20% image-group coverage, does preserving the spatial
  layout of each frozen active candidate's RGB response relative to no-op add
  non-futile signed action utility and tail safety beyond pooled, spatially
  shuffled, and generic-state controls?
- Rules commit used for design: `github/main@6443ec0da` plus the cloud-audit
  interpretation at
  `github/codex/haze4k-v5-r4b-a1-cloud-evidence-audit-20260718@e8de98ff5`.
- Model source: immutable
  `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch:
  `codex/haze4k-v5-r5-independent-route-contract-20260719`.
- Locked test/canary policy: confirmation identities and outcomes, historical
  A1X-432 outcomes, canary, and locked test are prohibited.

## Evidence Basis And Role

| Evidence | Role in the requested A-D hierarchy | Use in this contract |
| --- | --- | --- |
| R4B-A1 preregistration, typed closeout, conclusion, and terminal index | A, formal confirmatory evidence for the old route | fixes `COMPLETED_GATE_FAIL / R4B_A1_SETWISE_MECHANISM_FUTILITY_STOP / NONE` |
| Identity-verified raw cloud OOF and candidate-risk rows and exact metric reproduction | B, raw support for the formal result | establishes reproducibility and the persisted-field boundary |
| abstention, fold coverage, action-direction, and within-group risk concordance audits | C, post-hoc exploratory evidence | ranks mechanisms; cannot alter R4B-A1 or confirm R5 |
| recorded permutation tolerance miss `1.907e-6 > 1e-6` | D, engineering/numerical evidence | remains visible; does not invalidate the much larger scientific futility gap |

The cloud raw data improved mechanism resolution but did not change the old
terminal decision. R4B-A1 per-action mean/q05 score vectors, per-seed
predictions, composite confidence, and region rows were not persisted and must
not be regenerated under the R4B identity.

## Why This Is An Independent Factor

R4B used three candidate actions but reduced candidate-minus-no-op RGB and
first-encoder responses to summaries. v4a-A1R tested local spatial readout for
a privileged bounded Delta-u direction target. v4a-A1X tested a multiscale
global readout over five exact-half output-side tensors. Neither historical
route crossed the R4B three-action signed-utility problem with an explicitly
layout-preserving candidate-relative response representation.

R5 therefore does not test generic "more spatial capacity." It freezes one
new information factor: the non-DC layout of the active candidate's signed RGB
response relative to no-op. It does not change the ConvIR checkpoint, action
bank, action direction, evaluation operators, coverage, metric definitions, or
protected-data policy.

## Scientific Contract

- Population and grouping unit: the frozen 768-image Haze4K development ledger
  and its four 192-image folds; the provisional A0 test scope is outer folds
  0/1, with one clean-reference image as the independent bootstrap cluster and
  D_ref/D_rep paired within cluster.
- Data role: `development_screening`. Historical reuse and post-hoc route
  selection prohibit a confirmatory or external-validity claim.
- Active actions: exact frozen `state_positive_full` and
  `state_negative_full`; exact `reference_noop` is the fallback and zero
  reference. No action search or action expansion.
- Unique primary variable: whether all non-DC coefficients of the fixed 8x8
  signed RGB candidate-minus-no-op response grid are exposed to the scorer or
  masked to zero.
- Primary reference: parameter-identical pooled/DC-only scorer with the
  non-DC input block zero-masked.
- Required negative control: deterministic within-image permutation of the 64
  response cells before the same complete DCT, preserving channel marginals,
  DC, action identity, vector dimension, model, folds, seeds, and budget.
- Required nuisance control: no-op/state spatial coefficients with the
  candidate-relative block masked, testing generic spatial difficulty.
- Primary outcome: worse-operator selected-policy PSNR gain over no-op at
  exactly 20% group coverage, plus retention against the same three-action
  oracle.
- Primary mechanistic contrasts: spatial minus pooled, true spatial minus
  spatial shuffle, and spatial minus generic-state control.
- Strongest competing explanation: a fixed low-capacity readout or generic
  image difficulty, rather than missing candidate-relative spatial
  information, explains the R4B failure.
- Cheapest discriminating observation: a two-fold, complete-cell, fixed-budget
  OOF non-futility screen. It is cheaper and more identifiable than training a
  new restoration model, which would simultaneously change representation,
  action generation, loss, calibration, and execution.

## Fixed Representation

For each image, operator, and active action:

1. render the no-op and active candidate once under the new R5 identity using
   the unchanged official checkpoint and frozen action definitions;
2. compute signed RGB `candidate - no_op` on `[0,1]` outputs;
3. adaptive-average-pool each RGB channel to exactly 8x8 with no crop,
   padding, resize search, or alternative grid;
4. apply the orthonormal two-dimensional DCT to each 8x8 channel;
5. retain the DC term in every cell and expose either all 63 non-DC terms or
   zeros according to the assigned representation cell.

All 63 non-DC coefficients are retained. There is no frequency selection,
layer search, grid search, or feature subset search. The pooled, spatial,
shuffle, and nuisance cells have the same input dimension by masking, and the
same scorer parameter count.

## Fixed Readout And Decision

- Readout: one per-action MLP with input normalization fitted inside each outer
  training partition, hidden width 64, ReLU, and three outputs: mean utility,
  q05 utility, and severe probability. No attention, convolution, architecture
  search, or old R4/R4B prediction input.
- Initialization/seeds: only the readout is new; Xavier-uniform initialization
  under seeds `3407` and `3411`; arithmetic-average OOF outputs before one
  policy decision. ConvIR and candidate generation remain frozen.
- Loss: fixed equal-weight mean-utility Huber, q05 pinball, and severe binary
  cross-entropy terms. The q05 head is conditional quantile regression against
  the observed candidate utility; the severe label is the frozen clean-image
  any-operator event at `<=-0.2 dB`. No result-driven loss weights.
- Fit budget: 32 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch size 64,
  no early stopping or checkpoint selection.
- Decision: for each held-out group choose the active action with the largest
  averaged predicted q05 utility; rank groups by that value and act on exactly
  `ceil(0.20 * N_test)` groups. Remaining groups use exact no-op. Ties use the
  SHA-256 lexical order of the frozen group id. No threshold or calibration
  search is permitted.
- Descriptive-only coverage curve: fixed 10%, 30%, 40%, 60%, and 100% points.
  It cannot replace or rescue the 20% primary decision.

## Estimand, Uncertainty, And Gates

- Statistical unit: clean-reference image group. Action and operator rows are
  repeated paired measurements, never independent bootstrap units.
- Resampling: 4,000 paired PCG64(3407) group bootstrap draws. Each draw retains
  all cells, actions, and operators for sampled groups and takes the worse
  D_ref/D_rep value before interval construction.
- A0 is a non-futility screen. Utility gates use UCB95; safety and control
  direction use the bounds stated below. A0 PASS is not evidence of
  sufficiency and authorizes only a separately reviewed full-OOF contract.
- Materiality gates: primary gain point estimate `>0` and UCB95
  `>= +0.020 dB`; oracle-retention point estimate `>0` and UCB95 `>=0.25`;
  spatial-minus-pooled and true-minus-shuffle point estimates `>0` and UCB95
  `>=+0.005 dB`.
- Specificity gate: spatial-minus-generic-state point estimate `>0`; severe
  AUROC LCB95 `>0.5` and AUPRC-minus-prevalence LCB95 `>0` are required for the
  primary spatial cell.
- Coverage gate: exactly `ceil(0.20 * N_test)` acted clean-image groups, with
  no missing or duplicate group.
- Safety gates: zero selected severe groups at gain `<=-0.2 dB`; zero selected
  hard groups at gain `<=-0.5 dB`; exact one-sided 95% UCB of the all-group
  severe rate `<=0.010`; matched-coverage CVaR5 spatial-minus-pooled LCB95
  `>=-0.005 dB`; and no operator/native-shape mean below `-0.020 dB`.
- Fixed-cell protection audit: an 8x8 cell is protected only when no-op is the
  local oracle and its no-op MSE is below the outer-training-fold 25th
  percentile. Held-out definitions use that frozen fold threshold. The
  protected-cell spatial-minus-pooled mean-harm UCB95 must be `<=0.005 dB`,
  clustered by image. This is a preregistered non-semantic subgroup and cannot
  support sky, highlight, texture, color, or real-domain claims.
- Integrity gates: exact route/asset identities, complete folds/cells/seeds,
  strict checkpoint load, no protected access, finite outputs, exact 20%
  coverage, valid permutation identities, and coefficient masking equality.

## Terminal Mapping

- `PASS`: only when every integrity, utility non-futility, specificity,
  coverage, and safety gate passes. Decision label:
  `R5_A0_SPATIAL_RESPONSE_NONFUTILITY_PASS`; authorizes
  `R5_A1_FULL_OOF_CONTRACT_REVIEW_ONLY`.
- `FAIL`: a complete valid screen whose utility or spatial-increment UCB is
  below its frozen threshold, or any safety failure. Decision label:
  `R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP`; authorizes `NONE`.
- `INCONCLUSIVE`: missing cells/groups/seeds/operators, invalid identity or
  shuffle, nonfinite output, protected-data violation, or a valid interval that
  neither passes the lower-direction requirement nor proves UCB futility.
  Decision label: `R5_A0_SPATIAL_RESPONSE_INCONCLUSIVE_STOP`; authorizes
  `NONE`.
- Engineering failure tuple, if runtime is later approved:
  `FAILED_ENGINEERING / null / NONE`. It has no scientific interpretation.

No terminal permits grid, DCT subset, width, depth, seed, epoch, LR, loss,
coverage, threshold, operator, or subgroup search. A0 PASS does not authorize
A1 runtime, confirmation access, a deployable policy, or model training.

## Implementation And Asset Boundary

- The official checkpoint must strict-load every official tensor with exact
  shape. There are no new ConvIR parameters and no partial checkpoint load.
- Only the diagnostic MLP is trainable. It must use a route-specific prefix and
  never alter the official model builder/default behavior.
- Candidate generation is a new R5 operation from the immutable anchor. It is
  not a rerun, repair, or reconstruction of R4B-A1 and must not consume R4B-A1
  checkpoints, omitted scores, or old predictions.
- Allowed inputs: Haze4K development hazy/clean pairs for the frozen folds,
  official checkpoint, fixed action/operator definitions, candidate/no-op RGB,
  and the declared group/fold map. Clean RGB is a target/evaluation input only,
  never a readout input.
- Prohibited inputs: filenames as learned features, confirmation identities or
  outcomes, historical A1X-432 outcomes, canary, locked test, semantic labels,
  old R4/R4B predictions, GT-derived oracle action as a model input, and any
  unregistered feature.
- The draft asset manifest contains unresolved identities that block runtime
  authorization. No route operation, runtime spec, entrypoint, cloud workspace,
  output id, or launch receipt may be created from this draft.

## Planned Evidence If Runtime Is Separately Authorized

GitHub compact evidence: immutable route card, typed asset/provenance and data
role manifests, representation identity, factor/cell summary, oracle and margin
summaries, label/operator/fold/seed stability, harm prevalence, fixed-coverage
and descriptive risk-coverage results, calibration, protected-cell and tail
results, grouped bootstrap, typed closeout, scientific conclusion, and terminal
index record.

Cloud-only evidence: raw per-image/action/operator/cell rows, candidate tensors,
readout states, per-seed predictions, full logs, dataset content, images,
arrays, and checkpoints.

## Current Authorization

Allowed now: read-only evidence verification, documentation maintenance, and
review of this independent draft.

Forbidden now: materializing runtime code/specs, generating candidate responses,
training the diagnostic readout, running offline statistics, accessing any
protected role, or launching R5. The only current action is contract review.
