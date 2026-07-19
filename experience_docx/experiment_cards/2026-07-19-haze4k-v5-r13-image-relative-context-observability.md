# Haze4K v5 R13 Image-Relative Candidate Context Observability

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r13_image_relative_context_observability_20260719`.
- Question: does one frozen, target-free within-image candidate-relation
  transform expose specific signed regional utility that is absent from the R11
  local 3x3 interface, while preserving the frozen image-level tail contract?
- Rules commit: `github/main@0e9bda3163d0ab5fb658225cff09d1befd1d08d5`.
- Source branch/commit: terminal R11 implementation
  `c183817e2b3befdeeb12278aa6e6a0574883b6d5`, terminal R11 evidence at
  `893ba97790ad19d745ff676f5bbf28bd37395d50`, and terminal R12 evidence at
  `6bbd4419d754aee580bd859dd762a347a413a70c`. R11/R12 terminals remain
  unchanged.
- Route branch: `codex/haze4k-v5-r13-image-relative-context-observability-20260719`.
- Locked test/canary policy: confirmation identities/images/targets/outcomes,
  canary and locked test are prohibited; all permissions remain false.

## Scientific Contract

- Population and analysis/grouping unit: the exact 384 R11 development
  clean-image groups, folds 0/1 with 192 names each; tile/action rows are nested
  repeated observations and clean-image identity is the OOF split and grouped
  bootstrap unit.
- Intervention or factor contrast and reference: append one fixed 108-scalar
  image-relative context block to every frozen R11 303-scalar local row. Compare
  aligned context with zero-padded local-only, dimension-matched generic
  non-action context, within-image equal-area context shuffle, and fixed
  positive/negative action-identity permutation. All cells use an identical
  `411 -> 64 ReLU -> 2` readout and the exact privileged R10 action-count budget.
- Primary outcome, direction and aggregation: primary estimand is the
  worse-operator image-mean PSNR gain of aligned context minus the strongest
  matched control, with the operator and control maximum chosen conservatively
  inside each of 4,000 fold-stratified clean-image bootstrap draws. Larger
  specific increment, absolute gain, oracle retention and CVaR5 difference
  versus safe-global are better; zero image-level severe (`<=-0.2 dB`) and hard
  (`<=-0.5 dB`) events are required.
- Preferred mechanism and strongest competing explanation: H2 predicts that
  signed utility is present in the within-image relative organization of
  candidate responses but was compressed by R11. H1 predicts that frozen
  responses primarily encode generic difficulty, so aligned relation context
  will not beat generic, shuffled and action-permuted controls. R12 already
  failed a readout-only downside rescue; R13 does not repeat that mechanism.
- Evidence roles and candidate/freeze point: all inputs are reused
  `development_screening`; R13 is one new OOF mechanism screen, not validation.
  Transform, cells, folds, seeds, optimizer, epochs, exact budget, metrics,
  thresholds, bootstrap and terminal mapping freeze at the route commit.
- Primary gate, uncertainty and threshold source: PASS requires
  `LCB95(delta_specific)>+0.010 dB`, aligned absolute-gain LCB95 `>=+0.020 dB`,
  R10-oracle-retention LCB95 `>=0.25`, positive delta-specific in both folds,
  nonnegative fold-by-seed aligned gain, pooled seed range `<=0.020 dB`,
  aligned-minus-safe-global CVaR5 LCB95 `>=-0.005 dB`, zero severe/hard images,
  and complete exact replay. Margins predate outcomes in the R4/R5/R9-R13
  design chain.
- `PASS` authorizes: `R13_FIXED_CONTEXT_MECHANISM_CONTRACT_REVIEW_ONLY`; no
  policy, restoration training/inference, confirmation or protected access.
- `INCONCLUSIVE` authorizes: `NONE`; identity, coverage, OOF, finite-value,
  interval or protected-access failure stops as
  `R13_A0_INPUT_OR_CONTEXT_INCONCLUSIVE_STOP`.
- `FAIL` stops: a decisive utility miss returns
  `R13_A0_RELATIVE_CONTEXT_UTILITY_FAIL_STOP`; utility gates passing with a
  safety miss returns `R13_A0_RELATIVE_CONTEXT_SAFETY_FAIL_STOP`. Both authorize
  `NONE`; do not search relation statistics, features, contexts, heads, seeds,
  epochs, thresholds, action budgets or coverage.

## Frozen Relation Transform

For each image, active action and each of the 18 paired R11 candidate-response
channels, derive only from the 64 candidate-response tiles:

1. 18 percentile midranks normalized to `[-1,1]`, with average ranks for ties;
2. 18 robust z scores `(x-median)/max(IQR,1e-6)`;
3. 18 current-action minus opposite-action percentile-rank differences;
4. 54 current-action `q10/q50/q90` values broadcast to every tile, using the
   fixed linear quantile definition.

No clipping, scaling search, alternate quantile, window or feature selection is
allowed. Generic context applies the same transform to the tilewise mean
absolute response across both actions and fixes the cross-action difference to
zero. Shuffle uses one name/action-bound deterministic permutation within equal
pixel-count strata. Action permutation swaps the two aligned 108-vectors while
leaving the R11 local row and action sign unchanged.

## Implementation Contract

- Exact change and disabled mechanisms: add only the frozen relation transform,
  five matched 411-wide probe cells, grouped inference and typed finalization.
  Reuse the exact R11 cache replay, target reconstruction, tile partition,
  renderer, labels, action assignment and image replay from its SHA-bound source
  checkout. Disable ConvIR construction, candidate generation, restoration
  training/inference, semantic/subgroup selection, adaptive features, search and
  protected roles.
- Checkpoint/load/init/freeze contract: no restoration checkpoint or parameter
  is loaded. Only the matched diagnostic MLPs initialize under seeds 3407/3411.
- Input whitelist and prohibited inputs: allow the eight receipt-bound R11
  development assets, exact R11 source checkout and terminal R11 closeout.
  Learned inputs may contain only R11 local inference-visible rows and the frozen
  response-derived relation block. Prohibit targets, gains, oracle actions,
  filenames/folds/operators, saved predictions, R12 scores, semantic labels,
  confirmation, canary and locked test as features.
- Dataset/split/preprocessing/metric identities: exact R11 folds0/1, D_ref/D_rep,
  add-and-clamp rendering, RGB `[0,1]`, floor-boundary 8x8 tiles, float64 SSE,
  robust local targets and composite image PSNR replay.
- Matched baseline and budget: all five cells share rows, width, primary-arm
  outer-training normalizer, `411->64->2` model, seeds, data order, 24 epochs,
  optimizer, action slots, renderer, bootstrap and gates.
- Resource/cost limits or descriptive-only rationale: CPU-only cached diagnostic;
  expected 300 seconds, hard timeout 1,200 seconds, no resume. The CPU contract
  traverses all 49,152 formal-shape rows, five transforms, 20 one-epoch fits,
  exact assignment and 4,000 grouped bootstrap draws with wall time `<=240`
  seconds and peak RSS `<=3072 MiB`.
- Runner and required assets: unchanged generic runner; exact R11 source commit,
  terminal closeout, manifests, candidate cache, development targets, R5 fold
  rows and R10 terminal evidence declared in the asset manifest.
- Runtime spec and `contract --context` / `run --context` entrypoint:
  `experience_docx/route_runtime_specs/R13_A0_IMAGE_RELATIVE_CONTEXT_OBSERVABILITY_SCREEN.json`
  and `experience_docx/tools/r13_a0_image_relative_context_observability.py`.
- Representative engineering fixture or metadata-only exemption: the production
  transform, matched 411-wide MLP, assignment, replay, grouped bootstrap and
  finalizer run on a protected-data-free formal-shape synthetic fixture; no
  exemption.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R13_A0_IMAGE_RELATIVE_CONTEXT_OBSERVABILITY_SCREEN` | development OOF mechanism screen | context-specific utility plus exact-budget image-level safety | `R13_FIXED_CONTEXT_MECHANISM_CONTRACT_REVIEW_ONLY` |

- First operation: R13_A0_IMAGE_RELATIVE_CONTEXT_OBSERVABILITY_SCREEN
- Expected wall time and monitor profile: 300 seconds expected, 1,200 seconds
  hard timeout, `short` profile.
- Complete-unit resume policy: `none`; interruption requires typed engineering
  review and a fresh output without scientific changes.
- Cloud workspace/run/output/status/closeout: fresh workspace; output
  `r13-a0-image-relative-context-r1`; closeout
  `r13_a0_image_relative_context_observability_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract,
  provenance, input/representation identity, cell/fold-seed/operator/bootstrap,
  gate/resource summaries, typed closeout and one conclusion. Feature matrices,
  row predictions, action maps, model states, raw draws and logs remain
  cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. R11, R12 and all earlier terminals remain
unchanged.
