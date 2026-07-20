# R15 S0E Same-Action Population Precision Qualification

Date: 2026-07-20

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r15_s0e_same_action_population_precision_20260720`.
- Question: on all 55 already-used NH-HAZE development pairs, does the exact
  R10 `reference_noop/state_positive_full/state_negative_full` action family
  retain material, spatially specific and tail-safe regional headroom with
  adequate image-level precision, or is the F00 PSNR/8x8/action definition
  unsuitable for the planned target-alignment measurement?
- Rules commit: `github/main@7080dc2c44006f5b62c6a3d302e025c2fb046778`.
- Source branch/commit: R15-S0A terminal branch
  `4bbefc6ef`; immutable S0A launch `fa774910f`; R3 action source
  `207581b4abfff2224bc21d4d1ae4ad5c26118936`; R10 source
  `c455577905efa8bb6f5c0daa84c3ec43c2ee6ff5`.
- Route branch: `codex/haze4k-v5-r15-s0e-same-action-population-precision-20260720`.
- Locked test/canary policy: confirmation, canary and locked test are
  prohibited. NH-HAZE is `development_screening` because v2.7 already used it.

## Scientific Contract

- Population and analysis/grouping unit: all 55 paired NH-HAZE ids 01-55; one
  paired image is the bootstrap and generalization unit. The fixed SHA-balanced
  audit strata contain 27/28 images. Operators and 8x8 tiles are paired/nested,
  never independent samples.
- Intervention or factor contrast and reference: use the exact R3 production
  path to generate R10 no-op, positive-full and negative-full for D_ref/D_rep.
  Pass 1 generates every action without opening any GT and seals per-tensor
  SHA-256 identities. Pass 2 deterministically regenerates and verifies each
  hash before that image's GT is opened. The fixed dual-operator 8x8 regional
  oracle is compared with the exact safe whole-image action and 16
  action-histogram/pixel-area-preserving spatial permutations.
- Primary outcome, direction and aggregation: larger worse-operator regional
  PSNR gain, region-minus-global and region-minus-shuffle are better. Four
  thousand paired primary image-bootstrap draws with PCG64 seed 3407 choose the
  worse operator inside each draw. The two fixed audit strata use 4,000 draws
  each with seeds 3407 and 3408 only to distinguish directional consistency
  from decisive stratum failure. Report regional CVaR5 relative to global, severe,
  hard, mixed noop/active, bidirectional, model-support area and audit-stratum
  consistency.
- Preferred mechanism and strongest competing explanation: the preferred
  explanation is that real-data F00 action relations are sufficiently material
  and precise to justify collecting target-aligned F11 labels. The competitor
  is that the Haze4K action support/headroom collapses or becomes too variable on
  real haze, making action/region/target reformulation upstream of inference.
- Evidence roles and candidate/freeze point: the S0A closeout is the only prior
  authorization and remains unchanged. All NH-HAZE outcomes are development
  evidence. Population, source/checkpoints, actions, operators, grid, controls,
  thresholds, bootstrap, strata and terminal mapping freeze at the route commit.
- Primary gate, uncertainty and threshold source: PASS requires complete exact
  two-pass action replay, GT access after the complete pass-1 manifest, exact
  no-op/sign symmetry, all 55 pairs, pooled regional-gain LCB95 `>=+0.020 dB`,
  region-minus-global and region-minus-shuffle LCB95 each `>=+0.005 dB`,
  region-minus-global CVaR5 LCB95 `>=-0.005 dB`, zero regional severe
  (`<=-0.2 dB`) and hard (`<=-0.5 dB`) images, mixed fraction LCB95 `>=0.25`,
  bidirectional fraction LCB95 `>=0.10`, both audit strata meeting the same
  three materiality point thresholds, both operators directionally positive,
  and regional-gain bootstrap CI half-width `<=0.020 dB`. R10 supplies all
  scientific margins; the precision margin was frozen in R15-S0A before this run.
- `PASS` authorizes: only
  `R15_S0B_IDENTIFIABILITY_MEASUREMENT_CONTRACT_REVIEW_ONLY`; no annotation,
  training, confirmation or old-route reopening.
- `INCONCLUSIVE` authorizes: `R15_REAL_DEVELOPMENT_EVIDENCE_COMPLETION_ONLY`
  when complete valid intervals overlap a scientific gate or precision alone
  fails. It prohibits threshold changes and pseudoreplication.
- `FAIL` stops: if a materiality/support/tail UCB is below its old margin, any
  severe/hard image occurs, or either audit stratum has a materiality UCB below
  an old threshold, permanently
  close the current real-domain `PSNR + fixed 8x8 + R10 three-action` combination
  and authorize only S3 problem reformulation.

## Implementation Contract

- Exact change and disabled mechanisms: no training and no fitting. Run the
  frozen action generator twice, verify exact action hashes, compute full-image
  and fixed-tile SSE, exact R10 controls and grouped bootstrap. Disable all
  selectors, critics, feature extraction for fitting, semantic proxy labels,
  threshold search, sample exclusion and checkpoint selection.
- Checkpoint/load/init/freeze contract: exact S0A/R3 official checkpoint,
  control checkpoint, D_ref/D_rep artifacts and v4a final state. All parameters
  eval/frozen; no init, load relaxation or checkpoint comparison.
- Input whitelist and prohibited inputs: allow only declared NH-HAZE development
  pairs, S0A closeout, exact source checkouts and frozen action assets. Prohibit
  all confirmation/canary/locked data, human labels, undeclared models and file
  identities as decision features.
- Dataset/split/preprocessing/metric identities: flat paired RGB PNGs, ids
  01-55, native 1600x1200, legacy pad-to-factor, no resizing, output clamp
  `[0,1]`, render coefficient 0.25, exact floor-boundary 8x8 tiles, RGB SSE and
  PSNR. SHA-balanced strata sort `sha256(route_id:image_id)` tokens and assign
  the first 27 images to stratum 0 and the remaining 28 to stratum 1.
- Matched baseline and budget: regional/global/shuffle share identical images,
  rendered pixels, actions and operators. Each shuffle preserves the exact
  action-tile and action-pixel-area histogram. Work is 55 images, two full action
  passes, two operators, 64 tiles, 16 shuffles, 4,000 primary bootstrap draws
  and 4,000 fixed draws per audit stratum.
- Resource/cost limits or descriptive-only rationale: one GPU, expected 1,800
  seconds, timeout 3,600 seconds, no resume. Full-resolution actions are hashed
  and deterministically regenerated rather than retained as Git artifacts.
- Runner and required assets: unchanged generic runner plus the typed S0E asset
  manifest.
- Runtime spec and `contract --context` / `run --context` entrypoint:
  `route_runtime_specs/R15_S0E_SAME_ACTION_POPULATION_PRECISION.json` and
  `tools/r15_s0e_same_action_population_precision.py`.
- Representative engineering fixture or metadata-only exemption: CPU contract
  exercises exact R10 tile/oracle/global/shuffle/bootstrap logic on 55 synthetic
  groups, both strata and all gates. The production model path was already
  traversed by S0A at identical source/assets and native resolution; run repeats
  it over the full population.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R15_S0E_SAME_ACTION_POPULATION_PRECISION` | real-development evidence completion | same-action headroom, support, tail and image-level precision | S0B contract review only |

- First operation: R15_S0E_SAME_ACTION_POPULATION_PRECISION
- Expected wall time and monitor profile: 1,800 seconds, `standard`.
- Complete-unit resume policy: `none`.
- Cloud workspace/run/output/status/closeout: fresh MCP-owned workspace; output
  `r15-s0e-same-action-population-r1`; generic status/heartbeat; closeout
  `r15_s0e_same_action_population_precision_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git retains identity/access,
  action-support, bootstrap, controls, strata, gates, resource and scientific
  conclusion JSON/CSV. Per-image/tile rows and pass-1 action hashes remain
  cloud-only; no images, tensors, weights or datasets enter Git.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The route card is immutable after launch. R13, R14 and R15-S0A remain unchanged.
