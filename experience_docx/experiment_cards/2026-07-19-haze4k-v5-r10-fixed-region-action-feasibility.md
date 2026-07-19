# Haze4K v5 R10 Fixed-Region Action Feasibility

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r10_fixed_region_action_feasibility_20260719`.
- Question: on the exact R9 development population, does a fixed 8x8, dual-operator-safe three-action regional oracle provide material mean and tail benefit beyond the best dual-operator-safe whole-image action, with real spatial-layout and signed-direction heterogeneity?
- Rules commit: `github/main@57d87d653cebcbe781bce1930598db88bfd5d3b9`.
- Source branch/commit: immutable R3 A0 candidate cache and Haze4K development targets previously SHA-bound by R5; exact R5 folds0/1 candidate rows define the development population, while R5 target identity and R9 attribution PASS are mandatory inputs.
- Route branch: `codex/haze4k-v5-r10-fixed-region-action-feasibility-20260719`.
- Locked test/canary policy: confirmation images/targets/outcomes, historical protected outcomes, canary and locked test are prohibited; all permissions remain false.

## Scientific Contract

- Population and analysis/grouping unit: the same 384 development clean-image groups in R9 folds 0/1; D_ref/D_rep are paired repeated measurements, one normalized 8x8 tile coordinate is the fixed regional unit, and one clean-image name is the bootstrap unit.
- Intervention or factor contrast and reference: for every tile, choose one common action for D_ref/D_rep from no-op, positive-full and negative-full. An active action is eligible only if its exact tile MSE is no worse than no-op under both operators; among eligible actions choose the largest worse-operator local PSNR gain, with ties favoring no-op, positive, then negative. Compare this region oracle with the best whole-image action satisfying nonnegative image gain under both operators and with the mean of 16 name-seeded permutations of the same regional action histogram.
- Primary outcome, direction and aggregation: larger worse-operator population mean PSNR gain, region-minus-safe-global gain, region-minus-spatial-shuffle gain and region-minus-global CVaR5 are better; zero severe (`<=-0.2 dB`) and hard (`<=-0.5 dB`) region-oracle image events are required. Four thousand paired clean-image bootstrap draws retain operators and choose the conservative operator/contrast inside each draw.
- Preferred mechanism and strongest competing explanation: preferred mechanism is genuine within-image cancellation—different regions require no-op, positive and negative actions—while the competitor is that safe whole-image action identity already captures the useful headroom and apparent spatial structure is action-histogram or oracle noise.
- Evidence roles and candidate/freeze point: R9 is formal for its own post-hoc diagnostic and authorizes only this contract review. R10 is a new category-C privileged development feasibility audit. Cache/target identities, folds, grid, action bank, common-map rule, non-degrade constraint, controls, thresholds, bootstrap and terminal mapping freeze at the R10 route commit. R10 cannot change R5-R9 terminals.
- Primary gate, uncertainty and threshold source: PASS requires exact R5 whole-image target replay, complete 384x2 units, exact nonoverlapping 8x8 partitions, every selected active tile nonworse under both operators, region gain LCB95 `>=+0.020 dB`, region-minus-safe-global and region-minus-shuffle LCB95 each `>=+0.005 dB`, region-minus-global CVaR5 LCB95 `>=-0.005 dB`, zero region severe/hard images, exact-binomial mixed no-op/active image-fraction LCB95 `>=0.25`, and bidirectional positive/negative image-fraction LCB95 `>=0.10`. Margins predate R10 in R4/R5/R9; fractions are prespecified material-support thresholds.
- `PASS` authorizes: `R10_REGION_OBSERVABILITY_CONTRACT_REVIEW_ONLY`; no model training, inference, policy, confirmation or execution of a next route.
- `INCONCLUSIVE` authorizes: `NONE`; identity, completeness, target replay, partition, finite-value, local safety, positive-reference or interval failures stop as `R10_A0_INPUT_OR_FEASIBILITY_INCONCLUSIVE_STOP`.
- `FAIL` stops: valid evidence with any scientific gate decisively excluded returns `R10_A0_FIXED_REGION_ACTION_FEASIBILITY_FAIL_STOP / NONE`; do not search grid size, smoothing, tile budget, action magnitude, thresholds, seed or subgroup.

## Implementation Contract

- Exact change and disabled mechanisms: one deterministic CPU-only replay of existing cached no-op/positive/negative renders, exact float64 nonoverlapping tile SSE, fixed safe map, matched whole-image and permutation controls, grouped bootstrap and frozen gates; disable ConvIR/model construction, candidate generation, fitting, calibration, semantic/subgroup selection and adaptive grid search.
- Checkpoint/load/init/freeze contract: no checkpoint or trainable parameter is loaded; no random model initialization. Only fixed bootstrap seed 3407 and 16 SHA-256 name-seeded tile permutations are used.
- Input whitelist and prohibited inputs: allow the exact cache manifest/raw-unit manifest, per-unit cache files and development clean targets, R5 folds0/1 candidate-score rows for population and target replay, and R5/R9 typed closeouts. Do not open the broader R3 ledger. Prohibit confirmation identities/images/targets/outcomes, canary/locked inputs, semantic labels, R5 predictions as decision inputs, per-seed reconstruction and any new candidate.
- Dataset/split/preprocessing/metric identities: folds 0/1 only, 192 names each, D_ref/D_rep, historical add-and-clamp render, R5 label decode/crop, RGB `[0,1]`, exact floor-boundary 8x8 nonoverlapping tiles covering every pixel once, image PSNR from composite RGB SSE, no exclusion.
- Matched baseline and budget: region/global/shuffle share exact images, operators, three actions and rendered pixels. The global baseline chooses one dual-operator-safe action per image; each shuffle preserves the region map's exact action histogram. Work is fixed at 768 cache units, 64 tiles, 16 shuffles and 4,000 bootstrap draws.
- Resource/cost limits or descriptive-only rationale: CPU-only cached replay; expected 900 seconds, hard timeout 1,800 seconds, no resume. The contract exercises 384 synthetic groups, two operators, 64 tiles, 16 shuffles and 4,000 draws at the formal asymptotic scale.
- Runner and required assets: unchanged generic runner; cache manifest `55350511...`, raw manifest `3123f2da...`, cache directory, Haze4K development data, R5 candidate scores `53061bea...`, R5 closeout `e8d6151a...`, and R9 closeout `639820a2...`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `route_runtime_specs/R10_A0_FIXED_REGION_ACTION_FEASIBILITY_AUDIT.json` and `tools/r10_a0_fixed_region_action_feasibility.py`.
- Representative engineering fixture or metadata-only exemption: full-scale protected-data-free tile-loss fixture exercises common-map safety, global baseline, 16 shuffles, action heterogeneity, grouped bootstrap, deterministic finalizer and bounded work without creating workload output.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R10_A0_FIXED_REGION_ACTION_FEASIBILITY_AUDIT` | privileged post-hoc development feasibility | fixed-region materiality, spatial specificity, direction heterogeneity and dual-operator safety | `R10_REGION_OBSERVABILITY_CONTRACT_REVIEW_ONLY` |

- First operation: R10_A0_FIXED_REGION_ACTION_FEASIBILITY_AUDIT
- Expected wall time and monitor profile: 900 seconds expected, 1,800 seconds hard timeout, `standard` profile.
- Complete-unit resume policy: `none`.
- Cloud workspace/run/output/status/closeout: fresh workspace; output `r10-a0-fixed-region-feasibility-r1`; closeout `r10_a0_fixed_region_action_feasibility_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract/provenance/input/replay/action-distribution/feasibility/bootstrap/shuffle/operator/shape/gate/resource evidence, typed closeout, one conclusion and terminal index. Per-image region rows and action maps remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. R5-R9 terminals remain unchanged.
