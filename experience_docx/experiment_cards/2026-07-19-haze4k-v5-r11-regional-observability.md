# Haze4K v5 R11 Regional Action Observability

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r11_regional_action_observability_20260719`.
- Question: on the exact R10 development population, can a fixed local candidate-response representation recover material, dual-operator-safe regional action ranking out of fold when the R10 per-image action budget is held privileged and fixed?
- Rules commit: `github/main@9f6800e278a4c15b81dadf0ec53d3c987987631d`.
- Source branch/commit: exact SHA-bound R3 candidate cache, Haze4K development targets, R5 fold rows and the terminal R10 contract/evidence archived on the rules commit; no ConvIR architecture or checkpoint is changed.
- Route branch: `codex/haze4k-v5-r11-regional-observability-20260719`.
- Locked test/canary policy: confirmation identities/images/targets/outcomes, canary and locked test are prohibited; all permissions remain false.

## Scientific Contract

- Population and analysis/grouping unit: the same 384 clean-image groups in R10 folds 0/1, with 192 names per fold; D_ref/D_rep and the two active actions are paired repeated measurements, one fixed 8x8 tile/action row is the fitting unit, and one clean-image name is the split, replay and bootstrap unit. Fold 1 trains the fold-0 model and fold 0 trains the fold-1 model; no image or tile crosses its outer split.
- Intervention or factor contrast and reference: the primary `L1_LOCAL_CANDIDATE_CONTEXT` probe sees fixed 3x3 neighborhoods of per-tile no-op state and positive/negative candidate-response statistics from both operators, image-global summaries, normalized coordinates and action sign. It is compared with parameter-identical `P0_POOLED_CANDIDATE_RESPONSE`, `S2_WITHIN_IMAGE_RESPONSE_SHUFFLE` and `G0_LOCAL_STATE_ONLY` controls. At replay, each image and equal-pixel-count tile stratum receives the exact no-op/positive/negative slot counts of the R10 oracle, but all slot locations and signed identities are assigned only from OOF scores.
- Primary outcome, direction and aggregation: larger OOF worse-operator mean image PSNR gain, R10-oracle retention, primary-minus-control gains, primary-minus-safe-global mean gain and primary-minus-safe-global CVaR5 are better; zero severe (`<=-0.2 dB`) and hard (`<=-0.5 dB`) primary image events are required. Four thousand paired clean-image bootstrap draws retain folds, operators, actions and controls and choose the conservative operator/contrast inside each draw.
- Preferred mechanism and strongest competing explanation: preferred mechanism is that aligned local candidate responses contain observable signed regional utility beyond generic local state and image-level action summaries. The strongest competitor is that R10 headroom is target-only or structurally unidentifiable from the frozen deployable features; weaker competitors are global action identity, response-histogram information and generic spatial difficulty.
- Evidence roles and candidate/freeze point: R10 is terminal category-C privileged development evidence and authorizes only this contract review. R11 is a new `development_screening` OOF representation/ranking probe. Population, labels, features, cells, folds, seeds, optimizer, action-budget assignment, metrics, thresholds, uncertainty and terminal actions freeze at the R11 route commit. R11 cannot change R5-R10 terminals.
- Primary gate, uncertainty and threshold source: PASS requires exact R10 oracle/target/action-distribution replay; complete 384x2 units and 49,152 paired tile/action rows; primary OOF gain LCB95 `>=+0.020 dB`; R10-oracle retention LCB95 `>=0.25`; primary-minus-pooled, primary-minus-shuffle, primary-minus-generic and primary-minus-safe-global mean-gain LCB95 each `>=+0.005 dB`; primary-minus-safe-global CVaR5 LCB95 `>=-0.005 dB`; zero primary severe/hard images; both fold point estimates meeting the same absolute and four incremental mean-gain lines; every fold-by-seed primary gain nonnegative; and the pooled seed-specific gain range `<=0.020 dB`. These margins predate R11 in R4/R5/R9/R10.
- `PASS` authorizes: `R11_SAFE_REGIONAL_COVERAGE_CONTRACT_REVIEW_ONLY`; it does not authorize a coverage run, restoration-model training, inference, policy execution or protected-data access.
- `INCONCLUSIVE` authorizes: `NONE`; identity, completeness, exact-replay, finite-value, interval or protected-access failures stop as `R11_A0_INPUT_OR_OBSERVABILITY_INCONCLUSIVE_STOP`.
- `FAIL` stops: valid evidence that decisively misses a scientific line, any severe/hard event, fold failure, seed sign reversal or unstable seed range returns `R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP / NONE`. Do not search neighboring features, grids, contexts, heads, widths, depths, seeds, epochs, losses, thresholds, action budgets, coverage rates or subgroups.

## Independent Information And Decision Boundary

R5 used image/action-level 8x8 DCT response features and a fixed 20% whole-image coverage policy. It found material mean signal but failed with 10 severe and three hard images. R9 then localized the largest recoverable whole-image loss to coverage/readout conflict. R10 separately proved a large target-selected 8x8 regional ceiling. R11 does not widen the R5 MLP or tune its policy. It changes the statistical unit and target to fixed tile/action robust gain, introduces aligned local response neighborhoods, and holds action quantity to the exact R10 oracle budget so the primary estimand is ranking/action observability rather than coverage learning.

The oracle budget is leakage-ineligible privileged information. It makes R11 a mechanism screen, not a deployable policy. PASS means only that observable features can place and sign a fixed amount of regional intervention. Coverage, abstention and safety calibration remain a separate question.

## Frozen Representation And Readout

For every cached name/operator unit, reproduce the historical no-op, positive-full and negative-full renders. On each exact R10 tile, compute six no-op-state channels (RGB mean/std) and nine candidate-response channels (RGB mean/std/absolute mean). Pair D_ref/D_rep, forming 12 state and 18 response channels. The primary feature is the reflected 3x3 neighborhood of both maps, their 30 image-global means, two normalized tile coordinates and action sign (`303` scalars).

| Cell | Local state | Local response | Response alignment | Role |
| --- | --- | --- | --- | --- |
| `L1_LOCAL_CANDIDATE_CONTEXT` | true | true | true | primary |
| `P0_POOLED_CANDIDATE_RESPONSE` | true | image mean | none | image/action summary control |
| `S2_WITHIN_IMAGE_RESPONSE_SHUFFLE` | true | true | equal-area deterministic shuffle | response-layout control |
| `G0_LOCAL_STATE_ONLY` | true | zero | none | generic local-difficulty control |

A common training-fold normalizer is fitted on primary features and applied to all cells. Each fixed `303 -> 64 ReLU -> 2` MLP predicts local mean gain and worst-operator local gain. Their standardized Huber losses have equal weight. Seeds 3407/3411, 24 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch 256 and no early stopping/checkpoint selection are frozen. Ensemble worst-operator scores assign the R10 slots by exact SciPy linear-sum maximum-weight matching within equal-pixel-count strata; rows use ascending tile index and columns use no-op, positive, negative slot order, fixing exact-tie behavior. The `+0.005 dB` eligible label is used only for prespecified calibration reporting, not as a fitted third head or tunable gate.

## Implementation Contract

- Exact change and disabled mechanisms: add one deterministic CPU-only cache replay, fixed local-statistic extractor, four matched OOF probe cells, exact oracle-budget assignment, image replay, grouped bootstrap and typed finalizer. Disable ConvIR construction, candidate generation, restoration training/inference, semantic/subgroup selection, adaptive features, hyperparameter search, threshold calibration and protected roles.
- Checkpoint/load/init/freeze contract: no checkpoint or restoration parameter is loaded; cache tensors are immutable and each file hash is checked. Only the small diagnostic MLP initializes under seeds 3407/3411.
- Input whitelist and prohibited inputs: allow the exact cache/raw manifests, cache units, development targets for outcome labels only, R5 population/fold rows and R10 compact closeout/gate/action-distribution evidence. Learned inputs may contain only rendered no-op state, candidate response, coordinate and action sign. Prohibit clean targets/errors/oracle actions/gains as features, filename/fold/operator identifiers as features, saved R5 predictions/scores, broader R3 ledger, semantic labels, confirmation, canary and locked test.
- Dataset/split/preprocessing/metric identities: folds0/1 only, paired D_ref/D_rep, historical add-and-clamp render, R5 label decode/crop, RGB `[0,1]`, exact floor-boundary 8x8 tiles, float64 tile SSE, robust local gain as the minimum across operators, composite image PSNR and no exclusion.
- Matched baseline and budget: all four cells share rows, input width, primary normalizer, model, seeds, data order, epochs, optimizer, action slots, renderer, bootstrap and gates. Shuffle uses one name/action-seeded permutation within equal-pixel-count strata, preserving paired response channels and exact response/pixel-area distributions. Work is fixed at 768 cache units, 16 models, 24 epochs and 4,000 draws.
- Resource/cost limits or descriptive-only rationale: CPU-only cached diagnostic; expected 3,600 seconds, hard timeout 7,200 seconds, no resume. The protected-data-free contract uses 384 synthetic groups, 49,152 rows, 303 features, four cells, two folds, two seeds and a formal-size one-epoch timing/memory probe.
- Runner and required assets: unchanged generic runner; cache manifest `55350511...`, raw manifest `3123f2da...`, candidate-cache directory, Haze4K development directory, R5 fold rows `53061bea...`, and receipt-bound R10 closeout/gate/action-distribution files.
- Runtime spec and `contract --context` / `run --context` entrypoint: `experience_docx/route_runtime_specs/R11_A0_REGIONAL_ACTION_OBSERVABILITY_SCREEN.json` and `experience_docx/tools/r11_a0_regional_action_observability.py`.
- Representative engineering fixture or metadata-only exemption: the full production feature transforms, shared normalization, MLP/loss/optimizer, exact-slot assignment, replay, grouped bootstrap and finalizer run on a protected-data-free formal-shape synthetic fixture; no exemption.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R11_A0_REGIONAL_ACTION_OBSERVABILITY_SCREEN` | development OOF mechanism screen | local response observability, matched-control specificity, exact-budget safety and fold/seed stability | `R11_SAFE_REGIONAL_COVERAGE_CONTRACT_REVIEW_ONLY` |

- First operation: R11_A0_REGIONAL_ACTION_OBSERVABILITY_SCREEN
- Expected wall time and monitor profile: 3,600 seconds expected, 7,200 seconds hard timeout, `standard` profile.
- Complete-unit resume policy: `none`; an interruption requires typed engineering review and a new output identity without scientific changes.
- Cloud workspace/run/output/status/closeout: fresh workspace; output `r11-a0-regional-observability-r1`; closeout `r11_a0_regional_action_observability_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract/provenance/input/representation/oracle/control/fold-seed/calibration/operator/bootstrap/gate/resource evidence, typed closeout, one conclusion and terminal index. Per-image maps, tile features, predictions, model states, arrays and logs remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. R5-R10 terminals remain unchanged.
