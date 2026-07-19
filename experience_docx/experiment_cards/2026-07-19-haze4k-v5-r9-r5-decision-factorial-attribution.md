# Haze4K v5 R9 Frozen R5 Decision Factorial Attribution

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r9_r5_decision_factorial_attribution_20260719`.
- Question: at R5's fixed 20% per-fold coverage, is its utility/safety gap primarily attributable to action identity, coverage ranking, explicit severe-risk veto, or a prespecified pair interaction?
- Rules commit: `github/main@34d0bb6ffe30f28feb8610f32ce1e2d14f54a750`.
- Source branch/commit: authoritative R5 scores and policy are bound to `r5-a0-spatial-response-screen-r2@7e75eed504b2ead65a1971ec250dc7f59a79574d`; R8 formally verifies their score, label and policy identity.
- Route branch: `codex/haze4k-v5-r9-r5-decision-factorial-20260719`.
- Locked test/canary policy: confirmation images/targets/outcomes, historical protected outcomes, canary and locked test are prohibited; all access flags remain false.

## Scientific Contract

- Population and analysis/grouping unit: the 384 R5 development clean-image groups in folds 0/1; D_ref/D_rep and active actions are paired repeated measurements, and the bootstrap unit is one clean-image name.
- Intervention or factor contrast and reference: frozen `2 x 2 x 2` factorial over action identity (R5 saved robust-q05 action versus true worse-operator best active action), coverage ranking (R5 saved robust-q05 rank versus true worse-operator utility of the assigned action), and risk control (no veto versus veto when the assigned action is severe under either operator, followed by safe backfill in the same frozen rank order). Cells are `P000` through `P111`; `P000` exactly replays R5 and `P111` is a diagnostic ceiling only.
- Primary outcome, direction and aggregation: larger worse-operator population mean PSNR gain, R5 whole-population oracle retention, and recovered fraction of the `P111-P000` gain gap are better; fewer selected severe (gain `<=-0.2 dB`) and hard (`<=-0.5 dB`) groups and larger worse-operator CVaR5 are safer. Each cell selects exactly 39 names per fold. Four thousand paired name-group bootstrap draws retain operators and choose the worse operator inside each draw.
- Preferred mechanism and strongest competing explanation: preferred explanation is coverage/readout conflict that fails to convert candidate-conditioned signals into safe action; competitors are wrong signed action identity, missing risk veto, and multi-component interaction/aggregation failure.
- Evidence roles and candidate/freeze point: runtime role `development_screening`. R5/R6/R7/R8 closeouts are category A evidence for their own routes; R5 raw candidate/policy rows are category B support; R9 is category C post-hoc mechanism diagnostic. R9 cannot change earlier terminal decisions. Inputs, hashes, cells, factor definitions, fixed coverage, thresholds, metrics, bootstrap and terminal mapping freeze at the R9 route commit.
- Primary gate, uncertainty and threshold source: structure requires exact SHA-bound R5 candidate/policy rows, R5/R8 closeouts, 384 names, folds 0/1, D_ref/D_rep, both active actions, complete `S1_TRUE_SPATIAL_RESPONSE`, exact `P000` policy/metric replay, 78 selected groups, 10 severe and 3 hard. A single or pair cell is utility-attributable when gain-delta LCB95 `>=+0.005 dB`, regret-recovery LCB95 `>=0.25`, severe/hard do not exceed P000, and CVaR5-delta LCB95 `>=-0.005 dB`. It is safety-attributable when severe/hard are zero and gain/CVaR5 delta LCB95 values are each `>=-0.005 dB`. These thresholds predate R9 in R4/R5.
- `PASS` authorizes: `R9_NEXT_CONTRACT_REVIEW_ONLY` if at least one preregistered single (P100/P010/P001) or pair (P110/P101/P011) is utility- or safety-attributable. PASS is localization only; it does not validate deployment or regional recovery.
- `INCONCLUSIVE` authorizes: `NONE`; incomplete identity/base replay, non-finite estimates, nonpositive diagnostic gap, unsafe backfill, or intervals crossing all attribution boundaries stops as `R9_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP`.
- `FAIL` stops: valid inputs with every single and pair decisively below both attribution gates return `R9_A0_NO_SINGLE_OR_PAIR_ATTRIBUTION_STOP / NONE`; no threshold, coverage, subgroup, seed, representation or architecture search.

## Frozen Cells And Interactions

| Cell | Action | Coverage | Risk | Role |
| --- | --- | --- | --- | --- |
| P000 | saved q05 | saved q05 | none | exact R5 base |
| P100 | true utility | saved q05 | none | action-only |
| P010 | saved q05 | true utility | none | coverage-only |
| P001 | saved q05 | saved q05 | severe veto/backfill | risk-only |
| P110 | true utility | true utility | none | action+coverage |
| P101 | true utility | saved q05 | severe veto/backfill | action+risk |
| P011 | saved q05 | true utility | severe veto/backfill | coverage+risk |
| P111 | true utility | true utility | severe veto/backfill | diagnostic ceiling |

Interactions are frozen as `P110-P100-P010+P000`, `P101-P100-P001+P000`, and `P011-P010-P001+P000`. They are descriptive coupling estimates and cannot rescue an unqualified cell.

## Implementation Contract

- Exact change and disabled mechanisms: one deterministic CPU-only replay of the eight cells, paired bootstrap, component/interaction/risk summaries and frozen gates; disable training, inference, candidate generation, per-seed reconstruction, image/tensor/cache access, calibration fitting, threshold/coverage search and subgroups.
- Checkpoint/load/init/freeze contract: no checkpoint/model/random parameter initialization; only bootstrap RNG seed 3407. Authoritative ensemble scores are read directly, as required by R8 PASS.
- Input whitelist and prohibited inputs: allow exact R5 candidate-score and policy CSVs plus hash-bound R5 compact closeout/cell/bootstrap/gate and R8 closeout; prohibit per-seed reconstruction, images, tensors, semantic labels, protected roles and result-dependent inputs.
- Dataset/split/preprocessing/metric identities: folds 0/1, 192 names each, D_ref/D_rep, two active actions, primary S1 cell, saved target PSNR gain, any-operator severe and hard events, no exclusions, no-op gain zero.
- Matched baseline and budget: all cells share rows, folds, action bank, 39-per-fold coverage, tie rule, operators, metrics and 4,000 draws; a single replacement changes one bit only.
- Resource/cost limits or descriptive-only rationale: CPU-only audit of 6,144 candidate rows and 3,072 policy rows plus 4,000 paired draws; expected 180 seconds, hard timeout 900 seconds. Contract exercises the exact eight-cell and bootstrap path on 384 synthetic groups.
- Runner and required assets: unchanged generic runner; R5 candidate scores `53061bea...`, policy rows `63c1aa4c...`, closeout `e8d6151a...`, cell summary `8498491e...`, bootstrap `2e5154ff...`, gate `b8a28db1...`, and R8 closeout `4254840b...`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `route_runtime_specs/R9_A0_FROZEN_R5_DECISION_FACTORIAL_ATTRIBUTION_AUDIT.json` and `tools/r9_a0_r5_decision_factorial_attribution.py`.
- Representative engineering fixture or metadata-only exemption: full 384-group synthetic fixture validates eight cells, fixed fold coverage, risk backfill, the complete 4,000-draw paired bootstrap, deterministic finalizer and bounded work without creating workload output.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R9_A0_FROZEN_R5_DECISION_FACTORIAL_ATTRIBUTION_AUDIT` | post-hoc development mechanism diagnostic | exact base replay and frozen single/pair attribution | `R9_NEXT_CONTRACT_REVIEW_ONLY` |

- First operation: R9_A0_FROZEN_R5_DECISION_FACTORIAL_ATTRIBUTION_AUDIT
- Expected wall time and monitor profile: 180 seconds expected, 900 seconds hard timeout, `short` profile.
- Complete-unit resume policy: `none`.
- Cloud workspace/run/output/status/closeout: fresh workspace; output `r9-a0-r5-decision-factorial-r1`; closeout `r9_a0_decision_factorial_attribution_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract/provenance/input/base/factorial/bootstrap/component/interaction/risk/gate/resource, typed closeout, conclusion and terminal index. Raw R5 inputs and R9 per-image policies remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`

The card is immutable after launch. R5/R6/R7/R8 terminals remain unchanged.
