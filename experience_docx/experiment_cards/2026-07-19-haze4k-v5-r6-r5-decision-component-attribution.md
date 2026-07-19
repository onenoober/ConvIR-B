# Haze4K v5 R6 Frozen R5 Decision-Component Attribution Audit

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r6_r5_decision_component_attribution_20260719`.
- Question: at R5's fixed 20% per-fold coverage, which frozen decision component--action identity, coverage ranking, or explicit severe-risk veto--accounts for the recoverable utility and tail-safety gap?
- Rules commit: `github/main@058175c395c0c12d717919f67b56a3b93dc1321d`.
- Source branch/commit: current project memory `github/main@058175c395c0c12d717919f67b56a3b93dc1321d`; R5 formal route commit `7e75eed504b2ead65a1971ec250dc7f59a79574d` is an immutable upstream evidence identity, not a branch base.
- Route branch: `codex/haze4k-v5-r6-r5-decision-component-attribution-20260719`.
- Locked test/canary policy: confirmation images, targets and outcomes, historical protected outcomes, canary, and locked test are prohibited; every corresponding access flag must remain false.

## Scientific Contract

- Population and analysis/grouping unit: the 384 clean-image groups evaluated by R5-R2 on frozen outer folds 0/1; D_ref and D_rep are paired repeated measurements, actions are paired candidates, and the independent resampling unit is one clean-image name.
- Intervention or factor contrast and reference: a frozen `2 x 2 x 2` counterfactual decomposition over action identity (`A`: R5 q05 action versus true worse-operator best active action), coverage ranking (`C`: R5 robust-q05 rank versus true worse-operator utility of the action assigned by `A`), and risk control (`R`: no veto versus veto when that assigned action has gain `<=-0.2 dB` under either operator, followed by safe backfill in the same frozen rank order). Cells are `P000` through `P111`; `P000` must exactly replay R5 and `P111` is a diagnostic upper bound, never a deployable policy.
- Primary outcome, direction and aggregation: larger worse-operator population mean PSNR gain, R5 whole-population oracle retention, and fraction of the `P111-P000` regret gap recovered are better; fewer selected severe (`<=-0.2 dB`) and hard (`<=-0.5 dB`) clean-image groups and larger worse-operator CVaR5 are safer. Every cell selects exactly `ceil(0.20*192)=39` names per fold. Uncertainty uses 4,000 paired clean-image grouped bootstrap draws with D_ref/D_rep retained and the worse operator selected inside each draw.
- Preferred mechanism and strongest competing explanation: the preferred explanation is a coverage/readout conflict that fails to convert candidate-conditioned utility and risk signals into safe action; competitors are primarily wrong signed action identity, primarily wrong coverage ordering, and an interaction/aggregation failure for which no single component suffices.
- Evidence roles and candidate/freeze point: the R5 typed closeout and compact formal results are category A formal confirmation for the old R5 terminal only; the three R5 cloud-only CSVs are category B raw support; this R6 operation is category C `post-hoc mechanism diagnostic` carried in the runtime schema as `development_screening`; the metadata factcheck is category D engineering evidence. R6 cannot change the R5 terminal decision or authorize deployment, full OOF, confirmation, canary, locked test, training, or inference. Inputs, hashes, folds, primary cell, factors, thresholds, coverage, metrics, bootstrap, and terminal mapping freeze at the R6 route commit.
- Primary gate, uncertainty and threshold source: structure first requires the three SHA-bound R5-R2 raw CSVs, lifecycle identity, 384 names, folds 0/1, two operators, two active actions, both seeds, complete `S1_TRUE_SPATIAL_RESPONSE`, exact seed averaging, and exact replay of R5 action/coverage rows, operator gains, retention interval, 10 severe and 3 hard groups. A single or pair cell is utility-attributable when delta-gain LCB95 is at least `+0.005 dB`, regret-recovery LCB95 is at least `0.25`, severe/hard counts do not exceed `P000`, and CVaR5-delta LCB95 is at least `-0.005 dB`. It is safety-attributable when severe and hard counts are both zero while mean-gain and CVaR5 deltas each have LCB95 at least `-0.005 dB`. These materiality/safety thresholds predate R6 in R4/R5; no R6 outcome may change them.
- `PASS` authorizes: `R6_NEXT_CONTRACT_REVIEW_ONLY` when at least one preregistered single (`P100`, `P010`, `P001`) or pair (`P110`, `P101`, `P011`) is utility- or safety-attributable. PASS means component localization only. It does not validate a deployable policy or permit automatic execution.
- `INCONCLUSIVE` authorizes: `NONE`; input/base replay failure, incomplete safe backfill, non-finite results, nonpositive/unstable oracle gap, or intervals crossing every attribution boundary stops as `R6_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP`.
- `FAIL` stops: if structure and the oracle gap are valid but every single and pair is decisively below both utility and safety attribution gates, return `R6_A0_NO_SINGLE_OR_PAIR_COMPONENT_ATTRIBUTION_STOP / NONE`. No threshold, coverage, subgroup, seed, representation, action-bank, or architecture neighbor search is permitted.

## Frozen Factorial And Decision Meaning

| Cell | Action `A` | Coverage `C` | Risk `R` | Role |
| --- | --- | --- | --- | --- |
| `P000` | predicted q05 | predicted q05 | none | exact R5 base replay |
| `P100` | true utility | predicted q05 of assigned action | none | action-only replacement |
| `P010` | predicted q05 | true utility of assigned action | none | coverage-only replacement |
| `P001` | predicted q05 | predicted q05 | true severe veto/backfill | risk-only replacement |
| `P110` | true utility | true utility of assigned action | none | action-coverage pair |
| `P101` | true utility | predicted q05 of assigned action | true severe veto/backfill | action-risk pair |
| `P011` | predicted q05 | true utility of assigned action | true severe veto/backfill | coverage-risk pair |
| `P111` | true utility | true utility of assigned action | true severe veto/backfill | diagnostic ceiling only |

The three two-factor additive interactions are frozen as `P110-P100-P010+P000`, `P101-P100-P001+P000`, and `P011-P010-P001+P000`. They describe coupling and cannot rescue an unqualified cell by post-hoc relabeling. Descriptive 10/30/40/60/100% curves are not recomputed because R6 has one primary coverage and forbids coverage search.

## Implementation Contract

- Exact change and disabled mechanisms: add one deterministic CSV replay/attribution entrypoint, eight fixed counterfactual cells, paired bootstrap, interaction summaries, and compact gates; disable all training, model construction, checkpoint loading, candidate generation, image decoding, inference, threshold fitting, calibration fitting, subgroup selection, protected-data access, and result-dependent reruns.
- Checkpoint/load/init/freeze contract: no checkpoint or model is loaded and no random model initialization exists. The only stochastic operation is the frozen NumPy bootstrap RNG seed `3407`; the action/rank tie break is R5's SHA-256 lexical key.
- Input whitelist and prohibited inputs: allow only the exact R5-R2 lifecycle identity, three SHA-bound raw CSVs, and hash-bound R5 closeout/cell/bootstrap/gate/risk-coverage summaries. Prohibit images, clean RGB, candidate tensors, region arrays, semantic labels, filenames as learned features, other cells as a rescue analysis, confirmation, canary, locked test, and any unregistered outcome source. Names are grouping/tie identities only, never predictive inputs.
- Dataset/split/preprocessing/metric identities: R5 folds 0/1, 192 names per fold, paired D_ref/D_rep, active positive/negative full actions, `S1_TRUE_SPATIAL_RESPONSE`, saved target PSNR gain, severe `<=-0.2 dB`, hard `<=-0.5 dB`, no exclusions, and population gain zero for abstained names.
- Matched baseline and budget: all eight cells share identical rows, fold sizes, action bank, 39-per-fold coverage, tie breaking, paired operators, metrics and bootstrap draws. Exactly one component bit changes in a single replacement; no hyperparameter or candidate selection occurs.
- Resource/cost limits or descriptive-only rationale: CPU-only read/replay of 21,504 compact CSV rows plus 4,000 grouped bootstrap draws; expected wall time 180 seconds and hard timeout 900 seconds. The CPU contract exercises the exact eight-cell construction, veto/backfill, metrics and formal-size 384-group bootstrap path on protected-data-free synthetic rows.
- Runner and required assets: unchanged `experience_docx/tools/run_route_operation.sh`; R5 lifecycle identity SHA-256 `8fc05944...`, per-seed CSV `6cfcea93...`, candidate-score CSV `53061bea...`, policy-row CSV `63c1aa4c...`, closeout `e8d6151a...`, cell summary `8498491e...`, bootstrap summary `2e5154ff...`, gate summary `b8a28db1...`, and risk-coverage summary `caf43a17...`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `experience_docx/route_runtime_specs/R6_A0_FROZEN_R5_DECISION_COMPONENT_ATTRIBUTION_AUDIT.json` and `experience_docx/tools/r6_a0_r5_decision_component_attribution.py`.
- Representative engineering fixture or metadata-only exemption: `contract(context_path)` creates 384 synthetic groups with the production row schema and runs the exact action/rank/veto/backfill, cell metric, and bounded bootstrap functions; it validates eight cells, 39 selections per fold, zero risk after oracle veto, deterministic results, finite outputs, and projected workload cost without creating `workload/`.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R6_A0_FROZEN_R5_DECISION_COMPONENT_ATTRIBUTION_AUDIT` | category C post-hoc mechanism diagnostic over R5 development OOF rows | identity/base replay then frozen single/pair attribution | `R6_NEXT_CONTRACT_REVIEW_ONLY` |

- First operation: R6_A0_FROZEN_R5_DECISION_COMPONENT_ATTRIBUTION_AUDIT
- Expected wall time and monitor profile: 180 seconds expected, 900 seconds hard timeout, `short` monitor profile, one bounded startup observation and one finish observation.
- Complete-unit resume policy: `none`; interruption or engineering failure cannot be retried without a new output identity and the applicable engineering review.
- Cloud workspace/run/output/status/closeout: fresh route workspace; output id `r6-a0-r5-decision-attribution-r2`; generic `status.txt`, `heartbeat.json`, and `runtime.log`; closeout `r6_a0_decision_component_attribution_closeout.json`.
- Same-contract engineering repair: r1 stopped in the protected-data-free CPU contract because its synthetic fixture made the diagnostic ceiling gap nonpositive; r2 changes only that fixture's predicted q05 ordering so the unchanged production attribution/finalizer path has a strictly positive test gap.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract/provenance/input/base-replay/factorial/bootstrap/component/interaction/risk/gate/resource summaries, typed closeout, one scientific conclusion, and terminal index row. The three R5 raw inputs, R6 per-image factorial policy rows, logs, bootstrap draws, datasets, images, tensors and arrays remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`

The card is immutable after launch. Do not append terminal results here. R5 remains
`COMPLETED_GATE_FAIL / R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP / NONE`.
This post-hoc audit cannot change that terminal tuple. Write the single R6
scientific interpretation to the conclusion JSON required by
`SCIENCE_FASTPATH.md`; the typed closeout remains R6 terminal authority.
