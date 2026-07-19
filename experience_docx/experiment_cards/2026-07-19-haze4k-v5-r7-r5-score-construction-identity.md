# Haze4K v5 R7 Frozen R5 Score-Construction Identity Audit

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r7_r5_score_construction_identity_20260719`.
- Question: does the exact R5 source-code numerical contract--float32 seed predictions followed by `torch.stack(seed_predictions).mean(0)`--reproduce every saved ensemble score and the resulting four-cell fixed-coverage policy, and is the R6 mismatch explained by using Python double-precision averaging?
- Rules commit: `github/main@1d316d88e921aefac9524344f7126817191c5432`.
- Source branch/commit: R5 source semantics are frozen at `github/codex/haze4k-v5-r5-independent-route-contract-20260719@7e75eed504b2ead65a1971ec250dc7f59a79574d`; R5 and R6 terminal evidence are frozen on GitHub main.
- Route branch: `codex/haze4k-v5-r7-r5-score-construction-identity-20260719`.
- Locked test/canary policy: confirmation images, targets and outcomes, historical protected outcomes, canary, and locked test are prohibited; all corresponding access flags must remain false.

## Scientific Contract

- Population and analysis/grouping unit: all 12,288 R5 per-seed prediction rows, 6,144 R5 ensemble candidate-score rows, and 3,072 R5 policy rows from the 384 development clean-image groups in folds 0/1; each `(cell, fold, name, operator, action)` score is one reconstruction unit and one clean-image name is the policy grouping unit.
- Intervention or factor contrast and reference: reconstruct mean, q05 and severe scores using the authoritative source-code semantics `float32 tensor -> stack -> mean(0)`; compare against saved candidate scores. A Python float64 arithmetic mean is a prespecified sensitivity control only. Rebuild the R5 common action, SHA-tied top-39-per-fold coverage, oracle, and operator rows for all four cells.
- Primary outcome, direction and aggregation: exact native-float32 score match count, maximum absolute mismatch, exact policy-field replay, action-assignment changes, top-39 symmetric differences, and exact cell-level selected/severe/hard/negative counts and operator gains. No statistical hypothesis test or bootstrap is needed because the full frozen finite population is audited.
- Preferred mechanism and strongest competing explanation: preferred explanation is that R6 used a numerically different float64 reconstruction while R5 used float32 Torch reduction; the strongest competitor is a genuine inconsistency between saved per-seed scores, saved ensemble scores, and the R5 policy.
- Evidence roles and candidate/freeze point: the runtime schema role is `development_screening`. R5/R6 typed closeouts are category A terminal evidence for their own routes; the R5 cloud-only rows are category B raw support; R7 is a category D integrity audit with a category C post-hoc implication for whether a new attribution contract may be reviewed. It cannot change R5 or R6 terminal decisions. All source semantics, hashes, keys, row counts, folds, cells, actions, tie rule, coverage and terminal mapping freeze at the R7 route commit.
- Primary gate, uncertainty and threshold source: structure requires the exact SHA-bound R5-R2 lifecycle, two seeds `3407/3411`, folds 0/1, four cells, two operators, two active actions, 12,288/6,144/3,072 rows, unique complete keys and finite values. PASS requires zero exact mismatches for all three native-float32 ensemble fields, exact target/label identity, exact replay of every saved policy field, exactly 39 selected names per fold/cell, and exact reproduction of every R5 cell-summary count and operator gain. Float64 mismatch prevalence and downstream action/coverage changes are descriptive sensitivity results and cannot change the authoritative PASS if native semantics are exact.
- `PASS` authorizes: `R7_NEXT_ATTRIBUTION_CONTRACT_REVIEW_ONLY`; any new attribution route must use saved authoritative ensemble rows or the exact native float32 reconstruction, receive a new route/run identity, and preserve R5/R6 terminal history. PASS does not itself authorize execution, training, inference, protected data, full OOF or deployment.
- `INCONCLUSIVE` authorizes: `NONE`; missing/incomplete keys, non-finite values, unresolved source/run identity, or unavailable required compact support stops as `R7_A0_INPUT_IDENTITY_INCONCLUSIVE_STOP`.
- `FAIL` stops: complete inputs that do not reproduce under the authoritative float32 semantics, or an exact-score reconstruction that cannot replay the saved policy/cell results, return `R7_A0_SCORE_OR_POLICY_IDENTITY_FAIL_STOP / NONE` and require a separately approved integrity incident before any scientific attribution work.

## Implementation Contract

- Exact change and disabled mechanisms: add one deterministic CPU-only CSV identity audit, native float32 and float64 reconstruction summaries, exact four-cell policy replay and downstream sensitivity summaries; disable training, inference, candidate generation, image/tensor/cache access, score calibration, threshold/coverage search, subgroup analysis, bootstrap, and result-dependent reruns.
- Checkpoint/load/init/freeze contract: no checkpoint or model is loaded, no parameter initializes, and no stochastic computation exists. Torch is used only to execute the exact float32 two-seed stack/mean reduction frozen in the R5 source.
- Input whitelist and prohibited inputs: allow only the SHA-bound R5 lifecycle identity, three R5 raw CSVs, R5 closeout/cell summary, and the R6 closeout that records the triggering mismatch. Prohibit datasets, images, clean RGB, candidate tensors, model states, semantic labels, protected roles and any reconstructed score definition not declared here.
- Dataset/split/preprocessing/metric identities: R5 folds 0/1, cells `P0/S1/S2/G0`, D_ref/D_rep, positive/negative full actions, seeds 3407/3411, severe `<=-0.2 dB`, hard `<=-0.5 dB`, R5 SHA tie key, and exactly `ceil(0.20*192)=39` selected names per fold/cell.
- Matched baseline and budget: native and float64 reconstructions use identical decimal inputs and keys; only numerical dtype/reduction semantics differ. The native replay is authoritative and the float64 path is never used to rescue or redefine it.
- Resource/cost limits or descriptive-only rationale: CPU-only full finite-population audit over 21,504 rows; expected wall time 60 seconds and hard timeout 600 seconds. The CPU contract executes the same reconstruction, four-cell policy and comparison functions on the full formal row-count scale using synthetic protected-data-free rows.
- Runner and required assets: unchanged `experience_docx/tools/run_route_operation.sh`; R5 lifecycle `8fc05944...`, per-seed scores `6cfcea93...`, candidate scores `53061bea...`, policy rows `63c1aa4c...`, R5 closeout `e8d6151a...`, R5 cell summary `8498491e...`, and R6 closeout `7dd2ab48...`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `experience_docx/route_runtime_specs/R7_A0_FROZEN_R5_SCORE_CONSTRUCTION_IDENTITY_AUDIT.json` and `experience_docx/tools/r7_a0_r5_score_construction_identity.py`.
- Representative engineering fixture or metadata-only exemption: `contract(context_path)` generates the exact formal key cardinalities, produces saved ensembles with native float32 reduction, builds all four policies and verifies native exactness, float64 sensitivity reporting, fixed coverage, finite outputs and bounded resource class without creating `workload/`.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R7_A0_FROZEN_R5_SCORE_CONSTRUCTION_IDENTITY_AUDIT` | integrity audit over R5 development raw support | exact native score and policy identity | `R7_NEXT_ATTRIBUTION_CONTRACT_REVIEW_ONLY` |

- First operation: R7_A0_FROZEN_R5_SCORE_CONSTRUCTION_IDENTITY_AUDIT
- Expected wall time and monitor profile: 60 seconds expected, 600 seconds hard timeout, `short` profile, one bounded startup observation and one finish observation.
- Complete-unit resume policy: `none`; an interruption requires a new output identity and applicable engineering review.
- Cloud workspace/run/output/status/closeout: fresh route workspace; output id `r7-a0-r5-score-identity-r1`; generic status/heartbeat/runtime log; closeout `r7_a0_score_construction_identity_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract, provenance/access, input identity, reconstruction, policy replay, sensitivity, gate, resource, typed closeout, one scientific conclusion and terminal index. The three raw R5 CSVs and any row-level mismatch table remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`

The card is immutable after launch. R5 remains `COMPLETED_GATE_FAIL / NONE` and
R6 remains `COMPLETED_GATE_INCONCLUSIVE / NONE`; R7 cannot overwrite either.
The typed R7 closeout owns its terminal authority.
