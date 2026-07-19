# Haze4K v5 R12 Action-Conditioned Downside Observability

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r12_action_conditioned_downside_observability_20260719`.
- Question: do immutable R11 OOF regional score outputs contain action-conditioned severe-downside information beyond action-agnostic score, deterministic sign-swap and within-image label-shuffle controls?
- Rules commit: `github/main@893ba97790ad19d745ff676f5bbf28bd37395d50`.
- Source branch/commit: terminal R11 route `c183817e2b3befdeeb12278aa6e6a0574883b6d5`; immutable cloud-only R11 OOF tile predictions bound by SHA-256.
- Route branch: `codex/haze4k-v5-r12-action-conditioned-downside-observability-20260719`.
- Locked test/canary policy: confirmation identities/images/targets/outcomes, canary and locked test are prohibited; all permissions remain false.

## Scientific Contract

- Population and analysis/grouping unit: the exact 384 R11 development clean-image groups, folds 0/1 with 192 names each; 49,152 active tile/action rows are nested observations and clean-image identity is the outer split and grouped-bootstrap unit.
- Intervention or factor contrast and reference: fit a deterministic cross-fold linear severe-risk calibrator to immutable R11 OOF predicted mean/worst utility, their gap, normalized tile coordinates, action sign and prespecified action interactions. Compare with parameter-identical action-agnostic, positive/negative score-swap and within-image training-label-shuffle controls. No candidate, image render, restoration feature, policy or action budget is regenerated.
- Primary outcome, direction and aggregation: compute severe-action AUROC, top-20% severe capture and retained severe-prevalence ratio within each clean image containing both label classes, then macro-average across images; larger AUROC/increments/capture and smaller retained ratio are better. Four thousand fold-stratified clean-image bootstrap draws retain all nested tile/actions inside each sampled image.
- Preferred mechanism and strongest competing explanation: preferred mechanism is that utility and downside are separable only after an explicit action-conditioned risk objective. The strongest competitor is that R11 outputs contain generic difficulty/risk but no stable action-specific downside information; a weaker competitor is residual spatial or image prevalence learned without candidate alignment.
- Evidence roles and candidate/freeze point: R11 is terminal development-screening evidence and remains unchanged. R12 is a new `development_screening` readout-mechanism probe. Inputs, features, controls, labels, folds, optimizer, rejection fraction, metrics, thresholds, uncertainty and terminal mapping freeze at the R12 route commit.
- Primary gate, uncertainty and threshold source: PASS requires exact R11 terminal/input identity; 384 groups and 49,152 unique rows; at least 1,000 severe rows across at least 300 groups; primary AUROC LCB95 `>=0.75`; primary-minus-action-agnostic and primary-minus-sign-swap AUROC LCB95 each `>=0.03`; primary-minus-label-shuffle AUROC LCB95 `>=0.10`; top-20% severe-capture LCB95 `>=0.60`; retained severe-prevalence ratio UCB95 `<=0.60`; both folds primary AUROC `>=0.70`, severe capture `>=0.50`, and all three AUROC increments positive. Thresholds are fixed from the pre-R12 R4B/R5 severe-risk evidence and the R11 cloud-audit failure anatomy, not selected on R12 outcomes.
- `PASS` authorizes: `R12_DECOMPOSED_UTILITY_RISK_DECISION_CONTRACT_REVIEW_ONLY`; no policy replay, coverage search, restoration training/inference or protected-data access.
- `INCONCLUSIVE` authorizes: `NONE`; identity, completeness, finite-value, optimizer, interval or protected-access failures stop as `R12_A0_INPUT_OR_DOWNSIDE_INCONCLUSIVE_STOP`.
- `FAIL` stops: valid evidence decisively missing any scientific or fold line returns `R12_A0_ACTION_CONDITIONED_DOWNSIDE_FAIL_STOP / NONE`; do not tune features, interactions, regularization, rejection coverage, labels, folds or thresholds.

## Implementation Contract

- Exact change and disabled mechanisms: add one deterministic cross-fold balanced logistic severe-risk calibrator and three matched controls over immutable R11 OOF output rows. Disable ConvIR construction, cache/image/target access, candidate generation, policy replay, budget assignment, subgroup search, threshold calibration and all protected roles.
- Checkpoint/load/init/freeze contract: no checkpoint or model state is loaded; each convex calibrator starts from zeros and uses fixed full-batch L-BFGS-B, L2 `1e-4`, maximum 200 iterations and no selection or restart.
- Input whitelist and prohibited inputs: allow only the receipt-bound R11 typed closeout and R11 tile-prediction CSV columns `name,fold,action,tile,predicted_mean,predicted_worst,actual_worst`. Outcome is label-only. Prohibit filename/fold as model features, clean targets, images, cached tensors, oracle maps/budgets, per-image replay outcomes, semantic labels, confirmation, canary and locked test.
- Dataset/split/preprocessing/metric identities: exact R11 folds0/1, active actions 1/2, fixed severe label `actual_worst<=-0.2 dB`, normalized 8x8 coordinates, outer-training-only normalizer, balanced binary log loss, clean-image macro metrics with fold-stratified grouped bootstrap.
- Matched baseline and budget: all cells share rows, feature width, outer folds, normalizer, optimizer, iterations, labels at evaluation, bootstrap and gates. Action-agnostic zeros action terms; sign-swap swaps positive/negative predicted score pairs within name/tile; label-shuffle permutes outer-training labels within each image using a name-bound deterministic seed.
- Resource/cost limits or descriptive-only rationale: CPU-only calibration screen; expected 120 seconds, hard timeout 600 seconds, no resume. The protected-data-free contract uses all 49,152 formal-shape rows, eight deterministic fits and 4,000 grouped-bootstrap draws; wall time must be `<=120` seconds and peak RSS `<=1024 MiB`.
- Runner and required assets: unchanged generic runner SHA-256 `336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b`; R11 closeout SHA-256 `0e6bc5a70cd86f2aa58a34004bb895ad0295b6d7a5b2f619695878de38d4eec9`; R11 cloud tile predictions SHA-256 `e45ae4b11843063fc4cd0360ebcc7f7e872989bc6f9657228ee1de9b90bb16c7`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `experience_docx/route_runtime_specs/R12_A0_ACTION_CONDITIONED_DOWNSIDE_OBSERVABILITY_SCREEN.json` and `experience_docx/tools/r12_a0_action_conditioned_downside_observability.py`.
- Representative engineering fixture or metadata-only exemption: the entrypoint's production fit, controls, AUROC, risk-coverage, grouped bootstrap and finalizer run on a protected-data-free formal-shape synthetic fixture under the frozen cost bound.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R12_A0_ACTION_CONDITIONED_DOWNSIDE_OBSERVABILITY_SCREEN` | development OOF mechanism screen | action-specific severe-risk increment and fixed-coverage safety | `R12_DECOMPOSED_UTILITY_RISK_DECISION_CONTRACT_REVIEW_ONLY` |

- First operation: R12_A0_ACTION_CONDITIONED_DOWNSIDE_OBSERVABILITY_SCREEN
- Expected wall time and monitor profile: 120 seconds expected, 600 seconds hard timeout, `short` profile.
- Complete-unit resume policy: `none`; interruption requires typed engineering review and a new output identity without scientific changes.
- Cloud workspace/run/output/status/closeout: fresh workspace; output `r12-a0-action-downside-r1`; closeout `r12_a0_action_conditioned_downside_observability_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract/provenance/input/cell/fold/bootstrap/risk-coverage/gate/resource evidence, typed closeout and one conclusion. Row scores, bootstrap arrays and runtime logs remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. R11 and all earlier terminal decisions remain unchanged.
