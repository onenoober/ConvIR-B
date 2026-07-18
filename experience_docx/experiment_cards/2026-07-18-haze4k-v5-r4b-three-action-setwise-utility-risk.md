# Haze4K v5 R4B Three-Action Set-Wise Utility-Risk

Date: 2026-07-18

Status: `PLANNED`

## Identity

- Route id: `haze4k_v5_r4b_three_action_setwise_utility_risk_20260718`
- Question: can a permutation-equivariant model that jointly observes exact no-op, full positive and full negative actions separately identify signed mean utility, lower-tail utility and downside risk well enough to retain material three-action oracle value without severe or hard regression?
- Rules commit: `github/main@dae654c2512d110a2e64c65b4e4640113383a189`
- Source branch/commit: `github/main@dae654c2512d110a2e64c65b4e4640113383a189`
- Route branch: `codex/haze4k-v5-r4b-three-action-setwise-utility-risk`
- Locked test/canary policy: sealed 432 confirmation identities/outcomes, historical A1X-432 outcomes, canary and locked test are prohibited throughout development.

## Scientific Contract

- Population and analysis/grouping unit: frozen R3 S0 768 development images in four 192-image folds; one clean-reference image is independent and D_ref/D_rep remain paired. A risk event occurs when either operator triggers it.
- Intervention or factor contrast and reference: primary model is a small permutation-equivariant self-attention set encoder jointly consuming the three frozen actions and emitting per-action mean utility, q05 utility, harm probability and severe probability. Exact no-op has fixed zero utility/risk at decision time. The matched interaction-removed utility-risk scorer is the primary reference; set-wise mean-only independently tests risk factorization.
- Primary outcome, direction and aggregation: larger worse-operator OOF policy gain, oracle retention, true-minus-action-value-shuffle, set-wise-minus-independent, utility-risk-minus-mean-only, negative-oracle-subset gain and coverage are better; smaller selected severe/hard risk, calibration error and CVaR loss are better. Paired clean-image bootstraps retain both operators and use the worse operator per draw.
- Preferred mechanism and strongest competing explanation: explicit candidate interaction plus separate lower-tail/severe heads recover signed relative action value; the competitor is that the frozen inference-visible representation remains insufficient, so attention and risk factorization only learn abstention or action priors.
- Evidence roles and candidate/freeze point: S0 contracts, the three-action reduction, all model cells, feature masks, label shuffles, folds, ensemble seeds, losses, calibration, metrics and gates are frozen before R4B A0. A0/A1/A2 are `development_screening`; later operations require the exact prior typed PASS.
- Primary gate, uncertainty and threshold source: A0 first verifies three-action oracle and rare-event feasibility. A1 is a two-fold mechanism futility screen. A2, only if A1 survives, requires gain LCB95 `>=+0.020 dB`, retention `>=0.25`, both mechanistic increments and true-minus-shuffle LCB95 `>=+0.005 dB`, negative-oracle-subset gain LCB95 `>0`, coverage LCB95 `>=0.10`, zero severe/hard, selected-group severe exact one-sided UCB95 `<=0.005`, severe AUROC LCB95 `>0.5`, and AUPRC-minus-prevalence LCB95 `>0`. Thresholds come from the approved route, R3 materiality gates and the pre-result S0 register.
- `PASS` authorizes: A0 authorizes only `R4B_A1_SETWISE_MECHANISM_SCREEN`; A1 authorizes only `R4B_A2_FULL_OOF`; A2 authorizes only a separately approved confirmation review, never confirmation runtime.
- `INCONCLUSIVE` authorizes: `NONE`; structural incompleteness, protected-data violation or insufficient A0 grouped risk events stops without interpretation.
- `FAIL` stops: the current R4B mechanism with `NONE`; no same-contract rerun, tuning, neighbor architecture search or protected-stage access.

## Implementation Contract

- Exact change and disabled mechanisms: add only the three-action set encoder and frozen controls. Required A1 controls are candidate permutation audit, within-image action-value shuffle, risk-label shuffle, action-only, state-only, unsigned utility, interaction-removed utility-risk and set-wise mean-only. Disable nine-action training, feature search, ConvIR training, threshold rescue, checkpoint selection and all later stages.
- Checkpoint/load/init/freeze contract: A0 uses no model. A1/A2 strict-load official checkpoint SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`, freeze all ConvIR parameters, initialize only selector heads from fixed seeds `3407/3411`, and ensemble their predictions arithmetically before one policy decision.
- Input whitelist and prohibited inputs: whitelist frozen base/step/current/support summaries, action identity/sign, candidate-minus-no-op RGB summaries, frozen first-encoder response and within-set relative/disagreement context. Prohibit filename, fold id, GT RGB as model input, old R4 predictions, confirmation/canary/locked-test information and any result-driven feature addition.
- Dataset/split/preprocessing/metric identities: exact S0 ledger and sealed A0 cache; A1 evaluates outer folds 0/1 and A2 all folds, with the next development fold calibrating and remaining folds fitting; RGB image PSNR from MSE on `[0,1]`; no sample exclusion; 4,000 paired image/group bootstraps.
- Matched baseline and budget: primary set-wise utility-risk, interaction-removed utility-risk and set-wise mean-only have predeclared matched capacity; control feature masks and label shuffles use the same set architecture. A1 uses folds 0/1, two fixed ensemble members and 32 AdamW epochs; A2 uses all four folds only after A1 PASS.
- Resource/cost limits or descriptive-only rationale: A0 reads all 1,536 operator-image cache units. A1 and A2 use every planned fold, cell, seed and epoch; one GPU, no checkpoint selection, no reduced sample scope, and operation-specific hard timeouts.
- Runner and required assets: unchanged generic runner, stage-specific entrypoint/runtime spec, S0 JSON contracts, SHA-bound S0/A0 assets, Haze4K development data and official checkpoint only where required.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R4B_A0_THREE_ACTION_RISK_FEASIBILITY` | `development_screening`; all 768 images and both operators, no model training | oracle retention, materiality, grouped rare-event precision and operator consistency | `R4B_A1_SETWISE_MECHANISM_SCREEN` |
| `R4B_A1_SETWISE_MECHANISM_SCREEN` | planned only after exact A0 PASS; two folds, complete controls | mechanism non-futility, permutation, specificity and risk discrimination | `R4B_A2_FULL_OOF` |
| `R4B_A2_FULL_OOF` | planned only after exact A1 PASS; all four folds | complete formal utility, increment, risk, coverage and integrity gates | confirmation review only |

- First operation: R4B_A0_THREE_ACTION_RISK_FEASIBILITY
- Expected wall time and monitor profile: A0 expects 300 seconds with 1,800-second hard timeout and `short` monitor; later ETAs are frozen only in their separately committed runtime specs.
- Complete-unit resume policy: `none`; only one deterministic same-contract repair is allowed for an ordinary engineering root cause.
- Cloud workspace/run/output/status/closeout: fresh MCP workspace and receipt-bound output `r4b-a0-risk-feasibility-r1`; generic status/heartbeat; closeout `r4b_a0_three_action_risk_feasibility_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: each completed stage archives its frozen contract, typed closeout, scientific conclusion and all hash-bound compact JSON/CSV results to GitHub main before the next operation. Raw labels, per-image rows, tensors, predictions and caches remain cloud-only.
