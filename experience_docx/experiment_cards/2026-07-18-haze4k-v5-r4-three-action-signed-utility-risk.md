# Haze4K v5 R4 Three-Action Signed Utility-Risk Development

Date: 2026-07-18

Status: `PLANNED`

## Identity

- Route id: `haze4k_v5_r4_three_action_signed_utility_risk_20260718`
- Question: can one jointly trained, candidate-conditioned signed utility/harm/severe representation safely choose among exact no-op, the frozen full positive action and the frozen full negative action, overcoming R3 signed action collapse with material full-OOF development utility?
- Rules commit: `github/main@cd89248b8f073785b0efad85f5384d07a4bf3cf1`
- Source branch/commit: `github/main@cd89248b8f073785b0efad85f5384d07a4bf3cf1`
- Route branch: `codex/haze4k-v5-r4-three-action-signed-utility-risk`
- Locked test/canary policy: confirmation identities/outcomes, historical A1X-432 outcomes, canary and locked test are prohibited.

## Approved Research Basis

R3 A2 closed only its frozen C3 critic after low gain, low retention, no response
increment and nine severe cases. It did not test the approved independent
mechanism here. Read-only A0 development diagnostics established that exact
`reference_noop`, `state_positive_full` and `state_negative_full` retain about
99.6% of the nine-action oracle, while R3 selected the negative representative
zero times despite a substantial negative-action oracle subset. The target
bottleneck is therefore signed action-mode collapse together with
candidate-specific downside-risk identification and unstable abstention.

## Scientific Contract

- Population and analysis/grouping unit: exact frozen R3 S0 768 development images; one clean-reference image/group is independent and D_ref/D_rep remain paired.
- Intervention or factor contrast and reference: a shared candidate-conditioned head emits signed utility, harm probability and severe probability for exact no-op, full positive and full negative representatives; exact no-op is the reference and fixed safe fallback. The matched scalar head, risk-disabled joint ablation and within-image positive/negative feature swap are frozen controls.
- Primary outcome, direction and aggregation: larger worse-operator selected-policy PSNR gain, oracle retention, true-minus-sign-shuffle, joint-minus-independent increment, negative-oracle-subset gain and coverage are better; smaller severe/hard risk is better; 4,000 paired image/group bootstraps retain both operators and use the worse operator per draw.
- Preferred mechanism and strongest competing explanation: joint candidate-specific signed value and downside risk recover the negative mode safely; the competitor is that the oracle exists but inference-visible features cannot identify sign or harm, leaving calibration noise or generic abstention.
- Evidence roles and candidate/freeze point: this operation is `development_screening`; the exact three actions, joint head, matched scalar, risk-disabled ablation and sign swap are frozen before any R4 outcome. All four S0 folds are development; confirmation, canary and locked test are prohibited.
- Primary gate, uncertainty and threshold source: PASS requires gain LCB95 `>=+0.020 dB`, retention LCB95 `>=0.25`, true-minus-sign-shuffle LCB95 `>=+0.005 dB`, joint-minus-independent LCB95 `>=+0.005 dB`, negative-oracle-subset gain LCB95 `>0`, coverage LCB95 `>=0.10`, zero severe/hard points, severe-rate UCB95 `<=0.005`, native-shape/operator mean `>=-0.020 dB`, and structural/access validity. Thresholds are the approved R4 recommendation and unchanged R3 materiality scale, frozen pre-result.
- `PASS` authorizes: `R4_CONFIRMATION_REVIEW_ONLY`; no confirmation or later runtime.
- `INCONCLUSIVE` authorizes: `NONE`; only incomplete or structurally invalid planned evidence qualifies.
- `FAIL` stops: the mechanism with `NONE`; no rerun, tuning, neighbor search or later-role access.

## Implementation Contract

- Exact change and disabled mechanisms: add only the head-only three-action joint utility/risk development path, matched scalar, risk-disabled ablation and sign swap; disable action search, feature search, ConvIR architecture training, R3 prediction reuse, confirmation, canary and locked test.
- Checkpoint/load/init/freeze contract: strict-load official checkpoint SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`; freeze all ConvIR parameters; initialize only the new head from seed `3407` or `3411`; head-only trainable scope.
- Input whitelist and prohibited inputs: whitelist sealed A0 base/step/support/current summaries, frozen action identity/sign, candidate-minus-reference RGB summaries and official-checkpoint-frozen first-encoder response summaries. Prohibit filename, fold id, GT/clean RGB as model input, R3 predictions, confirmation identity/outcome, historical A1X-432 outcome, canary and locked-test information.
- Frozen actions: `reference_noop`, `state_positive_full`, `state_negative_full`; no candidate or action search.
- Loss: signed utility Huber plus within-image listwise ranking, with explicit harm and severe binary losses only in the joint cell. No R3 prediction-dependent dynamic 4x harm weighting.
- Decision rule: choose the higher predicted non-noop utility only when nested calibration's utility-margin and, for the joint cell, harm/severe constraints pass; otherwise exact no-op.
- Calibration: fixed finite margin/risk grid selected inside the held-out development calibration fold by maximum mean gain subject to at least 10% coverage and zero severe/hard calibration cases; deterministic tie ordering is frozen in code.
- Dataset/split/preprocessing/metric identities: frozen S0 ledger with four exact 192-image folds; next fold calibrates, remaining two fit, one tests; image PSNR from RGB MSE on `[0,1]`; no exclusion; repository-default development-only access.
- Matched baseline and budget: independent scalar with identical inputs/hidden width/folds/seeds/optimizer/epochs and within 1.5% parameters; all four folds x seeds `3407/3411` x three cells, 32 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch 64.
- Resource/cost limits or descriptive-only rationale: full 1,536 feature units plus 24 train/eval units on one GPU; 10,800 seconds expected, 21,600 seconds hard timeout, no resume.
- Runner and required assets: unchanged generic runner, `r4_three_action_signed_utility_risk.py`, frozen S0 ledger/A0 cache/Haze4K development data and strict official checkpoint.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R4_D0_THREE_ACTION_SIGNED_UTILITY_RISK_DEV` | `development_screening`; 4 folds x 2 seeds x 3 frozen cells | all formal utility, specificity, increment, coverage, subgroup, integrity and safety gates | `R4_CONFIRMATION_REVIEW_ONLY` |

- First operation: R4_D0_THREE_ACTION_SIGNED_UTILITY_RISK_DEV
- Expected wall time and monitor profile: 10,800 seconds expected, 21,600 seconds hard timeout, `standard` monitor, one finish near frozen ETA.
- Complete-unit resume policy: `none`; any allowed same-contract ordinary engineering repair uses lifecycle recovery without changing scientific identity.
- Cloud workspace/run/output/status/closeout: MCP fresh route workspace; run root sealed by receipt; output `r4-d0-three-action-dev-r1`; closeout `r4_d0_three_action_signed_utility_risk_dev_closeout.json`; generic status and heartbeat.
- Compact Git evidence and cloud-only raw artifacts: Git receives frozen contract, structural/bootstrap/gate, cell/training/risk-coverage/strata/resource/access summaries, typed closeout and scientific conclusion. Raw OOF predictions, tensors, labels, checkpoints and large outputs remain cloud-only.
