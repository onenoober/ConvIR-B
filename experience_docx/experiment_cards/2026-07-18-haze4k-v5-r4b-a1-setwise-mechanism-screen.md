# Haze4K v5 R4B A1 Set-Wise Mechanism Screen

Date: 2026-07-18

Status: launch contract sealed as `PLANNED`; current terminal is
`COMPLETED_GATE_FAIL / R4B_A1_SETWISE_MECHANISM_FUTILITY_STOP / NONE`.
The post-hoc cloud evidence audit is `COMPLETED_AUDIT` and does not change the
formal terminal tuple.

## Identity

- Route id: `haze4k_v5_r4b_three_action_setwise_utility_risk_20260718`
- Question: on two frozen development folds, does a genuine permutation-equivariant three-action set encoder show non-futile material utility, candidate-interaction increment, risk-factorization increment, action-value specificity and grouped severe-risk discrimination?
- Rules commit: `github/main@3fa309192b5ae69cc6b5d25f36c9c374afdcb23d`
- Source branch/commit: `github/main@3fa309192b5ae69cc6b5d25f36c9c374afdcb23d`
- Route branch: `codex/haze4k-v5-r4b-three-action-setwise-utility-risk-a1`
- Locked test/canary policy: confirmation identities/outcomes, historical A1X-432 outcomes, canary and locked test are prohibited.

## Scientific Contract

- Population and analysis/grouping unit: exact S0 768 development images; A1 evaluates outer folds 0/1, one clean-reference image is independent, and D_ref/D_rep remain paired. Risk labels use the clean-image any-operator grouping frozen at S0.
- Intervention or factor contrast and reference: primary `S_utility_risk` is a 48-dimensional, four-head, no-position self-attention set encoder over exact no-op/positive-full/negative-full with mean/q05/harm/severe outputs. Primary mechanism contrasts are `I_utility_risk` for interaction removal and `S_mean_only` for risk-factorization removal.
- Primary outcome, direction and aggregation: larger selected-policy gain, oracle retention, true-minus-action-value-shuffle, set-wise-minus-independent, utility-risk-minus-mean-only and nuisance-control increments are better; higher severe AUROC/AUPRC lift and lower calibration error are better; zero severe/hard selected groups is required. Two fixed seed members are averaged before one policy decision, then 4,000 paired image/group bootstraps use the worse operator.
- Preferred mechanism and strongest competing explanation: cross-candidate attention plus lower-tail/severe outputs identify the relative signed action and abstain on hazardous actions; the competitor is action prior, state-only difficulty or unsigned mean ranking without usable candidate interaction/risk information.
- Evidence roles and candidate/freeze point: A0 passed and authorized exactly this `development_screening` operation. Seven cells, feature masks, risk-label shuffle, candidate permutation audit, action-value shuffle, folds, ensemble seeds, optimizer, epochs, calibration grid and all gates are frozen before A1 outcomes.
- Primary gate, uncertainty and threshold source: A1 survives only if structurally complete, permutation error `<=1e-6`, selected severe/hard groups are zero, severe AUROC LCB95 `>0.5`, AUPRC-minus-prevalence and AUPRC-minus-risk-label-shuffle LCB95 are both `>0`, and UCB95 reaches gain `+0.020 dB`, retention `0.25`, true-minus-shuffle `+0.005 dB`, set-wise-minus-independent `+0.005 dB`, utility-risk-minus-mean-only `+0.005 dB`, and primary-minus-best-nuisance `+0.005 dB`. A1 uses UCB futility because it is a two-fold screen; thresholds were frozen at S0 and conventional no-skill risk discrimination.
- `PASS` authorizes: only `R4B_A2_FULL_OOF`; no A2 launch until A1 evidence is archived and a new exact operation commit is sealed.
- `INCONCLUSIVE` authorizes: `NONE`; incomplete folds/cells/ensemble, invalid permutation, protected-data violation or non-finite risk audit cannot be interpreted.
- `FAIL` stops: `NONE`; any valid mechanism futility or selected severe/hard event closes R4B without tuning, rerun or neighbor search.

## Implementation Contract

- Exact change and disabled mechanisms: implement one small permutation-equivariant self-attention encoder and exactly six controls: matched independent utility-risk, set-wise mean-only, action-only, state-only, unsigned utility and training-fold risk-label shuffle. Candidate permutation and action-value shuffle are non-training audits. Disable architecture/feature search, nine-action training, result-driven thresholds and ConvIR updates.
- Checkpoint/load/init/freeze contract: strict-load official checkpoint SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`; freeze ConvIR; initialize only selector models from seeds `3407/3411`; arithmetic-average their mean/q05/harm/severe predictions before calibration and decision; no checkpoint selection.
- Input whitelist and prohibited inputs: whitelist base/step/current/support state summaries, action identity/sign, candidate-minus-no-op RGB summaries and frozen first-encoder response. Action-only/state-only are fixed masks. Prohibit filename, fold id, GT RGB as input, confirmation/canary/locked-test information, old R4 predictions and any new feature.
- Dataset/split/preprocessing/metric identities: S0 folds 0/1 test; next fold calibrates and remaining two folds train; both operators paired; image PSNR from RGB MSE `[0,1]`; candidate risk uses gain `<0`, severe `<=-0.2 dB`, hard `<=-0.5 dB`; no exclusion; 4,000 paired group bootstraps.
- Matched baseline and budget: set model has dimension 48, four attention heads and FFN width 96; independent MLP hidden width 115 is within 1% trainable parameters. All seven cells use two fixed seeds, 32 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch 64.
- Resource/cost limits or descriptive-only rationale: read all 1,536 feature units and run 28 complete train units; one GPU, 10,800 seconds expected, 21,600 seconds hard timeout, no resume or checkpoint search.
- Runner and required assets: unchanged generic runner, `r4b_a1_setwise_mechanism_screen.py`, SHA-bound S0/A0 cache and A0 PASS closeout, Haze4K development data and official checkpoint.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R4B_A1_SETWISE_MECHANISM_SCREEN` | `development_screening`; folds 0/1 x seven cells x two ensemble seeds | mechanism non-futility, full controls, permutation, specificity, risk discrimination and safety | `R4B_A2_FULL_OOF` |

- First operation: R4B_A1_SETWISE_MECHANISM_SCREEN
- Expected wall time and monitor profile: 10,800 seconds expected, 21,600 seconds hard timeout, `standard` monitor, one finish near the observed completion window.
- Complete-unit resume policy: `none`; one deterministic same-contract repair is allowed only for an ordinary engineering root cause.
- Cloud workspace/run/output/status/closeout: fresh receipt-bound workspace; output `r4b-a1-setwise-screen-r1`; generic heartbeat/status; closeout `r4b_a1_setwise_mechanism_screen_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git receives contract, typed closeout, bootstrap/gate/cell/control/risk/calibration/permutation/risk-coverage/training/resource/access evidence and conclusion. Raw OOF predictions, candidate tensors, labels, checkpoints and per-image risk rows remain cloud-only.

## Terminal And Cloud Audit Addendum

The formal A1 run completed normally at route commit
`336f1132c9aaffebda365b63e26a67aab4643531`. Its point estimates, risk metrics,
calibration values, row counts, data roles and all 13 closeout-bound compact
artifact hashes reproduce from the receipt-bound cloud evidence. The recorded
permutation maximum remains `1.9073486328125e-06` against the frozen `1e-06`
tolerance; it does not invalidate the much larger utility and retention
futility gaps.

The bounded post-hoc audit adds the following interpretation without changing
the contract or terminal decision:

- `95.0521%` of primary operator-image rows are no-op;
- among `609` operator-image rows with positive oracle headroom, `577`
  (`94.7455%`) abstain;
- the primary acts only on fold 1 (`9.8958%` coverage) and never on fold 0;
- all `181` negative-oracle operator-image rows still receive zero negative
  selections;
- candidate severe scores are directionally concordant in `77.9006%` of the
  `181` groups where the two active actions have different severe labels;
- per-seed predictions, per-action mean/q05 utility vectors, composite
  risk-coverage confidence, region rows and semantic subgroup labels were not
  persisted and must not be reconstructed by rerunning training or inference.

Cloud audit closeout:
`../experiment_logs/haze4k_v5_r4b_three_action_setwise_utility_risk_20260718/cloud_audit_closeout.json`.
