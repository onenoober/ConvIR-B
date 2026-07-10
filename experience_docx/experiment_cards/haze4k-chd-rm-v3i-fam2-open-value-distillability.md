# Haze4K CHD-RM v3i FAM2 Open-Value Distillability

Date: 2026-07-11
Branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
Evidence:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711/`

## Purpose

v3g proved that FAM2 action-space oracle selection is strong. v3h showed that
the audited scalar/operator-site features cannot recover that oracle. v3i tests
whether the open-value target itself is compressible into a controller-realistic
spatial policy, then cross-checks several deployable signal families before any
controller training is allowed.

## Route Identity

This is a no-training diagnostic route after v3h. It is not a v3d continuation,
not v3f-B ranker training, not a router canary, and not a model-structure route.
The route used internal train-derived `val_inner` 600 evidence only.

## Locked Test Policy

Locked Haze4K test remained sealed. `no_locked_test_audit.json` reports
`locked_test_touched: false`. No checkpoints or runtime artifacts were produced
for promotion.

## Stage Gates And Results

| Stage | Gate | Result | Decision |
| --- | --- | --- | --- |
| v3i-A teacher compressibility | Compressed policy must retain at least 25% of the hard-to-oracle mean gap and not exceed hard severe regressions. | Passed. Hard D7c mean `+0.012784 dB`, `23` severe; GT open oracle mean `+0.411695 dB`, zero severe; best compressed `ALPHA_SECANT_Q3` mean `+0.412879 dB`, zero severe. | `V3I_A_ORACLE_COMPRESSIBLE_AUTHORIZE_FULL_CONTEXT_OOF_PROBE_ONLY` |
| v3i-B full-context OOF probe | Single-forward/full-context features must beat hard D7c with stable OOF replay. | Failed. Best OOF `OOF_DW3X3_TOP50` mean `+0.016627 dB`; bootstrap delta vs hard CI low `-0.000163`. | `V3I_B_SINGLE_FORWARD_OBSERVABILITY_FAIL_AUTHORIZE_COUNTERFACTUAL_FEATURE_AUDIT_ONLY` |
| v3i-C counterfactual response OOF probe | Alpha counterfactual response features must recover a useful open-action controller signal. | Failed. Best OOF `OOF_CF_RESPONSE_DW3X3_TOP50` mean `+0.008543 dB`; delta vs hard mean `-0.004241`. | `V3I_C_COUNTERFACTUAL_RESPONSE_FAIL_AUTHORIZE_DISAGREEMENT_AUDIT_ONLY` |
| v3i-D checkpoint/transform disagreement OOF probe | Disagreement/uncertainty features must identify safe useful open actions. | Failed. Best OOF `OOF_DISAGREE_LINEAR_TOP50` mean `+0.010534 dB`; delta vs hard mean `-0.002250`. | `V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE` |

## Final Decision

`V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE`

The FAM2 open-value actuator and privileged target are real: v3i-A reproduces a
strong, spatially compressible oracle near `+0.412 dB` mean with no severe
regressions. The blocker is deployability. Across independent OOF probes,
single-forward features, counterfactual RGB response, and checkpoint/transform
disagreement all fail to recover the oracle as a stable replay policy.

Stop the current FAM2 router/distillation route. Do not launch v3f-B scalar
ranker training, v3d continuation, 20-epoch continuation, canary expansion,
current-signal router training, or locked-test evaluation from this evidence.

## Next Route Shape

The next credible route should redesign the correction itself rather than keep
probing the same router surface. Candidate directions are joint
correction-confidence prediction or bounded experts with explicit false-action
protection. Any such route must first pass a no-training held-out
separability/replay gate before training.
