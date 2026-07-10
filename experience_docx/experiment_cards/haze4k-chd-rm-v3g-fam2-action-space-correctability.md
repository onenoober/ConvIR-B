# Haze4K CHD-RM v3g FAM2 Action-Space Correctability

Date: 2026-07-10

Branch:
`codex/haze4k-v5-v3g-fam2-action-space-correctability`

Status:
`COMPLETED_GATE_PASS_STOP_NO_TRAINING`

Decision:
`V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING`

## Objective

Determine whether the v3f output-space correction-selection oracle is
realizable by the actual half-resolution FAM2 actuator.

## Authorized Work Completed

A no-training action-space audit was run on internal `val_inner` 600 only:

- define FAM2 alpha action variable under D7c hard veto;
- compute gate-site MSE gradients;
- verify gradients with block finite differences;
- replay gradient-derived action-space oracle policies.

No locked test, no training, no v3d continuation, no 20-epoch run, no v4/RARM
expansion, and no backbone/FAM1/neighbor unfreeze were used.

## Result

The gate passed but stops at no-training evidence. Best action-space oracle
`ACTION_CLOSE_FILTER_POSITIVE_GRAD` reached mean `0.412676` dB and p10
`0.060075` dB with worst `-0.000241` dB.
Gradient/finite-difference alignment was strong: sign agreement
`0.910641`, Spearman `0.932047`.

Hard D7c and ungated action replay are still weak/tail-risky, so the bottleneck
is deployable operator-site context rather than FAM2 actuator capacity.

## Next Gate

Only a separate no-training operator-site context feature audit is authorized.
No router/ranker training is authorized by v3g alone.
