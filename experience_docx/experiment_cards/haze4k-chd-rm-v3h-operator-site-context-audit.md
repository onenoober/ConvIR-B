# Haze4K CHD-RM v3h Operator-Site Context Audit

Date: 2026-07-10

Branch:
`codex/haze4k-v5-v3h-operator-site-context-audit`

Status:
`COMPLETED_GATE_FAIL_STOP_NO_TRAINING`

Decision:
`V3H_OPERATOR_CONTEXT_FEATURES_WEAK_NO_ROUTER_TRAINING`

## Objective

Determine whether inference-time features at or near the true FAM2 action grid
can predict the v3g gradient-derived alpha target.

## Authorized Work Completed

A no-training feature separability and feature-replay audit was run on internal
`val_inner` 600 only:

- compute v3g open/keep gradient targets at the FAM2 action grid;
- collect operator-site features including D7c score, density, input/A0 context,
  FAM2 x1/x2/fused norms, gamma/beta, and action-delta magnitudes;
- select simple top/bottom fraction replay candidates from even-index
  calibration images;
- evaluate selected candidates on odd-index holdout images.

No locked test, no training, no checkpoint creation, no v3d continuation, no
v3f-B ranker training, no 20-epoch run, no v4/RARM expansion, and no
backbone/FAM1/neighbor unfreeze were used.

## Result

The gate failed. Best holdout feature `d7c_logit_mean` reached keep
dir AUROC `0.504729`; best feature replay
`FEATURE_04_residual_abs_high_0.25` had mean `0.008995`
dB versus hard D7c action mean `0.009352` dB. The
oracle reference remained mean `0.420325` dB.

## Next Gate

No current FAM2 scalar/operator-site feature training route is authorized. Future
work requires materially new information, target semantics, or a different
controller source.
