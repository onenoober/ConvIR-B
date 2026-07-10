# v3h Operator-Site Context Feature Audit

Date: 2026-07-10

Route id:
`haze4k_v5_chd_rm_v3h_operator_site_context_audit_20260710`

Route branch:
`codex/haze4k-v5-v3h-operator-site-context-audit`

Parent/source:
`origin/codex/haze4k-v5-v3g-fam2-action-space-correctability`
at `6edfbfcde45264effd4c083649077f9d017d21a9`.

## Purpose

Test whether inference-time features at the true FAM2 action site can recover
the v3g gradient-derived alpha target well enough to justify a later controller
route.

## Forbidden Flows

- No training.
- No new checkpoint.
- No v3d continuation.
- No v3f-B ranker training.
- No 20-epoch run.
- No v4/RARM expansion.
- No neighbor/FAM1/backbone unfreeze.
- No canary expansion.
- No locked Haze4K test.

## Metric Contract

Primary target is `keep_score = - dLoss/dAlpha` at `alpha=1` on the FAM2
action grid under the D7c hard action gate. Positive means the v3g close-filter
oracle would keep alpha active.

Feature separability uses even image indices for calibration and odd image
indices for holdout. Replay uses feature top/bottom fraction alpha masks under
the same D7c hard action gate and compares image-level PSNR delta against A0 on
internal `val_inner` only.

Gate:

- holdout operator-site feature `keep_dir_auroc >= 0.56`;
- best feature replay policy on holdout beats hard D7c action by `+0.02 dB`;
- best feature replay policy does not increase holdout `<= -0.2 dB`
  regressions versus hard D7c action.

## Result

Status: `COMPLETED_GATE_FAIL_STOP_NO_TRAINING`

Decision:
`V3H_OPERATOR_CONTEXT_FEATURES_WEAK_NO_ROUTER_TRAINING`

Both gates failed. Best holdout feature `d7c_logit_mean` reached only
`0.504729` keep dir AUROC. Best feature replay policy
`FEATURE_04_residual_abs_high_0.25` reached mean `0.008995`
dB, below hard D7c action mean `0.009352` dB, though with
fewer tail regressions. The gradient oracle reference remained strong at mean
`0.420325` dB.

Interpretation: the current bottleneck is missing deployable action-site signal.
No router/ranker training or continuation is authorized.
