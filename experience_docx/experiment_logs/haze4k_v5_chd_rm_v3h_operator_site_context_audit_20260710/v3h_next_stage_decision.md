# v3h Next Stage Decision

Date: 2026-07-10

Decision:
`V3H_OPERATOR_CONTEXT_FEATURES_WEAK_NO_ROUTER_TRAINING`

## Result

v3h completed the authorized no-training operator-site context feature audit and
failed both gates. It does not authorize training.

Key holdout evidence on odd-index internal `val_inner` images:

- Best feature: `d7c_logit_mean`, keep dir AUROC `0.504729`, AP `0.506163`, Spearman `0.007875`.
- Best feature replay policy: `FEATURE_04_residual_abs_high_0.25`, mean `0.008995` dB, p10 `-0.032779` dB, regressions <= -0.2 dB `0`.
- Hard D7c action replay: mean `0.009352` dB, p10 `-0.111293` dB, regressions <= -0.2 dB `9`.
- Gradient oracle reference: mean `0.420325` dB, p10 `0.059767` dB, regressions <= -0.2 dB `0`.

## Interpretation

The FAM2 action oracle remains strong, but the audited deployable operator-site
features do not recover it. The best replay policy is safer than hard D7c in
tail count, but it does not improve holdout mean utility and is not evidence for
training a router.

## Next Authorized Work

None within the current FAM2 scalar/operator-site feature route. A future route
would need materially new information, target semantics, or controller source.

No router training, v3d continuation, v3f-B training, locked test access,
canary expansion, 20-epoch run, v4/RARM expansion, or unfreeze route is
authorized by v3h.
