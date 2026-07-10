# v3g Next Stage Decision

Date: 2026-07-10

Decision:
`V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING`

## Result

The v3g no-training audit passed its action-space gate and then stops. It does
not authorize training.

Key evidence on internal `val_inner` 600:

- Best action-space oracle: `ACTION_CLOSE_FILTER_POSITIVE_GRAD` mean `0.412676` dB, median `0.338481` dB, p10 `0.060075` dB, worst `-0.000241` dB.
- Hard D7c action replay: mean `0.012784` dB, p10 `-0.121512` dB, regressions <= -0.2 dB `23`.
- Ungated W_U action: mean `0.033065` dB, p10 `-0.284573` dB, regressions <= -0.2 dB `91`.
- Output D7c oracle reference: mean `0.078254` dB.
- Gradient/finite-difference alignment: sign agreement `0.910641`, Spearman `0.932047`, rows `3640`.

## Interpretation

The FAM2 actuator is not the current limiting factor. A label-derived alpha
oracle at the true FAM2 action site is much stronger and safer than hard D7c or
ungated W_U replay. The deployable bottleneck is the missing operator-site
context/controller: scalar image-level proxies from v3f are too weak, but the
action target itself is strong.

## Next Authorized Work

Authorize only a separate no-training operator-site context feature audit on the
same internal `val_inner` split. The audit should test whether inference-time
features at or near the FAM2 action site can predict the gradient-derived action
benefit well enough to justify a later training route.

No router training, v3d continuation, v3f-B training, locked test access,
canary expansion, 20-epoch run, v4/RARM expansion, or unfreeze route is
authorized by v3g alone.
