# Haze4K v3.0 A0-Anchored Partial-Unfreeze Risk-Controlled ConvIR

Hypothesis: after v2.42 established `FROZEN_A0_GT_DESCENT_FAIL`, a materially changed route that moves beyond a frozen ConvIR-B small residual head may recover GT-aligned descent direction while preserving A0 tail safety.

Source: starts from immutable `github/codex/haze4k-official-arch-anchor`.

Architecture: `V300A0AnchoredConvIR` subclasses official ConvIR-B, preserves official parameter keys, and adds a zero-init `V300_*` low-frequency residual branch on the final output. Stage-0 must match official A0 with max absolute difference `<= 1e-7`.

Allowed new prefixes: `V300_` only. Official ConvIR-B keys must strict-shape load from `haze4k-base.pkl`; unexpected or shape-mismatched keys are fatal.

Scopes:
- `frozen_probe`: train only `V300_*` as a diagnostic richer frozen-carrier probe.
- `tier_a_partial`: train `V300_*`, `Decoder.2`, `Convs.1`, and `feat_extract.5` with low LR for official layers.

Loss: GT Charbonnier plus A0 anchor, hinge vs A0 MSE, top-k CVaR relative-MSE, residual direction loss `max(0, <A0-GT, Y-A0>)`, and residual norm penalty.

Metric contract: reuse v2.41 canary32 train-derived OOF split. Gates are mean `>= +0.15`, hard `>= +0.30`, easy `>= 0`, p05 `>= -0.01`, CVaR5 `>= -0.02`, severe `0`, strong-reference regressions `0`, fold pass `>= 4/5`, and severe_direction_bad `0`.

Stage policy: Stage-0 first. Canary32 OOF only if Stage-0 passes. Canary80 remains blocked unless canary32 passes. Locked test remains blocked unless a later fixed canary80 confirmation passes. No canary80 or locked test is authorized by this route card initially.

Tier-B addendum: after `frozen_probe` and `tier_a_partial` both failed canary32 with direction-dominant tail failures, one controlled Tier-B diagnostic is authorized to test whether moving the last two decoder stages changes residual direction. Tier-B trains `V300_*`, `Decoder.1`, `Decoder.2`, `Convs.0`, `Convs.1`, and `feat_extract.3/4/5`; canary80 and locked test remain blocked unless this canary32 passes the original gates.

## Closeout

Stage-0 passed with identity max abs vs A0 `0.0`, finite outputs, forbidden symbol hits `0`, and locked test untouched.

Canary32 OOF failed for all authorized scopes:
- `frozen_probe`: mean/hard/easy `-0.0008/+0.0146/+0.0089`, p05/CVaR5 `-0.1387/-0.2109`, severe `3`, fold pass `0/5`.
- `tier_a_partial`: mean/hard/easy `+0.0072/+0.0718/+0.0197`, p05/CVaR5 `-0.3751/-0.5918`, severe `22`, fold pass `0/5`.
- `tier_b_partial`: mean/hard/easy `+0.0050/+0.2226/-0.0889`, p05/CVaR5 `-0.6924/-1.0190`, severe `42`, fold pass `0/5`.

Decision: `V300_CANARY32_FAIL_TAIL_DIRECTION_RISK_LOCK_CANARY80_LOCKED_TEST`. Canary80 and locked test remain blocked. Tier-B increased hard movement but worsened easy/tail risk and kept severe failures mostly direction_bad (`40/42`), so this route does not authorize simple wider decoder tuning.
