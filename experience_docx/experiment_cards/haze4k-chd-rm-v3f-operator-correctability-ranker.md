# Haze4K CHD-RM v3f Operator-Correctability Ranker

Date: 2026-07-10

Branch:
`codex/haze4k-v5-v3f-operator-correctability-ranker`

Status:
`COMPLETED_GATE_STOP`

Decision:
`V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING`

## Objective

Determine whether current FAM2 correction has a deployable operator-specific
correctability signal that can be used under D7c as a safety veto.

## Authorized Work

Run v3f-A no-training correctability target/separability audit on internal
val-inner 600 only. No locked test, no v3d continuation, and no training are
authorized unless v3f-A writes a separate authorization.

## Closeout

v3f-A completed on `convir-4090` as a no-training replay/separability audit.
It sampled `4,915,200` pixels across the internal train-derived val-inner 600
split and compared current FAM2 marginal gain against deployable scalar proxies.

The best scalar proxy was FAM2 correction magnitude:

- positive-gain AUROC `0.532034`;
- positive-gain AUPRC `0.519730`;
- gain Spearman `0.033662`.

This is below the predeclared `0.56` AUROC gate for a lightweight ranker screen.
D7c score and D7c hard gate were near random for current FAM2 positive gain:
AUROC `0.492237` and `0.492251`.

Replay evidence confirms the safety/mean tradeoff:

- ungated control `W_U+G_1`: mean PSNR delta `+0.033065`, `91`
  `<= -0.2 dB` regressions;
- D7c-vetoed control `W_U+G_D`: mean PSNR delta `+0.012366`, `18`
  `<= -0.2 dB` regressions;
- D7c-vetoed gain oracle: mean PSNR delta `+0.078254`, zero
  `<= -0.2 dB` regressions.

The oracle result shows correctability exists in principle inside the D7c
safety region, but the audited deployable scalar features are too weak to
justify training a ranker. v3f therefore stops. Do not launch v3f-B, v3d
continuation, 20-epoch runs, v4/RARM expansion, neighbor/FAM1/backbone unfreeze,
canary expansion, or locked-test access from this evidence.
