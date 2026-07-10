# v3i Route Decision

Route: `haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711`
Branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
Date: 2026-07-11
Final decision: `V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE`

## Route Identity

This is a new diagnostic/no-training audit route after v3h. It is not a v3d
continuation, not v3f-B ranker training, not a router canary, and not a model
structure route.

## Source Of Truth

- GitHub `main` commit `b267d9f`: v3h evidence sync and prior CHD-RM state.
- Cloud runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3i-fam2-open-value-distillability`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711/`.
- Parent runnable code: v3h operator-site context audit branch at `0d0d79e`.

## Stage History

v3i-A tested whether the privileged open-value teacher target is spatially
compressible. It passed: hard D7c replay was only `+0.012784 dB`
with `23` severe regressions, while GT open top50
oracle was `+0.411695 dB` with zero severe regressions,
and best compressed `ALPHA_SECANT_Q3` was `+0.412879 dB`
with zero severe regressions.

v3i-B tested full-context single-forward OOF features. It failed: best OOF
`OOF_DW3X3_TOP50` reached `+0.016627 dB`,
but its bootstrap delta vs hard had CI low
`-0.000163` and no minimum or strong pass.

v3i-C tested counterfactual RGB response OOF features. It failed: best OOF
`OOF_CF_RESPONSE_DW3X3_TOP50` reached `+0.008543 dB`,
with bootstrap delta vs hard mean `-0.004241`
and CI low `-0.009492`.

v3i-D tested checkpoint and transform disagreement OOF features. It failed: best
OOF `OOF_DISAGREE_LINEAR_TOP50` reached `+0.010534 dB`,
with bootstrap delta vs hard mean `-0.002250`
and CI low `-0.007422`.

## Final Interpretation

The bottleneck is not FAM2 actuator realizability, not privileged open-value
oracle strength, and not spatial compressibility of the target. The bottleneck
is the missing deployable inference-time controller signal. Multiple independent
signal families failed OOF replay against hard D7c while the privileged oracle
remained strong.

## Forbidden Flow After Closeout

- No v3d continuation.
- No v3f-B scalar ranker training.
- No controller/router/distillation training from the current v3h/v3i signal sets.
- No canary expansion.
- No 20-epoch continuation.
- No v4/RARM expansion from this evidence.
- No backbone/FAM1/neighbor unfreeze from this evidence.
- No locked Haze4K test.
- No checkpoints, weights, raw arrays, images, or per-image raw tables in GitHub main evidence.

## Next Allowed Shape

Stop the FAM2 router/distillation route. The next route should redesign the
correction/controller formulation, such as joint correction-confidence or
bounded experts with explicit false-action protection. It must pass a
no-training held-out separability/replay gate before any training is authorized.
