# Haze4K v5 CHD-RM v3m Blockwise Counterfactual Advantage

Date: 2026-07-11

Status: `PREFLIGHT_A0A_COMMON_ACTION_ONLY`

Branch: `codex/haze4k-v5-v3m-blockwise-counterfactual-advantage`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/`

## Route Identity

v3m is a new diagnostic audit continuation of v3l, not a repair of the failed
v3l-B transmission policy. Its first purpose is to determine whether block16
remains the best deployable control granularity when image, block, and pixel
oracles all use the same discrete action set.

Parent evidence is GitHub `main` commit `5acaaa54d7aca7c9764dc3dd757ff58cdf6d96fa`.
The runnable source parent is cloud commit `0031b66799ce44574c555fa7bfb879cd5394b991` plus
the frozen v3l artifacts and analysis source recorded in the v3m source manifest.

## A0a Objective

Compare image, block32, block16, and pixel-grid privileged oracles using only
the common ladder `{0, 0.125, 0.25, 0.5, 1.0}`. The primary comparison is
relative to fixed `alpha=0.125` on the same train-derived clean-reference
grouped OOF rows for both frozen operators.

## Forbidden

- no Haze4K locked-test access;
- no canary;
- no controller, router, threshold, backbone, or direct-head training;
- no action-set expansion, post-hoc denominator change, or route-confirm
  strategy selection;
- no use of dense-grid or continuous pixel results to rescue a common-action
  gate failure;
- no checkpoint, tensor, image, raw per-image, or raw block table GitHub sync.

## A0a Gate

For both `D_ref` and `D_rep`, block16 must satisfy all of:

- clean-reference-grouped paired mean-lift CI95 low greater than zero;
- common-action retention CI95 low at least `0.80` relative to pixel-grid lift
  beyond fixed `alpha=0.125`;
- p10 and worst PSNR delta no lower than fixed `alpha=0.125`;
- severe count at `<= -0.2 dB` no higher than fixed `alpha=0.125`.

Pass authorizes only A0b dense-grid and continuous-pixel mechanism audits. Fail
records `V3M_A0_BLOCK16_GRANULARITY_LOCK_FAIL_NO_BLOCK16_CONTROLLER` and
authorizes neither a block controller nor physics policy work.

## Source And Runtime Contract

- Cloud worktree:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711`
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Inputs: frozen v3l `D_ref`/`D_rep` artifacts, exact A0 closeout, v3j split
  manifest, official Haze4K base checkpoint, D7c and density artifacts.
- Dataset: Haze4K `train` only. The v3j route-confirm panel is audit-only.
- Output: cloud-only per-image rows under `cloud_only_raw_common_action/` and
  compact summary/gate artifacts in this evidence root.

## Next Stage

No stage is authorized before A0a completes.
