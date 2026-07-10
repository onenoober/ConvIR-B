# Haze4K CHD-RM v3a D7c-Gated No-Op Connection Audit

Status: `COMPLETED_GATE_PASS`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3a_d7c_gated_noop_connection_audit_20260710/`

Source branch:
`github/codex/haze4k-official-arch-anchor`

Route branch:
`codex/haze4k-v5-v3a-d7c-gated-noop-connection-audit`

## Route Identity

v3a is a no-training architecture connection audit authorized by v2i. It tests
whether real D7c gate tensors can enter the FAM2 modulation path while the final
candidate remains mathematically no-op and exactly A0-equivalent.

This is not RARM, not training, and not a candidate-quality experiment.

## Architecture Contract

- Start from the official ConvIR-B Haze4K architecture anchor.
- Add `fam_mode='fam2_d7c_noop'`.
- Keep FAM1 original.
- FAM2 has the same zero-initialized `1x1` gamma/beta modulator shape as v2i.
- D7c gate is an external tensor passed into FAM2.
- D7c gate is resized to FAM2 scale and externally multiplies gamma/beta.
- Since gamma/beta are zero initialized, final output must equal A0 exactly.

Allowed new checkpoint-missing keys:

```text
FAM2.modulator.weight
FAM2.modulator.bias
```

Expected parameter delta: `8320`.

## Runtime Assets

- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- D3 density head: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt`
- D7c top-k head: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt`
- Split: train-derived internal `2400/600`; no locked Haze4K test.

## Metric Contract

Compare A0 original against `fam2_d7c_noop` on:

- random tensor;
- real train-derived batch;
- internal val-inner 600.

Pass gates:

- candidate missing keys exactly match the allowed list;
- unexpected keys and shape mismatches are empty;
- parameter delta is exactly `8320`;
- FAM2 modulator weights and bias are all zero;
- D7c gate has nontrivial coverage on real/internal samples;
- output max absolute difference is `<= 1e-7`;
- PSNR and SSIM deltas on internal val-inner 600 are `<= 1e-10`;
- no locked test, no training, no RARM, no adapter, no ConvIR-B unfreeze.

## Stop Rules

Pause immediately if the cloud assets are missing, the D7c gate is trivial, the
candidate is not exact no-op equivalent, or any command path touches locked test.

## Decision

`V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY`

The cloud attempt 5 audit passed. D7c gate tensors are connected into FAM2 as an
external gate tensor, while the zero-initialized gamma/beta modulation remains
exact no-op and A0-equivalent.

Final gate summary:

- expected missing candidate checkpoint keys:
  `FAM2.modulator.weight`, `FAM2.modulator.bias`;
- unexpected keys and shape mismatches: none;
- parameter delta: `8320`;
- random and real-batch no-op checks: pass;
- internal val-inner 600 output max absolute diff: `0.0`;
- internal val-inner 600 PSNR/SSIM delta max absolute values: `0.0`;
- nontrivial D7c gate images: `599/600`;
- locked Haze4K test, training, RARM, adapter training, and ConvIR-B unfreeze:
  not used.

This authorizes only a subsequent written preflight/design decision. It does not
authorize RARM, training, adapter experiments, canary expansion, or locked-test
access.
