# Haze4K CHD-RM v3c Gate Forward Contract

Date: 2026-07-10

Status: `COMPLETED_GATE_PASS`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3c_gate_forward_contract_20260710/`

Parent route branch:
`github/codex/haze4k-v5-v3b-rarm-preflight-design`

Route branch:
`codex/haze4k-v5-v3c-gate-forward-contract`

## Route Identity

v3c is the no-training forward-contract continuation required by v3b. It fixes
the entrypoint contract blocker only:

- build a frozen D7c gate producer from the already validated D3 density head,
  D7c need head, and official A0 checkpoint;
- pass the resulting gate into `fam2_d7c_noop` through train/valid/eval and
  modulation-stat helpers;
- allow official A0 partial initialization for the two zero-init FAM2 modulator
  keys.

This is not RARM, not adapter training, not ConvIR-B unfreeze, not a loss
change, and not a candidate-quality experiment.

## Fact Sources

- GitHub `main` CHD-RM index at v3b closeout.
- v3b route branch at `d5a9f76c488feb4d84325bd1a5cc0f4634cb3fb7`.
- v3a D7c-gated no-op connection audit and its gate-generation script.
- Current v3c source tree for entrypoint contract changes.

## Runtime Assets

- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Split:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`
- D3 density head:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt`
- D7c top-k head:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt`

## Metric Contract

The cloud no-training audit passes only if:

- source contract checks confirm train/valid/eval/modulation-stat helpers use
  the D7c gate path;
- official A0 partial init misses exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- the D7c gate producer is nontrivial on at least one checked internal sample;
- candidate outputs remain A0-equivalent with max absolute diff `<= 1e-7`;
- PSNR/SSIM deltas remain `<= 1e-10` on the checked internal samples;
- modulation stats include D7c gate stats;
- no locked test, training, RARM, adapter, ConvIR-B unfreeze, or canary
  expansion occurs.

The audit is intentionally small: `16` internal val-inner samples are enough to
verify the entrypoint contract because v3a already established the full val600
no-op equivalence for the same gate semantics.

## Stop Rules

Pause if any required asset is missing, the source contract is not wired, the
gate is trivial on all checked samples, the no-op equivalence breaks, or any
command path touches locked test.

## Decision

`V3C_GATE_FORWARD_CONTRACT_PASS_AUTHORIZE_NO_TRAINING_ENTRYPOINT_PREFLIGHT_ONLY`

The cloud no-training audit passed on `convir-4090` using a fresh workspace at
commit `0a350393776c4263386c72c8b81be076d9d984a5`.

Key evidence:

- source contract checks passed;
- official A0 partial init missed exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- checked samples: `16` internal val-inner images;
- nontrivial D7c gate images: `16/16`;
- D7c selected coverage mean/min/max:
  `0.3246304675703868` / `0.015908146277070045` /
  `0.6701125502586365`;
- output max absolute difference: `0.0`;
- PSNR/SSIM max absolute deltas: `0.0` / `0.0`;
- modulation stats include D7c gate stats;
- training, RARM, adapter training, ConvIR-B unfreeze, canary expansion, and
  locked Haze4K test were not used.

This closes the v3b entrypoint-contract blocker. It still does not authorize
RARM/training. Any next step toward RARM or training must be written as a
separate decision with its own resource and metric contract.
