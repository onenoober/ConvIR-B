# v3s Zero-Init Delta-u Direction Repair Evidence

Status: `COMPLETED_GATE_PASS` for S0; S1 fixed-32 trainability scout is authorized.

Cloud raw runtime root:
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713`.

This directory receives only compact stage closeouts, source manifests, fixed
training histories, and formal aggregate summaries. Checkpoints, raw image and
block tables, model outputs, and runtime logs remain in the cloud runtime root.

## S0 Exact No-op Smoke

`v3s_s0_noop32_r4` passed on the 32 fixed train-derived OOF names for both
frozen operators. `Delta u` and new-vs-old rendered prediction difference were
exactly zero, and the maximum old `.125` reference replay difference was
`0.0 dB` against the fixed `1e-6 dB` tolerance. It performed no training and
touched neither canary nor locked test. The closeout authorizes S1 only.
