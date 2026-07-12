# v3s Zero-Init Delta-u Direction Repair Evidence

Status: `COMPLETED_GATE_FAIL`; the written v3s low-capacity representation/loss contract is stopped.

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

## S1 Fixed-32 Trainability Scout

`v3s_s1_scout32` completed with finite loss and gradients but failed its
predeclared activity gate: final mean `|Delta u|` was `1.252e-7`, below
`1e-6`, and rendered `.25` loss `0.0003221426341042388` did not beat the
initial `0.0003221424813091289`. No formal five-fold training is authorized.
This stops this low-capacity input plus safety-loss contract; it is not a
