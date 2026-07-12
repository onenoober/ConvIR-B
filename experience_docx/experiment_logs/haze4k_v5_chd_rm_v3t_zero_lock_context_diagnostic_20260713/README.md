# v3t Zero-Lock Versus Context Diagnostic Evidence

Status: `COMPLETED_GATE_PASS` for S0; S1 factorial diagnostic is authorized.

Raw cloud runtime root:
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3t_zero_lock_context_diagnostic_20260713`.

Only compact no-op and factorial closeouts, source manifests, summaries, and
histories belong here. Cloud checkpoints, images, raw per-image tables, and
runtime logs are excluded.

## S0 Dual-form Exact No-op

`v3t_s0_noop32` passed all 32 fixed OOF names on both frozen operators and
both output-side/context forms. Across the 256 candidate checks, `Delta u` and
new-vs-old prediction difference were exactly zero, and the fixed `.125`
reference replay maximum was `0.0 dB` against `1e-6 dB`. No training, canary,
or locked-test operation occurred. The closeout authorizes S1 only.
