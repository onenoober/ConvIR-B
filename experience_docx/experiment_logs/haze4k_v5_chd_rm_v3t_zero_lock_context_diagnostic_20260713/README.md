# v3t Zero-Lock Versus Context Diagnostic Evidence

Status: `COMPLETED_GATE_PASS` diagnostic; no v3t formal training is authorized.

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

## S1 Four-cell Diagnosis

All four fixed 16-epoch cells completed with finite gradients but failed the
predeclared activity line (`|Delta u| >= 1e-6` and `.25` rendered-loss
reduction >= `0.1%`). Final `|Delta u|` was `2.72e-7` output-safe,
`2.40e-7` output-utility, `1.82e-7` context-safe, and `2.07e-7`
context-utility; relative rendered-loss reductions were only roughly
`0.00013%`, `0.00011%`, `0.000075%`, and `0.000026%`.

The v3t utility cells removed anchor/harm/CVaR but intentionally retained the
v3s minimal-repair penalty. The result therefore closes this regularized
rendered optimization form and requires a new activation-objective diagnostic;
it does not authorize formal training, policy work, canary, or locked test.
