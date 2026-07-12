# v3u Render-Only Activation Evidence

Status: `COMPLETED_GATE_PASS` activation diagnostic; only safety-curriculum
contract design is authorized.

Raw cloud runtime root:
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3u_render_only_activation_diagnostic_20260713`.

This route keeps the v3t output-side zero-init bounded `DIRT_*` form and
fixed-32 rendered `.25` activity gate, but removes the minimal-repair penalty
entirely. S0 must establish exact no-op before S1 is launched. S1 can authorize
only a new safety-curriculum contract design if both predeclared activity
criteria pass; it cannot authorize formal training, policy work, canary, or
locked-test access.

## S0 Output-Side Exact No-op

`v3u_s0_noop32` passed 32 fixed OOF names x two frozen operators. Maximum
`Delta u` and new-vs-old prediction difference were exactly `0.0`; fixed `.125`
reference replay difference was `0.0 dB` against the `1e-6 dB` tolerance.
No training, canary, or locked-test operation occurred.

## S1 Render-Only Activation

`v3u_s1_render_only32` optimized only real rendered `.25` MSE for 16 epochs.
It passed both fixed activity lines: final mean `|Delta u|=0.0041701270` and
relative MSE reduction `2.92747396%` (`0.00032214251 -> 0.00031271187`). The
history confirms `total == render` on all epochs and the source manifest fixes
`repair_weight=0.0`.

This is activation evidence, not safety evidence. Anchor/harm/margin increased
from `0`, `2.8721e-7`, and `2.2540e-6` to `6.7448e-7`, `2.2907e-6`, and
`7.8329e-6`; repair magnitude reached `0.445731`. No formal training, policy,
canary, or locked-test operation is authorized from this route.

Only compact manifests, closeouts, summary, history, and this README belong in
this directory. Cloud checkpoints, images, raw per-image tables, and runtime
logs remain outside Git.
