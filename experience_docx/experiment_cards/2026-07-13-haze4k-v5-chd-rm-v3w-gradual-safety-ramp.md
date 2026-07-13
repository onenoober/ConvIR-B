# Haze4K v5 CHD-RM v3w Gradual Safety Ramp

Date: 2026-07-13

Status: `V3W_S1_RAMP_LOSES_ACTIVITY_STOP_GRADUAL_RAMP`.

Fresh anchor route after v3v. It preserves the v3v output-side head, frozen
assets, 32 names, seed, 16 epochs, and all gates. Epochs 1-8 optimize rendered
`.25` MSE only. Epochs 9-16 linearly ramp the v3s anchor/margin/harm/CVaR
weights from `1/8` to `1`, with repair weight fixed at zero. Rules commit:
`github/main@8815d2849f8748e0484b8a2e9a0bb65ad938caab`; anchor:
`github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.

S0 passed exact no-op over all 32 fixed names and both frozen operators. S1
passed the epoch-8 warmup gate (`|Delta u|=0.00134179`, rendered-MSE reduction
`1.12999%`), then failed its final activity gate: final rendered MSE was
`0.00032232537` versus initial `0.00032214251` (relative reduction
`-0.05676%`), although anchor, harm, and margin were all no worse than their
fixed v3u references. The fixed linear ramp is stopped. Do not search another
fixed safety-weight schedule from this result; any fresh successor must use a
materially different direct low-haze-safety mechanism. No policy, canary,
formal candidate training, or locked test is allowed.
