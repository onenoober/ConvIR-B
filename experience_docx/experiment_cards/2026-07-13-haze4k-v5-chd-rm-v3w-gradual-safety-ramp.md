# Haze4K v5 CHD-RM v3w Gradual Safety Ramp

Date: 2026-07-13

Status: `PLANNED`.

Fresh anchor route after v3v. It preserves the v3v output-side head, frozen
assets, 32 names, seed, 16 epochs, and all gates. Epochs 1-8 optimize rendered
`.25` MSE only. Epochs 9-16 linearly ramp the v3s anchor/margin/harm/CVaR
weights from `1/8` to `1`, with repair weight fixed at zero. Rules commit:
`github/main@8815d2849f8748e0484b8a2e9a0bb65ad938caab`; anchor:
`github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.

S0 requires exact no-op. S1 requires the same warmup and final activity gates
as v3v, plus final anchor/harm/margin each no worse than fixed v3u references.
Pass authorizes safety-training-contract design only. Failure stops this ramp;
no policy, canary, formal candidate training, or locked test is allowed.
