# Haze4K v2.6 Residual Shrinkage Alpha Curves

Date: 2026-06-16

Status: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`

## Purpose

This route supplements the StrongExpert-GainMix evidence with explicit
train-derived residual shrinkage curves. It tests whether WD0375 is an isolated
fixed-alpha lucky point or part of a broader anchor-preserving shrinkage
pattern, and whether the same phenomenon appears for multiple strong experts.

## Governance

- Start point: `b163819` plus v2.6 evaluation scripts.
- Runtime host: `convir-4090`.
- Local checkout use: editing and compile/syntax checks only.
- Runtime scope: C8 train-derived `val_regular + val_hard` split.
- Locked Haze4K: forbidden. Prior locked WD0375/C11 results are not used for
  selection or tuning in this route.
- Architecture: unchanged; no ConvIR-B model structure modification.

## Fixed Evaluation Grid

```text
candidate(E, alpha) = A0 + alpha * (E - A0)

E:
- WDMamba
- FSNet+UDP
- MB-TaylorFormerV2-L

alpha:
- 0
- 0.125
- 0.25
- 0.375
- 0.50
- 0.75
- 1.0
```

Metrics:

```text
mean dPSNR
hard bottom-25 dPSNR
easy top-25 dPSNR
dSSIM
positive ratio
nonnegative ratio
severe / 600
worst-case dPSNR
group-min metrics across split and feature quartiles
```

## Success Readout

This route is not a promotion or locked-test route. It is considered useful if
it answers:

- whether WDMamba has a stable safety/utility interval around medium alpha;
- whether full expert alpha `1.0` increases tail risk relative to shrinkage;
- whether FSNet+UDP and/or MB-TaylorFormerV2-L show the same shrinkage pattern;
- whether the evidence should be stated as a general strong-expert residual
  shrinkage phenomenon or narrowed to WDMamba-style experts.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616/`

## Result

The parallel `convir-4090` run completed from source commit
`ee059bcb5278a878e8e3b1bf153dbd8bfd01eaaf` with locked Haze4K untouched.

Decision: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`.

Main train-derived result on C8 `val_regular + val_hard`:

- WDMamba has a broad safe shrinkage interval: alpha `0.125/0.25/0.375/0.50`
  all keep positive ratio at least `0.95` and severe no more than `22/600`.
  `WD0375` is therefore not an isolated lucky point. Full alpha `1.0` gives
  higher mean/hard gain but damages easy cases (`-1.048537 dB`) and raises
  severe regressions to `124/600`.
- FSNet+UDP also supports residual shrinkage: alpha `0.125/0.25/0.375/0.50/0.75`
  are positive/tail-safe by the route readout, while full alpha `1.0` drops
  positive ratio to `0.858333` and raises severe regressions to `71/600`.
- MB-TaylorFormerV2-L supports only a narrow small-alpha safety claim: alpha
  `0.125` has mean/hard/easy `+0.485463/+0.653630/+0.259786`, positive
  `0.905000`, severe `21/600`, but alpha `0.375` already reaches severe
  `99/600`, and full alpha has easy `-3.255472` with severe `294/600`.

This result strengthens the Haze4K train-derived mechanism claim from a single
fixed WDMamba alpha to a residual-shrinkage pattern for WDMamba and FSNet+UDP.
It does not claim cross-dataset transfer or adaptive-alpha deployment.
