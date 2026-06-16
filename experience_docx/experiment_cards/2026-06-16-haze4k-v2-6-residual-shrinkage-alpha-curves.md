# Haze4K v2.6 Residual Shrinkage Alpha Curves

Date: 2026-06-16

Status: `PLANNED_LOCKED_TEST_UNTOUCHED`

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
