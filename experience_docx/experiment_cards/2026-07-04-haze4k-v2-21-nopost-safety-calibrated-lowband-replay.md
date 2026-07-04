# Haze4K v2.21 NoPost Safety-Calibrated Lowband Replay

Date: 2026-07-04

Status: planned

## Scope

- Project: ConvIR-B Haze4K NoPost lowband policy.
- Model family: NoPost feature-lowband action policy.
- Dataset or task: Haze4K train-derived internal split only.
- Primary objective: test whether v2.20 unsafe/no-op signals can act as an internal action controller before any training.
- Main metric: safety-gated replay dPSNR versus official A0.
- Secondary metrics: hard/easy buckets, p05, CVaR5, severe rate, strong-reference regressions, calibration, threshold stability, factorial action/gate decomposition.
- Execution environment: `convir-4090` only for runtime validation.
- Artifact root: `experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/`.
- Branch or isolated workspace: `codex/haze4k-v2-21-nopost-safety-calibrated-lowband-replay`.

## Hypothesis

v2.20 showed that O3 mid+final/global context can learn strong feature-lowband actions, but P1-B failed because tail/easy/strong no-op calibration remained unsafe. v2.21 narrows the question:

```text
Can the train-derived unsafe/no-op signal from v2.20 gate or shrink the O3 action
so that p05/CVaR/severe/strong tail failures are repaired without collapsing
mean/hard gain to near-identity?
```

## Change

- Reuse v2.20 `NoPostMidFinalContextLowbandConvIR` as the action source.
- Add no deployed model training.
- Add replay-only safety controller audit:
  - hard risk-zero gating;
  - soft shrink gating;
  - piecewise shrink gating;
  - risk-temperature scaling;
  - shuffled-risk control;
  - oracle-risk upper bound;
  - 2x2 factorial audit for predicted/oracle action and predicted/oracle gate.
- Explicitly disabled:
  - direct v2.20 training;
  - N3 microfit unless v2.21 replay passes;
  - output-output deltas;
  - RGB output-level learned correction;
  - A0, WD0375, WDMamba, teacher, or expert outputs as forward inputs;
  - locked Haze4K test.

## Stages And Gates

| Stage | Purpose | Continue rule |
| --- | --- | --- |
| P0 | contract, forbidden scan, official checkpoint partial load, zero-init identity | pass before replay |
| P1 | safety-gated replay and factorial action/gate audit | only if fixed OOF candidate passes may N3 route-card review be considered |
| P2 | safety score calibration and fold stability | diagnostic only |
| P3 | post-gate action-shape residual audit | diagnostic only |
| P4 | objective replay after safety gating | guard audit only |

P1 training-authorization replay gate:

```text
mean dPSNR >= +1.00
hard bottom25 >= +2.00
easy top25 >= 0
positive_ratio >= 0.75
p05 >= -0.15
CVaR5 >= -0.35
severe_rate <= 0.035
strong-reference regression rate <= 0.075
fold tail pass >= 4/5
strong/easy p05 >= -0.15
no-op bin mean dPSNR >= -0.03
unsafe high-probability bin severe rate clearly reduced
factorial A predicted-action + predicted-gate also passes
```

## Factorial Audit

```text
A. predicted action + predicted risk gate
B. predicted action + oracle unsafe gate
C. oracle action + predicted risk gate
D. oracle action + oracle gate upper bound
```

Interpretation:

- If B passes and A fails, the action is usable but predicted gate calibration is weak.
- If C passes and A fails, the predicted gate has value but action shape/local error remains weak.
- If B and C both fail, v2.20 action and safety signal are insufficient together.
- If A passes, write a separate N3 microfit route-card review; do not auto-train.

## Stop Rules

- If P0 fails, stop as engineering/preflight failure.
- If P1 fails, pause normally and do not train.
- P2/P3/P4 cannot authorize training by themselves.
- Fold-specific thresholds are diagnostic only; final candidate must be fixed OOF.
- Locked Haze4K test remains untouched throughout this route.
