# Haze4K v2.20 NoPost Mid+Final Context Lowband Learnability

Date: 2026-07-03

Status: planned

## Scope

- Project: ConvIR-B Haze4K NoPost lowband policy.
- Model family: NoPost feature-lowband action policy.
- Dataset or task: Haze4K train-derived internal split only.
- Primary objective: test whether O3 mid+final/context signals make lowband actions safely learnable.
- Main metric: replay dPSNR versus official A0.
- Secondary metrics: hard/easy buckets, p05, CVaR5, severe rate, strong-reference regressions, action direction/shape, no-op classification.
- Execution environment: `convir-4090` only for runtime validation.
- Artifact root: `experience_docx/experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703/`.
- Branch or isolated workspace: `codex/haze4k-v2-20-nopost-midfinal-context-lowband-learnability`.

## Hypothesis

v2.19 proved the O2 final-only spatial lowband policy can move hard samples but cannot preserve easy/strong/tail samples. The v2.20 question is whether O3 mid+final feature-lowband plus context tokens provide enough information to identify no-op regions and safer local action shape.

Mechanism sentence:

```text
If mid-level feature-lowband and global context are added to the final lowband state,
the deployable predictor should preserve easy/strong/tail samples while keeping a
meaningful portion of the O2/O3 hard-sample oracle headroom.
```

## Change

- New model class: `NoPostMidFinalContextLowbandConvIR`.
- Insertions:
  - zero-init mid feature LL policy after decoder stage 1;
  - zero-init final feature LL policy after decoder stage 2;
  - final policy can see final LL, mid context, and a global token.
- Explicitly disabled:
  - output-output deltas;
  - RGB output-level learned correction;
  - A0, WD0375, WDMamba, teacher, or expert outputs as forward inputs;
  - locked Haze4K test.
- Initialization: zero projectors make the model exactly A0 at launch.

## Stages And Gates

| Stage | Purpose | Continue rule |
| --- | --- | --- |
| P0 | contract, forbidden scan, official checkpoint partial load, zero-init identity | pass before any P1 runtime audit |
| P1-A | O3 mechanism learnability versus controls | may continue diagnostics, but no training authorization |
| P1-B | training-authorization safety gate | only if this passes may a separate N3 microfit route-card review be written |
| P2 | no-op/action classifier audit | diagnostic only |
| P3 | action-shape decomposition | diagnostic only |
| P4 | objective replay for O3 predictions | guard audit only |

P1-A mechanism gate:

```text
mean dPSNR >= +0.25
hard bottom25 >= +0.50
positive_ratio >= 0.60
real beats shuffled by >= +0.20
wrong-direction rate <= 0.12
target MSE beats shuffled and final-only baseline
```

P1-B training-authorization safety gate:

```text
easy top25 >= -0.02
p05 >= -0.15
CVaR5 >= -0.35
severe_rate <= 0.035
strong-reference regression rate <= 0.075
fold tail pass >= 4/5
strong/easy mean >= 0
strong/easy p05 >= -0.15
```

## Controls

- exact O2 final-feature LL oracle upper bound;
- exact O3 mid+final LL oracle upper bound;
- v2.19-style final-only spatial CNN replicate;
- mid-only predictor;
- final+mid predictor;
- final+mid+global context predictor;
- shuffled-target control;
- global-broadcast control;
- no-op control.

## Stop Rules

- If P0 fails, stop as engineering/preflight failure.
- If P1-B fails, do not train; finish P2/P3/P4 diagnostics and pause normally.
- P4 guard pass does not authorize training without P1-B.
- Locked Haze4K test remains untouched throughout this route.
