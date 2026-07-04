# Haze4K v2.20 NoPost Mid+Final Context Lowband Learnability

Date: 2026-07-03

Status: completed gate-fail pause

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

## Closeout

Decision:

```text
V220_P1A_PASS_P1B_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING
```

Runtime source:

- cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-20-nopost-midfinal-context-lowband-learnability`
- branch: `codex/haze4k-v2-20-nopost-midfinal-context-lowband-learnability`
- run commit: `7dfa495`
- evidence root: `experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703/`

Results:

- P0 passed. Zero-init max abs versus A0 was `0.0`; official params loaded and only `nopost_midfinal_context_policy.*` keys were missing.
- P1-A passed for `P1_final_mid_global_context_predictor`: mean `+2.0684`, hard `+4.1450`, easy `+0.5199`, positive ratio `0.8508`, wrong-direction `0.00417`, and real-vs-shuffled gap `+3.1959`.
- P1-B failed safety: p05 `-0.7255`, CVaR5 `-1.6967`, severe rate `11.125%`, strong-reference regression rate `26.67%`, and fold tail pass `0/5`.
- P2 was diagnostic-positive for unsafe-action/no-op classification: easy/strong unsafe recall `0.90625`, predicted no-op mean dPSNR `-0.0300`.
- P3 diagnosed that remaining tail damage is not mainly explained by wrong direction or peakiness alone.
- P4 passed objective replay as guard evidence only; it does not authorize training.

Training and locked-test policy:

- training launched: `false`
- training authorized: `false`
- locked Haze4K touched: `false`

Next decision:

Do not train this v2.20 O3 context predictor. If the NoPost lowband direction is reopened, the next route should target the remaining safety/no-op calibration and tail preservation gap before any N3 microfit.
