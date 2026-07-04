# Haze4K v2.21 NoPost Safety-Calibrated Lowband Replay

Date: 2026-07-04

Status: completed gate-pass review-only

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

## Closeout

Decision:

```text
V221_P1_REPLAY_GATE_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED
```

Runtime source:

- cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay`
- branch: `codex/haze4k-v2-21-nopost-safety-calibrated-lowband-replay`
- run commit: `c652e86`
- evidence root: `experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/`

Results:

- P0 passed. v2.21 remains source-clean, zero-init identity preserved, and the route is replay-only.
- Raw v2.20 O3 action remained unsafe: mean `+2.0911 dB`, hard `+4.2277 dB`, easy `+0.5341 dB`, p05 `-0.7040 dB`, CVaR5 `-1.4501 dB`, severe rate `11.00%`, strong-reference regression rate `25.17%`.
- Selected fixed OOF gate was `V221_risk_temperature_gamma0p50`.
- Selected replay passed all P1 safety gates: mean `+2.2270 dB`, hard `+4.3031 dB`, easy `+0.7403 dB`, positive ratio `0.9479`, p05 `-0.0025 dB`, CVaR5 `-0.2089 dB`, severe rate `1.79%`, strong-reference regression rate `4.83%`, fold tail pass `5/5`.
- Shuffled-risk control failed, so the risk signal is not interchangeable noise: mean `+1.5726 dB`, p05 `-0.4712 dB`, CVaR5 `-1.2710 dB`, severe rate `7.83%`.
- Factorial audit passed:
  - A predicted action + predicted gate: mean `+2.2270 dB`, severe `1.79%`;
  - B predicted action + oracle gate: mean `+2.1915 dB`, severe `0`;
  - C oracle action + predicted gate: mean `+6.3872 dB`, severe `0`;
  - D oracle action + oracle upper bound: mean `+7.1107 dB`, severe `0`.
- P2 calibration was structured but not the sole authorization source: ROC AUC `0.9239`, PR AUC/AP `0.6976`, Brier `0.1143`, ECE10 `0.1591`.
- P3 still reports post-gate residual tail damage (`43/2400` severe cases), so N3 must carry this tail audit forward.
- P4 passed objective replay as guard evidence only.

Training and locked-test policy:

- training authorized for N3 route-card review: `true`
- training launched: `false`
- locked Haze4K touched: `false`

Next decision:

Write a separate N3 microfit route card for `V221_risk_temperature_gamma0p50`. This v2.21 replay does not itself launch training and does not authorize locked-test use.
