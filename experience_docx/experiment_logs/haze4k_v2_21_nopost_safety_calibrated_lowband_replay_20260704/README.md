# Haze4K v2.21 NoPost Safety-Calibrated Lowband Replay

Status: `COMPLETED_GATE_PASS_REVIEW_ONLY`

This evidence root belongs to route branch `codex/haze4k-v2-21-nopost-safety-calibrated-lowband-replay`.

Runtime validation is cloud-only on `convir-4090`. Local WSL is used only for editing and syntax/compile checks.

Declared stages:

- P0: contract, forbidden-symbol scan, official checkpoint partial load, zero-init identity.
- P1: safety-gated replay, fixed OOF threshold sweep, and factorial action/gate audit.
- P2: safety score calibration and fold stability.
- P3: post-gate action-shape residual audit.
- P4: objective replay after safety gating.

Initial policy:

```text
No training.
No N3 microfit.
No locked Haze4K test.
```

Closeout decision is pending cloud replay evidence.

Closeout decision:

```text
V221_P1_REPLAY_GATE_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED
```

Key result:

- P0 passed contract, source-clean scan, official checkpoint partial load, and zero-init identity.
- P1 safety-gated replay passed for fixed OOF candidate `V221_risk_temperature_gamma0p50`.
- Selected replay metrics: mean `+2.2270 dB`, hard bottom25 `+4.3031 dB`, easy top25 `+0.7403 dB`, positive ratio `0.9479`, p05 `-0.0025 dB`, CVaR5 `-0.2089 dB`, severe rate `1.79%`, strong-reference regression rate `4.83%`, fold tail pass `5/5`.
- Raw v2.20 action still failed safety: p05 `-0.7040 dB`, CVaR5 `-1.4501 dB`, severe rate `11.00%`, strong-reference regression rate `25.17%`, fold tail pass `0/5`.
- Factorial audit passed for A/B/C/D; predicted action + predicted gate passed, so the route can proceed to a separate N3 microfit route-card review.
- P2 found structured safety scores: ROC AUC `0.9239`, PR AUC/AP `0.6976`, Brier `0.1143`, ECE10 `0.1591`.
- P3 still found residual post-gate tail shape damage: `43/2400` severe cases remained, so N3 must preserve this risk audit rather than treating replay pass as solved.
- P4 passed as guard evidence only.

Training and locked-test policy:

- training authorized for route-card review: `true`
- training launched: `false`
- locked Haze4K touched: `false`

Next decision:

Write a separate N3 microfit route card for `V221_risk_temperature_gamma0p50`. Do not auto-train from this route and do not touch locked Haze4K test.
