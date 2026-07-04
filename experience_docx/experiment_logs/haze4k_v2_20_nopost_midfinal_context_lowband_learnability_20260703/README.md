# Haze4K v2.20 NoPost Mid+Final Context Lowband Learnability

Status: `COMPLETED_GATE_FAIL`

This evidence root belongs to route branch `codex/haze4k-v2-20-nopost-midfinal-context-lowband-learnability`.

Runtime validation is cloud-only on `convir-4090`. Local WSL is used only for editing and syntax/compile checks.

Declared stages:

- P0: contract, forbidden-symbol scan, official checkpoint partial load, zero-init identity.
- P1: O3 mid+final/context action learnability with controls.
- P2: no-op/action classifier audit.
- P3: action-shape decomposition.
- P4: objective replay for O3 context predictions.

Closeout decision:

```text
V220_P1A_PASS_P1B_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING
```

Key result:

- P0 passed contract, source-clean scan, official checkpoint partial load, and zero-init identity.
- P1-A mechanism gate passed for `P1_final_mid_global_context_predictor`.
- P1-B training-authorization safety gate failed, so no N3/N4 training was launched.
- P2 found a useful no-op/unsafe-action classifier signal.
- P3 found remaining tail damage not explained mainly by wrong direction or local peak shape alone.
- P4 objective replay passed as a guard audit only.

Primary P1 metrics:

| Metric | Value |
| --- | ---: |
| mean dPSNR | `+2.0684` |
| hard bottom25 dPSNR | `+4.1450` |
| easy top25 dPSNR | `+0.5199` |
| positive ratio | `0.8508` |
| p05 dPSNR | `-0.7255` |
| CVaR5 dPSNR | `-1.6967` |
| severe rate | `0.11125` |
| strong-reference regression rate | `0.2667` |
| wrong-direction rate | `0.00417` |
| fold tail pass | `0/5` |

Interpretation:

O3 mid+final/global context makes the action substantially more learnable than v2.19 final-only, but the tail/easy/strong safety gap remains too large for training authorization. Locked Haze4K test was not touched and training was not launched.

GitHub main sync note:

- This compact evidence root includes the route README, decisions, closeout JSON, status, reproducibility scripts, fold/control/replay summaries, P2/P3/P4 diagnostic reports, the v2.19 tail-rescue manifest, and the P3 tail-case shape manifest.
- The large 2400-row per-image `v220_p1_o3_target_energy_summary.csv` and `v220_p1_o3_minus_o2_residual_summary.csv` remain on the cloud runtime workspace and are not part of the main evidence archive.
- No checkpoints, weights, datasets, images, raw inference outputs, raw replay metrics, or locked-test outputs are included.
