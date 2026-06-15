# C8-Mini Decision

Decision: `C8_PASS_COMPLEMENTARITY_PROVEN_AUTHORIZE_C9_ROUTER_DESIGN_ONLY`

C8-Mini completed on `convir-4090` using only train-derived `val_regular` + `val_hard` splits. No MoE/router training was run, no distillation was run, and locked Haze4K test remained untouched.

## Key Results

| Stage | Mean gain over S0 | Hard-bottom25 gain over S0 | Severe selected oracle | Decision |
| --- | ---: | ---: | ---: | --- |
| S1 WDMamba | 2.824226 | 4.453624 | 0 | PASS |
| S2 WDMamba+FSNet+UDP | 3.116570 | 4.473811 | 0 | PASS |
| S3 +MB-TaylorFormerV2-L | 3.158518 | 4.559721 | 0 | PASS |

## Unique-Win Evidence

- WDMamba single-expert hard/red-flag unique-win rate over S0: `0.982143`.
- FSNet+UDP single-expert hard/red-flag unique-win rate over S0: `0.895833`.
- MB-Taylor single-expert hard/red-flag unique-win rate over S0: `0.857143`.
- S3 unique wins vs all other experts on hard/red-flag scope: WDMamba `0.806548`, FSNet+UDP `0.113095`, MB-Taylor `0.074405`.

## Removal Ablation

- Removing WDMamba from S3 drops mean/hard by `1.181882` / `1.966369` dB.
- Removing FSNet+UDP from S3 drops mean/hard by `0.275996` / `0.014031` dB.
- Removing MB-Taylor from S3 drops mean/hard by `0.041948` / `0.085910` dB.

## Group-Min Gain Check

- S2 group gain-over-S0 minimum mean/hard across fixed train-derived bins: `+1.546768` / `+1.966238` dB.
- S3 group gain-over-S0 minimum mean/hard across fixed train-derived bins: `+1.559336` / `+1.966238` dB.
- Transmission and haze-density bins are populated from `train/trans/<clean_id>.png`; no fixed group has negative mean gain or selected-oracle severe losses.

## Decision

C8-Mini proves multi-expert complementarity relative to the v2.1 FullUDP-only S0 oracle. The next stage may design C9 low-capacity group-min router using train-derived oracle labels/features. This document does not authorize locked-test tuning and does not claim locked-test performance.
