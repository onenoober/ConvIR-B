# Full-Model Line Family Summary

Date: 2026-07-07

Status: active full-model line after v3.1 diagnostic bakeoff and v3.2 P0/P1b aggregate route preflight/trainability pass.

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Haze4K v3.1 Full-Model Candidate Bakeoff | On 600 train-derived full-image samples, WDMamba standalone had mean/hard/easy `+3.5778/+8.2765/-1.0483` dB vs A0. ConvIR-L standalone had `+1.0945/+1.4339/+0.6015` dB. FullUDP standalone was negative at `-0.4313` dB mean. Oracle over A0/WDMamba/ConvIR-L/FullUDP reached `+4.4353 dB` mean with severe `0`. | `COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM`; use this as full-model headroom evidence, not as strict A0 dominance evidence. |
| Haze4K v3.2 ConvIR-WD Full Model Line | P0 strict partial-load from the official anchor passed: `602` official keys loaded, `24` WD keys allowed missing, no-op max abs vs A0 `0.0`, finite one-batch loss, locked test untouched. Original P1 trained over `8` train-derived `256` center crops but gated only the first mini-batch. Corrected P1b aggregate on all `8` crops passed: loss ratio `0.7841028225902132` against gate `<=0.95`, WD activity delta `0.005941152640540774`, finite outputs, locked test untouched. | `COMPLETED_P0_P1B_AGGREGATE_GATE_PASS_P2_DESIGN_OPEN_LOCKED_TEST_BLOCKED`; this validates architecture/load/trainability only and opens P2 design, not a quality claim. |

## Interpretation

v3.1 confirms that standard full-model headroom exists and is far larger than the A0-neighborhood residual/partial-unfreeze movements. It also confirms that standalone stronger models can still hurt easy/tail cases, so v3.2 must use a model-line success gate and train-derived validation, not strict safe-upgrade gates or locked-test selection.

v3.2 confirms the first ConvIR-WD route is runnable and trainable from the
official anchor without disturbing A0 at initialization. It does not yet answer
whether ConvIR-WD improves Haze4K quality.

## Reopen / Continue Condition

Continue only by writing the P2 train-derived validation design for v3.2
ConvIR-WD or another WDMamba-informed full-model protocol, with split, budget,
checkpoint policy, and gate fixed before launch. Locked test remains blocked
until fixed candidate selection. Do not reopen A0 residual, selector/alpha,
bridge/generator, or v3.0 rescue under this family.
