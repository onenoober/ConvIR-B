# Full-Model Line Family Summary

Date: 2026-07-07

Status: opened by v3.1 diagnostic bakeoff after the A0-anchored safe-upgrade family closed.

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Haze4K v3.1 Full-Model Candidate Bakeoff | On 600 train-derived full-image samples, WDMamba standalone had mean/hard/easy `+3.5778/+8.2765/-1.0483` dB vs A0. ConvIR-L standalone had `+1.0945/+1.4339/+0.6015` dB. FullUDP standalone was negative at `-0.4313` dB mean. Oracle over A0/WDMamba/ConvIR-L/FullUDP reached `+4.4353 dB` mean with severe `0`. | `COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM`; use this as full-model headroom evidence, not as strict A0 dominance evidence. |

## Interpretation

v3.1 confirms that standard full-model headroom exists and is far larger than the A0-neighborhood residual/partial-unfreeze movements. It also confirms that standalone stronger models can still hurt easy/tail cases, so v3.2 must use a model-line success gate and train-derived validation, not strict safe-upgrade gates or locked-test selection.

## Reopen / Continue Condition

Continue only with a written full-model route, preferably v3.2 ConvIR-WD or WDMamba-informed full-model training, with P0/P1/P2 gates and locked test blocked until fixed candidate selection. Do not reopen A0 residual, selector/alpha, bridge/generator, or v3.0 rescue under this family.
