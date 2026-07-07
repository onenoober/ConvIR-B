# Full-Model Line Family Summary

Date: 2026-07-07

Status: active full-model family, but the first ConvIR-WD v3.2 route is closed negative after fixed P2 train-derived validation.

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Haze4K v3.1 Full-Model Candidate Bakeoff | On 600 train-derived full-image samples, WDMamba standalone had mean/hard/easy `+3.5778/+8.2765/-1.0483` dB vs A0. ConvIR-L standalone had `+1.0945/+1.4339/+0.6015` dB. FullUDP standalone was negative at `-0.4313` dB mean. Oracle over A0/WDMamba/ConvIR-L/FullUDP reached `+4.4353 dB` mean with severe `0`. | `COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM`; use this as full-model headroom evidence, not as strict A0 dominance evidence. |
| Haze4K v3.2 ConvIR-WD Full Model Line | P0 strict partial-load from the official anchor passed: `602` official keys loaded, `24` WD keys allowed missing, no-op max abs vs A0 `0.0`, finite one-batch loss, locked test untouched. Corrected P1b aggregate on all `8` train-derived crops passed: loss ratio `0.7841028225902132` against gate `<=0.95`, WD activity delta `0.005941152640540774`, finite outputs, locked test untouched. Fixed P2R1 then completed the 480/120 train-derived validation screen, but Best vs A0 was only mean/hard/easy `+0.1387/+0.1959/+0.0476` dB with p05/CVaR5 `-0.5714/-0.7218`, below the predeclared P2 thresholds. | `COMPLETED_P0_P1B_AGGREGATE_PASS_P2_GATE_FAIL_LOCKED_TEST_BLOCKED`; v3.2 is not authorized for P3 or locked test. |

## Interpretation

v3.1 confirms that standard full-model headroom exists and is far larger than the A0-neighborhood residual/partial-unfreeze movements. It also confirms that standalone stronger models can still hurt easy/tail cases, so v3.2 must use a model-line success gate and train-derived validation, not strict safe-upgrade gates or locked-test selection.

v3.2 confirms the first ConvIR-WD route is runnable and trainable from the
official anchor without disturbing A0 at initialization, but the fixed P2
screen did not produce enough mean/hard gain or tail safety to justify P3.
This closes the current v3.2 continuation path rather than the broader
full-model family.

## Reopen / Continue Condition

Do not continue v3.2 to P3, canary expansion, locked test, or simple
epoch/fold/sample/loss-weight tuning. Continue only with a new written
full-model protocol or materially changed ConvIR-WD mechanism/training
contract, with split, budget, checkpoint policy, and gate fixed before
launch. Locked test remains blocked until fixed candidate selection. Do not
reopen A0 residual, selector/alpha, bridge/generator, or v3.0 rescue under
this family.
