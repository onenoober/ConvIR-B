# Haze4K v2.24 NoPost Train-Time Controller Failure Audit Evidence

Status: V224_DIAGNOSTIC_COMPLETE_CASE_A_RISK_HEAD_COLLAPSE_LOCKED_TEST_BLOCKED

Diagnostic-only route. No new training, no locked Haze4K test, and no checkpoint or threshold selection from locked test.

## Key Decisions

- P0: `P0_PASS`
- P1: `P1_CROP_STRONG_MASK_OVERBROAD_STRONG_GATE_SHOULD_USE_FULL_IMAGE_OR_PRECOMPUTED_MASK`
- P2: `P2_RISK_HEAD_COLLAPSE_OR_BASE_RATE_LEARNING_CONFIRMED`
- P3: `P3_TRAINED_ACTION_CAN_BE_RESCUED_BY_BETTER_GATE`
- P4: `P4_SUPERVISION_GRADIENT_IMBALANCE_RISK_CONFIRMED`
- P5: `P5_EPOCH2_MEAN_CAN_RISE_WHILE_TAIL_WORSENS_EXPANDING_EPOCHS_FORBIDDEN`

## Locked-Test Policy

Locked test remained untouched and blocked throughout v2.24.
