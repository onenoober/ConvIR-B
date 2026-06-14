# Haze4K v2.0 C1 Strong Expert Risk/Correctability Map

Decision: `C1_SIMPLE_FEATURE_POLICY_SCREEN_PASS_START_C2_TABLE_ROUTER`

This phase uses existing internal-validation A0/FullUDP endpoint evidence only. Locked test data was not touched.

## Best Simple Policy

- `policy_id`: `val_hard_and_name_param_2_le_1.39`
- `count`: `600`
- `coverage`: `0.20666666666666667`
- `mean_dPSNR`: `0.18447762489318847`
- `hard_bottom25_dPSNR`: `0.34585412343343097`
- `easy_top25_dPSNR`: `0.0`
- `dSSIM`: `1.1609892050425212e-05`
- `positive_ratio`: `0.15`
- `nonnegative_ratio`: `0.9433333333333334`
- `severe_loss_count`: `32`
- `severe_loss_per_600`: `32.0`
- `strong_loss_count`: `32`
- `strong_loss_per_600`: `32.0`
- `screen_gate_pass`: `True`
- `score`: `0.2069411557515462`

## Best Screen-Passing Simple Policy

- `policy_id`: `val_hard_and_name_param_2_le_1.39`
- `count`: `600`
- `coverage`: `0.20666666666666667`
- `mean_dPSNR`: `0.18447762489318847`
- `hard_bottom25_dPSNR`: `0.34585412343343097`
- `easy_top25_dPSNR`: `0.0`
- `dSSIM`: `1.1609892050425212e-05`
- `positive_ratio`: `0.15`
- `nonnegative_ratio`: `0.9433333333333334`
- `severe_loss_count`: `32`
- `severe_loss_per_600`: `32.0`
- `strong_loss_count`: `32`
- `strong_loss_per_600`: `32.0`
- `screen_gate_pass`: `True`
- `score`: `0.2069411557515462`

## Interpretation

- C1 is a risk-map and separability audit, not a final deployable router.
- If no simple policy passes the screen gate, the next efficient step is to reacquire or render strong-expert outputs on `convir-4090` and compute FullUDP-A0 output-difference features before C2.
