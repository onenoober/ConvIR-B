# Haze4K v2.0 C2b Multi-Rule OutputDiff Router Screen

Decision: `C2B_MULTIRULE_IN_SAMPLE_ONLY_FAIL_OOF`

C2b reuses the C2 output-difference feature CSV and searches transparent one- and two-condition deployable policies.
No raw images/tensors were read or written, and locked test data was not touched.

## Best In-Sample Policy

- `policy_id`: `udp_grad_mean_ge_0.012657589_AND_diff_signed_mean_le_-0.0031482906`
- `complexity`: `2`
- `count`: `600`
- `selected_count`: `121`
- `coverage`: `0.20166666666666666`
- `mean_dPSNR`: `0.2903993892669678`
- `hard_bottom25_dPSNR`: `0.4990957514444987`
- `easy_top25_dPSNR`: `-0.0007663472493489584`
- `dSSIM`: `0.0009586901466051737`
- `positive_ratio`: `0.16`
- `nonnegative_ratio`: `0.9583333333333334`
- `severe_loss_count`: `22`
- `severe_loss_per_600`: `22.0`
- `strong_loss_count`: `23`
- `strong_loss_per_600`: `23.0`
- `selected_precision`: `0.7933884297520661`
- `selected_nonnegative_ratio`: `0.7933884297520661`
- `selected_severe_count`: `22`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `True`
- `score`: `0.36145534473765983`

## OOF Replay

- `count`: `600`
- `selected_count`: `99`
- `coverage`: `0.165`
- `mean_dPSNR`: `0.23411861737569173`
- `hard_bottom25_dPSNR`: `0.4546847407023112`
- `easy_top25_dPSNR`: `-0.033001556396484374`
- `dSSIM`: `0.0007564600308736165`
- `positive_ratio`: `0.12333333333333334`
- `nonnegative_ratio`: `0.9583333333333334`
- `severe_loss_count`: `23`
- `severe_loss_per_600`: `23.0`
- `strong_loss_count`: `24`
- `strong_loss_per_600`: `24.0`
- `selected_precision`: `0.7474747474747475`
- `selected_nonnegative_ratio`: `0.7474747474747475`
- `selected_severe_count`: `23`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `0.2352475674099393`

## Interpretation

- C3 shifted validation is authorized only if the OOF screen passes.
- If OOF fails, do not touch locked test; improve features, expert compatibility, or router class first.
