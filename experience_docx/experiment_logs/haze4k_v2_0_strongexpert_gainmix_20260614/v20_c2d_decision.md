# Haze4K v2.0 C2d Alpha-Blend OutputDiff Router Screen

Decision: `C2D_ALPHA_STRICT_SCREEN_PASS_START_C3_SHIFTED`

C2d evaluates fixed alpha shrink for selected FullUDP residuals and chooses alpha+policy inside each train fold.
No raw images/tensors were written, and locked test data was not touched.

## Best In-Sample Policy

- `policy_id`: `alpha_a0p25__diff_signed_mean_le_0.0021860948`
- `alpha`: `0.25`
- `complexity`: `1`
- `count`: `600`
- `selected_count`: `480`
- `coverage`: `0.8`
- `mean_dPSNR`: `0.323956979115804`
- `hard_bottom25_dPSNR`: `0.2553900400797526`
- `easy_top25_dPSNR`: `0.4575859832763672`
- `dSSIM`: `0.00022891839345296223`
- `positive_ratio`: `0.655`
- `nonnegative_ratio`: `0.855`
- `severe_loss_count`: `32`
- `severe_loss_per_600`: `32.0`
- `strong_loss_count`: `72`
- `strong_loss_per_600`: `72.0`
- `selected_precision`: `0.81875`
- `selected_nonnegative_ratio`: `0.81875`
- `selected_severe_count`: `32`
- `strict_gate_pass`: `True`
- `abstention_gate_pass`: `False`
- `score`: `0.4087548866271972`

## OOF Replay

- `count`: `600`
- `selected_count`: `504`
- `coverage`: `0.84`
- `mean_dPSNR`: `0.3325235939025879`
- `hard_bottom25_dPSNR`: `0.25777137756347657`
- `easy_top25_dPSNR`: `0.47704737345377607`
- `dSSIM`: `0.00023827811082204184`
- `positive_ratio`: `0.6816666666666666`
- `nonnegative_ratio`: `0.8416666666666667`
- `severe_loss_count`: `37`
- `severe_loss_per_600`: `37.0`
- `strong_loss_count`: `79`
- `strong_loss_per_600`: `78.99999999999999`
- `selected_precision`: `0.8115079365079365`
- `selected_nonnegative_ratio`: `0.8115079365079365`
- `selected_severe_count`: `37`
- `strict_gate_pass`: `True`
- `abstention_gate_pass`: `False`
- `score`: `0.4076529093908885`

## Fold Policies

- fold `0`: `alpha_a0p25__diff_signed_mean_le_0.0029775854`, pass `False`
- fold `1`: `alpha_a0p25__diff_signed_mean_le_0.0030439693`, pass `False`
- fold `2`: `alpha_a0p25__diff_signed_mean_le_0.0027345826`, pass `False`
- fold `3`: `alpha_a0p25__diff_signed_mean_le_0.0022425749`, pass `False`
- fold `4`: `alpha_a0p25__diff_signed_mean_le_0.0026143706`, pass `False`
