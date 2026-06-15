# Haze4K v2.1 C6 Risk-Bounded Multi-Alpha Router

Decision: `C6_MULTIALPHA_OOF_SCREEN_PASS_STRONG_TARGET_NOT_YET_START_C7_C8`

C6 evaluates exact A0-preserving FullUDP residual alphas and searches train-only OOF image-level multi-alpha policies. Locked test data was not touched.

## OOF Summary

- `count`: `600`
- `selected_count`: `525`
- `coverage`: `0.875`
- `mean_dPSNR`: `0.42283898989359536`
- `hard_bottom25_dPSNR`: `0.479300422668457`
- `easy_top25_dPSNR`: `0.44730450948079425`
- `dSSIM`: `0.000275246798992157`
- `positive_ratio`: `0.6983333333333334`
- `nonnegative_ratio`: `0.8233333333333334`
- `severe_loss_count`: `46`
- `severe_loss_per_600`: `46.0`
- `strong_loss_count`: `89`
- `strong_loss_per_600`: `89.0`
- `selected_precision`: `0.7980952380952381`
- `selected_nonnegative_ratio`: `0.7980952380952381`
- `selected_severe_count`: `46`
- `screen_gate_pass`: `True`
- `formal_candidate_gate_pass`: `False`
- `score`: `1.1369145408812025`

## Image Multi-Alpha Oracle

- `count`: `600`
- `selected_count`: `478`
- `coverage`: `0.7966666666666666`
- `mean_dPSNR`: `0.8288999875386556`
- `hard_bottom25_dPSNR`: `0.9266457621256511`
- `easy_top25_dPSNR`: `0.8317298380533854`
- `dSSIM`: `0.0004145186146100362`
- `positive_ratio`: `0.7966666666666666`
- `nonnegative_ratio`: `1.0`
- `severe_loss_count`: `0`
- `severe_loss_per_600`: `0.0`
- `strong_loss_count`: `0`
- `strong_loss_per_600`: `0.0`
- `selected_precision`: `1.0`
- `selected_nonnegative_ratio`: `1.0`
- `selected_severe_count`: `0`
- `screen_gate_pass`: `True`
- `formal_candidate_gate_pass`: `True`
- `score`: `2.085789581044515`

## Interpretation

Only a C6 strong-candidate OOF pass can start C9 shifted-strong validation. It still does not authorize locked test.
