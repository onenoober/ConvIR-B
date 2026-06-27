# Haze4K v2.1 C7b Local-Alpha Deployable Prototype

Decision: `C7B_LOCAL_ALPHA_FAIL_START_C8_MULTIEXPERT_OR_RICHER_LOCAL_FEATURES`

C7b trains transparent patch-level alpha policies using train-derived image-fold OOF and re-renders held-out images for true PSNR/SSIM. Locked test data was not touched.

## Actual OOF Summary

- `count`: `600`
- `selected_count`: `599`
- `coverage`: `0.9983333333333333`
- `mean_dPSNR`: `0.3761114247639974`
- `hard_bottom25_dPSNR`: `0.3609487152099609`
- `easy_top25_dPSNR`: `0.4431714121500651`
- `dSSIM`: `0.00025761812925338744`
- `positive_ratio`: `0.7933333333333333`
- `nonnegative_ratio`: `0.795`
- `severe_loss_count`: `50`
- `severe_loss_per_600`: `50.0`
- `strong_loss_count`: `89`
- `strong_loss_per_600`: `89.0`
- `selected_precision`: `0.7946577629382304`
- `selected_nonnegative_ratio`: `0.7946577629382304`
- `selected_severe_count`: `50`
- `mean_patch_action_fraction_a0`: `0.20068750000000002`
- `mean_patch_action_fraction_a0p125`: `0.0`
- `mean_patch_action_fraction_a0p25`: `0.5222083333333334`
- `mean_patch_action_fraction_a0p375`: `0.21022916666666666`
- `mean_patch_action_fraction_a0p5`: `0.056375`
- `mean_patch_action_fraction_a0p75`: `0.0105`
- `screen_gate_pass`: `False`
- `strong_gate_pass`: `False`
- `score`: `0.9421875381469728`

## Interpretation

C9 shifted-strong validation is authorized only if the C7b strong gate passes. Otherwise locked test and distillation remain blocked.
