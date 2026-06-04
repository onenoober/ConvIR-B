# Haze4K APDR-v0.4E E1 OOF Fixed-Code Rerun

Date: 2026-06-04

Status: E1 OOF stop direction reproduced, but exact full numeric seal is blocked
until an alias-corrected full rerun is completed.

## Result

The expensive OOF action evaluation completed for 3000 images and wrote
`v04e_oof_candidate_action_per_image_sigma3.csv`. The original E1 script then
failed while writing `v04e_oof_locked_threshold_by_fold.csv` because rows had a
variable schema and the shared CSV writer used only the first row's fields.
`finalize_haze4k_apdr_v0_4e_oof_calibration.py` regenerated the missing summary
tables from the per-image intermediate CSV.

Locked-rule summary:

| Rule | Status | Keep | Coverage | Mean gain | Hard gain | Easy gain | Strong/severe | Oracle recovery | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule A `global_plus_spatial_kenel_knn_9` | missing candidate | - | - | - | - | - | - | - | fail |
| Rule B `spatial_priors_ridge_10` | present | `150/3000` | `0.0500` | `+0.0378 dB` | `+0.1352 dB` | `+0.0000 dB` | `0/1` | `0.0835` | fail |

The post-hoc OOF policy search retained `1600` candidates and found
`0` gate-passing policies. The best retained policy used
`spatial_priors_ridge_10`, scale `1.0`, with
`weighted_residual_norm >= 0.009209620478734751` and
`nn_distance <= 11.12091464996338`; it removed severe regressions and reached
hard `+0.2527 dB`, but coverage was only `0.0877`, below the `0.10` gate.

## Compatibility Finding

This rerun generated `24000` action rows instead of the historical `60000`
because `826caaf` still filtered out KNN mappers by requesting historical
`*_kenel_knn_9` names while the probe generated `*_kernel_knn_9` names. Current
code has been patched with mapper alias compatibility and variable-schema CSV
writing, but the full alias-corrected OOF numeric seal is still pending.

## Decision

The current v0.4E locked thresholds remain stopped. Do not run E2, full spatial
router, local correction, dense residual training, or stop20 from this route.
