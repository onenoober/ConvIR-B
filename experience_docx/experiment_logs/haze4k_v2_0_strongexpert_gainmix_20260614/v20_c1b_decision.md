# Haze4K v2.0 C1b Deployable-Feature Audit

Decision: `C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES`

This corrected C1 audit excludes validation split membership and filename-derived parameters.
It uses existing internal-validation A0/FullUDP endpoint metrics only; locked test data was not touched.

## Gates

- Strict gate includes all-sample positive ratio `>= 0.65`.
- Abstention-aware gate requires selected precision `>= 0.65`, all-sample nonnegative ratio `>= 0.9`, and coverage `>= 0.1`.

## Best In-Sample Deployable Proxy

- `policy_id`: `A0_PSNR_le_34.4278`
- `count`: `600`
- `selected_count`: `180`
- `coverage`: `0.3`
- `mean_dPSNR`: `0.2034080378214518`
- `hard_bottom25_dPSNR`: `0.6855227915445964`
- `easy_top25_dPSNR`: `0.0`
- `dSSIM`: `-4.1805307070414225e-05`
- `positive_ratio`: `0.2`
- `nonnegative_ratio`: `0.9`
- `severe_loss_count`: `54`
- `severe_loss_per_600`: `54.0`
- `strong_loss_count`: `56`
- `strong_loss_per_600`: `56.0`
- `selected_precision`: `0.6666666666666666`
- `selected_nonnegative_ratio`: `0.6666666666666666`
- `selected_severe_count`: `54`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `0.30012206904093425`

## OOF Threshold Replay

- `count`: `600`
- `selected_count`: `156`
- `coverage`: `0.26`
- `mean_dPSNR`: `0.17043327331542968`
- `hard_bottom25_dPSNR`: `0.6229673767089844`
- `easy_top25_dPSNR`: `0.0`
- `dSSIM`: `-4.448026418685913e-05`
- `positive_ratio`: `0.17`
- `nonnegative_ratio`: `0.91`
- `severe_loss_count`: `46`
- `severe_loss_per_600`: `46.0`
- `strong_loss_count`: `50`
- `strong_loss_per_600`: `50.0`
- `selected_precision`: `0.6538461538461539`
- `selected_nonnegative_ratio`: `0.6538461538461539`
- `selected_severe_count`: `46`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `0.2668674251849835`

## Interpretation

- C1b is a leakage-safe audit; it is not a trained image/patch router.
- A C2 router should not claim deployability from split/name-param policies.
- If C1b does not pass, the efficient next step is to reacquire/render FullUDP outputs and compute real output-difference, depth, texture, and artifact features before router training.
