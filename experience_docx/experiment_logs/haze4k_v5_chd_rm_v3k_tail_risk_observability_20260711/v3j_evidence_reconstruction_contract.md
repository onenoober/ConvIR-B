# v3j Evidence Reconstruction Contract

v3k A0 must reconstruct v3j-B replay rows against archived v3j CSVs before any
diagnostic interpretation is trusted.

Required comparisons:
- exact row count
- exact split/fold/index/name/policy order
- per-image psnr_delta max absolute difference <= 1e-6
- severe direct sets at <= -0.2 dB exactly match

If reconstruction fails, v3k stops or is explicitly relabeled as a new replicate;
it cannot be used to choose alpha, clipping, energy bound, risk thresholds, or
canary expansion.
