# Fixed Visual Notes

## Verdict

`B1-Surgery scale=0.70` is visually and numerically consistent with a preservation-first SafeRHFD candidate.

The panel does not show the B1 failure mode: no obvious global darkening, over-brightening, color cast, range collapse, or structural corruption. Candidate-original differences are small and mostly localized. Worst-regression rows appear to be mild texture, contrast, or residual-haze metric shifts rather than catastrophic output drift.

## Quantitative Notes

- A0 mean PSNR: `34.14065`
- Surgery scale 0.70 mean PSNR: `34.15128`
- Mean PSNR delta: `+0.01064 dB`
- Hard bottom-25% delta: `+0.03317 dB`
- Easy top-25% delta: `+0.00782 dB`
- Severe regressions (`delta <= -0.20 dB`): `0 / 1000`
- Strong-reference regressions (`delta <= -0.05 dB`): `0 / 250`
- Global regressions (`delta <= -0.05 dB`): `9 / 1000`

## Safety Stats

Candidate output safety over the fixed diagnostic set:

- `ratio_pred_lt_0` mean: `0.00303`, max: `0.02183`
- `ratio_pred_gt_1` mean: `0.01919`, max: `0.04990`
- `rgb_mean_shift_abs_mean` mean: `0.00929`, max: `0.02702`
- `luma_mean_shift` mean: `0.00002`, range: `[-0.02690, +0.02275]`
- `saturation_ratio` mean: `0.99228`, range: `[0.86531, 1.10766]`

These values are small compared with the B1 diagnostic failure pattern and do not indicate a systematic brightness, color, or saturation drift.

## RHFD Activity

RHFD activity is present but bounded:

- `rhfd2_delta_norm_ratio` mean: `0.00230`, max: `0.00282`
- `rhfd1_delta_norm_ratio` mean: `0.00280`, max: `0.00323`
- `rhfd2_delta_abs_mean` mean: `0.00101`
- `rhfd1_delta_abs_mean` mean: `0.00054`

The activity is enough to cross the hard-bucket gate while remaining much safer than the original B1 full-backbone route.

## Recommendation

Use `scale=0.70` as the primary B1-Surgery / SafeRHFD candidate. Keep `scale=1.00` as backup evidence only: it has higher mean and hard-bucket gains, but also more global, easy, and strong-reference regressions.

Do not commit panel PNGs or sample image outputs; keep them as local/cloud visual artifacts only.
