# Haze4K v2.7 NH-HAZE Haze4K-Weight Zero-Shot Transfer Decision

Superseded note: this archived file belongs to an intermediate run with stale
metadata wording. Use the parent evidence README and final `v27_decision.md` for
the corrected interpretation:
`V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`. Both A0 and
WDMamba used Haze4K checkpoints, so the run is a Haze4K-weight zero-shot
diagnostic, not an official NH-HAZE benchmark.

Decision: `V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`

This route evaluates NH-HAZE as a Haze4K-weight zero-shot fixed-transfer
diagnostic. The primary row is the Haze4K-selected fixed
`WD0375 = A0 + 0.375 * (WDMamba - A0)` profile. Other alpha rows are reported
only as a predeclared diagnostic curve and are not used to tune NH-HAZE.

## Primary Fixed Row

- count: `55`
- alpha: `0.375`
- mean/hard/easy dPSNR: `-0.018157` / `-0.003815` / `-0.042949`
- dSSIM: `+0.00887693`
- positive/nonnegative: `0.472727` / `0.472727`
- severe: `13/55` (`141.82/600`)
- worst dPSNR: `-0.750659`

## Full Expert Endpoint

- alpha `1.0` mean/hard/easy dPSNR: `-0.187173` / `-0.095121` / `-0.364553`
- alpha `1.0` severe: `26/55` (`283.64/600`), worst `-2.029044`

## Protocol Notes

- Haze4K locked test touched: `false`.
- NH-HAZE alpha tuning: `false`.
- NH-HAZE has 55 paired full-resolution PNG images, each 1600x1200.
- Metrics are PSNR/SSIM against NH-HAZE GT; hard/easy buckets are bottom/top quartiles by A0 PSNR.
