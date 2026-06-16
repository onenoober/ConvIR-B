# Haze4K v2.7 NH-HAZE Haze4K-Weight Zero-Shot Transfer Decision

Decision: `V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`

This route evaluates NH-HAZE as a Haze4K-weight zero-shot fixed-transfer
diagnostic. The primary row is the Haze4K-selected fixed
`WD0375 = A0 + 0.375 * (WDMamba - A0)` profile. Other alpha rows are reported
only as a predeclared diagnostic curve and are not used to tune NH-HAZE.

Scope caveat: this is not an official NH-HAZE benchmark and not a fair NH-HAZE
comparison between ConvIR-B and WDMamba. Both endpoints use Haze4K checkpoints:
`haze4k-base.pkl` for A0 and `haze4k_35.88.pth` for WDMamba. The result does
not prove either model is weak on NH-HAZE when evaluated with NH-HAZE-specific
weights.

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
- This route used Haze4K weights only; official NH-HAZE-trained ConvIR-B and
  WDMamba weights must be acquired or trained before any formal NH-HAZE claim.
