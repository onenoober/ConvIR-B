# Haze4K v2.8 NH-HAZE Official-Weight Evaluation Decision

Decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`

This route evaluates NH-HAZE with NH-HAZE-specific ConvIR-B and WDMamba checkpoints. The `alpha=1.0` row is the official-weight WDMamba endpoint relative to the official-weight ConvIR-B A0 baseline. The `alpha=0.375` row reuses the Haze4K-discovered fixed shrinkage coefficient only as a predeclared diagnostic; the alpha grid is not used to tune NH-HAZE.

## Inherited Alpha Diagnostic Row

- count: `55`
- alpha: `0.375`
- mean/hard/easy dPSNR: `-0.133584` / `+0.122348` / `-0.155758`
- dSSIM: `-0.00598572`
- positive/nonnegative: `0.309091` / `0.309091`
- severe: `26/55` (`283.64/600`)
- worst dPSNR: `-0.956678`

## WDMamba NH-HAZE Endpoint

- alpha `1.0` mean/hard/easy dPSNR: `-2.197751` / `-1.093919` / `-2.327606`
- alpha `1.0` severe: `52/55` (`567.27/600`), worst `-4.730593`

## Protocol Notes

- Haze4K locked test touched: `false`.
- NH-HAZE alpha tuning: `false`; alpha rows other than `1.0` are diagnostic.
- A0 and WDMamba both use NH-HAZE-specific checkpoints; A0 data argument is `NHR` and WDMamba DENet blocks is `4`.
- NH-HAZE has 55 paired full-resolution PNG images, each 1600x1200.
- Metrics are PSNR/SSIM against NH-HAZE GT; hard/easy buckets are bottom/top quartiles by A0 PSNR.
