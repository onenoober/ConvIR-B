# v3o Metric Contract

- Dataset view: Haze4K train-derived `v3j_controller_train`, 1,200 names, five
  clean-reference grouped OOF folds, both frozen operators.
- Fixed reference: `alpha=0.125` on the identical operator/name row.
- Primary block metric: RGB SSE, not block PSNR.
- Image reconstruction check: sum block SSE / total RGB element count must match
  direct image MSE for every candidate within `1e-10`.
- Replay check: fixed-alpha PSNR delta must match the frozen v3m reference within
  `1e-6 dB`.
- Any later image-risk metric must report per-image cumulative harmful SSE,
  grouped mean interval, p10/p05/worst, severe/hard counts, and both operators.
