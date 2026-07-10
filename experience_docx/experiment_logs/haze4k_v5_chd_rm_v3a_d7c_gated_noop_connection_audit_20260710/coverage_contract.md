# v3a Coverage Contract

- Baseline: official A0 ConvIR-B Haze4K checkpoint.
- Candidate: `fam2_d7c_noop` from official anchor.
- D7c gate: v2h fixed D7c threshold `0.5773006677627563`.
- Split: train-derived `val_inner` 600 from v1 internal split.
- Locked test: forbidden.
- Gate threshold: output max abs diff `<= 1e-7`.
- Metric threshold: PSNR/SSIM max absolute delta `<= 1e-10`.
- Required gate observability: D7c gate must be nontrivial on real/internal
  samples; otherwise the connection audit is inconclusive.
