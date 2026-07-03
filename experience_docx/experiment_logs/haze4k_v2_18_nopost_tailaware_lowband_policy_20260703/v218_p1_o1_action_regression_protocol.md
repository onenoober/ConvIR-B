# v2.18 P1 O1 Action Regression Protocol

This audit regenerates O1 global final-feature LL oracle targets on train-derived Haze4K images.
It then fits 5-fold deployable pooled-LL context predictors and replays predicted deltas through official ConvIR-B.
No deployable model is trained here and locked Haze4K is untouched.

Primary predictor: small MLP on final-feature LL channel mean and std context.
Diagnostics: ridge predictor and shuffled-target ridge control.
O1 optimization steps: `25`; delta scale: `0.5`.
MLP epochs: `320`; hidden: `64`.
