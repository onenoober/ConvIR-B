# v2.19 P1 O2 Spatial Action Learnability Protocol

This audit regenerates v2.17-style O2 spatial final-feature LL oracle targets on train-derived Haze4K images.
It then fits 5-fold deployable spatial predictors and replays predicted deltas through official ConvIR-B.
No deployable model is trained here and locked Haze4K is untouched.

Predictors and controls:

- small CNN spatial policy
- depthwise-small-CNN spatial policy
- shuffled-target spatial control
- global-broadcast O1-style ridge control
- exact oracle replay upper bound

O2 optimization steps: `25`; grid: `16`; delta scale: `0.5`.
CNN epochs: `220`; depthwise epochs: `220`; shuffled epochs: `120`.
