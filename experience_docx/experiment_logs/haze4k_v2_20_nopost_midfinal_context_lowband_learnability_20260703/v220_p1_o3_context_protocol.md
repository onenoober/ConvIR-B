# v2.20 P1 O3 Mid+Final/Context Learnability Protocol

This audit tests whether deployable mid+final/context features can learn safe NoPost lowband actions.
It is a train-derived learnability and replay audit only: no deployable model training and no locked Haze4K test.

Controls:

- exact O2 final-feature LL oracle upper bound
- exact O3 mid+final LL oracle upper bound
- v2.19-style final-only spatial CNN replicate
- mid-only predictor
- final+mid predictor
- final+mid+global context predictor
- shuffled-target control
- global-broadcast control
- no-op control

O2 steps: `25`; O3 steps: `18`; mid grid: `8`; final grid: `16`.
CNN epochs: `180`; shuffled epochs: `100`.
