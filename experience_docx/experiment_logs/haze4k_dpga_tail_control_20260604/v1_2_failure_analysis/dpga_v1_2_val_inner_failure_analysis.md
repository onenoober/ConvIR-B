# DPGA val_inner Failure Analysis

Gate pass: `false`
Locked test allowed: `false`

## Best Checkpoint

| group | count | mean delta | positive ratio | worst <= -0.20 | haze p1 | haze p2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 300 | 0.042656 | 0.620 | 16 | 0.7512 | 1.2302 |
| hard_bottom25 | 75 | 0.026225 | 0.600 | 2 | 0.7896 | 1.4597 |
| middle50 | 150 | 0.038195 | 0.580 | 9 | 0.7667 | 1.2569 |
| easy_top25 | 75 | 0.068009 | 0.720 | 5 | 0.6819 | 0.9472 |
| strong_reference_top25 | 75 | 0.068009 | 0.720 | 5 | 0.6819 | 0.9472 |
| worst_regressions_delta_le_-0.20 | 16 | -0.270311 | 0.000 | 16 | 0.6269 | 1.0506 |

## Final Checkpoint

- Mean delta: `0.039093`
- Hard bottom25 delta: `0.023173`
- Worst `<= -0.20 dB`: `14`

## Decision Hint

Gate failed on multiple axes; do not launch a higher-scale follow-up without a new diagnostic.
