# DPGA val_inner Failure Analysis

Gate pass: `false`
Locked test allowed: `false`

## Best Checkpoint

| group | count | mean delta | positive ratio | worst <= -0.20 | haze p1 | haze p2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 300 | 0.037036 | 0.623 | 9 | 0.7512 | 1.2302 |
| hard_bottom25 | 75 | 0.023367 | 0.613 | 1 | 0.7896 | 1.4597 |
| middle50 | 150 | 0.033300 | 0.580 | 5 | 0.7667 | 1.2569 |
| easy_top25 | 75 | 0.058178 | 0.720 | 3 | 0.6819 | 0.9472 |
| strong_reference_top25 | 75 | 0.058178 | 0.720 | 3 | 0.6819 | 0.9472 |
| worst_regressions_delta_le_-0.20 | 9 | -0.247753 | 0.000 | 9 | 0.6411 | 1.1622 |

## Final Checkpoint

- Mean delta: `0.034438`
- Hard bottom25 delta: `0.021114`
- Worst `<= -0.20 dB`: `9`

## Decision Hint

Mean gain and tail safety are acceptable, but hard-bottom gain is short. Prefer a small hard-gain follow-up: keep shallow-only, raise scale modestly, and reduce global anchor pressure before adding architecture.
