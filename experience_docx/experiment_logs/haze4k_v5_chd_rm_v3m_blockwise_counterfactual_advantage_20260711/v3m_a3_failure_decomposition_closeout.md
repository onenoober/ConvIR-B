# v3m A3 Failure Decomposition

Decision: `V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION`.

This is a diagnostic-only post-fail audit over already completed cloud A3/A2
rows. It does not train, tune thresholds, rerun inference, replay a new policy,
use route-confirm, touch canary, or touch locked test.

## Operator Summary

| Operator | Mean lift vs fixed | p10 lift vs fixed | Severe policy/fixed | Hard policy/fixed | Mean selected alpha | Mean alpha=1 fraction | Severe-subset oracle lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D_ref` | `+0.0828431` | `-0.2209025` | `148 / 0` | `39 / 0` | `0.2476405` | `0.1338205` | `+0.2754318` |
| `D_rep` | `+0.0826054` | `-0.2306944` | `146 / 0` | `41 / 0` | `0.2444203` | `0.1336124` | `+0.2682955` |

## Cross-Operator Tail Stability

- shared names: `1200`;
- severe overlap: `140` of union `154` (Jaccard `0.9090909`);
- hard overlap: `38` of union `42` (Jaccard `0.9047619`);
- policy-lift correlation: `0.9930474`;
- selected-alpha correlation: `0.9962972`;
- oracle-lift correlation: `0.9970668`.

## Interpretation

The failure remains a safe-utility calibration problem. A2 label calibration is
strong enough to create positive mean image PSNR, but A3's selected action mix
is too aggressive at image level and produces stable tail regressions across
the two frozen operators. Severe images still have positive block16-oracle
headroom on average, so the issue is not absence of oracle value; it is
incorrect allocation of aggressive local escalation under the current
deployable signal.

No route-confirm audit, canary, locked-test access, controller training,
learned ranker, physics/proxy continuation, or policy deployment is authorized
by this diagnostic.
