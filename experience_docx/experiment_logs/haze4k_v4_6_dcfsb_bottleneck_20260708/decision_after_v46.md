# Haze4K v4.6 DCFSB-Bottleneck Decision

Status: internal stage gate passed.

Selected candidate: `adapter4`.

Locked test: not touched; not enumerated.

## Internal256 Comparison

| Variant | Pass | Mean dPSNR | Positive ratio | p5 dPSNR | Mean dHighL1 | High-L1 improve ratio | High gate std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter5 | False | 0.032007 | 0.621094 | -0.297054 | -0.00000088 | 0.566406 | 0.042662 |
| adapter3 | False | 0.027318 | 0.589844 | -0.149409 | -0.00000109 | 0.582031 | 0.031827 |
| adapter5_lr5e5 | False | 0.015087 | 0.593750 | -0.205874 | -0.00000106 | 0.621094 | 0.025389 |
| adapter4 | True | 0.044404 | 0.625000 | -0.216141 | -0.00000057 | 0.527344 | 0.038074 |

## Decision

`adapter4` is selected because it is the only variant that passes all internal256 stage-gate thresholds: mean dPSNR >= `+0.03`, positive ratio >= `0.53`, p5 dPSNR >= `-0.25`, high-frequency L1 not worsened, and nontrivial high-gate variation.

The bracket probes were necessary: `adapter5` proved the route can exceed the mean-gain threshold but had too much tail harm, while `adapter3` made the tail safe but narrowly missed the mean-gain threshold. `adapter4` resolves that trade-off.

Next phase requires a separate written gate before any locked-test or broader external validation.

Raw cloud-only files not committed by default: per-image CSVs, module-stat JSONLs, checkpoints, and TensorBoard runs.
