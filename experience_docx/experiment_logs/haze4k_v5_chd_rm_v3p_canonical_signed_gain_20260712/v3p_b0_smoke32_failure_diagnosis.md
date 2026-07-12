# v3p B0 Smoke Failure Diagnosis

This is a read-only diagnosis of the typed smoke failure
`V3P_B0_SCALAR_A_SMOKE_FAIL_STOP_PHYSICS_ROUTE`. It does not modify the B0
contract, launch formal B0/B1, fit a model, or access canary/locked data.

## Frozen Smoke Result

- Scope: first 32 sorted names from `v3j_controller_train` only.
- Structural contract: pass. `train/haze` contains 3,000 PNG files plus only
  `.DS_Store`; `gt` and `trans` contain 3,000 numeric PNG ids each. Every
  selected triplet maps by numeric prefix, uses RGB uint8 haze/GT and exact
  replicated-RGB uint8 transmission, and has matching shape.
- Scalar A range: pass, `[0.525655, 0.918618]`.
- Primary sRGB forward residual: fail. Mean RMSE `0.077502`, p95 `0.123837`,
  p99 `0.148307`, maximum `0.159017`; preregistered maximum is `8/255 =
  0.031373`.
- Linear-space sensitivity is worse: p99 RMSE `0.215065`.

## Semantic Cross-Checks

All rows below use the same 32 mapped train triplets and a fitted A where the
method permits it. They are diagnostics, not alternative gates.

| Interpretation | Mean RMSE | p95 RMSE | Maximum RMSE |
| --- | ---: | ---: | ---: |
| raw `trans` as t, scalar A | 0.077502 | 0.123837 | 0.159017 |
| raw `trans` as t, channel-wise A | 0.076642 | 0.123229 | 0.158864 |
| `1 - trans`, scalar A | 0.193195 | 0.334074 | 0.350153 |
| `1 - trans`, channel-wise A | 0.193065 | 0.334064 | 0.350152 |
| `exp(-first_filename_parameter * trans)`, scalar A | 0.176792 | 0.320916 | 0.338111 |
| `exp(-second_filename_parameter * trans)`, scalar A | 0.175538 | 0.319446 | 0.334544 |

For the second filename-implied exponential interpretation, using the first
filename parameter directly as A gives mean RMSE `0.186999`; using the second
gives `0.324060`. Neither supports a depth-serialization repair.

## Decision

The residual gap is far above a PNG quantization margin, and no directly
specified alternative representation reduces it. The available package cannot
support the privileged `t+A` ceiling or any estimated physics route. A repair
requires authoritative external provenance for the Haze4K generator and
serialization, then a new explicitly authorized data-contract route. Do not
relax the residual tolerance or continue B0 formal/B1.
