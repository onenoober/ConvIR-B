# v2f F4b Tail-Rescue Authorization

Status: authorized as a narrow frozen-side diagnostic after F4 completed with
`COMPLETED_GATE_FAIL`.

Purpose: test whether the F4 failure was caused by insufficient low-density
hard-negative tail pressure rather than by a deeper target/head separability
limit.

Allowed:
- frozen ConvIR-B A0
- frozen D3 density
- frozen-side stratified R_need heads only
- train_inner calibration and val_inner evaluation
- original v2e global safety/LDHN gate

Forbidden:
- D2
- v3
- RARM connection or training
- locked Haze4K test
- relaxing false-tail or LDHN gates

If no F4b spec finds a selected gate pass or a safe+LDHN threshold point, keep
v2f paused and do not run v3/RARM. If a candidate point appears, run F5 controls
before any v3 no-op audit.
