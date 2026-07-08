# CHD-RM v2 Density-Need Calibration Evidence

Status: `PAUSE_V2_DUAL_HEAD_NOT_PASSED`

This evidence root stores compact text outputs from D0/D1/D3/D4/D5 on the fixed
Haze4K train_inner/val_inner split. Raw images, arrays, datasets, and head
checkpoints are not GitHub evidence.

Locked Haze4K test usage: none.

Main result:

- `H_density` is learnable and reliable on val_inner 600.
- `R_need` is not reliable enough yet; D4 is near on rank metrics but below the
  v2 gate, and D1 dual-head need fails.
- D5 shuffled-target control fails clearly, so the positive density result is
  not a generic training artifact.

Decision: pause before v3. Do not connect RARM until the `R_need` calibration
path is repaired or replaced inside the CHD-RM route scope.

Start with `v2_result_summary.md`, `decision_record.md`, and
`v2_run_summary.json` for the compact conclusion.
