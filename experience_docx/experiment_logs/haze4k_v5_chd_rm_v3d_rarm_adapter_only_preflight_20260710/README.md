# v3d RARM Adapter-Only Preflight

Status: `COMPLETED_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3d-rarm-adapter-only-preflight.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3d is the separate written decision required after v3c. It verifies that RARM
training can be constrained to the zero-init FAM2 modulator before any adapter
training is launched.

## Evidence

- `run_v3d_rarm_stage0_preflight.sh`
- `v3d_stage0_preflight.log`
- `v3d_stage0_preflight_summary.json`
- `v3d_stage0_preflight_per_sample.csv`
- `v3d_stage0_gradient_audit.csv`
- `v3d_stage0_preflight_closeout.json`
- `run_v3d_rarm_stage1_1epoch.sh`
- `v3d_stage1_1epoch_audit_summary.json`
- `run_v3d_rarm_stage1_5epoch.sh`
- `v3d_stage1_5epoch_audit_summary.json`
- `run_v3d_fam2_modres_matched_control_5epoch.sh`
- `v3d_fam2modres_control_5epoch_audit_summary.json`
- `run_v3d_matched_control_compare.sh`
- `v3d_matched_control_comparison.json`
- `v3d_final_closeout.json`
- `status.txt`

## Result

Decision:
`V3D_PAUSE_D7C_SAFER_BUT_NOT_MATCHED_CONTROL_UTILITY_NO_20EPOCH_NO_V4`

Stage 0 passed exact partial-load/no-op/freeze/gradient checks. D7c-gated RARM
then passed one-epoch and five-epoch adapter-only no-collapse gates, but the
five-epoch matched-budget comparison did not justify continuation:

- D7c mean PSNR delta: `+0.02947239875793457`;
- FAM2 modres control mean PSNR delta: `+0.033065325419108074`;
- paired D7c minus control mean: `-0.0035929266611735024`;
- D7c regressions `<= -0.2 dB`: `50`;
- control regressions `<= -0.2 dB`: `91`;
- locked test touched: `false`.

D7c is safer in mild-tail regression count, but it does not beat the matched
control on mean utility. No further RARM training, v4 expansion, neighbor
unfreeze, canary expansion, or locked-test access is authorized from v3d.
