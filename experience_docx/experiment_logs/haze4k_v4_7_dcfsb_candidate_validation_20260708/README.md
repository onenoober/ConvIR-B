# Haze4K v4.7 DCFSB Candidate-Lock Validation Evidence

Route id: `haze4k_v4_7_dcfsb_candidate_validation_20260708`

Branch: `codex/haze4k-v4-7-dcfsb-candidate-validation`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Policy: fixed v4.6 `adapter4`; train-derived internal validation first, then one fixed locked-test confirmation command after a written gate. Prediction images were not saved.

## Result

Final status: `COMPLETED_LOCKED_TEST_CONFIRM_FAIL_NO_PROMOTION`

Internal candidate-lock passed:

- mean dPSNR `0.044404`
- positive ratio `0.625000`
- p5 dPSNR `-0.216141`
- bootstrap 95% CI low `0.024481`
- sign-test p `3.802649e-05`
- systematic failure flags `0`

Locked-test confirmation failed:

- A0 mean PSNR `34.145502`
- candidate mean PSNR `34.149328`
- mean dPSNR `0.003826`
- positive ratio `0.484000`
- p5 dPSNR `-0.210819`
- mean dSSIM `0.00002084`

Decision: do not promote adapter4; do not run more locked-test commands for this candidate.

## Primary Artifacts

- `v47_candidate_lock.json`
- `v47_adapter4_internal256_bootstrap.json`
- `v47_adapter4_sign_test.json`
- `v47_adapter4_failure_atlas.md`
- `v47_adapter4_band_error_by_proxy_bins.csv`
- `v47_adapter4_worst32_compact.csv`
- `v47_adapter4_per_image_compact.csv`
- `decision_after_v47.md`
- `post_v47_locked_test_policy_note.md`
- `locked_test_confirmation_gate.md`
- `locked_test_once/v47_locked_test_summary.json`
- `locked_test_once/decision_after_locked_test.md`
- `v47_closeout.json`

Cloud-only by default: `locked_test_once/v47_locked_test_per_image.csv`.
