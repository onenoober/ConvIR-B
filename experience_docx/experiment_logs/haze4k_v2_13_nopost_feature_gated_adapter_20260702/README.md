# Haze4K v2.13 NoPost Feature-Gated Adapter Evidence

Status: `N1_MECHANISM_FAIL_STOP_BEFORE_TRAINING_LOCKED_TEST_UNTOUCHED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Data:
`/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`

A0 checkpoint:
`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`

Split source:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/v24_c12_split_manifest.json`

WD0375 teacher cache:
`/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher`

Locked Haze4K test: untouched and blocked.

## Stage Outputs

N0 contract audit:

- `v213_n0_nopost_contract.md`
- `v213_n0_source_audit.json`
- `v213_n0_forbidden_symbol_scan.txt`
- `v213_n0_forward_signature.json`

N1 feature separability:

- `v213_n1_feature_table_manifest.json`
- `v213_n1_oof_gain_risk_probe.csv`
- `v213_n1_feature_ablation_report.csv`
- `v213_n1_calibration_report.json`
- `v213_n1_decision.md`

N2 identity:

- `v213_n2_identity_summary.json`
- `v213_n2_identity_per_image.csv`
- `v213_n2_param_groups.json`
- `v213_n2_state_dict_load_report.txt`

N3 microfit:

- `v213_n3_microfit16_history.csv`
- `v213_n3_microfit64_history.csv`
- `v213_n3_microfit256_history.csv`
- `v213_n3_microfit_leaderboard.csv`
- `v213_n3_gate_action_stats.csv`
- `v213_n3_decision.md`

N4 staged screen:

- `v213_n4_fold_seed_summary.csv`
- `v213_n4_gate_action_distribution.csv`
- `v213_n4_failure_taxonomy.csv`
- `v213_n4_decision.md`

Raw feature tables, checkpoints, images, and large runtime outputs remain
cloud-only by default.

## Closeout

Cloud run: `convir-4090`, source commit `cd3442f`.

Decision: `N1_MECHANISM_FAIL_STOP_BEFORE_TRAINING`.

N0 contract passed:

- forbidden symbol hits: `0`;
- adapter forbidden args present: `false`;
- final `rgb_residual + x` count: `1`;
- synthetic max_abs_vs_A0: `0`;
- real-sample max_abs_vs_A0: `0`.

N1 failed the predeclared mechanism gate:

- rows: `2400`;
- benefit all-feature AUC: `0.811809`;
- severe-risk all-feature AUC: `0.824894`;
- benefit internal/hazy AUC: `0.802239` / `0.799183`;
- severe-risk internal/hazy AUC: `0.819616` / `0.833268`.

The AUC levels are usable, but severe-risk prediction is stronger from hazy-only
features than from internal ConvIR features. Training is therefore paused before
N3 to avoid building a route that behaves like input-rule post-processing.

N2 identity closeout passed:

- max_abs_vs_A0: `0`;
- trainable prefix: `nopost_adapter.`;
- trainable parameters: `74162`;
- frozen parameters: `8630665`;
- partial-load official keys loaded: `602`;
- missing new-module keys: `18`;
- unexpected/shape mismatch: `0`.

N3/N4/N5/N6/N7 were not launched. Locked Haze4K test remains untouched.
