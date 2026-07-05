# Haze4K v2.32 NoPost Bounded Internal Low-Frequency Correction Field Evidence

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field.md`

Central index: `experience_docx/EXPERIMENT_INDEX.md`

Status: `P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`

Runtime server: `convir-4090`
Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field`
Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Hard blocks:

- `p2b_selector_probe_launched: false`
- `locked_test_touched: false`
- `rgb_output_output_residual: false`
- `learned_rgb_post_output_correction: false`

## Key Results

- P0 architecture/identity passed: `identity_max_abs_vs_A0=0.0`,
  strict partial-load loaded `602` official keys, and only `8` `BILFCF_` keys
  were newly initialized.
- P1 bounded-field sanity passed: all-bucket field energy mean `5.6611e-06`,
  p95 `1.1682e-05`, high-frequency leakage `0.02363`, gate mean `0.01822`.
- P2 canary32 failed normally: mean/hard/easy delta
  `-0.4146/-0.3287/-0.4371 dB`, p05/CVaR5/severe
  `-1.7719/-2.0842/0.5000`.
- P2 canary80 OOF and P3 objective ablation were not launched because canary32
  did not meet the continuation gate.

## Primary Files

- `v232_p0_arch_contract_delta.md`
- `v232_p0_identity_zero_init_report.json`
- `v232_p1_field_sanity_report.csv`
- `v232_p1_highfreq_leakage_report.csv`
- `v232_p1_field_sanity_closeout.json`
- `v232_p2_canary32_trainability_report.csv`
- `v232_p2_canary32_field_energy_by_bucket.csv`
- `v232_p2_canary32_easy_strong_reference_preservation.csv`
- `v232_p2_canary32_closeout.json`
- `v232_local_optimum_escape_audit.md`
- `v232_closeout.json`
- `run_v232_p0_preflight.sh`
- `run_v232_p1_sanity.sh`
- `run_v232_p2_canary32.sh`
- `run_v232_p2_canary80_oof.sh`
- `monitor_v232.sh`
- `status.txt`

This directory is intended for compact text evidence only. It excludes
checkpoints, weights, datasets, images, arrays, archives, and raw feature tables
by default.
