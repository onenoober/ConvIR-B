# Haze4K v2.27 NoPost ILFRB-ACS Evidence

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill.md`

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`

Runtime server: `convir-4090`

Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-27-nopost-ilfrb-action-conditioned-selective-distill`

Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for P0-P5.

## Key Results

- P0 passed: strict partial load clean, forbidden symbol hits `0`, identity max abs versus A0 `0.0`.
- P1 passed: best insertion row `S6_early_mid_final` had mean `+7.8509 dB`, hard bottom25 `+9.4244 dB`, easy top25 `+6.1829 dB`, p05 `+4.5170 dB`, CVaR5 `+3.8851 dB`, severe `0`.
- P2 failed normally: no-op conservative preference count `0/80`; hard medium/strong preference rate `1.0`; strong unsafe rate `0.0`.
- P3/P4/P5/P6 were not launched.
- Training was not launched.
- Locked Haze4K test was untouched.

Decision: `P2_FAIL_ACTION_BANK_STRATIFICATION_PAUSE`.

## Primary Files

- `v227_closeout.json`
- `v227_p0_source_contract_report.md`
- `v227_p0_partial_load_manifest.json`
- `v227_p0_identity_vs_a0.json`
- `v227_p1_insertion_oracle_summary.csv`
- `v227_p1_fold_tail_report.csv`
- `v227_p2_action_bank_replay.csv`
- `v227_p2_noop_coverage_report.json`
- `v227_p2_strength_safety_curve.csv`
- `run_v227_p0.sh`
- `run_v227_p1_p5.sh`
- `monitor_v227.sh`
- `status.txt`

This directory is intended for compact text evidence only. Do not sync checkpoints, weights, image outputs, arrays, archives, or raw feature dumps by default.
