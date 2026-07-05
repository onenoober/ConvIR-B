# Haze4K v2.29 NoPost ILFRB-ACS Safe OOF Action-Bank Calibration

Date: 2026-07-05

Branch: `codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`

Route id: `haze4k_v2_29_nopost_ilfrb_acs_safe_oof_action_bank_calibration_20260705`

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`

## Hypothesis

v2.28 created OOF no-op/useful/unsafe action strata but failed because
negative-control and raw-candidate tail risk were too high. v2.29 tests whether
safety-envelope calibration, bucket-aware strength, stage/action pruning, and a
GT-free OOF table policy can preserve OOF gains while reducing plausible
negative-control unsafe exposure.

## Parent And Architecture Contract

- Parent route: `codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`
- Parent commit: `82f4752`
- Model architecture delta versus v2.28: none.
- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output, RGB output-output residual, or learned RGB
  post-output correction is introduced.
- Training is blocked for this route.
- P2B selector probe is blocked until P2A safe-envelope and GT-free table-policy
  gates pass.
- Locked Haze4K test is blocked.

## P2A Audit Scope

The audit reuses the v2.28 train-derived OOF prototype setup and evaluates
safety-envelope variants:

- `raw_v228_baseline`
- `energy_norm`
- `rms_clip`
- `absmax_clip`
- `alignment_gate`
- `bucket_strength_grid`
- `s5_only`
- `s5_plus_s6_hard_only`
- `s5_plus_s4_mild_only`
- `energy_norm_plus_bucket_strength`
- `energy_norm_plus_bucket_strength_plus_alignment_gate`

It also splits diagnostic negative controls into:

- impossible sanity controls: `sign_flip`, `overstrong_3.0`;
- plausible miscalibration controls: `overstrong_1.5`, `overstrong_2.0`;
- plausible routing errors: `cross_bucket_mismatch`, `wrong_stage`.

The impossible controls should remain unsafe enough to prove sensitivity. The
plausible miscalibration/routing controls must be reduced before any selector
or training work is authorized.

## P2A Gates

Safety-envelope selected policy must retain:

- overall no-op conservative preference between `0.10` and `0.50`;
- easy/top25 no-op or mild preference at least `0.40`;
- hard/bottom25 medium or strong preference at least `0.30`;
- selected mean at least `+0.40`;
- selected hard at least `+0.80`;
- selected easy at least `0.00`;
- p05 at least `-0.15`;
- CVaR5 at least `-0.35`;
- severe rate at most `0.035`;
- fold-tail pass at least `4/5`.

Safety-envelope raw/deployable and plausible-control bounds:

- deployable mild unsafe rate at most `0.20`;
- deployable medium unsafe rate at most `0.30`;
- deployable strong unsafe rate at most `0.35`;
- deployable strong unsafe rate on easy/top25 at most `0.20`;
- cross-bucket mismatch unsafe rate at most `0.35`;
- hard-to-easy cross-bucket severe rate at most `0.35`;
- overstrong 1.5 unsafe rate at most `0.35`;
- wrong-stage unsafe rate at most `0.30`;
- sign-flip unsafe rate at least `0.60`;
- overstrong 3.0 unsafe rate at least `0.55`.

GT-free table policy must pass:

- table-policy mean at least `+0.30`;
- table-policy hard at least `+0.60`;
- table-policy easy at least `0.00`;
- table-policy p05 at least `-0.15`;
- table-policy CVaR5 at least `-0.35`;
- table-policy severe rate at most `0.035`;
- table-policy fold-tail pass at least `4/5`.

Failure at these gates is a normal pause and does not authorize P2B, training,
or locked-test evaluation.

## Result Summary

The P2A audit completed on `convir-4090` at source commit `aa1bd1b` with
decision `P2A_FAIL_SAFE_OOF_ACTION_BANK_CALIBRATION_PAUSE`. This is a normal
gate pause. P2B selector probing, training, and locked Haze4K test evaluation
were not launched.

Best safety-envelope variant: `bucket_strength_grid`.

- selected mean/hard/easy: `+0.833159 / +1.375280 / +0.334240`
- p05/CVaR5/severe rate: `0.0 / 0.0 / 0.0`
- no-op rate: `0.2375`
- easy no-op or mild rate: `1.0`
- hard medium or strong rate: `0.7`
- fold-tail pass: `5/5`

The variant preserved useful OOF stratification but did not pass safety:

- deployable mild unsafe rate: `0.215625` above the `<=0.20` gate
- cross-bucket mismatch unsafe rate: `0.455556` above the `<=0.35` gate
- hard-to-easy cross-bucket severe rate: `0.716667` above the `<=0.35` gate
- overstrong 1.5 unsafe rate: `0.441667` above the `<=0.35` gate

Best GT-free table-policy variant: `energy_norm_plus_bucket_strength`.

- table-policy mean/hard/easy: `+0.077516 / +0.240312 / +0.037175`
- table-policy p05/CVaR5/severe rate: `-0.219939 / -0.433256 / 0.0625`
- table-policy fold-tail pass: `3/5`
- table pass: `0`

Conclusion: bucket-aware strength improves selected policy safety and
stratification relative to the raw v2.28 bank, but the current safety envelope
still fails on plausible cross-bucket, hard-to-easy, and overstrong risks. The
GT-free table policy is too weak to authorize selector probing. The next route,
if any, must first solve hard-to-easy cross-bucket leakage and GT-free table
selection before P2B, training, or locked-test use.

## Evidence

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_29_nopost_ilfrb_acs_safe_oof_action_bank_calibration_20260705/`

Compact text artifacts:

- `README.md`
- `status.txt`
- `v229_p0_arch_contract_delta.md`
- `v229_p2a_negative_control_taxonomy.csv`
- `v229_p2a_deployable_candidate_tail_by_bucket_stage.csv`
- `v229_p2a_safety_envelope_variant_summary.csv`
- `v229_p2a_oof_table_policy_report.csv`
- `v229_p2a_oracle_selected_table_policy_gap.md`
- `v229_p2a_cross_bucket_directionality_report.csv`
- `v229_p2a_stage_action_pruning_report.csv`
- `v229_p2a_fold_tail_report.csv`
- `v229_p2a_noop_useful_unsafe_base_rate_report.json`
- `v229_p2a_failonly_diagnostic_probe.csv`
- `v229_p2a_closeout.json`
- `run_v229_p2a.sh`
- `monitor_v229.sh`

## Locked-Test Policy

Locked test is blocked. This route uses only train-derived Haze4K samples and
does not launch training or selector fitting for deployment.
