# Haze4K v2.37 Tail-Safe Same-Context WDMamba Eligibility and Preservation Evidence

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for all phases in this route.

## Evidence Files

- `status.txt`
- `run_v237_p0_alpha_safety_sweep.sh`
- `run_v237_p1_p4_tail_safe_audits.sh`
- `runtime_logs/v237_p0_alpha_safety_sweep.log`
- `runtime_logs/v237_p1_tail_failure_atlas.log`
- `runtime_logs/v237_p2_mask_preservation_sweep.log`
- `runtime_logs/v237_p3_oof_mask_selection.log`
- `runtime_logs/v237_p4_target_only_eligibility.log`
- `v237_p0_alpha_safety_sweep_per_image.csv`
- `v237_p0_alpha_safety_sweep_summary.json`
- `v237_p0_closeout.json`
- `v237_p1_tail_failure_atlas.csv`
- `v237_p1_tail_failure_summary.json`
- `v237_p1_closeout.json`
- `v237_p2_mask_preservation_sweep_per_image.csv`
- `v237_p2_mask_preservation_sweep_summary.csv`
- `v237_p2_mask_preservation_sweep_summary.json`
- `v237_p2_closeout.json`
- `v237_p3_oof_mask_selection_per_image.csv`
- `v237_p3_oof_mask_selection_summary.csv`
- `v237_p3_closeout.json`
- `v237_p4_target_only_eligibility_features.csv`
- `v237_p4_target_only_eligibility_per_fold.csv`
- `v237_p4_target_only_eligibility_oof_summary.json`
- `v237_p4_closeout.json`
- `v237_decision_tree.md`
- `v237_closeout.json`

## Metric Contract

P0 uses cached full-image A0 and WDMamba tensors from v2.35 and recomputes
alpha blends offline against Haze4K train GT. P0 does not rerun WDMamba.

Buckets are based on full-image A0 PSNR: bottom 25% is hard, top 25% is easy,
and strong-reference is A0 PSNR greater than or equal to the 75th percentile.
Fold IDs reuse the v2.36 P0 fold manifest.

P1/P2/P3 are authorized only if no unmasked alpha passes P0. P4 is authorized
only if P3 passes. P5 is blocked unless both P3 and P4 pass.
