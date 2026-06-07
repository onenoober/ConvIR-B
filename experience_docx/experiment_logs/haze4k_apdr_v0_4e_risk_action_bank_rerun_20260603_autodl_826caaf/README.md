# Haze4K APDR-v0.4E Risk Action Bank Fixed-Code Rerun

Date: 2026-06-03

Status: fixed-code rerun archived; locked Rule A was not available in the clean
candidate table, Rule B passed E0 confirm, and the route remains blocked pending
OOF safety.

## Pointers

- Route card:
  `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4e-risk-calibrated-action-bank.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/apdr_family_summary.md`

## Primary Files

- `v04e_locked_threshold_confirm_summary.json`
- `v04e_candidate_action_table.csv`
- `v04e_candidate_action_per_image_sigma3.csv`
- `v04e_risk_feature_auc.csv`
- `v04e_oof_calibration_curve.csv`
- `v04e_accepted_vs_rejected_groups.csv`
- `v04e_strong_failure_signature.csv`
- `v04e_risk_action_bank_apdr_v0_4e_risk_action_bank_sigma3_seed3407.log`
- `status.txt`

## Key Result

The clean rerun used sigma `3.0`, train indices `0..127`, and confirm indices
`256..383`.

- Rule A, `global_plus_spatial_kenel_knn_9`, was marked
  `missing_candidate`, so the old Rule A pass is not sealed.
- Rule B, `spatial_priors_ridge_10`, K16, scale `1.0`, kept `45/128` images
  and passed the E0 confirm gate with mean `+0.21414 dB`, hard bottom-25%
  `+0.45278 dB`, easy top-25% `+0.06253 dB`, strong/severe `1/0`, and oracle
  recovery `0.23628`.

Decision label:

```text
FIXED_CODE_E0_PARTIAL_RULEB_PASS_OOF_REQUIRED
```

This is diagnostic evidence only. E2, full router, local correction, dense
residual training, and stop20 remain blocked unless OOF/held-out gates pass.

## Artifact Boundary

This directory is text-only evidence. Checkpoints, images, datasets, tensor
caches, and raw inference outputs are intentionally excluded.
