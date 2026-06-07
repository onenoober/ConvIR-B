# Haze4K APDR-v0.4E OOF Calibration Fixed-Code Rerun

Date: 2026-06-03

Status: fixed-code OOF rerun archived; OOF gate failed and no retained
low-capacity threshold policy passed.

## Pointers

- Route card:
  `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4e-oof-calibration.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/apdr_family_summary.md`

## Primary Files

- `v04e_oof_locked_threshold_summary.json`
- `v04e_oof_policy_search_summary_sigma3.json`
- `v04e_oof_candidate_action_per_image_sigma3.csv`
- `v04e_oof_candidate_action_table.csv`
- `v04e_oof_fold_assignments.csv`
- `v04e_oof_locked_threshold_by_fold.csv`
- `v04e_oof_risk_feature_auc.csv`
- `v04e_oof_policy_search_sigma3.csv`
- `v04e_oof_strong_failure_signature.csv`
- `v04e_oof_finalize_from_intermediate_826caaf.log`
- `status.txt`

## Key Result

The fixed-code OOF rerun covered `3000` train images, `5` folds, and `1324`
open images.

- Rule A, `global_plus_spatial_kenel_knn_9`, was marked
  `missing_candidate`.
- Rule B, `spatial_priors_ridge_10`, K16, scale `1.0`, kept `150/3000`
  images, coverage `0.0500`, mean `+0.03779 dB`, hard bottom-25%
  `+0.13524 dB`, easy top-25% `+0.00000 dB`, strong/severe `0/1`, and oracle
  recovery `0.08345`.
- Post-hoc policy search retained `1600` low-capacity rows and found
  `0` gate-passing policies. The best retained policy had coverage `0.08767`,
  mean `+0.07916 dB`, hard `+0.25271 dB`, and strong/severe `0/0`, but missed
  the predeclared `0.10` coverage line.

Decision label:

```text
FIXED_CODE_E1_FAIL_STOP_CURRENT_V04E_THRESHOLDS
```

Do not run E2, full spatial router, local correction, dense residual training,
or stop20 from this v0.4E route.

## Artifact Boundary

This directory is text-only evidence. Checkpoints, images, datasets, tensor
caches, and raw inference outputs are intentionally excluded.
