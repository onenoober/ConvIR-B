# v2.14 N1R Leakage Report

This replay separates GT/teacher-derived columns from runtime-available features.

## Excluded Runtime Columns

- `A0_PSNR`
- `A0_SSIM`
- `WD0375_PSNR`
- `WD0375_SSIM`
- `WD0375_dPSNR`
- `WD0375_dSSIM`
- `hazy_PSNR`

## Leakcheck Metrics

- `benefit_label` leakcheck `hazy_PSNR` ROC-AUC: `0.684934`
- `benefit_label` hazy-runtime ROC-AUC: `0.770909`
- `benefit_label` all-runtime ROC-AUC: `0.811898`
- `benefit_label` all-with-leak ROC-AUC: `0.811809`
- `severe_risk_label` leakcheck `hazy_PSNR` ROC-AUC: `0.698620`
- `severe_risk_label` hazy-runtime ROC-AUC: `0.798088`
- `severe_risk_label` all-runtime ROC-AUC: `0.826237`
- `severe_risk_label` all-with-leak ROC-AUC: `0.824894`

Conclusion: `hazy_PSNR` is treated only as an oracle-leak sentinel. It is excluded from `hazy_runtime`, `all_runtime`, and all pass/fail gates.
