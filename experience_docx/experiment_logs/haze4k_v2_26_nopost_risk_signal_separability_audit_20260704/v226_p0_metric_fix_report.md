# v2.26 P0 metric fix report

Decision: `V226_P0_V225A_AP_PASS_INVALID_DIAGNOSTIC_ARTIFACT`

- v2.25A OOF count: `480`
- probability std: `0.0016692509020246116`
- ROC-AUC: `0.5500802065808521`
- AP old tuple-sort implementation: `0.4937745923792122`
- AP tie-aware implementation: `0.13965163934426228`
- label base rate: `0.12708333333333333`

The gate failure remains valid because probability spread, ROC-AUC, and target-probability MAE still fail.
The old AP pass is not positive evidence when scores are tied or constant.
