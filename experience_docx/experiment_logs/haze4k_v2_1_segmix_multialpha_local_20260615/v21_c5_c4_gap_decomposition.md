# Haze4K v2.1 C5 C4 Failure Forensic

Decision: `C5_FORENSIC_COMPLETE_NO_POLICY_TUNING_START_C6_C7`

C5 only replays the sealed train-derived C2d/C4 family and decomposes the C4 gap. It does not select a new policy and does not touch locked test data.

## Gap Summary

- Mean positive-count deficit to 0.70: `12.000` images per seeded replay.
- Mean hard-bottom25 gap to +0.30 dB: `0.043611` dB.
- Hard-bottom25 rows with at least one safe high-alpha candidate in existing alpha grid: `97/150`.
- Selected-negative proxy rows written: `98`.

## Outputs

- `v21_c5_positive_deficit_report.csv`
- `v21_c5_false_positive_false_negative_bins.csv`
- `v21_c5_hard_bottom25_alpha_oracle.csv`
- `v21_c5_selected_negative_visual_proxy.csv`
- `v21_c5_summary.json`

## Interpretation Guardrail

C5 may motivate C6/C7 experiment design, but its replay is not a tuning source for locked data. Locked and distillation remain blocked.
