# Haze4K v2.0 C0 Strong Candidate Zoo Oracle

Decision: `C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED`

This audit uses existing train-derived/internal validation evidence only. Locked test data was not touched.

## A0 + FullUDP Endpoint Oracle

| metric | value |
| --- | ---: |
| `count` | `600` |
| `mean_dPSNR` | `0.7416947523752848` |
| `hard_bottom25_dPSNR` | `1.1109101994832358` |
| `easy_top25_dPSNR` | `0.39711181640625` |
| `dSSIM` | `0.00022958377997080485` |
| `positive_ratio` | `0.53` |
| `nonnegative_ratio` | `1.0` |
| `worst_per_600` | `0.0` |
| `intervention_rate` | `0.53` |

## Gate

- `mean_ge_0_30`: `True`
- `hard_ge_0_30`: `True`
- `positive_ge_0_75`: `False`
- `easy_ge_neg_0_02`: `True`
- `dssim_ge_0`: `True`
- `worst_per_600_le_5`: `True`

## Interpretation

- The FullUDP endpoint remains unsafe as a global replacement; use `v20_candidate_zoo_single_expert_summary.csv` and `v20_candidate_zoo_failure_bins.csv` for the damage profile.
- The A0-preserving endpoint oracle is the capacity signal for the next phase; any deployable route must learn abstention and preservation rather than transplanting FullUDP globally.
- ConvIR-L/DehazeFormer/PromptIR were not available on `convir-4090` during C0a, so they are logged as future candidate slots rather than silently skipped.
