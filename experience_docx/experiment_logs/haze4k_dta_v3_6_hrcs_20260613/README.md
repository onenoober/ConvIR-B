# Haze4K DTA-v3.6 HRCS Evidence

Date: 2026-06-13

Status: `PHASE_A_COMPLETED_RELAXED_PASS_STRICT_FAIL_FORMAL_QUEUE_PENDING`

Route card: `experience_docx/experiment_cards/2026-06-13-haze4k-dta-v3-6-hrcs.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-6-hrcs`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- v3.5 source evidence: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612/`.
- Locked test: blocked by default; user requested one later relaxed fixed-policy override.

## Primary Artifacts

- `status.txt`
- `run_dta_v3_6_hrcs_phase_a_convir4090.sh`
- `run_dta_v3_6_hrcs_candidate_convir4090.sh`
- `launch_dta_v3_6_hrcs_formal_convir4090.sh`
- `v36_high_coverage_rejection_curve.csv`
- `v36_high_coverage_rejection_curve_aggregate.csv`
- `v36_risk_feature_auc_report.csv`
- `v36_selector_reliability_bins.csv`
- `v36_selector_error_table.csv`
- `v36_action_bank_oracle_vs_selector.csv`
- `v36_selector_summary.json`
- `v36_selector_best_configs.csv`

## Phase A Completion

Phase A ran on `convir-4090` from branch commit `754d62a` and completed with marker:

```text
DTA_V3_6_HRCS_PHASE_A_OK
```

The run consumed the v3.5 OOF action table (`15600` rows) and evaluated L1/L2/L3 with logistic and shallow-GBDT selectors over high-coverage targets `1.00..0.90`.

Best deployable relaxed configs:

| Candidate | Selector | Feature group | Coverage | mean dPSNR | positive ratio | worst/600 | Strict | Relaxed |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| L1 `s004_g025` | logistic | `input_only` | `0.9000` | `+0.075882` | `0.5846` | `63.25` | fail | pass |
| L2 `s002_g025` | logistic | `input_only` | `0.9017` | `+0.055817` | `0.5783` | `44.50` | fail | pass |
| L3 `s004_g015` | logistic | `input_only` | `0.9042` | `+0.067108` | `0.5862` | `52.25` | fail | pass |

High-coverage oracle still shows substantial headroom:

| Candidate | Oracle coverage | mean dPSNR | positive ratio | worst/600 | Strict |
| --- | ---: | ---: | ---: | ---: | --- |
| L1 `s004_g025` | `0.93` | `+0.106708` | `0.6308` | `40.50` | pass |
| L3 `s004_g015` | `0.93` | `+0.094361` | `0.6304` | `27.50` | pass |
| L3 `s004_g015` | `0.95` | `+0.088241` | `0.6304` | `39.50` | pass |

Action-bank diagnostic:

| Policy | mean dPSNR | positive ratio | worst/600 |
| --- | ---: | ---: | ---: |
| A0 only | `+0.000000` | `0.0000` | `0.00` |
| L2 only | `+0.052706` | `0.6250` | `60.50` |
| L3 only | `+0.062959` | `0.6304` | `69.50` |
| L1 only | `+0.071183` | `0.6308` | `82.50` |
| Oracle choose `{A0,L2,L3,L1}` | `+0.146107` | `0.6542` | `0.00` |
| Deployable selector choose `{A0,L2,L3,L1}` | `+0.063409` | `0.5833` | `50.25` |

Interpretation: high-coverage oracle can pass strict for L1/L3, so the conservative FDF action family has enough headroom. The deployable selector still fails strict because false rejection of good/positive samples lowers all-image positive ratio, even when worst is close to the strict line. Severe-regression AUC is only moderate for deployable features (best fold-level severe ROC-AUC about `0.70`), so Phase C remains relaxed/exploratory rather than promotion-grade.

## Continuing Plan

Phase C launches 5 folds x 3 seeds candidate evidence for L1/L2/L3 on `convir-4090`, using the unchanged v3.5 model configs and the relaxed exploratory queue. Phase D is a one-shot locked test only after a fixed train-derived selector/action policy is sealed.

## Decision State

Phase A completed. Proceeding to relaxed formal train-derived queue per user instruction; locked test remains untouched until that queue finishes and the fixed policy is written down.
