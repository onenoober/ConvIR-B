# Haze4K DTA-v3.6 HRCS Evidence

Date: 2026-06-13

Status: `FORMAL_COMPLETED_RELAXED_PASS_STRICT_FAIL_FIXED_POLICY_READY_LOCKED_TEST_UNTOUCHED`

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
- `v36_formal_oof_per_image_action_table.csv`
- `v36_formal_v35style_oracle_risk_coverage_curve.csv`
- `v36_formal_v35style_selector_nested_calibration_report.{json,csv}`
- `formal_hrcs/v36_high_coverage_rejection_curve.csv`
- `formal_hrcs/v36_high_coverage_rejection_curve_aggregate.csv`
- `formal_hrcs/v36_risk_feature_auc_report.csv`
- `formal_hrcs/v36_selector_reliability_bins.csv`
- `formal_hrcs/v36_selector_error_table.csv`
- `formal_hrcs/v36_action_bank_oracle_vs_selector.csv`
- `formal_hrcs/v36_selector_summary.json`
- `formal_hrcs/v36_selector_best_configs.csv`

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

## Phase C Formal Train-Derived Completion

Formal 5-fold x 3-seed train-derived validation completed on `convir-4090` from branch commit `6f5965e` with marker:

```text
DTA_V3_6_HRCS_FORMAL_QUEUE_OK 2026-06-13T09:13:12+08:00
```

The queue completed `45` candidate train runs, `180` train-derived depth-control evals, and `45` aggregate jobs for L1/L2/L3 across folds `0..4` and seeds `3407/3411/2026`. The formal action table has `27000` rows. The v3.5-style nested selector postprocess completed with `DTA_V3_5_NESTED_SELECTOR_OK rows=27000 reports=15 variants=3`, and the HRCS selector postprocess completed with `DTA_V3_6_HRCS_SELECTOR_OK rows=27000 curve_rows=1830 best_configs=3`.

Best formal deployable relaxed configs:

| Candidate | Selector | Feature group | Coverage target | Actual coverage | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | Strict | Relaxed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| L1 `s004_g025` | logistic | `deployable_all` | `0.92` | `0.8961` | `+0.075380` | `+0.071317` | `+0.00000481` | `0.5876` | `56.60` | `71.67` | fail | pass |
| L2 `s002_g025` | logistic | `deployable_all` | `0.92` | `0.8931` | `+0.054269` | `+0.056842` | `+0.00001268` | `0.5788` | `39.47` | `50.33` | fail | pass |
| L3 `s004_g015` | logistic | `deployable_all` | `0.93` | `0.8887` | `+0.065404` | `+0.067793` | `+0.00001324` | `0.5817` | `47.07` | `59.33` | fail | pass |

High-coverage oracle remains strict-pass for L1/L3:

| Candidate | Oracle coverage | mean dPSNR | hard bottom-25 | positive ratio | worst/600 | max outer worst/600 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| L1 `s004_g025` | `0.93` | `+0.107726` | `+0.085384` | `0.6373` | `31.40` | `31.40` | pass |
| L1 `s004_g025` | `0.95` | `+0.101305` | `+0.081291` | `0.6373` | `43.40` | `43.40` | pass |
| L3 `s004_g015` | `0.93` | `+0.093538` | `+0.078527` | `0.6362` | `19.87` | `19.87` | pass |
| L3 `s004_g015` | `0.95` | `+0.087922` | `+0.073806` | `0.6362` | `31.87` | `31.87` | pass |
| L3 `s004_g015` | `0.97` | `+0.081127` | `+0.069427` | `0.6362` | `43.87` | `43.87` | pass |

Formal action-bank diagnostic:

| Policy | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | true-vs-zero | true-vs-shuffle | true-vs-normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 only | `+0.000000` | `+0.000000` | `+0.00000000` | `0.0000` | `0.00` | `+0.000000` | `+0.000000` | `+0.000000` |
| L2 only | `+0.051001` | `+0.041160` | `-0.00002317` | `0.6293` | `52.07` | `+0.054045` | `+0.040811` | `+0.044337` |
| L3 only | `+0.062992` | `+0.051758` | `-0.00002262` | `0.6362` | `61.87` | `+0.065923` | `+0.051136` | `+0.055593` |
| L1 only | `+0.072968` | `+0.058823` | `-0.00002455` | `0.6373` | `73.40` | `+0.080619` | `+0.063915` | `+0.069640` |
| Oracle choose `{A0,L2,L3,L1}` | `+0.143298` | `+0.121101` | `+0.00002551` | `0.6623` | `0.00` | `+0.146342` | `+0.133108` | `+0.136634` |
| Selector choose `{A0,L2,L3,L1}` | `+0.065880` | `+0.060523` | `+0.00000697` | `0.5980` | `48.87` | `+0.065107` | `+0.050217` | `+0.054715` |

Formal interpretation: strict gates still fail for every deployable selector, mainly because coverage and all-image positive ratio fall below the strict requirements. L3 logistic `deployable_all` is the best balanced fixed train-derived policy for any later relaxed one-shot locked test: it passes formal relaxed gates and the formal worst/max-outer-worst lines (`47.07/600`, max `59.33/600`), while preserving mean/hard/dSSIM and depth attribution. It still is not promotion-ready because coverage is only `0.8887` and positive ratio is `0.5817`.

Locked Haze4K test remains untouched in this evidence. If the user continues the explicit relaxed one-shot override, the fixed policy to seal before test is: `candidate=L3 l3_fdf_lite_s004_g015_bm2`, `selector=logistic`, `feature_group=deployable_all`, `coverage_target=0.93`, fallback to A0 on reject. No threshold, feature, checkpoint, or action-bank membership may be changed using locked-test feedback.

## Continuing Plan

Phase C is complete. Phase D is now technically ready only as a one-shot locked-test run under the user override, using the fixed L3 logistic `deployable_all` policy above. Because formal strict gates failed, any locked-test result must be labeled relaxed/exploratory and must not be used for further selection or tuning.

## Decision State

Formal train-derived validation completed. The route is relaxed-pass but strict-fail; locked test remains untouched. Candidate action and oracle headroom are strong, but deployable risk calibration still loses too many positive samples for promotion.
