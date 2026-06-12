# Haze4K DTA-v3.5 FDF-RCS-Lite Evidence

Date: 2026-06-12

Status: `COMPLETED_RELAXED_FLOW_PASS_STRICT_FAIL_SELECTOR_DIAGNOSTIC_LOCKED_TEST_BLOCKED`

Route card: `experience_docx/experiment_cards/2026-06-12-haze4k-dta-v3-5-fdf-rcs-lite.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- Locked test: blocked; train-derived OOF and nested calibration only.

## Primary Artifacts

- `status.txt`
- `run_dta_v3_5_fdf_rcs_lite_convir4090.sh`
- `launch_dta_v3_5_fdf_rcs_lite_triage_convir4090.sh`
- `dta_v3_5_fdf_rcs_triage_summary.json/csv`
- `dta_v3_5_fdf_rcs_triage_variant_summary.csv`
- `v35_oof_per_image_action_table.csv`
- `v35_oracle_risk_coverage_curve.csv`
- `v35_selector_nested_calibration_report.json/csv`
- `v35_selector_nested_selected_images.csv`
- per-run train/eval/aggregate logs and depth-control matrices.

## Completion State

The train-derived DTA-v3.5 queue completed on `convir-4090`. The initial queue
finished all non-L0 variants, then ended with `DTA_V3_5_TRIAGE_QUEUE_FAILED`
only because the original L0 A0 sanity eval path hit an engineering bug. The
L0 repair/postprocess queue was relaunched from commit `4c7589b` with two GPUs,
within the five-GPU follow-up cap, and completed successfully:

- `train_done_ok=24`: L1-L5 train runs, 6 variants x 2 folds x 2 seeds.
- `eval_done_ok=104`: all train-derived depth-control evals, including L0 repair.
- `aggregate_done_ok=26`: 24 trained runs plus two L0 A0 sanity runs.
- `DTA_V3_5_L0_REPAIR_POSTPROCESS_OK`: summary, oracle curve, and nested selector
  postprocess completed.
- Locked Haze4K test was not touched.

## Triage Summary

Decision: `RELAXED_FLOW_PASS_CONTINUE_NESTED_CALIBRATION_LOCKED_TEST_BLOCKED`.

| Variant | Runs | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | true-vs-zero | Relaxed | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `l0_a0_sanity` | 2 | `+0.000000` | `+0.000000` | `+0.00000000` | `0.0000` | `0.00` | `+0.000000` | no | no |
| `l1_fdf_lite_s004_g025_bm2` | 4 | `+0.071183` | `+0.081919` | `+0.00001536` | `0.6308` | `82.50` | `+0.069518` | yes | no |
| `l2_fdf_lite_s002_g025_bm2` | 4 | `+0.052706` | `+0.064818` | `+0.00001700` | `0.6250` | `60.50` | `+0.045710` | yes | no |
| `l3_fdf_lite_s004_g015_bm2` | 4 | `+0.062959` | `+0.075288` | `+0.00001736` | `0.6304` | `69.50` | `+0.056608` | yes | no |
| `l4_fdf_lite_tail_s004_g025_bm2` | 4 | `+0.066923` | `+0.078181` | `+0.00001603` | `0.6288` | `78.50` | `+0.061267` | yes | no |
| `l5_res010_tail_s004_g025_bm2` | 4 | `+0.066249` | `+0.078210` | `+0.00001698` | `0.6296` | `76.25` | `+0.055320` | yes | no |
| `l5_res015_tail_s004_g025_bm2` | 4 | `+0.066245` | `+0.078211` | `+0.00001696` | `0.6296` | `76.25` | `+0.055353` | yes | no |

## Nested Selector And Oracle

- The relaxed all-image flow succeeded for L1-L5, confirming that conservative
  FDF fixed much of the v3.4 over-action problem, but strict promotion gates
  still failed because worst-tail counts stayed above `48/600` and some positive
  ratios were only near the threshold.
- The nested threshold selector reduced selected-tail risk but was too low
  coverage for a strict route decision. The best relaxed selector diagnostic is
  L4 at about `0.21` coverage, selected mean about `+0.0197 dB`, selected
  positive ratio about `0.67`, and worst about `31.5/600`.
- The oracle risk-coverage curve is strong: non-L0 variants have positive
  oracle mean/hard at `0.50` coverage with zero worst regressions, and several
  variants remain tail-safe at much higher oracle coverage. This means the
  candidate has selector headroom, but the current nested selector is not yet
  strong enough.

## Decision

Keep DTA-v3.5 as a completed diagnostic, not a promotion candidate. Do not run
locked Haze4K test from this result. The next useful route should focus on a
stronger train-derived selector/calibration model and richer risk features,
rather than increasing residual/router capacity.
