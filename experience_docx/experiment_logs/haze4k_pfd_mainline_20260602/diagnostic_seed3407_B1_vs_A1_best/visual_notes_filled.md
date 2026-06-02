# B1 Fixed Diagnostic Visual Notes

Date: 2026-06-02

Status: B1 diagnostic sidecar completed; B1 remains diagnostic-only and should not be promoted.

## Inputs

- Sidecar command source: direct checkpoint sidecar, not the full training script.
- Diagnostic directory: `diagnostic_seed3407_B1_vs_A1_best/`
- Selected samples: `57`
- Missing selected samples: `0`
- Visual panel: `visual_panel_20.png` on AutoDL artifact storage and local review copy only; not intended for GitHub text sync.
- Text evidence files retained for sync: `diagnostic_summary.json`, `sample_manifest.csv`, `output_safety_stats.csv`, `pfd_branch_stats.csv`, `pfd_branch_stats_by_category.json`, and this note.

## Visual Read

The worst-regression rows show visible output failures, not a metric-only or image-read/write artifact. Several catastrophic and severe rows have obvious brightness/color drift and over-processing in the B1 output. The candidate-vs-A1 and candidate-vs-GT columns show large low-frequency image changes rather than tiny texture differences.

Hard-gain rows can look cleaner on haze removal, so RHFD can move the intended hard subset. However, the same branch also changes easy or strong-reference samples enough to create visible artifacts or preservation loss. This makes B1 a useful diagnostic, but not a stable replacement candidate.

Easy-regression and strong-reference rows show over-intervention risk. Some differences are small-looking but still measurable, while the catastrophic rows are clearly visible and dominate the route safety concern.

## Output Safety Evidence

The catastrophic-worst group has large candidate safety shifts:

- mean `pred_mean_delta_b1_minus_a1`: `-0.558286`
- mean `rgb_mean_shift_abs_mean_b1`: `0.559414`
- mean `luma_mean_shift_b1`: `-0.609393`
- mean `ratio_pred_lt_0_b1`: `0.182485`
- mean `ratio_pred_gt_1_b1`: `0.078795`

The worst-regression top-10 also shows a strong safety pattern:

- mean `pred_mean_delta_b1_minus_a1`: `-0.224593`
- mean `rgb_mean_shift_abs_mean_b1`: `0.228993`
- mean `luma_mean_shift_b1`: `-0.246092`
- mean `ratio_pred_lt_0_b1`: `0.073735`
- mean `ratio_pred_gt_1_b1`: `0.040179`

This supports the conclusion that B1 failures include real output range, brightness, and color/low-frequency drift problems.

## RHFD Activity Evidence

RHFD activity is low in absolute magnitude, but it is not selective enough across categories:

| Category | RHFD2 delta norm ratio mean | RHFD1 delta norm ratio mean |
| --- | ---: | ---: |
| best_gain_top10 | `0.00337822` | `0.00380220` |
| hard_gain | `0.00334674` | `0.00395850` |
| easy_regression | `0.00300373` | `0.00350040` |
| easy_preserved | `0.00291748` | `0.00347487` |
| catastrophic_worst | `0.00307366` | `0.00361298` |
| worst_regression_top10 | `0.00310331` | `0.00367933` |

The activation pattern does not cleanly separate hard gains from easy regressions or catastrophic failures. That means B1 RHFD is acting like a broad residual feature adapter rather than a preservation-aware hard-case intervention.

## Gate Interpretation

`gate_B1_stop20_with_diagnostic.json` is for closure evidence only:

- B1 fails global, easy-preservation, and strong-reference checks.
- The severe-regression check passes only against the loose diagnostic threshold `< 444`; this must not be treated as a promotion signal.
- A stricter deployable-candidate severe threshold such as `<= 150/1000` would still reject B1.

## Decision

B1 should remain diagnostic-only. The clearest failure cause is insufficient RHFD selectivity/preservation: the branch can help some hard samples, but it also over-intervenes on easy/strong-reference cases and produces real brightness/color/range failures in catastrophic rows.

Recommended next action: do not launch B2/B3 from this B1 as-is. Either redesign RHFD with an explicit preservation gate or move to the already isolated B1r decoder-RHFD preservation route.
