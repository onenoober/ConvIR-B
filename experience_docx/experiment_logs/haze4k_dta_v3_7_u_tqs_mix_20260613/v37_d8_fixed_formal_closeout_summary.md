# DTA-v3.7 D8 Fixed Formal Closeout Summary

Date: 2026-06-14

Decision: `D8_FIXED_FORMAL_STRICT_PASS_LOCKED_TEST_UNTOUCHED`.

D8 completed on `convir-4090` in
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9`
with locked Haze4K test untouched. It used only the sealed primary policy
`primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` and did not perform a
new policy search.

## Final Metrics

| policy id | coverage | mean | hard | easy | dSSIM | positive | worst/600 | max outer worst/600 | true vs zero | true vs shuffle | true vs normal | strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` | `1.0000` | `+0.078297` | `+0.085281` | `+0.062693` | `+0.00001904` | `0.65875` | `46.50` | `52.00` | `+0.083620` | `+0.064332` | `+0.068854` | pass |

Counts: `fixed_policy_count=1`, `strict_consistent_count=1`, `image_groups=2400`,
`rows=38400`, and `outputdiff_rows=135000`.

## Evidence Status

Completion markers copied locally:

```text
dta_v3_7_phase_d8_stage2_real_blend_done 2026-06-14T09:57:29+08:00
dta_v3_7_phase_d8_stage3_outputdiff_done 2026-06-14T10:23:58+08:00
DTA_V3_7_PHASE_D8_STAGE2_FAST_OK 2026-06-14T10:24:18+08:00
```

The cloud run generated exact final artifacts at:

- `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_d8_fixed_formal_summary.json`
- `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_d8_fixed_formal_policy_aggregate.csv`
- `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_d8_fixed_formal_per_outer_report.csv`
- `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_d8_fixed_formal_fixed_policy_config.json`
- `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_d8_fixed_formal_selected_actions.csv`

A later SSH probe failed with `connect to host 183.175.12.124 port 22:
Connection refused` at 2026-06-14T10:44:48+08:00, so the exact generated JSON/CSV
files are pending copy. The companion `*_recovered.*` files in this directory
record the final observed closeout values for GitHub sync.

## Next Action

The route is eligible for exactly one fixed one-shot locked Haze4K confirmation
using the sealed D8 policy. Do not tune thresholds, features, action-bank
membership, checkpoints, or code from the locked-test result.
