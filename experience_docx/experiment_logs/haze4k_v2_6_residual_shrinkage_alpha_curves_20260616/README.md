# Haze4K v2.6 Residual Shrinkage Alpha Curves Evidence

Status: `PLANNED_LOCKED_TEST_UNTOUCHED`

Route card:
`experience_docx/experiment_cards/2026-06-16-haze4k-v2-6-residual-shrinkage-alpha-curves.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v26-residual-shrinkage-alpha-curves`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Scope: C8 train-derived `val_regular + val_hard` only.
- Locked Haze4K: untouched and forbidden.

## Experiment Scope

This supplemental route addresses only the first two requested evidence layers:

1. WDMamba cross-alpha residual shrinkage curve.
2. Cross-expert residual shrinkage curves for WDMamba, FSNet+UDP, and
   MB-TaylorFormerV2-L.

The fixed candidates are:

```text
candidate(E, alpha) = A0 + alpha * (E - A0)
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
E in {WDMamba, FSNet+UDP, MB-TaylorFormerV2-L}
```

No router, selector, threshold, checkpoint, or locked-test result is tuned here.

## Planned Evidence Files

```text
commands/run_v26_alpha_curves_parallel.sh
commands/monitor_v26_alpha_curves.sh
runtime_logs/v26_<expert>_alpha_curve.log
status_v26_alpha_curves.txt
status_v26_<expert>_alpha_curve.txt
v26_<expert>_alpha_curve_alpha_grid.csv
v26_<expert>_alpha_curve_alpha_group_metrics.csv
v26_<expert>_alpha_curve_alpha_group_min.csv
v26_<expert>_alpha_curve_single_summary.csv
v26_<expert>_alpha_curve_per_image.csv
v26_all_expert_alpha_grid.csv
v26_all_expert_group_min.csv
v26_compact_comparison.csv
v26_summary.json
v26_decision.md
```

## Decision Labels

```text
PLANNED_LOCKED_TEST_UNTOUCHED
V26_ALPHA_CURVES_RUNNING
V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED
FAILED_INFRA
FAILED_COMMAND
```
