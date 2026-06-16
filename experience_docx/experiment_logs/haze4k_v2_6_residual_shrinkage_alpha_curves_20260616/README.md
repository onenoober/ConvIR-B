# Haze4K v2.6 Residual Shrinkage Alpha Curves Evidence

Status: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`

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

## Outcome

The parallel cloud run completed on `convir-4090` from source commit
`ee059bcb5278a878e8e3b1bf153dbd8bfd01eaaf`. The launcher selected GPUs
`2 3 4`, all three expert jobs exited with `rc=0`, and the main status recorded
`V26_ALPHA_CURVES_OK` at `2026-06-16T14:41:23+08:00`.

Decision: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`.

Key train-derived alpha-curve readout on C8 `val_regular + val_hard`:

| Expert | Alpha | Mean dPSNR | Hard dPSNR | Easy dPSNR | Positive | Severe / 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WDMamba | `0.125` | `+0.860475` | `+1.054049` | `+0.586756` | `0.991667` | `2` |
| WDMamba | `0.375` | `+2.512202` | `+3.505615` | `+1.189484` | `0.973333` | `11` |
| WDMamba | `1.0` | `+3.578052` | `+8.276923` | `-1.048537` | `0.768333` | `124` |
| FSNet+UDP | `0.125` | `+0.561473` | `+0.522577` | `+0.610280` | `0.978333` | `1` |
| FSNet+UDP | `0.375` | `+1.602301` | `+1.623987` | `+1.581052` | `0.970000` | `14` |
| FSNet+UDP | `0.75` | `+2.605198` | `+3.325834` | `+1.884767` | `0.916667` | `40` |
| FSNet+UDP | `1.0` | `+2.642392` | `+4.206927` | `+1.204795` | `0.858333` | `71` |
| MB-TaylorFormerV2-L | `0.125` | `+0.485463` | `+0.653630` | `+0.259786` | `0.905000` | `21` |
| MB-TaylorFormerV2-L | `0.375` | `+1.078024` | `+1.896762` | `+0.007126` | `0.803333` | `99` |
| MB-TaylorFormerV2-L | `1.0` | `+0.011106` | `+2.852408` | `-3.255472` | `0.486667` | `294` |

Interpretation:

- WDMamba has a broad train-derived safety/utility interval from `0.125` to
  `0.50`, and the already selected `WD0375` point is not isolated.
- FSNet+UDP also shows a strong residual-shrinkage interval; `0.125` through
  `0.75` meet the strict positive/tail readout, with `1.0` raising tail risk.
- MB-TaylorFormerV2-L only supports a narrow small-alpha safety claim; medium
  and full alpha are high-risk despite hard-sample gain.
- This route strengthens the claim from a single WDMamba alpha point to a
  train-derived residual-shrinkage phenomenon for at least WDMamba and FSNet+UDP.
  It does not establish cross-dataset transfer or sample-adaptive alpha.

Primary files:

- `v26_all_expert_alpha_grid.csv`
- `v26_all_expert_group_min.csv`
- `v26_compact_comparison.csv`
- `v26_decision.md`
- `v26_summary.json`

Locked-test status:

```text
locked_test_touched=false
locked_per_image_read=false
locked_outputs_as_targets=false
```
