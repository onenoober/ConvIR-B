# Haze4K v2.5 C13 A0-Frozen Residual Distillation Evidence

Status: `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`

Route card:
`experience_docx/experiment_cards/2026-06-15-haze4k-v2-5-c13-a0-frozen-residual-distillation.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Teacher: existing C12 train-core WD0375 cache only.
- Locked Haze4K: untouched and forbidden.

## Planned Evidence Files

```text
v25_c13_split_manifest.json
v25_c13_0_no_locked_status.txt
v25_c13_0_source_manifest.json
v25_c13_0_c12_failure_audit.md
v25_c13_0_c12_failure_audit.json
v25_c13_0_c12_failure_audit.csv
v25_c13_a3_adaptive_scalar_microfit_leaderboard.csv
v25_c13_a3_adaptive_scalar_microfit_decision.json
v25_c13_eval_c13a3_adaptive025_Best_summary.json
v25_c13_eval_c13a3_adaptive025_Best_per_image.csv
v25_c13_eval_c13a3_adaptive050_Best_summary.json
v25_c13_eval_c13a3_adaptive050_Best_per_image.csv
v25_c13_a4_fixed_scale_microfit_leaderboard.csv
v25_c13_a4_fixed_scale_microfit_decision.json
v25_c13_eval_c13a4_scale050_Best_summary.json
v25_c13_eval_c13a4_scale050_Best_per_image.csv
v25_c13_eval_c13a4_scale055_Best_summary.json
v25_c13_eval_c13a4_scale055_Best_per_image.csv
v25_c13_a5_a4_scale_sweep_leaderboard.csv
v25_c13_a5_a4_scale_sweep_decision.json
v25_c13_eval_a5_a4sweep_<tag>_summary.json
v25_c13_eval_a5_a4sweep_<tag>_per_image.csv
v25_c13_eval_<variant>_<checkpoint>_summary.json
v25_c13_eval_<variant>_<checkpoint>_per_image.csv
v25_c13_screen_leaderboard.csv
v25_c13_group_min_table.csv
v25_c13_screen_decision.json
v25_c13_decision.md
status_c13_a3_adaptive_scalar_microfit.txt
```

## Decision Labels

```text
PLANNED_LOCKED_TEST_UNTOUCHED
C13_AUDIT_FAILED_ENGINEERING
C13_MICROFIT_RUNNING
C13_MICROFIT_OK_START_B_SCREEN
C13_A3_MICROFIT_RUNNING
C13_A3_MICROFIT_OK_START_B_SCREEN
C13_A3_MICROFIT_FAIL_STOP_OR_REDESIGN
C13_A4_MICROFIT_RUNNING
C13_A4_MICROFIT_OK_START_B_SCREEN
C13_A4_MICROFIT_FAIL_STOP_OR_REDESIGN
C13_A5_SCALE_SWEEP_RUNNING
C13_A5_SCALE_SWEEP_OK_START_FULL_VAL
C13_A5_SCALE_SWEEP_FAIL_STOP_OR_REDESIGN
C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED
C13_SCREEN_RUNNING
C13_SCREEN_FAIL_STOP_OR_REDESIGN
C13_SCREEN_PASS_GROUP_MIN_FAIL_REVIEW
C13_SCREEN_PASS_GROUP_REVIEW
FAILED_INFRA
FAILED_COMMAND
```

## Current Outcome

The intermediate C13 sequence completed and stopped before C13-B.

- C13-0 audit passed.
- A3 adaptive scalar failed the quick gate.
- A4 fixed-scale microfit failed the quick gate.
- A5 post-hoc scale sweep failed the quick gate for all tested scales.

The current adapter family is learnable but not screen-ready. Locked Haze4K
remains untouched.
