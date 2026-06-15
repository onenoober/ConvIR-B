# Haze4K v2.5 C13 A0-Frozen Residual Distillation Evidence

Status: `PLANNED_LOCKED_TEST_UNTOUCHED`

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
v25_c13_eval_<variant>_<checkpoint>_summary.json
v25_c13_eval_<variant>_<checkpoint>_per_image.csv
v25_c13_screen_leaderboard.csv
v25_c13_group_min_table.csv
v25_c13_screen_decision.json
v25_c13_decision.md
```

## Decision Labels

```text
PLANNED_LOCKED_TEST_UNTOUCHED
C13_AUDIT_FAILED_ENGINEERING
C13_MICROFIT_RUNNING
C13_MICROFIT_OK_START_B_SCREEN
C13_SCREEN_RUNNING
C13_SCREEN_FAIL_STOP_OR_REDESIGN
C13_SCREEN_PASS_GROUP_MIN_FAIL_REVIEW
C13_SCREEN_PASS_GROUP_REVIEW
FAILED_INFRA
FAILED_COMMAND
```
