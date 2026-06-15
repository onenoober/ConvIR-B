# Haze4K v2.4 C12 WD0375 Distillation Feasibility

Date: 2026-06-15

Status: `PLANNED`

## Scope

- Objective: test whether the locked-pass fixed `WD0375` profile can be
  compressed into a single official ConvIR-B student without using locked
  outputs.
- Runtime host: `convir-4090` only.
- Route branch: `codex/haze4k-v2-4-c12-wd0375-distill`.
- Architecture anchor: `github/codex/haze4k-official-arch-anchor`.
- Remote workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Locked-Test Contract

C12 must not read locked per-image outputs, run locked Haze4K, tune from locked
metrics, or use locked outputs as distillation targets. The only teacher is the
train-derived `WD0375 = A0 + 0.375 * (WDMamba - A0)` profile generated on
Haze4K train-core images.

The C11 selector is not a C12 teacher because its locked one-shot worsened
positive ratio and severe risk relative to WD0375.

## Data Contract

Use C8 train-derived validation names as held-out validation:

```text
val_regular: 300 Haze4K train images from C8
val_hard:    300 Haze4K train images from C8
train_core:  matched Haze4K train images excluding val_regular and val_hard
locked test: forbidden
```

## Fixed Teacher

For each train-core image:

```text
A0 = official ConvIR-B haze4k-base.pkl
WDMamba = haze4k_35.88.pth
teacher = clamp(A0 + 0.375 * (WDMamba - A0), 0, 1)
```

Teacher cache is generated once and reused by all student variants.

## Screen Variants

The first C12 screen uses official ConvIR-B architecture initialized from
`haze4k-base.pkl` and trains only with train-core data:

```text
c12_gt075_teacher025_lr1e-5
c12_gt050_teacher050_lr1e-5
c12_gt025_teacher075_lr1e-5
c12_teacher100_lr1e-5
```

All variants use the same train-core split, random crop `256`, no locked data,
and a short screen budget before any longer formal run.

## C12 Screen Gates

Relative to A0 on the held-out 600 train-derived validation images:

```text
mean >= +1.00 dB
hard_bottom25 >= +1.00 dB
easy_top25 >= +0.80 dB
positive >= 0.90
severe <= 36/600
dSSIM >= 0
```

If no screen variant passes, stop C12 and keep WD0375 as the deployment teacher.
If one variant passes with margin, C12 may run a separate formal multi-seed
distillation route.

## Required Outputs

C12-0:

- `v24_c12_0_route_card.md`
- `v24_c12_0_no_locked_status.txt`
- `v24_c12_0_source_manifest.json`
- `v24_c12_split_manifest.json`

C12-A:

- `v24_c12_teacher_cache_manifest.json`
- `v24_c12_teacher_cache_metrics.csv`
- `status_teacher_cache.txt`

C12-B:

- one `status_train_<variant>.txt` per variant
- one train log per variant
- checkpoint existence report

C12-C:

- `v24_c12_eval_<variant>_<checkpoint>_summary.json`
- `v24_c12_eval_<variant>_<checkpoint>_per_image.csv`
- `v24_c12_screen_leaderboard.csv`
- `v24_c12_decision.md`

## Decision Labels

```text
C12_SCREEN_RUNNING
C12_SCREEN_PASS_FORMAL_DISTILLATION_REVIEW
C12_SCREEN_FAIL_KEEP_WD0375_TEACHER
C12_FAILED_INFRA
```
