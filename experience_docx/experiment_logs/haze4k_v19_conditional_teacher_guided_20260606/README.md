# Haze4K v1.9 Conditional Teacher-Guided Queue

Route card:
`experience_docx/experiment_cards/2026-06-06-haze4k-convir-v1-9-conditional-teacher-guided.md`

Primary runtime script:
`run_v19_conditional_teacher_queue.sh`

Monitor:
`monitor_v19_conditional_teacher_queue.sh`

Locked Haze4K test touched: no.

Status: completed gate fail.

This queue recorded independent failures and continued through all planned
experiments on `dehaze1`. Local WSL was used only for editing and syntax-side
work.

## Key Results

- Q0 physical-prior preflight found train transmission data:
  `3000` files under `dataset/HAZE4K/train/trans`; transmission candidate count
  `6266`, airlight candidate count `158`.
- Q1 teacher-delta predictability failed tail gates. Best pre-router OOF was
  mean `+0.4565 dB`, hard bottom-25 `+0.3846 dB`, easy top-25 `+0.5551 dB`,
  but worst regression ratio `0.2143` and strong regression ratio `0.2067`.
  Best pre-router heldout was mean `+0.4527 dB`, hard bottom-25 `+0.4623 dB`,
  easy top-25 `+0.5732 dB`, but worst ratio `0.2300` and strong ratio
  `0.1972`.
- Q2 patch alpha oracle passed mechanism reading strongly: count `3000`, tile
  count `193500`, mean `+1.6086 dB`, hard bottom-25 `+1.4500 dB`, easy top-25
  `+1.6249 dB`, SSIM `+0.000548`, worst/strong regression ratios `0`.
- Q3 patch mask head failed deployable mask gates. OOF was mean `+0.3714 dB`,
  hard bottom-25 `+0.2229 dB`, easy top-25 `+0.5235 dB`, but worst ratio
  `0.1083`. Heldout was mean `+0.3176 dB`, hard bottom-25 `+0.2557 dB`, easy
  top-25 `+0.4718 dB`, but worst ratio `0.1267` and strong ratio `0.1400`.
- Q4 conditional student failed the 3-seed screen. Aggregate regular mean PSNR
  delta was `-1.0591 dB`, regular easy top-25 `-1.1413 dB`, regular SSIM
  `-0.001251`; hard mean was `-0.6418 dB`, hard bottom-25 `-0.6731 dB`, hard
  SSIM `-0.001374`. All `3/3` selected checkpoint labels were `model_5`, and
  all decisions were `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`.
- Q5 optimizer hygiene panel completed training-only short runs for
  `clip0p001_noema`, `clip0p01_noema`, `clip0p1_noema`, and
  `clip0p01_ema`; each summary is marked
  `TRAIN_COMPLETE_PENDING_INTERNAL_EVAL`. These runs do not rescue the Q4
  screen result.

Decision label:
`MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`.

Locked Haze4K test remained blocked and was not touched.

## Artifact Boundary

The local evidence sync excludes checkpoints and large tile CSVs:

- excluded checkpoint files under `checkpoints/`;
- excluded `v19_patch_alpha_oracle_tiles.csv`;
- excluded `v19_patch_mask_head_oof_tiles.csv`.

The synced local evidence includes command scripts, logs, JSON summaries,
per-image CSVs, and small policy/inventory tables.
