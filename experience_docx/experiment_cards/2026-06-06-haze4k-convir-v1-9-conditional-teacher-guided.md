# ConvIR-Dehaze-v1.9-ConditionalTeacherGuided

Date: 2026-06-06

Status: completed cloud queue; internal gates failed. Local WSL is editing and
syntax-only; all runtime steps ran on `dehaze1`.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: conditional UDP/depth/teacher-guided route.
- Baseline expert `E0`: official ConvIR-B A0.
- Hard expert `E1`: official UDPNet checkpoint from v1.5.
- Locked Haze4K test policy: blocked. This route uses only train-derived
  `train_inner`, `val_regular`, and `val_hard`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_v19_conditional_teacher_guided_20260606/`.

## Motivation

v1.5 showed official UDPNet has hard-case headroom but fails global
preservation. v1.6/v1.7 showed a large A0+UDP oracle, but deployable image-level
switch/mix routers failed promotion or heldout gates. v1.8 closed the small
BiDPFM1 partial-unfreeze route as negative. v1.9 therefore moves up one level:
use UDPNet as a conditional teacher and learn where the teacher is safe.

## Queue

| ID | Experiment | Output | Continue rule |
| --- | --- | --- | --- |
| V19-Q0 | Physical-prior inventory | transmission/airlight candidate tables | Missing physical priors are a blocker record, not a queue stop. |
| V19-Q1 | Teacher-delta predictability ceiling | pre-router vs post-expert OOF/heldout summary | Always continue. |
| V19-Q2 | Patch alpha oracle | tile oracle CSV, per-image oracle CSV, summary JSON | Always continue if extraction succeeds; otherwise record engineering failure and continue later table steps when possible. |
| V19-Q3 | Patch mask head | OOF/heldout mask policy summary | Always continue. |
| V19-Q4 | Conditional distillation student | 3 seed train/eval, A0 comparison JSON/CSV, aggregate | Continue through all seeds even if one seed fails. |
| V19-Q5 | Optimizer hygiene panel | short student runs for grad clip and EMA variants | Continue through all panel rows. |

## Results

V19-Q0 physical-prior inventory completed with physical priors available:
`3000` train transmission files were found under `dataset/HAZE4K/train/trans`,
with transmission candidate count `6266` and airlight candidate count `158`.

V19-Q1 teacher-delta predictability found positive pre-router and post-expert
signals but failed tail gates. Best pre-router OOF was mean `+0.4565 dB`, hard
bottom-25 `+0.3846 dB`, easy top-25 `+0.5551 dB`, worst regression ratio
`0.2143`, and strong regression ratio `0.2067`. Best pre-router heldout was
mean `+0.4527 dB`, hard bottom-25 `+0.4623 dB`, easy top-25 `+0.5732 dB`,
worst ratio `0.2300`, and strong ratio `0.1972`. Decision:
`PRE_ROUTER_PREDICTABILITY_GATE_FAIL_ROUTE_NEEDS_INTERNAL_OR_PATCH_POLICY`.

V19-Q2 patch alpha oracle completed and passed the mechanism reading strongly:
count `3000`, tile count `193500`, mean `+1.6086 dB`, hard bottom-25
`+1.4500 dB`, easy top-25 `+1.6249 dB`, SSIM `+0.000548`, and worst/strong
regression ratios `0`.

V19-Q3 patch mask head failed deployable mask gates. OOF was mean `+0.3714 dB`,
hard bottom-25 `+0.2229 dB`, easy top-25 `+0.5235 dB`, worst ratio `0.1083`,
and strong ratio `0.1013`. Heldout was mean `+0.3176 dB`, hard bottom-25
`+0.2557 dB`, easy top-25 `+0.4718 dB`, worst ratio `0.1267`, and strong ratio
`0.1400`. Decision: `MASK_HEAD_GATE_FAIL_CONTINUE_STUDENT_EXPERIMENTS`.

V19-Q4 conditional distillation student failed the 3-seed screen. Aggregate
regular mean PSNR delta was `-1.0591 dB`, regular easy top-25 `-1.1413 dB`,
regular SSIM `-0.001251`, hard mean `-0.6418 dB`, hard bottom-25 `-0.6731 dB`,
and hard SSIM `-0.001374`. Regular worst `<= -0.20 dB` count mean was
`240.33`; hard worst count mean was `207.67`. All `3/3` seeds selected
`model_5`, and all `3/3` seed decisions were
`NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`.

V19-Q5 optimizer hygiene panel completed training-only runs for
`clip0p001_noema`, `clip0p01_noema`, `clip0p1_noema`, and `clip0p01_ema`; each
summary is `TRAIN_COMPLETE_PENDING_INTERNAL_EVAL`. These runs do not change the
Q4 screen decision.

## Gates

Mechanism gate:

```text
patch_oracle mean_delta >= +0.30 dB
patch_oracle hard_bottom25_delta >= +0.50 dB
patch_oracle easy_top25_delta >= 0.00 dB
worst_regression_ratio <= 0.02
```

Mask/router gate:

```text
OOF mean_delta >= +0.12 dB
OOF hard_bottom25_delta >= +0.20 dB
OOF easy_top25_delta >= -0.02 dB
OOF worst_ratio <= 0.06
Heldout mean_delta >= +0.12 dB
Heldout hard_bottom25_delta >= +0.20 dB
Heldout easy_top25_delta >= -0.02 dB
```

Student screen gate:

```text
n >= 3
regular mean PSNR delta CI lower >= +0.03 dB
hard-bottom25 PSNR delta CI lower >= +0.08 dB
easy-top25 mean >= -0.03 dB
regular/hard SSIM means >= 0
```

This card does not authorize locked test.

## Cloud Contract

- Remote workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-v1-9-conditional-teacher`.
- Python: `/root/miniconda3/envs/convir-cu128/bin/python`.
- Data root:
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- UDPNet checkpoint:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Split JSON:
  `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- tmux session: `v19_cond_teacher`.

## Decision

Decision label: `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`.

Locked Haze4K test remained blocked and was not touched.
