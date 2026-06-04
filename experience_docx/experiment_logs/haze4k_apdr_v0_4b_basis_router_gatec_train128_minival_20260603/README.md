# APDR-v0.4B Basis Router Gate C Train128/Mini-Val

Date: 2026-06-03

Status: completed on `autodl-dehaze3`; Gate C failed. Do not run local
correction, stop20, or a longer scout from this basis-only router.

## Purpose

Gate B showed that the K32 basis-only coefficient router can overfit a small
open subset. Gate C tests whether the same router generalizes when trained on
open samples from the first 128 train images and evaluated on the first 256
train images, with the second half treated as mini-val.

## Command

Run on `autodl-dehaze3`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/launch_apdr_v0_4b_basis_router_gatec_tmux.sh
```

The launched run used:

```text
BASIS_NUM_IMAGES=0
TRAIN_COUNT=128
EVAL_COUNT=256
FIT_COUNT=0
LOW_SIZE=32
K_VALUES=32
STEPS=2000
```

`FIT_COUNT=0` means train on every open sample inside the train scope. This run
trained on `90` open samples from the first 128 images. The derived basis was
fit from all `3000` train images.

## Artifacts

- `basis_router_gatec_train128_minival_summary_sigma3.json`
- `basis_router_gatec_train128_minival_history_sigma3.csv`
- `basis_router_gatec_train128_minival_per_image_sigma3.csv`
- `basis_router_gatec_train128_minival_groups_sigma3.csv`
- `basis_router_gatec_train128_minival_train_lowspace_sigma3.csv`
- `basis_router_gateb_apdr_v0_4b_basis_router_gatec_train128_minival_sigma3_seed3407.log`
- `status.txt`
- `tmux_exit_apdr_v04b_gatec_t128_20260603_172754.txt`

These are text diagnostics only. Checkpoints, tensor caches, arrays, datasets,
and image outputs stay out of the repo.

## Results

| Split | Gate | Count | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | pass | 128 | `0.7219` | `0.8880` | `0.8414` | `+1.2889 dB` | `+0.1836 dB` | `0 / 0` |
| mini-val | fail | 128 | `-0.3435` | `0.2154` | `0.0428` | `+0.4309 dB` | `-0.3551 dB` | `11 / 25` |
| overall | fail | 256 | `0.1364` | `0.5121` | `0.3548` | `+0.9117 dB` | `-0.0681 dB` | `11 / 25` |

Mini-val contained 114 open and 14 closed images. Closed images retained zero
output by gate construction, but open mini-val mean gain was only `+0.0570 dB`
against oracle mean gain `+1.1875 dB`.

## Decision

Decision label:
`GATEC_FAIL_STOP_BASIS_ROUTER_MAPPING_NO_LOCAL`.

The train split confirms the router still memorizes the train-scope mapping.
The mini-val split shows the learned image-to-basis coefficient mapping does
not generalize: target correlation collapses, weighted field error is worse
than zero output, easy images regress, and strong/severe regressions appear.

This stops APDR-v0.4B in its current basis-only router form. The next route, if
continued, should change the deployable mapping input or router family before
retesting Gate B/C. Do not use local correction to hide this failure, and do
not run stop20 from this evidence.
