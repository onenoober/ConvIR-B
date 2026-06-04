# APDR-v0.4B Basis-Only Coefficient Router Gate B

Date: 2026-06-03

Status: completed AutoDL Gate B router diagnostic. Follow-up Gate C later
failed; do not launch local correction or stop20 from this route.

## Purpose

Gate 0 showed that derived low-frequency bases are expressive enough. This
stage trains only an image-to-coefficients router for those frozen bases. It
does not add local correction, does not run Gate C, and does not authorize
stop20.

## Command

Run on `autodl-dehaze3`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/run_apdr_v0_4b_basis_router_gateb_sigma3.sh
```

Optional smoke:

```bash
BASIS_NUM_IMAGES=128 EVAL_COUNT=32 FIT_COUNT=0 K_VALUES=16 STEPS=300 bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/run_apdr_v0_4b_basis_router_gateb_sigma3.sh
```

## Expected Artifacts

- `basis_router_gateb_summary_sigma3.json`
- `basis_router_gateb_history_sigma3.csv`
- `basis_router_gateb_per_image_sigma3.csv`
- `basis_router_gateb_groups_sigma3.csv`
- `basis_router_gateb_train_lowspace_sigma3.csv`
- `basis_router_gateb_apdr_v0_4b_basis_router_gateb_sigma3_seed3407.log`
- `status.txt`

These are text diagnostics only. Checkpoints, tensor caches, arrays, datasets,
and image outputs stay out of the repo.

## Gate B

| Metric | Pass line |
| --- | ---: |
| weighted delta L1 drop | `>= 0.50` |
| pred-target corr | `>= 0.50` |
| oracle recovery | `>= 0.30` |
| hard train gain | `>= +0.30 dB` |
| easy train gain | `>= -0.010 dB` |
| strong/severe regressions | `0 / 0` |

Passing Gate B only authorizes a train128/mini-val Gate C diagnostic. It still
does not authorize local correction or stop20.

`FIT_COUNT=0` means train on every open sample inside the `EVAL_COUNT` subset.
That is the intended overfit32 Gate B mode.

## Results

The full run derived bases from all `3000` train images, then trained on every
open sample inside the first 32-image subset (`26` open samples). Both K16 and
K32 passed Gate B.

| K | Gate B | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | pass | `0.5419` | `0.8233` | `0.6962` | `+0.9796 dB` | `+0.2179 dB` | `0 / 0` |
| 32 | pass | `0.6361` | `0.8803` | `0.7891` | `+1.0860 dB` | `+0.2687 dB` | `0 / 0` |

Low-space training diagnostics also passed: K16 field L1 drop `0.5431`, corr
`0.8894`; K32 field L1 drop `0.6372`, corr `0.9279`.

## Decision

Decision label:
`GATEB_PASS_CONTINUE_BASIS_ROUTER_TRAINVAL_NO_LOCAL`.

K32 is the preferred next configuration because it has stronger field learning
and output recovery than K16 while still passing preservation. The next
experiment is a train128/mini-val Gate C diagnostic for basis-only K32. Do not
add local correction or run stop20 yet.

Follow-up note: the K32 train128/mini-val Gate C diagnostic in
`../haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/` failed on
mini-val. The route-level decision is now
`GATEC_FAIL_STOP_BASIS_ROUTER_MAPPING_NO_LOCAL`.
