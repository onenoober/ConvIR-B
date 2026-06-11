# Haze4K DTA-v3 DAPC Fine-Tune Evidence

Date: 2026-06-11

Status: `COMPLETED_GATE_FAIL_PHASE_A_R0_NO_PHASE_B`

This directory stores text evidence for `codex/haze4k-dta-v3-dapc-finetune`.
Checkpoints, model weights, datasets, images, arrays, archives, and raw inference
outputs are not committed by default. Contact-sheet PNGs are generated on
`convir-4090` for visual judgment and recorded by path only unless explicitly
requested otherwise.

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: to be verified by setup; expected `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.

## Execution Order

1. Setup remote workspace from GitHub branch and run cloud static `py_compile`.
2. Generate/verify train-derived OOF splits.
3. Run DTA-v3 preflight from A0 partial-load.
4. Run Phase A R0 zero-depth fine-tune and evaluate vs A0.
5. Run Phase B frozen-R0 depth fine-tune with invert/normal/zero/deterministic shuffle matrix.
6. Run output-refine-only, FiLM-only, trans-head-only, and phys-blend-only ablations.
7. Generate tail/win contact sheets on cloud for visual inspection.
8. Sync text evidence to GitHub and keep locked Haze4K test blocked unless internal gates pass.

## Planned Text Artifacts

- `setup_convir4090_dta_v3.sh`
- `run_dta_v3_preflight_convir4090.sh`
- `run_dta_v3_phase_a_r0_convir4090.sh`
- `run_dta_v3_phase_b_depth_matrix_convir4090.sh`
- `status.txt`
- `dta_v3_preflight.json/log`
- `depth_eval_pairing_audit.csv/json`
- `train_eval_depth_matrix.json/csv`
- `r0_vs_rdepth_attribution.csv`
- ablation JSON/CSV/log files
- contact-sheet generation logs and remote image paths

## Current Decision

`COMPLETED_GATE_FAIL_PHASE_A_R0_NO_PHASE_B`: convir-4090 Stage 0 preflight
passed, but Phase A `dta_r0_only` OOF20 fold0 failed the R0 safety/quality gate.
Phase B frozen-R0 depth training is blocked until R0 is redesigned or the user
explicitly overrides the stop rule. Locked Haze4K test remains blocked.

## 2026-06-11 Default Host Correction

The route default cloud host has been changed to `convir-4090` by user instruction.
The earlier `convir-5090` SSH blocker is superseded and is retained only in
`status.txt` as historical setup evidence. Runtime validation, training, eval,
and contact-sheet generation should now use `ssh convir-4090` and the
`/sda/home/wangyuxin/ConvIR-B/...` runtime paths.

## 2026-06-11 Convir-4090 Preflight

Stage 0 preflight completed on `convir-4090` with `DTA_V3_PREFLIGHT_OK`.

- R0 preflight: partial load `602` official keys, `29` missing keys all under `DTA.`, unexpected `[]`, synthetic no-op max diff `0.0`, real-batch DTA grad sum `0.06329656`.
- Depth-bounded preflight: partial load `602` official keys, `29` missing keys all under `DTA.`, unexpected `[]`, synthetic max diff `0.00024414` under the written bounded tolerance, real-batch DTA grad sum `0.19976439`.
- OOF split generation: five train-derived folds, `600` validation images per fold.
- Deterministic eval shuffle audit: `600` fold0 validation rows, `same_image_count=0`, `same_image_ratio=0.0`, density-bin match ratio `0.275`.
- Locked Haze4K test remains blocked.

## 2026-06-11 Phase A R0 OOF20 Fold0

Cloud run `oof20_phaseA_r0_seed3407_f0` completed training and evaluation on
`convir-4090`; the follow-up `t_pred` audit returned `KeyError: t_pred` because
`dta_phase=r0` / `dta_ablation=r0_only` intentionally disables the
transmission/depth branch. The Phase A launcher now records that audit as
not-applicable for R0 instead of treating it as model evidence.

Phase A metrics from `dta_v3_oof20_phaseA_r0_seed3407_f0_compare/`:

| Metric | Value |
| --- | ---: |
| common images | `600` |
| mean dPSNR vs A0 | `-0.012119` |
| hard bottom-25 dPSNR | `-0.069025` |
| easy top-25 dPSNR | `+0.044643` |
| mean dSSIM | `-0.00001218` |
| positive ratio | `0.4500` |
| strong regressions among top-25 A0 images (`<= -0.05 dB`) | `48/150` |
| worst regressions (`<= -0.20 dB`) | `70/600` |

Decision: Phase A R0 is not a safe generic zero-depth residual baseline. It is
negative on mean and hard samples, SSIM is slightly negative, and the positive
ratio is below the written continuation threshold. Do not launch Phase B from
this checkpoint. Recommended next route is a smaller/shorter R0 diagnostic or a
more conservative loss/scale variant declared as a new Phase A attempt.

## 2026-06-11 Phase A R0 Conservative Scout Queue

Next action follows the Phase A gate-fail recommendation: do not start Phase B;
instead run conservative R0-only fine-tune scouts in parallel on `convir-4090`.
The queue uses train-derived fold0 only, keeps locked Haze4K test blocked, and
generates cloud-only contact-sheet PNGs for visual judgment.

Scout variants:

| Variant | LR | R0 scale | preserve/ref/tail weights | Intent |
| --- | ---: | ---: | ---: | --- |
| `r0s005_lr3e5_ref005` | `3e-5` | `0.005` | `0.05/0.05/0.05` | near-no-op safety floor |
| `r0s010_lr3e5_ref005` | `3e-5` | `0.010` | `0.05/0.05/0.05` | conservative midpoint |
| `r0s020_lr3e5_ref005` | `3e-5` | `0.020` | `0.05/0.05/0.05` | reduced-capacity recovery vs failed `0.04` |
| `r0s010_lr1e5_ref010` | `1e-5` | `0.010` | `0.10/0.10/0.10` | strongest preservation pressure |

Initial stage is `scout5full`: 5 training epochs with full 600-image fold0 eval
and contact sheets. Only a variant with positive mean/hard movement, near-zero
or positive SSIM, and reduced tail risk should be promoted to OOF20.


## 2026-06-11 Conservative R0 Scout Results

`scout5full` completed on `convir-4090` using four GPUs in workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-r0variants`.
Each run trained 5 epochs on fold0 train and evaluated all `600` fold0 validation
images. Contact sheets were generated on cloud under `tail_regression_contact_sheet/`
and are not committed as Git evidence.

| Variant | mean dPSNR | hard bottom-25 | easy top-25 | dSSIM | pos ratio | strong | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r0s020_lr3e5_ref005` | `-0.021837` | `-0.056151` | `+0.030270` | `-0.00002035` | `0.4550` | `38` | `57` |
| `r0s010_lr3e5_ref005` | `-0.025856` | `-0.056900` | `+0.024449` | `-0.00002182` | `0.4417` | `35` | `56` |
| `r0s005_lr3e5_ref005` | `-0.031866` | `-0.057293` | `+0.014219` | `-0.00002624` | `0.4267` | `38` | `49` |
| `r0s010_lr1e5_ref010` | `-0.042601` | `-0.051842` | `-0.012940` | `-0.00003352` | `0.3567` | `43` | `43` |

Decision: all conservative R0 variants are still mean/hard/SSIM negative. The
least bad mean result is `r0s020_lr3e5_ref005` at `-0.021837 dB`, still worse
than A0 and hard-negative. No variant is promoted to OOF20, and Phase B remains
blocked under the original frozen-R0 plan. The next useful diagnostic is to test
a zero-R0 depth-direct branch from A0 as a separate no-promotion mechanism probe.


## 2026-06-11 Depth-Direct Scout Plan

Because both the original R0 and conservative R0-only scouts failed, Phase B
from a frozen learned R0 remains blocked. To avoid spending more cycles on a
bad generic residual, the next cloud-only diagnostic is `depthDirect`: initialize
from official A0 with partial DTA load, freeze A0, keep `R0` at zero
(`dta_r0_residual_scale=0.0`), and train only the depth/transmission branch.
This is a no-promotion mechanism probe, not a replacement for the blocked Phase
B gate.

The first `scout5full` queue trains four depth modes in parallel on
`convir-4090`: `invert`, `normal`, `zero`, and `shuffle`. Each trained model is
evaluated under `invert/normal/zero/shuffle` with deterministic eval shuffle,
aggregated into `train_eval_depth_matrix_*`, and gets cloud-only contact sheets.
Gate defaults are intentionally a bit wider for this diagnostic:
`gate_limit=0.12`, `gamma_limit=0.20`, `beta_limit=0.10`, dense mask budget
`0.14`, and depth residual scale `0.08`.
