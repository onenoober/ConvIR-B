# Haze4K DTA-v3 DAPC Fine-Tune Evidence

Date: 2026-06-11

Status: `COMPLETED_GATE_FAIL_TAILGUARD_WIDE_GATE_SCOUT`

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

## 2026-06-11 Depth-Direct Scout Results

`scout5full` depthDirect completed on `convir-4090` in workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-depthdirect`
from commit `12408fb`. This was a no-promotion mechanism probe: A0 stayed
frozen, `R0=0`, training scope was `dta_depth_only`, and each train-depth model
was evaluated under `invert/normal/zero/shuffle` using deterministic eval
shuffle. Locked Haze4K test remained blocked.

True-eval rows by train depth:

| train depth | true eval | mean dPSNR | hard bottom-25 | easy top-25 | dSSIM | pos ratio | strong | worst | true-vs-zero mean surplus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `invert` | `+0.013905` | `-0.005602` | `+0.036873` | `-0.00002676` | `0.5883` | `38` | `75` | `+0.032286` |
| `normal` | `normal` | `+0.008538` | `-0.008319` | `+0.030762` | `-0.00003284` | `0.5767` | `37` | `72` | `-0.009680` |
| `shuffle` | `shuffle` | `-0.004687` | `-0.020714` | `+0.018559` | `-0.00003603` | `0.5350` | `39` | `69` | `-0.002455` |
| `zero` | `zero` | `-0.004807` | `-0.020599` | `+0.018560` | `-0.00003586` | `0.5367` | `39` | `71` | n/a |

The important positive is attribution, not promotion: train=`invert` /
eval=`invert` beats eval=`zero` by `+0.032286 dB` mean with positive surplus
ratio `0.625`, which is the first DTA-v3 signal that satisfies the mean
true-vs-zero mechanism threshold. It still fails candidate gates: absolute mean
is only `+0.013905 dB`, hard is slightly negative, SSIM is negative, positive
ratio is below `0.65`, worst regressions rise to `75/600` versus eval-zero
`35/600`, and mean true-vs-shuffle is only `+0.028454 dB` (just under the
`+0.03 dB` gate).

Supporting text artifacts:

- `depth_direct_scout_summary.json` and `depth_direct_scout_summary.csv`.
- `train_eval_depth_matrix_scout5full_depthDirect_*_seed3407_f0.json/csv`.
- `r0_vs_rdepth_attribution_scout5full_depthDirect_*_seed3407_f0.csv`.
- deterministic shuffle audits `depth_eval_pairing_audit_scout5full_depthDirect_*_evalshuffle.json/csv`.
- cloud-only contact sheets under `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-depthdirect/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/tail_regression_contact_sheet/scout5full_depthDirect_*`.

Decision: `COMPLETED_MECHANISM_POSITIVE_NO_PROMOTION_DEPTHDIRECT_INVERT_ONLY`.
Keep locked test blocked. Do not revive learned R0 Phase B from the failed R0
checkpoints. Continue only with train=`invert` depth-direct variants that keep
the wider encoder gate but add stronger tail/SSIM and mask-budget protection.

## 2026-06-11 Depth-Direct Tail/SSIM Wide-Gate Scout Plan

Next cloud queue keeps the mechanism-positive setup (`R0=0`,
`train_scope=dta_depth_only`, train depth `invert`) and makes the DTA encoder
gates wider while varying the safety pressure. Each variant runs `scout5full` on
fold0, evaluates the full `invert/normal/zero/shuffle` matrix, and generates
cloud-only contact sheets.

| Variant | Gate/Gamma/Beta | depth scale | dense budget | preserve/ref/tail | intent |
| --- | ---: | ---: | ---: | ---: | --- |
| `wg16_tail08_s005_b10` | `0.16/0.24/0.12` | `0.05` | `0.10` | `0.08/0.08/0.08` | wider gate with lower depth action and moderate tail guard |
| `wg18_tail10_s006_b12` | `0.18/0.28/0.14` | `0.06` | `0.12` | `0.08/0.10/0.10` | recover surplus with stronger tail pressure |
| `wg20_tail12_s006_b10` | `0.20/0.30/0.15` | `0.06` | `0.10` | `0.10/0.12/0.12` | widest gate, lower LR, strongest guard |
| `wg16_tail06_s008_b08` | `0.16/0.24/0.12` | `0.08` | `0.08` | `0.06/0.06/0.06` | test whether smaller mask budget can keep surplus with less tail |

Continue condition: improve over the baseline depthDirect `invert` row by
reducing worst/strong regressions and SSIM loss while preserving at least
`+0.03 dB` mean true-vs-zero surplus. No locked test is allowed from these
scouts.

## 2026-06-11 Depth-Direct Tail/SSIM Wide-Gate Scout Results

The first wide-gate tail/SSIM queue completed on `convir-4090` in workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-tailguard`
from commit `debafdd`. All four variants used train=`invert`, `R0=0`,
`dta_depth_only`, wider encoder gates than the baseline depthDirect run, the
full eval matrix, and cloud-only contact sheets. Locked Haze4K test remained
blocked.

| Variant | true mean | true hard | dSSIM | pos ratio | true-vs-zero | worst true | worst zero | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wg16_tail06_s008_b08` | `-0.022292` | `-0.027152` | `-0.00002930` | `0.4450` | `+0.011628` | `37` | `34` | tail improved, quality/surplus failed |
| `wg16_tail08_s005_b10` | `-0.022990` | `-0.027368` | `-0.00002939` | `0.4317` | `+0.011423` | `37` | `34` | tail improved, quality/surplus failed |
| `wg18_tail10_s006_b12` | `-0.014497` | `-0.022719` | `-0.00002993` | `0.4917` | `+0.016380` | `46` | `35` | least negative mean, still failed |
| `wg20_tail12_s006_b10` | `-0.040233` | `-0.036566` | `-0.00002972` | `0.3533` | `+0.000958` | `39` | `37` | over-constrained |

Compared with the baseline depthDirect train=`invert` row (`+0.013905 dB` mean,
`+0.032286 dB` true-vs-zero surplus, `75/600` worst regressions), the wide-gate
tailguard queue reduced worst regressions to `37..46/600` but collapsed global
quality and mechanism surplus. The result is a useful negative: the current
strong guard/mask-budget setting protects the tail mostly by suppressing useful
depth action.

Supporting artifacts: `depth_direct_tailguard_scout_summary.json/csv`,
`train_eval_depth_matrix_scout5full_depthDirectTail_*`,
`r0_vs_rdepth_attribution_scout5full_depthDirectTail_*`, and cloud-only contact
sheets under `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-tailguard/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/tail_regression_contact_sheet/scout5full_depthDirectTail_*`.

Decision: `COMPLETED_GATE_FAIL_TAILGUARD_WIDE_GATE_SCOUT`. Do not promote. The
next quick scout should keep the wider gate but reduce guard pressure toward the
mechanism-positive baseline to find an intermediate point between `+0.032 dB`
surplus/unsafe tail and safe-tail/negative mean.

## 2026-06-11 Depth-Direct Tail-Lite Wide-Gate Scout Plan

Next queue keeps train=`invert`, `R0=0`, full eval matrix, and locked-test block.
It tests lighter guard settings that are closer to the baseline depthDirect loss
while still using wider encoder gates.

| Variant | Gate/Gamma/Beta | depth scale | dense budget | preserve/ref/tail | intent |
| --- | ---: | ---: | ---: | ---: | --- |
| `wg16_base_s008_b14` | `0.16/0.24/0.12` | `0.08` | `0.14` | `0.03/0.03/0.03` | isolate wider gate with baseline action budget |
| `wg18_base_s008_b14` | `0.18/0.28/0.14` | `0.08` | `0.14` | `0.03/0.03/0.03` | test a wider gate without added guard pressure |
| `wg16_lite_s006_b12` | `0.16/0.24/0.12` | `0.06` | `0.12` | `0.03/0.03/0.03` | lower action budget with baseline guard |
| `wg16_tail04_s008_b12` | `0.16/0.24/0.12` | `0.08` | `0.12` | `0.04/0.04/0.04` | mild guard between baseline and failed tailguard |

Continue only if a variant keeps mean true-vs-zero surplus near `+0.03 dB` while
reducing baseline depthDirect worst regressions materially below `75/600` and
not worsening SSIM.
