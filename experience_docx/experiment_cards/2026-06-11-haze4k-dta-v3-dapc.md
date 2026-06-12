# Haze4K ConvIR-B DTA-v3 DAPC Fine-Tune Route

Date: 2026-06-11

Status: `PLANNED_CTDG_SAFEMIX_AUDITS_LOCKED_TEST_BLOCKED`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: Innovation 1 / depth-guided transmission adapter.
- Route name: DTA-v3 DAPC, Depth-Attributed Preservation-Controlled Adapter.
- Branch: `codex/haze4k-dta-v3-dapc-finetune`.
- Anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`; this route imports DTA data/prior plumbing needed by the mechanism but does not fine-tune from a DTA-v2 checkpoint.
- Diagnostic predecessor: `github/codex/haze4k-dta-v2-calibrated` at `9e95408`, used only for mechanism diagnosis and evidence targets.
- Runtime host: `convir-4090`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Expected depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf` unless setup audit finds a different existing cache.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/`.
- Local policy: local WSL is for editing and compile/static checks only; no local tests, smoke tests, training, eval, inference, demos, or image generation.

## Hypothesis

DTA-v2 CalGate was a positive diagnostic but not a depth-attributed candidate:
zero/shuffle controls retained most of the gain, `normal` slightly beat calibrated
`invert`, SSIM was slightly negative, and tail regressions remained high. DTA-v3
DAPC tests whether separating a generic zero-depth residual `R0` from a frozen-R0
depth-specific physical correction `R_depth * M_depth` can increase true-vs-zero
surplus while reducing SSIM/tail risk.

## Architecture Change

`--arch dta_v3` adds `DTA.*` modules only:

- stage-2/stage-3 calibrated prior FiLM with wider but bounded gates;
- supervised `transmission_head` for `t_pred`;
- `r0_refine` generic zero-depth residual branch;
- `depth_mask_head` risk/preservation mask;
- physical correction blend using `J_phys = (I - A * (1 - t_pred)) / clamp(t_pred)` and a bounded mask.

The final output is:

```text
A0 output + R0(x) + M_depth(x, d, t_pred, conf) * bounded(J_phys - (A0 output + R0))
```

## Partial-Load And Initialization Contract

- Fine-tuning starts from official Haze4K A0, never from DTA-v2 weights.
- Use `--init_model_partial --partial_new_prefixes DTA.` for Phase A from A0.
- Official ConvIR-B keys must load with strict matching shapes.
- Missing keys are allowed only under `DTA.`.
- Unexpected checkpoint keys are fatal.
- New module init:
  - `DTA.stage2/3`: identity by zero last projection; gate bias defaults to `-5.0`, gate limit `0.10`, gamma limit `0.16`, beta limit `0.08`.
  - `DTA.transmission_head`: zero last projection.
  - `DTA.r0_refine`: zero last projection, so Phase A starts as exact no-op.
  - `DTA.depth_mask_head`: conservative bias `-4.0`, bounded budgets `0.04` easy / `0.12` dense.
- Phase B initializes from the same-arch Phase A route checkpoint using full strict load with `--init_model_allow_full_route`; `DTA.r0_refine` is frozen by train scope.

## Fine-Tune Ladder

| Stage | Trainable scope | Command intent | Continue only if |
| --- | --- | --- | --- |
| Stage 0 preflight | none / one-batch gradient probe | strict A0 partial load, no-op/bounded diff, depth/trans availability, finite losses | partial load clean, no locked test touched, DTA gradients finite |
| Phase A R0 | `dta_r0_only`, `dta_phase=r0`, `dta_ablation=r0_only`, `depth_mode=zero` | learn the generic zero-depth residual baseline safely | R0 is positive vs A0, SSIM non-negative or near zero, tail no worse than current DTA-v2 zero |
| Phase B depth surplus | `dta_depth_only`, `dta_phase=depth`, init from Phase A | freeze R0, train FiLM/trans/head/mask physical depth surplus | `invert/true` beats zero/shuffle/wrong orientation, tail and SSIM gates pass |
| Ablations | route-specific train scopes/modes | locate source of gains | output-refine-only, FiLM-only, trans-head-only, and phys-blend-only evidence is written |
| OOF/multi-seed | same fixed settings | verify stability | promotion gate passes on train-derived OOF only |

## Default DTA-v3 Fine-Tune Settings

- Gate: `gate_bias=-5.0`, `gate_limit=0.10`, `gamma_limit=0.16`, `beta_limit=0.08`, confidence floor `0.30`.
- Mask: easy budget `0.04`, dense budget `0.12`, density threshold `0.35`, mask bias `-4.0`, physical `t_min=0.10`.
- Loss: original multiscale L1 + `0.1 * FFT`, plus supervised transmission/physics/preserve losses when Haze4K `trans/A` are available.
- Added protection: optional A0 reference preserve and tail guard losses using frozen official ConvIR-B output.
- Depth controls: `invert` is the primary calibrated mode from DTA-v2 audit; `normal` is wrong-orientation control; `zero` and deterministic eval shuffle are mechanism controls.

## Required Intermediate Artifacts

| Artifact | Required purpose |
| --- | --- |
| `dta_v3_preflight.json/log` | partial-load, no-op/bounded diff, real batch gradients, finite losses |
| `depth_eval_pairing_audit.csv/json` | prove eval shuffle is deterministic and not batch-size-1 no-op |
| `train_eval_depth_matrix.json/csv` | separate training regularization from inference depth use |
| `r0_vs_rdepth_attribution.csv` | per-image generic residual vs depth surplus decomposition |
| `output_refine_only_ablation.json` | measure generic residual head contribution |
| `film_only_ablation.json` | measure stage2/stage3 depth FiLM contribution |
| `trans_head_only_no_rgb_residual.json` | prove transmission head can learn without RGB action |
| `phys_blend_only.json` | measure physical correction tail behavior |
| `t_to_image_coupling.csv/json` | correlate t quality with image delta |
| `tail_regression_contact_sheet/` | cloud PNG contact sheets for best wins and worst regressions; not committed to Git by default |
| `risk_router_calibration.json` | mask/gain/loss calibration if Phase B reaches selector analysis |
| `ssim_tail_report.json` | SSIM and regression distribution summary |

## Internal Promotion Gate

Locked Haze4K test is blocked unless all train-derived gates pass:

```text
mean_dPSNR(true - A0) >= +0.08
mean_dPSNR(true - zero/R0) >= +0.03
hard_dPSNR(true - zero/R0) >= +0.04
true - shuffle_eval_fixed_perm >= +0.03
true - normal_wrong_orientation >= +0.02
SSIM(true - A0) >= 0 or CI not significantly negative
worst regressions(true) <= zero/R0
strong regressions(true) <= zero/R0
positive_ratio >= 0.65
```

If the route only beats A0 but does not beat zero/shuffle/wrong-orientation, it
is a generic adapter and must not be called depth-guided.


## 2026-06-11 Phase A R0 OOF20 Fold0

Cloud run `oof20_phaseA_r0_seed3407_f0` used the fine-tune route from official
A0 with `train_scope=dta_r0_only`, `dta_phase=r0`, `dta_ablation=r0_only`,
`dta_depth_mode=zero`, and the wider-but-bounded DTA-v3 defaults. Training and
evaluation completed on `convir-4090`; locked Haze4K test remained untouched.

Evaluation against A0 on train-derived fold0 validation (`600` images):

| Metric | Value | Gate implication |
| --- | ---: | --- |
| mean dPSNR | `-0.012119` | fails positive R0 requirement |
| hard bottom-25 dPSNR | `-0.069025` | fails hard preservation/gain |
| easy top-25 dPSNR | `+0.044643` | easy improves, but not enough to offset hard/global loss |
| mean dSSIM | `-0.00001218` | slightly preservation-negative |
| positive ratio | `0.4500` | below continuation target |
| strong regressions (`<= -0.05 dB`) | `48/150` | too high for a safety baseline |
| worst regressions (`<= -0.20 dB`) | `70/600` | tail remains unsafe |

The post-eval `t_pred` audit failed with `KeyError: t_pred` because R0-only
intentionally has no active transmission/depth output. This is classified as a
script/audit applicability issue; the Phase A launcher now skips `t_pred` audit
for R0-only and reserves transmission coupling audits for Phase B/depth-active
runs.

Decision: `COMPLETED_GATE_FAIL_PHASE_A_R0_NO_PHASE_B`. Do not launch Phase B
from this R0 checkpoint. The next experiment should redesign Phase A as a
smaller or more conservative R0 diagnostic before any depth-attributed frozen-R0
training.


## 2026-06-11 Phase A Conservative R0 Scout Plan

After the original R0 OOF20 gate failure, the authorized next step is a parallel
R0-only scout queue, not Phase B. The queue keeps the architecture/fine-tune
route unchanged but reduces residual scale, LR, and/or increases A0 preserve and
tail-guard pressure:

| Variant | LR | R0 scale | preserve/ref/tail | Continue condition |
| --- | ---: | ---: | ---: | --- |
| `r0s005_lr3e5_ref005` | `3e-5` | `0.005` | `0.05/0.05/0.05` | no-op-safe, tail below failed R0 |
| `r0s010_lr3e5_ref005` | `3e-5` | `0.010` | `0.05/0.05/0.05` | positive mean without hard/tail loss |
| `r0s020_lr3e5_ref005` | `3e-5` | `0.020` | `0.05/0.05/0.05` | recover capacity while safer than scale `0.04` |
| `r0s010_lr1e5_ref010` | `1e-5` | `0.010` | `0.10/0.10/0.10` | strongest preservation baseline |

Run `scout5full` first on `convir-4090` using separate GPUs. Generate cloud-only
contact sheets for best wins and worst regressions. Promote only the best safe
variant to OOF20; do not launch Phase B until a safe R0 OOF20 baseline exists.

## Locked-Test Policy

Locked Haze4K test must not be used to select checkpoint, depth mode, gate,
loss, mask budget, train scope, or ablation. A locked test is allowed only once
for a fixed configuration that passes the internal OOF mechanism/preservation
gate. Current status: locked test blocked.

## Stop Rules

- Stop Phase A if R0 is not positive or worsens SSIM/tail beyond DTA-v2 zero.
- Stop Phase B if `invert` fails to beat zero/shuffle/wrong orientation by the written surplus gates.
- Stop ablations that are preservation-negative before OOF expansion.
- Do not continue `adapter_neighbors` from DTA-v2 unless a new Phase B gate passes first.

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
- Locked Haze4K test remains blocked. Phase A `dta_r0_only` completed and failed the R0 gate; Phase B is blocked until a safe R0 baseline is produced or the stop rule is explicitly overridden.


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

## 2026-06-11 Depth-Direct Tail-Lite Wide-Gate Scout Results

The tail-lite queue completed on `convir-4090` in workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-taillite`
from commit `023f226`. It kept train=`invert`, `R0=0`, wider gates, full
`invert/normal/zero/shuffle` eval matrices, and locked-test block.

| Variant | true mean | true hard | dSSIM | pos ratio | true-vs-zero | true-vs-shuffle | true-vs-normal | worst true | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wg18_base_s008_b14` | `+0.024404` | `+0.006360` | `-0.00002331` | `0.6050` | `+0.036631` | `+0.032141` | `+0.033084` | `76` | best mechanism row; tail/SSIM fail |
| `wg16_base_s008_b14` | `+0.020820` | `+0.002206` | `-0.00002448` | `0.6033` | `+0.035294` | `+0.031014` | `+0.031936` | `76` | mechanism-positive; tail fail |
| `wg16_lite_s006_b12` | `+0.019351` | `+0.001300` | `-0.00002463` | `0.5933` | `+0.035475` | `+0.031205` | `+0.032141` | `75` | mechanism-positive; tail fail |
| `wg16_tail04_s008_b12` | `+0.005806` | `-0.011557` | `-0.00002898` | `0.5633` | `+0.026700` | `+0.023430` | `+0.024054` | `70` | mild tail gain, surplus below gate |

This queue restores the depth mechanism after the over-constrained tailguard
run. `wg18_base_s008_b14` improves on the baseline depthDirect `invert` row in
mean (`+0.024404` vs `+0.013905`), hard (`+0.006360` vs `-0.005602`), and
true-vs-zero surplus (`+0.036631` vs `+0.032286`), and now clears the mean
true-vs-shuffle (`+0.032141`) and true-vs-normal (`+0.033084`) mechanism
thresholds. It still fails promotion because dSSIM is negative, positive ratio
is only `0.6050`, strong regressions are `39`, and worst regressions remain
`76/600` versus eval-zero `38/600`.

Decision: `COMPLETED_MECHANISM_POSITIVE_TAIL_FAIL_TAILLITE_WIDE_GATE`. No locked
test. The next evidence-supported step is a tail-aware variant centered on
`wg18_base_s008_b14`, adding only very mild tail pressure or post-hoc risk
selection; do not return to the strong guard settings that collapsed surplus.

## 2026-06-12 DTA-v3.1 WG18-RiskSelect-AConsistent Plan

User-approved follow-up keeps the fine-tune route and centers all work on
`wg18_base_s008_b14`. It does not revive R0 and does not touch locked Haze4K
test.

Implementation additions:

- eval-time DTA-v3 airlight mode: `fallback` (deployment proxy) or `gt`
  (oracle-A diagnostic) so train/eval airlight mismatch can be measured.
- `output_semantics_audit.json` to verify no-op DTA equivalence and confirm
  that DTA-v3 `refine_output()` receives a residual, so `out + hazy` is the
  correct physical base image.
- `airlight_train_eval_gap.csv` and `airlight_oracle_vs_pred_summary.json` to
  compare A0, DTA with fallback A, and DTA with Haze4K filename A.
- same-fold diagnostic risk selector artifacts:
  `risk_selector_oof_calibration_<run>.json`,
  `risk_selector_threshold_trace_<run>.csv`, and
  `per_image_delta_matrix_<run>_risk_selected.csv`.
- optional B4 light hinge training flags:
  `--dta_light_tail_hinge_weight` and `--dta_light_ssim_hinge_weight`, using the
  frozen A0 reference model with mild top-k MSE tail and SSIM no-worse-than-A0
  hinges.

Fold0 scout queue:

| ID | Candidate | Evidence |
| --- | --- | --- |
| B0 | existing `wg18_base_s008_b14` with fallback A | full depth matrix and contact sheets |
| B1 | existing `wg18_base_s008_b14` with GT/oracle A | oracle-A vs fallback A audit |
| B2 | B0 plus post-hoc risk selector | same-fold selector diagnostic only |
| B3 | B1 plus post-hoc risk selector | same-fold selector diagnostic only |
| B4 | new `wg18` light tail/SSIM hinge fine-tune | fold0 scout, then the same A/risk/depth matrix |

Scout gates remain unchanged: mean true-A0 at least `+0.020 dB`, hard true-A0 at
least `+0.010 dB`, true-vs-zero and true-vs-shuffle at least `+0.030 dB`,
true-vs-normal at least `+0.025 dB`, dSSIM no worse than `-0.000010`, positive
ratio at least `0.63`, and worst regressions at most `50/600`. If a fixed
selection passes, the next formal protocol is 5 folds x seeds `3407/3411/3413`
with eval depth `invert/zero/shuffle/normal`, A mode `fallback/gt`, selector
off/on, and locked test still blocked.

## 2026-06-12 DTA-v3.1 WG18-RiskSelect-AConsistent Results

The DTA-v3.1 fold0 queue completed on `convir-4090` from commit `b101196` in
workspace `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v31`.
It ran B0/B1/B2/B3 around the existing `wg18_base_s008_b14` checkpoint, trained
one B4 light tail/SSIM hinge scout, evaluated fallback-A and GT/oracle-A depth
matrices, and generated cloud-only contact sheets. Locked Haze4K test remained
untouched.

Pre-audits:

- `output_semantics_audit.json`: no-op DTA-v3 exactly matched A0
  (`max_abs_noop_diff=0.0`) and confirmed `DTA_refine_input_semantics=residual`,
  so the physical branch's `out + hazy` base formula is correct.
- `airlight_oracle_vs_pred_summary_v31_wg18_base_s008_b14_seed3407_f0.json`:
  fallback A was better than GT/oracle A on fold0 (`+0.024404 dB` vs
  `+0.020646 dB`), although GT A reduced worst regressions from `76` to `71`.
- `airlight_oracle_vs_pred_summary_v31_wg18_light_hinge_seed3407_f0_scout5full_post.json`:
  the same pattern held after B4 (`+0.025084 dB` fallback vs `+0.021298 dB` GT;
  worst `76` vs `72`).

| ID | mean | hard | dSSIM | pos ratio | true-vs-zero | worst | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 fallback-A `wg18` | `+0.024404` | `+0.006360` | `-0.00002331` | `0.6050` | `+0.036631` | `76` | mechanism kept; hard/SSIM/tail fail |
| B1 GT-A `wg18` | `+0.020646` | `+0.001292` | `-0.00002495` | `0.5983` | `+0.035193` | `71` | tail slightly better, quality worse |
| B2 fallback-A risk select | `+0.021610` | `+0.012073` | `+0.00000300` | `0.1800` | `+0.019600` | `9` | tail/SSIM fixed by low coverage; depth surplus fails |
| B3 GT-A risk select | `+0.020147` | `+0.011702` | `+0.00000293` | `0.1800` | `+0.018870` | `9` | tail/SSIM fixed by low coverage; depth surplus fails |
| B4 fallback-A light hinge | `+0.025084` | `+0.006701` | `-0.00002320` | `0.6033` | `+0.037015` | `76` | tiny mean/surplus gain; tail/SSIM still fail |
| B4 GT-A light hinge | `+0.021298` | `+0.001579` | `-0.00002485` | `0.5983` | `+0.035568` | `72` | same as B1 with slight gain |
| B4 fallback-A risk select | `+0.021976` | `+0.012167` | `+0.00000302` | `0.1800` | `+0.019838` | `9` | tail/SSIM fixed by low coverage; depth surplus fails |
| B4 GT-A risk select | `+0.020497` | `+0.011791` | `+0.00000295` | `0.1800` | `+0.019098` | `9` | tail/SSIM fixed by low coverage; depth surplus fails |

Decision: `COMPLETED_SCOUT_GATE_FAIL_LOCKED_TEST_BLOCKED`. The result is a
useful diagnostic but not an effective model improvement. Fallback A is not the
main tail cause; GT/oracle A does not improve mean or SSIM. The light hinge
slightly improves mean/surplus but does not move worst regressions. The
same-fold risk selector can make SSIM positive and reduce worst to `9/600`, but
only by selecting `25%` coverage and dropping true-vs-zero surplus below the
`+0.03 dB` mechanism gate. Therefore the planned 5-fold x 3-seed formal
validation is not launched from B0-B4; it remains blocked until a fixed scout
row passes the written fold0 gate.

## 2026-06-12 DTA-v3.2 CTDG-SafeMix No/Low-Training Audit Plan

Status: `PLANNED_CTDG_SAFEMIX_AUDITS_LOCKED_TEST_BLOCKED`.

The next user-approved route is DTA-v3.2 CTDG-SafeMix: Calibrated Transmission
+ Distributionally Safe Gated Mixing. It keeps the current DTA-v3.1 conclusion
intact: B0-B4 are not candidates, 5-fold x 3-seed validation remains blocked,
and locked Haze4K test remains blocked. The first step is no/low-training audit
only, centered on the two useful mechanism sources:

| Source | Checkpoint | Role |
| --- | --- | --- |
| `wg18_base_s008_b14` | tail-lite fold0 Final.pkl | best DTA-v3 depthDirect mechanism source |
| `wg18_light_hinge` | DTA-v3.1 B4 fold0 Final.pkl | mild hinge source; verify whether it changes action upper bound |

Required new artifacts:

- `oracle_action_upper_bound_by_coverage_<run>.json/csv`: image/patch/pixel
  oracle action upper bound at target coverages `25/40/60/80/100%`.
- `oracle_best_possible_contact_sheet_manifest_<run>.md`: text-only image list
  for later cloud PNG review; PNGs are not committed.
- `alpha_blend_sweep_matrix_<run>.json/csv`: alpha shrink sweep for
  `alpha=0.10/0.20/0.35/0.50/0.75/1.00` under fallback/GT A and
  `invert/zero/shuffle/normal` eval depth.
- `t_pred_vs_trans_gt_correlation_<run>.csv`,
  `t_error_to_regression_correlation_<run>.json`, and
  `transmission_bin_failure_report_<run>.json`: Haze4K GT transmission audit.
- `selector_metric_correction_report_<run>_<airlight>.json`: corrected selector
  metrics separating global coverage from selected conditional positive ratio.
- `nested_selector_smoke_f0_<run>_<airlight>.json`,
  `nested_selector_smoke_f0_thresholds_<run>_<airlight>.csv`, and
  `risk_coverage_curve_f0_nested_<run>_<airlight>.csv`: five internal fold0
  calibration splits to test same-fold threshold overfit.

Interpretation gates before any DTA-v3.2 training:

- If 60% coverage oracle cannot reach mean at least `+0.05 dB`, dSSIM at least
  `0`, worst at most `50/600`, and true-vs-zero at least `+0.03 dB`, then the
  direct physical delta is not good enough and DTA-v3.2 should pivot toward
  UDP-style multi-scale depth fusion instead of gate-only training.
- If alpha `0.35` or `0.50` preserves true-vs-zero at least `+0.03 dB` while
  cutting worst materially, the primary issue is action amplitude and C0/C1
  soft-gate experiments are justified.
- If t-error, low transmission, bright/low-texture bins, or large action stats
  correlate with regressions, then calibrated transmission + uncertainty gate is
  the correct C2/C3 path.
- If nested selector performance collapses relative to same-fold selection,
  selector-only post-hoc policies remain diagnostic-only.

No C0-C4 training is launched until these audits are parsed and the next row is
fixed in this card.

## 2026-06-12 DTA-v3.2 CTDG-SafeMix No/Low-Training Audit Results

Status: `COMPLETED_CTDG_AUDIT_C0_FAIL_C1_C3_JUSTIFIED_LOCKED_TEST_BLOCKED`.

The no/low-training queue completed on `convir-4090` in workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v32` from
commit `c0738dc`. It evaluated the two source checkpoints in parallel:
`wg18_base_s008_b14` and `wg18_light_hinge`. Locked Haze4K test remained
untouched.

Key result summary:

| Source | Audit | Main finding | Decision |
| --- | --- | --- | --- |
| `wg18_base_s008_b14` | alpha sweep | alpha `0.50` reduces worst to `23/600` but true-vs-zero falls to `+0.020439`; alpha `0.75` keeps mean `+0.020914` but true-vs-zero is only `+0.029073`, dSSIM is `-0.00001604`, and worst is `54/600` | C0 alpha-only fails |
| `wg18_light_hinge` | alpha sweep | alpha `0.50` gives true-vs-zero `+0.020706`, worst `23`; alpha `0.75` gives true-vs-zero `+0.029418`, dSSIM `-0.00001587`, worst `55` | C0 alpha-only fails |
| `wg18_base_s008_b14` | oracle 60% coverage | image/patch/pixel oracle all pass: image mean `+0.092568`, dSSIM `+0.00000963`, worst `0`, true-vs-zero `+0.056471`; patch/pixel are stronger | soft gate/SafeMix is justified |
| `wg18_light_hinge` | oracle 60% coverage | image mean `+0.093898`, dSSIM `+0.00000952`, worst `0`, true-vs-zero `+0.057479`; patch/pixel are stronger | soft gate/SafeMix is justified |
| both | t/transmission audit | log-t error correlation with dPSNR is weak (about `-0.066`), but low-t images are the failure concentration: `255` low-t images have mean about `-0.013 dB` and `45/76` worst regressions | use t/uncertainty as risk feature, not sole root cause |
| both | selector correction | same-fold selected subset has conditional positive ratio `0.72` and selected true-vs-zero about `+0.078..+0.079`, but global coverage is only `0.25` and global true-vs-zero remains about `+0.0196..+0.0198` | selector is diagnostic only |
| both | nested selector smoke | internal fold0 nested selection drops to global mean about `+0.0163..+0.0167`, hard near zero/negative, true-vs-zero about `+0.0196..+0.0199`, with one held-out split near zero mean | one-threshold selector is not deployable |

Artifacts:

- `dta_v3_2_ctdg_audit_summary.json/csv`.
- `alpha_blend_sweep_matrix_v32_ctdg_diag_wg18_base_s008_b14_seed3407_f0.json/csv`.
- `alpha_blend_sweep_matrix_v32_ctdg_diag_wg18_light_hinge_seed3407_f0.json/csv`.
- `oracle_action_upper_bound_by_coverage_v32_ctdg_diag_wg18_base_s008_b14_seed3407_f0.json/csv`.
- `oracle_action_upper_bound_by_coverage_v32_ctdg_diag_wg18_light_hinge_seed3407_f0.json/csv`.
- `t_pred_vs_trans_gt_correlation_*`, `t_error_to_regression_correlation_*`, and
  `transmission_bin_failure_report_*` for both source checkpoints.
- `selector_metric_correction_report_*` and `nested_selector_smoke_f0_*` for
  fallback/GT airlight.

Decision: do not launch 5-fold x 3-seed and do not touch locked test. C0
alpha-only is ruled out. The oracle gap shows that the existing depth action has
useful local direction but needs a learned soft pixel/image gate and safer mixing
rather than a global shrink or one-threshold selector. The next fixed scout row
should be C1/C3: start from `wg18_base_s008_b14`, disable R0, keep A0 frozen,
train a DTA-v3.2 SafeMix gate/residual with physical delta clipped and treated as
a hint, and use transmission/low-t/brightness/texture/action magnitude as risk
features. C2 transmission calibration should be included as an auxiliary risk
signal but not treated as the only rescue path.

## 2026-06-12 DTA-v3.2 SafeMix C1/C3 Scout Plan

Status: `PLANNED_SAFEMIX_SCOUT_CODE_READY_LOCKED_TEST_BLOCKED`.

The fixed scout after the CTDG no/low-training audit is DTA-v3.2 SafeMix. It
keeps the route as a fine-tune experiment from the existing fold0
`wg18_base_s008_b14` mechanism checkpoint, keeps A0 frozen, keeps `R0=0`, and
trains only newly added SafeMix modules unless the variant says otherwise. This
is still fold0 train-derived validation only; no 5-fold x 3-seed formal run and
no locked Haze4K test is allowed unless a fixed scout row passes the written
fold0 gate.

New module contract:

- `DTA.trans_uncertainty_head`: predicts low-resolution log transmission
  uncertainty for gating and optional NLL calibration.
- `DTA.safe_gate_head`: predicts a soft pixel gate over clipped depth action.
- `DTA.safe_residual_head`: predicts a bounded learned residual expert used by
  the full SafeMix variant.
- partial initialization from the old WG18 checkpoint allows missing keys only
  under `DTA.trans_uncertainty_head.`, `DTA.safe_residual_head.`, and
  `DTA.safe_gate_head.`; all pre-existing A0/DTA modules must load from the
  checkpoint.

Scout variants:

| ID | train scope | physics weight | learned weight | clip | LR | intent |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `C1 c1_gate` | `dta_safemix_gate_only` | `1.00` | `0.00` | `0.08` | `8e-5` | test whether a learned soft gate can safely select the existing clipped physical action |
| `C3 c3_full` | `dta_safemix_full` | `0.25` | `0.75` | `0.06` | `5e-5` | test safe learned residual plus clipped physics hint and transmission uncertainty |

Cloud scripts:

- `run_dta_v3_2_safemix_scout_convir4090.sh`: trains one variant, evaluates
  fallback/GT airlight under `invert/zero/shuffle/normal`, aggregates the depth
  matrix, writes a SafeMix scout summary, and generates cloud-only contact
  sheets.
- `launch_dta_v3_2_safemix_scouts_convir4090.sh`: launches `c1_gate` and
  `c3_full` in separate tmux sessions on separate GPUs.
- `summarize_haze4k_dta_v32_safemix_scouts.py`: summarizes available SafeMix
  fold0 matrices and applies the written scout gates.

Fold0 scout gate remains:

```text
mean true-A0 >= +0.020 dB
hard true-A0 >= +0.010 dB
true-vs-zero >= +0.030 dB
true-vs-shuffle >= +0.030 dB
true-vs-normal >= +0.025 dB
dSSIM >= -0.000010
positive_ratio >= 0.630
worst regressions <= 50/600
```

If neither C1 nor C3 passes this gate, the route is recorded as
`SCOUT_GATE_FAIL_LOCKED_TEST_BLOCKED` and no 5-fold x 3-seed validation is
launched. If one fixed fallback-A row passes, the next step is formal 5-fold x
3-seed nested validation with locked test still blocked until that formal gate
passes.

## 2026-06-12 DTA-v3.2 SafeMix C1/C3 Scout Results

Status: `SCOUT_GATE_FAIL_LOCKED_TEST_BLOCKED`.

The SafeMix scout queue completed on `convir-4090` from commit `931de83` in
workspace `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v32-safemix`.
It trained `c1_gate` and `c3_full` in parallel on fold0, evaluated fallback/GT
A under `invert/zero/shuffle/normal`, generated cloud-only contact sheets, and
kept locked Haze4K test untouched.

| Row | mean | hard | dSSIM | pos ratio | true-vs-zero | true-vs-shuffle | true-vs-normal | worst | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `c1_gate` fallback-A | `+0.028030` | `+0.005498` | `-0.00002350` | `0.5883` | `+0.030087` | `+0.026561` | `+0.027173` | `45` | fails hard, SSIM, pos, shuffle surplus |
| `c1_gate` GT-A | `+0.026010` | `+0.003591` | `-0.00002450` | `0.5950` | `+0.031785` | `+0.027784` | `+0.028384` | `46` | diagnostic only; same failure pattern |
| `c3_full` fallback-A | `+0.031636` | `+0.009309` | `-0.00002288` | `0.6133` | `+0.039813` | `+0.034174` | `+0.035193` | `48` | best SafeMix row; fails hard, SSIM, pos |
| `c3_full` GT-A | `+0.030135` | `+0.007771` | `-0.00002357` | `0.6017` | `+0.040193` | `+0.034531` | `+0.035574` | `48` | diagnostic only; GT-A still not a rescue |

Interpretation:

- SafeMix C3 is a real improvement over B4 on mean, depth-attributed surplus,
  and tail count: fallback-A worst drops from `76/600` to `48/600`, and
  true-vs-zero rises to `+0.039813 dB`.
- The written scout gate still fails because hard bottom-25 remains below
  `+0.010 dB`, dSSIM remains around `-2.3e-5`, and positive ratio remains below
  `0.630`.
- C1 gate-only proves that clipping/gating physical action can reduce worst
  count but is not enough; it loses true-vs-shuffle surplus and leaves hard/SSIM
  unsafe.
- GT/oracle A again does not rescue the row, so airlight remains a diagnostic
  feature rather than the main fix.

Decision: do not launch 5-fold x 3-seed formal validation and do not touch
locked Haze4K test. The route remains mechanism-positive but not candidate-ready.
The next route, if reopened, should build on C3 rather than C1: keep safe learned
residual, add explicit SSIM/texture-aware preservation or feature-space fusion,
and require the same depth-control gates before any formal validation.

## 2026-06-12 DTA-v3.3 RouterFusion-SafeMix++ Plan

Status: `PLANNED_ROUTERFUSION_TRiAGE_CODE_READY_LOCKED_TEST_BLOCKED`.

DTA-v3.2 C3 is mechanism-positive but fails hard, SSIM, and positive-ratio
scout gates. The next continuation is DTA-v3.3 RouterFusion-SafeMix++ rather
than formal 5-fold validation. It keeps the route as a fine-tune continuation
from the best C3 diagnostic checkpoint, keeps A0 frozen, keeps R0 disabled, and
adds only selection/structure-safety mechanisms needed to close the oracle
routing gap.

New module contract:

- `DTA.router_image_head`: image-level accept router over SafeMix risk/action
  features.
- `DTA.router_patch_head`: patch-level accept router, upsampled to the output
  grid and multiplied with the SafeMix pixel gate.
- Existing `DTA.safe_gate_head`, `DTA.safe_residual_head`,
  `DTA.trans_uncertainty_head`, and `DTA.transmission_head` remain the C3
  SafeMix foundation.
- Partial initialization from the C3 checkpoint allows missing keys only under
  `DTA.router_image_head.` and `DTA.router_patch_head.`; all C3 pre-existing
  modules must load from the checkpoint.

Training additions:

- `dta_routerfusion_router_only`, `dta_routerfusion_full`, and
  `dta_routerfusion_plus_film` train scopes.
- Low-phys/high-learned SafeMix variants that reduce direct physical RGB delta
  dominance while preserving depth-attributed learned residual action.
- SSIM-CVaR and group-tail losses comparing DTA output against the frozen A0
  reference only on top-q or high-risk regions, avoiding the previously failed
  strong global tailguard pattern.
- Optional counterfactual gate suppression for zero/wrong-orientation depth to
  discourage the router from re-learning a zero-depth residual path.

Fixed triage queue before any formal validation:

| ID | Variant | Intent |
| --- | --- | --- |
| `D1 d1_loss` | C3 + SSIM-CVaR/group tail losses | test whether loss-only structural protection closes the small C3 gate gap |
| `D2 d2_lowphys` | D1 + lower physical weight / higher learned residual / smaller clip | test whether residual structure and action amplitude are the main SSIM/tail issue |
| `D3 d3_router` | D2 + image/patch/pixel RouterFusion + counterfactual gate suppression | main candidate; test whether learned routing approaches the oracle coverage gap |

Low-cost triage protocol:

```text
folds: fold0, fold1
seeds: 3407, 3411
candidates: D1, D2, D3
locked Haze4K test: blocked
formal 5-fold x 3-seed: blocked until a fixed D-row passes triage
```

Triage gates:

```text
mean true-A0 >= +0.030 dB
hard true-A0 >= +0.010 dB
dSSIM >= -0.000010
positive_ratio >= 0.620
true-vs-zero >= +0.035 dB
true-vs-shuffle >= +0.030 dB
true-vs-normal >= +0.025 dB
worst regressions <= 48/600
no fold mean < +0.015 dB
no fold dSSIM < -0.000030
```

Required new artifacts:

- `dta_v3_3_routerfusion_triage_summary.json/csv` and
  `dta_v3_3_routerfusion_variant_summary.csv`.
- `train_eval_depth_matrix_v33_routerfusion_*_{fallback,gt}.json/csv` and
  `r0_vs_rdepth_attribution_v33_routerfusion_*_{fallback,gt}.csv`.
- `gate_oracle_gap_report_<run>.json/csv`, `action_failure_taxonomy_<run>.csv`,
  `router_metric_correction_report_<run>.json`, `risk_coverage_curve_<run>.csv`,
  `trans_uncertainty_calibration_<run>.json`,
  `t_pred_vs_gt_transmission_by_group_<run>.csv`, and
  `counterfactual_gate_matrix_<run>.csv` for the primary D3 diagnostic row.
- Cloud-only contact-sheet PNGs for worst regressions and best wins; text
  manifests are synced, images are not committed.

Decision rule: if no fixed fallback-A D-row passes triage, record
`TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED` and do not launch formal 5-fold x 3-seed
or locked test. If one D-row passes, the next step is a fresh formal validation
plan with fold-specific initialization rules; locked test remains blocked until
that formal gate passes.
