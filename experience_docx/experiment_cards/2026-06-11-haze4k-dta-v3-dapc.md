# Haze4K ConvIR-B DTA-v3 DAPC Fine-Tune Route

Date: 2026-06-11

Status: `COMPLETED_MECHANISM_POSITIVE_TAIL_FAIL_TAILLITE_WIDE_GATE`

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
