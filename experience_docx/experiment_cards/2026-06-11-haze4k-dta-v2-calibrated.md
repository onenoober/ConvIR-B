# Haze4K ConvIR-B DTA-v2 Calibrated Confidence-Gated Adapter

Date: 2026-06-11

Status: `IN_PROGRESS_ADAPTER_ONLY_FIVEFOLD_DONE_MULTI_SEED_NEXT`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: Innovation 1 / depth-guided transmission adapter.
- Branch: `codex/haze4k-dta-v2-calibrated`, starting from completed DTA low-gate evidence commit `04c356c`.
- Cloud runtime only: `convir-4090` under `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated`.
- Local policy: local WSL is restricted to code editing and compile/static checks; no local runtime tests, smoke, training, eval, inference, or demos.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/`.

## Reason To Reopen DTA

The completed low-gate DTA route was engineering-valid but not promotion-ready:
full gate20 was A0-level yet slightly negative and had no hard/far gain. This
route does not retune that same low-gate adapter. It implements the revised
analysis conclusion: first audit depth-vs-transmission calibration, then train a
confidence-gated, supervised-transmission DTA-v2 with mechanism controls.

## DTA-v2 Change

- Adds `--arch dta_v2` while preserving `--arch official_convir`, `--arch convir`, and the old `--arch dta` path.
- Extends the DTA prior to `[depth, calibrated t_proxy, -log(t_proxy), depth_grad_x, depth_grad_y, confidence]`.
- Adds confidence-gated bounded FiLM at stage-2/stage-3 and a zero-init decoder-side output residual refinement.
- Adds optional Haze4K `trans` loading with synchronized crop/flip and filename-derived airlight.
- Adds supervised transmission, physical hazy reconstruction, and easy/clear preservation auxiliary losses.
- Adds depth controls: `normal`, `invert`, `zero`, and `shuffle`.
- Adds DTA-specific gradient clipping and a gate ramp (`0.01 -> 0.03 -> 0.06`) for v2 runs.

## Initialization / Partial Load Contract

- Start from official Haze4K A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Use `--init_model_partial --partial_new_prefixes DTA.` for all DTA-v2 fine-tuning.
- Official ConvIR-B backbone, FAM, SCM, decoder, and convolution modules load from A0.
- New `DTA.*` parameters are the only missing partial-load keys.
- FiLM/output residual heads are zero-initialized so synthetic no-op equivalence must pass before training.

## Required Evidence Queue

| Stage | Required artifact | Purpose |
| --- | --- | --- |
| depth-transmission audit | `dta_depth_transmission_audit_summary.json/csv` | Verify depth orientation, alpha calibration, proxy error, low-texture/bright/dense-region risk. |
| OOF split generation | `dta_v2_haze4k_oof_splits_seed3407.json` | Keep route selection on train-derived splits, not locked test. |
| preflight | `dta_v2_preflight.json/log` | Partial load, no-op equivalence, real-batch finite loss, DTA gradients, trans/physics losses. |
| adapter-only DTA-v2 | train/eval logs and compare JSON/CSV | Test calibrated/confidence DTA with true depth. |
| depth controls | zero, shuffle, invert compare JSON/CSV | Verify any gain is depth-mechanism dependent. |
| adapter-neighbors | train/eval logs and compare JSON/CSV | Test whether neighboring FAM/Convs can consume the prior. |
| final confirmation | one fixed locked-test compare only if internal gates pass | Avoid repeated locked-test selection. |

## Default V2 Run Settings

- Train scope first: `adapter_only`; second: `adapter_neighbors` only after adapter-only/control evidence exists.
- Loss: original multiscale L1 + `0.1 * FFT` plus `rank=0.001`, `tv=0.0001`, `trans=0.02`, `phys=0.005`, `preserve=0.02`.
- Gate: `gate_bias=-6.0`, final `gate_limit=0.06`, `gamma_limit=0.12`, `beta_limit=0.06`, confidence floor `0.25`.
- Clip: DTA `0.05`, neighbors `0.005`, fallback global `0.001`.
- Depth controls: true/normal, zero, shuffle, invert.
- Seeds: start with `3407`; expand to multi-seed OOF when the first full mechanism pass is healthy.

## Locked-Test Policy

Locked Haze4K test must not be used to select checkpoint, gate, loss, depth mode,
or train scope. The queue starts with audit/preflight and train-derived OOF or
small diagnostic runs. A locked full test is allowed only once for a fixed config
that has already passed internal mechanism and preservation gates.

## Current State

- Code implementation, route scripts, card, and index were pushed at commit `2460b21`.
- convir-4090 setup/static py_compile passed at commit `2460b21`.
- OOF split generation passed with five `600`-image validation folds.
- DTA-v2 preflight passed: partial load `602` loaded / `25` missing all under `DTA.`, no-op max diff `0.0`, and real-batch DTA grad sum `0.66677364` with finite trans/physics losses.
- Depth-transmission audit passed with `4000` rows and `0` errors; it found the cached depth direction is reversed (`depth` vs `-log(t_gt)` median Spearman about `-0.93`, `1-depth` about `+0.93`).
- Primary calibrated-depth training must use `--dta_depth_mode invert`; `normal` becomes the wrong-orientation control, alongside `zero` and `shuffle` controls.
- Adapter-only fold0 scout5 and OOF20 controls have completed.
- Adapter-neighbors fold0 OOF20 controls have completed. They stayed slightly
  positive in mean dPSNR but lost easy/top preservation and SSIM, increased worst
  regressions versus adapter-only, and showed near-zero audited gate means.
- Current internal candidate remains `adapter_only`, not `adapter_neighbors`.
  Locked Haze4K test remains blocked; next queue is OOF expansion across
  additional train-derived folds and controls.
- Adapter-only folds `1-4` OOF20 controls completed with all train/eval/tpred
  jobs `rc=0`.
- Five-fold adapter-only aggregate completed on convir-4090 with bootstrap CI
  and Wilcoxon report. Locked Haze4K test remains blocked because depth
  attribution is positive but still not clean, and SSIM/tail regressions do not
  satisfy the promotion gate.


## 2026-06-11 Adapter-Only Fold0 Scout5 Controls

Four adapter-only scout5 jobs ran concurrently on convir-4090 GPUs 1-4 using
OOF `fold0_train` for training and the first `128` images from `fold0_val` for
comparison. All jobs completed train, A0 comparison, and post-run `t_pred`
quality audit.

| Depth mode | Role | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Spearman(depth,-log(t)) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | calibrated true-depth from audit | `+0.058864` | `+0.035617` | `+0.083542` | `-0.0000009` | `10` | `20` | `0.076015` | `0.922318` | `+0.898767` |
| `normal` | wrong raw orientation control | `+0.064201` | `+0.041953` | `+0.083473` | `-0.0000019` | `10` | `20` | `0.090006` | `0.918616` | `-0.898766` |
| `shuffle` | mismatched-depth control | `+0.035514` | `-0.013165` | `+0.099094` | `+0.0000336` | `10` | `18` | `0.084645` | `0.919693` | `-0.415015` |
| `zero` | no-depth control | `+0.024335` | `-0.028063` | `+0.080075` | `+0.0000468` | `12` | `17` | `0.079062` | `0.921572` | n/a |

Interpretation: the route is trainable and positive on this small internal
diagnostic, but mechanism attribution is not clean yet because the wrong raw
orientation is slightly higher than calibrated `invert` on 128 images. Zero and
shuffle controls improve mostly easy samples while hurting hard bottom-25,
which keeps depth-mechanism evidence open for the 20-epoch/full-fold run.


## 2026-06-11 Adapter-Only Fold0 OOF20 Controls

Four adapter-only OOF20 jobs ran concurrently on fold0 train/val. Evaluation used
all `600` images from `fold0_val`, followed by full-fold `t_pred` quality audits.

| Depth mode | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate mean | Stage3 gate mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `+0.106894` | `+0.099160` | `+0.091081` | `-0.0000075` | `56` | `102` | `0.077984` | `0.921685` | `0.013002` | `0.050007` |
| `normal` | `+0.106010` | `+0.104724` | `+0.087849` | `-0.0000045` | `56` | `98` | `0.088812` | `0.920188` | `0.005675` | `0.052407` |
| `shuffle` | `+0.098391` | `+0.095590` | `+0.084815` | `+0.0000089` | `55` | `90` | `0.084428` | `0.921052` | `0.011980` | `0.053491` |
| `zero` | `+0.095529` | `+0.091814` | `+0.085666` | `+0.0000107` | `52` | `88` | `0.079434` | `0.922128` | `0.013502` | `0.055354` |

Interpretation: adapter-only DTA-v2 is positive on fold0 OOF20 for all four
modes, with calibrated `invert` barely best on mean dPSNR and `normal` best on
hard bottom-25. The small spread versus zero/shuffle means image-quality gains
cannot yet be attributed solely to correct depth; however, the full-fold result
is strong enough to continue the predeclared adapter-neighbors experiment.


## 2026-06-11 Adapter-Neighbors Fold0 OOF20 Controls

Four adapter-neighbors OOF20 jobs ran concurrently on fold0 train/val. The
scope released `DTA.*` plus neighboring FAM/Conv parameters using the
predeclared DTA and neighbor clip settings. All jobs completed training,
full-fold comparison, and `t_pred` quality audit on convir-4090.

| Depth mode | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate mean | Stage3 gate mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `+0.015092` | `+0.009731` | `-0.063870` | `-0.0001458` | `54` | `144` | `0.077901` | `0.921738` | `0.000074` | `0.000072` |
| `normal` | `+0.015129` | `+0.008829` | `-0.062361` | `-0.0001398` | `56` | `142` | `0.088921` | `0.920107` | `0.000074` | `0.000072` |
| `shuffle` | `+0.009656` | `+0.003892` | `-0.072763` | `-0.0001245` | `53` | `145` | `0.084493` | `0.921071` | `0.000072` | `0.000070` |
| `zero` | `+0.007218` | `+0.001740` | `-0.074593` | `-0.0001274` | `53` | `146` | `0.079408` | `0.922233` | `0.000074` | `0.000074` |

Interpretation: adapter-neighbors is not a promotion candidate in this route.
It gives only `+0.0072` to `+0.0151 dB` mean gain, makes easy/top samples
negative by roughly `-0.06` to `-0.075 dB`, has negative SSIM, and raises worst
regressions to `142-146` versus `88-102` for adapter-only. The near-zero audited
gates also show that unfreezing neighbors did not improve consumption of the
depth/transmission prior. Continue with adapter-only OOF expansion and controls;
do not use locked test for this decision.


## 2026-06-11 Adapter-Only Fold1-2 OOF20 Controls

Folds `1-2` completed after the adapter-neighbors diagnostic. These results
continue to favor adapter-only over adapter-neighbors, but they also keep the
mechanism attribution question open because the zero/shuffle controls remain
close to the true-depth modes.

| Fold | Depth mode | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate mean | Stage3 gate mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f1` | `invert` | `+0.069177` | `+0.083142` | `+0.032640` | `-0.0000228` | `62` | `110` | `0.075932` | `0.928746` | `0.013875` | `0.050555` |
| `f1` | `normal` | `+0.069591` | `+0.089662` | `+0.030042` | `-0.0000217` | `61` | `108` | `0.086846` | `0.927034` | `0.008267` | `0.052497` |
| `f1` | `shuffle` | `+0.057955` | `+0.076627` | `+0.032887` | `-0.0000097` | `58` | `94` | `0.082336` | `0.928117` | `0.014126` | `0.053669` |
| `f1` | `zero` | `+0.054809` | `+0.073010` | `+0.031669` | `-0.0000093` | `57` | `92` | `0.077457` | `0.929085` | `0.016877` | `0.054727` |
| `f2` | `invert` | `+0.091881` | `+0.034875` | `+0.097953` | `-0.0000408` | `52` | `96` | `0.077050` | `0.929004` | `0.017288` | `0.049172` |
| `f2` | `normal` | `+0.090507` | `+0.040584` | `+0.095801` | `-0.0000411` | `48` | `91` | `0.087146` | `0.928178` | `0.016460` | `0.052335` |
| `f2` | `shuffle` | `+0.075840` | `+0.024227` | `+0.085528` | `-0.0000207` | `51` | `93` | `0.082814` | `0.928715` | `0.018315` | `0.053236` |
| `f2` | `zero` | `+0.070339` | `+0.020007` | `+0.082121` | `-0.0000189` | `47` | `92` | `0.078161` | `0.929554` | `0.020948` | `0.052771` |

Fold0-2 average: `invert` has the highest mean dPSNR (`+0.089317`), `normal`
has the highest hard bottom-25 average (`+0.078323`), and zero/shuffle controls
are still close (`+0.073559/+0.077395` mean dPSNR). Continue to five-fold OOF
before any locked-test or promotion decision.


## 2026-06-11 Adapter-Only Five-Fold OOF20 Aggregate

Folds `3-4` completed after folds `1-2`, then the aggregate audit ran on
convir-4090. The aggregate covers all `3000` train-derived validation images
from folds `0-4`; all rows below have Wilcoxon signed-rank `p_two_sided=0.0`
under the repository's normal-approximation report.

| Depth mode | Mean dPSNR | 95% bootstrap CI | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate | Stage3 gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `+0.088230` | `[+0.076466, +0.099691]` | `+0.070046` | `+0.071636` | `-0.0000253` | `275/750` | `479/3000` | `0.077442` | `0.925764` | `0.013341` | `0.050323` |
| `normal` | `+0.087934` | `[+0.076519, +0.099003]` | `+0.076317` | `+0.069672` | `-0.0000240` | `271/750` | `463/3000` | `0.088295` | `0.924241` | `0.008749` | `0.052278` |
| `shuffle` | `+0.076486` | `[+0.066115, +0.086681]` | `+0.063289` | `+0.064852` | `-0.0000106` | `270/750` | `425/3000` | `0.083860` | `0.925129` | `0.013612` | `0.053514` |
| `zero` | `+0.072830` | `[+0.062554, +0.082707]` | `+0.059540` | `+0.063010` | `-0.0000097` | `261/750` | `421/3000` | `0.078924` | `0.926136` | `0.015538` | `0.054852` |

Interpretation: DTA-v2 adapter-only is a real positive internal OOF signal, but
it is not yet a clean depth-mechanism or promotion result. `invert` and `normal`
are essentially tied in mean PSNR, `normal` is higher on hard bottom-25,
zero/shuffle controls retain most of the gain, SSIM is slightly negative for
all modes, and tail regressions remain high. Proceed to multi-seed adapter-only
controls; locked Haze4K test remains blocked.


## Next Internal Queue

- Run the predeclared multi-seed adapter-only controls for `invert`, `normal`,
  `shuffle`, and `zero` before any locked-test confirmation.
- Prefer train-derived OOF folds; if runtime pressure requires staging, run two
  additional seeds in the same five-fold schedule and aggregate across seeds.
- Only consider locked Haze4K test if multi-seed OOF preserves mean/hard gains,
  reduces ambiguity against zero/shuffle controls, and resolves the SSIM/tail
  risk gates.
