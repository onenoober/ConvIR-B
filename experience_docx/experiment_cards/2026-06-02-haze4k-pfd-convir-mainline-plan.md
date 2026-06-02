# Haze4K PFD-ConvIR Fast-Track Plan

Date: 2026-06-02

Status: completed gated stop20 scout; B1 failed preservation gate on 2026-06-02.

## Goal

Get a credible PFD-ConvIR result with the fewest useful runs.

The active path is:

```text
A1 -> B1 RHFD -> B2 RHFD + HSCM-Lite -> B3 RHFD + HSCM-Lite + PFFB-Low
   -> optional B4 teacher preservation only if easy cases regress
```

This keeps the three publishable structure changes, but avoids early grids,
extra branches, high-risk physical inversion, high-frequency paths, and default
multi-seed repetition.

## Current Evidence Boundary

No current Haze4K route is promotion-ready.

Facts that constrain this plan:

- A0 exists: official ConvIR-B Haze4K pretrained evaluation was reproduced at
  `34.14 dB PSNR / 0.98971 SSIM`.
- The Haze4K train split has a recorded count mismatch: `3001` hazy files and
  `3000` GT files. Pair audit is mandatory before training.
- A1 does not yet exist in the required form: official ConvIR-B checkpoint plus
  same fine-tuning schedule.
- Existing FAM/FAM2 evidence is diagnostic only. FAM2-only improved hard
  bottom-25% by `+0.8159 dB`, but easy top-25% fell `-0.2860 dB` and
  strong-reference regressions reached `138/250`.
- Stop20 is noisy: original stop20 mean PSNR sample std was `0.2206 dB`, hard
  bucket std was `0.4551 dB`, and easy bucket std was `0.3240 dB`.

Conclusion: hard samples can be moved, but easy/strong-reference preservation is
the core risk. Stop20 should reject bad routes quickly, not promote routes by
itself.

## Non-Negotiables

Keep these. They are the minimum scientific safeguards.

1. **A1 same-finetuning control**
   - Use official Haze4K ConvIR-B checkpoint through `--init_model`.
   - Do not treat optimizer `--resume` as pretrained initialization.
   - Reuse A1 by budget/seed/schedule. Example: one `A1_stop20_seed3407` can
     compare B1, B2, and B3 if all use the same seed and stop20 schedule.

2. **Haze4K pair audit**
   - Run once per dataset root plus loader mapping rule.
   - Check hazy/GT counts, missing labels, orphan GT files, duplicate target
     mappings, duplicate stems, and crop-size compatibility.
   - If dataset/root/loader do not change, do not repeat this per PFD version.

3. **`--init_model` and safe wrapper**
   - Add `--init_model` for model weights only.
   - Keep `--resume` for optimizer-state resume only.
   - Add `--arch {convir,pfd}` and PFD flags.
   - Put PFD code in new files:

```text
Dehazing/ITS/models/
  ConvIR.py
  PFDConvIR.py
  pfd_modules.py
  haze_priors.py
```

4. **Zero-init equivalence**
   - Use one no-op mechanism: zero-initialized projection or zero convolution.
   - Do not combine zero conv with an extra learned zero multiplier.
   - PFD all-flags-off or zero-init output must match ConvIR-B at FP32
     `max_abs_diff < 1e-6`.

## One Branch, Flag-Controlled Versions

Use one active branch/worktree:

```text
codex/haze4k-pfd-mainline
```

Use flags and experiment directories to separate stages:

```text
--arch pfd
--pfd_rhfd 0|1
--pfd_hscm 0|1
--pfd_pffb 0|1
--pfd_pffb_high 0
--pfd_teacher 0|1
```

Optional tags or commits:

```text
pfd-v0-wrapper
pfd-v1-rhfd
pfd-v2-hscm-lite
pfd-v3-pffb-low
```

Avoid separate V0/V1/V2/V3 branches unless the code becomes impossible to
audit in one branch.

## Stage 0: Engineering Preflight

No training.

Run once:

| ID | Item | Pass line |
| --- | --- | --- |
| E1 | Haze4K pair audit | no fatal missing GT or duplicate mapping |
| E2 | `--init_model` load | original ConvIR-B loads official checkpoint with no missing original keys |
| E3 | PFD wrapper equivalence | checkpoint keys, random tensor equivalence, real Haze4K batch equivalence all pass |

One artifact is enough:

```text
preflight_pfd_v0.json
```

It should contain:

- pair audit summary or path;
- checkpoint load missing/unexpected keys;
- random tensor output diffs;
- real Haze4K batch output diffs;
- output shape list;
- added parameter count.

Stop if any E1/E2/E3 item fails.

## Stage 1: B1 RHFD

Run only after Stage 0 passes and training is explicitly approved.

Matrix:

| ID | Model | Purpose |
| --- | --- | --- |
| A1_stop20 | ConvIR-B official pretrained + same fine-tuning | matched control |
| B1_stop20 | PFD + RHFD | first structure test |

Command shape:

```text
B1: --arch pfd --pfd_rhfd 1 --pfd_hscm 0 --pfd_pffb 0 --pfd_teacher 0
```

Disabled:

- haze priors;
- physical `t/A/v/j`;
- PFFB;
- teacher;
- loss changes.

Stop20 gate:

- global PSNR `>= A1 - 0.05 dB`;
- easy top-25% mean delta `>= -0.05 dB`;
- warning strong-reference regressions `<= 50/250`;
- severe all-image regressions clearly below the FAM2 failure pattern
  (`444/1000`);
- no concentrated visual artifact pattern.

If B1 fails, stop PFD structure expansion and fix RHFD. Do not start HSCM or
PFFB.

## Stage 2: B2 RHFD + HSCM-Lite

Run only if B1 passes.

Matrix:

| ID | Model | Purpose |
| --- | --- | --- |
| B1_stop20 | PFD + RHFD | direct predecessor |
| B2_stop20 | PFD + RHFD + HSCM-Lite | haze-aware SCM test |

Command shape:

```text
B2: --arch pfd --pfd_rhfd 1 --pfd_hscm 1 --pfd_pffb 0 --pfd_teacher 0
```

HSCM-Lite only:

- dark channel;
- saturation;
- local contrast or gradient;
- confidence gate for prior reliability;
- zero residual projection;
- keep `self.main` for original SCM checkpoint compatibility.

Explicitly disabled:

- transmission `t`;
- airlight `A`;
- veiling light `v`;
- clean estimate `j = (x - v) / t`;
- hard auxiliary loss;
- teacher.

Stop20 gate:

- global PSNR `>= B1 - 0.03 dB`;
- hard bottom-25% vs B1 has positive trend;
- easy top-25% mean delta `>= -0.05 dB`;
- warning strong-reference regressions `<= B1`;
- confidence and prior branch activity are finite and non-collapsed.

One allowed rescue:

- if B2 moves hard but mildly hurts easy, try one cheaper HSCM-Lite rescue:
  halve branch width or initialize confidence bias more conservatively.
- Do not tune repeatedly.
- Do not add physical `t/A/v/j` in the first execution cycle.

## Stage 3: B3 RHFD + HSCM-Lite + PFFB-Low

Run only if B2 passes or is clearly the best stable candidate needing the third
structure for a paper-ready model.

Matrix:

| ID | Model | Purpose |
| --- | --- | --- |
| B2_stop20 | PFD + RHFD + HSCM-Lite | direct predecessor |
| B3_stop20 | PFD + RHFD + HSCM-Lite + PFFB-Low | three-module PFD test |

Command shape:

```text
B3: --arch pfd --pfd_rhfd 1 --pfd_hscm 1 --pfd_pffb 1 --pfd_pffb_high 0 --pfd_teacher 0
```

PFFB-Low only:

- keep original `self.merge`;
- fixed low-pass or depthwise blur;
- low-frequency residual correction;
- high path disabled;
- zero-initialized delta projection;
- no extra FFT/FFL loss.

Stop20 gate, relaxed for single-seed noise:

- global PSNR `>= B2 - 0.03 dB`;
- hard bottom-25% `>= B2`;
- easy top-25% drop `< 0.03 dB`;
- warning strong-reference regressions do not increase;
- no halo, ringing, color cast, or over-sharpening cluster.

One allowed rescue:

```text
B3r: PFFB-Low-Narrow
  low path channel half
  optional constant delta scale 0.1
  high path still off
```

If B3r still fails, stop PFFB. The temporary final structure becomes:

```text
PFD-ConvIR-Lite = RHFD + HSCM-Lite
```

PFFB-Low remains an extension ablation, not the main model.

## Stage 4: Optional B4 Teacher Preservation

Teacher is conditional, not a default experiment.

Trigger:

- B2 or B3 improves hard bottom-25%;
- and easy top-25% mean delta `< -0.05 dB`, or warning strong-reference
  regressions increase.

Skip teacher if:

- easy preservation already passes;
- hard bucket does not improve;
- the structure collapses globally.

Apply teacher to the best structural candidate only:

- if B3 is stable, use B3;
- if B3 is unstable but B2 is stable, use B2.

Conservative first loss:

```text
lambda_h = 0.2
lambda_t = 0.1
extra hard FFT weight = 0.05
```

Training weights:

- precompute teacher hard/easy weights from the train split only;
- do not use test GT to generate training weights;
- test split is for bucket analysis only.

B4 gate:

- hard gain is retained;
- easy mean and median delta move toward zero;
- warning strong-reference regressions decrease;
- global PSNR does not fall below the structural predecessor by more than
  `0.03 dB`;
- loss terms stay finite and do not dominate restore loss.

## Deferred Work

Do not run these during discovery:

- V2b physical HSCM with `t/A/v/j`;
- PFFB-High;
- full FFL;
- S1 HSCM-only;
- P1 PFFB-only;
- RP1 RHFD+PFFB;
- default 3-seed stop20 for B1/B2/B3;
- default teacher after B3;
- paper-grade full evidence package for every failed stop20.

These are paper-stage or later-stage additions after one primary candidate
survives.

## Compute Plan

Fastest initial training count:

| Step | Run count | Notes |
| --- | ---: | --- |
| E1/E2/E3 | 0 training | mandatory engineering preflight |
| A1_stop20 | 1 | reused for B1/B2/B3 if seed/schedule match |
| B1_stop20 | 1 | RHFD |
| B2_stop20 | 1 | RHFD + HSCM-Lite |
| B3_stop20 | 1 | RHFD + HSCM-Lite + PFFB-Low |
| B4_stop20 | 0 or 1 | only if triggered |

Promotion:

- pick one best candidate from B1/B2/B3/B4;
- run either repeat stop20 or 80-epoch promotion, not both by default;
- prefer 80-epoch if compute allows and stop20 gain is near the noise boundary;
- run A1_80 only if no matched A1 at that budget exists;
- run full training only for the single best candidate;
- run A1_full only if reporting a full-budget replacement claim.

If compute is tight:

```text
A1_stop20 + B1 + B2 + B3 + A1_full + best_full
```

This is weaker scientifically than adding 80-epoch, but much faster.

## Outcome

- Stage 0 passed on `autodl-dehaze3`.
- `A1_stop20` completed.
- `B1_stop20` completed and failed the preservation gate.
- `B2_stop20` and `B3_stop20` were not launched.
- The route remains diagnostic only.

## Fixed Diagnostic Sidecar Rule

The existing A1/B1 logs were run with `save_image=False`; therefore the metric
gate is not enough for a final report or closure package. Backfill the fixed
diagnostic pack from the saved checkpoints, per-image CSV, and bucket JSON
before treating the visual/artifact question as answered.

For all future PFD stages, `compare_and_gate` must start a background
diagnostic sidecar immediately after per-image CSV and bucket JSON are written.
The sidecar must not interrupt training and must not dump all validation
images. It should use CPU by default when another training job may need the GPU,
or an explicitly chosen idle GPU when speed matters.

Required sidecar outputs:

- one 20-row visual panel with hazy input, GT, baseline output, candidate
  output, candidate-baseline difference, baseline error, candidate error,
  input-baseline residual, and input-candidate residual;
- per-sample multi-scale outputs: baseline and candidate `1/4`, `1/2`, and
  full predictions;
- output safety CSV with prediction range, out-of-range ratios, mean/std, RGB
  shift, luma shift, and saturation ratio;
- PFD branch activity CSV and bucket/category JSON for RHFD/HSCM/PFFB modules;
- visual notes template completed by human review.

Automatic gates may proceed while the sidecar runs in the background, but a
candidate report, route closure, or manual approval for the next structure
stage is blocked until `diagnostic_summary.json`, `visual_panel_20.png`,
`output_safety_stats.csv`, `pfd_branch_stats.csv`, and visual notes exist.

## Metrics By Stage

Stop20 screen requires only:

- global PSNR/SSIM;
- hard/medium/easy PSNR deltas;
- easy top-25% mean and median delta;
- warning strong-reference regressions;
- severe all-image regressions;
- one fixed 20-image visual panel;
- branch activity summary for active modules;
- multi-scale output panel for fixed samples;
- output safety statistics for fixed samples;
- RHFD activity by bucket/category when `--pfd_rhfd 1`;
- latency/memory only for A1 and the current best candidate unless cost looks
  risky.

Do not create a full paper package for every failed screen.

Record once unless the structure changes:

- pair audit;
- parameter count;
- FLOPs estimate;
- V0 equivalence;
- fixed visual panel definition.

Full candidate report requires the complete metric set:

- per-image CSV;
- bucket JSON;
- Best-vs-Last stability;
- latency and peak memory;
- params and FLOPs;
- visual notes;
- final decision label.

## Final Full-Training Target

Use these only for 80/full decisions, not for stop20 rejection unless the route
is clearly bad.

- global PSNR `>= A1 + 0.10 dB`;
- SSIM delta `>= -0.001`;
- hard bottom-25% `>= A1 + 0.20 dB`;
- easy top-25% mean and median delta `>= -0.05 dB`;
- warning strong-reference regressions `<= 25/250`;
- severe all-image regressions far below FAM2-only `444/1000`;
- cost limits pass, or the result is labeled as an ablation.

## Immediate Local Steps

No training should be started from this document.

Next local actions:

1. Create clean `main`-based worktree `codex/haze4k-pfd-mainline`.
2. Add or write a compact V0 route card.
3. Implement or run the Haze4K pair audit.
4. Add `--init_model`.
5. Add PFD wrapper and consolidated `preflight_pfd_v0.json`.
6. Run local compile/equivalence checks only.
7. Ask for explicit approval before any cloud real-batch preflight or training.
