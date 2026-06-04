# APDR-v0.3 Residual and Correctability Diagnostics

Date: 2026-06-03

Status: completed diagnostic, not promotion.

## Scope

- Project: ConvIR-B Haze4K.
- Model family: APDR v0.2RC action mask plus residual-source, learnability, crop-protocol, and correctability diagnostics.
- Dataset or task: Haze4K train/test diagnostics.
- Primary objective: decide whether APDR-v0.3 should proceed through direct SHED replay, the current residual head, a low-frequency/color residual expression, cached full-mask crop training, or CorrectabilityOpen routing.
- Main metric: PSNR delta versus official ConvIR-B anchor.
- Secondary metrics: SSIM delta, hard/easy bucket deltas, strong-reference regressions, severe regressions, raw expert versus safe replay delta.
- Execution environment: AutoDL `autodl-dehaze3`.
- Artifact roots:
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_shed_replay_20260603/`
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_delta_learnability_20260603/`
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_residual_source_oracle_20260603/`
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_low_color_learnability_20260603/`
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_crop_mask_mismatch_20260603/`
  - `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics/experience_docx/experiment_logs/haze4k_apdr_v0_3_correctability_proxy_20260603/`
- Branch or isolated workspace: `codex/haze4k-apdr-v0-3-shed-diagnostics`, `/home/ubuntu/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics`.
- Review package location: pending.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0.2RC O1 shows large safe-mask oracle headroom, while three residual stop20 runs recovered almost none of it. A no-training replay isolates whether the missing piece is residual expression or the mask/expert interface itself.
- Target failure or opportunity: preserve v0.2RC easy/strong-reference safety while borrowing hard-positive deltas from FAM2, HardFreq, PFD, and haze-prior SCM routes.
- Cheap preflight evidence: existing v0.2RC O1 oracle `hard +0.832 dB`, easy `+0.0085 dB`, strong/severe regressions `0`; historical experts have hard-positive deltas but failed preservation.
- Earliest decisive gate: full Haze4K safe replay for each expert, no training.
- Expected cost or attempt-count saving: one inference pass per expert avoids another 20-epoch residual scout before proving expert deltas are usable.
- What success decides: promote APDR-v0.3 SHED teacher or safe-expert route design.
- What failure decides: do not build v0.3 around historical expert deltas; prioritize delta learnability and residual-source ablation.
- Why a cheaper diagnostic is not enough: per-image PSNR CSVs alone do not contain pixel-level expert deltas, so they cannot test `J0 + M_safe * clamp(J_expert - J0)`.

## Hypothesis

- Observed failure: v0.2RC residual stop20 variants produced only noise-level gains despite high oracle headroom.
- Target mechanism: hard-positive expert residuals can provide a more learnable or reusable delta source than the tiny frozen-context residual head.
- Primary variable: residual source, with the v0.2RC safe action mask held fixed.

Mechanism sentence:

```text
If we replace the learned APDR residual with a bounded historical expert delta
under the same M_safe mask, hard-bucket PSNR should improve while easy and
strong-reference regressions remain controlled because preservation is supplied
by the APDR action mask rather than by the expert itself.
```

## Change

- Code branch: `codex/haze4k-apdr-v0-3-shed-diagnostics`.
- Exact code/config change: add a replay-only evaluator that computes `J_safe = J0 + M_safe * clamp(J_expert - J0, -r, r)` using the APDR-v0.2RC selector checkpoint.
- Enabled mechanisms: APDR v0.2RC full-image safe mask, bounded expert delta replay.
- Explicitly disabled mechanisms: training, gradient updates, checkpoint mutation, new depth/diffusion priors.
- Parameter/runtime/memory impact expected: no trained parameter change; inference loads APDR anchor/mask and one expert.
- Initialization or no-op behavior: anchor `J0` is read from the APDR full-scale adapter and should remain ConvIR-B equivalent.
- Resume policy: rerun per expert if a JSON/CSV output is missing; do not overwrite successful artifacts unless the script changes.
- Defaults changed: none for training.
- Defaults intentionally preserved: Haze4K test split, official A0 checkpoint, v0.2RC selector calibration.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| safe replay | mean PSNR delta `>= +0.020`, hard bottom-25% `>= +0.080`, easy top-25% `>= -0.010`, strong-reference regressions `<= 30/250`, severe regressions `<= 10/1000`, SSIM delta `>= 0` | safe replay must outperform raw failed route on preservation while retaining a hard-positive signal | if no expert passes, stop SHED as a direct replay route and use results only as teacher-source diagnostics |
| current residual learnability | 32-image overfit loss drop `>= 0.30`, recovery `>= 0.50`, corr `>= 0.40`, hard gain `>= +0.30 dB` | current residual branch must be able to memorize masked `Delta_star` before any longer train | if fail, stop current residual expression |
| residual source oracle | low/color source should retain most full oracle hard gain with no regressions | decide whether residual target is low-frequency/color or high-frequency/detail dominated | if low/color passes, prioritize veil/color expression over detail |
| crop mask protocol | crop recompute vs full-mask patch mean corr `>= 0.80`, p10 corr `>= 0.60`, mean abs diff `<= 0.020` | residual crop training must not break full-image selector calibration | if fail, require full-image `M_safe` cache plus crop patching |
| correctability proxy | held-out AUC `>= 0.80`, Spearman `>= 0.45`, easy score `<= 0.10`, positive-hard score `>= 0.50` | P_benefit must be predictable from deployable features before model integration | if ranking passes but calibration fails, keep as thresholded/calibrated preflight only |

## Analysis Plan

- Run safe replay for FAM2-only, FAM2 confidence-gated, HardFreq, PFD B1, and haze-prior SCM when checkpoints are available on AutoDL.
- Preserve compare JSON, bucket JSON, per-image CSV, gate JSON, and run log per expert.
- Do not sync checkpoints, model weights, image outputs, datasets, NumPy arrays, or raw inference images to GitHub.

## Results

### Safe Expert Replay

All direct SHED replay candidates failed the APDR stop20-style gate.

| Expert | Safe mean | Safe hard bottom-25% | Safe easy top-25% | Strong regressions | Severe regressions | Raw expert mean vs A0 anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAM2-only | `-0.1259` | `-0.1758` | `-0.0003` | `1/250` | `204/1000` | `-9.3243` |
| FAM2 confidence-gated | `-0.1365` | `-0.1913` | `+0.0004` | `2/250` | `203/1000` | `-9.0459` |
| HardFreq | `-0.1156` | `-0.1688` | `+0.0002` | `2/250` | `205/1000` | `-9.7109` |
| PFD B1 | `+0.0006` | `+0.0071` | `+0.0000` | `0/250` | `48/1000` | `-0.8031` |
| Haze-prior SCM | `-0.1106` | `-0.1498` | `+0.0015` | `2/250` | `210/1000` | `-9.5836` |

The FAM2/HardFreq/SCM checkpoints are hard-positive only relative to their
matched stop20 baselines; they are not usable direct deltas against the official
ConvIR-B A0 anchor. PFD B1 is closer to the A0 anchor, but its safe replay hard
gain remains only `+0.0071 dB`, far below the `+0.080 dB` gate.

### Delta Learnability

The 32-image full-image delta overfit also failed:

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted delta L1 drop | `0.0231` | `>= 0.30` | fail |
| oracle gain recovery | `0.0111` | `>= 0.50` | fail |
| residual/delta correlation | `0.1552` | `>= 0.40` | fail |
| hard train PSNR gain | `+0.0240 dB` | `>= +0.30 dB` | fail |

This supports the earlier stop20 interpretation: the v0.2RC mask is valuable,
but the current tiny residual head over frozen selector context is not a viable
delta-expression path.

### Residual Source Oracle

Full Haze4K residual-source decomposition showed that the APDR oracle target is
mainly low-frequency/color, not high-frequency detail:

| Variant | Mean | Hard bottom-25% | Easy top-25% | Strong regressions | Severe regressions | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `O_full_delta` | `+0.4195` | `+0.8326` | `+0.0085` | `0/250` | `0/1000` | pass |
| `O_low` | `+0.3820` | `+0.7893` | `+0.0062` | `0/250` | `0/1000` | pass |
| `O_high` | `+0.0296` | `+0.0333` | `+0.0022` | `0/250` | `0/1000` | fail |
| `O_color` | `+0.2116` | `+0.5548` | `+0.0010` | `0/250` | `0/1000` | pass |
| `O_low_plus_color` | `+0.3788` | `+0.7769` | `+0.0063` | `0/250` | `0/1000` | pass |

This preserves the value of the v0.2RC mask and changes the residual design
target: v0.3 should not be detail-first.

### Low-Color Learnability

A simple deployable low-resolution residual plus per-image color-affine branch
failed to overfit 32 full train images:

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted delta L1 drop | `0.0474` | `>= 0.30` | fail |
| oracle gain recovery | `0.0365` | `>= 0.30` | fail |
| predicted delta / target corr | `0.2320` | `>= 0.35` | fail |
| hard train PSNR gain | `+0.0624 dB` | `>= +0.20 dB` | fail |
| easy train PSNR gain | `+0.0053 dB` | `>= -0.010 dB` | pass |

This blocks the exact toy branch, not the low-frequency/color target itself.

### Crop Mask Mismatch

The full-image `M_safe` selector is not crop-recompute compatible:

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| mean mask correlation | `0.4455` | `>= 0.80` | fail |
| p10 mask correlation | `0.2512` | `>= 0.60` | fail |
| mean mask abs diff | `0.06734` | `<= 0.020` | fail |
| hard crop budget drop fraction | `0.0156` | `<= 0.10` | pass |
| near-zero crop mask fraction | `0.1582` | `<= 0.10` | fail |

Future crop training must cache full-image `M_safe` and crop the mask patch
instead of recomputing global budget on the crop.

### Correctability Proxy

A deployable-statistics tabular proxy for `O_low` correctability had strong
ranking but slightly failed the raw-score calibration gate:

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| held-out AUC | `1.0000` | `>= 0.80` | pass |
| held-out Spearman(score, low gain) | `0.9691` | `>= 0.45` | pass |
| easy top-25% mean score | `0.1137` | `<= 0.10` | fail |
| oracle-positive hard mean score | `0.9856` | `>= 0.50` | pass |

Threshold sanity on the full per-image CSV shows `score >= 0.5` gives easy
open rate `0.0438`, positive-hard recall `1.0000`, and negative false-open
`0.0000`. Treat this as a promising calibrated-routing signal, not as a final
P_benefit model.

## Decision

- Decision label: `STOP_DIRECT_SHED_AND_CURRENT_RESIDUAL_KEEP_MASK_PROMOTE_CALIBRATED_CORRECTABILITY_PREFLIGHT`.
- Image/global metric reason: direct SHED replay failed all experts; current residual overfit failed; simple low-color toy overfit failed; residual-source oracle still preserves `O_low hard +0.7893 dB` and `O_color hard +0.5548 dB`.
- Mechanism reason: the v0.2RC safe action mask is valuable, and the residual target is mostly low-frequency/color, but neither historical expert deltas nor the current frozen-context residual head can express it.
- Training-protocol reason: random-crop mask recomputation is invalid for this selector; future residual training must use full-image `M_safe` cache and crop-aligned mask patches.
- Correctability reason: deployable full-image statistics strongly rank `O_low` correctability, but raw score calibration needs a threshold/calibration layer before integration.
- Stop rules:
  - stop direct SHED replay;
  - stop current APDR residual head continuation;
  - stop the exact simple low-color toy branch;
  - do not run a new stop20 scout until full-mask cache training and a stronger low-frequency/color residual expression pass a small overfit gate.
- Next route design:
  - APDR-v0.3 should keep v0.2RC `M_safe`;
  - add calibrated `P_benefit` only as a thresholded/correctability gate;
  - redesign residual expression around low-frequency veil and color correction;
  - train any crop-based residual with precomputed full-image `M_safe` patches;
  - require a 32-image overfit pass before any 128-image or stop20 expansion.
