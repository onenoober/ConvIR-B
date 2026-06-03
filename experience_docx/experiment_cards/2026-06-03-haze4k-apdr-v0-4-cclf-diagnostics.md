# Haze4K APDR-v0.4 CCLF Diagnostics

Date: 2026-06-03

Status: preflight diagnostics completed; low-field-only v0.4A candidate supported; full low+color v0.4C stop20 remains blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR over ConvIR-B official anchor.
- Dataset or task: Haze4K train/test on `autodl-dehaze3`.
- Primary objective: verify whether cached full-image `M_safe` plus low-frequency/color targets can support a deployable v0.4 route.
- Main metric: oracle gain recovery and preservation gates before any stop20 run.
- Execution environment: `autodl-dehaze3`, `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_4_*_20260603/`.
- Branch or isolated workspace: `codex/haze4k-apdr-v0-4-cclf-diagnostics`.

## Baseline Contract

- Baseline implementation: ConvIR-B official Haze4K checkpoint and APDR-v0.2RC full-image selector checkpoint.
- Baseline checkpoint: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Selector checkpoint: `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Reproduced baseline result: Haze4K pretrained ConvIR-B around PSNR `34.14`, SSIM `0.98971`.
- Reference entrypoints that must remain stable: `Dehazing/ITS/main.py`, `Dehazing/ITS/models/APDRConvIR.py`, and APDR-v0.2RC selector behavior.

## Most Valuable Attempt

- Why this is the highest-value next attempt: current evidence says `M_safe` is safe, `O_low/O_color` has headroom, and residual failure is expression/protocol rather than route location.
- Target failure or opportunity: crop-recomputed masks break full-image routing, and tiny residual/color branches cannot learn the target.
- Cheap preflight evidence: cache roundtrip, lowpass scale sweep, free-parameter low/color overfit, and train-only correctability calibration.
- Earliest decisive gate: free-parameter low/color sanity and cache exactness.
- Expected cost or attempt-count saving: block stop20 until target/mask/loss/correctability are proven.
- What success decides: authorize implementation of a deployable low-frequency/color field branch.
- What failure decides: stop v0.4 before training and isolate the failed part.
- Why a cheaper diagnostic is not enough: v0.3 already showed toy branch and crop recompute are misleading; v0.4 must validate full-image cached targets directly.

## Hypothesis

If we replace crop-recomputed residual learning with cached full-image `M_safe` and train-calibrated correctability, then low-frequency/color residual targets should become learnable without damaging easy images because the route uses proven safe locations, train-only benefit gating, and a target family with strong oracle headroom.

## Change

- Code branch: `codex/haze4k-apdr-v0-4-cclf-diagnostics`.
- Exact code/config change: add diagnostic-only scripts for cache/scale, free-parameter low/color sanity, and train-calibrated correctability.
- Enabled mechanisms: APDR-v0.2RC full-image mask, Gaussian lowpass target sweep, weighted color affine target, per-image free-parameter overfit, train 5-fold correctability calibration.
- Explicitly disabled mechanisms: direct SHED replay, high-frequency residual-first, crop mask recompute training, stop20 training.
- Initialization or no-op behavior: no deployable v0.4 model is trained in this card.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| cache roundtrip | cached crop mask/target max diff `<= 1e-8` | pass: max diff `0.0` for mask and all lowpass targets |
| low target oracle | best lowpass hard gain remains decision-grade | pass: sigma `3` mean `+0.7804`, hard `+1.4176`, easy `+0.2977` |
| free-param low/color | recovery `>= 0.80`, corr `>= 0.70` | split: low target strong but strict loss-drop gate failed; color failed |
| correctability train calibration | easy open `<= 0.05`, positive-hard recall `>= 0.95`, negative false-open `<= 0.02` | pass: train easy `0.0493`, test easy `0.0200`, false-open `0.0`, hard recall `0.9750` |
| stop20 authorization | all above pass | blocked for full v0.4C; only a separate low-field-only implementation card is justified |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| cached patch max diff | proves crop training can use full-image routing without recompute drift | train cache subset | `cache_scale_summary_*.json` |
| lowpass scale hard/easy gain | chooses output resolution and target family | train subset | `cache_scale_per_image_*.csv` |
| free-param recovery | separates target/application bugs from deployable mapping failure | first 32 train images | `freeparam_lowcolor_*.json` |
| train-calibrated threshold | prevents test-derived threshold leakage | train/test split | `correctability_traincalib_*.json` |

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| cache | max diff `<= 1e-8` for cached crop mask and low targets | no crop-recompute path needed | stop if cache path is not exact |
| free-param | recovery `>= 0.80`, corr `>= 0.70`, easy gain `>= -0.010 dB` | target/loss/application are sound | stop if free parameters cannot recover oracle |
| correctability | train and test safety gates pass | train `tau` controls easy/negative false opens | use threshold only if train-calibrated gates pass |
| stop20 | not authorized by this card | requires a separate deployable branch card | do not run stop20 from diagnostics alone |

## Final Results

| Diagnostic | Artifact | Verdict | Key observations |
| --- | --- | --- | --- |
| cache-scale | `haze4k_apdr_v0_4_cache_scale_20260603/cache_scale_summary_*.json` | pass | full-image cache roundtrip exact; best lowpass sigma `3` gives hard `+1.4176 dB` and easy `+0.2977 dB` on the train128 audit |
| freeparam-low | `haze4k_apdr_v0_4_freeparam_low_20260603/freeparam_lowcolor_*.json` | partial pass signal, strict gate fail | recovery `1.0938`, corr `0.9322`, hard `+1.2484 dB`, easy `+0.4411 dB`, no strong/severe regressions; only loss-drop fraction `0.6762 < 0.80` failed |
| freeparam-color | `haze4k_apdr_v0_4_freeparam_color_20260603/freeparam_lowcolor_*.json` | fail | corr `0.3899`, hard `+0.3169 dB`, severe regressions `2`; do not merge color into v0.4A |
| correctability-traincalib | `haze4k_apdr_v0_4_correctability_traincalib_20260603/correctability_traincalib_*.json` | pass | train-calibrated tau `0.9897`; test AUC `1.0`, Spearman `0.9729`, easy open `0.0200`, false-open `0.0`, positive-hard recall `0.9750` |

## Decision

- Decision label: `PREFLIGHT_COMPLETE_LOW_FIELD_ONLY_CANDIDATE`.
- Stop20 status: `DO_NOT_RUN_V0_4C_STOP20`.
- What this decides next: implement a separate deployable v0.4A low-field-only branch that uses cached full-image `M_safe` and train-calibrated correctability; keep color correction as diagnostic/rework only.
- Rationale: cache and correctability gates passed, and the low-field target/application sanity is strong despite one conservative loss-drop threshold miss. The color branch failed both correlation/safety quality and should not be bundled into the next trainable route.
