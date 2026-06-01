# Haze4K FAM2 Selectivity Or Kill

Date: 2026-06-01

Status: no-training selectivity meta-analysis complete; FAM2 gate route fails
the offline deployable-selector gate.

## Scope

- Project: ConvIR-B dehazing.
- Dataset or task: Haze4K image dehazing.
- Branch: `codex/haze4k-fam2-selectivity-or-kill`.
- Objective: decide whether existing FAM2 evidence contains a deployable
  per-image selector before spending another GPU run on learned gates.
- Explicitly not run: no training, no checkpoint evaluation, no architecture
  change.

## Inputs

The analysis joins existing per-image evidence only:

| Input | Artifact |
| --- | --- |
| FAM2-only Best | `experience_docx/experiment_logs/haze4k_fam2_modres_stop20_20260531/scout_eval_per_image_seed3407_stop20.csv` |
| bounded gamma-only Best/Last | `experience_docx/experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/scout_eval_per_image_seed3407_stop20_{best,last}.csv` |
| confidence-gate Best/Last | `experience_docx/experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/scout_eval_per_image_seed3407_stop20_{best,last}.csv` |
| deployable proxies | `experience_docx/experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/proxy_separability_seed3407.csv` |
| original seed noise | `experience_docx/experiment_logs/haze4k_stop20_noise_floor_20260601/original_seed_noise_per_image.csv` |

Outputs:

- `experience_docx/experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/selectivity_meta.json`
- `experience_docx/experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/selectivity_meta.csv`
- `experience_docx/experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/selectivity_per_image.csv`

Command:

```bash
python3 experience_docx/tools/analyze_haze4k_fam2_selectivity_meta.py
```

## Definitions

- `positive_gain`: candidate `delta_psnr > 0`.
- `severe_regression`: candidate `delta_psnr <= -0.20`.
- `strong_reference_regression`: easy top-25% original image with
  `delta_psnr <= -0.05`; its AUC is computed inside easy top-25% only so a
  proxy cannot pass by merely separating hard from easy.
- `fam2_noise_margin_gain`: FAM2-only `delta_psnr > max(0.20, original
  per-image seed PSNR sample std)`.
- `bounded_gated_stable_regression`: at least three of bounded Best, bounded
  Last, confidence-gate Best, and confidence-gate Last have
  `delta_psnr <= -0.20`.
- Selector scores are oriented so higher score means "open FAM2"; severe
  regression avoidance is measured as `AUC(-open_score -> severe_regression)`.

## Hard Pass Line

A deployable selector, or a two/three-proxy linear combination, must satisfy
all of:

| Requirement | Threshold |
| --- | ---: |
| FAM2 positive-gain AUC | `>= 0.65` |
| severe-regression avoidance AUC | `>= 0.70` |
| proxy threshold-gate global mean delta | `>= +0.20 dB` |
| proxy threshold-gate easy top-25% mean delta | `>= -0.05 dB` |
| proxy threshold-gate strong-reference regressions | `<= 25/250` |

## Result

No deployable selector passes.

| Check | Best observed | Required |
| --- | ---: | ---: |
| passing deployable selectors | `0` | `>= 1` |
| best positive-gain AUC | `0.5874` | `>= 0.65` |
| best severe-regression avoidance AUC | `0.5888` | `>= 0.70` |
| best easy-only strong-reference avoidance AUC | `0.5204` | high enough to be useful |
| best feasible threshold-gate mean delta | `+0.1333 dB` | `>= +0.20 dB` |

The best positive-gain and severe-regression-avoidance selector was:

```text
-baseline_multiscale_consistency_mean
+baseline_residual_abs_mean
-input_dark_channel_p95
```

It reached only `0.5874` positive-gain AUC, `0.5888` severe-regression
avoidance AUC, and `+0.0573 dB` feasible threshold-gate mean delta.

The best threshold-gate selector was:

```text
-baseline_multiscale_consistency_mean
+input_saturation_mean
+input_saturation_std
```

It reached `+0.1333 dB` feasible global mean delta with easy top-25% at
`+0.0116 dB` and strong-reference regressions capped at `25/250`, but its
positive-gain AUC was only `0.5683` and severe-regression avoidance AUC was
only `0.5720`.

## FAM2 Signal And Failure Pattern

FAM2-only still contains a real but unselectable signal:

| Bucket | Mean delta | Positive gains | Severe regressions |
| --- | ---: | ---: | ---: |
| all | `+0.1739 dB` | `527/1000` | `444/1000` |
| hard bottom 25% | `+0.8159 dB` | `165/250` | `77/250` |
| medium middle 50% | `+0.0828 dB` | `253/500` | `234/500` |
| easy top 25% | `-0.2860 dB` | `109/250` | `133/250` |

Noise-aware FAM2 gains exist on `373/1000` images, but the bounded/gated
variants also show stable severe regressions on `328/1000` images. Worse,
`212/1000` images have FAM2-positive gain while regressing in at least two of
the bounded/gated checkpoints, so the failure is not a clean hard/easy split.

The non-deployable true positive oracle is large: opening FAM2 only where the
observed FAM2-only delta is positive would give `+0.9141 dB` global mean delta
and zero strong-reference regressions. The deployable proxies do not recover
that oracle. Even the non-deployable reference PSNR control stays weak on AUC
(`0.5980` positive-gain, `0.5952` severe-regression avoidance,
`0.4945` easy-only strong-reference avoidance), which confirms the issue is not
solved by predicting baseline hard/easy.

## Decision

Decision label: `FAIL_STOP_FAM_ROUTE`.

Do not run the target-budget gamma-only gate from this evidence. The offline
selector cannot predict positive FAM2 gain or severe/strong-reference
regression well enough, and the best deployable threshold gate misses the
`+0.20 dB` mean-delta floor.

Next route: stop FAM modulation search and switch to hard-aware loss / FFL while
keeping the ConvIR-B inference graph unchanged.
