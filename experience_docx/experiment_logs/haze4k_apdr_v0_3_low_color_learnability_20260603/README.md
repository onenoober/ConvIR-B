# APDR-v0.3 Low-Color Learnability

Date: 2026-06-03

Status: completed diagnostic, failed gate.

## Scope

This diagnostic freezes the APDR-v0.2RC selector/action mask and trains only a
small deployable toy residual branch on 32 full Haze4K train images.

The toy branch predicts:

- a low-resolution residual map upsampled to full resolution;
- a per-image RGB affine color correction;
- `J = J0 + M_safe * clamp(Delta_low + Delta_color, -r, r)`.

No checkpoint is saved, and no main model code is changed.

## Command

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_low_color_learnability_20260603/run_apdr_v0_3_low_color_learnability_32.sh
```

## Artifacts

- `low_color_learnability_apdr_v0_3_low_color_learnability_32_seed3407.json`
- `low_color_learnability_per_image_apdr_v0_3_low_color_learnability_32_seed3407.csv`
- `low_color_learnability_history_apdr_v0_3_low_color_learnability_32_seed3407.csv`
- `overfit_apdr_v0_3_low_color_learnability_32_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted delta L1 drop | `0.0474` | `>= 0.30` | fail |
| oracle gain recovery | `0.0365` | `>= 0.30` | fail |
| predicted delta / target corr | `0.2320` | `>= 0.35` | fail |
| hard train PSNR gain | `+0.0624 dB` | `>= +0.20 dB` | fail |
| easy train PSNR gain | `+0.0053 dB` | `>= -0.010 dB` | pass |

Final train-set context:

| Metric | Value |
| --- | ---: |
| mean output gain | `+0.0302 dB` |
| mean oracle gain | `+0.8285 dB` |
| hard oracle gain | `+1.3027 dB` |

## Decision

Decision label: `FAIL_STOP_SIMPLE_LOW_COLOR_TOY_BRANCH`.

The residual-source oracle shows that low-frequency/color targets contain most
of the available gain, but this simple deployable branch still recovers only
`3.65%` of the train oracle gain after 500 overfit steps. It is not worth
expanding this exact toy branch to 128 images or a stop20 scout.

The failure does not invalidate the low-frequency/color target; it says the
expression path must be stronger and trained under the correct full-mask cache
protocol.
