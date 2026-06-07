# Haze4K B1-Surgery Preserve Sweep

Date: 2026-06-02

Status: completed no-training diagnostic; preservation-first scale `0.70`
passed the minimum diagnostic gate but remains below promotion strength.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: PFD/RHFD preservation diagnostics.
- Dataset or task: Haze4K test split.
- Primary objective: test whether useful B1 RHFD branches can be reused on the
  official A0 backbone without carrying over B1 full-backbone regressions.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_b1_surgery_preserve_20260602/`.
- Branch or isolated workspace: `codex/haze4k-pfd-mainline`.

## Method

The surgery checkpoint uses the official A0 ConvIR-B Haze4K checkpoint as the
backbone, copies the `PFD_RHFD1` and `PFD_RHFD2` branches from the B1 `Best.pkl`,
and scales only the RHFD final `body.4` convolution weights and biases. No
training is performed.

## Gate

The minimum diagnostic gate required:

- mean PSNR delta vs A0 `>= 0.000 dB`;
- mean SSIM delta `>= -0.00005`;
- hard bottom-25% PSNR delta `>= +0.03 dB`;
- easy top-25% PSNR delta `>= -0.02 dB`;
- severe regressions, delta `<= -0.20 dB`, `<= 50 / 1000`;
- strong-reference regressions, A0 top-25% with delta `<= -0.05 dB`,
  `<= 50 / 250`.

## Result

`scale=0.70` is the preservation-first diagnostic candidate:

| Scale | Mean PSNR delta | Hard delta | Easy delta | Severe | Strong reg | Global reg <= -0.05 | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.70 | `+0.01064 dB` | `+0.03317 dB` | `+0.00782 dB` | `0` | `0` | `9` | pass |
| 1.00 | `+0.01268 dB` | `+0.03888 dB` | `+0.00980 dB` | `0` | `9` | `31` | pass, higher regression |

Lower scales were safer but missed the hard-gain line. Visual review for the
selected `0.70` diagnostic pack found no B1-style brightness/color/range
collapse; differences were small and mostly localized.

## Decision

Decision label:

```text
B1_SURGERY_DIAGNOSTIC_POSITIVE_NOT_PROMOTION_READY
```

The result proves that B1 branch surgery can recover a small preservation-safe
signal, but the gain is too thin and single-seed/test-derived to justify
promotion or B2/B3 expansion. The subsequent SafeRHFD-v2 stage-scale diagnostic
is the stricter robustness follow-up.

## Artifact Boundary

Synchronized GitHub evidence is text-only: JSON/CSV/log/txt/md/sh files.
Generated checkpoints, visual panels, sample images, datasets, and raw image
outputs are intentionally excluded.
