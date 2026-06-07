# Haze4K SafeRHFD-v2 Train

Date: 2026-06-02

Status: completed independent cloud stop20 run; automatic gate failed.

## Purpose

Train B1-v2 Selective SafeRHFD as a preservation-aware replacement for the
failed B1 RHFD full-model route.

This run is independent from the concurrent stage-scale surgery sweep:

- code/root: `ConvIR-B-saferhfd-v2-train`
- model name: `ConvIR-Haze4K-B1v2-SafeRHFD-pfdonly-stop20-seed3407-20260602`
- output log dir: `experience_docx/experiment_logs/haze4k_saferhfd_v2_train_20260602`
- comparison target: A0 official ConvIR-B Haze4K checkpoint

## Gate

The automatic gate compares against A0:

- global mean PSNR delta `>= +0.02 dB`
- mean SSIM delta `>= 0`
- hard bottom-25% delta `>= +0.08 dB`
- easy top-25% delta `>= -0.02 dB`
- strong-reference regressions `<= 30 / 250`
- severe regressions `<= 20 / 1000`
- worst10 mean delta `> -0.50 dB`

## Result

The run completed and was evaluated against A0 official ConvIR-B on the Haze4K
test split.

| Metric | Result |
| --- | ---: |
| Mean PSNR delta vs A0 | `+0.00568 dB` |
| Mean SSIM delta vs A0 | `+0.000024` |
| Hard bottom-25% delta | `+0.04890 dB` |
| Easy top-25% delta | `-0.00920 dB` |
| Strong-reference regressions | `70 / 250` |
| Severe regressions | `18 / 1000` |
| Worst10 mean delta | `-0.30090 dB` |

Decision: `FAIL_STOP_SAFERHFD_V2_TRAIN`. The run missed the mean and hard-gain
lines and exceeded the strong-reference regression limit, so it is diagnostic
only and does not justify B2/B3 expansion.

## Pointers

- Route card:
  `experience_docx/experiment_cards/2026-06-02-haze4k-saferhfd-v2-train.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/pfd_rhfd_family_summary.md`

## Artifact Boundary

Commit text-only evidence only. Do not commit checkpoints, visual panels,
datasets, or raw image outputs.
