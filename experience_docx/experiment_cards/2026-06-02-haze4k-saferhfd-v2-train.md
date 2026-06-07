# Haze4K SafeRHFD-v2 Train

Date: 2026-06-02

Status: completed independent stop20 training diagnostic; automatic gate failed.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: PFD/RHFD preservation diagnostics.
- Dataset or task: Haze4K dehazing.
- Primary objective: train B1-v2 Selective SafeRHFD as a preservation-aware
  replacement for the failed B1 RHFD full-model route.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_saferhfd_v2_train_20260602/`.
- Branch or isolated workspace: `codex/haze4k-pfd-mainline` derivative
  `ConvIR-B-saferhfd-v2-train`.

## Gate

The stop20 gate compared against A0:

- global mean PSNR delta `>= +0.02 dB`;
- mean SSIM delta `>= 0`;
- hard bottom-25% delta `>= +0.08 dB`;
- easy top-25% delta `>= -0.02 dB`;
- strong-reference regressions `<= 30 / 250`;
- severe regressions `<= 20 / 1000`;
- worst10 mean delta `> -0.50 dB`.

## Result

The run completed on `autodl-dehaze3` with model name
`ConvIR-Haze4K-B1v2-SafeRHFD-pfdonly-stop20-seed3407-20260602`.

| Metric | Result |
| --- | ---: |
| Mean PSNR delta vs A0 | `+0.00568 dB` |
| Mean SSIM delta vs A0 | `+0.000024` |
| Hard bottom-25% delta | `+0.04890 dB` |
| Easy top-25% delta | `-0.00920 dB` |
| Strong-reference regressions | `70 / 250` |
| Severe regressions | `18 / 1000` |
| Worst10 mean delta | `-0.30090 dB` |

The route stayed near A0 globally and was not catastrophic, but it missed the
mean and hard-gain gates and exceeded the strong-reference regression limit.

## Decision

Decision label:

```text
FAIL_STOP_SAFERHFD_V2_TRAIN
```

SafeRHFD-v2 training is diagnostic only. It should not be promoted and should
not be used to justify B2/B3 expansion without a new mechanism that materially
improves hard gain while reducing strong-reference regressions.

## Artifact Boundary

Synchronized GitHub evidence is text-only: JSON/CSV/log/txt/md/sh files.
Checkpoints, generated images, datasets, and raw inference outputs are excluded.
