# Haze4K FAM Modres 5-Epoch Scout

Date: 2026-05-31

Remote root:

```text
/root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam-modres-scout-20260531-stop5
```

## Run Contract

- Data: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`
- Seed: `3407`
- Batch size: `8`
- Learning rate: `4e-4`
- Scheduler horizon: `--num_epoch 1000`
- Early stop: `--stop_epoch 5`
- Validation frequency: every epoch
- Compared modes: `--fam_mode original` and `--fam_mode modres`

## Matched Train-Validation PSNR

| Epoch | Original | Modres | Delta |
| --- | ---: | ---: | ---: |
| 1 | 20.60 | 20.77 | +0.17 |
| 2 | 20.64 | 20.58 | -0.06 |
| 3 | 21.37 | 20.54 | -0.83 |
| 4 | 21.90 | 22.40 | +0.50 |
| 5 | 22.31 | 21.94 | -0.37 |

The epoch-5 gap is inside the written scout tolerance of `-0.50 dB`.

## Full-Test Best Checkpoints

| Metric | Original Best | Modres Best | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | 22.3070 | 22.4023 | +0.0953 |
| Mean SSIM | 0.92093 | 0.91905 | -0.00187 |
| Avg synchronized time | 0.05279 s | 0.05196 s | -0.00083 s |
| Peak CUDA allocated | 570.69 MiB | 572.05 MiB | +1.36 MiB |

## Regression Screen

- Median PSNR delta: `-0.0490 dB`
- p5 / p95 PSNR delta: `-3.3088 / +4.0541 dB`
- Worst-10% / best-10% mean PSNR delta: `-3.4940 / +4.1806 dB`
- Strong-reference cutoff: original PSNR >= `24.9883 dB`
- Strong-reference regressions: `142/250` with delta <= `-0.05 dB`
- Worst regressions: `469/1000` with delta <= `-0.20 dB`

## Decision

`modres` is an active mechanism and has a small average PSNR gain on the
5-epoch best checkpoint, but the per-image regression profile is too poor for
unchanged promotion. Do not run an unchanged 20-epoch replacement scout from
this card; either add an explicit preservation guard/narrower target or switch
to the lower-risk FFL/loss route.
