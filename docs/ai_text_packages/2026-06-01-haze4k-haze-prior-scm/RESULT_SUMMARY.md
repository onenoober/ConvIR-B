# Haze4K Haze-Prior SCM Stop20 Result

Date: 2026-06-01

## Configuration

- Control: `ConvIR-B original SCM + original FAM + hard_aux`.
- Candidate: `ConvIR-B haze_prior SCM + original FAM + hard_aux`.
- Seed: `3407`.
- Budget: 20 epochs, full Haze4K test evaluation.
- Route: replace ConvIR SCM with `HazePriorSCM = SCM_rgb(x) + PriorBranch(prior_maps(x))`.
- Prior maps: min RGB, max RGB, dark channel, saturation, grayscale gradient.
- Safety design: final prior-branch conv zero-init, so candidate equals original
  SCM at initialization.

## Completion

- Remote host: `autodl-dehaze3`.
- Remote workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm`.
- Local evidence mirror:
  `/home/ubuntu/workspace/ConvIR-B-haze-prior-scm/experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/`.
- Completion marker: `complete 2026-06-01T18:39:12+08:00`.

## Best Checkpoint

| Metric | original SCM + hard_aux | haze-prior SCM + hard_aux | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | 24.935965 | 24.557032 | -0.378932 |
| Mean SSIM | 0.948732 | 0.943324 | -0.005407 |
| Strong-reference regressions, delta <= -0.05 | n/a | 185 / 250 | n/a |
| Regressions, delta <= -0.20 | n/a | 528 / 1000 | n/a |

Best bucket deltas:

| Bucket by original PSNR | Mean PSNR delta | Median PSNR delta | Positive ratio | Regressions <= -0.20 | Mean SSIM delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard bottom 25% | +0.350125 | +0.269632 | 0.556 | 100 / 250 | +0.000909 |
| medium middle 50% | -0.107365 | -0.182564 | 0.462 | 248 / 500 | -0.005532 |
| easy top 25% | -1.651124 | -1.392050 | 0.256 | 180 / 250 | -0.011474 |

## Last Checkpoint

| Metric | original SCM + hard_aux | haze-prior SCM + hard_aux | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | 22.362059 | 23.946890 | +1.584831 |
| Mean SSIM | 0.913947 | 0.947844 | +0.033897 |
| Strong-reference regressions, delta <= -0.05 | n/a | 93 / 250 | n/a |
| Regressions, delta <= -0.20 | n/a | 229 / 1000 | n/a |

The Last checkpoint is diagnostic only because the matched original-SCM final
checkpoint collapsed to `22.3621 dB`.

## Mechanism

- Synthetic neutral-init output diffs: `[0.0, 0.0, 0.0]`.
- Real-batch neutral-init output diffs: `[0.0, 0.0, 0.0]`.
- Real-batch prior branch nonzero gradients: `10432/35776`.
- Parameter delta: `+35776`, or `+0.4145%`.
- Epoch 20 prior/rgb abs ratios: SCM1 `0.6943`, SCM2 `0.4843`.

## Decision

Decision label: `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`.

The route has a real hard-bucket signal, but the exact `haze_prior SCM +
hard_aux` configuration fails the first promotion gate because Best checkpoint
global PSNR/SSIM are worse than matched control and preservation regressions are
too large.

Next candidates should not continue this exact configuration to 80 epochs.
Reasonable follow-up options are either isolating SCM with original loss or
moving to the next documented loss candidate with matched controls and the same
strong/easy preservation gates.
