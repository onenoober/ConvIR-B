# CHD-RM Stage Gate Policy

Date: 2026-07-08

## Stage Order

```text
v0 route lock
  -> v1 data and ConvIR-B baseline lock
  -> v2 density/need calibration
  -> v3 no-op RARM audit
  -> v4 single-scale RARM with matched-budget controls
  -> v5 low-haze protection
  -> v6 multiscale haze modulation
  -> v7 OOF candidate lock
  -> v8 final Haze4K confirmation
```

## Gate Rules

| Stage | Pass Rule | If It Fails |
| --- | --- | --- |
| v1 | `train=3000`, `test=1000`, no leakage, A0 finite, metrics repeatable | do not enter v2 |
| v2 | density and need calibration pass; shuffled target control fails | do not connect RARM |
| v3 | no-op RARM equals A0 within tolerance; cost gates pass | do not train RARM |
| v4 | single-scale CHD-RM beats random/shuffled and matched-budget controls | do not enter final candidate pool |
| v5 | low-haze protection passes while heavy-haze benefit remains positive | do not enter final candidate pool |
| v6 | multiscale improves over v5 with acceptable cost | use v5 instead of v6 |
| v7 | one OOF candidate passes all pre-registered gates | do not use locked test |
| v8 | one-shot final confirmation passes | record confirmed result |

## v2 Pass Thresholds

```text
H_density Pearson >= 0.25
H_density Spearman >= 0.30
H_density AUROC heavy-vs-low >= 0.65
R_need Pearson >= 0.20
R_need Spearman >= 0.25
R_need AUROC high-need-vs-low-need >= 0.65
density / need calibration bins at least 4/5 monotonic
low-haze false-strong-recovery rate <= 0.10
shuffled target control must not pass
```

## v3 Pass Thresholds

```text
max_abs_diff(O_new_init, O_A0) <= 1e-6
mean_abs_diff(O_new_init, O_A0) <= 1e-8
PSNR(O_new_init, O_A0) >= 80 dB
finite output ratio = 100%
unexpected checkpoint keys = 0
params increase <= 15%
FLOPs increase <= 15%
FPS drop <= 20%
```

## v7 Candidate Thresholds

```text
OOF mean dPSNR >= +0.08 dB
bootstrap 95% CI low > 0
positive ratio >= 0.56
heavy-haze dPSNR >= +0.12 dB
low-haze dPSNR >= -0.03 dB
p5 dPSNR >= -0.18 dB
CVaR5 dPSNR >= -0.25 dB
dLPIPS <= 0
beats PlainAdapter-MB
beats ConstantGamma-MB
beats RandomDensity-MB
cost increase acceptable
```
