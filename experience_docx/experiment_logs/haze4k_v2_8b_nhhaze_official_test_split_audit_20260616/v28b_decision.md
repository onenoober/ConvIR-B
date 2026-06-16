# Haze4K v2.8b NH-HAZE Official-Test Split Audit Decision

Decision: `V28B_NHHAZE_OFFICIAL_TEST51_55_REPRO_ALIGNED_WITH_EXPECTED_BASELINES`

The v2.8 official-weight run is engineering-valid for checkpoint loading and
per-image inference, but its all-55 aggregate is not a valid official NH-HAZE
benchmark. The local dataset was a flat `01..55` directory, and the v2.8
launcher evaluated all `55` pairs. This mixed official train, validation, and
test images.

The v2.8 evaluator pair discovery used `data_dir.glob("*_hazy.png")` with no
split argument, then paired each `NN_hazy.png` with `NN_GT.png`. The shard
manifests therefore correctly report `pair_count_total=55`, but that count is
the flat directory count, not an official test count.

## Serious Protocol Issue

The all-55 aggregate must be relabeled:

```text
V28_ALL55_MIXED_SPLIT_REPRO_INVALID_FOR_OFFICIAL_BENCHMARK
```

The inflated A0_NH result `26.1047/0.9296` is explained by evaluating
`01-45` train images and `46-50` validation images together with `51-55` test.
Split aggregation shows:

```text
01-45 train:  A0 26.7379/0.9444, WDMamba 24.3867/0.8781
46-50 val:    A0 25.8473/0.9292, WDMamba 22.6659/0.8314
51-55 test:   A0 20.6636/0.7968, WDMamba 20.8307/0.8182
01-55 mixed:  A0 26.1047/0.9296, WDMamba 23.9070/0.8684
```

This is a data split contamination / leakage issue for official reproduction,
not evidence that the NH-specific ConvIR-B checkpoint reaches 26 dB on the
official NH-HAZE test set.

## What Was Not The Main Problem

No evidence was found that v2.8 accidentally used Haze4K weights:

- ConvIR-B used `nhhaze-base.pkl` with `build_net("base", "NHR", "original")`;
- WDMamba used `NH_20.83.pth`;
- WDMamba strict loading required the NH-style `DENet(3, 4)` head.

The dominant issue is the evaluated image set.

## Official-Test Result

On the official `51-55` test subset, the reproduced baselines align with the
expected public numbers:

```text
ConvIR-B README NH-HAZE base: 20.66 / 0.802
v2.8b A0_NH on 51-55:        20.6636 / 0.7968

WDMamba checkpoint name:      NH_20.83.pth
v2.8b WDMamba_NH on 51-55:   20.8307 / 0.8182
```

On `51-55`, inherited `alpha=0.375` gives mean dPSNR `+0.515796`, dSSIM
`+0.02203439`, positive `1.0`, severe `0/5`, and worst `+0.078772`. This is
only a five-image post-run diagnostic from an alpha grid that was already
computed in v2.8. It must not be promoted as an NH-HAZE-selected fixed alpha
without a separate validation/OOF protocol.

## Closeout

Use v2.8b for any NH-HAZE official-test discussion. Keep v2.8 all-55 evidence
only as a mixed-split diagnostic and cautionary audit artifact.
