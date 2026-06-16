# Haze4K v2.8 NH-HAZE Official-Weight Evaluation Decision

Original decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`

Audit decision: `V28_ALL55_MIXED_SPLIT_REPRO_INVALID_FOR_OFFICIAL_BENCHMARK`

This route evaluated NH-HAZE with NH-HAZE-specific ConvIR-B and WDMamba checkpoints. The engineering setup was correct for weights and model construction, but the evaluated image set was wrong for official benchmark reproduction: v2.8 used all `55` flat NH-HAZE pairs instead of the official-style `51-55` test subset. Therefore the all-55 aggregate below is retained only as a mixed-split diagnostic.

The corrected official-test split audit is recorded in:

```text
experience_docx/experiment_logs/haze4k_v2_8b_nhhaze_official_test_split_audit_20260616/
```

On `51-55`, the reproduced baselines are A0_NH `20.6636/0.7968` and WDMamba_NH `20.8307/0.8182`, aligning with the expected ConvIR-B README and WDMamba checkpoint-name baselines.

## Mixed All-55 Inherited Alpha Diagnostic Row

- count: `55`
- alpha: `0.375`
- mean/hard/easy dPSNR: `-0.133584` / `+0.122348` / `-0.155758`
- dSSIM: `-0.00598572`
- positive/nonnegative: `0.309091` / `0.309091`
- severe: `26/55` (`283.64/600`)
- worst dPSNR: `-0.956678`

## Mixed All-55 WDMamba NH-HAZE Endpoint

- alpha `1.0` mean/hard/easy dPSNR: `-2.197751` / `-1.093919` / `-2.327606`
- alpha `1.0` severe: `52/55` (`567.27/600`), worst `-4.730593`

## Protocol Notes

- Haze4K locked test touched: `false`.
- NH-HAZE alpha tuning: `false`; alpha rows other than `1.0` are diagnostic.
- A0 and WDMamba both use NH-HAZE-specific checkpoints; A0 data argument is `NHR` and WDMamba DENet blocks is `4`.
- NH-HAZE has 55 paired full-resolution PNG images, each 1600x1200, but official benchmark evaluation must not aggregate all 55 together.
- Metrics are PSNR/SSIM against NH-HAZE GT; hard/easy buckets are bottom/top quartiles by A0 PSNR.

## Official-Test Correction

v2.8b re-aggregated the existing per-image results by split:

```text
01-45 train:  A0 26.7379/0.9444, WDMamba 24.3867/0.8781
46-50 val:    A0 25.8473/0.9292, WDMamba 22.6659/0.8314
51-55 test:   A0 20.6636/0.7968, WDMamba 20.8307/0.8182
```

Use v2.8b, not this all-55 aggregate, for official NH-HAZE discussion.
