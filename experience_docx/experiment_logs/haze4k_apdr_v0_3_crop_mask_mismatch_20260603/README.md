# APDR-v0.3 Crop Mask Mismatch Audit

Date: 2026-06-03

Status: completed diagnostic, failed gate.

## Scope

This audit tests whether APDR-v0.2RC full-image `M_safe` can be safely replaced
by recomputing the selector/action mask on random 256 crops during residual
training.

For each sampled train image:

1. run the frozen APDR-v0.2RC selector on the full image;
2. crop the full-image mask at the sampled crop coordinates;
3. run the same selector on the hazy crop;
4. compare the full-image mask patch with the crop-recomputed mask.

No weights are trained or changed.

## Command

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_crop_mask_mismatch_20260603/run_apdr_v0_3_crop_mask_mismatch_128x4.sh
```

## Artifacts

- `crop_mask_mismatch_apdr_v0_3_crop_mask_mismatch_128x4_seed3407.json`
- `crop_mask_mismatch_per_crop_apdr_v0_3_crop_mask_mismatch_128x4_seed3407.csv`
- `audit_apdr_v0_3_crop_mask_mismatch_128x4_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| mean mask correlation | `0.4455` | `>= 0.80` | fail |
| p10 mask correlation | `0.2512` | `>= 0.60` | fail |
| mean mask abs diff | `0.06734` | `<= 0.020` | fail |
| hard crop budget drop fraction | `0.0156` | `<= 0.10` | pass |
| near-zero crop mask fraction | `0.1582` | `<= 0.10` | fail |

Additional summary:

| Metric | Value |
| --- | ---: |
| images | `128` |
| crops | `512` |
| mean crop/full mask-mean ratio | `3.4047` |
| median crop/full mask-mean ratio | `1.3710` |
| mean full global budget | `0.2655` |
| mean crop global budget | `0.3978` |

## Decision

Decision label: `FAIL_STOP_CROP_RECOMPUTED_MASK_PROTOCOL`.

The APDR-v0.2RC selector is full-image calibrated. Recomputing `M_safe` on
random 256 crops produces a different mask distribution, with low correlation
and excessive near-zero crop masks. Any future residual training that uses
crops should precompute `M_safe` on the full image and crop the corresponding
mask patch, rather than recomputing the global budget on the crop.
