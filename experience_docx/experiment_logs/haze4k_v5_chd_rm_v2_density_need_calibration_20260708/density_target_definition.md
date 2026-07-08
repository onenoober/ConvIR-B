# Density Target Definition

`H_density_target` is a paired proxy for regional haze density:

```text
H_density_target = normalize_train_p1_p99(blur(gray_abs(I_hazy - I_gt)))
```

The blur is a local average filter. Normalization percentiles and bucket
thresholds are computed from `train_inner` only, then reused for `val_inner`.
The Haze4K locked test is not used for target definition, thresholding, or
variant selection.

