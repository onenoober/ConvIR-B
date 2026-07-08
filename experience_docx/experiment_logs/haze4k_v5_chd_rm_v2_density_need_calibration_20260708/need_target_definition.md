# Restoration-Need Target Definition

`R_need_target` is a paired proxy for where the frozen ConvIR-B baseline still
has residual error:

```text
R_need_target = normalize_train_p1_p99(blur(gray_abs(O_A0 - I_gt)))
```

`O_A0` is produced by the official ConvIR-B Haze4K checkpoint. Normalization
percentiles and bucket thresholds are computed from `train_inner` only, then
reused for `val_inner`. The Haze4K locked test is not used.

