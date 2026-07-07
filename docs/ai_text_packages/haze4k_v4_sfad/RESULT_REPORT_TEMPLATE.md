# Experiment Report Template

复制本文件为：

```text
experiments/haze4k_v4/{run_id}/report.md
```

## 1. 基本信息

```text
Run ID:
Experiment ID:
Model:
Branch:
Commit:
Seed:
Dataset:
Train split:
Test split:
Start time:
End time:
GPU:
```

## 2. 实验目的

```text
本实验验证：
-
```

## 3. 相比 A0 的改动

```text
Architecture changes:
-

Loss changes:
-

Training changes:
-

Logging changes:
-
```

## 4. 主结果

| Metric | A0 Baseline | Current | Delta |
|---|---:|---:|---:|
| PSNR |  |  |  |
| SSIM |  |  |  |
| LPIPS |  |  |  |
| Params(M) |  |  |  |
| FLOPs(G) |  |  |  |
| Runtime(ms/img) |  |  |  |

## 5. 训练稳定性

```text
Best epoch:
Last epoch:
PSNR curve trend:
Loss curve trend:
NaN/overflow:
GPU memory:
```

## 6. Per-image delta 分析

Top improved cases:

| image_id | scene_tag | A0 PSNR | Current PSNR | Delta | Observation |
|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |

Top degraded cases:

| image_id | scene_tag | A0 PSNR | Current PSNR | Delta | Suspected reason |
|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |

## 7. 中间结果分析

### 7.1 SDFM / R_s

```text
R_mean:
R_std:
R_lt_005:
R_gt_095:
corr_R_input_gt_error:
corr_R_pred_gt_error:
Interpretation:
```

结论：

```text
R_s 是否有空间区分度：
R_s 是否与雾/误差区域对应：
是否存在 collapse：
```

### 7.2 GST / G_s

```text
G_mean:
G_std:
G_lt_005:
G_gt_095:
G_high_R_mean:
G_low_R_mean:
Interpretation:
```

结论：

```text
skip gate 是否退化感知：
是否过度关闭 skip：
是否改善重雾区域：
是否损伤纹理：
```

### 7.3 DCFSB

```text
low_gate_mean:
high_gate_mean:
high_freq_energy_before:
high_freq_energy_after:
edge_error_delta_vs_A0:
Interpretation:
```

结论：

```text
低频去雾是否增强：
高频细节是否改善：
是否出现 halo / noise / oversharpening：
```

## 8. 可视化结论

必须附带以下路径或图表：

```text
visual/fixed_samples/
intermediate/epoch_best/
failure_atlas/
```

观察：

```text
Sky:
Distant buildings:
Tree branches:
Road texture:
Low contrast:
Failure cases:
```

## 9. 决策

选择一项：

```text
[ ] keep
[ ] drop
[ ] needs_rerun
[ ] modify_and_rerun
```

理由：

```text

```

下一步：

```text

```

## 10. 备注

```text

```
