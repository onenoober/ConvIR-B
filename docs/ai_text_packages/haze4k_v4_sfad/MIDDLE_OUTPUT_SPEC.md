# Middle Output Specification

本文件规定 Haze4K v4 每个实验必须输出的中间结果。后续模型优化只依据这些中间结果、最终指标和失败样本分析做决策，避免凭主观视觉单点判断继续堆模块。

## 1. 总体输出目录

```text
experiments/haze4k_v4/{run_id}/
├── config.yaml
├── run_meta.json
├── git_info.txt
├── model_summary.txt
├── complexity.json
├── train_log.csv
├── val_log.csv
├── best_metrics.json
├── per_image_metrics.csv
├── module_stats.jsonl
├── checkpoints/
│   ├── latest.pth
│   ├── best_psnr.pth
│   └── best_ssim.pth
├── visual/
│   ├── fixed_samples/
│   ├── best_epoch/
│   └── failure_cases/
├── intermediate/
│   ├── epoch_000/
│   ├── epoch_001/
│   ├── epoch_005/
│   ├── epoch_010/
│   ├── epoch_025/
│   ├── epoch_best/
│   └── epoch_last/
├── failure_atlas/
│   ├── worst_psnr.md
│   ├── largest_negative_delta.md
│   ├── thick_haze_failures.md
│   ├── sky_artifacts.md
│   ├── texture_loss.md
│   └── color_shift.md
└── report.md
```

## 2. run_meta.json

字段：

```json
{
  "run_id": "A1_sdfm_only_seed3407_20260707_2300",
  "experiment_id": "A1",
  "model_name": "ConvIR-B-SDFM",
  "dataset": "Haze4K",
  "train_split": "official_or_local_split_name",
  "test_split": "official_or_local_split_name",
  "seed": 3407,
  "start_time": "YYYY-MM-DD HH:MM:SS",
  "end_time": "YYYY-MM-DD HH:MM:SS",
  "git_branch": "codex/haze4k-v4-1-sdfm-only",
  "git_commit": "commit_hash",
  "baseline_run_id": "A0_baseline_seed3407_...",
  "hardware": "GPU name",
  "notes": "short notes"
}
```

## 3. complexity.json

字段：

```json
{
  "params_m": 0.0,
  "flops_g": 0.0,
  "runtime_ms_per_image": 0.0,
  "input_size": [256, 256],
  "extra_params_m_vs_baseline": 0.0,
  "extra_flops_g_vs_baseline": 0.0
}
```

## 4. train_log.csv / val_log.csv

最低字段：

```text
epoch,iter,lr,loss_total,loss_l1,loss_fft,loss_density,loss_edge,psnr,ssim,time_sec,gpu_mem_mb
```

注意：

```text
没有使用的 loss 字段填 0，不要省略列。
```

## 5. best_metrics.json

字段：

```json
{
  "best_epoch_by_psnr": 0,
  "best_psnr": 0.0,
  "best_ssim": 0.0,
  "last_psnr": 0.0,
  "last_ssim": 0.0,
  "delta_psnr_vs_A0": 0.0,
  "delta_ssim_vs_A0": 0.0,
  "lpips": null,
  "niqe": null,
  "decision": "keep/drop/needs_rerun",
  "decision_reason": "short reason"
}
```

## 6. per_image_metrics.csv

最低字段：

```text
image_id,scene_tag,psnr,ssim,lpips,baseline_psnr,baseline_ssim,delta_psnr,delta_ssim,input_gt_l1,input_gt_fft_l1,pred_gt_l1
```

scene_tag 建议：

```text
sky
distant
tree
building
road
low_contrast
heavy_haze
normal
unknown
```

用途：

```text
1. 找出模块真正提升的场景。
2. 找出 largest negative delta 样本。
3. 判断提升是否只来自少数简单样本。
```

## 7. module_stats.jsonl

每行一个 epoch 或一个 eval batch 的统计：

```json
{
  "epoch": 25,
  "split": "val",
  "module": "SDFM",
  "scale": "1/2",
  "R_mean": 0.42,
  "R_std": 0.13,
  "R_min": 0.02,
  "R_max": 0.91,
  "R_lt_005": 0.01,
  "R_gt_095": 0.00,
  "corr_R_input_gt_error": 0.36,
  "corr_R_pred_gt_error": 0.28
}
```

GST 统计：

```json
{
  "epoch": 25,
  "split": "val",
  "module": "GST",
  "scale": "1/2",
  "G_mean": 0.31,
  "G_std": 0.11,
  "G_lt_005": 0.00,
  "G_gt_095": 0.00,
  "G_high_R_mean": 0.45,
  "G_low_R_mean": 0.22
}
```

DCFSB 统计：

```json
{
  "epoch": 25,
  "split": "val",
  "module": "DCFSB",
  "scale": "1/4",
  "low_gate_mean": 0.55,
  "low_gate_std": 0.12,
  "high_gate_mean": 0.38,
  "high_gate_std": 0.10,
  "high_freq_energy_before": 0.18,
  "high_freq_energy_after": 0.21,
  "edge_error_delta_vs_A0": -0.004
}
```

## 8. 可视化输出

每个 fixed sample 至少保存：

```text
input.png
gt.png
pred.png
baseline_pred.png
error_map.png
baseline_error_map.png
delta_error_map.png
```

SDFM 输出：

```text
R_1_full.png
R_2_half.png
R_4_quarter.png
R_overlay_input.png
R_histogram.png
```

GST 输出：

```text
G_skip_full.png
G_skip_half.png
G_overlay_input.png
skip_original_norm.png
skip_clean_norm.png
skip_delta_norm.png
```

DCFSB 输出：

```text
low_component.png
high_component.png
low_gate.png
high_gate.png
low_gate_overlay.png
high_gate_overlay.png
frequency_energy_curve.png
```

局部 crop：

```text
crop_sky_input_gt_pred_error.png
crop_tree_input_gt_pred_error.png
crop_building_input_gt_pred_error.png
crop_road_input_gt_pred_error.png
crop_text_or_edge_input_gt_pred_error.png
```

## 9. 失败样本分析

必须生成以下四类失败榜单：

```text
1. worst_psnr: 当前模型 PSNR 最低的 20 张
2. largest_negative_delta: 相对 A0 下降最多的 20 张
3. high_R_failure: R_s 高但恢复差的样本
4. high_frequency_artifact: 高频增强后出现 halo / noise 的样本
```

每个失败样本记录：

```text
image_id
scene_tag
baseline_psnr
current_psnr
delta_psnr
failure_type
suspected_reason
next_action
```

## 10. 关键异常判定

SDFM 异常：

```text
R_std < 0.03 for most epochs => R collapse
R_gt_095 > 0.8 or R_lt_005 > 0.8 => binary saturation
corr_R_input_gt_error <= 0 for best epoch => degradation field lacks meaning
```

GST 异常：

```text
G_mean < 0.05 => skip gate nearly unused
G_mean > 0.95 => skip almost fully replaced
G_high_R_mean <= G_low_R_mean => gate not degradation-aware
```

DCFSB 异常：

```text
high_gate_mean too high on sky/no-texture regions => likely halo/noise
high_freq_energy_after >> before with PSNR drop => over-sharpening
low_gate not higher in heavy haze scenes => low-frequency branch ineffective
```
