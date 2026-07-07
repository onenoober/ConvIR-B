# Haze4K v4 SFAD Experiment Protocol

## 1. 实验路线名称

```text
Haze4K v4: SFAD-ConvIR-B
Spatial-Frequency Adaptive Degradation ConvIR-B
```

## 2. 基线

```text
Baseline: ConvIR-B
Dataset: Haze4K
Primary metrics: PSNR / SSIM
Secondary metrics: LPIPS, Params, FLOPs, Runtime
Current anchor: local reproduction around 34.14 PSNR / 0.98971 SSIM
Official reference: ConvIR-B Haze4K around 34.15 PSNR / 0.99 SSIM
```

基线锁定原则：

```text
A0 必须固定随机种子、数据划分、训练配置、评估脚本和 checkpoint 选择规则。
后续所有 delta 均相对于 A0 同配置比较。
```

## 3. 核心假设

### H1: 空间退化场有助于非均匀雾恢复

普通特征融合默认所有空间位置使用同样的恢复强度。Haze4K 中雾分布存在局部差异，模型应该学习一个空间退化场 `R_s` 来控制不同区域的融合和恢复强度。

可验证现象：

```text
1. R_s 在天空、远景、厚雾区域响应更强。
2. R_s 与 |input - gt| 或 |pred - gt| 的空间分布存在正相关。
3. 加入 SDFM 后，厚雾/远景/天空样本的 per-image PSNR delta 更明显。
```

### H2: skip 连接会传递浅层污染特征，需要门控净化

Encoder skip 能传递纹理，但在去雾中也可能携带 haze residual。GST 应在原始 skip 与净化 skip 之间自适应选择。

可验证现象：

```text
1. GST gate 在重雾和远景区域更倾向于使用 clean skip。
2. GST 不应整体关闭 skip，否则细节会下降。
3. 加入 GST 后，边缘区域 error map 应下降，且天空区域残雾不增加。
```

### H3: 低频去雾和高频细节保护需要分开处理

雾幕、亮度漂移和对比度下降更偏低频；边缘、纹理和小结构更偏高频。DCFSB 应根据退化条件自适应路由低/高频处理。

可验证现象：

```text
1. 低频 gate 在厚雾和大面积 veil 区域更强。
2. 高频 gate 在树枝、建筑边缘、文字、道路纹理区域更强。
3. full-res 高频增强如果产生 halo 或天空噪声，应回退到 bottleneck-only 或 1/2 decoder-only。
```

## 4. 模块定义

### 4.1 SDFM: Spatial Degradation Field Modulation

输入：

```text
I_s: 当前尺度输入图像
F_s: 当前尺度主干特征
C_s: 当前尺度条件特征，可来自 SCM 或其他多尺度输入分支
```

输出：

```text
R_s: single-channel spatial degradation field, shape B x 1 x H_s x W_s
Z'_s: degradation-modulated fused feature
```

推荐公式：

```text
Z_s = Fuse(F_s, C_s)

gamma_s, beta_s = Conv(R_s)

Z'_s = Z_s * (1 + alpha_s * tanh(gamma_s)) + alpha_s * tanh(beta_s)
```

初始化：

```text
alpha_s = 0
```

必要统计：

```text
R_s.mean
R_s.std
R_s.min
R_s.max
ratio(R_s < 0.05)
ratio(R_s > 0.95)
corr(R_s, |input - gt|)
corr(R_s, |pred - gt|)
```

### 4.2 GST: Gated Skip Transfer

输入：

```text
S: encoder skip feature
U: decoder upsampled feature
R_s: spatial degradation field
```

输出：

```text
S_out: transferred skip feature
G_s: skip transfer gate
```

推荐公式：

```text
S_clean = CleanConv(S)
G_s = sigmoid(Conv3x3(concat(U, S, R_s)))
S_out = S + G_s * (S_clean - S)
```

必要统计：

```text
G_s.mean
G_s.std
G_s.min
G_s.max
ratio(G_s < 0.05)
ratio(G_s > 0.95)
mean(G_s in high-R region)
mean(G_s in low-R region)
```

### 4.3 DCFSB-lite: Degradation-Conditioned Frequency Selective Block

频率近似分解：

```text
Low_k(F) = Upsample(AvgPool_k(F)), k in {2, 4, 8}
High_k(F) = F - Low_k(F)
```

退化条件路由：

```text
M_low = sigmoid(Conv3x3(concat(F, R_s)))
M_high = sigmoid(Conv3x3(concat(High, R_s)))

F_low = LowBranch(Low) * M_low
F_high = HighBranch(High) * M_high

F_out = F + alpha * Conv1x1(concat(F_low, F_high))
```

初始化：

```text
alpha = 0
```

必要统计：

```text
M_low.mean/std/min/max
M_high.mean/std/min/max
high-frequency energy ratio before/after
edge error before/after
sky/no-texture noise indicator
```

## 5. 实验矩阵

### Phase A: 空间模块验证

| ID | Branch | Model | 目标 |
|---|---|---|---|
| A0 | codex/haze4k-v4-0-baseline-lock | ConvIR-B | 锁定复现基线 |
| A1 | codex/haze4k-v4-1-sdfm-only | ConvIR-B + SDFM | 验证空间退化场调制 |
| A2 | codex/haze4k-v4-2-gst-only | ConvIR-B + GST | 验证 skip 污染控制 |
| A3 | codex/haze4k-v4-3-sdfm-gst | ConvIR-B + SDFM + GST | 验证协同 |
| A4 | codex/haze4k-v4-4-sdfm-gst-density-aux | A3 + weak density loss | 验证辅助密度监督是否必要 |

### Phase B: 频率模块验证

| ID | Branch | Model | 目标 |
|---|---|---|---|
| B1 | codex/haze4k-v4-5-dcfsb-bottleneck | ConvIR-B + DCFSB@bottleneck | 验证基础频率选择收益 |
| B2 | codex/haze4k-v4-6-dcfsb-bottleneck-decoder12 | B1 + DCFSB@1/2 decoder | 验证细节恢复收益 |
| B3 | codex/haze4k-v4-7-dcfsb-all-decoder | B2 + full-res DCFSB | 检查是否过锐化，仅消融 |

### Phase C: 组合模型

| ID | Branch | Model | 目标 |
|---|---|---|---|
| C1 | codex/haze4k-v4-8-sfad-spatial | SDFM + GST | 空间主模型 |
| C2 | codex/haze4k-v4-9-sfad-final-lite | SDFM + GST + DCFSB@bottleneck | 最稳候选 |
| C3 | codex/haze4k-v4-10-sfad-final-full | C2 + DCFSB@1/2 decoder | 指标候选 |

## 6. 每个实验必须保存的内容

每个实验 run 使用统一 run_id：

```text
{experiment_id}_{model_name}_seed{seed}_{YYYYMMDD_HHMM}
```

目录：

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
├── visual/
├── intermediate/
├── failure_atlas/
└── report.md
```

## 7. 训练阶段中间输出频率

推荐保存频率：

```text
epoch 0: 保存初始化输出和模块图，确认 identity 初始化是否接近 baseline
epoch 1: 保存早期模块是否 collapse
epoch 5: 保存收敛早期趋势
epoch 10: 保存中期模块图
每 25 epoch: 保存一次完整中间结果
best epoch: 保存完整中间结果
last epoch: 保存完整中间结果
```

如果训练轮数较少，至少保留：

```text
init / epoch1 / best / last
```

## 8. 固定可视化样本集

需要维护一个固定文件：

```text
docs/ai_text_packages/haze4k_v4_sfad/fixed_eval_samples.txt
```

样本类别建议：

```text
sky_heavy_haze: 5 images
distant_building: 5 images
tree_branch_texture: 5 images
road_scene: 5 images
low_contrast_scene: 5 images
failure_cases_from_A0: 20 images
random_normal_cases: 20 images
```

每个 run 都必须在相同样本集上输出中间结果，这样后续可直接横向比较。
