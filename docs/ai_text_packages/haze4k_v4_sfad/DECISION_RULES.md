# Haze4K v4 SFAD Decision Rules

## 1. 总决策原则

后续架构优化只依据三类证据：

```text
1. 主指标：PSNR / SSIM / Params / FLOPs / Runtime
2. 中间证据：R_s、G_s、low/high gate、error map、failure atlas
3. 稳定性证据：多 seed、per-image delta、跨数据集或跨架构验证
```

不要因为单张图好看就保留模块，也不要因为单次 +0.02 dB 就进入最终模型。

## 2. Baseline A0 通过条件

A0 必须满足：

```text
PSNR 接近当前本地 anchor，允许小幅浮动。
SSIM 接近当前本地 anchor。
评估脚本、数据划分和 checkpoint 选择规则固定。
所有后续实验使用同一评估协议。
```

如果 A0 不稳定：

```text
暂停模块实验，先排查 dataset split、crop size、eval mode、RGB/BGR、image range、checkpoint loading。
```

## 3. SDFM 保留条件

强保留：

```text
A1 相对 A0 平均 PSNR >= +0.05 dB
且 R_s 不 collapse
且厚雾/远景/天空类样本 per-image delta 为正
```

弱保留：

```text
A1 PSNR 持平或 +0.02 dB ~ +0.05 dB
但 R_s 可解释，且 A3 与 GST 组合明显提升
```

删除或重做：

```text
A1 PSNR <= -0.05 dB
或 R_s collapse
或 R_s 与 error 完全无关
```

## 4. GST 保留条件

强保留：

```text
A2 相对 A0 平均 PSNR >= +0.05 dB
且 largest_negative_delta 样本没有明显增加
且 G_high_R_mean > G_low_R_mean
```

弱保留：

```text
A2 单独不升，但 A3 明显提升。
这种情况说明 GST 需要 SDFM 提供更可靠的 R_s。
```

删除或重做：

```text
G_s 长期接近 0 或 1
或纹理样本显著下降
或天空区域残雾增加
```

## 5. SDFM-GST 组合保留条件

A3 是空间模块能否成为主线的关键。

强进入下一阶段：

```text
A3 相对 A0 平均 PSNR >= +0.10 dB
SSIM 持平或提升
Params/FLOPs 增量可接受
R_s 和 G_s 均有可解释性
```

可进入下一阶段但标记风险：

```text
A3 相对 A0 +0.05 dB ~ +0.10 dB
但 failure atlas 显示厚雾/远景样本稳定改善
```

暂停优化：

```text
A3 <= A0
且中间图没有明显可解释性。
此时不要继续上 DCFSB，应先重构 SDFM/GST。
```

## 6. density auxiliary loss 使用规则

A4 不作为默认主模型。

保留条件：

```text
A4 比 A3 >= +0.05 dB
且 R_s 可解释性更强
且没有过拟合到 |input - gt| 的假密度图
```

不保留：

```text
A4 PSNR 持平或下降
或真实/跨数据集泛化下降
```

如果 A4 只提升可视化，不提升指标：

```text
作为 appendix 或 analysis，不进入主模型。
```

## 7. DCFSB 保留条件

B1 强保留：

```text
B1 相对 A0 >= +0.05 dB
且天空/平滑区域无明显噪声
且 edge crop 不变差
```

B2 强保留：

```text
B2 比 B1 >= +0.03 dB
且 high-frequency artifact 榜单没有增加
```

B3 默认不进主模型：

```text
只有当 B3 明显优于 B2 >= +0.05 dB
且没有 halo / noise / oversharpening 时，才考虑进入 full 模型。
```

## 8. 最终组合选择

候选：

```text
C1 = SDFM + GST
C2 = SDFM + GST + DCFSB@bottleneck
C3 = SDFM + GST + DCFSB@bottleneck + 1/2 decoder
```

主模型选择优先级：

```text
1. 指标提升稳定
2. 复杂度增量较低
3. 中间图可解释
4. failure atlas 改善
5. 跨数据集或跨架构泛化不下降
```

推荐选择：

```text
如果 C2 与 C3 差距 < 0.03 dB，选择 C2。
如果 C3 明显更高且没有副作用，选择 C3。
如果 DCFSB 组合后拖累空间模块，选择 C1 作为 final-spatial，DCFSB 作为 optional extension。
```

## 9. 多 seed 规则

每个重要节点至少跑 2 个 seed：

```text
A0, A3, B1, C2, C3
```

最终论文表建议 3 个 seed：

```text
seed3407
seed42
seed2026
```

保留条件：

```text
mean_delta_psnr > 0
且至少 2/3 seeds 为正
```

## 10. 下一步优化方向选择

根据中间结果决定：

```text
R_s collapse:
    降低 SDFM 容量，引入 entropy/variance regularization，或改为 residual prior + feature estimator。

G_s collapse:
    将 G_s 初始化偏向 0，减少对 skip 的早期扰动；或使用 S_out = S + alpha * G * (S_clean - S)。

DCFSB 过锐化:
    移除 full-res DCFSB；降低 high branch alpha；对天空区域加入 high-gate suppression。

指标涨但视觉差:
    检查 color shift、halo、over-smoothing；不优先加新模块，先调 loss 权重或 gate 初始化。

视觉好但 PSNR 不涨:
    进入真实数据/无参考指标验证，不作为 Haze4K 主模型优先。
```
