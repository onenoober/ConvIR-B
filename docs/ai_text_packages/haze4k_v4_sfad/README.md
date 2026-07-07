# Haze4K v4 SFAD Experiment Package

## 目标

本实验包用于开启 Haze4K v4 新路线。新路线以 ConvIR-B 为强基线，围绕两个更高成功率、更通用的图像恢复痛点进行模块设计和消融验证：

1. **空间非均匀退化与污染特征传递**
   - 模块：`SDFM-GST`
   - 核心思想：显式估计空间退化场，并用它调制多尺度融合与 skip 特征传递。
2. **低频去雾与高频细节保护冲突**
   - 模块：`DCFSB-lite`
   - 核心思想：根据退化强度选择性处理低频雾幕、对比度偏移和高频纹理细节。

最终候选模型：

```text
SFAD-ConvIR-B-lite
= ConvIR-B
+ SDFM at 1/2 and 1/4 scale fusion
+ GST at two skip connections
+ DCFSB at bottleneck
```

增强候选模型：

```text
SFAD-ConvIR-B-full
= ConvIR-B
+ SDFM at 1/2 and 1/4 scale fusion
+ GST at two skip connections
+ DCFSB at bottleneck and 1/2 decoder
```

## 文件说明

```text
docs/ai_text_packages/haze4k_v4_sfad/
├── README.md
├── EXPERIMENT_PROTOCOL.md
├── MIDDLE_OUTPUT_SPEC.md
├── RESULT_REPORT_TEMPLATE.md
├── DECISION_RULES.md
├── RUN_INDEX.md
└── GIT_SYNC_COMMANDS.md
```

## 使用方式

每个实验分支必须同步更新：

```text
1. 实验配置
2. 训练日志
3. 指标结果
4. 中间可视化
5. 模块统计
6. 失败样本分析
7. RESULT_REPORT_TEMPLATE.md 派生出的单次实验报告
8. RUN_INDEX.md 中的一行索引
```

## 不允许直接跳过的步骤

不要直接训练全量组合模型。必须按如下顺序逐步验证：

```text
A0 baseline lock
A1 SDFM only
A2 GST only
A3 SDFM + GST
A4 SDFM + GST + density auxiliary loss, optional
B1 DCFSB bottleneck
B2 DCFSB bottleneck + 1/2 decoder
C1 SDFM + GST
C2 SDFM + GST + DCFSB bottleneck
C3 SDFM + GST + DCFSB bottleneck + 1/2 decoder
```

实验结果不足时，优先回看中间结果，而不是继续堆模块。
