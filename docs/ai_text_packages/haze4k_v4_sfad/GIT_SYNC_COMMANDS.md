# Git Sync Commands

## 1. 新建 v4 文档协议分支

```bash
git checkout main
git pull origin main
git checkout -b codex/haze4k-v4-sfad-experiment-protocol

mkdir -p docs/ai_text_packages/haze4k_v4_sfad
```

将本实验包文件复制到：

```text
docs/ai_text_packages/haze4k_v4_sfad/
```

提交：

```bash
git add docs/ai_text_packages/haze4k_v4_sfad
git commit -m "docs(haze4k-v4): add SFAD experiment protocol and middle-output spec"
git push -u origin codex/haze4k-v4-sfad-experiment-protocol
```

## 2. 建议分支创建顺序

```bash
git checkout main
git pull origin main

git checkout -b codex/haze4k-v4-0-baseline-lock
git push -u origin codex/haze4k-v4-0-baseline-lock

git checkout main
git checkout -b codex/haze4k-v4-1-sdfm-only
git push -u origin codex/haze4k-v4-1-sdfm-only

git checkout main
git checkout -b codex/haze4k-v4-2-gst-only
git push -u origin codex/haze4k-v4-2-gst-only

git checkout main
git checkout -b codex/haze4k-v4-3-sdfm-gst
git push -u origin codex/haze4k-v4-3-sdfm-gst

git checkout main
git checkout -b codex/haze4k-v4-5-dcfsb-bottleneck
git push -u origin codex/haze4k-v4-5-dcfsb-bottleneck
```

## 3. 每个代码实验分支的最小提交结构

```bash
git add Dehazing/ITS/models
git add Dehazing/ITS/options 2>/dev/null || true
git add Dehazing/ITS/utils 2>/dev/null || true
git add docs/ai_text_packages/haze4k_v4_sfad
git commit -m "exp(haze4k-v4): add <experiment-id> <module-name>"
```

示例：

```bash
git commit -m "exp(haze4k-v4): add A1 SDFM-only degradation field modulation"
git commit -m "exp(haze4k-v4): add A2 GST-only gated skip transfer"
git commit -m "exp(haze4k-v4): add A3 SDFM-GST spatial degradation module"
git commit -m "exp(haze4k-v4): add B1 DCFSB bottleneck frequency selection"
git commit -m "exp(haze4k-v4): add C2 SFAD final lite candidate"
```

## 4. 每次实验完成后的文档提交

实验完成后至少同步：

```text
docs/ai_text_packages/haze4k_v4_sfad/RUN_INDEX.md
experiments/haze4k_v4/{run_id}/report.md
experiments/haze4k_v4/{run_id}/best_metrics.json
experiments/haze4k_v4/{run_id}/per_image_metrics.csv
experiments/haze4k_v4/{run_id}/module_stats.jsonl
```

如果大文件不适合进 Git：

```text
checkpoints/
visual/
intermediate/
```

可以只提交索引和摘要，把大文件放到外部存储路径，并在 report.md 中记录路径。

提交命令：

```bash
git add docs/ai_text_packages/haze4k_v4_sfad/RUN_INDEX.md
git add experiments/haze4k_v4/{run_id}/report.md
git add experiments/haze4k_v4/{run_id}/best_metrics.json
git add experiments/haze4k_v4/{run_id}/per_image_metrics.csv
git add experiments/haze4k_v4/{run_id}/module_stats.jsonl

git commit -m "result(haze4k-v4): add <run_id> metrics and middle-output analysis"
git push
```

## 5. 推荐 tag

当 C2 或 C3 成为阶段最佳模型时：

```bash
git tag -a haze4k-v4-sfad-lite-v0.1 -m "Haze4K v4 SFAD-lite candidate"
git push origin haze4k-v4-sfad-lite-v0.1
```

## 6. 不建议提交到 Git 的内容

```text
*.pth
*.pt
large visual dumps
tensorboard event files if too large
raw dataset
```

建议 `.gitignore` 增加：

```gitignore
experiments/haze4k_v4/*/checkpoints/
experiments/haze4k_v4/*/visual/
experiments/haze4k_v4/*/intermediate/
experiments/haze4k_v4/*/tensorboard/
*.pth
*.pt
```
