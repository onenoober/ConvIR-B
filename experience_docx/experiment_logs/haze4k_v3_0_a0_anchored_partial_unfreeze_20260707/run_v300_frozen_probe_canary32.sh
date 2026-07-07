#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
PY=$BASE/envs/convir-cu121/bin/python
REPO=$BASE/repos/ConvIR-B-haze4k-v3-0-a0-anchored-partial-unfreeze
E=$REPO/experience_docx/experiment_logs/haze4k_v3_0_a0_anchored_partial_unfreeze_20260707
DATA=$BASE/datasets/Haze4K/Haze4K
CKPT=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=$BASE/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v2_41_a0_proximal_supervised_residual_20260706/v241_canary32_oof_summary.json
OUT=$REPO/Dehazing/ITS/results/ConvIR-Haze4K-v3.0-a0-anchored-partial-unfreeze-20260707
cd "$REPO"
STATUS=$E/status.txt
LOG=$E/runtime_logs/v300_frozen_probe_canary32.log
trap 'echo "V300_FROZEN_PROBE_FAILED time=$(date -Iseconds)" >> "$STATUS"' ERR
echo "V300_FROZEN_PROBE_START time=$(date -Iseconds)" >> "$STATUS"
$PY experience_docx/tools/run_haze4k_v300_a0_partial_unfreeze.py   --mode canary32   --scope frozen_probe   --repo-root "$REPO"   --data-dir "$DATA"   --checkpoint "$CKPT"   --split-source-summary "$SPLIT"   --evidence-root "$E"   --output-dir "$OUT"   --epochs 10   --batch-size 2   2>&1 | tee "$LOG"
echo "V300_FROZEN_PROBE_OK time=$(date -Iseconds)" >> "$STATUS"
