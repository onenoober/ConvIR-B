#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
PY=$BASE/envs/convir-cu121/bin/python
REPO=$BASE/repos/ConvIR-B-haze4k-v2-42-a0prox-failure-atlas
EVIDENCE=$REPO/experience_docx/experiment_logs/haze4k_v2_42_a0prox_failure_atlas_20260707
V241=$BASE/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual/experience_docx/experiment_logs/haze4k_v2_41_a0_proximal_supervised_residual_20260706
V240=$BASE/repos/ConvIR-B-haze4k-v2-40-teacher-residual-alignment-atlas/experience_docx/experiment_logs/haze4k_v2_40_teacher_residual_alignment_atlas_20260706/v240_teacher_residual_alignment_atlas_per_image.csv
DATA=$BASE/datasets/Haze4K/Haze4K
CKPT=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
LOG=$EVIDENCE/runtime_logs/v242_a0prox_failure_atlas.log
STATUS=$EVIDENCE/status.txt
trap 'echo "V242_FAILED time=$(date -Iseconds)" >> "$STATUS"' ERR
cd "$REPO"
echo "V242_START time=$(date -Iseconds)" > "$STATUS"
echo "repo_branch=$(git branch --show-current)" >> "$STATUS"
echo "repo_commit=$(git rev-parse --short HEAD)" >> "$STATUS"
echo "diagnostic_only=true" >> "$STATUS"
echo "training_touched=false" >> "$STATUS"
echo "canary80_touched=false" >> "$STATUS"
echo "locked_test_touched=false" >> "$STATUS"
"$PY" experience_docx/tools/run_haze4k_v242_a0prox_failure_atlas.py \
  --repo-root "$REPO" \
  --data-dir "$DATA" \
  --official-checkpoint "$CKPT" \
  --v241-evidence-root "$V241" \
  --v240-atlas-csv "$V240" \
  --evidence-root "$EVIDENCE" \
  2>&1 | tee "$LOG"
echo "V242_OK time=$(date -Iseconds)" >> "$STATUS"
