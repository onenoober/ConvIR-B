#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=$BASE/repos/ConvIR-B-v24-c12-wd0375-distill/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/v24_c12_split_manifest.json
TEACHER=$BASE/runtime_cache/v24_c12_wd0375_teacher
STATUS=$EVID/status.txt
mkdir -p "$EVID"
choose_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 < 2000 && $3 < 20) {print $1; exit}}'
}
GPU="$(choose_gpu)"
if [ -z "$GPU" ]; then
  echo "FAILED_INFRA no_free_gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 90
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
{
  echo "v213_n012_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "split=$SPLIT"
  echo "teacher=$TEACHER"
  echo "gpu=$GPU"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"
cd "$WORK"

run_step() {
  local name="$1"
  shift
  local log="$EVID/${name}.log"
  echo "${name}_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
  set +e
  PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1
  local rc=$?
  set -e
  echo "${name}_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [ "$rc" -ne 0 ]; then
    echo "${name}_FAILED" | tee -a "$STATUS"
    exit "$rc"
  fi
  echo "${name}_OK" | tee -a "$STATUS"
}

run_step v213_n0_contract "$PY" experience_docx/tools/nopost_contract_audit.py \
  --repo-root "$WORK" \
  --data-dir "$DATA" \
  --checkpoint "$A0" \
  --split-manifest "$SPLIT" \
  --scope fold_val \
  --fold 0 \
  --max-images 8 \
  --out-dir "$EVID"

run_step v213_n1_feature_table "$PY" experience_docx/tools/build_nopost_feature_table.py \
  --data-dir "$DATA" \
  --checkpoint "$A0" \
  --split-manifest "$SPLIT" \
  --teacher-cache-dir "$TEACHER" \
  --scope train_core \
  --out-dir "$EVID" \
  --print-freq 50

run_step v213_n1_probe "$PY" experience_docx/tools/oof_probe_gain_risk.py \
  --feature-table "$EVID/v213_n1_feature_rows_cloud_only.csv" \
  --out-dir "$EVID"

run_step v213_n2_identity "$PY" Dehazing/ITS/tools/nopost_identity_check.py \
  --data-dir "$DATA" \
  --checkpoint "$A0" \
  --split-manifest "$SPLIT" \
  --scope fold_val \
  --fold 0 \
  --max-images 64 \
  --out-dir "$EVID"

echo "v213_n012_done rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "V213_N012_ALL_OK_READY_FOR_N3" | tee -a "$STATUS"
