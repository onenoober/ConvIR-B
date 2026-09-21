#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
REMOTE_ROOT=$BASE/repos/ConvIR-B-aecr-ffa-repro-20260921
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/aecr_ffa_repro_20260921
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
OUTPUT=$BASE/runs/aecr_ffa_repro_20260921_smoke
STATUS=$EVID/status.txt
AECR=$BASE/repos/aecr-net-upstream-20260921
FFA=$BASE/repos/ffa-net-upstream-20260921

mkdir -p "$EVID"
printf 'PREFLIGHT_RUNNING date=%s branch=%s commit=%s data=%s output=%s locked_test=false\n' \
  "$(date --iso-8601=seconds)" "$(git -C "$REMOTE_ROOT" branch --show-current)" \
  "$(git -C "$REMOTE_ROOT" rev-parse HEAD)" "$DATA" "$OUTPUT" | tee -a "$STATUS"
for path in "$PY" "$DATA/train" "$AECR/models/AECRNet.py" "$FFA/net/models/FFA.py"; do
  if [[ ! -e "$path" ]]; then
    printf 'PREFLIGHT_FAILED_ENGINEERING missing=%s\n' "$path" | tee -a "$STATUS"
    exit 1
  fi
done
if [[ -e "$OUTPUT/aecr_smoke.pt" || -e "$OUTPUT/ffa_smoke.pt" ]]; then
  printf 'PREFLIGHT_FAILED_ENGINEERING output_exists=%s\n' "$OUTPUT" | tee -a "$STATUS"
  exit 1
fi

for model in ffa aecr; do
  source=$FFA
  [[ "$model" == aecr ]] && source=$AECR
  log=$EVID/smoke_${model}.log
  report=$EVID/smoke_${model}.json
  printf 'smoke_start model=%s date=%s\n' "$model" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
  set +e
  CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" "$REMOTE_ROOT/reproductions/dehazing/smoke.py" \
    --model "$model" --source "$source" --data "$DATA" \
    --output "$OUTPUT" --report "$report" > "$log" 2>&1
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    printf 'PREFLIGHT_FAILED_ENGINEERING model=%s rc=%s log=%s\n' "$model" "$rc" "$log" | tee -a "$STATUS"
    tail -n 35 "$log"
    exit "$rc"
  fi
  printf '%s_SMOKE_OK date=%s report=%s\n' "${model^^}" "$(date --iso-8601=seconds)" "$report" | tee -a "$STATUS"
done
printf 'COMPLETED_GATE_PASS AECR_FFA_SMOKE_OK date=%s locked_test=false\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
