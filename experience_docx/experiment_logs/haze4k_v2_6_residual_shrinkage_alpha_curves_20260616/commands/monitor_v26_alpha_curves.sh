#!/usr/bin/env bash
set -euo pipefail

ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v26-residual-shrinkage-alpha-curves"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616"

echo "remote_time=$(date -Is)"
if tmux has-session -t v26_alpha_curves 2>/dev/null; then
  echo "v26_alpha_curves=ACTIVE"
else
  echo "v26_alpha_curves=NOT_ACTIVE"
fi

if [ -f "$EVID/status_v26_alpha_curves.txt" ]; then
  echo "main_status_tail_begin"
  tail -n 30 "$EVID/status_v26_alpha_curves.txt"
  echo "main_status_tail_end"
else
  echo "main_status=MISSING"
fi

for prefix in v26_wdmamba_alpha_curve v26_fsudp_alpha_curve v26_mbtaylor_alpha_curve; do
  status="$EVID/status_${prefix}.txt"
  log="$EVID/runtime_logs/${prefix}.log"
  if [ -f "$status" ]; then
    echo "${prefix}_status_tail_begin"
    tail -n 8 "$status"
    echo "${prefix}_status_tail_end"
  else
    echo "${prefix}_status=MISSING"
  fi
  if [ -f "$log" ]; then
    echo "${prefix}_progress_tail_begin"
    { grep -E 'progress|V26_RESIDUAL_SHRINKAGE_ALPHA_CURVE_OK|Traceback|Error|FAILED' "$log" || true; } | tail -n 8
    echo "${prefix}_progress_tail_end"
  else
    echo "${prefix}_log=MISSING"
  fi
done

echo "evidence_files_begin"
find "$EVID" -maxdepth 2 -type f | sed "s#^$EVID/##" | sort | tail -n 60
echo "evidence_files_end"
echo "REMOTE_MONITOR_OK"
