#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-$(pwd)}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DATA=${DATA:-/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K}
DEPTH=${DEPTH:-/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$WORK}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/dta_v3_6_haze4k_oof_splits_seed3407.json}
EVID=${EVID:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613}
GROUP_DIR=${GROUP_DIR:-$EVID/phase_d6_outputdiff_groups}
STATUS=${STATUS:-$EVID/status_phase_d6_outputdiff_policy.txt}
MAX_PARALLEL=${MAX_PARALLEL:-2}
FREE_GPU_MAX_USED_MIB=${FREE_GPU_MAX_USED_MIB:-1200}
FEATURE_MAX_SIDE=${FEATURE_MAX_SIDE:-384}
INCLUDE_RUN_SUBSTRING=${INCLUDE_RUN_SUBSTRING:-quick5full}

mkdir -p "$EVID" "$GROUP_DIR"
{
  echo "dta_v3_7_phase_d6_outputdiff_start $(date -Is)"
  echo "state=RUNNING_D6_OUTPUTDIFF_FEATURE_POLICY"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "checkpoint_root=$CHECKPOINT_ROOT"
  echo "split_json=$SPLIT_JSON"
  echo "feature_max_side=$FEATURE_MAX_SIDE"
  echo "locked_test_touched=false"
  git -C "$WORK" branch --show-current
  git -C "$WORK" rev-parse --short HEAD
  git -C "$WORK" status --short
} | tee "$STATUS"

for required in "$PY" "$A0" "$CHECKPOINT_ROOT" "$SPLIT_JSON" "$EVID/v37_tau_oof_per_image_action_table.csv" "$EVID/v37_tau_real_blend_single_actions_all.csv"; do
  if [[ ! -e "$required" ]]; then
    echo "DTA_V3_7_D6_OUTPUTDIFF_FAILED missing=$required $(date -Is)" | tee -a "$STATUS"
    exit 1
  fi
done

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk 'NR==1 {print int($1)}'
}

pick_gpu() {
  local best_gpu="" best_mem=999999 mem idx
  while IFS=, read -r idx _rest; do
    idx="${idx// /}"
    [[ -z "$idx" ]] && continue
    mem="$(gpu_used_mib "$idx" || echo 999999)"
    if [[ "$mem" -lt "$best_mem" && "$mem" -le "$FREE_GPU_MAX_USED_MIB" ]]; then
      best_gpu="$idx"
      best_mem="$mem"
    fi
  done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
  [[ -n "$best_gpu" ]] && printf '%s\n' "$best_gpu"
}

run_group() {
  local fold="$1" seed="$2" gpu="$3"
  local log="$GROUP_DIR/d6_outputdiff_seed${seed}_f${fold}.log"
  echo "d6_group_start fold=$fold seed=$seed gpu=$gpu log=$log $(date -Is)" | tee -a "$STATUS"
  (
    cd "$WORK"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$WORK/Dehazing/ITS:$WORK:${PYTHONPATH:-}"
    "$PY" experience_docx/tools/extract_haze4k_dta_v37_outputdiff_features.py \
      --data_dir "$DATA" \
      --a0_checkpoint "$A0" \
      --checkpoint_root "$CHECKPOINT_ROOT" \
      --action_table_csv "$EVID/v37_tau_oof_per_image_action_table.csv" \
      --include_run_substring "$INCLUDE_RUN_SUBSTRING" \
      --depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --fold "$fold" \
      --seed "$seed" \
      --output_dir "$GROUP_DIR" \
      --feature_max_side "$FEATURE_MAX_SIDE"
  ) >"$log" 2>&1
  local rc=$?
  echo "d6_group_done fold=$fold seed=$seed rc=$rc log=$log $(date -Is)" | tee -a "$STATUS"
  return "$rc"
}

RUN_GROUPS=("0 3407" "1 3407" "0 3411" "1 3411")
pids=()
for item in "${RUN_GROUPS[@]}"; do
  fold="${item%% *}"
  seed="${item##* }"
  out="$GROUP_DIR/v37_d6_outputdiff_features_seed${seed}_f${fold}.csv"
  if [[ -s "$out" ]]; then
    echo "d6_group_skip_existing fold=$fold seed=$seed out=$out $(date -Is)" | tee -a "$STATUS"
    continue
  fi
  while :; do
    while [[ "${#pids[@]}" -ge "$MAX_PARALLEL" ]]; do
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          new_pids+=("$pid")
        else
          wait "$pid"
        fi
      done
      pids=("${new_pids[@]}")
      sleep 20
    done
    gpu="$(pick_gpu || true)"
    if [[ -n "${gpu:-}" ]]; then
      run_group "$fold" "$seed" "$gpu" &
      pids+=("$!")
      sleep 25
      break
    fi
    echo "d6_wait_no_gpu free_threshold_mib=$FREE_GPU_MAX_USED_MIB $(date -Is)" | tee -a "$STATUS"
    sleep 60
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

COMBINED="$EVID/v37_d6_outputdiff_features_all.csv"
"$PY" - <<PY
import csv
from pathlib import Path
group_dir = Path("$GROUP_DIR")
paths = sorted(group_dir.glob("v37_d6_outputdiff_features_seed*_f*.csv"))
if len(paths) != 4:
    raise SystemExit(f"expected 4 group csvs, found {len(paths)}: {paths}")
rows = []
fields = []
for path in paths:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for key in reader.fieldnames:
                if key not in fields:
                    fields.append(key)
        rows.extend(reader)
with Path("$COMBINED").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})
print(f"DTA_V3_7_D6_OUTPUTDIFF_COMBINE_OK files={len(paths)} rows={len(rows)} output=$COMBINED", flush=True)
PY

cd "$WORK"
export PYTHONPATH="$WORK/Dehazing/ITS:$WORK:${PYTHONPATH:-}"
"$PY" experience_docx/tools/train_haze4k_dta_v37_d6_outputdiff_policy.py \
  --single_actions_csv "$EVID/v37_tau_real_blend_single_actions_all.csv" \
  --feature_action_table_csv "$EVID/v37_tau_oof_per_image_action_table.csv" \
  --outputdiff_features_csv "$COMBINED" \
  --output_dir "$EVID" \
  --output_prefix v37_d6_outputdiff \
  --include_run_substring "$INCLUDE_RUN_SUBSTRING" | tee "$EVID/phase_d6_outputdiff_policy.log"

echo "dta_v3_7_phase_d6_outputdiff_done rc=0 $(date -Is)" | tee -a "$STATUS"
echo "DTA_V3_7_PHASE_D6_OUTPUTDIFF_POLICY_OK $(date -Is)" | tee -a "$STATUS"
