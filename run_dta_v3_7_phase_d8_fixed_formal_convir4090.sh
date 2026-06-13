#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$(pwd)}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/dta_v3_6_haze4k_oof_splits_seed3407.json}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d8_fixed_formal.txt
RUN_TAG=${DTA_V37_D8_RUN_TAG:-d8formal}
VARIANTS_CSV=${VARIANTS_CSV:-u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006}
FOLDS_CSV=${FOLDS_CSV:-0,1,2,3,4}
SEEDS_CSV=${SEEDS_CSV:-3407,3411,2026}
STAGE=${STAGE:-quick5full}
MAX_PARALLEL_TRAIN=${MAX_PARALLEL_TRAIN:-}
MAX_PARALLEL_RENDER=${MAX_PARALLEL_RENDER:-}
MAX_PARALLEL_FEATURES=${MAX_PARALLEL_FEATURES:-2}
FREE_GPU_MAX_USED_MIB=${FREE_GPU_MAX_USED_MIB:-2500}
FREE_GPU_MAX_UTIL_PCT=${FREE_GPU_MAX_UTIL_PCT:-20}
FEATURE_FREE_GPU_MAX_USED_MIB=${FEATURE_FREE_GPU_MAX_USED_MIB:-1200}
FEATURE_MAX_SIDE=${FEATURE_MAX_SIDE:-384}
PRIMARY_POLICY_ID=${PRIMARY_POLICY_ID:-primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100}

mkdir -p "$EVID"
{
  echo "dta_v3_7_phase_d8_fixed_formal_start $(date -Is)"
  echo "state=RUNNING_D8_FIXED_FORMAL_CONFIRMATION"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "split_json=$SPLIT_JSON"
  echo "variants=$VARIANTS_CSV folds=$FOLDS_CSV seeds=$SEEDS_CSV stage=$STAGE run_tag=$RUN_TAG"
  echo "sealed_policy_id=$PRIMARY_POLICY_ID"
  echo "policy_search=false"
  echo "locked_test_touched=false"
  echo "raw_d1_full_5x3_run=true"
  echo "raw_d1_full_5x3_reason=D7 fixed policy sealed; D8 broader train-derived confirmation"
  echo "dynamic_gpu=true free_gpu_max_used_mib=$FREE_GPU_MAX_USED_MIB free_gpu_max_util_pct=$FREE_GPU_MAX_UTIL_PCT"
  git -C "$WORK" branch --show-current || true
  git -C "$WORK" rev-parse --short HEAD || true
  git -C "$WORK" status --short || true
} | tee "$STATUS"

for required in "$PY" "$DATA" "$DEPTH" "$A0" "$SPLIT_JSON"; do
  if [[ ! -e "$required" ]]; then
    echo "DTA_V3_7_D8_MISSING_PATH $required $(date -Is)" | tee -a "$STATUS"
    exit 3
  fi
done

D1_LAUNCH=$EVID/launch_dta_v3_7_tau_training_convir4090.sh
D3_LAUNCH=$EVID/run_dta_v3_7_phase_d3_tau_real_blend_convir4090.sh
for script in "$D1_LAUNCH" "$D3_LAUNCH"; do
  if [[ ! -f "$script" ]]; then
    echo "DTA_V3_7_D8_MISSING_SCRIPT $script $(date -Is)" | tee -a "$STATUS"
    exit 3
  fi
  chmod +x "$script"
done

echo "dta_v3_7_phase_d8_stage1_train_candidates_start $(date -Is)" | tee -a "$STATUS"
DTA_V37_STAGE_SCREEN_ONLY=0 \
DTA_V37_RUN_BATCH_TAG="$RUN_TAG" \
VARIANTS_CSV="$VARIANTS_CSV" \
FOLDS_CSV="$FOLDS_CSV" \
SEEDS_CSV="$SEEDS_CSV" \
STAGE="$STAGE" \
MAX_PARALLEL="${MAX_PARALLEL_TRAIN:-}" \
FREE_GPU_MAX_USED_MIB="$FREE_GPU_MAX_USED_MIB" \
FREE_GPU_MAX_UTIL_PCT="$FREE_GPU_MAX_UTIL_PCT" \
WORK="$WORK" \
PY="$PY" \
bash "$D1_LAUNCH" 2>&1 | tee "$EVID/phase_d8_stage1_train_candidates.log"
echo "dta_v3_7_phase_d8_stage1_train_candidates_done $(date -Is)" | tee -a "$STATUS"

ACTION_TABLE=$EVID/v37_tau_oof_per_image_action_table.csv
if [[ ! -s "$ACTION_TABLE" ]]; then
  echo "DTA_V3_7_D8_MISSING_ACTION_TABLE $ACTION_TABLE $(date -Is)" | tee -a "$STATUS"
  exit 3
fi

echo "dta_v3_7_phase_d8_stage2_real_blend_start $(date -Is)" | tee -a "$STATUS"
WORK="$WORK" \
TAU_WORK="$WORK" \
PY="$PY" \
DATA="$DATA" \
A0="$A0" \
DEPTH="$DEPTH" \
SPLIT_JSON="$SPLIT_JSON" \
ACTION_TABLE="$ACTION_TABLE" \
FOLDS_CSV="$FOLDS_CSV" \
SEEDS_CSV="$SEEDS_CSV" \
STAGE="$STAGE" \
INCLUDE_RUN_SUBSTRING="$RUN_TAG" \
MAX_PARALLEL="${MAX_PARALLEL_RENDER:-}" \
FREE_GPU_MAX_USED_MIB="$FREE_GPU_MAX_USED_MIB" \
FREE_GPU_MAX_UTIL_PCT="$FREE_GPU_MAX_UTIL_PCT" \
bash "$D3_LAUNCH" 2>&1 | tee "$EVID/phase_d8_stage2_real_blend.log"
echo "dta_v3_7_phase_d8_stage2_real_blend_done $(date -Is)" | tee -a "$STATUS"

REAL_ACTIONS=$EVID/v37_tau_real_blend_single_actions_all.csv
if [[ ! -s "$REAL_ACTIONS" ]]; then
  echo "DTA_V3_7_D8_MISSING_REAL_ACTIONS $REAL_ACTIONS $(date -Is)" | tee -a "$STATUS"
  exit 3
fi

GROUP_DIR=$EVID/phase_d8_outputdiff_groups
mkdir -p "$GROUP_DIR"

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk 'NR==1 {print int($1)}'
}

pick_feature_gpu() {
  local best_gpu="" best_mem=999999 mem idx
  while IFS=, read -r idx _rest; do
    idx="${idx// /}"
    [[ -z "$idx" ]] && continue
    mem="$(gpu_used_mib "$idx" || echo 999999)"
    if [[ "$mem" -lt "$best_mem" && "$mem" -le "$FEATURE_FREE_GPU_MAX_USED_MIB" ]]; then
      best_gpu="$idx"
      best_mem="$mem"
    fi
  done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
  [[ -n "$best_gpu" ]] && printf '%s\n' "$best_gpu"
}

run_feature_group() {
  local fold="$1" seed="$2" gpu="$3"
  local log="$GROUP_DIR/d8_outputdiff_seed${seed}_f${fold}.log"
  echo "d8_outputdiff_group_start fold=$fold seed=$seed gpu=$gpu log=$log $(date -Is)" | tee -a "$STATUS"
  (
    cd "$WORK"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$WORK/Dehazing/ITS:$WORK:${PYTHONPATH:-}"
    "$PY" experience_docx/tools/extract_haze4k_dta_v37_outputdiff_features.py \
      --data_dir "$DATA" \
      --a0_checkpoint "$A0" \
      --checkpoint_root "$WORK" \
      --action_table_csv "$ACTION_TABLE" \
      --include_run_substring "$RUN_TAG" \
      --depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --fold "$fold" \
      --seed "$seed" \
      --output_dir "$GROUP_DIR" \
      --feature_max_side "$FEATURE_MAX_SIDE"
  ) >"$log" 2>&1
  local rc=$?
  echo "d8_outputdiff_group_done fold=$fold seed=$seed rc=$rc log=$log $(date -Is)" | tee -a "$STATUS"
  return "$rc"
}

IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
TASKS=()
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    TASKS+=("$fold,$seed")
  done
done

echo "dta_v3_7_phase_d8_stage3_outputdiff_start tasks=${#TASKS[@]} $(date -Is)" | tee -a "$STATUS"
pids=()
task_idx=0
while [[ "$task_idx" -lt "${#TASKS[@]}" ]]; do
  while [[ "${#pids[@]}" -ge "$MAX_PARALLEL_FEATURES" ]]; do
    next=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        next+=("$pid")
      else
        wait "$pid"
      fi
    done
    pids=("${next[@]}")
    sleep 10
  done
  gpu="$(pick_feature_gpu || true)"
  if [[ -z "${gpu:-}" ]]; then
    if [[ "${#pids[@]}" -gt 0 ]]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    else
      echo "d8_wait_no_feature_gpu free_threshold_mib=$FEATURE_FREE_GPU_MAX_USED_MIB $(date -Is)" | tee -a "$STATUS"
      sleep 60
    fi
    continue
  fi
  IFS=',' read -r fold seed <<< "${TASKS[$task_idx]}"
  task_idx=$((task_idx + 1))
  out="$GROUP_DIR/v37_d6_outputdiff_features_seed${seed}_f${fold}.csv"
  if [[ -s "$out" ]]; then
    echo "d8_outputdiff_group_skip_existing fold=$fold seed=$seed out=$out $(date -Is)" | tee -a "$STATUS"
    continue
  fi
  run_feature_group "$fold" "$seed" "$gpu" &
  pids+=("$!")
  sleep 15
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

COMBINED=$EVID/v37_d8_outputdiff_features_all.csv
"$PY" - <<PY
import csv
from pathlib import Path
group_dir = Path("$GROUP_DIR")
paths = sorted(group_dir.glob("v37_d6_outputdiff_features_seed*_f*.csv"))
expected = ${#TASKS[@]}
if len(paths) != expected:
    raise SystemExit(f"expected {expected} group csvs, found {len(paths)}: {paths}")
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
print(f"DTA_V3_7_D8_OUTPUTDIFF_COMBINE_OK files={len(paths)} rows={len(rows)} output=$COMBINED", flush=True)
PY
echo "dta_v3_7_phase_d8_stage3_outputdiff_done $(date -Is)" | tee -a "$STATUS"

echo "dta_v3_7_phase_d8_stage4_fixed_policy_start $(date -Is)" | tee -a "$STATUS"
export PYTHONPATH="$WORK/Dehazing/ITS:$WORK:${PYTHONPATH:-}"
"$PY" experience_docx/tools/confirm_haze4k_dta_v37_d7_fixed_outputdiff_policy.py \
  --single_actions_csv "$REAL_ACTIONS" \
  --feature_action_table_csv "$ACTION_TABLE" \
  --outputdiff_features_csv "$COMBINED" \
  --d6_aggregate_csv "$EVID/v37_d6_outputdiff_policy_aggregate.csv" \
  --output_dir "$EVID" \
  --output_prefix v37_d8_fixed_formal \
  --include_run_substring "$RUN_TAG" \
  --policy_ids "$PRIMARY_POLICY_ID" \
  --skip_d6_consistency \
  2>&1 | tee "$EVID/phase_d8_fixed_formal_policy.log"

echo "dta_v3_7_phase_d8_fixed_formal_done rc=0 $(date -Is)" | tee -a "$STATUS"
echo "DTA_V3_7_PHASE_D8_FIXED_FORMAL_OK $(date -Is)" | tee -a "$STATUS"
