#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$(pwd)}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d9_locked_fixed_policy.txt
LOG=$EVID/phase_d9_locked_fixed_policy.log
GROUP_DIR=$EVID/phase_d9_locked_groups
ACTION_TABLE=$EVID/v37_tau_oof_per_image_action_table.csv
TRAIN_ACTIONS=$EVID/v37_tau_real_blend_single_actions_all.csv
TRAIN_OUTPUTDIFF=$EVID/v37_d8_outputdiff_features_all.csv
PRIMARY_POLICY_ID=primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100
OUTER_GROUPS=${OUTER_GROUPS:-0:3407,0:3411,1:3407,1:3411}
GPU_LIST=${GPU_LIST:-1,2,3,4}
FEATURE_MAX_SIDE=${FEATURE_MAX_SIDE:-384}
MAX_IMAGES=${MAX_IMAGES:-0}
INCLUDE_RUN_SUBSTRING=${INCLUDE_RUN_SUBSTRING:-d8formal}
D9_RUNNER_SOURCE_COMMIT=${D9_RUNNER_SOURCE_COMMIT:-unknown}

mkdir -p "$EVID" "$GROUP_DIR"
{
  echo "dta_v3_7_phase_d9_locked_fixed_policy_start $(date -Is)"
  echo "state=RUNNING_LOCKED_ONE_SHOT_CONFIRMATION"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "sealed_policy_id=$PRIMARY_POLICY_ID"
  echo "outer_groups=$OUTER_GROUPS"
  echo "gpu_list=$GPU_LIST"
  echo "locked_test_touched=true"
  echo "one_shot_locked_confirmation=true"
  echo "post_test_tuning_allowed=false"
  echo "d9_runner_source_commit=$D9_RUNNER_SOURCE_COMMIT"
  branch="$(git -C "$WORK" branch --show-current 2>/dev/null || true)"
  head_commit="$(git -C "$WORK" rev-parse --short HEAD 2>/dev/null || true)"
  echo "git_branch=${branch:-unknown}"
  echo "git_head=${head_commit:-unknown}"
  echo "git_status_short_untracked_no_first80_begin"
  git -C "$WORK" status --short --untracked-files=no 2>/dev/null | head -n 80 || true
  echo "git_status_short_untracked_no_first80_end"
  echo "D9_STATUS_HEADER_OK $(date -Is)"
} | tee "$STATUS"

for required in "$PY" "$DATA" "$DEPTH" "$A0" "$ACTION_TABLE" "$TRAIN_ACTIONS" "$TRAIN_OUTPUTDIFF"; do
  if [[ ! -e "$required" ]]; then
    echo "DTA_V3_7_D9_MISSING_PATH $required $(date -Is)" | tee -a "$STATUS"
    exit 3
  fi
done

if [[ -s "$EVID/v37_d9_locked_fixed_policy_summary.json" ]]; then
  echo "DTA_V3_7_D9_OUTPUT_EXISTS $EVID/v37_d9_locked_fixed_policy_summary.json $(date -Is)" | tee -a "$STATUS"
  exit 4
fi

export PYTHONPATH="$WORK/Dehazing/ITS:$WORK:${PYTHONPATH:-}"
"$PY" -m py_compile "$WORK/experience_docx/tools/eval_haze4k_dta_v37_locked_fixed_policy.py"
echo "DTA_V3_7_D9_PREFLIGHT_PY_COMPILE_OK $(date -Is)" | tee -a "$STATUS"

IFS=',' read -r -a OUTER_GROUP_ITEMS <<< "$OUTER_GROUPS"
IFS=',' read -r -a GPU_ITEMS <<< "$GPU_LIST"
echo "DTA_V3_7_D9_GROUP_PARSE_OK groups=${#OUTER_GROUP_ITEMS[@]} gpus=${#GPU_ITEMS[@]} $(date -Is)" | tee -a "$STATUS"
if [[ "${#GPU_ITEMS[@]}" -lt "${#OUTER_GROUP_ITEMS[@]}" ]]; then
  echo "DTA_V3_7_D9_GPU_LIST_TOO_SHORT groups=${#OUTER_GROUP_ITEMS[@]} gpus=${#GPU_ITEMS[@]} $(date -Is)" | tee -a "$STATUS"
  exit 3
fi

pids=()
for idx in "${!OUTER_GROUP_ITEMS[@]}"; do
  group="${OUTER_GROUP_ITEMS[$idx]}"
  gpu="${GPU_ITEMS[$idx]}"
  fold="${group%%:*}"
  seed="${group##*:}"
  prefix="v37_d9_locked_fixed_policy_f${fold}_s${seed}"
  group_log="$GROUP_DIR/${prefix}.log"
  echo "d9_locked_group_start group=$group gpu=$gpu log=$group_log $(date -Is)" | tee -a "$STATUS"
  (
    cd "$WORK"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" experience_docx/tools/eval_haze4k_dta_v37_locked_fixed_policy.py \
      --data_dir "$DATA" \
      --a0_checkpoint "$A0" \
      --checkpoint_root "$WORK" \
      --feature_action_table_csv "$ACTION_TABLE" \
      --train_single_actions_csv "$TRAIN_ACTIONS" \
      --train_outputdiff_features_csv "$TRAIN_OUTPUTDIFF" \
      --include_run_substring "$INCLUDE_RUN_SUBSTRING" \
      --depth_cache_dir "$DEPTH" \
      --outer_groups "$group" \
      --output_dir "$GROUP_DIR" \
      --output_prefix "$prefix" \
      --feature_group outputdiff_plus_Q \
      --action_bank micro_shrink \
      --score_mode pred_gain \
      --target_intervention 1.0 \
      --feature_max_side "$FEATURE_MAX_SIDE" \
      --max_images "$MAX_IMAGES"
  ) >"$group_log" 2>&1 &
  pids+=("$!")
  sleep 10
done

rc=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  group="${OUTER_GROUP_ITEMS[$idx]}"
  if wait "$pid"; then
    echo "d9_locked_group_done group=$group rc=0 $(date -Is)" | tee -a "$STATUS"
  else
    group_rc=$?
    echo "d9_locked_group_done group=$group rc=$group_rc $(date -Is)" | tee -a "$STATUS"
    rc=$group_rc
  fi
done
if [[ "$rc" -ne 0 ]]; then
  echo "DTA_V3_7_D9_LOCKED_GROUP_FAILED rc=$rc $(date -Is)" | tee -a "$STATUS"
  exit "$rc"
fi

"$PY" - <<PY | tee "$LOG"
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from experience_docx.tools.eval_haze4k_dta_v37_locked_fixed_policy import add_locked_metric_aliases
from experience_docx.tools.eval_haze4k_dta_v37_tau_real_blend_oracle import finite_float, write_csv
from experience_docx.tools.train_haze4k_dta_v37_d3_high_positive_policy import STRICT_GATES, gate_checks, summarize

evid = Path("$EVID")
group_dir = Path("$GROUP_DIR")
groups = [item.strip() for item in "$OUTER_GROUPS".split(",") if item.strip()]
selected = []
per_outer = []
for group in groups:
    fold, seed = group.split(":", 1)
    prefix = f"v37_d9_locked_fixed_policy_f{fold}_s{seed}"
    selected_path = group_dir / f"{prefix}_selected_actions.csv"
    summary_path = group_dir / f"{prefix}_summary.json"
    if not selected_path.is_file() or not summary_path.is_file():
        raise SystemExit(f"missing group output for {group}")
    with selected_path.open(newline="", encoding="utf-8") as handle:
        selected.extend(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_outer.extend(summary.get("per_outer", []))

aggregate = add_locked_metric_aliases(summarize(selected), selected)
aggregate.update({
    "policy_id": "$PRIMARY_POLICY_ID",
    "feature_group": "outputdiff_plus_Q",
    "action_bank": "micro_shrink",
    "score_mode": "pred_gain",
    "target_intervention": 1.0,
    "outer_groups": len(groups),
    "selected_rows": len(selected),
    "mean_alpha": sum(finite_float(row.get("alpha")) for row in selected) / max(1, len(selected)),
    "intervention_rate": sum(row.get("variant") != "A0" for row in selected) / max(1, len(selected)),
    "locked_test_touched": True,
    "one_shot_locked_confirmation": True,
    "post_test_tuning_allowed": False,
})
aggregate["strict_gate_checks"] = gate_checks(aggregate)
aggregate["strict_gate_pass"] = all(aggregate["strict_gate_checks"].values())
aggregate["decision"] = "D9_LOCKED_FIXED_POLICY_PASS" if aggregate["strict_gate_pass"] else "D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING"

for row in per_outer:
    row["policy_id"] = "$PRIMARY_POLICY_ID"

selected_out = evid / "v37_d9_locked_fixed_policy_selected_actions.csv"
aggregate_out = evid / "v37_d9_locked_fixed_policy_aggregate.csv"
per_outer_out = evid / "v37_d9_locked_fixed_policy_per_outer.csv"
summary_out = evid / "v37_d9_locked_fixed_policy_summary.json"
write_csv(selected_out, selected)
write_csv(aggregate_out, [aggregate])
write_csv(per_outer_out, per_outer)
summary = {
    "route": "DTA-v3.7 U-TQS-Mix",
    "phase": "D9_one_shot_locked_fixed_policy_confirmation",
    "policy_id": "$PRIMARY_POLICY_ID",
    "decision": aggregate["decision"],
    "aggregate": aggregate,
    "per_outer": per_outer,
    "outer_groups": groups,
    "selected_actions_csv": str(selected_out),
    "aggregate_csv": str(aggregate_out),
    "per_outer_csv": str(per_outer_out),
    "strict_gates": STRICT_GATES,
    "locked_test_touched": True,
    "one_shot_locked_confirmation": True,
    "post_test_tuning_allowed": False,
    "combined_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(
    "DTA_V3_7_D9_LOCKED_FIXED_POLICY_OK "
    f"decision={aggregate['decision']} rows={len(selected)} images={aggregate['test_image_count']} "
    f"mean={aggregate['mean_dPSNR']:.6f} hard={aggregate['hard_bottom25_dPSNR']:.6f} "
    f"positive={aggregate['positive_ratio']:.6f} worst_per_600={aggregate['worst_per_600']:.2f}",
    flush=True,
)
PY

echo "dta_v3_7_phase_d9_locked_fixed_policy_done rc=0 $(date -Is)" | tee -a "$STATUS"
echo "DTA_V3_7_PHASE_D9_LOCKED_FIXED_POLICY_OK $(date -Is)" | tee -a "$STATUS"
