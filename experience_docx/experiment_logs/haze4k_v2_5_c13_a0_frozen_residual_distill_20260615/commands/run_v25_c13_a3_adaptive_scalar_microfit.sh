#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615
C12_EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v25_c13_a0_frozen_residual_distill
STATUS=$EVID/status_c13_a3_adaptive_scalar_microfit.txt

mkdir -p "$EVID/runtime_logs" "$EVID/commands" "$RUNROOT"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_a3_adaptive_scalar_microfit"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "source_branch=$(git -C "$ROOT" branch --show-current)"
  echo "design=direct residual, zero head, adaptive scalar image gate, clamped output"
  echo "quick_gate=mean>=0.25 hard>=0.35 easy>=0.10 positive>=0.80 severe<=60/600 dSSIM>=0"
} > "$STATUS"

for path in "$PY" "$DATA" "$A0" "$CACHE" "$EVID/v25_c13_split_manifest.json" "$C12_EVID/v24_c12_teacher_cache_metrics.csv"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

{
  echo "state=C13_A3_MICROFIT_RUNNING"
  echo "microfit_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

pids=()
specs=(
  c13a3_adaptive025:0.25:1:rgb_wavelet:32:3:10
  c13a3_adaptive050:0.50:2:rgb_wavelet:32:3:10
)
for spec in "${specs[@]}"; do
  IFS=: read -r variant scale_init gpu feature width depth epochs <<< "$spec"
  out="$RUNROOT/$variant"
  if [ -e "$out" ]; then
    echo "REFUSE_EXISTING_OUTPUT variant=$variant out=$out" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 3
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/train_haze4k_v25_c13_residual_adapter.py" \
      --variant "$variant" \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --teacher-dir "$CACHE" \
      --teacher-metrics "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
      --a0-checkpoint "$A0" \
      --out-dir "$out" \
      --feature-mode "$feature" \
      --adapter-width "$width" \
      --adapter-depth "$depth" \
      --bootstrap-scale 0.0 \
      --residual-mode adaptive_scalar \
      --residual-scale 1.0 \
      --scale-init "$scale_init" \
      --head-init zero \
      --clamp-output \
      --seed 3407 \
      --epochs "$epochs" \
      --batch-size 1 \
      --num-workers 2 \
      --crop-size 0 \
      --max-images 256 \
      --keep-partial-batch \
      --learning-rate 1e-4 \
      --weight-decay 0 \
      --teacher-margin 0.70 \
      --gt-weight 0.50 \
      --teacher-weight 0.50 \
      --preserve-weight 2.50 \
      --freq-weight 0.05 \
      --color-weight 0.05 \
      --tv-weight 0.03 \
      --raw-weight 0.001 \
      --grad-clip-norm 1.0 \
      --print-freq 20
  ) > "$EVID/runtime_logs/train_${variant}.log" 2>&1 &
  pids+=("$!")
  echo "launched_a3_microfit variant=$variant gpu=$gpu scale_init=$scale_init pid=${pids[-1]}" | tee -a "$STATUS"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
if [ "$rc" -ne 0 ]; then
  echo "C13_A3_MICROFIT_FAILED" | tee -a "$STATUS"
  echo "state=FAILED_COMMAND" >> "$STATUS"
  exit "$rc"
fi

for spec in "${specs[@]}"; do
  IFS=: read -r variant scale_init gpu feature width depth epochs <<< "$spec"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
      --variant "$variant" \
      --checkpoint Best \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --a0-checkpoint "$A0" \
      --student-checkpoint "$RUNROOT/$variant/checkpoints/Best.pkl" \
      --out-dir "$EVID" \
      --feature-mode "$feature" \
      --adapter-width "$width" \
      --adapter-depth "$depth" \
      --bootstrap-scale 0.0 \
      --residual-mode adaptive_scalar \
      --residual-scale 1.0 \
      --scale-init "$scale_init" \
      --head-init zero \
      --clamp-output \
      --max-train 256 \
      --max-val 128
  ) > "$EVID/runtime_logs/eval_${variant}_Best.log" 2>&1
  echo "eval_a3_quick_done variant=$variant" | tee -a "$STATUS"
done

"$PY" - "$EVID" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

evid = Path(sys.argv[1])
variants = ["c13a3_adaptive025", "c13a3_adaptive050"]
rows = []
passes = []
for variant in variants:
    path = evid / f"v25_c13_eval_{variant}_Best_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    val = payload["val_summary"]
    row = {
        "variant": variant,
        "eval_scope": "quick_val128",
        "mean_dPSNR": val["mean_dPSNR"],
        "hard_bottom25_dPSNR": val["hard_bottom25_dPSNR"],
        "easy_top25_dPSNR": val["easy_top25_dPSNR"],
        "dSSIM": val["dSSIM"],
        "positive_ratio": val["positive_ratio"],
        "severe_loss_per_600": val["severe_loss_per_600"],
    }
    row["quick_gate_pass"] = (
        row["mean_dPSNR"] >= 0.25
        and row["hard_bottom25_dPSNR"] >= 0.35
        and row["easy_top25_dPSNR"] >= 0.10
        and row["positive_ratio"] >= 0.80
        and row["severe_loss_per_600"] <= 60.0
        and row["dSSIM"] >= 0.0
    )
    rows.append(row)
    if row["quick_gate_pass"]:
        passes.append(variant)

leader = evid / "v25_c13_a3_adaptive_scalar_microfit_leaderboard.csv"
with leader.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

decision = {
    "route": "Haze4K v2.5 C13 A3 adaptive scalar microfit",
    "locked_test_touched": False,
    "locked_per_image_read": False,
    "quick_gate_pass_variants": passes,
    "quick_gate_pass_count": len(passes),
    "decision": "C13_A3_QUICK_PASS_RUN_FULL_VAL" if passes else "C13_A3_QUICK_FAIL_STOP_OR_REDESIGN",
    "rows": rows,
}
(evid / "v25_c13_a3_adaptive_scalar_microfit_decision.json").write_text(
    json.dumps(decision, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("C13_A3_QUICK_DECISION_OK", json.dumps(decision, sort_keys=True))
PY

pass_variants=$("$PY" - "$EVID/v25_c13_a3_adaptive_scalar_microfit_decision.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(" ".join(payload.get("quick_gate_pass_variants", [])))
PY
)

if [ -n "$pass_variants" ]; then
  echo "a3_full_val_start variants=$pass_variants" | tee -a "$STATUS"
  for variant in $pass_variants; do
    scale_init=0.25
    gpu=1
    if [ "$variant" = "c13a3_adaptive050" ]; then
      scale_init=0.50
      gpu=2
    fi
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
        --variant "${variant}_fullval" \
        --checkpoint Best \
        --convir-dir "$ROOT/Dehazing/ITS" \
        --data-dir "$DATA" \
        --split-manifest "$EVID/v25_c13_split_manifest.json" \
        --a0-checkpoint "$A0" \
        --student-checkpoint "$RUNROOT/$variant/checkpoints/Best.pkl" \
        --out-dir "$EVID" \
        --feature-mode rgb_wavelet \
        --adapter-width 32 \
        --adapter-depth 3 \
        --bootstrap-scale 0.0 \
        --residual-mode adaptive_scalar \
        --residual-scale 1.0 \
        --scale-init "$scale_init" \
        --head-init zero \
        --clamp-output \
        --max-train 256 \
        --max-val 0
    ) > "$EVID/runtime_logs/eval_${variant}_fullval_Best.log" 2>&1
    echo "eval_a3_full_val_done variant=$variant" | tee -a "$STATUS"
  done
fi

{
  echo "finish_time=$(date --iso-8601=seconds)"
  if [ -n "$pass_variants" ]; then
    echo "state=C13_A3_MICROFIT_DONE_FULL_VAL_REVIEW_BEFORE_B_SCREEN"
  else
    echo "state=C13_A3_MICROFIT_FAIL_STOP_OR_REDESIGN"
  fi
  echo "C13_A3_ADAPTIVE_SCALAR_MICROFIT_OK"
} >> "$STATUS"
