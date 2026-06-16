#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v25_c13_a0_frozen_residual_distill
STATUS=$EVID/status_c13_a5_a4_scale_sweep.txt
BASE_CKPT=$RUNROOT/c13a4_scale050/checkpoints/Best.pkl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
SOURCE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)

mkdir -p "$EVID/runtime_logs"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_a5_a4_scale_sweep"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$SOURCE_COMMIT"
  echo "source_branch=$SOURCE_BRANCH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "base_checkpoint=$BASE_CKPT"
  echo "design=post-hoc residual scale sweep on best A4 checkpoint"
  echo "quick_gate=mean>=0.25 hard>=0.35 easy>=0.10 positive>=0.80 severe<=60/600 dSSIM>=0"
} > "$STATUS"

for path in "$PY" "$A0" "$BASE_CKPT" "$EVID/v25_c13_split_manifest.json"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

{
  echo "state=C13_A5_SCALE_SWEEP_RUNNING"
  echo "sweep_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

tags=(
  s025:0.25
  s030:0.30
  s035:0.35
  s040:0.40
  s045:0.45
)
for spec in "${tags[@]}"; do
  IFS=: read -r tag scale <<< "$spec"
  (
    "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
      --variant "a5_a4sweep_${tag}" \
      --checkpoint Best \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir /sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --a0-checkpoint "$A0" \
      --student-checkpoint "$BASE_CKPT" \
      --out-dir "$EVID" \
      --feature-mode rgb_wavelet \
      --adapter-width 32 \
      --adapter-depth 3 \
      --bootstrap-scale 0.0 \
      --residual-mode direct \
      --residual-scale "$scale" \
      --head-init zero \
      --clamp-output \
      --max-train 256 \
      --max-val 128
  ) > "$EVID/runtime_logs/eval_a5_a4sweep_${tag}.log" 2>&1
  echo "eval_a5_done tag=$tag scale=$scale" | tee -a "$STATUS"
done

"$PY" - "$EVID" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

evid = Path(sys.argv[1])
tags = ["s025", "s030", "s035", "s040", "s045"]
rows = []
passes = []
for tag in tags:
    path = evid / f"v25_c13_eval_a5_a4sweep_{tag}_Best_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    val = payload["val_summary"]
    row = {
        "tag": tag,
        "variant": f"a5_a4sweep_{tag}",
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
        passes.append(tag)

leader = evid / "v25_c13_a5_a4_scale_sweep_leaderboard.csv"
with leader.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

decision = {
    "route": "Haze4K v2.5 C13 A5 A4 scale sweep",
    "locked_test_touched": False,
    "locked_per_image_read": False,
    "quick_gate_pass_tags": passes,
    "quick_gate_pass_count": len(passes),
    "decision": "C13_A5_SCALE_SWEEP_OK_START_FULL_VAL" if passes else "C13_A5_SCALE_SWEEP_FAIL_STOP_OR_REDESIGN",
    "rows": rows,
}
(evid / "v25_c13_a5_a4_scale_sweep_decision.json").write_text(
    json.dumps(decision, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("C13_A5_SCALE_SWEEP_OK", json.dumps(decision, sort_keys=True))
PY

pass_tags=$("$PY" - "$EVID/v25_c13_a5_a4_scale_sweep_decision.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(" ".join(payload.get("quick_gate_pass_tags", [])))
PY
)

if [ -n "$pass_tags" ]; then
  echo "a5_full_val_start tags=$pass_tags" | tee -a "$STATUS"
  for tag in $pass_tags; do
    scale=0.25
    case "$tag" in
      s025) scale=0.25 ;;
      s030) scale=0.30 ;;
      s035) scale=0.35 ;;
      s040) scale=0.40 ;;
      s045) scale=0.45 ;;
    esac
    "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
      --variant "a5_a4sweep_${tag}_fullval" \
      --checkpoint Best \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir /sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --a0-checkpoint "$A0" \
      --student-checkpoint "$BASE_CKPT" \
      --out-dir "$EVID" \
      --feature-mode rgb_wavelet \
      --adapter-width 32 \
      --adapter-depth 3 \
      --bootstrap-scale 0.0 \
      --residual-mode direct \
      --residual-scale "$scale" \
      --head-init zero \
      --clamp-output \
      --max-train 256 \
      --max-val 0 > "$EVID/runtime_logs/eval_a5_a4sweep_${tag}_fullval.log" 2>&1
    echo "eval_a5_full_val_done tag=$tag" | tee -a "$STATUS"
  done
fi

{
  echo "finish_time=$(date --iso-8601=seconds)"
  if [ -n "$pass_tags" ]; then
    echo "state=C13_A5_SCALE_SWEEP_DONE_FULL_VAL_REVIEW_BEFORE_B_SCREEN"
  else
    echo "state=C13_A5_SCALE_SWEEP_FAIL_STOP_OR_REDESIGN"
  fi
  echo "C13_A5_A4_SCALE_SWEEP_OK"
} >> "$STATUS"
