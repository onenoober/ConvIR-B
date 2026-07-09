#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=${WORK:-$BASE/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign-f4-044b7798}
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$WORK/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709
STATUS=$EVID/status.txt
LOG=$EVID/v2f_f4b_tail_rescue_matrix.log

SPLIT=$BASE/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
V2_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json
V2B_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json
D3=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
exec > >(tee -a "$LOG") 2>&1

echo "F4B_TAIL_RESCUE_MATRIX_START $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "work=$WORK" | tee -a "$STATUS"
echo "python=$PY" | tee -a "$STATUS"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
echo "locked_haze4k_test_usage=none" | tee -a "$STATUS"
echo "D2=not_run RARM=not_connected_or_trained v3=not_run" | tee -a "$STATUS"

for path in "$PY" "$DATA" "$A0" "$SPLIT" "$V2_THRESH" "$V2B_THRESH" "$D3" "$WORK/experience_docx/tools/run_chd_rm_v2f_stratified_head_canary.py"; do
  if [[ ! -e "$path" ]]; then
    echo "PREFLIGHT_FAILED missing_path=$path" | tee -a "$STATUS"
    exit 2
  fi
done

cd "$WORK"
BRANCH=$(git branch --show-current)
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "branch=$BRANCH" | tee -a "$STATUS"
echo "source_commit=$SOURCE_COMMIT" | tee -a "$STATUS"

cat > "$EVID/f4b_authorization_record.md" <<EOF
# v2f F4b Tail-Rescue Authorization

Status: authorized as a narrow frozen-side diagnostic after F4 completed with
\`COMPLETED_GATE_FAIL\`.

Purpose: test whether the F4 failure was caused by insufficient low-density
hard-negative tail pressure rather than by a deeper target/head separability
limit.

Allowed:
- frozen ConvIR-B A0
- frozen D3 density
- frozen-side stratified R_need heads only
- train_inner calibration and val_inner evaluation
- original v2e global safety/LDHN gate

Forbidden:
- D2
- v3
- RARM connection or training
- locked Haze4K test
- relaxing false-tail or LDHN gates

If no F4b spec finds a selected gate pass or a safe+LDHN threshold point, keep
v2f paused and do not run v3/RARM. If a candidate point appears, run F5 controls
before any v3 no-op audit.
EOF

run_spec() {
  local name="$1"
  shift
  local out="$EVID/$name"
  mkdir -p "$out"
  echo "F4B_SPEC_START name=$name $(date --iso-8601=seconds)" | tee -a "$STATUS"
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v2f_stratified_head_canary.py \
    --data_dir "$DATA" \
    --checkpoint "$A0" \
    --split_json "$SPLIT" \
    --v2_thresholds "$V2_THRESH" \
    --v2b_thresholds "$V2B_THRESH" \
    --density_artifact "$D3" \
    --output_dir "$out" \
    --source_commit "$SOURCE_COMMIT" \
    --batch_size 8 \
    --density_bins 5 \
    --metric_sample_size 64 \
    --fit_grid 64 \
    --variants f4_global_strat_control f4_excess_strat_ldhn \
    "$@" \
    2>&1 | tee "$out/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "F4B_SPEC_DONE name=$name rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$rc" -ne 0 ]]; then
    echo "F4B_SPEC_FAILED name=$name" | tee -a "$STATUS"
    exit "$rc"
  fi
}

run_spec tail2_topk10 \
  --epochs 6 \
  --hn_weight 2.50 \
  --topk_fraction 0.10 \
  --tau_neg 0.45 \
  --pos_weight 0.80 \
  --pair_weight 0.30 \
  --head_hidden 96 \
  --density_temperature 0.08

run_spec tail4_topk20 \
  --epochs 6 \
  --hn_weight 4.00 \
  --topk_fraction 0.20 \
  --tau_neg 0.40 \
  --pos_weight 0.60 \
  --pair_weight 0.20 \
  --head_hidden 96 \
  --density_temperature 0.08

run_spec tail3_cap128_temp04 \
  --epochs 8 \
  --hn_weight 3.00 \
  --topk_fraction 0.15 \
  --tau_neg 0.45 \
  --pos_weight 0.80 \
  --pair_weight 0.35 \
  --head_hidden 128 \
  --density_temperature 0.04

"$PY" - "$EVID" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/stratified_head_ablation_summary.csv")):
    spec = summary_path.parent.name
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["spec"] = spec
            rows.append(row)

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with (root / "f4b_tail_rescue_matrix_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def as_float(row, key, default=math.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

global_rows = [r for r in rows if r.get("target_contract") == "global_v2e_contract"]
gate_pass = [
    r for r in global_rows
    if str(r.get("selected_gate_pass", "")).lower() == "true"
]
safe_ldhn = [
    r for r in global_rows
    if as_float(r, "safe_and_ldhn_points", 0.0) > 0
]
best_safe_ldhn = max((as_float(r, "best_safe_ldhn_recall", 0.0) for r in global_rows), default=math.nan)
best_selected_ldhn = max((as_float(r, "selected_ldhn_recall", 0.0) for r in global_rows), default=math.nan)
min_first_ldhn_p95 = min((as_float(r, "first_ldhn_false_p95", math.inf) for r in global_rows), default=math.nan)

if gate_pass:
    status = "COMPLETED_GATE_PASS_NEEDS_F5_CONTROLS"
elif safe_ldhn:
    status = "COMPLETED_SAFE_LDHN_POINT_NEEDS_F5_CONTROLS"
else:
    status = "COMPLETED_GATE_FAIL"

closeout = {
    "status": status,
    "phase": "F4B_TAIL_RESCUE_MATRIX",
    "spec_count": len({r["spec"] for r in rows}),
    "global_contract_rows": len(global_rows),
    "selected_gate_pass_any_variant": bool(gate_pass),
    "safe_and_ldhn_point_any_variant": bool(safe_ldhn),
    "best_safe_ldhn_recall": best_safe_ldhn,
    "best_selected_ldhn_recall": best_selected_ldhn,
    "min_first_ldhn_false_p95": min_first_ldhn_p95,
    "locked_haze4k_test_usage": "none",
    "D2": "not_run",
    "RARM": "not_connected_or_trained",
    "v3": "not_run",
    "ConvIR_B": "frozen",
    "D3_density": "frozen",
    "next_gate": "Run F5 controls only if a selected pass or safe+LDHN point exists; otherwise keep v2f paused and do not run v3/RARM.",
}
(root / "v2f_f4b_tail_rescue_closeout.json").write_text(json.dumps(closeout, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# v2f F4b Tail-Rescue Matrix Summary",
    "",
    f"Status: `{status}`",
    "",
    "Policy: ConvIR-B and D3 frozen; D2/v3/RARM not run; locked Haze4K test not used.",
    "",
    "| Spec | Variant | Gate | Spearman | AUROC | AUPRC | Coverage | False p95 | LDHN recall | Safe+LDHN points |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in global_rows:
    lines.append(
        f"| {row['spec']} | {row['variant']} | {row['selected_gate_pass']} | "
        f"{as_float(row, 'selected_spearman'):.4f} | {as_float(row, 'selected_auroc'):.4f} | "
        f"{as_float(row, 'selected_auprc'):.4f} | {as_float(row, 'selected_coverage'):.4f} | "
        f"{as_float(row, 'selected_false_p95'):.4f} | {as_float(row, 'selected_ldhn_recall'):.4f} | "
        f"{as_float(row, 'safe_and_ldhn_points', 0.0):.0f} |"
    )
lines.append("")
lines.append("Decision: keep the original v2e global safety/LDHN gate as the primary contract.")
(root / "v2f_f4b_tail_rescue_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(closeout, indent=2, sort_keys=True))
PY

echo "F4B_TAIL_RESCUE_MATRIX_DONE $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "V2F_F4B_TAIL_RESCUE_MATRIX_OK" | tee -a "$STATUS"
