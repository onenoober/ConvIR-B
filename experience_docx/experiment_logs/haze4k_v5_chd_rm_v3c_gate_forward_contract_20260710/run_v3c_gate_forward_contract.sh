#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3c-gate-forward-contract}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3c_gate_forward_contract_20260710"

mkdir -p "$OUT"
cd "$ROOT"

PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v3c_gate_forward_contract.py \
  --checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$OUT" \
  --source_split train \
  --split_key val_inner \
  --max_samples 16 \
  --progress_every 8 \
  2>&1 | tee "$OUT/v3c_gate_forward_contract.log"

echo V3C_GATE_FORWARD_CONTRACT_SCRIPT_OK | tee "$OUT/status.txt"
