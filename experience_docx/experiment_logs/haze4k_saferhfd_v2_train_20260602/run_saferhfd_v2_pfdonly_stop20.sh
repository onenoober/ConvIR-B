#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-saferhfd-v2-train}"
ITS_ROOT="$ROOT/Dehazing/ITS"
PY="${PY:-/root/miniconda3/envs/convir-cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}"
PRETRAIN="${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}"
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_saferhfd_v2_train_20260602"
STATUS="$LOG_DIR/status.txt"

MODEL_NAME="ConvIR-Haze4K-B1v2-SafeRHFD-pfdonly-stop20-seed3407-20260602"
TAG="saferhfd_v2_pfdonly_stop20_seed3407_vs_a0"
CANDIDATE_NAME="saferhfd_v2_pfdonly"
BEST="$ITS_ROOT/results/$MODEL_NAME/Training-Results/Best.pkl"

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start root=$ROOT data_dir=$DATA_DIR"

"$PY" - <<'PY' > "$LOG_DIR/preflight_saferhfd_v2.json"
import json
import contextlib
import io
import torch
from argparse import Namespace
from models.PFDConvIR import build_pfd_net
from train import _configure_pfd_train_scope

torch.manual_seed(3407)
base = build_pfd_net("small", "Haze4K", pfd_safe_rhfd=0).eval()
torch.manual_seed(3407)
safe = build_pfd_net("small", "Haze4K", pfd_safe_rhfd=1).eval()
x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    base_out = base(x)
    safe_out = safe(x)
max_abs_diff = max((a - b).abs().max().item() for a, b in zip(base_out, safe_out))

train_model = build_pfd_net("base", "Haze4K", pfd_safe_rhfd=1)
args = Namespace(arch="pfd", pfd_train_scope="pfd_only")
with contextlib.redirect_stdout(io.StringIO()):
    trainable_params = _configure_pfd_train_scope(train_model, args)
print(json.dumps({
    "safe_zero_init_max_abs_diff": max_abs_diff,
    "safe_zero_init_pass": max_abs_diff < 1e-6,
    "trainable_param_count": sum(p.numel() for p in trainable_params),
    "total_param_count": sum(p.numel() for p in train_model.parameters()),
    "stat_keys": sorted(safe.collect_pfd_stats(x).keys()),
}, indent=2))
if max_abs_diff >= 1e-6:
    raise SystemExit(2)
PY
log_status "preflight_pass"

if [[ -f "$BEST" ]]; then
  log_status "skip_existing_train $MODEL_NAME"
else
  log_status "train_start $MODEL_NAME"
  "$PY" main.py \
    --mode train \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    --init_model "$PRETRAIN" \
    --arch pfd \
    --pfd_rhfd 0 \
    --pfd_hscm 0 \
    --pfd_pffb 0 \
    --pfd_teacher 0 \
    --pfd_safe_rhfd 1 \
    --pfd_safe_rhfd_gate_max 1.0 \
    --pfd_safe_rhfd_norm_cap 0.0035 \
    --pfd_safe_rhfd_lowpass_ratio 0.20 \
    --pfd_train_scope pfd_only \
    --mod_stats_freq 1 \
    --mod_stats_batches 64 \
    > "$LOG_DIR/train_${MODEL_NAME}.log" 2>&1
  log_status "train_done $MODEL_NAME rc=$?"
fi

log_status "eval_start $TAG"
"$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
  --data_dir "$DATA_DIR" \
  --original_checkpoint "$PRETRAIN" \
  --original_arch convir \
  --original_mode original \
  --original_name a0 \
  --candidate_checkpoint "$BEST" \
  --candidate_arch pfd \
  --candidate_mode original \
  --candidate_name "$CANDIDATE_NAME" \
  --candidate_pfd_rhfd 0 \
  --candidate_pfd_hscm 0 \
  --candidate_pfd_pffb 0 \
  --candidate_pfd_pffb_high 0 \
  --candidate_pfd_teacher 0 \
  --candidate_pfd_safe_rhfd 1 \
  --candidate_pfd_safe_rhfd_gate_max 1.0 \
  --candidate_pfd_safe_rhfd_norm_cap 0.0035 \
  --candidate_pfd_safe_rhfd_lowpass_ratio 0.20 \
  --output_dir "$LOG_DIR" \
  --tag "$TAG"

"$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
  --csv "$LOG_DIR/scout_eval_per_image_${TAG}.csv" \
  --candidate_name "$CANDIDATE_NAME" \
  --output "$LOG_DIR/scout_eval_bucket_analysis_${TAG}.json"

if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_saferhfd_v2.py" \
  --compare_json "$LOG_DIR/scout_eval_compare_${TAG}.json" \
  --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${TAG}.json" \
  --output "$LOG_DIR/gate_${TAG}.json"; then
  log_status "gate_pass $TAG"
else
  log_status "gate_fail $TAG"
fi

log_status "complete"
