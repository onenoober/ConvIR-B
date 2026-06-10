#!/usr/bin/env bash
set -euo pipefail
BASE=/data/caozhiyang/ConvIR-B
WORKTREE="$BASE/repos/ConvIR-B-official-arch-anchor"
ITS="$WORKTREE/Dehazing/ITS"
PY="$BASE/envs/convir-cu128/bin/python"
DATA="$BASE/datasets/Haze4K/Haze4K"
CKPT="$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
EVID="$WORKTREE/experience_docx/experiment_logs/haze4k_official_arch_anchor_convir5090_train_test_flow_20260610"
MODEL="ConvIR-Haze4K-official-anchor-convir5090-flow-20260610"
STATUS="$EVID/status.txt"
OFFICIAL_TEST_LOG="$EVID/official_pretrained_test.log"
TRAIN_LOG="$EVID/train_smoke_from_pretrained_1epoch.log"
SMOKE_TEST_LOG="$EVID/smoke_best_test.log"
SUMMARY_JSON="$EVID/train_test_flow_summary.json"
export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
mkdir -p "$EVID"
{
  echo "flow_start haze4k_official_anchor_convir5090_train_test $(date --iso-8601=seconds)"
  echo "worktree=$WORKTREE"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "model=$MODEL"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "torch_force_no_weights_only_load=$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
  echo "locked_test_policy=official_pretrained_and_flow_test_only_no_model_selection"
} | tee -a "$STATUS"
cd "$ITS"
"$PY" - <<'PY' | tee "$EVID/environment_probe.txt"
import json, os, subprocess, sys, torch
from pathlib import Path
base = Path('/data/caozhiyang/ConvIR-B')
ckpt = base / 'checkpoints/official/Haze4K/haze4k-base.pkl'
data = base / 'datasets/Haze4K/Haze4K'
info = {
    'host': subprocess.getoutput('hostname'),
    'python': sys.executable,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'cuda_device0': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'data': str(data),
    'train_haze_count': len(list((data/'train'/'haze').glob('*'))),
    'train_gt_count': len(list((data/'train'/'gt').glob('*'))),
    'test_haze_count': len(list((data/'test'/'haze').glob('*'))),
    'test_gt_count': len(list((data/'test'/'gt').glob('*'))),
    'checkpoint': str(ckpt),
}
print(json.dumps(info, indent=2, sort_keys=True))
PY

echo "official_pretrained_test_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" main.py \
  --model_name "$MODEL-official-pretrained-test" \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir "$DATA" \
  --test_model "$CKPT" \
  --save_image False 2>&1 | tee "$OFFICIAL_TEST_LOG"
official_rc=${PIPESTATUS[0]}
set -e
echo "official_pretrained_test_done rc=$official_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$official_rc" -ne 0 ]]; then
  echo "FLOW_FAILED_OFFICIAL_PRETRAINED_TEST" | tee -a "$STATUS"
  exit "$official_rc"
fi

echo "train_smoke_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" main.py \
  --model_name "$MODEL" \
  --mode train \
  --version base \
  --data Haze4K \
  --data_dir "$DATA" \
  --init_model "$CKPT" \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 1 \
  --print_freq 100 \
  --num_worker 4 \
  --save_freq 1 \
  --valid_freq 1 2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "train_smoke_done rc=$train_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then
  echo "FLOW_FAILED_TRAIN_SMOKE" | tee -a "$STATUS"
  exit "$train_rc"
fi

BEST="results/$MODEL/Training-Results/Best.pkl"
if [[ ! -f "$BEST" ]]; then
  echo "FLOW_FAILED_BEST_MISSING path=$BEST" | tee -a "$STATUS"
  exit 20
fi
sha256sum "$BEST" | tee "$EVID/smoke_best_sha256.txt"
ls -lh "$BEST" | tee "$EVID/smoke_best_ls.txt"

echo "smoke_best_test_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" main.py \
  --model_name "$MODEL-smoke-best-test" \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir "$DATA" \
  --test_model "$BEST" \
  --save_image False 2>&1 | tee "$SMOKE_TEST_LOG"
smoke_test_rc=${PIPESTATUS[0]}
set -e
echo "smoke_best_test_done rc=$smoke_test_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$smoke_test_rc" -ne 0 ]]; then
  echo "FLOW_FAILED_SMOKE_BEST_TEST" | tee -a "$STATUS"
  exit "$smoke_test_rc"
fi

"$PY" - <<'PY'
import json, re, hashlib
from pathlib import Path
E = Path('/data/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor/experience_docx/experiment_logs/haze4k_official_arch_anchor_convir5090_train_test_flow_20260610')
base = Path('/data/caozhiyang/ConvIR-B')
model = 'ConvIR-Haze4K-official-anchor-convir5090-flow-20260610'
best = Path('/data/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS/results') / model / 'Training-Results' / 'Best.pkl'

def parse_eval(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    out = {'log': str(path)}
    m = re.search(r'The average PSNR is\s+([0-9.]+)\s+dB', text)
    if m: out['psnr'] = float(m.group(1))
    m = re.search(r'The average SSIM is\s+([0-9.]+)\s+dB', text)
    if m: out['ssim'] = float(m.group(1))
    m = re.search(r'Average time:\s+([0-9.]+)', text)
    if m: out['avg_time'] = float(m.group(1))
    return out

def parse_train(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    out = {'log': str(path)}
    vals = re.findall(r'(\d+) epoch\s+\n Average PSNR ([0-9.]+) dB', text)
    out['valid_epochs'] = [{'epoch': int(e), 'psnr': float(p)} for e, p in vals]
    out['finite_loss_lines'] = len(re.findall(r'Loss content:', text))
    return out

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
summary = {
    'status': 'OFFICIAL_ANCHOR_CONVIR5090_TRAIN_TEST_FLOW_OK',
    'locked_test_policy': 'official pretrained evaluation and smoke test-entry verification only; no model selection',
    'worktree': '/data/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor',
    'branch': 'codex/haze4k-official-arch-anchor',
    'commit': '2d529d4',
    'python': '/data/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python',
    'data': '/data/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K',
    'checkpoint': '/data/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl',
    'checkpoint_sha256': sha(base/'checkpoints/official/Haze4K/haze4k-base.pkl'),
    'official_pretrained_test': parse_eval(E/'official_pretrained_test.log'),
    'train_smoke': parse_train(E/'train_smoke_from_pretrained_1epoch.log'),
    'smoke_best_checkpoint': str(best),
    'smoke_best_checkpoint_size': best.stat().st_size,
    'smoke_best_checkpoint_sha256': sha(best),
    'smoke_best_test': parse_eval(E/'smoke_best_test.log'),
}
(E/'train_test_flow_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "flow_done rc=0 haze4k_official_anchor_convir5090_train_test $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "OFFICIAL_ANCHOR_CONVIR5090_TRAIN_TEST_FLOW_OK" | tee -a "$STATUS"
