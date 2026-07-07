#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-0-baseline-lock
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_0_baseline_lock_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/v4_a0_preflight.log
JSON_OUT=$EVID/v4_a0_preflight.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "preflight_start v4_a0 $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from models.ConvIR import build_net
from data import train_dataloader

WORK = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-0-baseline-lock')
DATA = Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K')
A0 = Path('/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl')
JSON_OUT = WORK / 'experience_docx/experiment_logs/haze4k_v4_0_baseline_lock_20260707/v4_a0_preflight.json'

random.seed(3407)
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def image_count(path: Path) -> int:
    exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    if not path.is_dir():
        return -1
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)

result = {
    'route_id': 'haze4k_v4_0_baseline_lock_20260707',
    'branch': None,
    'commit': None,
    'python': sys.executable,
    'python_version': sys.version.replace('\n', ' '),
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count_visible': torch.cuda.device_count(),
    'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'data_dir': str(DATA),
    'train_haze_count': image_count(DATA / 'train' / 'haze'),
    'train_gt_count': image_count(DATA / 'train' / 'gt'),
    'test_split_enumerated': False,
    'checkpoint': str(A0),
    'checkpoint_size': A0.stat().st_size,
    'checkpoint_sha256': sha256(A0),
    'locked_test_touched': False,
    'pass': False,
}

import subprocess
result['branch'] = subprocess.check_output(['git', '-C', str(WORK), 'branch', '--show-current'], text=True).strip()
result['commit'] = subprocess.check_output(['git', '-C', str(WORK), 'rev-parse', 'HEAD'], text=True).strip()
result['status_short'] = subprocess.check_output(['git', '-C', str(WORK), 'status', '--short'], text=True).strip().splitlines()

model = build_net('base', 'Haze4K', 'original')
state = torch.load(str(A0), map_location='cpu')
if isinstance(state, dict) and 'model' in state:
    state = state['model']
load_ret = model.load_state_dict(state, strict=True)
result['strict_load'] = {
    'missing': list(load_ret.missing_keys),
    'unexpected': list(load_ret.unexpected_keys),
    'loaded_key_count': len(state),
}
result['parameter_count_total'] = sum(p.numel() for p in model.parameters())
result['parameter_count_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
result['forbidden_state_key_hits'] = [
    k for k in model.state_dict().keys()
    if any(token in k.lower() for token in ('apdr', 'dpga', 'udp', 'pfd', 'a0prox', 'wd_'))
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
with torch.no_grad():
    x = torch.rand(1, 3, 256, 256, device=device)
    outs = model(x)
    result['synthetic_output_shapes'] = [list(o.shape) for o in outs]
    result['synthetic_forward_finite'] = all(torch.isfinite(o).all().item() for o in outs)

loader = train_dataloader(str(DATA), batch_size=1, num_workers=0, data='Haze4K', use_transform=True)
input_img, label_img = next(iter(loader))
input_img = input_img.to(device)
label_img = label_img.to(device)
with torch.no_grad():
    pred = model(input_img)
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
    one_batch_loss = (
        F.l1_loss(pred[0], label_img4)
        + F.l1_loss(pred[1], label_img2)
        + F.l1_loss(pred[2], label_img)
    )
result['one_batch_input_shape'] = list(input_img.shape)
result['one_batch_label_shape'] = list(label_img.shape)
result['one_batch_output_shapes'] = [list(o.shape) for o in pred]
result['one_batch_forward_finite'] = all(torch.isfinite(o).all().item() for o in pred)
result['one_batch_loss'] = float(one_batch_loss.detach().cpu())

expected_hash = '6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088'
result['pass'] = bool(
    result['checkpoint_sha256'] == expected_hash
    and result['strict_load']['missing'] == []
    and result['strict_load']['unexpected'] == []
    and result['parameter_count_total'] == 8630665
    and result['synthetic_forward_finite']
    and result['one_batch_forward_finite']
    and result['train_haze_count'] > 0
    and result['train_gt_count'] > 0
    and not result['locked_test_touched']
    and not result['forbidden_state_key_hits']
)
JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, sort_keys=True))
if not result['pass']:
    raise SystemExit(2)
PYCODE
rc=$?
set -e
echo "preflight_done rc=$rc v4_a0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V4_A0_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "V4_A0_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
