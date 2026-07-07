#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-2-gst-only
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_2_gst_only_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/v4_a2_gst_preflight.log
JSON_OUT=$EVID/v4_a2_gst_preflight.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "preflight_start v4_a2_gst $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_policy=train split count and one train crop only; test split not enumerated"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from models.ConvIR import build_net as build_official_net
from models.SFADGSTConvIR import (
    build_sfad_gst_net,
    configure_sfad_train_scope,
    load_haze4k_partial,
)
from data import train_dataloader

WORK = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-2-gst-only')
DATA = Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K')
A0 = Path('/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl')
JSON_OUT = WORK / 'experience_docx/experiment_logs/haze4k_v4_2_gst_only_20260707/v4_a2_gst_preflight.json'

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

def max_abs_delta(a, b):
    return max(float((x - y).abs().max().detach().cpu()) for x, y in zip(a, b))

result = {
    'route_id': 'haze4k_v4_2_gst_only_20260707',
    'branch': subprocess.check_output(['git', '-C', str(WORK), 'branch', '--show-current'], text=True).strip(),
    'commit': subprocess.check_output(['git', '-C', str(WORK), 'rev-parse', 'HEAD'], text=True).strip(),
    'status_short': subprocess.check_output(['git', '-C', str(WORK), 'status', '--short'], text=True).strip().splitlines(),
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(str(A0), map_location='cpu')
if isinstance(state, dict) and 'model' in state:
    state = state['model']

official = build_official_net('base', 'Haze4K', 'original')
official_load = official.load_state_dict(state, strict=True)
official.to(device).eval()

route = build_sfad_gst_net('base', 'Haze4K', 'original')
partial = load_haze4k_partial(route, str(A0), allowed_new_prefixes=('SFAD_GST',))
route.to(device).eval()

scope_report = configure_sfad_train_scope(route, 'adapter_only')
route.eval()

result['official_strict_load'] = {
    'missing': list(official_load.missing_keys),
    'unexpected': list(official_load.unexpected_keys),
    'loaded_key_count': len(state),
}
result['partial_load'] = partial
result['parameter_count_total'] = sum(p.numel() for p in route.parameters())
result['official_parameter_count_total'] = sum(p.numel() for p in official.parameters())
result['parameter_count_added'] = result['parameter_count_total'] - result['official_parameter_count_total']
result['adapter_scope'] = scope_report
result['forbidden_state_key_hits'] = [
    k for k in route.state_dict().keys()
    if any(token in k.lower() for token in ('apdr', 'dpga', 'udp', 'pfd', 'a0prox', 'wd_', 'sdfm', 'dcfsb', 'density'))
]

with torch.no_grad():
    x = torch.rand(1, 3, 256, 256, device=device)
    official_synth = official(x)
    route_synth = route(x)
    result['synthetic_output_shapes'] = [list(o.shape) for o in route_synth]
    result['synthetic_forward_finite'] = all(torch.isfinite(o).all().item() for o in route_synth)
    result['noop_max_abs_synthetic_vs_a0'] = max_abs_delta(route_synth, official_synth)

loader = train_dataloader(str(DATA), batch_size=1, num_workers=0, data='Haze4K', use_transform=True)
input_img, label_img = next(iter(loader))
input_img = input_img.to(device)
label_img = label_img.to(device)
with torch.no_grad():
    official_pred = official(input_img)
    route_pred = route(input_img)
    label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
    label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
    one_batch_loss = (
        F.l1_loss(route_pred[0], label_img4)
        + F.l1_loss(route_pred[1], label_img2)
        + F.l1_loss(route_pred[2], label_img)
    )
result['one_batch_input_shape'] = list(input_img.shape)
result['one_batch_output_shapes'] = [list(o.shape) for o in route_pred]
result['one_batch_forward_finite'] = all(torch.isfinite(o).all().item() for o in route_pred)
result['one_batch_loss'] = float(one_batch_loss.detach().cpu())
result['noop_max_abs_train_crop_vs_a0'] = max_abs_delta(route_pred, official_pred)
result['gst_stats'] = route.collect_modulation_stats(input_img)

missing_ok = all(k.startswith('SFAD_GST') for k in partial['missing_new_modules'])
gate_not_collapsed = all(
    stats.get('gate_std', 0.0) > 1e-5
    and stats.get('gate_lt_005', 1.0) < 0.95
    and stats.get('gate_gt_095', 1.0) < 0.95
    and stats.get('delta_abs_mean', 0.0) > 1e-7
    for stats in result['gst_stats'].values()
)
expected_hash = '6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088'
result['pass'] = bool(
    result['checkpoint_sha256'] == expected_hash
    and result['official_strict_load']['missing'] == []
    and result['official_strict_load']['unexpected'] == []
    and result['partial_load']['unexpected'] == []
    and result['partial_load']['shape_mismatch'] == []
    and missing_ok
    and result['parameter_count_added'] > 0
    and result['adapter_scope']['trainable_prefixes'] == ['SFAD_GST1', 'SFAD_GST2']
    and result['synthetic_forward_finite']
    and result['one_batch_forward_finite']
    and result['noop_max_abs_synthetic_vs_a0'] <= 1e-7
    and result['noop_max_abs_train_crop_vs_a0'] <= 1e-7
    and gate_not_collapsed
    and not result['locked_test_touched']
    and not result['test_split_enumerated']
    and not result['forbidden_state_key_hits']
)
JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, sort_keys=True))
if not result['pass']:
    raise SystemExit(2)
PYCODE
rc=$?
set -e
echo "preflight_done rc=$rc v4_a2_gst $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V4_A2_GST_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "V4_A2_GST_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
