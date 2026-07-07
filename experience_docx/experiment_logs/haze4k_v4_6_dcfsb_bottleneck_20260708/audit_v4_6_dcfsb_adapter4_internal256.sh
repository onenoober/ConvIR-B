#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_6_dcfsb_bottleneck_20260708_adapter4
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
CAND=$ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl
TRAINFIT=$WORK/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_diagnosis_trainfit128.txt
HOLDOUT=$WORK/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_internal_holdout256.txt
STATUS=$EVID/status.txt
LOG=$EVID/audit_v4_6_dcfsb_adapter4_internal256.log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "audit_start v4_6_dcfsb_adapter4_internal256 $(date --iso-8601=seconds)"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CAND"
  echo "trainfit=$TRAINFIT"
  echo "holdout=$HOLDOUT"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_policy=train-derived trainfit/internal only"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from pytorch_msssim import ssim

from models.ConvIR import build_net as build_official_net
from models.DCFSBConvIR import build_dcfsb_bottleneck_net

DATA = Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K')
A0 = Path('/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl')
CAND = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent/Dehazing/ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl')
EVID = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent/experience_docx/experiment_logs/haze4k_v4_6_dcfsb_bottleneck_20260708_adapter4')
TRAINFIT = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_diagnosis_trainfit128.txt')
HOLDOUT = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_internal_holdout256.txt')

def load_state(path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state

def label_path_for(image_name):
    label_dir = DATA / 'train' / 'gt'
    stem, ext = os.path.splitext(image_name)
    candidates = [image_name]
    if '_' in stem:
        candidates.append(stem.split('_')[0] + ext)
        candidates.append(stem.split('_')[0] + '.png')
    for candidate in candidates:
        p = label_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError((image_name, candidates))

def psnr(pred, label):
    mse = F.mse_loss(pred, label)
    return float((10 * torch.log10(1 / mse)).detach().cpu())

def lowpass(t):
    return F.avg_pool2d(t, kernel_size=9, stride=1, padding=4, count_include_pad=False)

def frequency_l1(pred, label):
    pred_low = lowpass(pred)
    label_low = lowpass(label)
    pred_high = pred - pred_low
    label_high = label - label_low
    return (
        float(F.l1_loss(pred_low, label_low).detach().cpu()),
        float(F.l1_loss(pred_high, label_high).detach().cpu()),
    )

def eval_model(model, image):
    h, w = image.shape[-2:]
    H = ((h + 31) // 32) * 32
    W = ((w + 31) // 32) * 32
    padded = F.pad(image, (0, W - w, 0, H - h), 'reflect')
    pred = model(padded)[2][:, :, :h, :w]
    stats = getattr(model, 'DCFSB_Bottleneck', None)
    return torch.clamp(pred, 0, 1), (stats.collect_stats() if stats is not None else {}), H, W

def percentile(values, pct):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
official = build_official_net('base', 'Haze4K', 'original').to(device).eval()
official.load_state_dict(load_state(A0), strict=True)
route = build_dcfsb_bottleneck_net('base', 'Haze4K', 'original').to(device).eval()
route.load_state_dict(load_state(CAND), strict=True)

splits = {'trainfit128': TRAINFIT.read_text().splitlines(), 'internal256': HOLDOUT.read_text().splitlines()}
rows = []
module_rows = []
start = time.time()
with torch.no_grad():
    for split, names in splits.items():
        for name in names:
            hazy = TF.to_tensor(Image.open(DATA / 'train' / 'haze' / name).convert('RGB')).unsqueeze(0).to(device)
            label = TF.to_tensor(Image.open(label_path_for(name)).convert('RGB')).unsqueeze(0).to(device)
            a0_pred, _, H, W = eval_model(official, hazy)
            cand_pred, stats, _, _ = eval_model(route, hazy)
            a0_psnr = psnr(a0_pred, label)
            cand_psnr = psnr(cand_pred, label)
            down_ratio = max(1, round(min(H, W) / 256))
            pooled_label = F.adaptive_avg_pool2d(label, (int(H / down_ratio), int(W / down_ratio)))
            a0_ssim = float(ssim(F.adaptive_avg_pool2d(a0_pred, pooled_label.shape[-2:]), pooled_label, data_range=1, size_average=False).mean().detach().cpu())
            cand_ssim = float(ssim(F.adaptive_avg_pool2d(cand_pred, pooled_label.shape[-2:]), pooled_label, data_range=1, size_average=False).mean().detach().cpu())
            a0_low_l1, a0_high_l1 = frequency_l1(a0_pred, label)
            cand_low_l1, cand_high_l1 = frequency_l1(cand_pred, label)
            rows.append({
                'split': split,
                'image_id': name,
                'a0_psnr': a0_psnr,
                'candidate_psnr': cand_psnr,
                'delta_psnr': cand_psnr - a0_psnr,
                'a0_ssim': a0_ssim,
                'candidate_ssim': cand_ssim,
                'delta_ssim': cand_ssim - a0_ssim,
                'a0_low_l1': a0_low_l1,
                'candidate_low_l1': cand_low_l1,
                'delta_low_l1': cand_low_l1 - a0_low_l1,
                'a0_high_l1': a0_high_l1,
                'candidate_high_l1': cand_high_l1,
                'delta_high_l1': cand_high_l1 - a0_high_l1,
            })
            mod = {'split': split, 'image_id': name, 'module': 'DCFSB_Bottleneck'}
            mod.update(stats)
            module_rows.append(mod)
        split_deltas = [r['delta_psnr'] for r in rows if r['split'] == split]
        print(f'audit {split} {len(names)} done mean_delta={statistics.mean(split_deltas):.6f}', flush=True)

def summarize(split):
    subset = [r for r in rows if r['split'] == split]
    mods = [m for m in module_rows if m['split'] == split]
    deltas = [r['delta_psnr'] for r in subset]
    high_l1 = [r['delta_high_l1'] for r in subset]
    low_l1 = [r['delta_low_l1'] for r in subset]
    return {
        'count': len(subset),
        'mean_delta_psnr': statistics.mean(deltas),
        'median_delta_psnr': statistics.median(deltas),
        'p5_delta_psnr': percentile(deltas, 5),
        'p95_delta_psnr': percentile(deltas, 95),
        'positive_ratio': sum(d > 0 for d in deltas) / len(deltas),
        'mean_delta_ssim': statistics.mean(r['delta_ssim'] for r in subset),
        'mean_delta_low_l1': statistics.mean(low_l1),
        'mean_delta_high_l1': statistics.mean(high_l1),
        'high_l1_improve_ratio': sum(d < 0 for d in high_l1) / len(high_l1),
        'low_gate_std_mean': statistics.mean(m['low_gate_std'] for m in mods),
        'high_gate_std_mean': statistics.mean(m['high_gate_std'] for m in mods),
        'alpha_abs_mean': statistics.mean(m['alpha_abs_mean'] for m in mods),
        'high_low_energy_ratio_mean': statistics.mean(m['high_low_energy_ratio'] for m in mods),
    }

summary = {
    'route_id': 'haze4k_v4_6_dcfsb_bottleneck_20260708_adapter4',
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'trainfit128': summarize('trainfit128'),
    'internal256': summarize('internal256'),
    'runtime_sec': time.time() - start,
}
gate = {
    'route_id': summary['route_id'],
    'gate': 'dcfsb_bottleneck_internal256_stage_gate',
    'thresholds': {
        'mean_delta_psnr_min': 0.03,
        'positive_ratio_min': 0.53,
        'p5_delta_psnr_min': -0.25,
        'mean_delta_high_l1_max': 0.0005,
        'high_gate_std_mean_min': 0.01,
    },
    'metrics': summary['internal256'],
    'pass': bool(
        summary['internal256']['mean_delta_psnr'] >= 0.03
        and summary['internal256']['positive_ratio'] >= 0.53
        and summary['internal256']['p5_delta_psnr'] >= -0.25
        and summary['internal256']['mean_delta_high_l1'] <= 0.0005
        and summary['internal256']['high_gate_std_mean'] >= 0.01
    ),
}
(EVID / 'v46_dcfsb_adapter4_audit_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(EVID / 'v46_dcfsb_adapter4_gate.json').write_text(json.dumps(gate, indent=2, sort_keys=True) + '\n', encoding='utf-8')
with (EVID / 'v46_dcfsb_adapter4_per_image.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with (EVID / 'v46_dcfsb_adapter4_module_stats.jsonl').open('w', encoding='utf-8') as f:
    for row in module_rows:
        f.write(json.dumps(row, sort_keys=True) + '\n')
print(json.dumps({'summary': summary, 'gate': gate}, indent=2, sort_keys=True))
PYCODE
rc=$?
set -e
echo "audit_done rc=$rc v4_6_dcfsb_adapter4_internal256 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V4_6_DCFSB_ADAPTER4_INTERNAL256_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V4_6_DCFSB_ADAPTER4_INTERNAL256_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
