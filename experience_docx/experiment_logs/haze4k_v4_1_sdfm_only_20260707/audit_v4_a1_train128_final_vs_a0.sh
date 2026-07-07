#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-1-sdfm-only
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
CAND=$ITS/results/ConvIR-Haze4K-v4A1-SDFM-adapter-notest-seed3407-20260707/Training-Results/Final.pkl
STATUS=$EVID/status.txt
LOG=$EVID/audit_v4_a1_train128_final_vs_a0.log
SUMMARY=$EVID/a1_train128_compare_final_vs_a0.json
PER_IMAGE=$EVID/a1_train128_per_image_final_vs_a0.csv
MODULE_STATS=$EVID/a1_train128_module_stats_final.jsonl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "audit_start v4_a1_train128_final_vs_a0 $(date --iso-8601=seconds)"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CAND"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_policy=train/haze sorted first128 only"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from pytorch_msssim import ssim

from models.ConvIR import build_net as build_official_net
from models.SFADConvIR import build_sfad_sdfm_net

DATA = Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K')
A0 = Path('/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl')
CAND = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-1-sdfm-only/Dehazing/ITS/results/ConvIR-Haze4K-v4A1-SDFM-adapter-notest-seed3407-20260707/Training-Results/Final.pkl')
EVID = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-1-sdfm-only/experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707')
SUMMARY = EVID / 'a1_train128_compare_final_vs_a0.json'
PER_IMAGE = EVID / 'a1_train128_per_image_final_vs_a0.csv'
MODULE_STATS = EVID / 'a1_train128_module_stats_final.jsonl'

def load_state(path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state

def label_path_for(image_name):
    label_dir = DATA / 'train' / 'gt'
    candidates = [image_name]
    stem, ext = os.path.splitext(image_name)
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

def eval_model(model, image):
    factor = 32
    h, w = image.shape[2], image.shape[3]
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    padded = F.pad(image, (0, padw, 0, padh), 'reflect')
    pred = model(padded)[2][:, :, :h, :w]
    return torch.clamp(pred, 0, 1), H, W

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
route = build_sfad_sdfm_net('base', 'Haze4K', 'original').to(device).eval()
route.load_state_dict(load_state(CAND), strict=True)

exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
names = sorted(p.name for p in (DATA / 'train' / 'haze').iterdir() if p.is_file() and p.suffix.lower() in exts)[:128]
rows = []
module_sums = {}
start_all = time.time()
with torch.no_grad():
    for idx, name in enumerate(names, 1):
        hazy = TF.to_tensor(Image.open(DATA / 'train' / 'haze' / name).convert('RGB')).unsqueeze(0).to(device)
        label = TF.to_tensor(Image.open(label_path_for(name)).convert('RGB')).unsqueeze(0).to(device)
        a0_pred, H, W = eval_model(official, hazy)
        cand_pred, _, _ = eval_model(route, hazy)
        a0_psnr = psnr(a0_pred, label)
        cand_psnr = psnr(cand_pred, label)
        down_ratio = max(1, round(min(H, W) / 256))
        a0_ssim = float(ssim(
            F.adaptive_avg_pool2d(a0_pred, (int(H / down_ratio), int(W / down_ratio))),
            F.adaptive_avg_pool2d(label, (int(H / down_ratio), int(W / down_ratio))),
            data_range=1,
            size_average=False,
        ).mean().detach().cpu())
        cand_ssim = float(ssim(
            F.adaptive_avg_pool2d(cand_pred, (int(H / down_ratio), int(W / down_ratio))),
            F.adaptive_avg_pool2d(label, (int(H / down_ratio), int(W / down_ratio))),
            data_range=1,
            size_average=False,
        ).mean().detach().cpu())
        stats = {
            'SDFM_1_4': route.SFAD_SDFM1.collect_stats(),
            'SDFM_1_2': route.SFAD_SDFM2.collect_stats(),
        }
        for module, values in stats.items():
            module_sums.setdefault(module, {})
            for key, value in values.items():
                module_sums[module][key] = module_sums[module].get(key, 0.0) + float(value)
        rows.append({
            'name': name,
            'a0_psnr': a0_psnr,
            'candidate_psnr': cand_psnr,
            'delta_psnr': cand_psnr - a0_psnr,
            'a0_ssim': a0_ssim,
            'candidate_ssim': cand_ssim,
            'delta_ssim': cand_ssim - a0_ssim,
            'input_gt_l1': float(F.l1_loss(hazy, label).detach().cpu()),
            'a0_gt_l1': float(F.l1_loss(a0_pred, label).detach().cpu()),
            'candidate_gt_l1': float(F.l1_loss(cand_pred, label).detach().cpu()),
        })
        if idx % 32 == 0:
            print(f'audit {idx}/{len(names)} mean_delta={statistics.mean(r["delta_psnr"] for r in rows):.6f}', flush=True)

deltas = [r['delta_psnr'] for r in rows]
ssim_deltas = [r['delta_ssim'] for r in rows]
summary = {
    'route_id': 'haze4k_v4_1_sdfm_only_20260707',
    'audit': 'train_sorted_first128_final_vs_a0',
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'sample_policy': 'sorted first 128 files from Haze4K train/haze; this is train-fit/mechanism sanity, not generalization evidence',
    'count': len(rows),
    'a0_checkpoint': str(A0),
    'candidate_checkpoint': str(CAND),
    'mean_delta_psnr': statistics.mean(deltas),
    'median_delta_psnr': statistics.median(deltas),
    'p5_delta_psnr': percentile(deltas, 5),
    'p95_delta_psnr': percentile(deltas, 95),
    'mean_delta_ssim': statistics.mean(ssim_deltas),
    'positive_ratio': sum(d > 0 for d in deltas) / len(deltas),
    'worst_delta_psnr': min(deltas),
    'best_delta_psnr': max(deltas),
    'runtime_sec': time.time() - start_all,
}
SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
with PER_IMAGE.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with MODULE_STATS.open('w', encoding='utf-8') as f:
    for module, values in sorted(module_sums.items()):
        averaged = {k: v / len(rows) for k, v in sorted(values.items())}
        averaged.update({'module': module, 'split': 'train_sorted_first128', 'sample_count': len(rows)})
        f.write(json.dumps(averaged, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PYCODE
rc=$?
set -e
echo "audit_done rc=$rc v4_a1_train128_final_vs_a0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V4_A1_TRAIN128_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V4_A1_TRAIN128_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
