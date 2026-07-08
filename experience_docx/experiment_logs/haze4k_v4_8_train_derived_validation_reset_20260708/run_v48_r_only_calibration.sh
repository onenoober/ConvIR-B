#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-1}
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt
V45=$BASE/repos/ConvIR-B-haze4k-v4-5-sdc-lite
ROUTE_ID=haze4k_v4_8_train_derived_validation_reset_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
SDC_CKPT=$V45/Dehazing/ITS/results/ConvIR-Haze4K-v4A5-SDC-Lite-adapter-notest-seed3407-20260708/Training-Results/Final.pkl
OOF=$EVID/v48_oof_per_image_compact.csv
OUT=$EVID/r_only_calibration_sdc_lite_oof3000
STATUS=$OUT/status.txt
LOG=$OUT/r_only_calibration.log
mkdir -p "$OUT"
{
  echo "r_only_start v48_sdc_lite_oof3000 $(date --iso-8601=seconds)"
  echo "state=RUNNING_R_ONLY_AUDIT"
  echo "gpu=$GPU_ID"
  echo "work=$WORK"
  echo "v45_work=$V45"
  echo "v45_branch=$(cd "$V45" && git branch --show-current)"
  echo "v45_commit=$(cd "$V45" && git rev-parse HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "sdc_checkpoint=$SDC_CKPT"
  echo "oof_table=$OOF"
  echo "out=$OUT"
  echo "locked_test_policy=train-derived OOF image ids only; no test enumeration; restoration outputs ignored"
} | tee -a "$STATUS"
if [ ! -x "$PY" ]; then echo "V48_R_ONLY_FAILED python_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$SDC_CKPT" ]; then echo "V48_R_ONLY_FAILED checkpoint_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$OOF" ]; then echo "V48_R_ONLY_FAILED oof_table_missing" | tee -a "$STATUS"; exit 2; fi
if [ -f "$OUT/r_only_summary.json" ]; then echo "V48_R_ONLY_FAILED output_exists" | tee -a "$STATUS"; exit 3; fi
set +e
CUDA_VISIBLE_DEVICES=$GPU_ID TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 "$PY" - <<'PY' > "$LOG" 2>&1
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

BASE = Path('/sda/home/wangyuxin/ConvIR-B')
WORK = BASE / 'repos' / 'ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt'
V45 = BASE / 'repos' / 'ConvIR-B-haze4k-v4-5-sdc-lite'
EVID = WORK / 'experience_docx' / 'experiment_logs' / 'haze4k_v4_8_train_derived_validation_reset_20260708'
DATA = BASE / 'datasets' / 'Haze4K' / 'Haze4K'
SDC_CKPT = V45 / 'Dehazing' / 'ITS' / 'results' / 'ConvIR-Haze4K-v4A5-SDC-Lite-adapter-notest-seed3407-20260708' / 'Training-Results' / 'Final.pkl'
OOF = EVID / 'v48_oof_per_image_compact.csv'
OUT = EVID / 'r_only_calibration_sdc_lite_oof3000'
OUT.mkdir(parents=True, exist_ok=True)
ROUTE_ID = 'haze4k_v4_8_train_derived_validation_reset_20260708'
THRESHOLDS = {
    'R_1_2_std_mean_min': 0.10,
    'corr_R_input_gt_l1_min': 0.10,
    'corr_R_a0_error_proxy_min': 0.10,
    'corr_R_dark_channel_mean_min': 0.0,
    'heavy_haze_q4_vs_q1_rel_min': 0.10,
    'a0_error_q4_vs_q1_rel_min': 0.10,
    'low_saturation_no_reverse_min_diff': 0.0,
}
PROXY_KEYS = [
    'input_gt_l1',
    'a0_error_proxy_low_plus_high_l1',
    'input_dark_channel_mean',
    'input_brightness_mean',
    'input_saturation_proxy',
    'gt_texture_proxy',
    'hazy_texture_proxy',
    'a0_psnr',
]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

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

def pearson(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-20 or vy <= 1e-20:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))

def pct(values, q):
    vals = sorted(float(v) for v in values)
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

def quantile_labels(rows, key):
    ordered = sorted((float(r[key]), idx) for idx, r in enumerate(rows))
    n = len(ordered)
    labels = [None] * n
    for rank, (_, idx) in enumerate(ordered):
        labels[idx] = f'q{min(3, int(rank * 4 / n)) + 1}'
    return labels

def summarize_r(rows):
    return {
        'count': len(rows),
        'R_1_2_mean': float(statistics.mean(float(r['R_mean']) for r in rows)),
        'R_1_2_std_mean': float(statistics.mean(float(r['R_std']) for r in rows)),
        'R_1_2_std_median': float(statistics.median(float(r['R_std']) for r in rows)),
        'R_1_2_entropy_mean': float(statistics.mean(float(r['R_entropy']) for r in rows)),
        'R_1_2_mean_p5': pct([r['R_mean'] for r in rows], 5),
        'R_1_2_mean_p95': pct([r['R_mean'] for r in rows], 95),
    }

def rel_delta(high, low):
    denom = abs(low) if abs(low) > 1e-12 else 1.0
    return float((high - low) / denom)

start = time.time()
sys.path.insert(0, str(V45 / 'Dehazing' / 'ITS'))
from models.SDCConvIR import build_sdc_lite_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_sdc_lite_net('base', 'Haze4K', 'original').to(device).eval()
model.load_state_dict(load_state(SDC_CKPT), strict=True)

with OOF.open(newline='', encoding='utf-8') as f:
    base_rows = [dict(r) for r in csv.DictReader(f)]
if len(base_rows) != 3000:
    raise RuntimeError(f'expected 3000 OOF rows, got {len(base_rows)}')
if len(base_rows) != len({r['image_id'] for r in base_rows}):
    raise RuntimeError('duplicate image ids in OOF table')

rows = []
with torch.no_grad():
    for idx, src in enumerate(base_rows, start=1):
        name = src['image_id']
        input_path = DATA / 'train' / 'haze' / name
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        image = TF.to_tensor(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
        h, w = image.shape[-2:]
        H = ((h + 31) // 32) * 32
        W = ((w + 31) // 32) * 32
        padded = F.pad(image, (0, W - w, 0, H - h), 'reflect')
        model.collect_modulation_stats(padded)
        stats = model.SFAD_SDC.collect_stats()
        row = {
            'fold': int(src['fold']),
            'image_id': name,
            'R_mean': float(stats['R_mean']),
            'R_std': float(stats['R_std']),
            'R_min': float(stats['R_min']),
            'R_max': float(stats['R_max']),
            'R_entropy': float(stats['R_entropy']),
            'R_lt_005': float(stats['R_lt_005']),
            'R_gt_095': float(stats['R_gt_095']),
            'gamma_mean': float(stats['gamma_mean']),
            'gamma_std': float(stats['gamma_std']),
            'beta_mean': float(stats['beta_mean']),
            'beta_std': float(stats['beta_std']),
            'alpha': float(stats['alpha']),
        }
        for key in PROXY_KEYS:
            row[key] = float(src[key])
        rows.append(row)
        if idx % 100 == 0 or idx == len(base_rows):
            print(f"r_only progress={idx}/{len(base_rows)} R_mean={statistics.mean(r['R_mean'] for r in rows):.6f} R_std_mean={statistics.mean(r['R_std'] for r in rows):.6f}", flush=True)

for key in PROXY_KEYS:
    labels = quantile_labels(rows, key)
    for row, label in zip(rows, labels):
        row[f'{key}_quartile'] = label

summary = summarize_r(rows)
correlations = {f'corr_R_mean_{key}': pearson([r['R_mean'] for r in rows], [r[key] for r in rows]) for key in PROXY_KEYS}
correlations.update({f'corr_R_std_{key}': pearson([r['R_std'] for r in rows], [r[key] for r in rows]) for key in PROXY_KEYS})

bin_rows = []
for key in PROXY_KEYS:
    for label in ['q1', 'q2', 'q3', 'q4']:
        subset = [r for r in rows if r[f'{key}_quartile'] == label]
        out = summarize_r(subset)
        vals = [r[key] for r in subset]
        out.update({
            'proxy': key,
            'bin': label,
            'proxy_min': min(vals),
            'proxy_max': max(vals),
            'proxy_mean': float(statistics.mean(vals)),
        })
        bin_rows.append(out)

by_proxy = {(r['proxy'], r['bin']): r for r in bin_rows}
heavy_q1 = by_proxy[('input_gt_l1', 'q1')]['R_1_2_mean']
heavy_q4 = by_proxy[('input_gt_l1', 'q4')]['R_1_2_mean']
a0err_q1 = by_proxy[('a0_error_proxy_low_plus_high_l1', 'q1')]['R_1_2_mean']
a0err_q4 = by_proxy[('a0_error_proxy_low_plus_high_l1', 'q4')]['R_1_2_mean']
sat_q1 = by_proxy[('input_saturation_proxy', 'q1')]['R_1_2_mean']
sat_q4 = by_proxy[('input_saturation_proxy', 'q4')]['R_1_2_mean']
response = {
    'heavy_haze_input_gt_l1_q1_R_mean': heavy_q1,
    'heavy_haze_input_gt_l1_q4_R_mean': heavy_q4,
    'heavy_haze_q4_vs_q1_rel': rel_delta(heavy_q4, heavy_q1),
    'a0_error_q1_R_mean': a0err_q1,
    'a0_error_q4_R_mean': a0err_q4,
    'a0_error_q4_vs_q1_rel': rel_delta(a0err_q4, a0err_q1),
    'low_saturation_q1_R_mean': sat_q1,
    'high_saturation_q4_R_mean': sat_q4,
    'low_saturation_q1_minus_q4': float(sat_q1 - sat_q4),
}

gates = {
    'R_1_2_std_mean': summary['R_1_2_std_mean'] >= THRESHOLDS['R_1_2_std_mean_min'],
    'corr_R_input_gt_l1': (correlations['corr_R_mean_input_gt_l1'] or -999) > THRESHOLDS['corr_R_input_gt_l1_min'],
    'corr_R_a0_error_proxy': (correlations['corr_R_mean_a0_error_proxy_low_plus_high_l1'] or -999) > THRESHOLDS['corr_R_a0_error_proxy_min'],
    'corr_R_dark_channel_direction': (correlations['corr_R_mean_input_dark_channel_mean'] or -999) > THRESHOLDS['corr_R_dark_channel_mean_min'],
    'heavy_haze_q4_gt_q1_by_10pct': response['heavy_haze_q4_vs_q1_rel'] >= THRESHOLDS['heavy_haze_q4_vs_q1_rel_min'],
    'a0_error_q4_gt_q1_by_10pct': response['a0_error_q4_vs_q1_rel'] >= THRESHOLDS['a0_error_q4_vs_q1_rel_min'],
    'low_saturation_no_reverse': response['low_saturation_q1_minus_q4'] >= THRESHOLDS['low_saturation_no_reverse_min_diff'],
    'locked_test_not_touched': True,
    'test_split_not_enumerated': True,
}
pass_gate = all(gates.values())
closeout = {
    'route_id': ROUTE_ID,
    'probe': 'v48_r_only_calibration_sdc_lite_oof3000',
    'status': 'COMPLETED_GATE_PASS' if pass_gate else 'COMPLETED_GATE_FAIL',
    'decision': 'R field passes calibration gate; bounded R-gated residual experiments may be considered without locked test.' if pass_gate else 'R field fails calibration gate; block SDC-Lite v2 and any skip/FAM/restoration modulation using this R.',
    'scope': 'R-only audit on v4.8 train-derived OOF union; restoration outputs ignored; no training; no locked test.',
    'count': len(rows),
    'sdc_checkpoint': str(SDC_CKPT),
    'sdc_checkpoint_sha256': sha256_file(SDC_CKPT),
    'summary': summary,
    'correlations': correlations,
    'response': response,
    'thresholds': THRESHOLDS,
    'gates': gates,
    'failed_gates': [k for k, v in gates.items() if not v],
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'runtime_sec': time.time() - start,
}
fields = [
    'fold', 'image_id', 'R_mean', 'R_std', 'R_min', 'R_max', 'R_entropy', 'gamma_mean', 'gamma_std', 'beta_mean', 'beta_std', 'alpha',
    'input_gt_l1', 'a0_error_proxy_low_plus_high_l1', 'input_dark_channel_mean', 'input_brightness_mean', 'input_saturation_proxy', 'gt_texture_proxy', 'hazy_texture_proxy', 'a0_psnr',
    'input_gt_l1_quartile', 'a0_error_proxy_low_plus_high_l1_quartile', 'input_dark_channel_mean_quartile', 'input_saturation_proxy_quartile', 'a0_psnr_quartile',
]
with (OUT / 'r_only_stats_compact.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in sorted(rows, key=lambda r: (r['fold'], r['image_id'])):
        w.writerow({k: row.get(k, '') for k in fields})
with (OUT / 'r_only_proxy_bins.csv').open('w', newline='', encoding='utf-8') as f:
    fieldnames = list(bin_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(bin_rows)
(OUT / 'r_only_summary.json').write_text(json.dumps(closeout, indent=2, sort_keys=True) + '\n', encoding='utf-8')
failed = '\n'.join(f"- `{name}`" for name in closeout['failed_gates']) or '- None'
bin_table = '\n'.join(
    f"| `{key}` | q1 `{by_proxy[(key, 'q1')]['R_1_2_mean']:.6f}` | q4 `{by_proxy[(key, 'q4')]['R_1_2_mean']:.6f}` | corr `{correlations.get('corr_R_mean_' + key):.6f}` |"
    for key in ['input_gt_l1', 'a0_error_proxy_low_plus_high_l1', 'input_dark_channel_mean', 'input_saturation_proxy', 'a0_psnr']
)
md = f"""# v4.8 R-only Calibration Decision

Probe: `v48_r_only_calibration_sdc_lite_oof3000`

Decision: **{'PASS' if pass_gate else 'FAIL'}**.

Scope: train-derived OOF union only (`3000` images). Restoration outputs were ignored; no training, no prediction images, no locked-test enumeration.

## Summary

- R mean: `{summary['R_1_2_mean']:.6f}`
- R std mean: `{summary['R_1_2_std_mean']:.6f}` (gate `>= 0.10`)
- corr(R, input-GT L1): `{correlations['corr_R_mean_input_gt_l1']:.6f}` (gate `> 0.10`)
- corr(R, A0 low+high error proxy): `{correlations['corr_R_mean_a0_error_proxy_low_plus_high_l1']:.6f}` (gate `> 0.10`)
- corr(R, dark-channel mean): `{correlations['corr_R_mean_input_dark_channel_mean']:.6f}` (direction gate `> 0`)
- heavy haze q4 vs q1 relative response: `{response['heavy_haze_q4_vs_q1_rel']:.6f}` (gate `>= 0.10`)
- A0-error q4 vs q1 relative response: `{response['a0_error_q4_vs_q1_rel']:.6f}` (gate `>= 0.10`)
- low-saturation q1 minus high-saturation q4 R mean: `{response['low_saturation_q1_minus_q4']:.6f}` (gate `>= 0`)

## Failed Gates

{failed}

## Proxy Direction Snapshot

| Proxy | q1 R mean | q4 R mean | corr(R, proxy) |
| --- | ---: | ---: | ---: |
{bin_table}

## Interpretation

This audit tests whether the existing v4.5 SDC-Lite response field behaves like a usable haze/error controller. It does not test restoration quality and does not authorize connecting R to skip/FAM/restoration outputs.
"""
(OUT / 'decision_after_v48_r_only.md').write_text(md, encoding='utf-8')
print(json.dumps({
    'summary': summary,
    'correlations': correlations,
    'response': response,
    'failed_gates': closeout['failed_gates'],
    'pass': pass_gate,
    'runtime_sec': closeout['runtime_sec'],
}, indent=2, sort_keys=True))
print('V48_R_ONLY_AUDIT_OK')
PY
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "state=R_ONLY_DONE" | tee -a "$STATUS"
  echo "r_only_done rc=0 v48_sdc_lite_oof3000 $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_R_ONLY_OK" | tee -a "$STATUS"
else
  echo "state=FAILED_R_ONLY" | tee -a "$STATUS"
  echo "r_only_done rc=$rc v48_sdc_lite_oof3000 $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_R_ONLY_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
