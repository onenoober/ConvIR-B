#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent
ROUTE_ID=haze4k_v4_7_dcfsb_candidate_validation_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
LOCK_DIR=$EVID/locked_test_once
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
CAND=$WORK/Dehazing/ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl
STATUS=$LOCK_DIR/status.txt
LOG=$LOCK_DIR/run_v47_locked_test_once.log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$LOCK_DIR"
if [ -f "$LOCK_DIR/v47_locked_test_summary.json" ]; then
  echo "LOCKED_TEST_ALREADY_HAS_SUMMARY refusing_second_metric_command" | tee -a "$STATUS"
  exit 3
fi
{
  echo "locked_test_run_start v47_adapter4_once $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CAND"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "save_prediction_images=false"
  echo "locked_test_command_count=1"
} | tee -a "$STATUS"
set +e
cd "$WORK/Dehazing/ITS"
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from data import test_dataloader
from models.ConvIR import build_net as build_official_net
from models.DCFSBConvIR import build_dcfsb_bottleneck_net

BASE = Path('/sda/home/wangyuxin/ConvIR-B')
WORK = BASE / 'repos' / 'ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent'
ROUTE_ID = 'haze4k_v4_7_dcfsb_candidate_validation_20260708'
EVID = WORK / 'experience_docx' / 'experiment_logs' / ROUTE_ID
LOCK_DIR = EVID / 'locked_test_once'
DATA = BASE / 'datasets' / 'Haze4K' / 'Haze4K'
A0 = BASE / 'checkpoints' / 'official' / 'Haze4K' / 'haze4k-base.pkl'
CAND = WORK / 'Dehazing' / 'ITS' / 'results' / 'ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708' / 'Training-Results' / 'Final.pkl'
THRESHOLDS = {
    'mean_delta_psnr_min_exclusive': 0.0,
    'positive_ratio_min': 0.50,
    'p5_delta_psnr_min': -0.50,
    'mean_delta_ssim_min': -0.0001,
    'metric_command_count': 1,
    'save_prediction_images': False,
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model_state(path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def pct(values, q):
    vals = sorted(float(v) for v in values)
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def psnr(pred, label):
    mse = F.mse_loss(pred, label)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def eval_pred(model, input_img):
    factor = 32
    h, w = input_img.shape[2], input_img.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    padded = F.pad(input_img, (0, padw, 0, padh), 'reflect')
    pred = model(padded)[2]
    return torch.clamp(pred[:, :, :h, :w], 0, 1), H, W


def ssim_val(pred, label, H, W):
    down_ratio = max(1, round(min(H, W) / 256))
    return float(ssim(
        F.adaptive_avg_pool2d(pred, (int(H / down_ratio), int(W / down_ratio))),
        F.adaptive_avg_pool2d(label, (int(H / down_ratio), int(W / down_ratio))),
        data_range=1,
        size_average=False,
    ).mean().detach().cpu())

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
official = build_official_net('base', 'Haze4K', 'original').to(device).eval()
official.load_state_dict(load_model_state(A0), strict=True)
candidate = build_dcfsb_bottleneck_net('base', 'Haze4K', 'original').to(device).eval()
candidate.load_state_dict(load_model_state(CAND), strict=True)
loader = test_dataloader(str(DATA), 'Haze4K', batch_size=1, num_workers=0)
rows = []
with torch.no_grad():
    for idx, batch in enumerate(loader, start=1):
        input_img, label_img, name = batch
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        t0 = time.time()
        a0_pred, H, W = eval_pred(official, input_img)
        a0_time = time.time() - t0
        t1 = time.time()
        cand_pred, H2, W2 = eval_pred(candidate, input_img)
        cand_time = time.time() - t1
        if H != H2 or W != W2:
            raise RuntimeError('pad shape mismatch')
        a0_psnr = psnr(a0_pred, label_img)
        cand_psnr = psnr(cand_pred, label_img)
        a0_ssim = ssim_val(a0_pred, label_img, H, W)
        cand_ssim = ssim_val(cand_pred, label_img, H, W)
        rows.append({
            'image_id': name[0],
            'a0_psnr': a0_psnr,
            'candidate_psnr': cand_psnr,
            'delta_psnr': cand_psnr - a0_psnr,
            'a0_ssim': a0_ssim,
            'candidate_ssim': cand_ssim,
            'delta_ssim': cand_ssim - a0_ssim,
            'a0_time_sec': a0_time,
            'candidate_time_sec': cand_time,
        })
        if idx % 50 == 0:
            mean_delta = statistics.mean(r['delta_psnr'] for r in rows)
            print(f'locked_test_progress count={idx} mean_delta_psnr={mean_delta:.6f}', flush=True)

deltas = [r['delta_psnr'] for r in rows]
ssim_d = [r['delta_ssim'] for r in rows]
summary_metrics = {
    'count': len(rows),
    'a0_mean_psnr': statistics.mean(r['a0_psnr'] for r in rows),
    'candidate_mean_psnr': statistics.mean(r['candidate_psnr'] for r in rows),
    'mean_delta_psnr': statistics.mean(deltas),
    'median_delta_psnr': statistics.median(deltas),
    'p5_delta_psnr': pct(deltas, 5),
    'p95_delta_psnr': pct(deltas, 95),
    'positive_ratio': sum(d > 0 for d in deltas) / len(deltas),
    'a0_mean_ssim': statistics.mean(r['a0_ssim'] for r in rows),
    'candidate_mean_ssim': statistics.mean(r['candidate_ssim'] for r in rows),
    'mean_delta_ssim': statistics.mean(ssim_d),
    'median_delta_ssim': statistics.median(ssim_d),
    'a0_mean_time_sec': statistics.mean(r['a0_time_sec'] for r in rows),
    'candidate_mean_time_sec': statistics.mean(r['candidate_time_sec'] for r in rows),
}
gates = {
    'mean_delta_psnr': summary_metrics['mean_delta_psnr'] > THRESHOLDS['mean_delta_psnr_min_exclusive'],
    'positive_ratio': summary_metrics['positive_ratio'] >= THRESHOLDS['positive_ratio_min'],
    'p5_delta_psnr': summary_metrics['p5_delta_psnr'] >= THRESHOLDS['p5_delta_psnr_min'],
    'mean_delta_ssim': summary_metrics['mean_delta_ssim'] >= THRESHOLDS['mean_delta_ssim_min'],
    'save_prediction_images_false': THRESHOLDS['save_prediction_images'] is False,
    'metric_command_count_one': THRESHOLDS['metric_command_count'] == 1,
}
summary = {
    'route_id': ROUTE_ID,
    'status': 'LOCKED_TEST_CONFIRM_PASS' if all(gates.values()) else 'LOCKED_TEST_CONFIRM_FAIL',
    'policy_note': 'A prior post-v47 directory-count preflight enumerated the locked test split but produced no metrics; this script is the sole metric-producing locked-test command for the fixed adapter4 checkpoint.',
    'branch': 'codex/haze4k-v4-7-dcfsb-candidate-validation',
    'cloud_commit_at_run': os.popen(f'cd {WORK} && git rev-parse HEAD').read().strip(),
    'data_root': str(DATA),
    'a0_checkpoint': str(A0),
    'a0_sha256': sha256_file(A0),
    'candidate_checkpoint': str(CAND),
    'candidate_sha256': sha256_file(CAND),
    'metrics': summary_metrics,
    'thresholds': THRESHOLDS,
    'gates': gates,
    'pass': all(gates.values()),
    'save_prediction_images': False,
    'metric_producing_locked_test_command_count': 1,
    'runtime_sec': time.time() - start,
}
LOCK_DIR.mkdir(parents=True, exist_ok=True)
with (LOCK_DIR / 'v47_locked_test_per_image.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
(LOCK_DIR / 'v47_locked_test_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

decision = f"""# Haze4K v4.7 Locked-Test Confirmation Decision

Status: `{summary['status']}`

Route id: `{ROUTE_ID}`

Fixed candidate: v4.6 DCFSB-bottleneck `adapter4` checkpoint.

Policy note: a prior post-v4.7 directory-count preflight enumerated the locked test split but produced no metric. This decision records the sole metric-producing locked-test command for the fixed candidate.

Prediction images saved: `false`

Metric-producing locked-test command count: `1`

## Metrics

| Metric | Value |
| --- | ---: |
| A0 mean PSNR | `{summary_metrics['a0_mean_psnr']:.6f}` |
| Candidate mean PSNR | `{summary_metrics['candidate_mean_psnr']:.6f}` |
| mean dPSNR | `{summary_metrics['mean_delta_psnr']:.6f}` |
| median dPSNR | `{summary_metrics['median_delta_psnr']:.6f}` |
| p5 dPSNR | `{summary_metrics['p5_delta_psnr']:.6f}` |
| p95 dPSNR | `{summary_metrics['p95_delta_psnr']:.6f}` |
| positive ratio | `{summary_metrics['positive_ratio']:.6f}` |
| A0 mean SSIM | `{summary_metrics['a0_mean_ssim']:.6f}` |
| Candidate mean SSIM | `{summary_metrics['candidate_mean_ssim']:.6f}` |
| mean dSSIM | `{summary_metrics['mean_delta_ssim']:.8f}` |

## Gate Results

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```

## Decision

The locked-test confirmation gate {'passed' if all(gates.values()) else 'failed'}. Do not run additional locked-test commands for this candidate. Do not tune from locked-test results.
"""
(LOCK_DIR / 'decision_after_locked_test.md').write_text(decision, encoding='utf-8')
print(json.dumps(summary, indent=2, sort_keys=True))
PYCODE
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "state=LOCKED_TEST_COMMAND_DONE" | tee -a "$STATUS"
  echo "locked_test_run_done rc=0 v47_adapter4_once $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V47_LOCKED_TEST_ONCE_OK
else
  echo "state=FAILED_COMMAND" | tee -a "$STATUS"
  echo "locked_test_run_done rc=$rc v47_adapter4_once $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V47_LOCKED_TEST_ONCE_FAILED
fi
exit "$rc"
