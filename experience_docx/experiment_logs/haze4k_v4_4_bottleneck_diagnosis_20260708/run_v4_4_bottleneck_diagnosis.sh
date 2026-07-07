#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-4-bottleneck-diagnosis
A3_WORK=$BASE/repos/ConvIR-B-haze4k-v4-3-sdfm-gst
A3_ITS=$A3_WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_4_bottleneck_diagnosis_20260708
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
STATUS=$EVID/status.txt
LOG=$EVID/v4_4_bottleneck_diagnosis.log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "audit_start v4_4_bottleneck_diagnosis $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "a3_code=$A3_ITS"
  echo "data=$DATA"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_policy=train-derived splits only; test split not enumerated"
} | tee -a "$STATUS"
cd "$A3_ITS"
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
from models.SFADConvIR import build_sfad_sdfm_net, SpatialDegradationFieldModulation
from models.SFADGSTConvIR import build_sfad_gst_net, GuidedSkipTransfer
from models.SFADSDFMGSTConvIR import build_sfad_sdfm_gst_net

BASE = Path('/sda/home/wangyuxin/ConvIR-B')
WORK = BASE / 'repos/ConvIR-B-haze4k-v4-4-bottleneck-diagnosis'
DATA = BASE / 'datasets/Haze4K/Haze4K'
EVID = WORK / 'experience_docx/experiment_logs/haze4k_v4_4_bottleneck_diagnosis_20260708'
SPLIT_DIR = WORK / 'docs/ai_text_packages/haze4k_v4_sfad/splits'
A0 = BASE / 'checkpoints/official/Haze4K/haze4k-base.pkl'
A1 = BASE / 'repos/ConvIR-B-haze4k-v4-1-sdfm-only/Dehazing/ITS/results/ConvIR-Haze4K-v4A1-SDFM-adapter-notest-seed3407-20260707/Training-Results/Final.pkl'
A2 = BASE / 'repos/ConvIR-B-haze4k-v4-2-gst-only/Dehazing/ITS/results/ConvIR-Haze4K-v4A2-GST-adapter-notest-seed3407-20260707/Training-Results/Final.pkl'
A3 = BASE / 'repos/ConvIR-B-haze4k-v4-3-sdfm-gst/Dehazing/ITS/results/ConvIR-Haze4K-v4A3-SDFM-GST-adapter-notest-seed3407-20260707/Training-Results/Final.pkl'

def entropy01(tensor, bins=16):
    values = tensor.detach().float().clamp(0, 1).reshape(-1).cpu()
    hist = torch.histc(values, bins=bins, min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(1e-12)
    prob = prob[prob > 0]
    return float((-(prob * torch.log(prob)).sum() / math.log(bins)).item())

def sdfm_summary(self, degradation_field, gamma, beta):
    field = degradation_field.detach()
    gamma = gamma.detach()
    beta = beta.detach()
    return {
        'R_mean': float(field.mean().cpu()),
        'R_std': float(field.std(unbiased=False).cpu()),
        'R_min': float(field.min().cpu()),
        'R_max': float(field.max().cpu()),
        'R_lt_005': float((field < 0.05).float().mean().cpu()),
        'R_gt_095': float((field > 0.95).float().mean().cpu()),
        'R_entropy': entropy01(field),
        'gamma_mean': float(gamma.mean().cpu()),
        'gamma_std': float(gamma.std(unbiased=False).cpu()),
        'beta_mean': float(beta.mean().cpu()),
        'beta_std': float(beta.std(unbiased=False).cpu()),
        'alpha': float(self.alpha.detach().cpu().reshape(-1)[0]),
    }

def gst_summary(self, gate, delta, skip_high, decoder_high):
    gate = gate.detach()
    delta = delta.detach()
    skip_high = skip_high.detach()
    decoder_high = decoder_high.detach()
    effective = self.alpha.detach() * gate * delta
    return {
        'gate_mean': float(gate.mean().cpu()),
        'gate_std': float(gate.std(unbiased=False).cpu()),
        'gate_min': float(gate.min().cpu()),
        'gate_max': float(gate.max().cpu()),
        'gate_lt_005': float((gate < 0.05).float().mean().cpu()),
        'gate_gt_095': float((gate > 0.95).float().mean().cpu()),
        'delta_mean': float(delta.mean().cpu()),
        'delta_std': float(delta.std(unbiased=False).cpu()),
        'delta_abs_mean': float(delta.abs().mean().cpu()),
        'effective_update_abs_mean': float(effective.abs().mean().cpu()),
        'effective_update_signed_mean': float(effective.mean().cpu()),
        'skip_high_abs_mean': float(skip_high.abs().mean().cpu()),
        'decoder_high_abs_mean': float(decoder_high.abs().mean().cpu()),
        'alpha': float(self.alpha.detach().cpu().reshape(-1)[0]),
    }

SpatialDegradationFieldModulation._summarize = sdfm_summary
GuidedSkipTransfer._summarize = gst_summary

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

def pad_to_32(image):
    h, w = image.shape[-2:]
    H = ((h + 31) // 32) * 32
    W = ((w + 31) // 32) * 32
    padh = H - h
    padw = W - w
    return F.pad(image, (0, padw, 0, padh), 'reflect'), h, w, H, W

def extract_stats(model):
    result = {}
    modules = [
        ('SDFM_1_4', 'SFAD_SDFM1'),
        ('SDFM_1_2', 'SFAD_SDFM2'),
        ('GST_1_2', 'SFAD_GST1'),
        ('GST_1_1', 'SFAD_GST2'),
    ]
    for public_name, attr in modules:
        if hasattr(model, attr):
            result[public_name] = getattr(model, attr).collect_stats()
    return result

def eval_model(model, image):
    padded, h, w, H, W = pad_to_32(image)
    pred = model(padded)[2][:, :, :h, :w]
    return torch.clamp(pred, 0, 1), extract_stats(model), H, W

def lowpass(x, factor):
    h, w = x.shape[-2:]
    low = F.interpolate(x, size=(max(1, h // factor), max(1, w // factor)), mode='bilinear', align_corners=False)
    return F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)

def bands(x):
    low16 = lowpass(x, 16)
    low4 = lowpass(x, 4)
    return {
        'low': low16,
        'mid': low4 - low16,
        'high': x - low4,
    }

def edge_masks(label):
    gray = label.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3) / 8.0
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)
    flat = mag.reshape(-1)
    q50 = torch.quantile(flat, 0.50)
    q75 = torch.quantile(flat, 0.75)
    return (mag <= q50).float(), (mag >= q75).float(), float(mag.mean().detach().cpu())

def band_errors(pred, label):
    pb = bands(pred)
    lb = bands(label)
    smooth, edge, edge_mean = edge_masks(label)
    high_err = (pb['high'] - lb['high']).abs().mean(dim=1, keepdim=True)
    smooth_den = smooth.sum().clamp_min(1.0)
    edge_den = edge.sum().clamp_min(1.0)
    return {
        'low_l1': float((pb['low'] - lb['low']).abs().mean().detach().cpu()),
        'mid_l1': float((pb['mid'] - lb['mid']).abs().mean().detach().cpu()),
        'high_l1': float((pb['high'] - lb['high']).abs().mean().detach().cpu()),
        'smooth_high_l1': float((high_err * smooth).sum().detach().cpu() / smooth_den.detach().cpu()),
        'edge_high_l1': float((high_err * edge).sum().detach().cpu() / edge_den.detach().cpu()),
        'edge_density': edge_mean,
    }

def percentile(values, pct):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)

def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out

def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return pearson(ranks(list(xs)), ranks(list(ys)))

def flatten_stats(prefix, stats):
    out = {}
    for module_name, values in stats.items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                out[f'{prefix}_{module_name}_{key}'] = value
    return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

models = {
    'a0': build_official_net('base', 'Haze4K', 'original').to(device).eval(),
    'a1': build_sfad_sdfm_net('base', 'Haze4K', 'original').to(device).eval(),
    'a2': build_sfad_gst_net('base', 'Haze4K', 'original').to(device).eval(),
    'a3': build_sfad_sdfm_gst_net('base', 'Haze4K', 'original').to(device).eval(),
}
models['a0'].load_state_dict(load_state(A0), strict=True)
models['a1'].load_state_dict(load_state(A1), strict=True)
models['a2'].load_state_dict(load_state(A2), strict=True)
models['a3'].load_state_dict(load_state(A3), strict=True)

splits = {
    'trainfit128': (SPLIT_DIR / 'haze4k_train_diagnosis_trainfit128.txt').read_text().splitlines(),
    'internal256': (SPLIT_DIR / 'haze4k_train_internal_holdout256.txt').read_text().splitlines(),
}

joint_rows = []
module_rows = []
band_rows = []
start = time.time()
with torch.no_grad():
    for split_name, image_names in splits.items():
        for idx, image_name in enumerate(image_names, 1):
            hazy = TF.to_tensor(Image.open(DATA / 'train' / 'haze' / image_name).convert('RGB')).unsqueeze(0).to(device)
            label = TF.to_tensor(Image.open(label_path_for(image_name)).convert('RGB')).unsqueeze(0).to(device)
            preds = {}
            stats = {}
            shapes = {}
            for key, model in models.items():
                pred, model_stats, H, W = eval_model(model, hazy)
                preds[key] = pred
                stats[key] = model_stats
                shapes[key] = (H, W)
            metrics = {}
            for key, pred in preds.items():
                metrics[key] = {
                    'psnr': psnr(pred, label),
                    'gt_l1': float(F.l1_loss(pred, label).detach().cpu()),
                    'bands': band_errors(pred, label),
                }
            down_ratio = max(1, round(min(shapes['a0']) / 256))
            pooled_label = F.adaptive_avg_pool2d(label, (int(shapes['a0'][0] / down_ratio), int(shapes['a0'][1] / down_ratio)))
            for key, pred in preds.items():
                pooled_pred = F.adaptive_avg_pool2d(pred, pooled_label.shape[-2:])
                metrics[key]['ssim'] = float(ssim(pooled_pred, pooled_label, data_range=1, size_average=False).mean().detach().cpu())

            a1_delta = metrics['a1']['psnr'] - metrics['a0']['psnr']
            a2_delta = metrics['a2']['psnr'] - metrics['a0']['psnr']
            a3_delta = metrics['a3']['psnr'] - metrics['a0']['psnr']
            expected = a1_delta + a2_delta
            interaction = a3_delta - expected
            input_gt_l1 = float(F.l1_loss(hazy, label).detach().cpu())
            texture_energy = float(bands(label)['high'].abs().mean().detach().cpu())
            joint_row = {
                'split': split_name,
                'image_id': image_name,
                'a0_psnr': metrics['a0']['psnr'],
                'a1_delta_psnr': a1_delta,
                'a2_delta_psnr': a2_delta,
                'a3_delta_psnr': a3_delta,
                'expected_additive_delta_psnr': expected,
                'interaction_delta_psnr': interaction,
                'a1_delta_ssim': metrics['a1']['ssim'] - metrics['a0']['ssim'],
                'a2_delta_ssim': metrics['a2']['ssim'] - metrics['a0']['ssim'],
                'a3_delta_ssim': metrics['a3']['ssim'] - metrics['a0']['ssim'],
                'input_gt_l1': input_gt_l1,
                'a0_gt_l1': metrics['a0']['gt_l1'],
                'label_texture_high_abs_mean': texture_energy,
                'edge_density': metrics['a0']['bands']['edge_density'],
            }
            joint_rows.append(joint_row)

            mod_row = {'split': split_name, 'image_id': image_name}
            mod_row.update(flatten_stats('a1', stats['a1']))
            mod_row.update(flatten_stats('a2', stats['a2']))
            mod_row.update(flatten_stats('a3', stats['a3']))
            module_rows.append(mod_row)

            for key in ('a1', 'a2', 'a3'):
                row = {'split': split_name, 'image_id': image_name, 'candidate': key}
                for metric_name in ('low_l1', 'mid_l1', 'high_l1', 'smooth_high_l1', 'edge_high_l1'):
                    row[f'a0_{metric_name}'] = metrics['a0']['bands'][metric_name]
                    row[f'candidate_{metric_name}'] = metrics[key]['bands'][metric_name]
                    row[f'delta_{metric_name}'] = metrics[key]['bands'][metric_name] - metrics['a0']['bands'][metric_name]
                row['candidate_delta_psnr'] = joint_row[f'{key}_delta_psnr']
                row['interaction_delta_psnr'] = interaction if key == 'a3' else ''
                row['edge_density'] = metrics['a0']['bands']['edge_density']
                band_rows.append(row)
            if idx % 32 == 0:
                subset = [r for r in joint_rows if r['split'] == split_name]
                print(f'audit {split_name} {idx}/{len(image_names)} mean_a3={statistics.mean(r["a3_delta_psnr"] for r in subset):.6f} mean_interaction={statistics.mean(r["interaction_delta_psnr"] for r in subset):.6f}', flush=True)

def write_csv(path, rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

write_csv(EVID / 'joint_delta_matrix.csv', joint_rows)
write_csv(EVID / 'joint_delta_matrix_trainfit128.csv', [r for r in joint_rows if r['split'] == 'trainfit128'])
write_csv(EVID / 'module_interaction_stats.csv', module_rows)
write_csv(EVID / 'band_error_report.csv', band_rows)

def split_summary(split_name):
    rows = [r for r in joint_rows if r['split'] == split_name]
    both_pos = [r for r in rows if r['a1_delta_psnr'] > 0 and r['a2_delta_psnr'] > 0]
    a3_neg_on_both = [r for r in both_pos if r['a3_delta_psnr'] < 0]
    return {
        'count': len(rows),
        'a1_mean_delta_psnr': statistics.mean(r['a1_delta_psnr'] for r in rows),
        'a2_mean_delta_psnr': statistics.mean(r['a2_delta_psnr'] for r in rows),
        'a3_mean_delta_psnr': statistics.mean(r['a3_delta_psnr'] for r in rows),
        'expected_additive_mean_delta_psnr': statistics.mean(r['expected_additive_delta_psnr'] for r in rows),
        'interaction_mean_delta_psnr': statistics.mean(r['interaction_delta_psnr'] for r in rows),
        'interaction_median_delta_psnr': statistics.median(r['interaction_delta_psnr'] for r in rows),
        'interaction_p5_delta_psnr': percentile([r['interaction_delta_psnr'] for r in rows], 5),
        'interaction_p95_delta_psnr': percentile([r['interaction_delta_psnr'] for r in rows], 95),
        'a3_positive_ratio': sum(r['a3_delta_psnr'] > 0 for r in rows) / len(rows),
        'both_a1_a2_positive_count': len(both_pos),
        'a3_negative_given_both_positive_ratio': (len(a3_neg_on_both) / len(both_pos)) if both_pos else None,
        'a3_mean_delta_on_both_positive': statistics.mean(r['a3_delta_psnr'] for r in both_pos) if both_pos else None,
        'interaction_mean_on_both_positive': statistics.mean(r['interaction_delta_psnr'] for r in both_pos) if both_pos else None,
    }

internal_rows = [r for r in joint_rows if r['split'] == 'internal256']
module_by_key = {f"{r['split']}::{r['image_id']}": r for r in module_rows}
for r in joint_rows:
    m = module_by_key[f"{r['split']}::{r['image_id']}"]
    r['a1_sdfm_1_2_R_std'] = m.get('a1_SDFM_1_2_R_std', float('nan'))
    r['a3_sdfm_1_2_R_std'] = m.get('a3_SDFM_1_2_R_std', float('nan'))
    r['a2_gst_1_2_effective'] = m.get('a2_GST_1_2_effective_update_abs_mean', float('nan'))
    r['a3_gst_1_2_effective'] = m.get('a3_GST_1_2_effective_update_abs_mean', float('nan'))

corr_vars = ['a1_delta_psnr', 'a2_delta_psnr', 'a3_delta_psnr', 'input_gt_l1', 'a0_gt_l1', 'label_texture_high_abs_mean', 'edge_density', 'a1_sdfm_1_2_R_std', 'a3_sdfm_1_2_R_std', 'a2_gst_1_2_effective', 'a3_gst_1_2_effective']
correlations = {}
for var in corr_vars:
    correlations[var] = {
        'pearson_vs_interaction': pearson([r[var] for r in internal_rows], [r['interaction_delta_psnr'] for r in internal_rows]),
        'spearman_vs_interaction': spearman([r[var] for r in internal_rows], [r['interaction_delta_psnr'] for r in internal_rows]),
    }

band_summary = {}
for candidate in ('a1', 'a2', 'a3'):
    rows = [r for r in band_rows if r['split'] == 'internal256' and r['candidate'] == candidate]
    band_summary[candidate] = {
        metric: statistics.mean(r[f'delta_{metric}'] for r in rows)
        for metric in ('low_l1', 'mid_l1', 'high_l1', 'smooth_high_l1', 'edge_high_l1')
    }

module_means = {}
for prefix in ('a1_SDFM_1_2', 'a1_SDFM_1_4', 'a2_GST_1_2', 'a2_GST_1_1', 'a3_SDFM_1_2', 'a3_SDFM_1_4', 'a3_GST_1_2', 'a3_GST_1_1'):
    rows = [r for r in module_rows if r['split'] == 'internal256']
    vals = {}
    for key in rows[0]:
        if key.startswith(prefix):
            numeric = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if numeric:
                vals[key.replace(prefix + '_', '')] = statistics.mean(numeric)
    module_means[prefix] = vals

report = {
    'route_id': 'haze4k_v4_4_bottleneck_diagnosis_20260708',
    'audit': 'after_a3_bottleneck_diagnosis',
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'splits': {name: split_summary(name) for name in splits},
    'correlations_internal256': correlations,
    'band_summary_internal256': band_summary,
    'module_means_internal256': module_means,
    'runtime_sec': time.time() - start,
}
(EVID / 'correlation_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')

internal_summary = report['splits']['internal256']
trainfit_summary = report['splits']['trainfit128']
scale_md = f"""# v4.4 Scale Collision Report

Route id: `haze4k_v4_4_bottleneck_diagnosis_20260708`

Locked test touched: `false`

## Primary Internal256 Result

- A1 mean delta PSNR: `{internal_summary['a1_mean_delta_psnr']:.6f}`
- A2 mean delta PSNR: `{internal_summary['a2_mean_delta_psnr']:.6f}`
- A3 mean delta PSNR: `{internal_summary['a3_mean_delta_psnr']:.6f}`
- Expected additive mean: `{internal_summary['expected_additive_mean_delta_psnr']:.6f}`
- Mean interaction delta: `{internal_summary['interaction_mean_delta_psnr']:.6f}`
- A3 positive ratio: `{internal_summary['a3_positive_ratio']:.6f}`
- Both A1/A2 positive count: `{internal_summary['both_a1_a2_positive_count']}`
- A3 negative given both A1/A2 positive: `{internal_summary['a3_negative_given_both_positive_ratio']}`

## Legacy Trainfit128 Check

- A3 mean delta PSNR: `{trainfit_summary['a3_mean_delta_psnr']:.6f}`
- Mean interaction delta: `{trainfit_summary['interaction_mean_delta_psnr']:.6f}`

## Module Read

- A1 SDFM_1_2 R_std mean: `{module_means['a1_SDFM_1_2'].get('R_std', float('nan')):.6f}`
- A3 SDFM_1_2 R_std mean: `{module_means['a3_SDFM_1_2'].get('R_std', float('nan')):.6f}`
- A2 GST_1_2 effective update mean: `{module_means['a2_GST_1_2'].get('effective_update_abs_mean', float('nan')):.8f}`
- A3 GST_1_2 effective update mean: `{module_means['a3_GST_1_2'].get('effective_update_abs_mean', float('nan')):.8f}`

Interpretation: negative interaction on `internal_holdout256` supports the after-A3 bottleneck diagnosis. If A3 is also negative when A1 and A2 are both positive, the strongest hypothesis is same-scale intervention collision rather than insufficient training.
"""
(EVID / 'scale_collision_report.md').write_text(scale_md, encoding='utf-8')

worst = sorted(internal_rows, key=lambda r: r['interaction_delta_psnr'])[:30]
best = sorted(internal_rows, key=lambda r: r['interaction_delta_psnr'], reverse=True)[:10]
atlas = ['# v4.4 Failure Atlas After A3', '', 'Locked test touched: `false`', '', '## Worst Internal256 Interaction Cases', '']
for r in worst:
    atlas.append(f"- `{r['image_id']}`: interaction `{r['interaction_delta_psnr']:.6f}`, A1 `{r['a1_delta_psnr']:.6f}`, A2 `{r['a2_delta_psnr']:.6f}`, A3 `{r['a3_delta_psnr']:.6f}`, input_gt_l1 `{r['input_gt_l1']:.6f}`, texture `{r['label_texture_high_abs_mean']:.6f}`")
atlas += ['', '## Best Internal256 Interaction Cases', '']
for r in best:
    atlas.append(f"- `{r['image_id']}`: interaction `{r['interaction_delta_psnr']:.6f}`, A1 `{r['a1_delta_psnr']:.6f}`, A2 `{r['a2_delta_psnr']:.6f}`, A3 `{r['a3_delta_psnr']:.6f}`")
(EVID / 'failure_atlas_after_a3.md').write_text('\n'.join(atlas) + '\n', encoding='utf-8')

decision = 'AUTHORIZE_V45_AND_V46_INDEPENDENT_FROM_ANCHOR'
if internal_summary['a3_mean_delta_psnr'] >= 0 or internal_summary['interaction_mean_delta_psnr'] > -0.02:
    decision = 'DIAGNOSIS_WEAK_REVIEW_BEFORE_MODEL_ROUTES'
decision_md = f"""# v4.4 Decision After Diagnosis

Decision: `{decision}`

Locked test touched: `false`

Internal256 A3 mean delta PSNR: `{internal_summary['a3_mean_delta_psnr']:.6f}`

Internal256 mean interaction delta: `{internal_summary['interaction_mean_delta_psnr']:.6f}`

Band summary internal256:

```json
{json.dumps(band_summary, indent=2, sort_keys=True)}
```

If authorized, v4.5 and v4.6 must branch from the official architecture anchor, train only on `haze4k_train_adapter_train.txt`, audit on `haze4k_train_internal_holdout256.txt`, and keep locked test blocked.
"""
(EVID / 'decision_after_diagnosis.md').write_text(decision_md, encoding='utf-8')

print(json.dumps({
    'route_id': 'haze4k_v4_4_bottleneck_diagnosis_20260708',
    'pass': True,
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'internal256': internal_summary,
    'decision': decision,
    'runtime_sec': report['runtime_sec'],
}, indent=2, sort_keys=True))
PYCODE
rc=$?
set -e
echo "audit_done rc=$rc v4_4_bottleneck_diagnosis $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V4_4_BOTTLENECK_DIAGNOSIS_OK" | tee -a "$STATUS"
else
  echo "V4_4_BOTTLENECK_DIAGNOSIS_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
