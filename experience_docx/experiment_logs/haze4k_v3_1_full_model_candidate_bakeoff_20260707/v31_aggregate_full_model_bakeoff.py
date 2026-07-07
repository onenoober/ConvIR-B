#!/usr/bin/env python3
import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim

V237 = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv')
V239 = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-39-convirl-same-family-teacher-contract-audit/experience_docx/experiment_logs/haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706/v239_p0_convirl_fullimage_teacher_sweep_per_image.csv')
UDP = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_2_fsudp_full_per_image.csv')
OUT = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707')

SEVERE = -0.20
STRONG_REG = -0.05
USEFUL = 0.20
GAIN = 0.0


def fnum(x):
    if x is None or x == '':
        return None
    return float(x)


def percentile(values, pct):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def cvar(values, pct=5):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    n = max(1, math.ceil(len(vals) * pct / 100.0))
    return statistics.mean(vals[:n])


def load_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def first_by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], r)
    return out


def image_to_tensor(path):
    img = Image.open(path).convert('RGB')
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    data = data.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float().div_(255.0)
    return data.unsqueeze(0)


def load_prediction_tensor(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ('pred', 'output', 'tensor', 'image'):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f'Unsupported tensor cache object at {path}: {type(obj)}')
    if obj.ndim == 3:
        obj = obj.unsqueeze(0)
    if obj.ndim != 4:
        raise ValueError(f'Unsupported tensor shape at {path}: {tuple(obj.shape)}')
    return obj.float().cpu().clamp(0, 1)


def compute_ssim(pred, gt):
    h, w = gt.shape[2], gt.shape[3]
    pred = pred[:, :, :h, :w].clamp(0, 1)
    factor = max(1, round(min(h, w) / 256))
    pred_small = F.adaptive_avg_pool2d(pred, (int(h / factor), int(w / factor)))
    gt_small = F.adaptive_avg_pool2d(gt, (int(h / factor), int(w / factor)))
    return float(ssim(pred_small, gt_small, data_range=1, size_average=False).mean().item())


def maybe_compute_ssims(records, candidate_name, path_key, gt_key):
    vals = []
    failures = []
    for name, rec in records.items():
        try:
            pred_path = rec.get(path_key, '')
            gt_path = rec.get(gt_key, '')
            if not pred_path or not gt_path:
                continue
            vals.append((name, compute_ssim(load_prediction_tensor(pred_path), image_to_tensor(gt_path))))
        except Exception as exc:
            failures.append({'image_name': name, 'candidate': candidate_name, 'error': repr(exc)})
    return dict(vals), failures


def metric_row(candidate, psnrs, ssims, a0_psnrs, buckets, strong_flags):
    names = sorted(psnrs)
    deltas = [psnrs[n] - a0_psnrs[n] for n in names]
    hard = [n for n in names if buckets[n].get('hardness_bucket') == 'hard']
    easy = [n for n in names if buckets[n].get('easy_bucket') == 'easy_top25']
    strong = [n for n in names if strong_flags[n]]
    strong_reg = [n for n in strong if (psnrs[n] - a0_psnrs[n]) <= STRONG_REG]
    severe = [n for n in names if (psnrs[n] - a0_psnrs[n]) <= SEVERE]
    return {
        'candidate': candidate,
        'count': len(names),
        'absolute_psnr_mean': mean([psnrs[n] for n in names]),
        'absolute_ssim_mean': mean([ssims.get(n) for n in names]),
        'delta_vs_a0_mean': mean(deltas),
        'delta_vs_a0_median': statistics.median(deltas),
        'delta_vs_a0_p01': percentile(deltas, 1),
        'delta_vs_a0_p05': percentile(deltas, 5),
        'delta_vs_a0_cvar5': cvar(deltas, 5),
        'delta_vs_a0_worst': min(deltas),
        'delta_vs_a0_p95': percentile(deltas, 95),
        'hard_bottom25_delta': mean([psnrs[n] - a0_psnrs[n] for n in hard]),
        'easy_top25_delta': mean([psnrs[n] - a0_psnrs[n] for n in easy]),
        'hard_easy_tradeoff': (mean([psnrs[n] - a0_psnrs[n] for n in hard]) or 0) - (mean([psnrs[n] - a0_psnrs[n] for n in easy]) or 0),
        'positive_ratio_delta_gt_0': sum(d > 0 for d in deltas) / len(deltas),
        'gain_ge_0p20_count': sum(d >= USEFUL for d in deltas),
        'severe_count_delta_le_-0p20': len(severe),
        'strong_reference_count': len(strong),
        'strong_reference_regression_count_delta_le_-0p05': len(strong_reg),
        'locked_test_touched': False,
    }


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v237_rows = load_rows(V237)
    v239_rows = load_rows(V239)
    udp_rows = load_rows(UDP)

    base = first_by(v237_rows, 'image_name')
    convirl = first_by(v239_rows, 'image_name')
    udp = first_by(udp_rows, 'name')
    common = sorted(set(base) & set(convirl) & set(udp))
    if len(common) != 600:
        raise RuntimeError(f'Expected 600 common train-derived images, got {len(common)}')

    audit = {
        'route': 'haze4k_v3_1_full_model_candidate_bakeoff_20260707',
        'route_type': 'diagnostic_only_full_model_bakeoff',
        'locked_test_policy': 'locked test forbidden; all joined sources report train-derived/full-image evidence only',
        'source_files': {
            'v237_wdmamba_fullimage': str(V237),
            'v239_convirl_fullimage': str(V239),
            'v22_fsudp_fullimage': str(UDP),
        },
        'common_count': len(common),
        'forbidden_continuations': ['A0 residual rescue', 'selector', 'alpha deployment', 'bridge', 'generator', 'canary80', 'locked test'],
        'metric_contract': {
            'baseline': 'official ConvIR-B A0 same-context PSNR from v2.37/v2.39 full-image cache',
            'split': '600 train-derived Haze4K full-image samples; 300 val_regular + 300 val_hard labels where available',
            'hard_easy': 'hardness_bucket/easy_bucket from A0 same-context full-image PSNR buckets in v2.37/v2.39',
            'severe_definition': 'delta_vs_A0 <= -0.20 dB',
            'strong_reference_regression_definition': 'strong_reference_bucket and delta_vs_A0 <= -0.05 dB',
        },
    }

    a0_psnr = {n: fnum(base[n]['A0_same_context_psnr']) for n in common}
    buckets = {n: {
        'hardness_bucket': base[n].get('hardness_bucket', ''),
        'easy_bucket': base[n].get('easy_bucket', ''),
        'table_split': base[n].get('table_split', ''),
        'fold_id': base[n].get('fold_id', ''),
    } for n in common}
    strong = {n: base[n].get('strong_reference_bucket') == 'strong_reference' for n in common}

    # A0 SSIM and UDP SSIM come from the UDP full-image table; A0 PSNR is anchored to v2.37/v2.39 cache.
    candidate_psnr = {
        'A0_official_ConvIR-B': {n: a0_psnr[n] for n in common},
        'WDMamba_standalone_fullimage': {n: fnum(base[n]['WDMamba_full_psnr']) for n in common},
        'ConvIR-L_standalone_fullimage': {n: fnum(convirl[n]['ConvIRL_full_psnr']) for n in common},
        'FullUDP_standalone_fullimage': {n: fnum(udp[n]['FullUDP_PSNR']) for n in common},
    }
    raw_udp_ssim = {n: fnum(udp[n].get('FullUDP_SSIM')) for n in common}
    udp_ssim_valid = all(v is not None and 0.0 <= v <= 1.0 for v in raw_udp_ssim.values())
    candidate_ssim = {
        'A0_official_ConvIR-B': {n: fnum(udp[n].get('A0_SSIM')) for n in common},
    }
    if udp_ssim_valid:
        candidate_ssim['FullUDP_standalone_fullimage'] = raw_udp_ssim

    ssim_audit = {'computed': {}, 'failures': []}
    a0_ssim_calc, failures = maybe_compute_ssims({n: base[n] for n in common}, 'A0_official_ConvIR-B', 'A0_full_output_path', 'gt_path')
    ssim_audit['failures'].extend(failures)
    if len(a0_ssim_calc) == len(common):
        candidate_ssim['A0_official_ConvIR-B'] = a0_ssim_calc
    ssim_audit['computed']['A0_official_ConvIR-B'] = len(a0_ssim_calc)

    wd_ssim, failures = maybe_compute_ssims({n: base[n] for n in common}, 'WDMamba_standalone_fullimage', 'WDMamba_full_output_path', 'gt_path')
    ssim_audit['failures'].extend(failures)
    candidate_ssim['WDMamba_standalone_fullimage'] = wd_ssim
    ssim_audit['computed']['WDMamba_standalone_fullimage'] = len(wd_ssim)

    cl_ssim, failures = maybe_compute_ssims({n: convirl[n] for n in common}, 'ConvIR-L_standalone_fullimage', 'ConvIRL_full_output_path', 'gt_path')
    ssim_audit['failures'].extend(failures)
    candidate_ssim['ConvIR-L_standalone_fullimage'] = cl_ssim
    ssim_audit['computed']['ConvIR-L_standalone_fullimage'] = len(cl_ssim)
    ssim_audit['computed']['FullUDP_standalone_fullimage'] = len(candidate_ssim.get('FullUDP_standalone_fullimage', {}))
    ssim_audit['FullUDP_source_ssim_valid_0_to_1'] = udp_ssim_valid
    if not udp_ssim_valid:
        ssim_audit['FullUDP_ssim_note'] = 'Historical FullUDP_SSIM source values fall outside [0,1]; SSIM is omitted for FullUDP in v3.1 compact matrix.'
    audit['ssim_audit'] = ssim_audit

    metric_rows = []
    for cand in candidate_psnr:
        metric_rows.append(metric_row(cand, candidate_psnr[cand], candidate_ssim.get(cand, {}), a0_psnr, buckets, strong))
    metric_fields = list(metric_rows[0].keys())
    write_csv(OUT / 'v31_candidate_metric_matrix.csv', metric_rows, metric_fields)

    tail_rows = []
    bucket_defs = {
        'all': common,
        'hard_bottom25': [n for n in common if buckets[n]['hardness_bucket'] == 'hard'],
        'easy_top25': [n for n in common if buckets[n]['easy_bucket'] == 'easy_top25'],
        'strong_reference': [n for n in common if strong[n]],
        'val_hard': [n for n in common if buckets[n]['table_split'] == 'val_hard'],
        'val_regular': [n for n in common if buckets[n]['table_split'] == 'val_regular'],
    }
    for cand, psnrs in candidate_psnr.items():
        for bucket_name, names in bucket_defs.items():
            deltas = [psnrs[n] - a0_psnr[n] for n in names]
            tail_rows.append({
                'candidate': cand,
                'bucket': bucket_name,
                'count': len(names),
                'mean_delta': mean(deltas),
                'median_delta': statistics.median(deltas) if deltas else None,
                'p01_delta': percentile(deltas, 1),
                'p05_delta': percentile(deltas, 5),
                'cvar5_delta': cvar(deltas, 5),
                'worst_delta': min(deltas) if deltas else None,
                'severe_count_delta_le_-0p20': sum(d <= SEVERE for d in deltas),
                'positive_ratio_delta_gt_0': sum(d > 0 for d in deltas) / len(deltas) if deltas else None,
            })
    write_csv(OUT / 'v31_candidate_tail_matrix.csv', tail_rows, list(tail_rows[0].keys()))

    # Per-image table is kept cloud-only for audit; not intended for GitHub sync by default.
    per_image_path = OUT / 'v31_candidate_per_image_cloud_only.csv'
    with per_image_path.open('w', newline='', encoding='utf-8') as f:
        fields = ['image_name', 'fold_id', 'table_split', 'hardness_bucket', 'easy_bucket', 'strong_reference_bucket', 'A0_psnr']
        for cand in candidate_psnr:
            fields += [f'{cand}_psnr', f'{cand}_delta_vs_A0']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n in common:
            row = {
                'image_name': n,
                'fold_id': buckets[n]['fold_id'],
                'table_split': buckets[n]['table_split'],
                'hardness_bucket': buckets[n]['hardness_bucket'],
                'easy_bucket': buckets[n]['easy_bucket'],
                'strong_reference_bucket': 'strong_reference' if strong[n] else 'not_strong_reference',
                'A0_psnr': a0_psnr[n],
            }
            for cand, psnrs in candidate_psnr.items():
                row[f'{cand}_psnr'] = psnrs[n]
                row[f'{cand}_delta_vs_A0'] = psnrs[n] - a0_psnr[n]
            w.writerow(row)

    winner = Counter()
    oracle_psnr = {}
    for n in common:
        best = max(candidate_psnr, key=lambda c: candidate_psnr[c][n])
        winner[best] += 1
        oracle_psnr[n] = candidate_psnr[best][n]
    oracle_delta = [oracle_psnr[n] - a0_psnr[n] for n in common]
    oracle = {
        'candidate_set': list(candidate_psnr),
        'count': len(common),
        'winner_histogram': dict(winner),
        'oracle_mean_psnr': mean([oracle_psnr[n] for n in common]),
        'oracle_delta_vs_A0_mean': mean(oracle_delta),
        'oracle_delta_vs_A0_hard': mean([oracle_psnr[n] - a0_psnr[n] for n in bucket_defs['hard_bottom25']]),
        'oracle_delta_vs_A0_easy': mean([oracle_psnr[n] - a0_psnr[n] for n in bucket_defs['easy_top25']]),
        'oracle_delta_vs_A0_p05': percentile(oracle_delta, 5),
        'oracle_delta_vs_A0_cvar5': cvar(oracle_delta, 5),
        'oracle_severe_count': sum(d <= SEVERE for d in oracle_delta),
    }
    (OUT / 'v31_candidate_oracle_upper_bound.json').write_text(json.dumps(oracle, indent=2), encoding='utf-8')

    overlap_rows = []
    cands = list(candidate_psnr)
    for a in cands:
        for b in cands:
            da = {n for n in common if candidate_psnr[a][n] - a0_psnr[n] > GAIN}
            db = {n for n in common if candidate_psnr[b][n] - a0_psnr[n] > GAIN}
            ua = {n for n in common if candidate_psnr[a][n] - a0_psnr[n] >= USEFUL}
            ub = {n for n in common if candidate_psnr[b][n] - a0_psnr[n] >= USEFUL}
            sa = {n for n in common if candidate_psnr[a][n] - a0_psnr[n] <= SEVERE}
            sb = {n for n in common if candidate_psnr[b][n] - a0_psnr[n] <= SEVERE}
            overlap_rows.append({
                'candidate_a': a,
                'candidate_b': b,
                'positive_jaccard': len(da & db) / len(da | db) if (da | db) else 1.0,
                'useful_ge_0p20_jaccard': len(ua & ub) / len(ua | ub) if (ua | ub) else 1.0,
                'severe_le_-0p20_jaccard': len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0,
                'both_positive_count': len(da & db),
                'both_useful_count': len(ua & ub),
                'both_severe_count': len(sa & sb),
                'a_positive_count': len(da),
                'b_positive_count': len(db),
            })
    write_csv(OUT / 'v31_candidate_overlap_matrix.csv', overlap_rows, list(overlap_rows[0].keys()))

    cost = {
        'A0_official_ConvIR-B': {
            'params_m': 8.63, 'flops_g': 71.22, 'checkpoint': '/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl', 'source': 'official ConvIR Haze4K B'},
        'ConvIR-L_standalone_fullimage': {
            'params_m': 14.83, 'flops_g': 129.34, 'checkpoint': '/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-large.pkl', 'source': 'official ConvIR Haze4K L'},
        'WDMamba_standalone_fullimage': {
            'params_m': None, 'flops_g': None, 'checkpoint': '/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth', 'source': 'external WDMamba Haze4K checkpoint; cost not measured in this diagnostic'},
        'FullUDP_standalone_fullimage': {
            'params_m': None, 'flops_g': None, 'checkpoint': '/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt', 'source': 'UDPNet official Haze4K ConvIR_UDPNet; cost not measured in this diagnostic'},
    }
    (OUT / 'v31_candidate_cost_manifest.json').write_text(json.dumps(cost, indent=2), encoding='utf-8')

    metric_by = {r['candidate']: r for r in metric_rows}
    decision = 'COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM'
    if metric_by['WDMamba_standalone_fullimage']['delta_vs_a0_mean'] < 0.30:
        decision = 'COMPLETED_GATE_FAIL_NO_STANDALONE_MODEL_HEADROOM'
    closeout = {
        **audit,
        'decision': decision,
        'primary_result': {
            'best_mean_candidate': max(metric_rows, key=lambda r: r['delta_vs_a0_mean'])['candidate'],
            'best_mean_delta_vs_A0': max(r['delta_vs_a0_mean'] for r in metric_rows),
            'wdmamba_mean_delta_vs_A0': metric_by['WDMamba_standalone_fullimage']['delta_vs_a0_mean'],
            'wdmamba_hard_delta_vs_A0': metric_by['WDMamba_standalone_fullimage']['hard_bottom25_delta'],
            'wdmamba_easy_delta_vs_A0': metric_by['WDMamba_standalone_fullimage']['easy_top25_delta'],
            'convirl_mean_delta_vs_A0': metric_by['ConvIR-L_standalone_fullimage']['delta_vs_a0_mean'],
            'fulludp_mean_delta_vs_A0': metric_by['FullUDP_standalone_fullimage']['delta_vs_a0_mean'],
        },
        'next_authorized': ['draft v3.2 ConvIR-WD/WDMamba-informed full model line P0/P1 protocol'],
        'still_forbidden': ['locked test', 'canary80', 'A0 residual rescue', 'selector/alpha deployment', 'bridge/generator from v3.0'],
        'cloud_only_raw_table': str(per_image_path),
        'compact_outputs': [
            'v31_candidate_metric_matrix.csv',
            'v31_candidate_tail_matrix.csv',
            'v31_candidate_oracle_upper_bound.json',
            'v31_candidate_overlap_matrix.csv',
            'v31_candidate_cost_manifest.json',
            'v31_closeout.json',
        ],
    }
    (OUT / 'v31_closeout.json').write_text(json.dumps(closeout, indent=2), encoding='utf-8')

    readme = f"""# Haze4K v3.1 Full-Model Candidate Bakeoff

Status: {decision}.

Purpose: separate standard full-model quality from strict A0-dominance safe-upgrade. This route is diagnostic-only and train-derived only.

Sources: v2.37 WDMamba full-image table, v2.39 ConvIR-L full-image table, and v2.2 FullUDP full-image table joined on the same 600 train-derived image names. Locked test remains untouched.

Forbidden: no canary80, no locked test, no A0+alpha deployable selector, no bridge/generator, no v3.0 rescue.

Key result: WDMamba standalone is the strongest full-model candidate in this bakeoff. Mean delta vs official ConvIR-B A0 is {metric_by['WDMamba_standalone_fullimage']['delta_vs_a0_mean']:.4f} dB; hard/easy deltas are {metric_by['WDMamba_standalone_fullimage']['hard_bottom25_delta']:.4f}/{metric_by['WDMamba_standalone_fullimage']['easy_top25_delta']:.4f} dB. ConvIR-L standalone is also positive at {metric_by['ConvIR-L_standalone_fullimage']['delta_vs_a0_mean']:.4f} dB mean, while FullUDP standalone is negative at {metric_by['FullUDP_standalone_fullimage']['delta_vs_a0_mean']:.4f} dB mean on this joined table.

Decision: use v3.1 as evidence to pivot away from ConvIR-B A0-anchored rescue and draft v3.2 as a full model line centered on WDMamba/ConvIR-WD-style low-frequency haze modeling.

Compact artifacts:
- `v31_candidate_metric_matrix.csv`
- `v31_candidate_tail_matrix.csv`
- `v31_candidate_oracle_upper_bound.json`
- `v31_candidate_overlap_matrix.csv`
- `v31_candidate_cost_manifest.json`
- `v31_closeout.json`

Cloud-only raw table: `v31_candidate_per_image_cloud_only.csv`.
"""
    (OUT / 'README.md').write_text(readme, encoding='utf-8')

    print(json.dumps(closeout['primary_result'], indent=2))
    print('V31_AGGREGATE_OK')

if __name__ == '__main__':
    main()
