import argparse
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
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def pct(values, q):
    vals = sorted(float(v) for v in values)
    if not vals:
        return float('nan')
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def label_path_for(data_root: Path, image_name: str) -> Path:
    label_dir = data_root / 'train' / 'gt'
    stem, ext = os.path.splitext(image_name)
    candidates = [image_name]
    if '_' in stem:
        candidates.append(stem.split('_')[0] + ext)
        candidates.append(stem.split('_')[0] + '.png')
    for candidate in candidates:
        p = label_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError({'image': image_name, 'candidates': candidates})


def texture_proxy(arr: np.ndarray) -> float:
    dy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean() if arr.shape[0] > 1 else 0.0
    dx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean() if arr.shape[1] > 1 else 0.0
    return float(dx + dy)


def psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def lowpass(t: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(t, kernel_size=9, stride=1, padding=4, count_include_pad=False)


def frequency_l1(pred: torch.Tensor, label: torch.Tensor):
    pred_low = lowpass(pred)
    label_low = lowpass(label)
    pred_high = pred - pred_low
    label_high = label - label_low
    return (
        float(F.l1_loss(pred_low, label_low).detach().cpu()),
        float(F.l1_loss(pred_high, label_high).detach().cpu()),
    )


def eval_model(model, image: torch.Tensor):
    h, w = image.shape[-2:]
    H = ((h + 31) // 32) * 32
    W = ((w + 31) // 32) * 32
    padded = F.pad(image, (0, W - w, 0, H - h), 'reflect')
    pred = model(padded)[2][:, :, :h, :w]
    stats_mod = getattr(model, 'DCFSB_Bottleneck', None)
    stats = stats_mod.collect_stats() if stats_mod is not None else {}
    return torch.clamp(pred, 0, 1), stats, H, W


def summarize(rows):
    deltas = [float(r['delta_psnr']) for r in rows]
    dssim = [float(r['delta_ssim']) for r in rows]
    dlow = [float(r['delta_low_l1']) for r in rows]
    dhigh = [float(r['delta_high_l1']) for r in rows]
    return {
        'count': len(rows),
        'mean_delta_psnr': float(statistics.mean(deltas)),
        'median_delta_psnr': float(statistics.median(deltas)),
        'p1_delta_psnr': pct(deltas, 1),
        'p5_delta_psnr': pct(deltas, 5),
        'p95_delta_psnr': pct(deltas, 95),
        'positive_ratio': sum(d > 0 for d in deltas) / len(deltas),
        'mean_delta_ssim': float(statistics.mean(dssim)),
        'mean_delta_low_l1': float(statistics.mean(dlow)),
        'mean_delta_high_l1': float(statistics.mean(dhigh)),
        'high_l1_improve_ratio': sum(d < 0 for d in dhigh) / len(dhigh),
        'a0_mean_psnr': float(statistics.mean(float(r['a0_psnr']) for r in rows)),
        'candidate_mean_psnr': float(statistics.mean(float(r['candidate_psnr']) for r in rows)),
        'a0_mean_ssim': float(statistics.mean(float(r['a0_ssim']) for r in rows)),
        'candidate_mean_ssim': float(statistics.mean(float(r['candidate_ssim']) for r in rows)),
    }


def quantile_bins(rows, key):
    ordered = sorted((float(r[key]), idx) for idx, r in enumerate(rows))
    n = len(ordered)
    labels = [None] * n
    for rank, (_, idx) in enumerate(ordered):
        labels[idx] = f'q{min(3, int(rank * 4 / n)) + 1}'
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--a0', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    start = time.time()
    work = Path(args.work)
    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    a0_path = Path(args.a0)
    cand_path = Path(args.candidate)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    its = work / 'Dehazing' / 'ITS'
    sys.path.insert(0, str(its))
    from models.ConvIR import build_net as build_official_net
    from models.DCFSBConvIR import build_dcfsb_bottleneck_net

    names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f'empty split: {split_file}')
    if len(names) != len(set(names)):
        raise RuntimeError(f'duplicate names in split: {split_file}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    official = build_official_net('base', 'Haze4K', 'original').to(device).eval()
    official.load_state_dict(load_state(a0_path), strict=True)
    route = build_dcfsb_bottleneck_net('base', 'Haze4K', 'original').to(device).eval()
    route.load_state_dict(load_state(cand_path), strict=True)

    rows = []
    factor_note = 'padding factor 32; low/high L1 uses v4.6 9x9 avg-pool lowpass'
    with torch.no_grad():
        for idx, name in enumerate(names, start=1):
            input_path = data_root / 'train' / 'haze' / name
            label_path = label_path_for(data_root, name)
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            hazy_np = np.asarray(Image.open(input_path).convert('RGB'), dtype=np.float32) / 255.0
            gt_np = np.asarray(Image.open(label_path).convert('RGB'), dtype=np.float32) / 255.0
            if hazy_np.shape != gt_np.shape:
                raise RuntimeError({'shape_mismatch': name, 'hazy': hazy_np.shape, 'gt': gt_np.shape})
            hazy = TF.to_tensor(Image.fromarray((hazy_np * 255.0 + 0.5).astype(np.uint8))).unsqueeze(0).to(device)
            label = TF.to_tensor(Image.fromarray((gt_np * 255.0 + 0.5).astype(np.uint8))).unsqueeze(0).to(device)

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
            row = {
                'fold': args.fold,
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
                'input_gt_l1': float(np.abs(hazy_np - gt_np).mean()),
                'a0_error_proxy_low_plus_high_l1': float(a0_low_l1 + a0_high_l1),
                'candidate_error_proxy_low_plus_high_l1': float(cand_low_l1 + cand_high_l1),
                'input_dark_channel_mean': float(np.min(hazy_np, axis=2).mean()),
                'input_brightness_mean': float(hazy_np.mean()),
                'input_saturation_proxy': float((hazy_np.max(axis=2) - hazy_np.min(axis=2)).mean()),
                'gt_texture_proxy': texture_proxy(gt_np),
                'hazy_texture_proxy': texture_proxy(hazy_np),
            }
            for key in ['low_energy', 'high_energy', 'high_low_energy_ratio', 'low_gate_mean', 'low_gate_std', 'high_gate_mean', 'high_gate_std', 'low_update_abs_mean', 'high_update_abs_mean', 'alpha_low', 'alpha_high', 'alpha_abs_mean']:
                row[key] = float(stats.get(key, float('nan')))
            rows.append(row)
            if idx % 50 == 0 or idx == len(names):
                mean_delta = statistics.mean(float(r['delta_psnr']) for r in rows)
                print(f"fold={args.fold} progress={idx}/{len(names)} mean_delta={mean_delta:.6f}", flush=True)

    proxy_keys = [
        'input_gt_l1',
        'a0_error_proxy_low_plus_high_l1',
        'a0_psnr',
        'input_dark_channel_mean',
        'input_brightness_mean',
        'input_saturation_proxy',
        'gt_texture_proxy',
        'hazy_texture_proxy',
        'high_gate_std',
        'high_low_energy_ratio',
    ]
    for key in proxy_keys:
        labels = quantile_bins(rows, key)
        for row, label in zip(rows, labels):
            row[f'{key}_quartile'] = label

    summary = summarize(rows)
    summary.update({
        'fold': args.fold,
        'split_file': str(split_file),
        'a0_checkpoint': str(a0_path),
        'a0_sha256': sha256_file(a0_path),
        'candidate_checkpoint': str(cand_path),
        'candidate_sha256': sha256_file(cand_path),
        'metric_note': factor_note,
        'locked_test_touched': False,
        'test_split_enumerated': False,
        'runtime_sec': time.time() - start,
    })

    fields = [
        'fold', 'image_id', 'delta_psnr', 'delta_ssim', 'delta_low_l1', 'delta_high_l1',
        'a0_psnr', 'candidate_psnr', 'a0_ssim', 'candidate_ssim',
        'a0_low_l1', 'candidate_low_l1', 'a0_high_l1', 'candidate_high_l1',
        'input_gt_l1', 'a0_error_proxy_low_plus_high_l1', 'candidate_error_proxy_low_plus_high_l1',
        'input_dark_channel_mean', 'input_brightness_mean', 'input_saturation_proxy',
        'gt_texture_proxy', 'hazy_texture_proxy', 'high_gate_std', 'high_low_energy_ratio',
        'high_gate_mean', 'low_gate_mean', 'alpha_abs_mean',
        'input_gt_l1_quartile', 'a0_error_proxy_low_plus_high_l1_quartile', 'a0_psnr_quartile',
        'input_dark_channel_mean_quartile', 'input_brightness_mean_quartile', 'input_saturation_proxy_quartile',
        'gt_texture_proxy_quartile', 'hazy_texture_proxy_quartile', 'high_gate_std_quartile', 'high_low_energy_ratio_quartile',
    ]
    per_image_path = out_dir / 'val_per_image_compact.csv'
    with per_image_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r['image_id']):
            writer.writerow({key: row[key] for key in fields})

    bin_rows = []
    for key in proxy_keys:
        for label in ['q1', 'q2', 'q3', 'q4']:
            subset = [r for r in rows if r[f'{key}_quartile'] == label]
            vals = [float(r[key]) for r in subset]
            sm = summarize(subset)
            bin_rows.append({
                'fold': args.fold,
                'proxy': key,
                'bin': label,
                'count': len(subset),
                'proxy_min': min(vals),
                'proxy_max': max(vals),
                'proxy_mean': float(statistics.mean(vals)),
                'mean_delta_psnr': sm['mean_delta_psnr'],
                'median_delta_psnr': sm['median_delta_psnr'],
                'p5_delta_psnr': sm['p5_delta_psnr'],
                'positive_ratio': sm['positive_ratio'],
                'mean_delta_high_l1': sm['mean_delta_high_l1'],
            })
    with (out_dir / 'val_proxy_bins.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    worst = sorted(rows, key=lambda r: float(r['delta_psnr']))[:max(16, min(32, len(rows)))]
    worst_lines = ['| Rank | Image | dPSNR | A0 PSNR | Candidate PSNR | input-GT L1 | A0 proxy L1 | saturation | high gate std |', '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i, r in enumerate(worst[:32], start=1):
        worst_lines.append(f"| {i} | `{r['image_id']}` | {r['delta_psnr']:.6f} | {r['a0_psnr']:.6f} | {r['candidate_psnr']:.6f} | {r['input_gt_l1']:.6f} | {r['a0_error_proxy_low_plus_high_l1']:.6f} | {r['input_saturation_proxy']:.6f} | {r['high_gate_std']:.6f} |")
    atlas = f"""# v4.8 Fold {args.fold} Failure Atlas

Split: `{split_file}`

Locked test touched/enumerated: `false` / `false`

Mean dPSNR: `{summary['mean_delta_psnr']:.6f}`; positive ratio: `{summary['positive_ratio']:.6f}`; p5: `{summary['p5_delta_psnr']:.6f}`; mean dHighL1: `{summary['mean_delta_high_l1']:.10f}`.

## Worst Images

""" + '\n'.join(worst_lines) + '\n'
    (out_dir / 'val_failure_atlas.md').write_text(atlas, encoding='utf-8')
    (out_dir / 'val_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"V48_FOLD{args.fold}_OOF_EVAL_OK", flush=True)


if __name__ == '__main__':
    main()
