#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF

TOOLS = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked/experience_docx/tools')
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from eval_udpnet_v15_phase0_repro import load_convir_builders, load_a0_model, infer_one  # noqa: E402

DEFAULT_ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
SEVERE = -0.20


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == '':
            return default
        return float(x)
    except Exception:
        return default


def alpha_key(alpha: float) -> str:
    return (('a%.6f' % alpha).rstrip('0').rstrip('.')).replace('.', 'p')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), 'reflect'), h + ph, w + pw


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f'none of {names} exists under {root}')


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if '_' in stem:
        clean = stem.split('_', 1)[0]
        candidates.extend([f'{clean}{ext}', f'{clean}.png'])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f'no GT for {image_name}')


def list_images(data_dir: Path, split: str) -> list[str]:
    input_dir = first_existing_dir(data_dir / split, ('IN', 'haze', 'hazy'))
    return sorted(p.name for p in input_dir.iterdir() if p.is_file())


def load_sample(data_dir: Path, split: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = data_dir / split
    input_dir = first_existing_dir(root, ('IN', 'haze', 'hazy'))
    gt_dir = first_existing_dir(root, ('GT', 'gt'))
    hazy = Image.open(input_dir / name).convert('RGB')
    gt = Image.open(label_path(gt_dir, name)).convert('RGB')
    return TVF.to_tensor(hazy).float(), TVF.to_tensor(gt).float()


def load_wdmamba(repo: Path, checkpoint: Path, device: torch.device):
    try:
        import transformers.generation as tg
        for name in ['GreedySearchDecoderOnlyOutput', 'SampleDecoderOnlyOutput']:
            if not hasattr(tg, name):
                setattr(tg, name, type(name, (object,), {}))
    except Exception:
        pass

    def pkg(name: str, path: Path) -> None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod

    def load_mod(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for key in list(sys.modules):
        if key == 'basicsr' or key.startswith('basicsr.'):
            del sys.modules[key]
    pkg('basicsr', repo / 'basicsr')
    pkg('basicsr.archs', repo / 'basicsr/archs')
    pkg('basicsr.utils', repo / 'basicsr/utils')
    load_mod('basicsr.utils.registry', repo / 'basicsr/utils/registry.py')
    load_mod('basicsr.archs.Ublock', repo / 'basicsr/archs/Ublock.py')
    load_mod('basicsr.archs.detail_enhance_net', repo / 'basicsr/archs/detail_enhance_net.py')
    load_mod('basicsr.archs.wavelet', repo / 'basicsr/archs/wavelet.py')
    wavemamba = load_mod('basicsr.archs.wavemamba_arch', repo / 'basicsr/archs/wavemamba_arch.py')
    model = wavemamba.WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['params'], strict=True)
    model.eval()
    return model


def infer_wdmamba(model, rgb: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
    x, hp, wp = pad_to(rgb, 4)
    out = model.restoration_network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp


def evaluate(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_names = list_images(args.data_dir, args.data_split)
    names = [name for idx, name in enumerate(all_names) if idx % args.shard_count == args.shard_index]
    if args.max_images:
        names = names[:args.max_images]
    if not names:
        raise RuntimeError(f'empty shard {args.shard_index}/{args.shard_count}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, build_convir = load_convir_builders(args.convir_its_dir)
    a0 = load_a0_model(build_convir, args.a0_checkpoint, device)
    wdmamba = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    alphas = sorted({round(float(a), 6) for a in args.alphas})
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy, gt = load_sample(args.data_dir, args.data_split, name)
            rgb = hazy.unsqueeze(0).to(device)
            lab = gt.unsqueeze(0).to(device)
            _, _, h, w = rgb.shape
            rgb32, hp32, wp32 = pad_to(rgb, 32)
            a0_pred = infer_one(a0, rgb32, h, w)
            a0_psnr, a0_ssim = metric(a0_pred, lab, hp32, wp32)
            expert_pred, hp4, wp4 = infer_wdmamba(wdmamba, rgb, h, w)
            expert_psnr, expert_ssim = metric(expert_pred, lab, hp4, wp4)
            expert_grid_psnr, expert_grid_ssim = metric(expert_pred, lab, hp32, wp32)
            row: dict[str, Any] = {
                'image_id': Path(name).stem,
                'name': name,
                'split': args.data_split,
                'width': w,
                'height': h,
                'shard_index': args.shard_index,
                'shard_count': args.shard_count,
                'A0_PSNR': a0_psnr,
                'A0_SSIM': a0_ssim,
                'WDMamba_PSNR': expert_psnr,
                'WDMamba_SSIM': expert_ssim,
                'WDMamba_grid32_PSNR': expert_grid_psnr,
                'WDMamba_grid32_SSIM': expert_grid_ssim,
                'WDMamba_dPSNR': expert_psnr - a0_psnr,
                'WDMamba_dSSIM': expert_ssim - a0_ssim,
                'WDMamba_grid32_dSSIM': expert_grid_ssim - a0_ssim,
            }
            for alpha in alphas:
                key = alpha_key(alpha)
                pred = torch.clamp(a0_pred + alpha * (expert_pred - a0_pred), 0, 1)
                psnr, ss = metric(pred, lab, hp32, wp32)
                row[f'alpha_{key}_PSNR'] = psnr
                row[f'alpha_{key}_SSIM'] = ss
                row[f'alpha_{key}_dPSNR'] = psnr - a0_psnr
                row[f'alpha_{key}_dSSIM'] = ss - a0_ssim
            rows.append(row)
            if idx % args.print_freq == 0 or idx == len(names):
                k = alpha_key(0.375)
                mean = statistics.mean(fnum(r[f'alpha_{k}_dPSNR']) for r in rows)
                print(f'progress shard={args.shard_index}/{args.shard_count} {idx}/{len(names)} mean_wd0375={mean:.6f} elapsed={time.time()-t0:.1f}s', flush=True)
    write_csv(args.out_dir / f'{args.prefix}_per_image.csv', rows)
    manifest = {
        'mode': 'evaluate',
        'count': len(rows),
        'shard_index': args.shard_index,
        'shard_count': args.shard_count,
        'alpha_grid': alphas,
        'metric_protocol': 'v2.2 locked one-shot compatible: alpha candidates use A0 32-pad SSIM; WDMamba standalone also records 4-pad endpoint SSIM',
    }
    (args.out_dir / f'{args.prefix}_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'V210_EVAL_SHARD_OK shard={args.shard_index} rows={len(rows)}')


def summarize_alpha(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    key = alpha_key(alpha)
    n = len(rows)
    ps = [fnum(r[f'alpha_{key}_PSNR']) for r in rows]
    ss = [fnum(r[f'alpha_{key}_SSIM']) for r in rows]
    ds = [fnum(r[f'alpha_{key}_dPSNR']) for r in rows]
    dss = [fnum(r[f'alpha_{key}_dSSIM']) for r in rows]
    a0 = [fnum(r['A0_PSNR']) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    bucket = max(1, n // 4)
    hard = [ds[i] for i in order[:bucket]]
    easy = [ds[i] for i in order[-bucket:]]
    severe = sum(d <= SEVERE for d in ds)
    pos = sum(d > 0 for d in ds)
    out = {
        'alpha': alpha,
        'label': 'A0 / ConvIR-B' if alpha == 0 else 'WD0375' if abs(alpha - 0.375) < 1e-9 else 'WDMamba full' if alpha == 1 else f'alpha={alpha:.3f}',
        'count': n,
        'mean_PSNR': statistics.mean(ps),
        'mean_SSIM_grid32': statistics.mean(ss),
        'mean_dPSNR': statistics.mean(ds),
        'hard_bottom25_dPSNR': statistics.mean(hard),
        'easy_top25_dPSNR': statistics.mean(easy),
        'mean_dSSIM_grid32': statistics.mean(dss),
        'positive_ratio': pos / n,
        'nonnegative_ratio': sum(d >= 0 for d in ds) / n,
        'severe_loss_count': severe,
        'severe_loss_per_600': severe / n * 600.0,
        'worst_dPSNR': min(ds),
        'best_dPSNR': max(ds),
        'median_dPSNR': statistics.median(ds),
    }
    if alpha == 1.0:
        out['mean_SSIM_v22_wdmamba_endpoint'] = statistics.mean(fnum(r['WDMamba_SSIM']) for r in rows)
        out['mean_dSSIM_v22_wdmamba_endpoint'] = statistics.mean(fnum(r['WDMamba_dSSIM']) for r in rows)
    return out


def aggregate(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for p in args.input_csvs:
        rows.extend(read_csv(Path(p)))
    rows = sorted(rows, key=lambda r: r['image_id'])
    if len(rows) != args.expected_count:
        raise RuntimeError(f'expected {args.expected_count} rows got {len(rows)}')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise RuntimeError('duplicate image_id detected')
    alphas = sorted({round(float(a), 6) for a in args.alphas})
    write_csv(args.out_dir / f'{args.prefix}_per_image.csv', rows)
    alpha_rows = [summarize_alpha(rows, a) for a in alphas]
    write_csv(args.out_dir / f'{args.prefix}_absolute_metrics.csv', alpha_rows)
    compact_keys = ['alpha', 'label', 'count', 'mean_PSNR', 'mean_SSIM_grid32', 'mean_dPSNR', 'hard_bottom25_dPSNR', 'easy_top25_dPSNR', 'mean_dSSIM_grid32', 'positive_ratio', 'severe_loss_per_600', 'worst_dPSNR', 'mean_SSIM_v22_wdmamba_endpoint']
    write_csv(args.out_dir / f'{args.prefix}_compact_metrics.csv', [{k: r.get(k, '') for k in compact_keys} for r in alpha_rows])

    v22 = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked/experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/v22_locked_wd0375_one_shot_per_image.csv')
    parity: dict[str, Any] = {'v22_path': str(v22), 'available': v22.is_file()}
    if v22.is_file():
        old = read_csv(v22)
        oldmap = {Path(r.get('name', '')).stem: r for r in old}
        pairs = [(r, oldmap[r['image_id']]) for r in rows if r['image_id'] in oldmap]
        def max_abs(new_key: str, old_key: str) -> float:
            return max(abs(fnum(n[new_key]) - fnum(o[old_key])) for n, o in pairs)
        def mean_abs(new_key: str, old_key: str) -> float:
            return statistics.mean(abs(fnum(n[new_key]) - fnum(o[old_key])) for n, o in pairs)
        parity.update({
            'v22_count': len(old),
            'matched_count': len(pairs),
            'max_abs_A0_PSNR': max_abs('A0_PSNR', 'A0_PSNR'),
            'max_abs_A0_SSIM': max_abs('A0_SSIM', 'A0_SSIM'),
            'max_abs_WDMamba_PSNR': max_abs('WDMamba_PSNR', 'WDMamba_PSNR'),
            'max_abs_WDMamba_SSIM': max_abs('WDMamba_SSIM', 'WDMamba_SSIM'),
            'max_abs_WD0375_PSNR': max_abs('alpha_a0p375_PSNR', 'WD0375_PSNR'),
            'max_abs_WD0375_SSIM': max_abs('alpha_a0p375_SSIM', 'WD0375_SSIM'),
            'mean_abs_A0_PSNR': mean_abs('A0_PSNR', 'A0_PSNR'),
            'mean_abs_A0_SSIM': mean_abs('A0_SSIM', 'A0_SSIM'),
            'mean_abs_WDMamba_PSNR': mean_abs('WDMamba_PSNR', 'WDMamba_PSNR'),
            'mean_abs_WDMamba_SSIM': mean_abs('WDMamba_SSIM', 'WDMamba_SSIM'),
            'mean_abs_WD0375_PSNR': mean_abs('alpha_a0p375_PSNR', 'WD0375_PSNR'),
            'mean_abs_WD0375_SSIM': mean_abs('alpha_a0p375_SSIM', 'WD0375_SSIM'),
        })
        parity['parity_pass'] = parity['matched_count'] == args.expected_count and max(
            parity['max_abs_A0_PSNR'], parity['max_abs_A0_SSIM'], parity['max_abs_WDMamba_PSNR'],
            parity['max_abs_WDMamba_SSIM'], parity['max_abs_WD0375_PSNR'], parity['max_abs_WD0375_SSIM']
        ) < 1e-8
    else:
        parity['parity_pass'] = False
    (args.out_dir / 'v210_parity_with_v22_wd0375.json').write_text(json.dumps(parity, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    endpoint = next(r for r in alpha_rows if fnum(r['alpha']) == 1.0)
    summary = {
        'route': 'haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616',
        'decision': 'V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY',
        'state_label': 'COMPLETED_GATE_PASS' if parity.get('parity_pass') else 'COMPLETED_GATE_FAIL',
        'locked_test_touched': True,
        'locked_scope': 'Haze4K test split, 1000 images',
        'locked_policy': 'diagnostic alpha-grid audit only; do not select or retune alpha from this locked grid',
        'metric_protocol': 'v2.2 locked one-shot compatible. A0 and alpha candidates use the same A0 32-pad SSIM convention as v2.2 WD0375; WDMamba standalone endpoint SSIM is recorded separately and matches v2.2.',
        'count': len(rows),
        'alpha_grid': alphas,
        'a0_checkpoint': str(args.a0_checkpoint),
        'a0_sha256': sha256(args.a0_checkpoint),
        'wdmamba_checkpoint': str(args.wdmamba_checkpoint),
        'wdmamba_sha256': sha256(args.wdmamba_checkpoint),
        'alpha_summaries': alpha_rows,
        'alpha1_grid32_SSIM': endpoint['mean_SSIM_grid32'],
        'alpha1_v22_wdmamba_endpoint_SSIM': endpoint.get('mean_SSIM_v22_wdmamba_endpoint'),
        'v22_parity': parity,
        'invalidated_preliminary_note': 'A preliminary v28/NH metric reuse run was discarded before sync because A0/WD0375 SSIM did not parity-match v2.2; final evidence is this v2.2-compatible rerun only.',
    }
    (args.out_dir / f'{args.prefix}_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    lines = [
        '# Haze4K v2.10 Locked WDMamba Alpha Grid',
        '',
        'Decision: `V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`',
        '',
        'This route evaluates the predeclared WDMamba residual-shrinkage alpha grid on the Haze4K locked test split (`1000` images). It is a diagnostic audit only and must not be used to select or retune alpha.',
        '',
        'Metric protocol: v2.2 locked one-shot compatible. A0 and alpha candidates use the same A0 32-pad SSIM convention as v2.2 WD0375; WDMamba standalone endpoint SSIM is recorded separately and parity-matches v2.2.',
        '',
        '## Absolute Metrics',
        '',
        '| alpha | label | PSNR | SSIM grid32 | WDMamba endpoint SSIM | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 | worst dPSNR |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in alpha_rows:
        endpoint_ssim = r.get('mean_SSIM_v22_wdmamba_endpoint', '')
        endpoint_cell = f'{endpoint_ssim:.6f}' if isinstance(endpoint_ssim, float) else ''
        lines.append(
            f"| {r['alpha']:.3f} | {r['label']} | {r['mean_PSNR']:.6f} | {r['mean_SSIM_grid32']:.6f} | {endpoint_cell} | "
            f"{r['mean_dPSNR']:+.6f} | {r['hard_bottom25_dPSNR']:+.6f} | {r['easy_top25_dPSNR']:+.6f} | "
            f"{r['positive_ratio']:.6f} | {r['severe_loss_per_600']:.2f} | {r['worst_dPSNR']:+.6f} |"
        )
    lines += [
        '',
        '## Reliability Checks',
        '',
        f"- Pair count: `{len(rows)}` locked-test images.",
        f"- v2.2 parity pass: `{parity.get('parity_pass')}` with matched count `{parity.get('matched_count')}`.",
        f"- Max abs parity diffs: A0 PSNR `{parity.get('max_abs_A0_PSNR')}`, A0 SSIM `{parity.get('max_abs_A0_SSIM')}`, WD0375 PSNR `{parity.get('max_abs_WD0375_PSNR')}`, WD0375 SSIM `{parity.get('max_abs_WD0375_SSIM')}`, WDMamba PSNR `{parity.get('max_abs_WDMamba_PSNR')}`, WDMamba SSIM `{parity.get('max_abs_WDMamba_SSIM')}`.",
        '- Preliminary v28/NH metric reuse output was discarded before sync because A0/WD0375 SSIM did not match v2.2; this directory contains only the corrected v2.2-compatible rerun.',
        '- Locked policy: diagnostic-only; no locked-grid alpha selection or retuning.',
        '',
        '## Evidence Files',
        '',
        f'- `{args.prefix}_absolute_metrics.csv`',
        f'- `{args.prefix}_compact_metrics.csv`',
        f'- `{args.prefix}_per_image.csv`',
        f'- `{args.prefix}_summary.json`',
        '- `v210_parity_with_v22_wd0375.json`',
        '- `commands/run_v210_locked_alpha_grid_parallel.sh`',
        '- `runtime_logs/`',
    ]
    text = '\n'.join(lines) + '\n'
    (args.out_dir / 'README.md').write_text(text, encoding='utf-8')
    (args.out_dir / 'v210_decision.md').write_text(text, encoding='utf-8')
    print(f'V210_AGGREGATE_OK rows={len(rows)} parity_pass={parity.get("parity_pass")}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['evaluate', 'aggregate'], required=True)
    ap.add_argument('--convir-its-dir', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked/Dehazing/ITS'))
    ap.add_argument('--data-dir', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K'))
    ap.add_argument('--data-split', default='test')
    ap.add_argument('--a0-checkpoint', type=Path, required=True)
    ap.add_argument('--wdmamba-repo', type=Path, required=True)
    ap.add_argument('--wdmamba-checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--prefix', default='v210_haze4k_locked_wdmamba_alpha_grid')
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--print-freq', type=int, default=10)
    ap.add_argument('--input-csvs', nargs='*', default=[])
    ap.add_argument('--expected-count', type=int, default=1000)
    args = ap.parse_args()
    args.alphas = sorted({round(float(a), 6) for a in args.alphas})
    if args.mode == 'evaluate':
        evaluate(args)
    else:
        aggregate(args)


if __name__ == '__main__':
    main()
