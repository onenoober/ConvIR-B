#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
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
from eval_udpnet_v15_phase0_repro import load_convir_builders, load_a0_model, load_udpnet_model, infer_one  # noqa: E402

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


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
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


def metric_grid(pred: torch.Tensor, label: torch.Tensor, hp: int, wp: int) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hp, wp) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hp / down), int(wp / down))),
        F.adaptive_avg_pool2d(label, (int(hp / down), int(wp / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def metric_direct(pred: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    return tensor_psnr(pred, label), float(ssim(pred, label, data_range=1, size_average=False).mean().item())


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
    input_dir = first_existing_dir(data_dir / split, ('IN', 'haze', 'hazy', 'Haze'))
    return sorted(p.name for p in input_dir.iterdir() if p.is_file())


def load_sample(data_dir: Path, split: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = data_dir / split
    input_dir = first_existing_dir(root, ('IN', 'haze', 'hazy', 'Haze'))
    gt_dir = first_existing_dir(root, ('GT', 'gt'))
    hazy = Image.open(input_dir / name).convert('RGB')
    gt = Image.open(label_path(gt_dir, name)).convert('RGB')
    return TVF.to_tensor(hazy).float(), TVF.to_tensor(gt).float()


def load_depth(depth_cache_dir: Path, name: str, h: int, w: int, depth2l_dir: Path | None = None) -> torch.Tensor:
    if depth2l_dir is not None:
        path = depth2l_dir / f'{Path(name).stem}.png'
        if not path.is_file():
            raise FileNotFoundError(f'missing official-style depth2l PNG {path}')
        img = Image.open(path).convert('L')
        if img.size != (w, h):
            img = img.resize((w, h), resample=Image.BICUBIC)
        return TVF.to_tensor(img).float()
    path = depth_cache_dir / f'{name}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'missing depth cache {path}')
    import numpy as np
    arr = np.load(path)
    x = torch.from_numpy(arr).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x.permute(2, 0, 1)
    elif x.ndim == 3 and x.shape[0] != 1:
        x = x[:1]
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False).squeeze(0)
    return x.clamp(0, 1).float()


def load_udp_builder_file(udp_repo: Path, filename: str, fsudp_heads2_patch: bool = False):
    models_dir = udp_repo / 'Dehazing/ITS/models'
    model_file = models_dir / filename
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    package_name = 'udpnet_v211_models_' + Path(filename).stem.lower() + ('_patched' if fsudp_heads2_patch else '')
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    module_name = f'{package_name}.{Path(filename).stem}'
    if fsudp_heads2_patch:
        src = model_file.read_text(encoding='utf-8').replace('num_heads=1', 'num_heads=2')
        mod = types.ModuleType(module_name)
        mod.__file__ = str(model_file)
        mod.__package__ = package_name
        sys.modules[module_name] = mod
        exec(compile(src, str(model_file) + '#v211_fsudp_heads2', 'exec'), mod.__dict__)
        return mod.build_net
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_net


def load_fsudp(args: argparse.Namespace, device: torch.device):
    build = load_udp_builder_file(args.udp_repo, 'FSNet_UDPNet.py', fsudp_heads2_patch=True)
    model, meta = load_udpnet_model(build, args.expert_checkpoint, device)
    model.v211_load_meta = meta
    return model


def fake_optional_modules() -> None:
    if 'matplotlib' not in sys.modules:
        m = types.ModuleType('matplotlib')
        p = types.ModuleType('matplotlib.pyplot')
        m.pyplot = p
        sys.modules['matplotlib'] = m
        sys.modules['matplotlib.pyplot'] = p
    if 'thop' not in sys.modules:
        t = types.ModuleType('thop')
        t.profile = lambda *a, **k: (0, 0)
        t.clever_format = lambda vals, *a, **k: vals
        sys.modules['thop'] = t


def load_mbtaylor(args: argparse.Namespace, device: torch.device):
    fake_optional_modules()
    repo = args.mbtaylor_repo
    file = repo / 'basicsr/models/archs/MB_TaylorFormerV2.py'
    spec = importlib.util.spec_from_file_location('mbtaylor_v211_arch', file)
    if spec is None or spec.loader is None:
        raise RuntimeError(file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    import yaml
    opt = yaml.safe_load((repo / 'Dehazing/Options/MB-TaylorFormerV2-L.yml').read_text(encoding='utf-8'))['network_g']
    opt.pop('type', None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = mod.MB_TaylorFormer(**opt).to(device)
    ckpt = torch.load(args.expert_checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['params'] if isinstance(ckpt, dict) and 'params' in ckpt else ckpt
    res = model.load_state_dict(state, strict=False)
    model.eval()
    model.v211_missing_keys = list(res.missing_keys)
    model.v211_unexpected_keys = list(res.unexpected_keys)
    return model


def infer_expert(expert: str, model, rgb: torch.Tensor, depth: torch.Tensor, h: int, w: int, fsudp_pad_factor: int) -> tuple[torch.Tensor, int, int]:
    if expert == 'fsudp':
        x, hp, wp = pad_to(torch.cat([rgb, depth], 1), fsudp_pad_factor)
        return infer_one(model, x, h, w), hp, wp
    if expert == 'mbtaylor':
        x, hp, wp = pad_to(rgb, 8)
        out = model(x)
        return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp
    raise ValueError(expert)


def endpoint_name(expert: str) -> str:
    return 'FSNetUDP' if expert == 'fsudp' else 'MBTaylorV2L'


def expert_label(expert: str) -> str:
    return 'FSNet+UDP' if expert == 'fsudp' else 'MB-TaylorFormerV2-L'


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    names = list_images(args.data_dir, args.data_split)
    root = args.data_dir / args.data_split
    input_dir = first_existing_dir(root, ('IN', 'haze', 'hazy', 'Haze'))
    gt_dir = first_existing_dir(root, ('GT', 'gt'))
    missing_gt = []
    missing_depth = []
    missing_depth2l = []
    size_mismatch = []
    for name in names:
        try:
            gp = label_path(gt_dir, name)
        except FileNotFoundError:
            missing_gt.append(name)
            continue
        dp = args.depth_cache_dir / f'{name}.npy'
        if not dp.is_file():
            missing_depth.append(name)
        if args.depth2l_dir is not None:
            d2 = args.depth2l_dir / f'{Path(name).stem}.png'
            if not d2.is_file():
                missing_depth2l.append(name)
        with Image.open(input_dir / name) as hazy, Image.open(gp) as gt:
            if hazy.size != gt.size:
                size_mismatch.append({'name': name, 'hazy': hazy.size, 'gt': gt.size})
    return {
        'data_dir': str(args.data_dir),
        'data_split': args.data_split,
        'input_dir': str(input_dir),
        'gt_dir': str(gt_dir),
        'depth_cache_dir': str(args.depth_cache_dir),
        'image_count': len(names),
        'missing_gt_count': len(missing_gt),
        'missing_gt_examples': missing_gt[:10],
        'missing_depth_count': len(missing_depth),
        'missing_depth_examples': missing_depth[:10],
        'depth2l_dir': str(args.depth2l_dir) if args.depth2l_dir is not None else '',
        'missing_depth2l_count': len(missing_depth2l),
        'missing_depth2l_examples': missing_depth2l[:10],
        'size_mismatch_count': len(size_mismatch),
        'size_mismatch_examples': size_mismatch[:10],
        'expected_count_pass': len(names) == args.expected_count,
        'alignment_pass': len(names) == args.expected_count and not missing_gt and not missing_depth and not missing_depth2l and not size_mismatch,
    }


def run_preflight(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_preflight(args)
    report.update({
        'mode': 'preflight',
        'route': 'haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616',
        'locked_test_touched': False,
        'a0_checkpoint': str(args.a0_checkpoint),
        'a0_sha256': sha256(args.a0_checkpoint),
        'fsudp_checkpoint': str(args.fsudp_checkpoint),
        'fsudp_sha256': sha256(args.fsudp_checkpoint),
        'mbtaylor_checkpoint': str(args.mbtaylor_checkpoint),
        'mbtaylor_sha256': sha256(args.mbtaylor_checkpoint),
        'udp_repo': str(args.udp_repo),
        'mbtaylor_repo': str(args.mbtaylor_repo),
            'fsudp_loader_note': 'official UDPNet FSNet_UDPNet.py with num_heads=1->2 builder patch to match checkpoint OCAB bias tables; state_dict strict=True after patch; official-style depth2l PNG input is required for FSNet+UDP formal metrics',
        'mbtaylor_loader_note': 'official MB-TaylorFormerV2-L.yml, HAZE4K-L.pth params, strict=False as official Dehazing/test.py',
    })
    out = args.out_dir / 'v211_preflight.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if not report['alignment_pass']:
        raise RuntimeError(f'preflight failed: {report}')
    print(f'V211_PREFLIGHT_OK count={report["image_count"]}')


def evaluate(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_names = list_images(args.data_dir, args.data_split)
    names = [name for idx, name in enumerate(all_names) if idx % args.shard_count == args.shard_index]
    if args.max_images:
        names = names[:args.max_images]
    if not names:
        raise RuntimeError(f'empty shard {args.expert} {args.shard_index}/{args.shard_count}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, build_convir = load_convir_builders(args.convir_its_dir)
    a0 = load_a0_model(build_convir, args.a0_checkpoint, device)
    expert_model = load_fsudp(args, device) if args.expert == 'fsudp' else load_mbtaylor(args, device)
    alphas = sorted({round(float(a), 6) for a in args.alphas})
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    ename = endpoint_name(args.expert)
    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy, gt = load_sample(args.data_dir, args.data_split, name)
            _, h, w = hazy.shape
            depth = load_depth(args.depth_cache_dir, name, h, w, args.depth2l_dir if args.expert == 'fsudp' else None)
            rgb = hazy.unsqueeze(0).to(device)
            lab = gt.unsqueeze(0).to(device)
            dep = depth.unsqueeze(0).to(device)
            rgb32, hp32, wp32 = pad_to(rgb, 32)
            a0_pred = infer_one(a0, rgb32, h, w)
            a0_psnr, a0_ssim = metric_grid(a0_pred, lab, hp32, wp32)
            expert_pred, ehp, ewp = infer_expert(args.expert, expert_model, rgb, dep, h, w, args.fsudp_pad_factor)
            expert_psnr, expert_ssim_endpoint = metric_grid(expert_pred, lab, ehp, ewp)
            expert_grid32_psnr, expert_grid32_ssim = metric_grid(expert_pred, lab, hp32, wp32)
            expert_direct_psnr, expert_direct_ssim = metric_direct(expert_pred, lab)
            row: dict[str, Any] = {
                'image_id': Path(name).stem,
                'name': name,
                'split': args.data_split,
                'expert': args.expert,
                'expert_label': expert_label(args.expert),
                'width': w,
                'height': h,
                'shard_index': args.shard_index,
                'shard_count': args.shard_count,
                'A0_PSNR': a0_psnr,
                'A0_SSIM_grid32': a0_ssim,
                f'{ename}_PSNR_endpoint': expert_psnr,
                f'{ename}_SSIM_endpoint': expert_ssim_endpoint,
                f'{ename}_PSNR_grid32': expert_grid32_psnr,
                f'{ename}_SSIM_grid32': expert_grid32_ssim,
                f'{ename}_PSNR_direct': expert_direct_psnr,
                f'{ename}_SSIM_direct': expert_direct_ssim,
                f'{ename}_dPSNR_endpoint': expert_psnr - a0_psnr,
                f'{ename}_dSSIM_endpoint': expert_ssim_endpoint - a0_ssim,
                f'{ename}_dSSIM_grid32': expert_grid32_ssim - a0_ssim,
                f'{ename}_dSSIM_direct_minus_A0_grid32': expert_direct_ssim - a0_ssim,
            }
            for alpha in alphas:
                key = alpha_key(alpha)
                pred = torch.clamp(a0_pred + alpha * (expert_pred - a0_pred), 0, 1)
                psnr, ss = metric_grid(pred, lab, hp32, wp32)
                row[f'alpha_{key}_PSNR'] = psnr
                row[f'alpha_{key}_SSIM_grid32'] = ss
                row[f'alpha_{key}_dPSNR'] = psnr - a0_psnr
                row[f'alpha_{key}_dSSIM_grid32'] = ss - a0_ssim
            rows.append(row)
            if idx % args.print_freq == 0 or idx == len(names):
                k = alpha_key(0.375)
                mean = statistics.mean(fnum(r[f'alpha_{k}_dPSNR']) for r in rows)
                print(
                    f'progress expert={args.expert} shard={args.shard_index}/{args.shard_count} '
                    f'{idx}/{len(names)} mean_a0p375={mean:.6f} elapsed={time.time()-t0:.1f}s',
                    flush=True,
                )
    write_csv(args.out_dir / f'{args.prefix}_per_image.csv', rows)
    manifest = {
        'mode': 'evaluate',
        'expert': args.expert,
        'expert_label': expert_label(args.expert),
        'count': len(rows),
        'shard_index': args.shard_index,
        'shard_count': args.shard_count,
        'alpha_grid': alphas,
        'metric_protocol': 'v2.2 locked one-shot compatible: alpha candidates use A0 32-pad PSNR/SSIM. Expert endpoints also record official-style padding/direct SSIM as separate fields.',
        'loader_note': (
            f'official UDPNet FSNet_UDPNet.py with num_heads=1->2 builder patch, strict=True checkpoint load, depth2l PNG input, pad factor {args.fsudp_pad_factor}'
            if args.expert == 'fsudp'
            else 'official MB-TaylorFormerV2-L.yml with HAZE4K-L.pth params, strict=False as official test.py'
        ),
        'mbtaylor_missing_keys': getattr(expert_model, 'v211_missing_keys', None),
        'mbtaylor_unexpected_keys': getattr(expert_model, 'v211_unexpected_keys', None),
        'fsudp_load_meta': getattr(expert_model, 'v211_load_meta', None),
    }
    (args.out_dir / f'{args.prefix}_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'V211_EVAL_SHARD_OK expert={args.expert} shard={args.shard_index} rows={len(rows)}')


def summarize_alpha(rows: list[dict[str, Any]], expert: str, alpha: float) -> dict[str, Any]:
    key = alpha_key(alpha)
    n = len(rows)
    ps = [fnum(r[f'alpha_{key}_PSNR']) for r in rows]
    ss = [fnum(r[f'alpha_{key}_SSIM_grid32']) for r in rows]
    ds = [fnum(r[f'alpha_{key}_dPSNR']) for r in rows]
    dss = [fnum(r[f'alpha_{key}_dSSIM_grid32']) for r in rows]
    a0 = [fnum(r['A0_PSNR']) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    bucket = max(1, n // 4)
    hard = [ds[i] for i in order[:bucket]]
    easy = [ds[i] for i in order[-bucket:]]
    severe = sum(d <= SEVERE for d in ds)
    pos = sum(d > 0 for d in ds)
    label = 'A0 / ConvIR-B' if alpha == 0 else f'{expert_label(expert)} full' if alpha == 1 else f'{expert_label(expert)} alpha={alpha:.3f}'
    return {
        'expert': expert,
        'expert_label': expert_label(expert),
        'alpha': alpha,
        'label': label,
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


def summarize_endpoint(rows: list[dict[str, Any]], expert: str) -> dict[str, Any]:
    ename = endpoint_name(expert)
    n = len(rows)
    ds = [fnum(r[f'{ename}_dPSNR_endpoint']) for r in rows]
    dss_endpoint = [fnum(r[f'{ename}_dSSIM_endpoint']) for r in rows]
    dss_grid = [fnum(r[f'{ename}_dSSIM_grid32']) for r in rows]
    dss_direct = [fnum(r[f'{ename}_dSSIM_direct_minus_A0_grid32']) for r in rows]
    a0 = [fnum(r['A0_PSNR']) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    bucket = max(1, n // 4)
    hard = [ds[i] for i in order[:bucket]]
    easy = [ds[i] for i in order[-bucket:]]
    severe = sum(d <= SEVERE for d in ds)
    return {
        'expert': expert,
        'expert_label': expert_label(expert),
        'count': n,
        'A0_PSNR': statistics.mean(fnum(r['A0_PSNR']) for r in rows),
        'A0_SSIM_grid32': statistics.mean(fnum(r['A0_SSIM_grid32']) for r in rows),
        'expert_PSNR_endpoint': statistics.mean(fnum(r[f'{ename}_PSNR_endpoint']) for r in rows),
        'expert_SSIM_endpoint': statistics.mean(fnum(r[f'{ename}_SSIM_endpoint']) for r in rows),
        'expert_PSNR_grid32': statistics.mean(fnum(r[f'{ename}_PSNR_grid32']) for r in rows),
        'expert_SSIM_grid32': statistics.mean(fnum(r[f'{ename}_SSIM_grid32']) for r in rows),
        'expert_PSNR_direct': statistics.mean(fnum(r[f'{ename}_PSNR_direct']) for r in rows),
        'expert_SSIM_direct': statistics.mean(fnum(r[f'{ename}_SSIM_direct']) for r in rows),
        'mean_dPSNR_endpoint': statistics.mean(ds),
        'hard_bottom25_dPSNR_endpoint': statistics.mean(hard),
        'easy_top25_dPSNR_endpoint': statistics.mean(easy),
        'mean_dSSIM_endpoint': statistics.mean(dss_endpoint),
        'mean_dSSIM_grid32': statistics.mean(dss_grid),
        'mean_dSSIM_direct_minus_A0_grid32': statistics.mean(dss_direct),
        'positive_ratio_endpoint': sum(d > 0 for d in ds) / n,
        'nonnegative_ratio_endpoint': sum(d >= 0 for d in ds) / n,
        'severe_loss_count_endpoint': severe,
        'severe_loss_per_600_endpoint': severe / n * 600.0,
        'worst_dPSNR_endpoint': min(ds),
        'official_reported_haze4k_psnr': 35.31 if expert == 'fsudp' else None,
        'official_reported_haze4k_ssim': 0.99 if expert == 'fsudp' else None,
        'official_source_note': (
            'UDPNet README Table 2 reports FSNet+UDP Haze4K 35.31 / 0.99'
            if expert == 'fsudp'
            else 'MB-TaylorFormerV2 repo README/test.py does not expose a clear Haze4K V2-L table row in checked files; checkpoint is HAZE4K-L.pth'
        ),
    }


def aggregate_one(args: argparse.Namespace, expert: str, input_csvs: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for p in input_csvs:
        rows.extend(read_csv(Path(p)))
    rows = sorted(rows, key=lambda r: r['image_id'])
    if len(rows) != args.expected_count:
        raise RuntimeError(f'{expert}: expected {args.expected_count} rows got {len(rows)}')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise RuntimeError(f'{expert}: duplicate image_id detected')
    alphas = sorted({round(float(a), 6) for a in args.alphas})
    prefix = f'{args.prefix}_{expert}'
    write_csv(args.out_dir / f'{prefix}_per_image.csv', rows)
    alpha_rows = [summarize_alpha(rows, expert, a) for a in alphas]
    write_csv(args.out_dir / f'{prefix}_alpha_grid_absolute_metrics.csv', alpha_rows)
    compact_keys = [
        'expert_label', 'alpha', 'label', 'count', 'mean_PSNR', 'mean_SSIM_grid32', 'mean_dPSNR',
        'hard_bottom25_dPSNR', 'easy_top25_dPSNR', 'mean_dSSIM_grid32',
        'positive_ratio', 'severe_loss_per_600', 'worst_dPSNR',
    ]
    write_csv(args.out_dir / f'{prefix}_alpha_grid_compact_metrics.csv', [{k: r.get(k, '') for k in compact_keys} for r in alpha_rows])
    endpoint = summarize_endpoint(rows, expert)
    return {
        'expert': expert,
        'expert_label': expert_label(expert),
        'endpoint_summary': endpoint,
        'alpha_summaries': alpha_rows,
        'per_image_file': f'{prefix}_per_image.csv',
        'absolute_metrics_file': f'{prefix}_alpha_grid_absolute_metrics.csv',
        'compact_metrics_file': f'{prefix}_alpha_grid_compact_metrics.csv',
    }


def aggregate(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fsudp = aggregate_one(args, 'fsudp', args.fsudp_input_csvs)
    mbtaylor = aggregate_one(args, 'mbtaylor', args.mbtaylor_input_csvs)
    endpoint_rows = [fsudp['endpoint_summary'], mbtaylor['endpoint_summary']]
    write_csv(args.out_dir / f'{args.prefix}_endpoint_reproduction_metrics.csv', endpoint_rows)
    combined_alpha_rows = fsudp['alpha_summaries'] + mbtaylor['alpha_summaries']
    write_csv(args.out_dir / f'{args.prefix}_combined_alpha_grid_absolute_metrics.csv', combined_alpha_rows)
    compact_keys = [
        'expert_label', 'alpha', 'label', 'count', 'mean_PSNR', 'mean_SSIM_grid32', 'mean_dPSNR',
        'hard_bottom25_dPSNR', 'easy_top25_dPSNR', 'mean_dSSIM_grid32',
        'positive_ratio', 'severe_loss_per_600', 'worst_dPSNR',
    ]
    write_csv(args.out_dir / f'{args.prefix}_combined_alpha_grid_compact_metrics.csv', [{k: r.get(k, '') for k in compact_keys} for r in combined_alpha_rows])
    preflight_path = args.out_dir / 'v211_preflight.json'
    preflight = json.loads(preflight_path.read_text(encoding='utf-8')) if preflight_path.is_file() else {}
    summary = {
        'route': 'haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616',
        'decision': 'V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY',
        'state_label': 'COMPLETED_GATE_PASS',
        'locked_test_touched': True,
        'locked_scope': 'Haze4K test split, 1000 images',
        'locked_policy': 'diagnostic alpha-grid audit only; do not select or retune alpha from this locked grid',
        'metric_protocol': 'v2.2 locked one-shot compatible for alpha candidates: A0 32-pad PSNR/SSIM. Expert endpoint/direct metrics are recorded separately for official-standard reproduction context.',
        'count_per_expert': args.expected_count,
        'alpha_grid': sorted({round(float(a), 6) for a in args.alphas}),
        'preflight': preflight,
        'a0_checkpoint': str(args.a0_checkpoint),
        'a0_sha256': sha256(args.a0_checkpoint),
        'fsudp_checkpoint': str(args.fsudp_checkpoint),
        'fsudp_sha256': sha256(args.fsudp_checkpoint),
        'mbtaylor_checkpoint': str(args.mbtaylor_checkpoint),
        'mbtaylor_sha256': sha256(args.mbtaylor_checkpoint),
        'udp_repo': str(args.udp_repo),
        'udp_repo_head': args.udp_repo_head,
        'mbtaylor_repo': str(args.mbtaylor_repo),
        'mbtaylor_repo_head': args.mbtaylor_repo_head,
        'experts': [fsudp, mbtaylor],
    }
    (args.out_dir / f'{args.prefix}_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    lines = [
        '# Haze4K v2.11 Locked Cross-Expert Alpha Grid',
        '',
        'Decision: `V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`',
        '',
        'This route evaluates FSNet+UDP and MB-TaylorFormerV2-L residual-shrinkage alpha grids on the Haze4K locked test split (`1000` images). It is a diagnostic audit only and must not be used to select or retune alpha.',
        '',
        'Metric protocol: alpha candidates use the v2.2 locked one-shot compatible A0 32-pad PSNR/SSIM convention, so the curves are directly comparable with the WDMamba v2.10 locked alpha grid. Expert endpoint/direct metrics are also recorded separately for official-standard reproduction context.',
        '',
        'Loader provenance:',
        '',
        '- FSNet+UDP: official UDPNet `Dehazing/ITS/models/FSNet_UDPNet.py`, with documented `num_heads=1 -> 2` builder patch needed to strict-load `FSNet_UDPNet_haze4k.ckpt` OCAB bias tables; depth input uses official-style `depth2l/*.png` files generated from DepthAnything V2 raw predictions by per-image min-max normalization, then read through PIL `L` mode like the official dataloader.',
        '- MB-TaylorFormerV2-L: official `Dehazing/Options/MB-TaylorFormerV2-L.yml`, `HAZE4K-L.pth`, `strict=False` matching official `Dehazing/test.py`; factor-8 reflect padding for model inference.',
        '',
        '## Endpoint Reproduction Metrics',
        '',
        '| expert | A0 PSNR | A0 SSIM grid32 | endpoint PSNR | endpoint SSIM | grid32 SSIM | direct SSIM | official reference | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in endpoint_rows:
        ref = '35.31 / 0.99 (UDPNet README Table 2)' if r['expert'] == 'fsudp' else 'not clearly reported for Haze4K V2-L in checked repo files'
        lines.append(
            f"| {r['expert_label']} | {r['A0_PSNR']:.6f} | {r['A0_SSIM_grid32']:.6f} | "
            f"{r['expert_PSNR_endpoint']:.6f} | {r['expert_SSIM_endpoint']:.6f} | "
            f"{r['expert_SSIM_grid32']:.6f} | {r['expert_SSIM_direct']:.6f} | {ref} | "
            f"{r['mean_dPSNR_endpoint']:+.6f} | {r['hard_bottom25_dPSNR_endpoint']:+.6f} | "
            f"{r['easy_top25_dPSNR_endpoint']:+.6f} | {r['positive_ratio_endpoint']:.6f} | "
            f"{r['severe_loss_per_600_endpoint']:.2f} |"
        )
    lines += [
        '',
        '## Alpha Grid Absolute Metrics',
        '',
        '| expert | alpha | label | PSNR | SSIM grid32 | mean dPSNR | hard dPSNR | easy dPSNR | dSSIM grid32 | positive | severe/600 | worst dPSNR |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in combined_alpha_rows:
        lines.append(
            f"| {r['expert_label']} | {r['alpha']:.3f} | {r['label']} | {r['mean_PSNR']:.6f} | {r['mean_SSIM_grid32']:.6f} | "
            f"{r['mean_dPSNR']:+.6f} | {r['hard_bottom25_dPSNR']:+.6f} | {r['easy_top25_dPSNR']:+.6f} | "
            f"{r['mean_dSSIM_grid32']:+.8f} | {r['positive_ratio']:.6f} | {r['severe_loss_per_600']:.2f} | {r['worst_dPSNR']:+.6f} |"
        )
    lines += [
        '',
        '## Reliability Checks',
        '',
        f"- Preflight alignment pass: `{preflight.get('alignment_pass')}`; image count `{preflight.get('image_count')}`, missing GT `{preflight.get('missing_gt_count')}`, missing raw depth `{preflight.get('missing_depth_count')}`, missing depth2l `{preflight.get('missing_depth2l_count')}`, size mismatch `{preflight.get('size_mismatch_count')}`.",
        f"- A0 checkpoint sha256: `{sha256(args.a0_checkpoint)}`.",
        f"- FSNet+UDP checkpoint sha256: `{sha256(args.fsudp_checkpoint)}`.",
        f"- MB-TaylorFormerV2-L checkpoint sha256: `{sha256(args.mbtaylor_checkpoint)}`.",
        '- Locked policy: diagnostic-only; no locked-grid alpha selection or retuning.',
        '',
        '## Evidence Files',
        '',
        f'- `{args.prefix}_summary.json`',
        f'- `{args.prefix}_endpoint_reproduction_metrics.csv`',
        f'- `{args.prefix}_combined_alpha_grid_absolute_metrics.csv`',
        f'- `{args.prefix}_combined_alpha_grid_compact_metrics.csv`',
        f'- `{args.prefix}_fsudp_per_image.csv`',
        f'- `{args.prefix}_mbtaylor_per_image.csv`',
        '- `v211_preflight.json`',
        '- `commands/run_v211_locked_cross_expert_alpha_grid_parallel.sh`',
        '- `runtime_logs/`',
    ]
    text = '\n'.join(lines) + '\n'
    (args.out_dir / 'README.md').write_text(text, encoding='utf-8')
    (args.out_dir / 'v211_decision.md').write_text(text, encoding='utf-8')
    print('V211_AGGREGATE_OK rows_per_expert=%d' % args.expected_count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['preflight', 'evaluate', 'aggregate'], required=True)
    ap.add_argument('--expert', choices=['fsudp', 'mbtaylor'], default='fsudp')
    ap.add_argument('--convir-its-dir', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked/Dehazing/ITS'))
    ap.add_argument('--data-dir', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K'))
    ap.add_argument('--data-split', default='test')
    ap.add_argument('--depth-cache-dir', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf/test'))
    ap.add_argument('--depth2l-dir', type=Path, default=None)
    ap.add_argument('--fsudp-pad-factor', type=int, default=8)
    ap.add_argument('--a0-checkpoint', type=Path, required=True)
    ap.add_argument('--fsudp-checkpoint', type=Path, required=True)
    ap.add_argument('--mbtaylor-checkpoint', type=Path, required=True)
    ap.add_argument('--expert-checkpoint', type=Path, required=True)
    ap.add_argument('--udp-repo', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/repos/UDPNet'))
    ap.add_argument('--mbtaylor-repo', type=Path, default=Path('/sda/home/wangyuxin/ConvIR-B/repos/external_experts/MB-TaylorFormerV2'))
    ap.add_argument('--udp-repo-head', default='')
    ap.add_argument('--mbtaylor-repo-head', default='')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--prefix', default='v211_haze4k_locked_cross_expert_alpha_grid')
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--print-freq', type=int, default=10)
    ap.add_argument('--expected-count', type=int, default=1000)
    ap.add_argument('--fsudp-input-csvs', nargs='*', default=[])
    ap.add_argument('--mbtaylor-input-csvs', nargs='*', default=[])
    args = ap.parse_args()
    args.alphas = sorted({round(float(a), 6) for a in args.alphas})
    if args.mode == 'preflight':
        run_preflight(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        aggregate(args)


if __name__ == '__main__':
    main()
