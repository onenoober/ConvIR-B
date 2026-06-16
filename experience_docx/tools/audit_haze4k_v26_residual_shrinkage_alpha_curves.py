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
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF

# Reuse validated ConvIR/UDPNet/Haze4K IO helpers from v2.1/v2.2.
import audit_haze4k_v20_c2_outputdiff_router as c2
from eval_udpnet_v15_phase0_repro import load_convir_builders, load_a0_model, load_udpnet_model, infer_one

DEFAULT_ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0]
SEVERE = -0.20
BOOT_N = 400


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def alpha_key(a: float) -> str:
    return (('a%.6f' % a).rstrip('0').rstrip('.')).replace('.', 'p')


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fields = keys
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).item())


def metric(pred: torch.Tensor, label: torch.Tensor, hpad: int, wpad: int) -> tuple[float, float]:
    psnr = tensor_psnr(pred, label)
    down = max(1, round(min(hpad, wpad) / 256))
    ss = ssim(
        F.adaptive_avg_pool2d(pred, (int(hpad / down), int(wpad / down))),
        F.adaptive_avg_pool2d(label, (int(hpad / down), int(wpad / down))),
        data_range=1,
        size_average=False,
    ).mean().item()
    return psnr, float(ss)


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int, int, int]:
    _, _, h, w = x.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(x, (0, pw, 0, ph), 'reflect'), h, w, h + ph, w + pw


def wilson_lcb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - rad) / denom)


def wilson_ucb(pos: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (center + rad) / denom)


def bootstrap_lcb(vals: list[float], rng: np.random.Generator, q: float = 0.05) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    idx = rng.integers(0, len(arr), size=(BOOT_N, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, q))


def summarize(rows: list[dict[str, Any]], dkey: str = 'dPSNR') -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {'count': 0}
    ds = [fnum(r[dkey]) for r in rows]
    ss = [fnum(r.get('dSSIM', 0)) for r in rows]
    a0 = [fnum(r['A0_PSNR']) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    k = max(1, n // 4)
    hard_vals = [ds[i] for i in order[:k]]
    easy_vals = [ds[i] for i in order[-k:]]
    severe = sum(d <= SEVERE for d in ds)
    positive = sum(d > 0 for d in ds)
    rng = np.random.default_rng(3407 + n)
    return {
        'count': n,
        'mean_dPSNR': statistics.mean(ds),
        'mean_bootstrap_LCB': bootstrap_lcb(ds, rng),
        'hard_bottom25_dPSNR': statistics.mean(hard_vals),
        'hard_bootstrap_LCB': bootstrap_lcb(hard_vals, rng),
        'easy_top25_dPSNR': statistics.mean(easy_vals),
        'worst_dPSNR': min(ds),
        'dSSIM': statistics.mean(ss),
        'positive_ratio': positive / n,
        'positive_Wilson_LCB': wilson_lcb(positive, n),
        'nonnegative_ratio': sum(d >= 0 for d in ds) / n,
        'severe_loss_count': severe,
        'severe_loss_per_600': severe / n * 600.0,
        'severe_Wilson_UCB': wilson_ucb(severe, n),
    }


def qbins(vals: list[float]) -> list[str]:
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        return []
    qs = np.quantile(arr, [0.25, 0.5, 0.75])
    out = []
    for v in arr:
        out.append('q1' if v <= qs[0] else 'q2' if v <= qs[1] else 'q3' if v <= qs[2] else 'q4')
    return out


def load_udp_builder_file(udp_repo: Path, filename: str):
    models_dir = udp_repo / 'Dehazing/ITS/models'
    model_file = models_dir / filename
    package_name = 'udpnet_c8_models_' + Path(filename).stem.lower()
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    module_name = f'{package_name}.{Path(filename).stem}'
    if filename == 'FSNet_UDPNet.py':
        # The available FSNet+UDP checkpoint stores 2-head OCAB bias tables while
        # the checked-out builder has fusion OCAB blocks hardcoded to 1 head.
        src = model_file.read_text(encoding='utf-8').replace('num_heads=1', 'num_heads=2')
        mod = types.ModuleType(module_name)
        mod.__file__ = str(model_file)
        mod.__package__ = package_name
        sys.modules[module_name] = mod
        exec(compile(src, str(model_file) + '#c8_fsudp_heads2', 'exec'), mod.__dict__)
        return mod.build_net
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_net


def load_fsudp(args, device):
    build = load_udp_builder_file(Path(args.udp_repo), 'FSNet_UDPNet.py')
    return load_udpnet_model(build, Path(args.expert_checkpoint), device)[0]


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


def load_mbtaylor(args, device):
    fake_optional_modules()
    repo = Path(args.mbtaylor_repo)
    file = repo / 'basicsr/models/archs/MB_TaylorFormerV2.py'
    spec = importlib.util.spec_from_file_location('mbtaylor_c8_arch', file)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    import yaml
    opt = yaml.safe_load((repo / 'Dehazing/Options/MB-TaylorFormerV2-L.yml').read_text())['network_g']
    opt.pop('type', None)
    with contextlib.redirect_stdout(io.StringIO()):
        model = mod.MB_TaylorFormer(**opt).to(device)
    ckpt = torch.load(args.expert_checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['params'] if isinstance(ckpt, dict) and 'params' in ckpt else ckpt
    res = model.load_state_dict(state, strict=False)
    model.eval()
    model.c8_load_missing = list(res.missing_keys)
    model.c8_load_unexpected = list(res.unexpected_keys)
    return model


def load_wdmamba(args, device):
    repo = Path(args.wdmamba_repo)
    # mamba_ssm==2.2.x imports text-generation classes at package import time.
    # The WDMamba image model only needs selective_scan ops, so provide harmless
    # compatibility aliases for newer transformers builds where names moved.
    try:
        import transformers.generation as tg
        for name in ['GreedySearchDecoderOnlyOutput', 'SampleDecoderOnlyOutput']:
            if not hasattr(tg, name):
                setattr(tg, name, type(name, (object,), {}))
    except Exception:
        pass

    # Avoid WDMamba basicsr/__init__.py because it imports training losses (pyiqa).
    # Build a minimal package namespace and load only registry + arch modules.
    def pkg(name: str, path: Path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod
        return mod

    def load_mod(name: str, file: Path):
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise RuntimeError(file)
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
    WaveMamba = wavemamba.WaveMamba
    model = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
    ckpt = torch.load(args.expert_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['params'], strict=True)
    model.eval()
    return model


def infer_candidate(kind: str, model, rgb: torch.Tensor, depth: torch.Tensor, h: int, w: int):
    if kind == 'fsudp':
        x, _, _, hp, wp = pad_to(torch.cat([rgb, depth], 1), 32)
        return infer_one(model, x, h, w), hp, wp
    if kind == 'mbtaylor':
        x, _, _, hp, wp = pad_to(rgb, 8)
        out = model(x)
        return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp
    if kind == 'wdmamba':
        x, _, _, hp, wp = pad_to(rgb, 4)
        out = model.restoration_network(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return torch.clamp(out[:, :, :h, :w], 0, 1), hp, wp
    raise ValueError(kind)


def load_transmission(data_dir: Path, name: str, size: tuple[int, int]) -> torch.Tensor | None:
    trans_dir = data_dir / 'train' / 'trans'
    stem = Path(name).stem
    clean_id = stem.split('_', 1)[0]
    candidates = [trans_dir / name, trans_dir / f'{stem}.png', trans_dir / f'{clean_id}.png']
    path = next((p for p in candidates if p.is_file()), candidates[-1])
    if not path.is_file():
        matches = list(trans_dir.glob(clean_id + '.*'))
        path = matches[0] if matches else path
    if not path.is_file():
        return None
    img = Image.open(path).convert('L')
    if img.size != size:
        img = img.resize(size, resample=Image.BICUBIC)
    return TVF.to_tensor(img).float()


def grad_mean(x: torch.Tensor) -> float:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return float((gx.abs().mean() + gy.abs().mean()).item() * 0.5)


def qvalue(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.detach().flatten().float().cpu(), q).item())


def add_feature_values(rec: dict[str, Any], rgb: torch.Tensor, lab: torch.Tensor, dep: torch.Tensor, trans: torch.Tensor | None, a0_pred: torch.Tensor, full_pred: torch.Tensor, cand_pred: torch.Tensor) -> None:
    base_feats = c2.feature_dict(rgb, dep, a0_pred, full_pred)
    for k, v in base_feats.items():
        rec[f'feature_{k}'] = v
    full_res = full_pred - a0_pred
    exp_res = cand_pred - a0_pred
    rec['expert_residual_signed_mean'] = float(exp_res.mean().item())
    rec['expert_residual_abs_mean'] = float(exp_res.abs().mean().item())
    rec['expert_residual_grad_mean'] = grad_mean(exp_res)
    rec['expert_fulludp_mae'] = float((cand_pred - full_pred).abs().mean().item())
    rec['expert_fulludp_mse'] = float(F.mse_loss(cand_pred, full_pred).item())
    rec['expert_fulludp_psnr'] = tensor_psnr(cand_pred, full_pred)
    denom = (full_res.flatten().norm() * exp_res.flatten().norm()).clamp_min(1e-12)
    rec['residual_cosine_fulludp_expert'] = float(torch.dot(full_res.flatten(), exp_res.flatten()).div(denom).item())
    rec['input_edge_density'] = grad_mean(rgb)
    rec['input_low_texture_proxy'] = -float(rgb.std().item())
    dark = rgb.min(dim=1, keepdim=True).values
    rec['dark_channel_mean'] = float(dark.mean().item())
    brightness = rgb.mean(dim=1, keepdim=True)
    sat = rgb.max(dim=1, keepdim=True).values - rgb.min(dim=1, keepdim=True).values
    rec['sky_highlight_proxy'] = float(((brightness > 0.78) & (sat < 0.18)).float().mean().item())
    rec['airlight_proxy_p99'] = qvalue(brightness, 0.99)
    if trans is not None:
        t = trans.unsqueeze(0).to(rgb.device) if trans.ndim == 3 else trans.to(rgb.device)
        rec['transmission_mean'] = float(t.mean().item())
        rec['transmission_std'] = float(t.std().item())
        rec['haze_density_mean'] = float((1.0 - t).mean().item())
        rec['haze_density_p90'] = qvalue(1.0 - t, 0.90)
    else:
        rec['transmission_mean'] = float('nan')
        rec['transmission_std'] = float('nan')
        rec['haze_density_mean'] = float('nan')
        rec['haze_density_p90'] = float('nan')


def group_rows(rows: list[dict[str, Any]], args_splits: list[str]) -> list[dict[str, Any]]:
    specs = [
        ('A0-PSNR_q4', 'A0_PSNR'),
        ('FullUDP-A0_diff_signed_q4', 'feature_diff_signed_mean'),
        ('FullUDP-A0_diff_abs_q4', 'feature_diff_abs_mean'),
        ('Expert-A0_diff_signed_q4', 'expert_residual_signed_mean'),
        ('expert_disagreement_q4', 'expert_fulludp_mae'),
        ('residual_cosine_q4', 'residual_cosine_fulludp_expert'),
        ('haze_density_q4', 'haze_density_mean'),
        ('haze_density_p90_q4', 'haze_density_p90'),
        ('transmission_q4', 'transmission_mean'),
        ('airlight_proxy_q4', 'airlight_proxy_p99'),
        ('depth_q4', 'feature_depth_mean'),
        ('dark_channel_q4', 'dark_channel_mean'),
        ('low_texture_q4', 'input_low_texture_proxy'),
        ('edge_density_q4', 'input_edge_density'),
        ('sky_highlight_proxy_q4', 'sky_highlight_proxy'),
    ]
    groups: list[dict[str, Any]] = []
    for label, key in specs:
        vals = [fnum(r.get(key), float('nan')) for r in rows]
        finite_pairs = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
        if not finite_pairs:
            continue
        bins_finite = qbins([v for _, v in finite_pairs])
        idx_to_bin = {i: b for (i, _), b in zip(finite_pairs, bins_finite)}
        for b in ['q1', 'q2', 'q3', 'q4']:
            sub0 = [r for i, r in enumerate(rows) if idx_to_bin.get(i) == b]
            sub = [{**r, 'dPSNR': r['S1_alpha_oracle_dPSNR'], 'dSSIM': 0} for r in sub0]
            if not sub:
                continue
            groups.append({
                'group': label,
                'bin': b,
                'expert_selected_rate': sum(r['S1_best_family'] == 'expert' for r in sub0) / len(sub0),
                'unique_win_rate': sum(bool(r['expert_unique_win']) for r in sub0) / len(sub0),
                **summarize(sub),
            })
    for split in sorted(set(r['split'] for r in rows).union(args_splits)):
        sub0 = [r for r in rows if r['split'] == split]
        if not sub0:
            continue
        sub = [{**r, 'dPSNR': r['S1_alpha_oracle_dPSNR'], 'dSSIM': 0} for r in sub0]
        groups.append({
            'group': 'split',
            'bin': split,
            'expert_selected_rate': sum(r['S1_best_family'] == 'expert' for r in sub0) / len(sub0),
            'unique_win_rate': sum(bool(r['expert_unique_win']) for r in sub0) / len(sub0),
            **summarize(sub),
        })
    return groups


def alpha_group_rows(rows: list[dict[str, Any]], args_splits: list[str], alphas: list[float]) -> list[dict[str, Any]]:
    specs = [
        ('A0-PSNR_q4', 'A0_PSNR'),
        ('Expert-A0_diff_signed_q4', 'expert_residual_signed_mean'),
        ('Expert-A0_diff_abs_q4', 'expert_residual_abs_mean'),
        ('expert_fulludp_disagreement_q4', 'expert_fulludp_mae'),
        ('residual_cosine_q4', 'residual_cosine_fulludp_expert'),
        ('haze_density_q4', 'haze_density_mean'),
        ('haze_density_p90_q4', 'haze_density_p90'),
        ('transmission_q4', 'transmission_mean'),
        ('airlight_proxy_q4', 'airlight_proxy_p99'),
        ('depth_q4', 'feature_depth_mean'),
        ('dark_channel_q4', 'dark_channel_mean'),
        ('low_texture_q4', 'input_low_texture_proxy'),
        ('edge_density_q4', 'input_edge_density'),
        ('sky_highlight_proxy_q4', 'sky_highlight_proxy'),
    ]
    groups: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str, list[int]]] = []
    for label, key in specs:
        vals = [fnum(r.get(key), float('nan')) for r in rows]
        finite_pairs = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
        if not finite_pairs:
            continue
        bins_finite = qbins([v for _, v in finite_pairs])
        idx_to_bin = {i: b for (i, _), b in zip(finite_pairs, bins_finite)}
        for b in ['q1', 'q2', 'q3', 'q4']:
            idxs = [i for i in range(len(rows)) if idx_to_bin.get(i) == b]
            if idxs:
                group_specs.append((label, b, idxs))
    for split in sorted(set(r['split'] for r in rows).union(args_splits)):
        idxs = [i for i, r in enumerate(rows) if r['split'] == split]
        if idxs:
            group_specs.append(('split', split, idxs))

    for fam in ['expert', 'fulludp']:
        for a in alphas:
            key = alpha_key(a)
            for label, b, idxs in group_specs:
                sub = [
                    {**rows[i], 'dPSNR': rows[i][f'{fam}_{key}_dPSNR'], 'dSSIM': rows[i][f'{fam}_{key}_dSSIM']}
                    for i in idxs
                ]
                groups.append({'family': fam, 'alpha': a, 'group': label, 'bin': b, **summarize(sub)})
    return groups


def alpha_group_min_rows(group_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r['family'], fnum(r['alpha'])) for r in group_metrics}, key=lambda x: (x[0], x[1]))
    for fam, alpha in keys:
        sub = [r for r in group_metrics if r['family'] == fam and fnum(r['alpha']) == alpha]
        if not sub:
            continue
        mean_min = min(sub, key=lambda r: fnum(r['mean_dPSNR']))
        hard_min = min(sub, key=lambda r: fnum(r['hard_bottom25_dPSNR']))
        easy_min = min(sub, key=lambda r: fnum(r['easy_top25_dPSNR']))
        pos_min = min(sub, key=lambda r: fnum(r['positive_ratio']))
        worst_min = min(sub, key=lambda r: fnum(r['worst_dPSNR']))
        severe_max = max(sub, key=lambda r: fnum(r['severe_loss_per_600']))
        out.append({
            'family': fam,
            'alpha': alpha,
            'group_count': len(sub),
            'min_group_mean_dPSNR': fnum(mean_min['mean_dPSNR']),
            'min_group_mean_group': mean_min['group'],
            'min_group_mean_bin': mean_min['bin'],
            'min_group_hard_bottom25_dPSNR': fnum(hard_min['hard_bottom25_dPSNR']),
            'min_group_hard_group': hard_min['group'],
            'min_group_hard_bin': hard_min['bin'],
            'min_group_easy_top25_dPSNR': fnum(easy_min['easy_top25_dPSNR']),
            'min_group_easy_group': easy_min['group'],
            'min_group_easy_bin': easy_min['bin'],
            'min_group_positive_ratio': fnum(pos_min['positive_ratio']),
            'min_group_positive_group': pos_min['group'],
            'min_group_positive_bin': pos_min['bin'],
            'worst_group_worst_dPSNR': fnum(worst_min['worst_dPSNR']),
            'worst_group_worst_group': worst_min['group'],
            'worst_group_worst_bin': worst_min['bin'],
            'max_group_severe_per_600': fnum(severe_max['severe_loss_per_600']),
            'max_group_severe_group': severe_max['group'],
            'max_group_severe_bin': severe_max['bin'],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--expert', choices=['fsudp', 'mbtaylor', 'wdmamba'], required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--convir-its-dir', required=True)
    ap.add_argument('--udp-repo', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--depth-cache-dir', required=True)
    ap.add_argument('--a0-checkpoint', required=True)
    ap.add_argument('--fulludp-checkpoint', required=True)
    ap.add_argument('--expert-checkpoint', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--splits', nargs='+', default=['val_regular', 'val_hard'])
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--print-freq', type=int, default=25)
    ap.add_argument('--mbtaylor-repo', default='/sda/home/wangyuxin/ConvIR-B/repos/external_experts/MB-TaylorFormerV2')
    ap.add_argument('--wdmamba-repo', default='/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba')
    args = ap.parse_args()
    alphas = sorted(set(round(float(a), 6) for a in args.alphas))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, build_convir = load_convir_builders(Path(args.convir_its_dir))
    a0 = load_a0_model(build_convir, Path(args.a0_checkpoint), device)
    full = load_udpnet_model(load_udp_builder_file(Path(args.udp_repo), 'ConvIR_UDPNet.py'), Path(args.fulludp_checkpoint), device)[0]
    cand = {'fsudp': load_fsudp, 'mbtaylor': load_mbtaylor, 'wdmamba': load_wdmamba}[args.expert](args, device)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with torch.no_grad():
        for split in args.splits:
            names = c2.load_split_names(Path(args.split_json), split)
            if args.max_images:
                names = names[:args.max_images]
            for idx, name in enumerate(names, 1):
                inp, label, depth = c2.load_sample(Path(args.data_dir), Path(args.depth_cache_dir), name, 'train')
                rgb = inp.unsqueeze(0).to(device)
                lab = label.unsqueeze(0).to(device)
                dep = depth.unsqueeze(0).to(device)
                _, _, h, w = rgb.shape
                trans = load_transmission(Path(args.data_dir), name, (w, h))
                rgb_pad, _, _, hp, wp = pad_to(rgb, 32)
                dep_pad = F.interpolate(dep, size=(hp, wp), mode='bicubic', align_corners=False)
                a0_pred = infer_one(a0, rgb_pad, h, w)
                a0_psnr, a0_ssim = metric(a0_pred, lab, hp, wp)
                full_pred = infer_one(full, torch.cat([rgb_pad, dep_pad], 1), h, w)
                full_psnr, full_ssim = metric(full_pred, lab, hp, wp)
                cand_pred, chp, cwp = infer_candidate(args.expert, cand, rgb, dep, h, w)
                cand_psnr, cand_ssim = metric(cand_pred, lab, chp, cwp)
                rec: dict[str, Any] = {
                    'name': name,
                    'split': split,
                    'A0_PSNR': a0_psnr,
                    'A0_SSIM': a0_ssim,
                    'FullUDP_PSNR': full_psnr,
                    'FullUDP_SSIM': full_ssim,
                    'expert_PSNR': cand_psnr,
                    'expert_SSIM': cand_ssim,
                    'dPSNR_endpoint': cand_psnr - a0_psnr,
                    'dSSIM_endpoint': cand_ssim - a0_ssim,
                    'fulludp_dPSNR': full_psnr - a0_psnr,
                    'fulludp_dSSIM': full_ssim - a0_ssim,
                }
                add_feature_values(rec, rgb, lab, dep, trans, a0_pred, full_pred, cand_pred)
                for a in alphas:
                    if a == 0:
                        p = a0_pred
                        pf = a0_pred
                    else:
                        p = torch.clamp(a0_pred + a * (cand_pred - a0_pred), 0, 1)
                        pf = torch.clamp(a0_pred + a * (full_pred - a0_pred), 0, 1)
                    ps, ss = metric(p, lab, hp, wp)
                    rec[f'expert_{alpha_key(a)}_PSNR'] = ps
                    rec[f'expert_{alpha_key(a)}_dPSNR'] = ps - a0_psnr
                    rec[f'expert_{alpha_key(a)}_dSSIM'] = ss - a0_ssim
                    psf, ssf = metric(pf, lab, hp, wp)
                    rec[f'fulludp_{alpha_key(a)}_PSNR'] = psf
                    rec[f'fulludp_{alpha_key(a)}_dPSNR'] = psf - a0_psnr
                    rec[f'fulludp_{alpha_key(a)}_dSSIM'] = ssf - a0_ssim
                full_ds = [rec[f'fulludp_{alpha_key(a)}_dPSNR'] for a in alphas]
                exp_ds = [rec[f'expert_{alpha_key(a)}_dPSNR'] for a in alphas]
                s0 = max(full_ds)
                s1 = max(full_ds + exp_ds)
                rec['S0_alpha_oracle_dPSNR'] = s0
                rec['S1_alpha_oracle_dPSNR'] = s1
                rec['oracle_gain_over_S0'] = s1 - s0
                rec['S1_best_family'] = 'expert' if max(exp_ds) > s0 else 'S0'
                rec['expert_unique_win'] = max(exp_ds) > s0 + 1e-9
                rows.append(rec)
                if idx % args.print_freq == 0:
                    print('progress', args.expert, split, idx, 'elapsed', time.time() - t0, flush=True)

    write_csv(out / f'{args.prefix}_per_image.csv', rows)
    endpoint = [{**r, 'dPSNR': r['dPSNR_endpoint'], 'dSSIM': r['dSSIM_endpoint']} for r in rows]
    single = []
    for scope, sub in [('all', endpoint)] + [(s, [r for r in endpoint if r['split'] == s]) for s in args.splits]:
        single.append({'scope': scope, **summarize(sub)})
    write_csv(out / f'{args.prefix}_single_summary.csv', single)

    ag = []
    for fam in ['expert', 'fulludp']:
        for a in alphas:
            sub = [{**r, 'dPSNR': r[f'{fam}_{alpha_key(a)}_dPSNR'], 'dSSIM': r[f'{fam}_{alpha_key(a)}_dSSIM']} for r in rows]
            ag.append({'family': fam, 'alpha': a, 'scope': 'all', **summarize(sub)})
    write_csv(out / f'{args.prefix}_alpha_grid.csv', ag)
    ag_groups = alpha_group_rows(rows, args.splits, alphas)
    write_csv(out / f'{args.prefix}_alpha_group_metrics.csv', ag_groups)
    write_csv(out / f'{args.prefix}_alpha_group_min.csv', alpha_group_min_rows(ag_groups))

    s0_rows = [{**r, 'dPSNR': r['S0_alpha_oracle_dPSNR'], 'dSSIM': 0} for r in rows]
    s1_rows = [{**r, 'dPSNR': r['S1_alpha_oracle_dPSNR'], 'dSSIM': 0} for r in rows]
    gain_rows = [{**r, 'dPSNR': r['oracle_gain_over_S0'], 'dSSIM': 0} for r in rows]
    oracle = [
        {'oracle': 'S0_fulludp_alpha', 'scope': 'all', **summarize(s0_rows)},
        {'oracle': 'S1_plus_expert_alpha', 'scope': 'all', **summarize(s1_rows)},
        {'oracle': 'gain_over_S0', 'scope': 'all', **summarize(gain_rows)},
    ]
    write_csv(out / f'{args.prefix}_oracle_vs_s0.csv', oracle)
    write_csv(out / f'{args.prefix}_group_metrics.csv', group_rows(rows, args.splits))

    hard_cut = np.quantile([r['A0_PSNR'] for r in rows], 0.25)
    hard = [r for r in rows if r['A0_PSNR'] <= hard_cut or r['split'] == 'val_hard']
    unique = [r for r in hard if r['expert_unique_win']]
    unique_summary = [{
        'scope': 'hard_or_val_hard',
        'count': len(hard),
        'unique_win_count': len(unique),
        'unique_win_rate': len(unique) / max(1, len(hard)),
        'mean_oracle_gain_unique': statistics.mean([r['oracle_gain_over_S0'] for r in unique]) if unique else 0.0,
        'residual_cosine_mean_unique': statistics.mean([r['residual_cosine_fulludp_expert'] for r in unique]) if unique else 0.0,
    }]
    write_csv(out / f'{args.prefix}_unique_wins.csv', unique_summary)

    gain = summarize(gain_rows)
    decision = 'PASS_COMPLEMENTARITY_SIGNAL' if (
        len(unique) / max(1, len(hard)) >= 0.05
        or gain['hard_bottom25_dPSNR'] >= 0.05
        or gain['mean_dPSNR'] >= 0.05
    ) else 'NO_COMPLEMENTARITY_SIGNAL'
    (out / f'{args.prefix}_decision.md').write_text(
        f'# {args.prefix} decision\n\n'
        f'Decision: `{decision}`\n\n'
        f'Unique win rate on hard/red-flag scope: `{len(unique) / max(1, len(hard)):.6f}`. '
        f'Mean oracle gain over S0: `{gain["mean_dPSNR"]:.6f}`. '
        f'Hard-bottom25 oracle gain over S0: `{gain["hard_bottom25_dPSNR"]:.6f}`. '
        f'Positive Wilson LCB: `{gain["positive_Wilson_LCB"]:.6f}`. '
        f'Severe Wilson UCB: `{gain["severe_Wilson_UCB"]:.6f}`. Locked test untouched.\n',
        encoding='utf-8',
    )
    manifest = {
        'expert': args.expert,
        'expert_checkpoint': str(args.expert_checkpoint),
        'expert_sha256': sha256(Path(args.expert_checkpoint)),
        'fulludp_sha256': sha256(Path(args.fulludp_checkpoint)),
        'a0_sha256': sha256(Path(args.a0_checkpoint)),
        'image_count': len(rows),
        'alpha_grid': alphas,
        'fsudp_builder_patch': 'FSNet_UDPNet fusion OCAB num_heads=1->2 at import time' if args.expert == 'fsudp' else None,
        'decision': decision,
        'locked_test_touched': False,
    }
    (out / f'{args.prefix}_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('V26_RESIDUAL_SHRINKAGE_ALPHA_CURVE_OK', args.expert, decision)


if __name__ == '__main__':
    main()
