#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--haze-dir', type=Path, required=True)
    ap.add_argument('--depth-cache-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    names = sorted(p.name for p in args.haze_dir.iterdir() if p.is_file())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0
    skipped = 0
    missing = []
    for name in names:
        src = args.depth_cache_dir / f'{name}.npy'
        dst = args.out_dir / f'{Path(name).stem}.png'
        if not src.is_file():
            missing.append(name)
            continue
        if dst.is_file() and not args.overwrite:
            skipped += 1
            continue
        arr = np.load(src)
        out = normalize_to_uint8(arr)
        Image.fromarray(out, mode='L').save(dst)
        rows.append({
            'name': name,
            'source': str(src),
            'output': str(dst),
            'shape': list(arr.shape),
            'src_min': float(np.nanmin(arr)),
            'src_max': float(np.nanmax(arr)),
            'src_mean': float(np.nanmean(arr)),
            'src_std': float(np.nanstd(arr)),
        })
        written += 1
    report = {
        'method': 'per-image min-max normalization of raw DepthAnything V2 cache to uint8 PNG, then official UDPNet dataloader reads PIL L image and ToTensor scales to 0..1',
        'haze_dir': str(args.haze_dir),
        'depth_cache_dir': str(args.depth_cache_dir),
        'out_dir': str(args.out_dir),
        'input_count': len(names),
        'written': written,
        'skipped': skipped,
        'missing_count': len(missing),
        'missing_examples': missing[:10],
        'sample_rows': rows[:10],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if missing:
        raise RuntimeError(f'missing depth cache for {len(missing)} images')
    print(f'V211_DEPTH2L_OK input={len(names)} written={written} skipped={skipped}')


if __name__ == '__main__':
    main()
