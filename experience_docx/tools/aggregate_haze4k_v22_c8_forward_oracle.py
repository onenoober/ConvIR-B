#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ALPHAS = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.50]
SEVERE = -0.20
BOOT_N = 400


def alpha_key(a: float) -> str:
    return (('a%.6f' % a).rstrip('0').rstrip('.')).replace('.', 'p')


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


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
    return float(np.quantile(arr[idx].mean(axis=1), q))


def summarize(rows: list[dict[str, Any]], dkey: str = 'dPSNR') -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {'count': 0}
    ds = [fnum(r[dkey]) for r in rows]
    a0 = [fnum(r['A0_PSNR']) for r in rows]
    order = sorted(range(n), key=lambda i: a0[i])
    k = max(1, n // 4)
    hard = [ds[i] for i in order[:k]]
    easy = [ds[i] for i in order[-k:]]
    positive = sum(d > 0 for d in ds)
    severe = sum(d <= SEVERE for d in ds)
    rng = np.random.default_rng(3407 + n)
    return {
        'count': n,
        'mean_dPSNR': statistics.mean(ds),
        'mean_bootstrap_LCB': bootstrap_lcb(ds, rng),
        'hard_bottom25_dPSNR': statistics.mean(hard),
        'hard_bootstrap_LCB': bootstrap_lcb(hard, rng),
        'easy_top25_dPSNR': statistics.mean(easy),
        'positive_ratio': positive / n,
        'positive_Wilson_LCB': wilson_lcb(positive, n),
        'nonnegative_ratio': sum(d >= 0 for d in ds) / n,
        'severe_loss_count': severe,
        'severe_loss_per_600': severe / n * 600.0,
        'severe_Wilson_UCB': wilson_ucb(severe, n),
    }


def qbins(vals: list[float]) -> list[str]:
    arr = np.asarray(vals, dtype=float)
    qs = np.quantile(arr, [0.25, 0.5, 0.75])
    return ['q1' if v <= qs[0] else 'q2' if v <= qs[1] else 'q3' if v <= qs[2] else 'q4' for v in arr]


def best_action(row: dict[str, str], prefix: str) -> tuple[float, str]:
    vals = [(fnum(row[f'{prefix}_{alpha_key(a)}_dPSNR']), f'{prefix}:alpha={a}') for a in ALPHAS]
    return max(vals, key=lambda x: x[0])


def stage_best(base_row: dict[str, Any], experts: list[str]) -> tuple[float, str]:
    candidates = [(base_row['S0_dPSNR'], 'S0:fulludp_alpha_oracle')]
    for e in experts:
        candidates.append((base_row[f'{e}_best_dPSNR'], base_row[f'{e}_best_action']))
    return max(candidates, key=lambda x: x[0])


def build_rows(out_dir: Path, expert_files: dict[str, Path]) -> list[dict[str, Any]]:
    expert_rows = {name: read_csv(path) for name, path in expert_files.items()}
    names = None
    for e, rows in expert_rows.items():
        cur = [(r['split'], r['name']) for r in rows]
        if names is None:
            names = cur
        elif cur != names:
            raise RuntimeError(f'row order mismatch for {e}')
    assert names is not None
    merged: list[dict[str, Any]] = []
    first = next(iter(expert_rows.values()))
    for idx, row0 in enumerate(first):
        rec: dict[str, Any] = {
            'split': row0['split'],
            'name': row0['name'],
            'A0_PSNR': fnum(row0['A0_PSNR']),
            'A0_SSIM': fnum(row0.get('A0_SSIM')),
            'FullUDP_PSNR': fnum(row0['FullUDP_PSNR']),
            'fulludp_dPSNR': fnum(row0['fulludp_dPSNR']),
        }
        # Carry stable feature columns from WDMamba row when available, otherwise first row.
        feature_src = expert_rows.get('wdmamba', first)[idx]
        for k, v in feature_src.items():
            if k.startswith('feature_') or k in [
                'transmission_mean', 'transmission_std', 'haze_density_mean', 'haze_density_p90',
                'dark_channel_mean', 'input_edge_density', 'input_low_texture_proxy',
                'sky_highlight_proxy', 'airlight_proxy_p99',
            ]:
                rec[k] = fnum(v, float('nan'))
        s0_vals = [(fnum(row0[f'fulludp_{alpha_key(a)}_dPSNR']), f'fulludp:alpha={a}') for a in ALPHAS]
        rec['S0_dPSNR'], rec['S0_action'] = max(s0_vals, key=lambda x: x[0])
        for e, rows in expert_rows.items():
            er = rows[idx]
            best, action = best_action(er, 'expert')
            rec[f'{e}_best_dPSNR'] = best
            rec[f'{e}_best_action'] = action.replace('expert', e)
            rec[f'{e}_endpoint_dPSNR'] = fnum(er['dPSNR_endpoint'])
            rec[f'{e}_fulludp_mae'] = fnum(er.get('expert_fulludp_mae'), float('nan'))
            rec[f'{e}_residual_cosine'] = fnum(er.get('residual_cosine_fulludp_expert'), float('nan'))
        for stage, experts in [('S1', ['wdmamba']), ('S2', ['wdmamba', 'fsudp']), ('S3', ['wdmamba', 'fsudp', 'mbtaylor'])]:
            available = [e for e in experts if e in expert_rows]
            best, action = stage_best(rec, available)
            rec[f'{stage}_dPSNR'] = best
            rec[f'{stage}_action'] = action
            rec[f'{stage}_gain_over_S0'] = best - rec['S0_dPSNR']
            rec[f'{stage}_selected_expert'] = action.split(':', 1)[0]
        merged.append(rec)
    return merged


def composition(rows: list[dict[str, Any]], stage: str, group: str, key: str | None = None) -> list[dict[str, Any]]:
    if key is None:
        bins = [r['split'] for r in rows]
    else:
        vals = [fnum(r.get(key), float('nan')) for r in rows]
        finite = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
        qb = qbins([v for _, v in finite]) if finite else []
        m = {i: b for (i, _), b in zip(finite, qb)}
        bins = [m.get(i, 'nan') for i in range(len(rows))]
    out: list[dict[str, Any]] = []
    for b in sorted(set(bins)):
        if b == 'nan':
            continue
        sub0 = [r for r, bb in zip(rows, bins) if bb == b]
        if not sub0:
            continue
        sub = [{**r, 'dPSNR': r[f'{stage}_dPSNR']} for r in sub0]
        counts: dict[str, int] = {}
        for r in sub0:
            counts[r[f'{stage}_selected_expert']] = counts.get(r[f'{stage}_selected_expert'], 0) + 1
        rec = {'stage': stage, 'group': group, 'bin': b, **summarize(sub)}
        for expert in ['S0', 'wdmamba', 'fsudp', 'mbtaylor']:
            rec[f'composition_{expert}'] = counts.get(expert, 0) / len(sub0)
        out.append(rec)
    return out


def write_stage_outputs(out_dir: Path, rows: list[dict[str, Any]], stage: str, prefix: str, experts: list[str]) -> None:
    oracle_rows = []
    for label, key in [('S0_fulludp_alpha', 'S0_dPSNR'), (stage, f'{stage}_dPSNR'), (f'{stage}_gain_over_S0', f'{stage}_gain_over_S0')]:
        oracle_rows.append({'oracle': label, 'scope': 'all', **summarize([{**r, 'dPSNR': r[key]} for r in rows])})
        for split in sorted(set(r['split'] for r in rows)):
            sub0 = [r for r in rows if r['split'] == split]
            oracle_rows.append({'oracle': label, 'scope': split, **summarize([{**r, 'dPSNR': r[key]} for r in sub0])})
    write_csv(out_dir / f'{prefix}_forward_selection_oracle.csv', oracle_rows)

    comp: list[dict[str, Any]] = []
    group_specs = [
        ('split', None),
        ('A0-PSNR_q4', 'A0_PSNR'),
        ('FullUDP-A0_diff_signed_q4', 'feature_diff_signed_mean'),
        ('FullUDP-A0_diff_abs_q4', 'feature_diff_abs_mean'),
        ('haze_density_q4', 'haze_density_mean'),
        ('transmission_q4', 'transmission_mean'),
        ('airlight_proxy_q4', 'airlight_proxy_p99'),
        ('depth_q4', 'feature_depth_mean'),
        ('dark_channel_q4', 'dark_channel_mean'),
        ('low_texture_q4', 'input_low_texture_proxy'),
        ('edge_density_q4', 'input_edge_density'),
        ('sky_highlight_proxy_q4', 'sky_highlight_proxy'),
    ]
    for g, k in group_specs:
        comp.extend(composition(rows, stage, g, k))
    write_csv(out_dir / f'{prefix}_expert_composition_by_group.csv', comp)

    hard_cut = float(np.quantile([r['A0_PSNR'] for r in rows], 0.25))
    hard = [r for r in rows if r['A0_PSNR'] <= hard_cut or r['split'] == 'val_hard']
    unique_rows = []
    for e in experts:
        unique = [r for r in hard if r[f'{e}_best_dPSNR'] > max([r['S0_dPSNR']] + [r[f'{o}_best_dPSNR'] for o in experts if o != e]) + 1e-9]
        unique_rows.append({
            'expert': e,
            'scope': 'hard_or_val_hard',
            'count': len(hard),
            'unique_win_count_vs_all_others': len(unique),
            'unique_win_rate_vs_all_others': len(unique) / max(1, len(hard)),
            'mean_margin_unique': statistics.mean([r[f'{e}_best_dPSNR'] - max([r['S0_dPSNR']] + [r[f'{o}_best_dPSNR'] for o in experts if o != e]) for r in unique]) if unique else 0.0,
        })
    write_csv(out_dir / f'{prefix}_unique_wins_by_expert.csv', unique_rows)

    removal = []
    full_key = f'{stage}_dPSNR'
    removal.append({'ablation': 'full', **summarize([{**r, 'dPSNR': r[full_key]} for r in rows])})
    for e in experts:
        kept = [o for o in experts if o != e]
        vals = []
        for r in rows:
            best = max([r['S0_dPSNR']] + [r[f'{o}_best_dPSNR'] for o in kept])
            vals.append({**r, 'dPSNR': best})
        summ = summarize(vals)
        full_s = summarize([{**r, 'dPSNR': r[full_key]} for r in rows])
        removal.append({'ablation': f'remove_{e}', **summ, 'mean_drop_vs_full': full_s['mean_dPSNR'] - summ['mean_dPSNR'], 'hard_drop_vs_full': full_s['hard_bottom25_dPSNR'] - summ['hard_bottom25_dPSNR']})
    write_csv(out_dir / f'{prefix}_expert_removal_ablation.csv', removal)

    labels = [{
        'split': r['split'], 'name': r['name'], 'A0_PSNR': r['A0_PSNR'], 'S0_dPSNR': r['S0_dPSNR'],
        f'{stage}_dPSNR': r[f'{stage}_dPSNR'], f'{stage}_gain_over_S0': r[f'{stage}_gain_over_S0'],
        f'{stage}_action': r[f'{stage}_action'], f'{stage}_selected_expert': r[f'{stage}_selected_expert'],
    } for r in rows]
    write_csv(out_dir / f'{prefix}_oracle_labels.csv', labels)

    gain = summarize([{**r, 'dPSNR': r[f'{stage}_gain_over_S0']} for r in rows])
    full = summarize([{**r, 'dPSNR': r[f'{stage}_dPSNR']} for r in rows])
    unique_ok = any(r['unique_win_rate_vs_all_others'] >= 0.05 for r in unique_rows)
    pass_gate = unique_ok and gain['hard_bottom25_dPSNR'] >= 0.05 and gain['mean_dPSNR'] >= 0.02 and full['severe_loss_count'] == 0
    decision = 'PASS_C8_COMPLEMENTARITY' if pass_gate else 'STOP_OR_EXPAND_EXPERT_POOL'
    (out_dir / f'{prefix}_decision.md').write_text(
        f'# {prefix} decision\n\n'
        f'Decision: `{decision}`\n\n'
        f'- stage: `{stage}`\n'
        f'- experts: `{", ".join(experts)}`\n'
        f'- mean oracle gain over S0: `{gain["mean_dPSNR"]:.6f}`\n'
        f'- hard-bottom25 oracle gain over S0: `{gain["hard_bottom25_dPSNR"]:.6f}`\n'
        f'- positive Wilson LCB for gain: `{gain["positive_Wilson_LCB"]:.6f}`\n'
        f'- severe count in selected oracle: `{full["severe_loss_count"]}`\n'
        f'- hard/red-flag unique-win gate: `{unique_ok}`\n'
        f'- locked test untouched: `true`\n',
        encoding='utf-8',
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    files = {
        'wdmamba': out / 'v22_c8_1_wdmamba_full_per_image.csv',
        'fsudp': out / 'v22_c8_2_fsudp_full_per_image.csv',
        'mbtaylor': out / 'v22_c8_3_mbtaylor_full_per_image.csv',
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    rows = build_rows(out, files)
    write_csv(out / 'v22_c8_forward_selection_per_image.csv', rows)
    write_stage_outputs(out, rows, 'S2', 'v22_c8_2_s2', ['wdmamba', 'fsudp'])
    write_stage_outputs(out, rows, 'S3', 'v22_c8_3_s3', ['wdmamba', 'fsudp', 'mbtaylor'])
    manifest = {
        'rows': len(rows),
        'inputs': {k: str(v) for k, v in files.items()},
        'stages': {'S0': 'FullUDP alpha oracle', 'S1': 'S0 + WDMamba', 'S2': 'S1 + FSNet+UDP', 'S3': 'S2 + MB-TaylorFormerV2-L'},
        'locked_test_touched': False,
    }
    (out / 'v22_c8_forward_selection_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('C8_FORWARD_ORACLE_OK', len(rows))


if __name__ == '__main__':
    main()
