#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPERTS = [
    ('wdmamba', 'WDMamba', 'v26_wdmamba_alpha_curve'),
    ('fsudp', 'FSNet+UDP', 'v26_fsudp_alpha_curve'),
    ('mbtaylor', 'MB-TaylorFormerV2-L', 'v26_mbtaylor_alpha_curve'),
]


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def best_alpha(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r['family'] == 'expert' and fnum(r['alpha']) > 0]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            fnum(r['positive_ratio']),
            -fnum(r['severe_loss_per_600']),
            fnum(r['mean_dPSNR']),
            fnum(r['hard_bottom25_dPSNR']),
        ),
    )


def metric_delta(row: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, float]:
    return {
        'mean_vs_full': fnum(row['mean_dPSNR']) - fnum(endpoint['mean_dPSNR']),
        'hard_vs_full': fnum(row['hard_bottom25_dPSNR']) - fnum(endpoint['hard_bottom25_dPSNR']),
        'easy_vs_full': fnum(row['easy_top25_dPSNR']) - fnum(endpoint['easy_top25_dPSNR']),
        'positive_vs_full': fnum(row['positive_ratio']) - fnum(endpoint['positive_ratio']),
        'severe_per_600_vs_full': fnum(row['severe_loss_per_600']) - fnum(endpoint['severe_loss_per_600']),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--prefix', default='v26')
    args = ap.parse_args()

    alpha_rows: list[dict[str, Any]] = []
    group_min_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    decisions: list[str] = []

    for expert_id, expert_name, prefix in EXPERTS:
        alpha_path = args.evidence_dir / f'{prefix}_alpha_grid.csv'
        group_min_path = args.evidence_dir / f'{prefix}_alpha_group_min.csv'
        if not alpha_path.is_file():
            raise FileNotFoundError(alpha_path)
        if not group_min_path.is_file():
            raise FileNotFoundError(group_min_path)
        rows = read_csv(alpha_path)
        gm_rows = read_csv(group_min_path)
        for row in rows:
            row = dict(row)
            row['expert_id'] = expert_id
            row['expert_name'] = expert_name
            alpha_rows.append(row)
        for row in gm_rows:
            row = dict(row)
            row['expert_id'] = expert_id
            row['expert_name'] = expert_name
            group_min_rows.append(row)

        expert_rows = [r for r in rows if r['family'] == 'expert']
        endpoint = next((r for r in expert_rows if abs(fnum(r['alpha']) - 1.0) < 1e-9), None)
        wd0375 = next((r for r in expert_rows if abs(fnum(r['alpha']) - 0.375) < 1e-9), None)
        best = best_alpha(rows)
        if endpoint is None or best is None:
            raise RuntimeError(f'missing endpoint or best alpha for {expert_id}')

        for role, row in [('endpoint_alpha1', endpoint), ('alpha0375', wd0375), ('best_safety_alpha', best)]:
            if row is None:
                continue
            out = {
                'expert_id': expert_id,
                'expert_name': expert_name,
                'role': role,
                'alpha': fnum(row['alpha']),
                'mean_dPSNR': fnum(row['mean_dPSNR']),
                'hard_bottom25_dPSNR': fnum(row['hard_bottom25_dPSNR']),
                'easy_top25_dPSNR': fnum(row['easy_top25_dPSNR']),
                'worst_dPSNR': fnum(row['worst_dPSNR']),
                'dSSIM': fnum(row['dSSIM']),
                'positive_ratio': fnum(row['positive_ratio']),
                'nonnegative_ratio': fnum(row['nonnegative_ratio']),
                'severe_loss_per_600': fnum(row['severe_loss_per_600']),
            }
            out.update(metric_delta(row, endpoint))
            compact_rows.append(out)

        safe = [
            r for r in expert_rows
            if fnum(r['positive_ratio']) >= 0.90
            and fnum(r['severe_loss_per_600']) <= 48.0
            and fnum(r['mean_dPSNR']) > 0
            and fnum(r['hard_bottom25_dPSNR']) > 0
        ]
        alpha_list = ', '.join(str(fnum(r['alpha'])) for r in safe) if safe else 'none'
        decisions.append(
            f'- {expert_name}: safe positive/tail alpha set `{alpha_list}`; '
            f'endpoint severe `{fnum(endpoint["severe_loss_per_600"]):.1f}/600`; '
            f'best safety alpha `{fnum(best["alpha"]):.3f}`.'
        )

    write_csv(args.evidence_dir / f'{args.prefix}_all_expert_alpha_grid.csv', alpha_rows)
    write_csv(args.evidence_dir / f'{args.prefix}_all_expert_group_min.csv', group_min_rows)
    write_csv(args.evidence_dir / f'{args.prefix}_compact_comparison.csv', compact_rows)

    wdmamba_rows = [r for r in alpha_rows if r['expert_id'] == 'wdmamba' and r['family'] == 'expert']
    cross_experts = sorted({r['expert_name'] for r in alpha_rows if r['family'] == 'expert'})
    summary = {
        'route': 'haze4k_v2_6_residual_shrinkage_alpha_curves_20260616',
        'locked_test_touched': False,
        'splits': ['val_regular', 'val_hard'],
        'experts': cross_experts,
        'alpha_count': len({fnum(r['alpha']) for r in alpha_rows if r['family'] == 'expert'}),
        'row_count': len(alpha_rows),
        'wdmamba_best_mean_alpha': max(wdmamba_rows, key=lambda r: fnum(r['mean_dPSNR']))['alpha'] if wdmamba_rows else None,
        'wdmamba_best_positive_tail_alpha': best_alpha(wdmamba_rows)['alpha'] if wdmamba_rows and best_alpha(wdmamba_rows) else None,
    }
    (args.evidence_dir / f'{args.prefix}_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    decision = (
        '# Haze4K v2.6 Residual Shrinkage Alpha Curves Decision\n\n'
        'Decision: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`\n\n'
        'This supplemental route evaluates fixed anchor-preserving residual shrinkage on '
        'the C8 train-derived `val_regular + val_hard` scope only. It does not read or '
        'write locked Haze4K evidence and does not tune from the prior WD0375 locked result.\n\n'
        '## Main Readout\n\n'
        + '\n'.join(decisions)
        + '\n\n## Evidence Files\n\n'
        f'- `{args.prefix}_all_expert_alpha_grid.csv`\n'
        f'- `{args.prefix}_all_expert_group_min.csv`\n'
        f'- `{args.prefix}_compact_comparison.csv`\n'
        f'- `{args.prefix}_summary.json`\n'
        '\nLocked-test status: `locked_test_touched=false`.\n'
    )
    (args.evidence_dir / f'{args.prefix}_decision.md').write_text(decision, encoding='utf-8')
    print('V26_SUMMARY_OK')


if __name__ == '__main__':
    main()
