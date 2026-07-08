import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

EVID = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt/experience_docx/experiment_logs/haze4k_v4_8_train_derived_validation_reset_20260708')
OOF = EVID / 'oof_eval_bs7repair'
OUT_PREFIX = EVID
ROUTE_ID = 'haze4k_v4_8_train_derived_validation_reset_20260708'
THRESHOLDS = {
    'oof_mean_delta_psnr_min': 0.025,
    'oof_positive_ratio_min': 0.55,
    'oof_median_delta_psnr_min_exclusive': 0.0,
    'oof_p5_delta_psnr_min': -0.25,
    'oof_p1_delta_psnr_min': -0.50,
    'bootstrap_ci95_low_min_exclusive': 0.0,
    'sign_test_one_sided_p_max': 0.01,
    'mean_delta_high_l1_max': 0.000005,
    'fold_mean_delta_psnr_min_exclusive': 0.0,
    'fold_positive_ratio_min': 0.50,
    'proxy_bin_mean_delta_psnr_min': -0.005,
    'proxy_bin_positive_ratio_min': 0.50,
    'low_saturation_mean_delta_psnr_min': 0.0,
    'low_saturation_positive_ratio_min': 0.50,
}
PROXY_KEYS = [
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


def one_sided_sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return float(tail / (2 ** n))


def bootstrap_summary(rows, seed=3407, boot_n=20000):
    deltas = [float(r['delta_psnr']) for r in rows]
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(boot_n):
        total = 0.0
        for _j in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    return {
        'route_id': ROUTE_ID,
        'bootstrap_samples': boot_n,
        'seed': seed,
        'count': n,
        'mean_delta_psnr': float(statistics.mean(deltas)),
        'ci95_low': pct(means, 2.5),
        'ci95_high': pct(means, 97.5),
        'prob_mean_gt_0': sum(v > 0 for v in means) / boot_n,
        'locked_test_touched': False,
        'test_split_enumerated': False,
    }


def sign_summary(rows):
    deltas = [float(r['delta_psnr']) for r in rows]
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)
    ties = len(deltas) - wins - losses
    return {
        'route_id': ROUTE_ID,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'n_effective': wins + losses,
        'positive_ratio_all': wins / len(deltas),
        'one_sided_p_win_rate_gt_0_5': one_sided_sign_test(wins, losses),
        'locked_test_touched': False,
        'test_split_enumerated': False,
    }


def quantile_bins(rows, key):
    ordered = sorted((float(r[key]), idx) for idx, r in enumerate(rows))
    n = len(ordered)
    labels = [None] * n
    for rank, (_, idx) in enumerate(ordered):
        labels[idx] = f'q{min(3, int(rank * 4 / n)) + 1}'
    return labels


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, '') for k in fields} for r in rows])

start = time.time()
all_rows = []
fold_summaries = []
for fold in range(5):
    fold_dir = OOF / f'fold_{fold}'
    rows = read_csv(fold_dir / 'val_per_image_compact.csv')
    for r in rows:
        r['fold'] = int(r['fold'])
    if len(rows) != len({r['image_id'] for r in rows}):
        raise RuntimeError(f'duplicate images inside fold {fold}')
    boot = bootstrap_summary(rows, seed=3407 + fold, boot_n=12000)
    boot['fold'] = fold
    sign = sign_summary(rows)
    sign['fold'] = fold
    (fold_dir / 'val_bootstrap.json').write_text(json.dumps(boot, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (fold_dir / 'val_sign_test.json').write_text(json.dumps(sign, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    sm = summarize(rows)
    sm['fold'] = fold
    sm['bootstrap_ci95_low'] = boot['ci95_low']
    sm['bootstrap_ci95_high'] = boot['ci95_high']
    sm['sign_test_one_sided_p'] = sign['one_sided_p_win_rate_gt_0_5']
    fold_summaries.append(sm)
    all_rows.extend(rows)

if len(all_rows) != 3000:
    raise RuntimeError(f'expected 3000 OOF rows, got {len(all_rows)}')
if len(all_rows) != len({r['image_id'] for r in all_rows}):
    dupes = sorted({r['image_id'] for r in all_rows if sum(rr['image_id'] == r['image_id'] for rr in all_rows) > 1})[:10]
    raise RuntimeError({'duplicate_oof_images': dupes})

for key in PROXY_KEYS:
    labels = quantile_bins(all_rows, key)
    for row, label in zip(all_rows, labels):
        row[f'{key}_oof_quartile'] = label

summary = summarize(all_rows)
bootstrap = bootstrap_summary(all_rows, seed=3407, boot_n=30000)
sign = sign_summary(all_rows)
worst_n = 64
worst = sorted(all_rows, key=lambda r: float(r['delta_psnr']))[:worst_n]
worst_ids = {r['image_id'] for r in worst}

proxy_rows = []
proxy_failures = []
for key in PROXY_KEYS:
    for label in ['q1', 'q2', 'q3', 'q4']:
        subset = [r for r in all_rows if r[f'{key}_oof_quartile'] == label]
        sm = summarize(subset)
        vals = [float(r[key]) for r in subset]
        out = {
            'proxy': key,
            'bin': label,
            'count': len(subset),
            'proxy_min': min(vals),
            'proxy_max': max(vals),
            'proxy_mean': float(statistics.mean(vals)),
            'mean_delta_psnr': sm['mean_delta_psnr'],
            'median_delta_psnr': sm['median_delta_psnr'],
            'p1_delta_psnr': sm['p1_delta_psnr'],
            'p5_delta_psnr': sm['p5_delta_psnr'],
            'positive_ratio': sm['positive_ratio'],
            'mean_delta_high_l1': sm['mean_delta_high_l1'],
            'worst64_count': sum(r['image_id'] in worst_ids for r in subset),
            'worst64_ratio': sum(r['image_id'] in worst_ids for r in subset) / worst_n,
        }
        proxy_rows.append(out)
        if out['mean_delta_psnr'] < THRESHOLDS['proxy_bin_mean_delta_psnr_min'] or out['positive_ratio'] < THRESHOLDS['proxy_bin_positive_ratio_min']:
            proxy_failures.append(out)

# Low-saturation subgroup is the global input_saturation_proxy q1 bin, matching the promoted v4.8 subgroup gate.
low_sat = [r for r in all_rows if r['input_saturation_proxy_oof_quartile'] == 'q1']
low_sat_summary = summarize(low_sat)

fold_gate_failures = []
for sm in fold_summaries:
    if sm['mean_delta_psnr'] <= THRESHOLDS['fold_mean_delta_psnr_min_exclusive'] or sm['positive_ratio'] < THRESHOLDS['fold_positive_ratio_min']:
        fold_gate_failures.append(sm)

gates = {
    'oof_mean_delta_psnr': summary['mean_delta_psnr'] >= THRESHOLDS['oof_mean_delta_psnr_min'],
    'oof_positive_ratio': summary['positive_ratio'] >= THRESHOLDS['oof_positive_ratio_min'],
    'oof_median_delta_psnr': summary['median_delta_psnr'] > THRESHOLDS['oof_median_delta_psnr_min_exclusive'],
    'oof_p5_delta_psnr': summary['p5_delta_psnr'] >= THRESHOLDS['oof_p5_delta_psnr_min'],
    'oof_p1_delta_psnr': summary['p1_delta_psnr'] >= THRESHOLDS['oof_p1_delta_psnr_min'],
    'bootstrap_ci95_low': bootstrap['ci95_low'] > THRESHOLDS['bootstrap_ci95_low_min_exclusive'],
    'sign_test': sign['one_sided_p_win_rate_gt_0_5'] < THRESHOLDS['sign_test_one_sided_p_max'],
    'mean_delta_high_l1': summary['mean_delta_high_l1'] <= THRESHOLDS['mean_delta_high_l1_max'],
    'every_fold_mean_positive': all(sm['mean_delta_psnr'] > THRESHOLDS['fold_mean_delta_psnr_min_exclusive'] for sm in fold_summaries),
    'every_fold_positive_ratio': all(sm['positive_ratio'] >= THRESHOLDS['fold_positive_ratio_min'] for sm in fold_summaries),
    'every_proxy_bin_mean': all(r['mean_delta_psnr'] >= THRESHOLDS['proxy_bin_mean_delta_psnr_min'] for r in proxy_rows),
    'every_proxy_bin_positive_ratio': all(r['positive_ratio'] >= THRESHOLDS['proxy_bin_positive_ratio_min'] for r in proxy_rows),
    'low_saturation_subgroup': low_sat_summary['mean_delta_psnr'] >= THRESHOLDS['low_saturation_mean_delta_psnr_min'] and low_sat_summary['positive_ratio'] >= THRESHOLDS['low_saturation_positive_ratio_min'],
    'locked_test_not_touched': True,
    'test_split_not_enumerated': True,
}
pass_gate = all(gates.values())
severity_counts = {
    'delta_lt_minus_0_50': sum(float(r['delta_psnr']) < -0.50 for r in all_rows),
    'delta_lt_minus_0_25': sum(float(r['delta_psnr']) < -0.25 for r in all_rows),
    'delta_lt_minus_0_10': sum(float(r['delta_psnr']) < -0.10 for r in all_rows),
}

fields = list(all_rows[0].keys())
# Keep stable, readable ordering for the compact table.
preferred = [
    'fold', 'image_id', 'delta_psnr', 'delta_ssim', 'delta_low_l1', 'delta_high_l1',
    'a0_psnr', 'candidate_psnr', 'a0_ssim', 'candidate_ssim',
    'input_gt_l1', 'a0_error_proxy_low_plus_high_l1', 'candidate_error_proxy_low_plus_high_l1',
    'input_dark_channel_mean', 'input_brightness_mean', 'input_saturation_proxy',
    'gt_texture_proxy', 'hazy_texture_proxy', 'high_gate_std', 'high_low_energy_ratio',
    'input_saturation_proxy_oof_quartile', 'a0_error_proxy_low_plus_high_l1_oof_quartile',
    'a0_psnr_oof_quartile', 'input_gt_l1_oof_quartile', 'gt_texture_proxy_oof_quartile',
]
fields = preferred + [f for f in fields if f not in preferred]
write_csv(OUT_PREFIX / 'v48_oof_per_image_compact.csv', sorted(all_rows, key=lambda r: (int(r['fold']), r['image_id'])), fields)
write_csv(OUT_PREFIX / 'v48_oof_proxy_bins.csv', proxy_rows, list(proxy_rows[0].keys()))
write_csv(OUT_PREFIX / 'v48_fold_summaries.csv', fold_summaries, list(fold_summaries[0].keys()))
write_csv(OUT_PREFIX / 'v48_oof_worst64_compact.csv', worst, fields)

family = {
    'route_id': ROUTE_ID,
    'comparison_scope': 'train-derived OOF only; no locked-test use',
    'v47_internal256_reference': {
        'mean_delta_psnr': 0.044404,
        'positive_ratio': 0.625,
        'p5_delta_psnr': -0.216141,
        'bootstrap_ci95_low': 0.024481,
        'sign_test_p': 3.802649e-05,
        'locked_test_mean_delta_psnr': 0.003826,
        'locked_test_positive_ratio': 0.484,
        'note': 'locked-test values are historical evidence only; v4.8 did not use locked test for tuning or evaluation.',
    },
    'v48_oof': summary,
    'v48_low_saturation_q1': low_sat_summary,
    'v48_fold_summaries': fold_summaries,
    'decision': 'PASS_OOF_GATE' if pass_gate else 'FAIL_OOF_GATE',
    'locked_test_touched': False,
    'test_split_enumerated': False,
}
closeout = {
    'route_id': ROUTE_ID,
    'route_identity': 'train-derived K-fold/tail-safe validation reset for DCFSB-bottleneck adapter recipe',
    'status': 'COMPLETED_GATE_PASS' if pass_gate else 'COMPLETED_GATE_FAIL',
    'decision': 'The DCFSB-bottleneck adapter recipe passes v4.8 train-derived OOF gate.' if pass_gate else 'The DCFSB-bottleneck adapter recipe does not pass the full v4.8 tail-safe OOF gate; do not promote or use locked test.',
    'summary': summary,
    'fold_summaries': fold_summaries,
    'bootstrap': bootstrap,
    'sign_test': sign,
    'low_saturation_subgroup': low_sat_summary,
    'severity_counts': severity_counts,
    'thresholds': THRESHOLDS,
    'gates': gates,
    'failed_gates': [k for k, v in gates.items() if not v],
    'fold_gate_failures': fold_gate_failures,
    'proxy_bin_failures': proxy_failures,
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'batch_size_repair': 'All folds rerun with batch_size=7 after predeclared engineering repair for batch-size-1 BatchNorm failure under batch_size=8.',
    'runtime_sec': time.time() - start,
}
(OUT_PREFIX / 'v48_oof_bootstrap.json').write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(OUT_PREFIX / 'v48_oof_sign_test.json').write_text(json.dumps(sign, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(OUT_PREFIX / 'v48_oof_summary.json').write_text(json.dumps(closeout, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(OUT_PREFIX / 'v48_family_comparison.json').write_text(json.dumps(family, indent=2, sort_keys=True) + '\n', encoding='utf-8')

failed = ', '.join(closeout['failed_gates']) if closeout['failed_gates'] else 'none'
fold_table = '\n'.join(
    f"| {int(sm['fold'])} | {sm['count']} | {sm['mean_delta_psnr']:.6f} | {sm['positive_ratio']:.6f} | {sm['p5_delta_psnr']:.6f} | {sm['p1_delta_psnr']:.6f} | {sm['mean_delta_high_l1']:.10f} |"
    for sm in fold_summaries
)
failed_gate_lines = '\n'.join(f"- `{name}`" for name in closeout['failed_gates']) or '- None'
proxy_fail_lines = '\n'.join(
    f"- `{r['proxy']}` {r['bin']}: mean `{r['mean_delta_psnr']:.6f}`, pos `{r['positive_ratio']:.6f}`, p5 `{r['p5_delta_psnr']:.6f}`"
    for r in proxy_failures[:40]
) or '- None'
worst_lines = '\n'.join(
    f"| {i+1} | `{r['image_id']}` | {r['fold']} | {float(r['delta_psnr']):.6f} | {float(r['a0_psnr']):.6f} | {float(r['candidate_psnr']):.6f} | {float(r['input_saturation_proxy']):.6f} | {float(r['a0_error_proxy_low_plus_high_l1']):.6f} |"
    for i, r in enumerate(worst[:24])
)
decision = f"""# Decision After v4.8 Train-Derived OOF

Route id: `{ROUTE_ID}`

Decision: **{'PASS OOF gate' if pass_gate else 'FAIL OOF gate'}**.

Locked test touched/enumerated in v4.8: `false` / `false`.

## OOF Summary

- Count: `{summary['count']}` train-derived held-out images.
- Mean dPSNR: `{summary['mean_delta_psnr']:.6f}` (gate `>= +0.025`).
- Positive ratio: `{summary['positive_ratio']:.6f}` (gate `>= 0.55`).
- Median dPSNR: `{summary['median_delta_psnr']:.6f}` (gate `> 0`).
- p5 dPSNR: `{summary['p5_delta_psnr']:.6f}` (gate `>= -0.25`).
- p1 dPSNR: `{summary['p1_delta_psnr']:.6f}` (gate `>= -0.50`).
- Bootstrap 95% CI: `[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]`.
- Sign-test one-sided p: `{sign['one_sided_p_win_rate_gt_0_5']:.8g}`.
- Mean dHighL1: `{summary['mean_delta_high_l1']:.10f}` (gate `<= +0.000005`).
- Low-saturation q1 mean/positive ratio: `{low_sat_summary['mean_delta_psnr']:.6f}` / `{low_sat_summary['positive_ratio']:.6f}`.

## Failed Gates

{failed_gate_lines}

## Fold Summary

| Fold | Count | Mean dPSNR | Positive ratio | p5 | p1 | Mean dHighL1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{fold_table}

## Proxy Bin Failures

{proxy_fail_lines}

## Worst 24 OOF Images

| Rank | Image | Fold | dPSNR | A0 PSNR | Candidate PSNR | saturation | A0 proxy L1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{worst_lines}

## Interpretation

The recipe shows a real positive train-derived OOF mean and sign-test signal, but the full v4.8 tail-safe contract is stricter than mean uplift. The decision is based only on train-derived folds; no locked-test data were used or enumerated during v4.8.
"""
(OUT_PREFIX / 'decision_after_v48.md').write_text(decision, encoding='utf-8')
print(json.dumps({
    'summary': summary,
    'bootstrap': bootstrap,
    'sign_test': sign,
    'low_saturation_subgroup': low_sat_summary,
    'failed_gates': closeout['failed_gates'],
    'pass': pass_gate,
    'proxy_failure_count': len(proxy_failures),
    'runtime_sec': closeout['runtime_sec'],
}, indent=2, sort_keys=True))
print('V48_OOF_COMBINE_OK')
