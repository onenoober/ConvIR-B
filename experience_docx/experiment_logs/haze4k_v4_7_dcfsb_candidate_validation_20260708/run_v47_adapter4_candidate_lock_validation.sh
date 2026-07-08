#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent
ROUTE_ID=haze4k_v4_7_dcfsb_candidate_validation_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
CAND=$WORK/Dehazing/ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl
SRC_EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_6_dcfsb_bottleneck_20260708_adapter4
PER_IMAGE=$SRC_EVID/v46_dcfsb_adapter4_per_image.csv
MODULE_STATS=$SRC_EVID/v46_dcfsb_adapter4_module_stats.jsonl
HOLDOUT=$WORK/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_internal_holdout256.txt
STATUS=$EVID/status.txt
LOG=$EVID/run_v47_adapter4_candidate_lock_validation.log
mkdir -p "$EVID"
{
  echo "run_start v47_adapter4_candidate_lock $(date --iso-8601=seconds)"
  echo "state=RUNNING_AUDIT"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CAND"
  echo "source_per_image=$PER_IMAGE"
  echo "source_module_stats=$MODULE_STATS"
  echo "holdout=$HOLDOUT"
  echo "locked_test_policy=blocked; train-derived internal_holdout256 only"
} | tee -a "$STATUS"
if [ ! -x "$PY" ]; then echo "V47_PRECHECK_FAILED python_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$PER_IMAGE" ]; then echo "V47_PRECHECK_FAILED per_image_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$MODULE_STATS" ]; then echo "V47_PRECHECK_FAILED module_stats_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$HOLDOUT" ]; then echo "V47_PRECHECK_FAILED holdout_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$CAND" ]; then echo "V47_PRECHECK_FAILED candidate_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$A0" ]; then echo "V47_PRECHECK_FAILED a0_missing" | tee -a "$STATUS"; exit 2; fi
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PYCODE' > "$LOG" 2>&1
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('/sda/home/wangyuxin/ConvIR-B')
WORK = BASE / 'repos' / 'ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent'
ROUTE_ID = 'haze4k_v4_7_dcfsb_candidate_validation_20260708'
EVID = WORK / 'experience_docx' / 'experiment_logs' / ROUTE_ID
DATA = BASE / 'datasets' / 'Haze4K' / 'Haze4K'
A0 = BASE / 'checkpoints' / 'official' / 'Haze4K' / 'haze4k-base.pkl'
CAND = WORK / 'Dehazing' / 'ITS' / 'results' / 'ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708' / 'Training-Results' / 'Final.pkl'
SRC_EVID = WORK / 'experience_docx' / 'experiment_logs' / 'haze4k_v4_6_dcfsb_bottleneck_20260708_adapter4'
PER_IMAGE = SRC_EVID / 'v46_dcfsb_adapter4_per_image.csv'
MODULE_STATS = SRC_EVID / 'v46_dcfsb_adapter4_module_stats.jsonl'
HOLDOUT = WORK / 'docs' / 'ai_text_packages' / 'haze4k_v4_sfad' / 'splits' / 'haze4k_train_internal_holdout256.txt'
CARD = WORK / 'experience_docx' / 'experiment_cards' / '2026-07-08-haze4k-v4-7-dcfsb-candidate-validation.md'
README = EVID / 'README.md'

THRESHOLDS = {
    'mean_delta_psnr_min': 0.035,
    'positive_ratio_min': 0.58,
    'p5_delta_psnr_min': -0.25,
    'mean_delta_high_l1_max': 0.000005,
    'bootstrap_ci95_low_min_exclusive': 0.0,
    'sign_test_one_sided_p_max': 0.01,
    'worst32_max_proxy_quartile_ratio_soft_limit': 0.60,
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


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


def mean(values):
    return float(statistics.mean(float(v) for v in values))


def median(values):
    return float(statistics.median(float(v) for v in values))


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


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def texture_proxy(arr):
    dy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean() if arr.shape[0] > 1 else 0.0
    dx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean() if arr.shape[1] > 1 else 0.0
    return float(dx + dy)


def quantile_bins(rows, key):
    ordered = sorted((float(r[key]), idx) for idx, r in enumerate(rows))
    n = len(ordered)
    labels = [None] * n
    for rank, (_, idx) in enumerate(ordered):
        bin_idx = min(3, int(rank * 4 / n))
        labels[idx] = f'q{bin_idx + 1}'
    return labels


def summarize_subset(rows):
    deltas = [float(r['delta_psnr']) for r in rows]
    high = [float(r['delta_high_l1']) for r in rows]
    return {
        'count': len(rows),
        'mean_delta_psnr': mean(deltas),
        'median_delta_psnr': median(deltas),
        'p5_delta_psnr': pct(deltas, 5),
        'p95_delta_psnr': pct(deltas, 95),
        'positive_ratio': sum(d > 0 for d in deltas) / len(deltas),
        'mean_delta_high_l1': mean(high),
        'high_l1_improve_ratio': sum(d < 0 for d in high) / len(high),
    }


def one_sided_sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return float(tail / (2 ** n))

start = time.time()
EVID.mkdir(parents=True, exist_ok=True)
holdout_names = [line.strip() for line in HOLDOUT.read_text(encoding='utf-8').splitlines() if line.strip()]
with PER_IMAGE.open(newline='', encoding='utf-8') as f:
    source_rows = [dict(r) for r in csv.DictReader(f)]
rows = []
for r in source_rows:
    if r.get('split') != 'internal256':
        continue
    converted = dict(r)
    for key, value in list(converted.items()):
        if key not in {'split', 'image_id'}:
            converted[key] = float(value)
    rows.append(converted)
if len(rows) != 256:
    raise RuntimeError(f'expected 256 internal rows, got {len(rows)}')
if set(r['image_id'] for r in rows) != set(holdout_names):
    missing = sorted(set(holdout_names) - set(r['image_id'] for r in rows))[:10]
    extra = sorted(set(r['image_id'] for r in rows) - set(holdout_names))[:10]
    raise RuntimeError({'holdout_mismatch': True, 'missing': missing, 'extra': extra})

module_by_image = {}
with MODULE_STATS.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get('split') == 'internal256':
            module_by_image[item['image_id']] = item
if len(module_by_image) != 256:
    raise RuntimeError(f'expected 256 module rows, got {len(module_by_image)}')

for r in rows:
    name = r['image_id']
    hazy = load_rgb(DATA / 'train' / 'haze' / name)
    gt = load_rgb(label_path_for(name))
    if hazy.shape != gt.shape:
        raise RuntimeError(f'shape mismatch {name}: {hazy.shape} vs {gt.shape}')
    mod = module_by_image[name]
    r['input_gt_l1'] = float(np.abs(hazy - gt).mean())
    r['input_dark_channel_mean'] = float(np.min(hazy, axis=2).mean())
    r['input_brightness_mean'] = float(hazy.mean())
    r['input_saturation_proxy'] = float((hazy.max(axis=2) - hazy.min(axis=2)).mean())
    r['gt_texture_proxy'] = texture_proxy(gt)
    r['hazy_texture_proxy'] = texture_proxy(hazy)
    r['a0_error_proxy_low_plus_high_l1'] = float(r['a0_low_l1'] + r['a0_high_l1'])
    r['candidate_error_proxy_low_plus_high_l1'] = float(r['candidate_low_l1'] + r['candidate_high_l1'])
    for key in ['low_gate_std', 'high_gate_std', 'high_gate_mean', 'low_gate_mean', 'alpha_abs_mean', 'high_low_energy_ratio', 'high_energy', 'low_energy']:
        r[key] = float(mod[key])

metrics = summarize_subset(rows)
deltas = [float(r['delta_psnr']) for r in rows]
rng = random.Random(3407)
boot_n = 20000
boot_means = []
n = len(deltas)
for _ in range(boot_n):
    total = 0.0
    for _j in range(n):
        total += deltas[rng.randrange(n)]
    boot_means.append(total / n)
bootstrap = {
    'route_id': ROUTE_ID,
    'split': 'internal256',
    'bootstrap_samples': boot_n,
    'seed': 3407,
    'mean_delta_psnr': metrics['mean_delta_psnr'],
    'ci95_low': pct(boot_means, 2.5),
    'ci95_high': pct(boot_means, 97.5),
    'prob_mean_gt_0': sum(v > 0 for v in boot_means) / boot_n,
    'locked_test_touched': False,
    'test_split_enumerated': False,
}
wins = sum(d > 0 for d in deltas)
losses = sum(d < 0 for d in deltas)
ties = len(deltas) - wins - losses
sign = {
    'route_id': ROUTE_ID,
    'split': 'internal256',
    'wins': wins,
    'losses': losses,
    'ties': ties,
    'n_effective': wins + losses,
    'positive_ratio_all': wins / len(deltas),
    'one_sided_p_win_rate_gt_0_5': one_sided_sign_test(wins, losses),
    'locked_test_touched': False,
    'test_split_enumerated': False,
}

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
    for r, label in zip(rows, labels):
        r[f'{key}_quartile'] = label

worst32 = sorted(rows, key=lambda r: float(r['delta_psnr']))[:32]
worst_ids = {r['image_id'] for r in worst32}
bin_rows = []
systematic_flags = []
for key in proxy_keys:
    for label in ['q1', 'q2', 'q3', 'q4']:
        subset = [r for r in rows if r[f'{key}_quartile'] == label]
        summary = summarize_subset(subset)
        worst_count = sum(r['image_id'] in worst_ids for r in subset)
        values = [float(r[key]) for r in subset]
        out = {
            'proxy': key,
            'bin': label,
            'count': len(subset),
            'proxy_min': min(values),
            'proxy_max': max(values),
            'proxy_mean': mean(values),
            'mean_delta_psnr': summary['mean_delta_psnr'],
            'median_delta_psnr': summary['median_delta_psnr'],
            'p5_delta_psnr': summary['p5_delta_psnr'],
            'positive_ratio': summary['positive_ratio'],
            'mean_delta_high_l1': summary['mean_delta_high_l1'],
            'worst32_count': worst_count,
            'worst32_ratio': worst_count / 32,
        }
        bin_rows.append(out)
        if out['worst32_ratio'] >= 0.60 and out['mean_delta_psnr'] < -0.03 and out['positive_ratio'] < 0.50:
            systematic_flags.append({
                'proxy': key,
                'bin': label,
                'worst32_ratio': out['worst32_ratio'],
                'mean_delta_psnr': out['mean_delta_psnr'],
                'positive_ratio': out['positive_ratio'],
            })

compact_fields = [
    'image_id', 'delta_psnr', 'delta_ssim', 'delta_low_l1', 'delta_high_l1',
    'a0_psnr', 'candidate_psnr', 'input_gt_l1', 'a0_error_proxy_low_plus_high_l1',
    'input_dark_channel_mean', 'input_brightness_mean', 'input_saturation_proxy',
    'gt_texture_proxy', 'hazy_texture_proxy', 'high_gate_std', 'high_low_energy_ratio',
    'input_gt_l1_quartile', 'a0_error_proxy_low_plus_high_l1_quartile',
    'a0_psnr_quartile', 'input_dark_channel_mean_quartile', 'gt_texture_proxy_quartile',
]
for r in rows:
    r['worst32'] = r['image_id'] in worst_ids
compact_fields.append('worst32')

worst_fields = compact_fields + ['candidate_error_proxy_low_plus_high_l1', 'high_gate_mean', 'low_gate_mean', 'alpha_abs_mean']
with (EVID / 'v47_adapter4_per_image_compact.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=compact_fields)
    writer.writeheader()
    for r in sorted(rows, key=lambda x: x['image_id']):
        writer.writerow({k: r[k] for k in compact_fields})
with (EVID / 'v47_adapter4_worst32_compact.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=worst_fields)
    writer.writeheader()
    for r in worst32:
        writer.writerow({k: r[k] for k in worst_fields})
with (EVID / 'v47_adapter4_band_error_by_proxy_bins.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
    writer.writeheader()
    writer.writerows(bin_rows)

severity_counts = {
    'delta_lt_minus_0_50': sum(d < -0.50 for d in deltas),
    'delta_lt_minus_0_25': sum(d < -0.25 for d in deltas),
    'delta_lt_minus_0_10': sum(d < -0.10 for d in deltas),
}
all_gates = {
    'mean_delta_psnr': metrics['mean_delta_psnr'] >= THRESHOLDS['mean_delta_psnr_min'],
    'positive_ratio': metrics['positive_ratio'] >= THRESHOLDS['positive_ratio_min'],
    'p5_delta_psnr': metrics['p5_delta_psnr'] >= THRESHOLDS['p5_delta_psnr_min'],
    'mean_delta_high_l1': metrics['mean_delta_high_l1'] <= THRESHOLDS['mean_delta_high_l1_max'],
    'bootstrap_ci95_low': bootstrap['ci95_low'] > THRESHOLDS['bootstrap_ci95_low_min_exclusive'],
    'sign_test': sign['one_sided_p_win_rate_gt_0_5'] < THRESHOLDS['sign_test_one_sided_p_max'],
    'no_systematic_failure_bin': len(systematic_flags) == 0,
    'locked_test_not_touched': True,
    'test_split_not_enumerated': True,
}
pass_gate = all(all_gates.values())
lock = {
    'route_id': ROUTE_ID,
    'status': 'COMPLETED_GATE_PASS' if pass_gate else 'COMPLETED_GATE_FAIL',
    'route_identity': 'v4.6 adapter4 candidate-lock validation, train-derived audit only',
    'source_branch': 'codex/haze4k-v4-6-dcfsb-bottleneck-independent',
    'source_closeout_commit': '1277b61788fd2969e2bfdac9455a1a317db61f48',
    'validation_branch': 'codex/haze4k-v4-7-dcfsb-candidate-validation',
    'cloud_commit_at_run': os.popen(f'cd {WORK} && git rev-parse HEAD').read().strip(),
    'python': str(BASE / 'envs' / 'convir-cu121' / 'bin' / 'python'),
    'data_root': str(DATA),
    'a0_checkpoint': str(A0),
    'a0_sha256': sha256_file(A0),
    'candidate_checkpoint': str(CAND),
    'candidate_sha256': sha256_file(CAND),
    'split': 'haze4k_train_internal_holdout256.txt',
    'count': len(rows),
    'metrics': metrics,
    'bootstrap': bootstrap,
    'sign_test': sign,
    'severity_counts': severity_counts,
    'systematic_failure_flags': systematic_flags,
    'thresholds': THRESHOLDS,
    'gates': all_gates,
    'pass': pass_gate,
    'locked_test_touched': False,
    'test_split_enumerated': False,
    'runtime_sec': time.time() - start,
}
(EVID / 'v47_adapter4_internal256_bootstrap.json').write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(EVID / 'v47_adapter4_sign_test.json').write_text(json.dumps(sign, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(EVID / 'v47_candidate_lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n', encoding='utf-8')

worst_by_proxy_lines = []
for key in proxy_keys:
    counts = {label: sum(r[f'{key}_quartile'] == label for r in worst32) for label in ['q1', 'q2', 'q3', 'q4']}
    worst_by_proxy_lines.append((key, counts, max(counts.values()) / 32))
worst_table = '\n'.join(
    f"| {idx+1} | `{r['image_id']}` | {r['delta_psnr']:.6f} | {r['a0_psnr']:.6f} | {r['input_gt_l1']:.6f} | {r['a0_error_proxy_low_plus_high_l1']:.6f} | {r['input_dark_channel_mean']:.6f} | {r['gt_texture_proxy']:.6f} |"
    for idx, r in enumerate(worst32[:16])
)
proxy_table = '\n'.join(
    f"| `{key}` | {counts['q1']} | {counts['q2']} | {counts['q3']} | {counts['q4']} | {ratio:.3f} |"
    for key, counts, ratio in worst_by_proxy_lines
)
flags_text = 'None.' if not systematic_flags else '\n'.join(f"- `{f['proxy']}` {f['bin']}: worst32_ratio={f['worst32_ratio']:.3f}, mean_delta={f['mean_delta_psnr']:.6f}, positive_ratio={f['positive_ratio']:.3f}" for f in systematic_flags)
atlas = f"""# Haze4K v4.7 Adapter4 Failure Atlas

Route id: `{ROUTE_ID}`

Split: `internal_holdout256` train-derived holdout only. Locked test touched/enumerated: `false` / `false`.

## Gate Summary

- mean dPSNR: `{metrics['mean_delta_psnr']:.6f}`
- positive ratio: `{metrics['positive_ratio']:.6f}`
- p5 dPSNR: `{metrics['p5_delta_psnr']:.6f}`
- mean dHighL1: `{metrics['mean_delta_high_l1']:.10f}`
- bootstrap 95% CI: `[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]`
- sign-test one-sided p: `{sign['one_sided_p_win_rate_gt_0_5']:.8g}`
- severe counts: `{severity_counts}`
- systematic failure flags: {len(systematic_flags)}

## Worst32 Proxy Concentration

| Proxy | q1 | q2 | q3 | q4 | max ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
{proxy_table}

Systematic flags:

{flags_text}

## Worst 16 Images

| Rank | Image | dPSNR | A0 PSNR | input-GT L1 | A0 proxy L1 | dark channel | GT texture |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{worst_table}

## Interpretation

This atlas is a compact text audit, not a visual inspection set. It checks whether adapter4's worst internal_holdout regressions concentrate in a single proxy quartile strongly enough to suggest a systematic failure mode. The gate treats a bin as systematic only when worst32 concentration is high and the whole bin has negative mean movement and low positive ratio.
"""
(EVID / 'v47_adapter4_failure_atlas.md').write_text(atlas, encoding='utf-8')

decision = 'PASS' if pass_gate else 'FAIL'
next_action = (
    'Adapter4 is eligible for a separate written confirmation gate for exactly one fixed-checkpoint locked-test command. Do not run locked test from this v4.7 script.'
    if pass_gate else
    'Locked test remains blocked. Next allowed work is K-fold train-derived validation or a tail-safe adapter variant, not locked-test access.'
)
decision_md = f"""# Decision After Haze4K v4.7

Status: `{decision}`

Route id: `{ROUTE_ID}`

Candidate: fixed v4.6 DCFSB-bottleneck `adapter4` checkpoint.

Locked test: not touched; not enumerated.

## Metrics

| Metric | Value | Gate |
| --- | ---: | --- |
| mean dPSNR | `{metrics['mean_delta_psnr']:.6f}` | `>= +0.035` |
| positive ratio | `{metrics['positive_ratio']:.6f}` | `>= 0.58` |
| p5 dPSNR | `{metrics['p5_delta_psnr']:.6f}` | `>= -0.25` |
| mean dHighL1 | `{metrics['mean_delta_high_l1']:.10f}` | `<= +0.000005` |
| bootstrap CI low | `{bootstrap['ci95_low']:.6f}` | `> 0` |
| sign-test p | `{sign['one_sided_p_win_rate_gt_0_5']:.8g}` | `< 0.01` |
| systematic failure flags | `{len(systematic_flags)}` | `0` |

## Gate Results

```json
{json.dumps(all_gates, indent=2, sort_keys=True)}
```

## Decision

{next_action}
"""
(EVID / 'decision_after_v47.md').write_text(decision_md, encoding='utf-8')
readme = f"""# Haze4K v4.7 DCFSB Candidate-Lock Validation Evidence

Route id: `{ROUTE_ID}`

Branch: `codex/haze4k-v4-7-dcfsb-candidate-validation`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Policy: fixed v4.6 `adapter4` on train-derived `internal_holdout256` only. Locked Haze4K test was not touched or enumerated.

## Result

Status: `{lock['status']}`

- mean dPSNR `{metrics['mean_delta_psnr']:.6f}`
- positive ratio `{metrics['positive_ratio']:.6f}`
- p5 dPSNR `{metrics['p5_delta_psnr']:.6f}`
- mean dHighL1 `{metrics['mean_delta_high_l1']:.10f}`
- bootstrap 95% CI `[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]`
- sign-test one-sided p `{sign['one_sided_p_win_rate_gt_0_5']:.8g}`
- systematic failure flags `{len(systematic_flags)}`

## Primary Artifacts

- `v47_candidate_lock.json`
- `v47_adapter4_internal256_bootstrap.json`
- `v47_adapter4_sign_test.json`
- `v47_adapter4_failure_atlas.md`
- `v47_adapter4_band_error_by_proxy_bins.csv`
- `v47_adapter4_worst32_compact.csv`
- `v47_adapter4_per_image_compact.csv`
- `decision_after_v47.md`
- `run_v47_adapter4_candidate_lock_validation.sh`
- `run_v47_adapter4_candidate_lock_validation.log`
"""
README.write_text(readme, encoding='utf-8')
print(json.dumps({'candidate_lock': lock, 'outputs': sorted(p.name for p in EVID.iterdir())}, indent=2, sort_keys=True))
PYCODE
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "state=COMPLETED_GATE_WRITTEN" | tee -a "$STATUS"
  echo "run_done rc=0 v47_adapter4_candidate_lock $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V47_CANDIDATE_LOCK_OK
else
  echo "state=FAILED_COMMAND" | tee -a "$STATUS"
  echo "run_done rc=$rc v47_adapter4_candidate_lock $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V47_CANDIDATE_LOCK_FAILED
fi
exit "$rc"
