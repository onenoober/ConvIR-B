#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B
WS=$ROOT/repos/ConvIR-B-haze4k-v5-v1-chd-rm-data-baseline-lock
PY=$ROOT/envs/convir-cu121/bin/python
DATA=$ROOT/datasets/Haze4K/Haze4K
CKPT=$ROOT/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$WS/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708
SPLIT_JSON=$EVID/haze4k_internal_split_2400_600.json
OUT=$EVID/a0_val600_eval
LOG=$EVID/v1_a0_val600_eval.log
STATUS=$EVID/status.txt
mkdir -p "$OUT"
{
  echo "v1_a0_val600_eval_start $(date --iso-8601=seconds)"
  echo "workspace=$WS"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "split_json=$SPLIT_JSON"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  "$PY" "$WS/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA" \
    --original_checkpoint "$CKPT" \
    --original_arch official_convir \
    --original_mode original \
    --original_name A0 \
    --candidate_checkpoint "$CKPT" \
    --candidate_arch official_convir \
    --candidate_mode original \
    --candidate_name A0_repeat \
    --split_json "$SPLIT_JSON" \
    --split_name val_inner \
    --output_dir "$OUT" \
    --tag chdrm_v1_a0_val600_repeat
  "$PY" - <<'PY'
import csv, json, statistics
from pathlib import Path
E = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v1-chd-rm-data-baseline-lock/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708')
OUT = E / 'a0_val600_eval'
compare_json = OUT / 'scout_eval_compare_chdrm_v1_a0_val600_repeat.json'
per_csv = OUT / 'scout_eval_per_image_chdrm_v1_a0_val600_repeat.csv'
data = json.loads(compare_json.read_text())
rows = []
with per_csv.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rows.append(row)
with (E / 'a0_val600_per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    fieldnames = ['name','a0_psnr','a0_ssim','a0_repeat_psnr','a0_repeat_ssim','delta_psnr','delta_ssim','a0_time_sec','a0_repeat_time_sec']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({
            'name': r['name'],
            'a0_psnr': r['original_psnr'],
            'a0_ssim': r['original_ssim'],
            'a0_repeat_psnr': r['A0_repeat_psnr'],
            'a0_repeat_ssim': r['A0_repeat_ssim'],
            'delta_psnr': r['delta_psnr'],
            'delta_ssim': r['delta_ssim'],
            'a0_time_sec': r['original_time_sec'],
            'a0_repeat_time_sec': r['A0_repeat_time_sec'],
        })
psnr = [float(r['original_psnr']) for r in rows]
ssim = [float(r['original_ssim']) for r in rows]
dpsnr = [float(r['delta_psnr']) for r in rows]
dssim = [float(r['delta_ssim']) for r in rows]
runs = data['runs']
global_row = {
    'split': 'val_inner',
    'count': len(rows),
    'mean_psnr': statistics.mean(psnr),
    'mean_ssim': statistics.mean(ssim),
    'repeat_mean_delta_psnr': statistics.mean(dpsnr),
    'repeat_max_abs_delta_psnr': max(abs(x) for x in dpsnr),
    'repeat_mean_delta_ssim': statistics.mean(dssim),
    'repeat_max_abs_delta_ssim': max(abs(x) for x in dssim),
}
with (E / 'a0_val600_global_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(global_row))
    w.writeheader(); w.writerow(global_row)
metric_audit = {
    'count': len(rows),
    'repeat_mean_delta_psnr': global_row['repeat_mean_delta_psnr'],
    'repeat_max_abs_delta_psnr': global_row['repeat_max_abs_delta_psnr'],
    'repeat_mean_delta_ssim': global_row['repeat_mean_delta_ssim'],
    'repeat_max_abs_delta_ssim': global_row['repeat_max_abs_delta_ssim'],
    'pass': global_row['repeat_max_abs_delta_psnr'] <= 1e-8 and global_row['repeat_max_abs_delta_ssim'] <= 1e-10,
}
eff = {
    'A0': runs['A0'],
    'A0_repeat': runs['A0_repeat'],
}
(E / 'metric_repro_audit.json').write_text(json.dumps(metric_audit, indent=2), encoding='utf-8')
(E / 'a0_efficiency_metrics.json').write_text(json.dumps(eff, indent=2), encoding='utf-8')
print(json.dumps({'global': global_row, 'metric_audit': metric_audit}, indent=2))
PY
  echo "v1_a0_val600_eval_done $(date --iso-8601=seconds)"
  echo "CHDRM_V1_A0_VAL600_OK"
} 2>&1 | tee "$LOG"
echo "CHDRM_V1_A0_VAL600_OK $(date --iso-8601=seconds)" >> "$STATUS"
