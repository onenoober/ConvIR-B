#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v2h_c_oof_stability.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "v2h_c_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import torch

BASE=Path('/sda/home/wangyuxin/ConvIR-B')
ROOT=BASE/'repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency'
EVID=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709'
V1=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708'
DATA=BASE/'datasets/Haze4K/Haze4K'
A0=BASE/'checkpoints/official/Haze4K/haze4k-base.pkl'
V2_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json'
V2B_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json'
D3=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt'
D7C=BASE/'repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt'
V2F=ROOT/'experience_docx/tools/run_chd_rm_v2f_need_target_head_redesign.py'
A_CLOSE=EVID/'v2h_a_risk_coverage_closeout.json'
B_CLOSE=EVID/'shadow_modulation_closeout.json'
FOLDS=V1/'haze4k_oof_folds.csv'
MANIFEST=V1/'haze4k_manifest_train.csv'

spec=importlib.util.spec_from_file_location('chdrm_v2f_tool', V2F)
v2f=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2f)
v2e=v2f.v2e; d7c=v2e.d7c; v2d=v2e.v2d; v2b=v2d.v2b

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames=[]
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def safe_div(a,b):
    return float(a)/float(b) if b else math.nan

def matched_threshold(score, coverage):
    return float(np.quantile(score.astype(np.float64), max(0.0, min(1.0, 1.0-float(coverage)))))

def metric_row(arr, score_key, threshold):
    pred=arr[score_key] >= threshold
    action=arr['action']; lowadj=arr['low_adjacent']; neg=arr['negative']; ignore=arr['ignore']; iso=arr['isolated']
    return {
        'threshold':float(threshold),
        'selected_coverage':float(pred.mean()),
        'action_recall':safe_div((pred & action).sum(), action.sum()),
        'low_adjacent_recall':safe_div((pred & lowadj).sum(), lowadj.sum()),
        'negative_false_rate':safe_div((pred & neg).sum(), neg.sum()),
        'ignore_hit_rate':safe_div((pred & ignore).sum(), ignore.sum()),
        'isolated_ldhn_hit_rate':safe_div((pred & iso).sum(), iso.sum()),
        'action_precision_vs_all_selected':safe_div((pred & action).sum(), pred.sum()),
        'action_pixels':int(action.sum()),
        'low_adjacent_pixels':int(lowadj.sum()),
        'negative_pixels':int(neg.sum()),
        'ignore_pixels':int(ignore.sum()),
        'isolated_pixels':int(iso.sum()),
    }

def qstats(vals):
    vals=[float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {'mean':math.nan,'std':math.nan,'min':math.nan,'max':math.nan}
    return {'mean':float(np.mean(vals)),'std':float(np.std(vals)),'min':float(np.min(vals)),'max':float(np.max(vals))}

def aggregate(records, names):
    out={}
    for key in ['action','low_adjacent','negative','ignore','isolated','d7c','density_pred']:
        out[key]=np.concatenate([records[name][key] for name in names])
    return out

print('V2H_C_PY_START')
for path in [DATA/'train'/'haze', DATA/'train'/'gt', A0, V2_THRESH, V2B_THRESH, D3, D7C, V2F, A_CLOSE, B_CLOSE, FOLDS, MANIFEST]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if '/test/' in str(path).lower() or 'locked' in str(path).lower():
        raise RuntimeError(f'forbidden path: {path}')
if not json.loads(A_CLOSE.read_text(encoding='utf-8')).get('gate_pass'):
    raise RuntimeError('v2h-A did not pass')
if not json.loads(B_CLOSE.read_text(encoding='utf-8')).get('gate_pass'):
    raise RuntimeError('v2h-B did not pass')

with MANIFEST.open(newline='', encoding='utf-8') as handle:
    manifest={row['image_id']:row['hazy_name'] for row in csv.DictReader(handle)}
fold_val={i:[] for i in range(5)}
with FOLDS.open(newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        if row['fold_role'] == 'val':
            fold_val[int(row['fold_id'])].append(manifest[row['image_id']])
all_names=sorted(manifest.values())
target_coverage=float(json.loads(A_CLOSE.read_text(encoding='utf-8'))['fixed_train_calib_coverage'])
fixed_threshold=float(json.loads(A_CLOSE.read_text(encoding='utf-8'))['primary_operating_point']['threshold'])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, 'images', len(all_names), 'target_coverage', target_coverage, 'fixed_threshold', fixed_threshold, flush=True)
model=v2b.load_model(A0, device); model.eval()
density_head=v2d.load_density_head(D3, device); density_head.eval()
topk_head=v2e.load_head(D7C, device); topk_head.eval()
for module in [model,density_head,topk_head]:
    for param in module.parameters():
        param.requires_grad_(False)

density_stats=json.loads(V2_THRESH.read_text(encoding='utf-8'))
target_info=json.loads(V2B_THRESH.read_text(encoding='utf-8'))
q33=float(target_info['quantile']['q33']); q66=float(target_info['quantile']['q66']); q80=float(target_info['quantile']['q80'])
density_q33=float(density_stats['density']['q33']); density_q66=float(density_stats['density']['q66'])
f1=json.loads((ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/ldhn_target_autopsy_summary.json').read_text())
grad_p90=float(f1['density_gradient_p90'])
map_grid=64; blur_kernel=9; blur_radii=[5,9,15]; near_haze_radius=3

records={}
ds=v2e.Haze4KPairDataset(all_names, DATA, max_items=0, seed=3407)
start=time.time()
with torch.no_grad():
    for idx,(name,hazy,gt) in enumerate(ds):
        hazy=hazy.unsqueeze(0).to(device); gt=gt.unsqueeze(0).to(device)
        padded,h,w=v2b.v2.pad32(hazy)
        a0,context=d7c.convir_a0_context(model,density_head,padded)
        a0=a0[:,:,:h,:w]; context=context[:,:,:h,:w]
        pred,_=d7c.predict_head(topk_head, context)
        raw_need=v2b.v2.raw_need(a0, gt, blur_kernel)
        target=v2b.make_target(raw_need, target_info, 'quantile')
        density=v2b.v2.normalize(v2b.v2.raw_density(hazy, gt, blur_kernel), density_stats['density']['raw_p1'], density_stats['density']['raw_p99'])
        target_map=v2f.pool_map(target,map_grid); density_map=v2f.pool_map(density,map_grid); pred_map=v2f.pool_map(pred,map_grid); density_pred_map=v2f.pool_map(context[:,-1:],map_grid)
        blur_maps=[]
        for radius in blur_radii:
            rn=v2b.v2.raw_need(a0, gt, radius); tb=v2b.make_target(rn, target_info, 'quantile')
            blur_maps.append(v2f.pool_map(tb,map_grid))
        stable66=np.logical_and.reduce([tb>=q66 for tb in blur_maps])
        density_low=density_map<=density_q33
        high_need=target_map>=q66
        low_need=target_map<=q33
        low_mid=density_low & (target_map>q33) & (target_map<q66)
        local_high_density=v2f.local_max_2d((density_map>=density_q66).astype(np.float32), near_haze_radius)>0
        grad=v2f.gradient_mag(density_map)
        adjacent=(density_low & high_need) & (local_high_density | (grad>=grad_p90))
        isolated=(density_low & high_need) & (~adjacent)
        unstable=(density_low & high_need) & (~stable66)
        boundary_band=density_low & (target_map>=q66) & (target_map<q80)
        boundary=(density_low & high_need) & (boundary_band | unstable)
        action=(high_need & (~density_low)) | adjacent
        negative=density_low & low_need
        ignore=isolated | low_mid | boundary
        records[name]={
            'action':action.reshape(-1).astype(bool),
            'low_adjacent':adjacent.reshape(-1).astype(bool),
            'negative':negative.reshape(-1).astype(bool),
            'ignore':ignore.reshape(-1).astype(bool),
            'isolated':isolated.reshape(-1).astype(bool),
            'd7c':pred_map.reshape(-1).astype(np.float32),
            'density_pred':density_pred_map.reshape(-1).astype(np.float32),
        }
        if (idx+1)%250==0:
            print(f'oof_collect {idx+1}/{len(ds)} elapsed={time.time()-start:.1f}s', flush=True)

fold_rows=[]
for fold_id in range(5):
    val_names=sorted(fold_val[fold_id])
    train_names=sorted(set(all_names)-set(val_names))
    train=aggregate(records, train_names)
    val=aggregate(records, val_names)
    d7_thr=matched_threshold(train['d7c'], target_coverage)
    den_thr=matched_threshold(train['density_pred'], target_coverage)
    for selector,score_key,thr in [('d7c_calibrated','d7c',d7_thr),('density_matched','density_pred',den_thr),('d7c_fixed_A','d7c',fixed_threshold)]:
        row={'fold_id':fold_id,'selector':selector,'train_images':len(train_names),'val_images':len(val_names),'target_train_coverage':target_coverage}
        row.update(metric_row(val, score_key, thr))
        fold_rows.append(row)
write_csv(EVID/'oof_stability_by_fold.csv', fold_rows)

d7_rows=[r for r in fold_rows if r['selector']=='d7c_calibrated']
den_rows=[r for r in fold_rows if r['selector']=='density_matched']
fixed_rows=[r for r in fold_rows if r['selector']=='d7c_fixed_A']
summary={}
for selector,rows in [('d7c_calibrated',d7_rows),('density_matched',den_rows),('d7c_fixed_A',fixed_rows)]:
    for key in ['threshold','selected_coverage','action_recall','low_adjacent_recall','negative_false_rate','isolated_ldhn_hit_rate']:
        st=qstats([r[key] for r in rows])
        for sk,sv in st.items():
            summary[f'{selector}_{key}_{sk}']=sv
gate_pass=(
    summary['d7c_calibrated_selected_coverage_std'] <= 0.035 and
    summary['d7c_calibrated_action_recall_mean'] >= 0.50 and
    summary['d7c_calibrated_action_recall_min'] >= 0.45 and
    summary['d7c_calibrated_low_adjacent_recall_mean'] >= 0.10 and
    summary['d7c_calibrated_negative_false_rate_max'] <= 0.01 and
    summary['d7c_calibrated_isolated_ldhn_hit_rate_max'] <= 0.08 and
    summary['d7c_calibrated_action_recall_mean'] > summary['density_matched_action_recall_mean'] and
    summary['d7c_calibrated_negative_false_rate_mean'] < summary['density_matched_negative_false_rate_mean']
)
closeout={
    'status':'COMPLETED_GATE_PASS' if gate_pass else 'COMPLETED_GATE_FAIL',
    'decision_label':'V2H_C_OOF_STABILITY_PASS_AUTHORIZE_FAM2_NOOP_REVIEW' if gate_pass else 'V2H_C_OOF_STABILITY_FAIL_PAUSE',
    'locked_haze4k_test_usage':'none',
    'D2':'not_run','F5':'not_run','v3':'not_run','RARM':'not_connected_or_trained',
    'scope':'no training; fold-wise threshold calibration over v1 fixed five-fold train OOF table',
    'target_train_coverage':target_coverage,
    'gate_pass':gate_pass,
    'summary':summary,
    'next_authorized_stage':'v2h-D FAM2 no-op equivalence review only' if gate_pass else 'pause; no FAM2/no-op/RARM/training expansion',
}
write_json(EVID/'oof_stability_closeout.json', closeout)
md=[
    '# v2h-C OOF Stability Audit',
    '',
    f"Status: `{closeout['status']}`",
    '',
    f"Decision label: `{closeout['decision_label']}`",
    '',
    'Policy: no training, no locked Haze4K test, no D2/F5/v3/RARM/adapter training/canary expansion.',
    '',
    '## Fold Summary',
    '',
    '| Selector | Action recall mean/min | Low-adj mean | Negative false mean/max | Coverage std |',
    '| --- | ---: | ---: | ---: | ---: |',
    f"| D7c calibrated | {summary['d7c_calibrated_action_recall_mean']:.6f}/{summary['d7c_calibrated_action_recall_min']:.6f} | {summary['d7c_calibrated_low_adjacent_recall_mean']:.6f} | {summary['d7c_calibrated_negative_false_rate_mean']:.6f}/{summary['d7c_calibrated_negative_false_rate_max']:.6f} | {summary['d7c_calibrated_selected_coverage_std']:.6f} |",
    f"| Density matched | {summary['density_matched_action_recall_mean']:.6f}/{summary['density_matched_action_recall_min']:.6f} | {summary['density_matched_low_adjacent_recall_mean']:.6f} | {summary['density_matched_negative_false_rate_mean']:.6f}/{summary['density_matched_negative_false_rate_max']:.6f} | {summary['density_matched_selected_coverage_std']:.6f} |",
    '',
    '## Decision',
    '',
    closeout['next_authorized_stage'],
    '',
]
(EVID/'oof_stability_summary.md').write_text('\n'.join(md), encoding='utf-8')
readme=EVID/'README.md'
readme.write_text(readme.read_text(encoding='utf-8').rstrip() + '\n\n## v2h-C Result\n\n' + '\n'.join(md) + '\n', encoding='utf-8')
print('V2H_C_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "v2h_c_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2H_C_OOF_STABILITY_OK | tee -a "$STATUS"; else echo V2H_C_OOF_STABILITY_FAILED | tee -a "$STATUS"; fi
exit "$rc"
