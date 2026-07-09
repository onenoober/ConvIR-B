#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v2h_a_d7c_risk_coverage.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "v2h_a_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
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
DATA=BASE/'datasets/Haze4K/Haze4K'
A0=BASE/'checkpoints/official/Haze4K/haze4k-base.pkl'
SPLIT=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json'
V2_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json'
V2B_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json'
D3=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt'
D7C=BASE/'repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt'
V2F=ROOT/'experience_docx/tools/run_chd_rm_v2f_need_target_head_redesign.py'
FIXED_D7C_THRESHOLD=0.5773006677627563
COVERAGES=[0.15,0.20,0.25,0.30,0.35,0.40,0.45]

spec=importlib.util.spec_from_file_location('chdrm_v2f_tool', V2F)
v2f=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2f)
v2e=v2f.v2e; d7c=v2e.d7c; v2d=v2e.v2d; v2b=v2d.v2b

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames=[]
        for r in rows:
            for k in r:
                if k not in fieldnames: fieldnames.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')

def safe_div(a,b): return float(a)/float(b) if b else math.nan

def qstats(vals):
    vals=sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals: return {'n':0,'mean':math.nan,'p50':math.nan,'p75':math.nan,'p90':math.nan,'p95':math.nan,'max':math.nan}
    def q(p): return vals[min(len(vals)-1,max(0,int(round((len(vals)-1)*p))))]
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'max':vals[-1]}

def auroc(y, s): return float(v2b.auroc(np.asarray(y).astype(bool), np.asarray(s).astype(np.float32)))
def auprc(y, s): return float(v2b.auprc(np.asarray(y).astype(bool), np.asarray(s).astype(np.float32)))

def matched_threshold(score, coverage):
    return float(np.quantile(score.astype(np.float64), max(0.0, min(1.0, 1.0-float(coverage)))))

def metric_row(arr, score_key, threshold, score_name, kind, split_name, target_coverage=None, op='coverage'):
    pred=arr[score_key] >= threshold
    action=arr['action']; neg=arr['negative']; ignore=arr['ignore']; iso=arr['isolated']; lowadj=arr['low_adjacent']
    row={
        'score':score_name,'kind':kind,'split':split_name,'operating_point':op,'target_train_calib_coverage':target_coverage,
        'threshold':threshold,'selected_coverage':float(pred.mean()),
        'action_positive_coverage':float(action.mean()),'negative_low_risk_coverage':float(neg.mean()),'ignore_coverage':float(ignore.mean()),
        'low_adjacent_coverage':float(lowadj.mean()),'isolated_ldhn_coverage':float(iso.mean()),
        'action_recall':safe_div((pred & action).sum(), action.sum()),
        'low_adjacent_recall':safe_div((pred & lowadj).sum(), lowadj.sum()),
        'negative_false_rate':safe_div((pred & neg).sum(), neg.sum()),
        'ignore_hit_rate':safe_div((pred & ignore).sum(), ignore.sum()),
        'isolated_ldhn_hit_rate':safe_div((pred & iso).sum(), iso.sum()),
        'action_precision_vs_all_selected':safe_div((pred & action).sum(), pred.sum()),
    }
    if int(action.sum()) and int(neg.sum()):
        y=np.concatenate([np.ones(int(action.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)])
        s=np.concatenate([arr[score_key][action], arr[score_key][neg]])
        row['auroc_action_vs_negative']=auroc(y,s); row['auprc_action_vs_negative']=auprc(y,s)
    if int(lowadj.sum()) and int(neg.sum()):
        y=np.concatenate([np.ones(int(lowadj.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)])
        s=np.concatenate([arr[score_key][lowadj], arr[score_key][neg]])
        row['auroc_lowadjacent_vs_negative']=auroc(y,s); row['auprc_lowadjacent_vs_negative']=auprc(y,s)
    return row

def per_image_tail(arr, score_key, threshold, score_name, kind, split_name, target_coverage=None, op='coverage'):
    pred=arr[score_key] >= threshold
    pix_per_img=4096
    out={'score':score_name,'kind':kind,'split':split_name,'operating_point':op,'target_train_calib_coverage':target_coverage,'threshold':threshold}
    for label,mask in [('action_recall',arr['action']),('low_adjacent_recall',arr['low_adjacent']),('negative_false_rate',arr['negative']),('ignore_hit_rate',arr['ignore']),('isolated_ldhn_hit_rate',arr['isolated'])]:
        vals=[]
        n=len(mask)//pix_per_img
        for i in range(n):
            sl=slice(i*pix_per_img,(i+1)*pix_per_img)
            denom=mask[sl].sum()
            if denom:
                vals.append(safe_div((pred[sl] & mask[sl]).sum(), denom))
        st=qstats(vals)
        for k,v in st.items(): out[f'{label}_{k}']=v
    return out

print('V2H_A_PY_START')
for p in [DATA/'train'/'haze', DATA/'train'/'gt', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C, V2F]:
    if not Path(p).exists(): raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp: raise RuntimeError(f'forbidden path: {p}')

split=json.loads(SPLIT.read_text(encoding='utf-8'))
train_names=sorted(split['splits']['train_inner']); val_names=sorted(split['splits']['val_inner'])
rng=np.random.default_rng(3407)
train_calib_names=sorted(rng.choice(train_names, size=min(360,len(train_names)), replace=False).tolist())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, 'train_calib', len(train_calib_names), 'val_inner', len(val_names), flush=True)
model=v2b.load_model(A0, device); model.eval()
density_head=v2d.load_density_head(D3, device); density_head.eval()
topk_head=v2e.load_head(D7C, device); topk_head.eval()
for m in [model,density_head,topk_head]:
    for p in m.parameters(): p.requires_grad_(False)

density_stats=json.loads(V2_THRESH.read_text(encoding='utf-8'))
target_info=json.loads(V2B_THRESH.read_text(encoding='utf-8'))
q33=float(target_info['quantile']['q33']); q66=float(target_info['quantile']['q66']); q80=float(target_info['quantile']['q80'])
density_q33=float(density_stats['density']['q33']); density_q66=float(density_stats['density']['q66'])
f1=json.loads((ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/ldhn_target_autopsy_summary.json').read_text())
grad_p90=float(f1['density_gradient_p90'])
map_grid=64; blur_kernel=9; blur_radii=[5,9,15]; near_haze_radius=3

def collect(names, split_name):
    ds=v2e.Haze4KPairDataset(names, DATA, max_items=0, seed=3407)
    parts={k:[] for k in ['action','low_adjacent','negative','ignore','isolated','d7c','density_pred']}
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
            parts['action'].append(action.reshape(-1)); parts['low_adjacent'].append(adjacent.reshape(-1)); parts['negative'].append(negative.reshape(-1)); parts['ignore'].append(ignore.reshape(-1)); parts['isolated'].append(isolated.reshape(-1)); parts['d7c'].append(pred_map.reshape(-1)); parts['density_pred'].append(density_pred_map.reshape(-1))
            if (idx+1)%100==0: print(f'collect {split_name} {idx+1}/{len(ds)} elapsed={time.time()-start:.1f}s', flush=True)
    return {k:np.concatenate(v).astype(np.float32 if k in ['d7c','density_pred'] else bool) for k,v in parts.items()}

train=collect(train_calib_names,'train_calib')
val=collect(val_names,'val_inner')
curve=[]; tail=[]; density_curve=[]
for cov in COVERAGES:
    d7_thr=matched_threshold(train['d7c'], cov)
    den_thr=matched_threshold(train['density_pred'], cov)
    curve.append(metric_row(val,'d7c',d7_thr,'d7c_topk_score','deployable_prior_candidate','val_inner',cov,'coverage'))
    tail.append(per_image_tail(val,'d7c',d7_thr,'d7c_topk_score','deployable_prior_candidate','val_inner',cov,'coverage'))
    density_curve.append(metric_row(val,'density_pred',den_thr,'d3_density_pred_matched','deployable_density_control','val_inner',cov,'coverage'))
    tail.append(per_image_tail(val,'density_pred',den_thr,'d3_density_pred_matched','deployable_density_control','val_inner',cov,'coverage'))
fixed_cov=float((train['d7c']>=FIXED_D7C_THRESHOLD).mean())
fixed=metric_row(val,'d7c',FIXED_D7C_THRESHOLD,'d7c_topk_score','deployable_prior_candidate','val_inner',fixed_cov,'fixed_v2g_d7c')
fixed_tail=per_image_tail(val,'d7c',FIXED_D7C_THRESHOLD,'d7c_topk_score','deployable_prior_candidate','val_inner',fixed_cov,'fixed_v2g_d7c')
curve.append(fixed); tail.append(fixed_tail)
den_fixed_thr=matched_threshold(train['density_pred'], fixed_cov)
den_fixed=metric_row(val,'density_pred',den_fixed_thr,'d3_density_pred_matched','deployable_density_control','val_inner',fixed_cov,'matched_fixed_v2g_d7c')
density_curve.append(den_fixed); tail.append(per_image_tail(val,'density_pred',den_fixed_thr,'d3_density_pred_matched','deployable_density_control','val_inner',fixed_cov,'matched_fixed_v2g_d7c'))
write_csv(EVID/'d7c_risk_coverage_curve.csv', curve)
write_csv(EVID/'density_only_matched_control_curve.csv', density_curve)
write_csv(EVID/'d7c_per_image_tail_metrics.csv', tail)
bins=[]
score=val['d7c']; qs=np.quantile(score.astype(np.float64), np.linspace(0,1,11))
for i in range(10):
    lo,hi=qs[i],qs[i+1]
    mask=(score>=lo) & (score<=hi if i==9 else score<hi)
    row={'bin':i,'score_lo':float(lo),'score_hi':float(hi),'pixels':int(mask.sum()),'score_mean':float(score[mask].mean()) if mask.any() else math.nan}
    for key in ['action','low_adjacent','negative','ignore','isolated']:
        row[f'{key}_rate']=safe_div((mask & val[key]).sum(), mask.sum())
    bins.append(row)
write_csv(EVID/'d7c_calibration_reliability_bins.csv', bins)
def fget(row,key): return float(row.get(key, math.nan))
passed=(0.25 <= fget(fixed,'selected_coverage') <= 0.35 and fget(fixed,'low_adjacent_recall') >= 0.15 and fget(fixed,'negative_false_rate') <= 0.005 and fget(fixed,'isolated_ldhn_hit_rate') <= 0.03 and float(fixed_tail['negative_false_rate_p95']) <= 0.05 and fget(fixed,'action_recall') >= 0.5383118150551728 and fget(fixed,'action_recall') > fget(den_fixed,'action_recall') and fget(fixed,'negative_false_rate') < fget(den_fixed,'negative_false_rate'))
closeout={
  'status':'COMPLETED_GATE_PASS' if passed else 'COMPLETED_GATE_FAIL',
  'decision_label':'V2H_A_D7C_RISK_COVERAGE_PASS_AUTHORIZE_SHADOW' if passed else 'V2H_A_D7C_RISK_COVERAGE_FAIL_NO_SHADOW',
  'locked_haze4k_test_usage':'none','D2':'not_run','F5':'not_run','v3':'not_run','RARM':'not_connected_or_trained',
  'train_calib_images':len(train_calib_names),'val_inner_images':len(val_names),'fixed_train_calib_coverage':fixed_cov,
  'primary_operating_point':fixed,'primary_per_image_tail':fixed_tail,'density_matched_at_primary':den_fixed,
  'gate_pass':passed,
  'next_authorized_stage':'v2h-B shadow-modulation only' if passed else 'pause; no v2h-B, no v3, no RARM'
}
write_json(EVID/'v2h_a_risk_coverage_closeout.json', closeout)
md=['# v2h-A D7c Risk-Coverage Calibration','',f"Status: `{closeout['status']}`",'',f"Decision label: `{closeout['decision_label']}`",'', 'Policy: train-calib selects thresholds; val-inner is report-only. No locked test, D2, F5, v3, RARM, adapter training, or new probe training was run.','', '## Primary Operating Point','', '| Score | Coverage | Action recall | Low-adj recall | Negative false | Negative false p95 | Isolated hit | Density action recall | Density negative false |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |', f"| D7c fixed | {fget(fixed,'selected_coverage'):.6f} | {fget(fixed,'action_recall'):.6f} | {fget(fixed,'low_adjacent_recall'):.6f} | {fget(fixed,'negative_false_rate'):.6f} | {float(fixed_tail['negative_false_rate_p95']):.6f} | {fget(fixed,'isolated_ldhn_hit_rate'):.6f} | {fget(den_fixed,'action_recall'):.6f} | {fget(den_fixed,'negative_false_rate'):.6f} |", '', '## Decision','', closeout['next_authorized_stage'], '']
(EVID/'d7c_risk_coverage_summary.md').write_text('\n'.join(md), encoding='utf-8')
readme=EVID/'README.md'
text=readme.read_text(encoding='utf-8').replace('Status: `PLANNED`', f"Status: `{closeout['status']}`")
text += '\n## v2h-A Result\n\n' + '\n'.join(md) + '\n'
readme.write_text(text, encoding='utf-8')
print('V2H_A_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "v2h_a_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2H_A_RISK_COVERAGE_OK | tee -a "$STATUS"; else echo V2H_A_RISK_COVERAGE_FAILED | tee -a "$STATUS"; fi
exit "$rc"
