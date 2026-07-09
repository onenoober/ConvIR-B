#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign-f4-044b7798
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
V2_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json
V2B_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json
D3=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt
D7C_TOPK=$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt
STATUS=$EVID/status.txt
LOG=$EVID/v2g_g4a_actionability_control_audit.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "g4a_control_audit_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import torch

BASE=Path('/sda/home/wangyuxin/ConvIR-B')
ROOT=BASE/'repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign-f4-044b7798'
EVID=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709'
DATA=BASE/'datasets/Haze4K/Haze4K'
A0=BASE/'checkpoints/official/Haze4K/haze4k-base.pkl'
SPLIT=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json'
V2_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json'
V2B_THRESH=BASE/'repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json'
D3=BASE/'repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt'
D7C_TOPK=BASE/'repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt'
V2F=ROOT/'experience_docx/tools/run_chd_rm_v2f_need_target_head_redesign.py'

spec=importlib.util.spec_from_file_location('chdrm_v2f_tool', V2F)
v2f=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2f)
v2e=v2f.v2e; d7c=v2e.d7c; v2d=v2e.v2d; v2b=v2d.v2b

print('V2G_G4A_PY_START')
for p in [DATA/'train'/'haze', DATA/'train'/'gt', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C_TOPK]:
    if not Path(p).exists(): raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp: raise RuntimeError(f'Forbidden runtime path: {p}')

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
        fieldnames=fields
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def write_json(path, obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')

def qstats(vals):
    vals=sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals: return {'n':0}
    def q(p): return vals[min(len(vals)-1,max(0,int(round((len(vals)-1)*p))))]
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'max':vals[-1]}

def matched_threshold(score, coverage):
    return float(np.quantile(score.astype(np.float64), max(0.0, min(1.0, 1.0-float(coverage)))))

def safe_div(a,b): return float(a)/float(b) if b else math.nan

def auroc(y, s): return float(v2e.v2d.v2b.auroc(np.asarray(y).astype(bool), np.asarray(s).astype(np.float32)))
def auprc(y, s): return float(v2e.v2d.v2b.auprc(np.asarray(y).astype(bool), np.asarray(s).astype(np.float32)))

split=json.loads(SPLIT.read_text(encoding='utf-8'))
train_names=sorted(split['splits']['train_inner']); val_names=sorted(split['splits']['val_inner'])
if len(train_names)!=2400 or len(val_names)!=600: raise RuntimeError('bad split')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=v2b.load_model(A0, device); model.eval()
for p in model.parameters(): p.requires_grad_(False)
density_head=v2d.load_density_head(D3, device); density_head.eval()
topk_head=v2e.load_head(D7C_TOPK, device); topk_head.eval()
for m in [density_head, topk_head]:
    for p in m.parameters(): p.requires_grad_(False)

density_stats=json.loads(V2_THRESH.read_text(encoding='utf-8'))
target_info=json.loads(V2B_THRESH.read_text(encoding='utf-8'))
q33=float(target_info['quantile']['q33']); q66=float(target_info['quantile']['q66']); q80=float(target_info['quantile']['q80'])
density_q33=float(density_stats['density']['q33']); density_q66=float(density_stats['density']['q66'])
f1=json.loads((ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/ldhn_target_autopsy_summary.json').read_text())
grad_p90=float(f1['density_gradient_p90'])
candidate_threshold=0.5773006677627563
map_grid=64; blur_kernel=9; blur_radii=[5,9,15]; near_haze_radius=3

def collect(names, split_name):
    ds=v2e.Haze4KPairDataset(names, DATA, max_items=0, seed=3407)
    parts={k:[] for k in ['action','low_adjacent','negative','ignore','isolated','d7c','density_pred','density_true','dark_proxy','need_target']}
    per=[]; start=time.time()
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
            dark=v2b.v2.normalize(v2b.v2.raw_dark_density(hazy, 15), density_stats['d0_dark_density']['raw_p1'], density_stats['d0_dark_density']['raw_p99'])
            target_map=v2f.pool_map(target,map_grid); density_map=v2f.pool_map(density,map_grid); pred_map=v2f.pool_map(pred,map_grid)
            density_pred_map=v2f.pool_map(context[:,-1:],map_grid); dark_map=v2f.pool_map(dark,map_grid)
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
            parts['action'].append(action.reshape(-1)); parts['low_adjacent'].append(adjacent.reshape(-1)); parts['negative'].append(negative.reshape(-1)); parts['ignore'].append(ignore.reshape(-1)); parts['isolated'].append(isolated.reshape(-1))
            parts['d7c'].append(pred_map.reshape(-1)); parts['density_pred'].append(density_pred_map.reshape(-1)); parts['density_true'].append(density_map.reshape(-1)); parts['dark_proxy'].append((1.0-dark_map).reshape(-1)); parts['need_target'].append(target_map.reshape(-1))
            per.append({'split':split_name,'name':name,'action_pixels':int(action.sum()),'negative_pixels':int(negative.sum()),'ignore_pixels':int(ignore.sum()),'low_adjacent_pixels':int(adjacent.sum()),'isolated_pixels':int(isolated.sum())})
            if (idx+1)%200==0: print(f'g4a_collect {split_name} {idx+1}/{len(ds)} elapsed={time.time()-start:.1f}s', flush=True)
    arr={k:np.concatenate(v).astype(np.float32 if k in ['d7c','density_pred','density_true','dark_proxy','need_target'] else bool) for k,v in parts.items()}
    return arr, per

train, train_per=collect(train_names,'train_inner')
val, val_per=collect(val_names,'val_inner')
train_selected_coverage=float((train['d7c']>=candidate_threshold).mean())
# Controls: matched to D7c selected train coverage. Need target is an oracle sanity row, not deployable.
scores={
    'd7c_topk_score':('deployable_prior_candidate', 'd7c'),
    'd3_density_pred_matched':('deployable_density_control', 'density_pred'),
    'true_density_oracle_matched':('diagnostic_density_oracle_not_deployable', 'density_true'),
    'dark_channel_proxy_matched':('handcrafted_proxy_control', 'dark_proxy'),
    'need_target_oracle_not_deployable':('diagnostic_target_oracle_upper_bound', 'need_target'),
}
rows=[]; per_image=[]
for name,(kind,key) in scores.items():
    thr = candidate_threshold if name=='d7c_topk_score' else matched_threshold(train[key], train_selected_coverage)
    for split_name,arr,per_base in [('train_inner',train,train_per),('val_inner',val,val_per)]:
        pred=arr[key] >= thr
        action=arr['action']; neg=arr['negative']; ignore=arr['ignore']; iso=arr['isolated']; lowadj=arr['low_adjacent']
        rows.append({
            'score':name,'kind':kind,'split':split_name,'threshold':thr,
            'selected_coverage':float(pred.mean()),
            'action_positive_coverage':float(action.mean()),
            'negative_low_risk_coverage':float(neg.mean()),
            'ignore_coverage':float(ignore.mean()),
            'action_recall':safe_div((pred & action).sum(), action.sum()),
            'low_adjacent_recall':safe_div((pred & lowadj).sum(), lowadj.sum()),
            'negative_false_rate':safe_div((pred & neg).sum(), neg.sum()),
            'ignore_hit_rate':safe_div((pred & ignore).sum(), ignore.sum()),
            'isolated_ldhn_hit_rate':safe_div((pred & iso).sum(), iso.sum()),
            'action_precision_vs_neg_ignore':safe_div((pred & action).sum(), pred.sum()),
            'auroc_action_vs_negative':auroc(np.concatenate([np.ones(int(action.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)]), np.concatenate([arr[key][action], arr[key][neg]])) if int(action.sum()) and int(neg.sum()) else math.nan,
            'auprc_action_vs_negative':auprc(np.concatenate([np.ones(int(action.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)]), np.concatenate([arr[key][action], arr[key][neg]])) if int(action.sum()) and int(neg.sum()) else math.nan,
            'auroc_lowadjacent_vs_negative':auroc(np.concatenate([np.ones(int(lowadj.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)]), np.concatenate([arr[key][lowadj], arr[key][neg]])) if int(lowadj.sum()) and int(neg.sum()) else math.nan,
            'auprc_lowadjacent_vs_negative':auprc(np.concatenate([np.ones(int(lowadj.sum()), dtype=bool), np.zeros(int(neg.sum()), dtype=bool)]), np.concatenate([arr[key][lowadj], arr[key][neg]])) if int(lowadj.sum()) and int(neg.sum()) else math.nan,
        })
        # per-image tail false/hit distributions on val/train from flat chunks of 4096
        pix_per_img=4096
        action_rec=[]; neg_false=[]; ignore_hit=[]; iso_hit=[]; lowadj_rec=[]
        n_img=len(per_base)
        for i in range(n_img):
            sl=slice(i*pix_per_img,(i+1)*pix_per_img)
            pp=pred[sl]; aa=action[sl]; nn=neg[sl]; ii=ignore[sl]; isom=iso[sl]; la=lowadj[sl]
            if aa.sum(): action_rec.append(safe_div((pp & aa).sum(), aa.sum()))
            if nn.sum(): neg_false.append(safe_div((pp & nn).sum(), nn.sum()))
            if ii.sum(): ignore_hit.append(safe_div((pp & ii).sum(), ii.sum()))
            if isom.sum(): iso_hit.append(safe_div((pp & isom).sum(), isom.sum()))
            if la.sum(): lowadj_rec.append(safe_div((pp & la).sum(), la.sum()))
        srow={'score':name,'split':split_name}
        for label,vals in [('action_recall',action_rec),('negative_false_rate',neg_false),('ignore_hit_rate',ignore_hit),('isolated_ldhn_hit_rate',iso_hit),('low_adjacent_recall',lowadj_rec)]:
            st=qstats(vals)
            for k,v in st.items(): srow[f'{label}_{k}']=v
        per_image.append(srow)

write_csv(EVID/'g4a_actionability_control_audit_summary.csv', rows)
write_csv(EVID/'g4a_actionability_control_per_image_tail.csv', per_image)
write_csv(EVID/'g4a_actionability_control_per_image_support.csv', train_per+val_per)
# Decision against density controls.
val_rows={r['score']:r for r in rows if r['split']=='val_inner'}
d7=val_rows['d7c_topk_score']; d3=val_rows['d3_density_pred_matched']; true_d=val_rows['true_density_oracle_matched']
decision={
    'status':'COMPLETED_G4A_ACTIONABILITY_CONTROL_AUDIT',
    'locked_haze4k_test_usage':'none','D2':'not_run','RARM':'not_connected_or_trained','v3':'not_run','F5':'not_run',
    'threshold_selection':'train_inner matched to D7c selected coverage; val_inner reported',
    'train_selected_coverage_reference':train_selected_coverage,
    'val_d7c':d7,
    'val_d3_density_control':d3,
    'val_true_density_oracle':true_d,
    'd7c_minus_d3_action_recall':float(d7['action_recall'])-float(d3['action_recall']),
    'd7c_minus_d3_low_adjacent_recall':float(d7['low_adjacent_recall'])-float(d3['low_adjacent_recall']),
    'd7c_minus_d3_negative_false_rate':float(d7['negative_false_rate'])-float(d3['negative_false_rate']),
    'interpretation': 'D7c must beat density-only on action recall/low-adjacent recall without losing low-risk safety before any selective-head screen is considered.',
    'next_recommended_stage':'If D7c beats density controls, G4b may be a small selective-head screen under three-state target; still no F5/v3/RARM.'
}
write_json(EVID/'v2g_g4a_actionability_control_closeout.json', decision)
md='# v2g G4a Actionability Control Audit\n\nStatus: `COMPLETED_G4A_ACTIONABILITY_CONTROL_AUDIT`\n\nPolicy: no locked test, no D2, no RARM, no v3, no F5, no new head training. Thresholds for controls are selected on train_inner to match D7c selected coverage.\n\n| Score | Kind | Val coverage | Action recall | Low-adj recall | Negative false | Ignore hit | Isolated hit | AUROC action-vs-neg | AUROC lowadj-vs-neg |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
for r in [rr for rr in rows if rr['split']=='val_inner']:
    md += f"| {r['score']} | {r['kind']} | {float(r['selected_coverage']):.6f} | {float(r['action_recall']):.6f} | {float(r['low_adjacent_recall']):.6f} | {float(r['negative_false_rate']):.6f} | {float(r['ignore_hit_rate']):.6f} | {float(r['isolated_ldhn_hit_rate']):.6f} | {float(r['auroc_action_vs_negative']):.6f} | {float(r['auroc_lowadjacent_vs_negative']):.6f} |\n"
md += '\nInterpretation: compare D7c against deployable D3 density and diagnostic true-density oracle before any new training.\n'
(EVID/'v2g_g4a_actionability_control_summary.md').write_text(md, encoding='utf-8')
with open(EVID/'v2g_overall_result_summary.md','a',encoding='utf-8') as f:
    f.write('\n## G4a Actionability Controls\n\n')
    f.write(md)
print('V2G_G4A_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "g4a_control_audit_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2G_G4A_CONTROL_AUDIT_OK; else echo V2G_G4A_CONTROL_AUDIT_FAILED; fi
exit "$rc"
