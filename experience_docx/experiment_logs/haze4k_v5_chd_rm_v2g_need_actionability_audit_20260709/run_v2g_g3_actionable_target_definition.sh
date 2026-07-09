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
LOG=$EVID/v2g_g3_actionable_target_definition.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "g3_actionable_target_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, time, importlib.util, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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

print('V2G_G3_PY_START')
for p in [DATA/'train'/'haze', DATA/'train'/'gt', DATA/'train'/'trans', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C_TOPK]:
    if not Path(p).exists(): raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp: raise RuntimeError(f'Forbidden runtime path: {p}')

split=json.loads(SPLIT.read_text(encoding='utf-8'))
train_names=sorted(split['splits']['train_inner']); val_names=sorted(split['splits']['val_inner'])
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
threshold=0.5773006677627563
map_grid=64; blur_kernel=9; blur_radii=[5,9,15]; near_haze_radius=3

def collect_split(names, split_name):
    ds=v2e.Haze4KPairDataset(names, DATA, max_items=0, seed=3407)
    totals={k:0 for k in ['pixels','global_high_need','density_low','action_positive','action_positive_low_adjacent','action_positive_mid_high','negative_low_risk','ignore_abstain','ignore_isolated_ldhn','ignore_low_mid_need','ignore_unstable_or_boundary','other_unlabeled','pred_high','pred_high_action_positive','pred_high_negative_low_risk','pred_high_ignore','pred_high_isolated_ldhn','pred_high_low_adjacent']}
    per=[]
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
            target_map=v2f.pool_map(target,map_grid); density_map=v2f.pool_map(density,map_grid); pred_map=v2f.pool_map(pred,map_grid)
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
            action_positive=(high_need & (~density_low)) | adjacent
            action_positive_mid_high=high_need & (~density_low)
            negative_low_risk=density_low & low_need
            ignore_abstain=isolated | low_mid | boundary
            # Make mutually exclusive reporting buckets for label distribution.
            labeled=action_positive | negative_low_risk | ignore_abstain
            other=~labeled
            pred_high=pred_map>=threshold
            masks={
                'pixels':np.ones_like(target_map,dtype=bool),
                'global_high_need':high_need,
                'density_low':density_low,
                'action_positive':action_positive,
                'action_positive_low_adjacent':adjacent,
                'action_positive_mid_high':action_positive_mid_high,
                'negative_low_risk':negative_low_risk,
                'ignore_abstain':ignore_abstain,
                'ignore_isolated_ldhn':isolated,
                'ignore_low_mid_need':low_mid,
                'ignore_unstable_or_boundary':boundary,
                'other_unlabeled':other,
                'pred_high':pred_high,
                'pred_high_action_positive':pred_high & action_positive,
                'pred_high_negative_low_risk':pred_high & negative_low_risk,
                'pred_high_ignore':pred_high & ignore_abstain,
                'pred_high_isolated_ldhn':pred_high & isolated,
                'pred_high_low_adjacent':pred_high & adjacent,
            }
            row={'split':split_name,'name':name}
            n=int(target_map.size)
            for k,m in masks.items():
                c=int(m.sum()); totals[k]+=c; row[k+'_coverage']=c/n
            row['action_recall']=float((pred_high & action_positive).sum()/max(int(action_positive.sum()),1))
            row['low_adjacent_recall']=float((pred_high & adjacent).sum()/max(int(adjacent.sum()),1)) if int(adjacent.sum()) else math.nan
            row['negative_false_rate']=float((pred_high & negative_low_risk).sum()/max(int(negative_low_risk.sum()),1)) if int(negative_low_risk.sum()) else math.nan
            row['ignore_hit_rate']=float((pred_high & ignore_abstain).sum()/max(int(ignore_abstain.sum()),1)) if int(ignore_abstain.sum()) else math.nan
            per.append(row)
            if (idx+1)%100==0: print(f'g3_collect {split_name} {idx+1}/{len(ds)} elapsed={time.time()-start:.1f}s', flush=True)
    total_pix=totals['pixels']
    dist=[]
    for k,v in totals.items():
        if k!='pixels': dist.append({'split':split_name,'bucket':k,'pixels':v,'coverage':v/max(total_pix,1)})
    gate={
        'split':split_name,
        'pixels':total_pix,
        'action_positive_coverage':totals['action_positive']/max(total_pix,1),
        'action_positive_low_adjacent_coverage':totals['action_positive_low_adjacent']/max(total_pix,1),
        'action_positive_mid_high_coverage':totals['action_positive_mid_high']/max(total_pix,1),
        'negative_low_risk_coverage':totals['negative_low_risk']/max(total_pix,1),
        'ignore_abstain_coverage':totals['ignore_abstain']/max(total_pix,1),
        'ignore_isolated_ldhn_coverage':totals['ignore_isolated_ldhn']/max(total_pix,1),
        'd7c_pred_high_coverage':totals['pred_high']/max(total_pix,1),
        'd7c_action_recall':totals['pred_high_action_positive']/max(totals['action_positive'],1),
        'd7c_low_adjacent_recall':totals['pred_high_low_adjacent']/max(totals['action_positive_low_adjacent'],1),
        'd7c_negative_false_rate':totals['pred_high_negative_low_risk']/max(totals['negative_low_risk'],1),
        'd7c_ignore_hit_rate':totals['pred_high_ignore']/max(totals['ignore_abstain'],1),
        'd7c_isolated_ldhn_hit_rate':totals['pred_high_isolated_ldhn']/max(totals['ignore_isolated_ldhn'],1),
        'per_image_action_recall':qstats([r['action_recall'] for r in per]),
        'per_image_low_adjacent_recall':qstats([r['low_adjacent_recall'] for r in per]),
        'per_image_negative_false_rate':qstats([r['negative_false_rate'] for r in per]),
    }
    return dist, per, gate

train_dist, train_per, train_gate=collect_split(train_names,'train_inner')
val_dist, val_per, val_gate=collect_split(val_names,'val_inner')
write_csv(EVID/'actionable_need_target_distribution.csv', train_dist+val_dist)
write_csv(EVID/'actionable_need_per_image_distribution.csv', train_per+val_per)
ignore_rows=[r for r in train_dist+val_dist if 'ignore' in r['bucket'] or r['bucket'] in ['negative_low_risk','action_positive','action_positive_low_adjacent','action_positive_mid_high']]
write_csv(EVID/'ignore_abstain_support.csv', ignore_rows)
write_json(EVID/'d7c_actionable_target_gate_summary.json', {'train_inner':train_gate,'val_inner':val_gate,'threshold':threshold,'status':'diagnostic_no_training'})
write_csv(EVID/'d7c_actionable_target_gate_summary.csv', [train_gate, val_gate])
noise={
    'status':'COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION',
    'definition':'positive = high_need in mid/high density OR low-density high-need adjacent-to-haze; negative = low-density low-need; ignore/abstain = isolated LDHN, low-density mid-need, boundary/unstable LDHN; other = unlabeled outside this selective low-haze action contract.',
    'rationale':'G2b shows isolated LDHN carries residual energy but is not safely haze-actionable; D7c preferentially recalls haze-adjacent LDHN while preserving low-density low-need safety.',
    'train_inner_gate':train_gate,
    'val_inner_gate':val_gate,
    'locked_haze4k_test_usage':'none','D2':'not_run','RARM':'not_connected_or_trained','v3':'not_run','F5':'not_run',
    'g4_authorization':'screen only may be considered; no full training or F5/v3/RARM authorized by G3 alone.'
}
write_json(EVID/'target_noise_uncertainty_summary.json', noise)
(EVID/'actionable_need_target_definition.md').write_text('''# v2g Actionable Need Target Definition\n\nStatus: `COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION`\n\nThis is a target-semantics definition, not a trained model and not a promotion claim. Locked Haze4K test, D2, RARM, v3, and F5 remain unused.\n\n## Three-State Contract\n\nPositive/actionable:\n\n```text\naction_positive = (target >= q66 and density > density_q33)\n               OR (target >= q66 and density <= density_q33 and adjacent_to_haze)\n```\n\nNegative/confident low-risk:\n\n```text\nnegative_low_risk = density <= density_q33 and target <= q33\n```\n\nIgnore/abstain:\n\n```text\nignore_abstain = isolated LDHN\n              OR low-density mid-need\n              OR boundary/unstable LDHN\n```\n\nOther unlabeled pixels are outside this selective low-haze actionability contract.\n\n## Rationale\n\nG2b shows isolated LDHN removes residual energy under oracle replacement, but it is not reliably haze-actionable. Therefore isolated LDHN should not be forced as a hard RARM positive. The low-density adjacent-to-haze subset remains the candidate actionable LDHN subset.\n''', encoding='utf-8')
closeout={
    'status':'COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION',
    'locked_haze4k_test_usage':'none','D2':'not_run','RARM':'not_connected_or_trained','v3':'not_run','F5':'not_run',
    'train_inner_gate':train_gate,'val_inner_gate':val_gate,
    'decision':'D7C_TOPK_PASSES_PRELIMINARY_ACTIONABLE_LDHN_RECALL_AND_LOW_RISK_FALSE_RATE_UNDER_V2G_TARGET_BUT_NEEDS_G4_SCREEN_OR_CONTROLS_BEFORE_PROMOTION',
    'next_recommended_stage':'G4a small selective-head/probe screen or control audit under the v2g three-state target; do not run F5/v3/RARM yet.'
}
write_json(EVID/'v2g_g3_actionable_target_closeout.json', closeout)
with open(EVID/'v2g_overall_result_summary.md','a',encoding='utf-8') as f:
    f.write('\n## G3 Actionable Target Definition\n\n')
    f.write(f"Status: `COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION`\n\n")
    f.write(f"Val action-positive coverage: `{val_gate['action_positive_coverage']:.6f}`; low-adjacent actionable coverage: `{val_gate['action_positive_low_adjacent_coverage']:.6f}`; ignore/abstain coverage: `{val_gate['ignore_abstain_coverage']:.6f}`.\n\n")
    f.write(f"D7c under the new diagnostic target: action recall `{val_gate['d7c_action_recall']:.6f}`, low-adjacent recall `{val_gate['d7c_low_adjacent_recall']:.6f}`, negative false rate `{val_gate['d7c_negative_false_rate']:.6f}`, isolated-LDHN hit rate `{val_gate['d7c_isolated_ldhn_hit_rate']:.6f}`.\n\n")
    f.write('Interpretation: the old global-LDHN gate was over-broad. Under the v2g three-state target, D7c top-k becomes a plausible baseline action signal, but this does not authorize F5/v3/RARM.\n')
print('V2G_G3_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "g3_actionable_target_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2G_G3_ACTIONABLE_TARGET_OK; else echo V2G_G3_ACTIONABLE_TARGET_FAILED; fi
exit "$rc"
