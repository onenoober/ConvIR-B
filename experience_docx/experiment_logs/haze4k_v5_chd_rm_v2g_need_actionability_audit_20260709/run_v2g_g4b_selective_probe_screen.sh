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
LOG=$EVID/v2g_g4b_selective_probe_screen.log

mkdir -p "$EVID"
cp "$0" "$EVID/run_v2g_g4b_selective_probe_screen.sh"
{
  echo "g4b_selective_probe_screen_start $(date --iso-8601=seconds) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}"
  echo "g4b_policy=no_locked_test_no_D2_no_RARM_no_v3_no_F5_no_weights_saved"
} | tee -a "$STATUS"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, statistics, time, importlib.util, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')

def qstats(vals):
    vals=sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals: return {'n':0}
    def q(p): return vals[min(len(vals)-1,max(0,int(round((len(vals)-1)*p))))]
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'max':vals[-1]}

def safe_div(a,b):
    return float(a)/float(b) if b else math.nan

def matched_threshold(score, target_coverage):
    score=np.asarray(score, dtype=np.float32)
    if score.size == 0: return math.nan
    q=max(0.0, min(1.0, 1.0-float(target_coverage)))
    return float(np.quantile(score, q))

class ProbeNet(nn.Module):
    def __init__(self, in_dim, kind):
        super().__init__()
        if kind == 'linear':
            self.net = nn.Linear(in_dim, 1)
        elif kind == 'mlp':
            self.net = nn.Sequential(nn.Linear(in_dim, 96), nn.ReLU(inplace=True), nn.Dropout(0.05), nn.Linear(96, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        else:
            raise ValueError(kind)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_probe(train_x, train_y, kind, seed, device, epochs=6, batch_size=2048, lr=8e-4, wd=1e-3, permute_labels=False):
    rng=np.random.default_rng(seed + (70000 if permute_labels else 0))
    y=train_y.copy().astype(np.float32)
    if permute_labels:
        y=rng.permutation(y)
    mean=train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std=(train_x.std(axis=0, keepdims=True)+1e-6).astype(np.float32)
    x=((train_x-mean)/std).astype(np.float32)
    model=ProbeNet(x.shape[1], kind).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn=nn.BCEWithLogitsLoss()
    xt=torch.from_numpy(x); yt=torch.from_numpy(y)
    losses=[]
    model.train()
    for epoch in range(epochs):
        order=rng.permutation(x.shape[0])
        ep=[]
        for start in range(0, x.shape[0], batch_size):
            idx=order[start:start+batch_size]
            logits=model(xt[idx].to(device))
            loss=loss_fn(logits, yt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ep.append(float(loss.detach().cpu()))
        losses.append(float(statistics.mean(ep)))
        print(f'g4b_train kind={kind} permuted={permute_labels} epoch={epoch+1} loss={losses[-1]:.6f}', flush=True)
    model.eval()
    return {'model':model, 'mean':mean, 'std':std, 'kind':kind, 'permuted':permute_labels, 'loss_final':losses[-1], 'loss_history':losses}

def score_probe(probe, x_np, device, batch_size=8192):
    x=((x_np.astype(np.float32)-probe['mean'])/probe['std']).astype(np.float32)
    out=[]
    with torch.no_grad():
        xt=torch.from_numpy(x)
        for start in range(0, x.shape[0], batch_size):
            logits=probe['model'](xt[start:start+batch_size].to(device))
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)

print('V2G_G4B_PY_START', flush=True)
for p in [DATA/'train'/'haze', DATA/'train'/'gt', DATA/'train'/'trans', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C_TOPK]:
    if not Path(p).exists(): raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp:
        raise RuntimeError(f'Forbidden runtime path: {p}')

split=json.loads(SPLIT.read_text(encoding='utf-8'))
train_names=sorted(split['splits']['train_inner']); val_names=sorted(split['splits']['val_inner'])
rng=np.random.default_rng(3407)
train_calib_names=sorted(rng.choice(train_names, size=min(360, len(train_names)), replace=False).tolist())

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
map_grid=64; blur_kernel=9; blur_radii=[5,9,15]; near_haze_radius=3; d7c_fixed_threshold=0.5773006677627563

protocol={
    'status':'AUTHORIZED_G4B_SMALL_SCREEN_ONLY',
    'route_identity':'G4b small selective-head/probe screen under v2g three-state target',
    'forbidden':['locked Haze4K test','D2','RARM connection/training','v3','F5','saving checkpoints/weights','model promotion'],
    'training_scope':'balanced train_inner pixel samples; tiny linear/MLP probes only; thresholds selected on train_inner calibration subset',
    'evaluation_scope':'val_inner only; D7c and D3 density controls reported with matched train coverage',
    'primary_screen_gate':{
        'candidate_must_beat_d7c_action_recall_by_at_least':0.02,
        'candidate_low_adjacent_recall_must_be_at_least_d7c':True,
        'candidate_negative_false_rate_max':0.005,
        'candidate_ignore_hit_rate_max':0.05,
        'candidate_isolated_ldhn_hit_rate_max':0.03,
        'candidate_selected_coverage_range':[0.25,0.35],
        'passing_does_not_authorize':['F5','v3','RARM','D2','locked test']
    }
}
(EVID/'v2g_g4b_selective_probe_protocol.md').write_text('# v2g G4b Selective Probe Screen Protocol\n\n```json\n'+json.dumps(protocol, indent=2, ensure_ascii=False)+'\n```\n', encoding='utf-8')

def feature_slices(context_np, density_pred_np, d7c_np):
    c=context_np.astype(np.float32, copy=False)
    learned_end=max(1, c.shape[0]-10)
    image_end=max(1, c.shape[0]-1)
    d=density_pred_np.reshape(1,*density_pred_np.shape).astype(np.float32)
    g=v2f.gradient_mag(density_pred_np).reshape(1,*density_pred_np.shape).astype(np.float32)
    pooled=v2f.local_max_2d(density_pred_np,2).reshape(1,*density_pred_np.shape).astype(np.float32)
    ds=d7c_np.reshape(1,*d7c_np.shape).astype(np.float32)
    return {
        'context_core': c[:learned_end],
        'context_image_density': np.concatenate([c[:image_end], d, g], axis=0),
        'context_plus_d7c_density': np.concatenate([c, g, pooled, ds], axis=0),
    }

def make_label_maps(hazy, gt, a0):
    raw_need=v2b.v2.raw_need(a0, gt, blur_kernel)
    target=v2b.make_target(raw_need, target_info, 'quantile')
    density=v2b.v2.normalize(v2b.v2.raw_density(hazy, gt, blur_kernel), density_stats['density']['raw_p1'], density_stats['density']['raw_p99'])
    target_map=v2f.pool_map(target, map_grid); density_map=v2f.pool_map(density, map_grid)
    blur_maps=[]
    for radius in blur_radii:
        rn=v2b.v2.raw_need(a0, gt, radius); tb=v2b.make_target(rn, target_info, 'quantile')
        blur_maps.append(v2f.pool_map(tb, map_grid))
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
    other=~(action | negative | ignore)
    return {'target':target_map,'density':density_map,'action':action,'negative':negative,'ignore':ignore,'isolated':isolated,'low_adjacent':adjacent,'other':other}

def make_pack_for_image(name):
    ds=v2e.Haze4KPairDataset([name], DATA, max_items=0, seed=3407)
    _,hazy,gt=ds[0]
    hazy=hazy.unsqueeze(0).to(device); gt=gt.unsqueeze(0).to(device)
    with torch.no_grad():
        padded,h,w=v2b.v2.pad32(hazy)
        a0,context=d7c.convir_a0_context(model,density_head,padded)
        a0=a0[:,:,:h,:w]; context=context[:,:,:h,:w]
        d7c_pred,_=d7c.predict_head(topk_head,context)
        context_small=F.adaptive_avg_pool2d(context,(map_grid,map_grid)).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        d7c_map=v2f.pool_map(d7c_pred,map_grid).astype(np.float32)
    labels=make_label_maps(hazy,gt,a0)
    density_pred=context_small[-1].astype(np.float32)
    features=feature_slices(context_small,density_pred,d7c_map)
    labels['d7c']=d7c_map
    labels['density_pred']=density_pred
    return features, labels

train_features={}
train_y=[]
start=time.time()
per_image_per_class=10
max_rows=64000
for idx,name in enumerate(train_names):
    features, labels=make_pack_for_image(name)
    pos=np.argwhere(labels['action'])
    neg=np.argwhere(labels['negative'])
    take=min(len(pos),len(neg),per_image_per_class)
    if take>0:
        pidx=pos[rng.choice(len(pos), size=take, replace=False)]
        nidx=neg[rng.choice(len(neg), size=take, replace=False)]
        coords=np.concatenate([pidx,nidx],axis=0)
        y=np.concatenate([np.ones(take,dtype=np.float32),np.zeros(take,dtype=np.float32)])
        for key,fmap in features.items():
            vals=fmap[:,coords[:,0],coords[:,1]].T.astype(np.float32, copy=False)
            train_features.setdefault(key,[]).append(vals)
        train_y.append(y)
    if (idx+1)%150==0:
        rows=sum(x.shape[0] for x in train_y)
        print(f'g4b_collect_train_samples {idx+1}/{len(train_names)} rows={rows} elapsed={time.time()-start:.1f}s', flush=True)
train_y=np.concatenate(train_y).astype(np.float32)
for key in list(train_features):
    train_features[key]=np.concatenate(train_features[key],axis=0).astype(np.float32)
if train_y.size>max_rows:
    keep=rng.choice(train_y.size, size=max_rows, replace=False)
    train_y=train_y[keep]
    for key in train_features:
        train_features[key]=train_features[key][keep]
print(f'g4b_train_sample_final rows={train_y.size} positives={int(train_y.sum())} negatives={int((train_y<0.5).sum())}', flush=True)

probe_specs=[]
for feature_set in ['context_core','context_image_density','context_plus_d7c_density']:
    for kind in ['linear','mlp']:
        probe=fit_probe(train_features[feature_set], train_y, kind, 3407, device)
        probe_specs.append({'name':f'{feature_set}_{kind}', 'feature_set':feature_set, 'kind':kind, 'probe':probe, 'control':'normal'})
probe_specs.append({'name':'context_plus_d7c_density_mlp_labelperm_control','feature_set':'context_plus_d7c_density','kind':'mlp','probe':fit_probe(train_features['context_plus_d7c_density'], train_y, 'mlp', 9407, device, epochs=4, permute_labels=True),'control':'label_permutation'})

def score_split(names, split_name):
    score_parts={'d7c_topk_score':[], 'd3_density_pred':[]}
    for spec in probe_specs:
        score_parts[spec['name']]=[]
    masks={k:[] for k in ['action','negative','ignore','isolated','low_adjacent','other']}
    support_rows=[]
    start=time.time()
    for idx,name in enumerate(names):
        features, labels=make_pack_for_image(name)
        action=labels['action'].reshape(-1); neg=labels['negative'].reshape(-1); ign=labels['ignore'].reshape(-1); iso=labels['isolated'].reshape(-1); lowadj=labels['low_adjacent'].reshape(-1); other=labels['other'].reshape(-1)
        for k,arr in [('action',action),('negative',neg),('ignore',ign),('isolated',iso),('low_adjacent',lowadj),('other',other)]:
            masks[k].append(arr.astype(bool))
        score_img={'d7c_topk_score':labels['d7c'].reshape(-1).astype(np.float32),'d3_density_pred':labels['density_pred'].reshape(-1).astype(np.float32)}
        for spec in probe_specs:
            fmap=features[spec['feature_set']]
            x=fmap.reshape(fmap.shape[0], -1).T.astype(np.float32, copy=False)
            score_img[spec['name']]=score_probe(spec['probe'], x, device)
        for key,score in score_img.items():
            score_parts[key].append(score)
        support_rows.append({'name':name,'split':split_name,'action_pixels':int(action.sum()),'negative_pixels':int(neg.sum()),'ignore_pixels':int(ign.sum()),'isolated_pixels':int(iso.sum()),'low_adjacent_pixels':int(lowadj.sum()),'other_pixels':int(other.sum())})
        if (idx+1)%100==0:
            print(f'g4b_score_{split_name} {idx+1}/{len(names)} elapsed={time.time()-start:.1f}s', flush=True)
    return {'support':support_rows, 'masks':{k:np.concatenate(v).astype(bool) for k,v in masks.items()}, 'scores':{k:np.concatenate(v).astype(np.float32) for k,v in score_parts.items()}}

train_calib=score_split(train_calib_names,'train_calib')
val=score_split(val_names,'val_inner')

d7c_train_selected=float((train_calib['scores']['d7c_topk_score']>=d7c_fixed_threshold).mean())
thresholds={'d7c_topk_score':d7c_fixed_threshold}
for key,score in train_calib['scores'].items():
    if key!='d7c_topk_score':
        thresholds[key]=matched_threshold(score, d7c_train_selected)

score_kinds={'d7c_topk_score':'deployable_prior_baseline','d3_density_pred':'deployable_density_control'}
for spec in probe_specs:
    score_kinds[spec['name']]='label_permutation_control' if spec['control']=='label_permutation' else 'small_selective_probe'

def binary_auc_metrics(score, pos_mask, neg_mask):
    if int(pos_mask.sum())==0 or int(neg_mask.sum())==0: return (math.nan, math.nan)
    y=np.concatenate([np.ones(int(pos_mask.sum()), dtype=bool), np.zeros(int(neg_mask.sum()), dtype=bool)])
    s=np.concatenate([score[pos_mask], score[neg_mask]]).astype(np.float32)
    return float(v2b.auroc(y,s)), float(v2b.auprc(y,s))

def summarize_at_threshold(split_pack, split_name, key, threshold):
    score=split_pack['scores'][key]; pred=score>=threshold
    m=split_pack['masks']
    action=m['action']; neg=m['negative']; ign=m['ignore']; iso=m['isolated']; lowadj=m['low_adjacent']
    auroc_an, auprc_an=binary_auc_metrics(score, action, neg)
    auroc_ln, auprc_ln=binary_auc_metrics(score, lowadj, neg)
    return {
        'score':key,'kind':score_kinds[key],'split':split_name,'threshold':float(threshold),'threshold_source':'train_calib_match_d7c_fixed_coverage',
        'selected_coverage':float(pred.mean()),
        'action_positive_coverage':float(action.mean()),
        'negative_low_risk_coverage':float(neg.mean()),
        'ignore_coverage':float(ign.mean()),
        'action_recall':safe_div((pred & action).sum(), action.sum()),
        'low_adjacent_recall':safe_div((pred & lowadj).sum(), lowadj.sum()),
        'negative_false_rate':safe_div((pred & neg).sum(), neg.sum()),
        'ignore_hit_rate':safe_div((pred & ign).sum(), ign.sum()),
        'isolated_ldhn_hit_rate':safe_div((pred & iso).sum(), iso.sum()),
        'action_precision_vs_all_selected':safe_div((pred & action).sum(), pred.sum()),
        'auroc_action_vs_negative':auroc_an,'auprc_action_vs_negative':auprc_an,
        'auroc_lowadjacent_vs_negative':auroc_ln,'auprc_lowadjacent_vs_negative':auprc_ln,
    }

summary=[]
for key,thr in thresholds.items():
    summary.append(summarize_at_threshold(train_calib,'train_calib',key,thr))
    summary.append(summarize_at_threshold(val,'val_inner',key,thr))
write_csv(EVID/'g4b_selective_probe_summary.csv', summary)

per_tail=[]
pix_per_img=map_grid*map_grid
n_img=len(val_names)
for key,thr in thresholds.items():
    score=val['scores'][key]; pred=score>=thr; m=val['masks']
    vals={label:[] for label in ['action_recall','low_adjacent_recall','negative_false_rate','ignore_hit_rate','isolated_ldhn_hit_rate']}
    for i in range(n_img):
        sl=slice(i*pix_per_img,(i+1)*pix_per_img)
        pp=pred[sl]; action=m['action'][sl]; neg=m['negative'][sl]; ign=m['ignore'][sl]; iso=m['isolated'][sl]; lowadj=m['low_adjacent'][sl]
        if action.sum(): vals['action_recall'].append(safe_div((pp & action).sum(), action.sum()))
        if lowadj.sum(): vals['low_adjacent_recall'].append(safe_div((pp & lowadj).sum(), lowadj.sum()))
        if neg.sum(): vals['negative_false_rate'].append(safe_div((pp & neg).sum(), neg.sum()))
        if ign.sum(): vals['ignore_hit_rate'].append(safe_div((pp & ign).sum(), ign.sum()))
        if iso.sum(): vals['isolated_ldhn_hit_rate'].append(safe_div((pp & iso).sum(), iso.sum()))
    row={'score':key,'kind':score_kinds[key],'split':'val_inner'}
    for label,arr in vals.items():
        st=qstats(arr)
        for sk,sv in st.items(): row[f'{label}_{sk}']=sv
    per_tail.append(row)
write_csv(EVID/'g4b_selective_probe_per_image_tail.csv', per_tail)

curve=[]
coverage_targets=[0.20,0.25,0.30,0.35,0.40]
for key,score in train_calib['scores'].items():
    for cov in coverage_targets:
        thr=matched_threshold(score,cov)
        row=summarize_at_threshold(val,'val_inner',key,thr)
        row['threshold_source']=f'train_calib_coverage_{cov:.2f}'
        row['target_train_calib_coverage']=cov
        curve.append(row)
write_csv(EVID/'g4b_selective_probe_threshold_curve.csv', curve)

train_rows=[]
for spec in probe_specs:
    p=spec['probe']
    train_rows.append({'score':spec['name'],'feature_set':spec['feature_set'],'kind':spec['kind'],'control':spec['control'],'train_rows':int(train_y.size),'train_positive_rows':int(train_y.sum()),'train_negative_rows':int((train_y<0.5).sum()),'loss_final':p['loss_final'],'loss_history':json.dumps(p['loss_history'])})
write_csv(EVID/'g4b_selective_probe_train_log.csv', train_rows)
write_csv(EVID/'g4b_selective_probe_support.csv', train_calib['support'] + val['support'])

val_rows={r['score']:r for r in summary if r['split']=='val_inner'}
d7=val_rows['d7c_topk_score']; d3=val_rows['d3_density_pred']
probe_candidates=[r for r in val_rows.values() if r['kind']=='small_selective_probe']
best=sorted(probe_candidates, key=lambda r:(-float(r['action_recall']), float(r['negative_false_rate']), float(r['isolated_ldhn_hit_rate'])))[0]
pass_gate=(
    float(best['action_recall']) >= float(d7['action_recall']) + 0.02 and
    float(best['low_adjacent_recall']) >= float(d7['low_adjacent_recall']) and
    float(best['negative_false_rate']) <= 0.005 and
    float(best['ignore_hit_rate']) <= 0.05 and
    float(best['isolated_ldhn_hit_rate']) <= 0.03 and
    0.25 <= float(best['selected_coverage']) <= 0.35
)
perm_rows=[r for r in val_rows.values() if r['kind']=='label_permutation_control']
perm_clean=all(float(r['action_recall']) < float(best['action_recall'])-0.10 for r in perm_rows) if perm_rows else False
closeout={
    'status':'COMPLETED_G4B_SELECTIVE_PROBE_SCREEN',
    'decision_label':'G4B_SCREEN_PASS_NEXT_CONTROLS_ONLY' if pass_gate and perm_clean else 'PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3',
    'locked_haze4k_test_usage':'none','D2':'not_run','RARM':'not_connected_or_trained','v3':'not_run','F5':'not_run','weights_or_checkpoints_saved':'none',
    'threshold_policy':'All non-D7c thresholds selected on train_calib to match fixed D7c selected coverage; val_inner is report-only.',
    'train_sample_rows':int(train_y.size),'train_calib_images':len(train_calib_names),'val_inner_images':len(val_names),
    'd7c_val':d7,'d3_density_val':d3,'best_probe_val':best,'label_permutation_controls_val':perm_rows,
    'best_probe_minus_d7c':{
        'action_recall':float(best['action_recall'])-float(d7['action_recall']),
        'low_adjacent_recall':float(best['low_adjacent_recall'])-float(d7['low_adjacent_recall']),
        'negative_false_rate':float(best['negative_false_rate'])-float(d7['negative_false_rate']),
        'ignore_hit_rate':float(best['ignore_hit_rate'])-float(d7['ignore_hit_rate']),
        'isolated_ldhn_hit_rate':float(best['isolated_ldhn_hit_rate'])-float(d7['isolated_ldhn_hit_rate']),
        'auroc_action_vs_negative':float(best['auroc_action_vs_negative'])-float(d7['auroc_action_vs_negative']),
    },
    'screen_gate':protocol['primary_screen_gate'],
    'screen_gate_pass':bool(pass_gate and perm_clean),
    'label_permutation_control_clean':bool(perm_clean),
    'interpretation':('A tiny selective probe found a train-selected val_inner operating point that improves over D7c under the v2g target, but this only authorizes further controls, not F5/v3/RARM.' if pass_gate and perm_clean else 'The small selective probe screen did not produce a safe improvement over D7c under the predeclared G4b gate. Keep F5/v3/RARM/D2/locked test blocked.'),
    'next_recommended_stage':('G4c control audit/repeatability screen only; no F5/v3/RARM/D2/locked test.' if pass_gate and perm_clean else 'Pause or redesign G4b target/features; no F5/v3/RARM/D2/locked test.')
}
write_json(EVID/'v2g_g4b_selective_probe_closeout.json', closeout)

md=['# v2g G4b Selective Probe Screen','',f"Status: `{closeout['status']}`",'',f"Decision: `{closeout['decision_label']}`",'', 'Policy: no locked Haze4K test, no D2, no RARM, no v3, no F5, and no saved probe weights/checkpoints.', '', 'Thresholds for probe and density controls were selected on a train_inner calibration subset to match fixed D7c selected coverage; val_inner is report-only.', '', '## Primary Val Rows','', '| Score | Kind | Coverage | Action recall | Low-adj recall | Negative false | Ignore hit | Isolated hit | AUROC action-vs-neg |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in sorted(val_rows.values(), key=lambda x:(x['kind'], x['score'])):
    md.append(f"| {r['score']} | {r['kind']} | {float(r['selected_coverage']):.6f} | {float(r['action_recall']):.6f} | {float(r['low_adjacent_recall']):.6f} | {float(r['negative_false_rate']):.6f} | {float(r['ignore_hit_rate']):.6f} | {float(r['isolated_ldhn_hit_rate']):.6f} | {float(r['auroc_action_vs_negative']):.6f} |")
md += ['', '## Interpretation', '', closeout['interpretation'], '', '## Next', '', closeout['next_recommended_stage'], '']
(EVID/'v2g_g4b_selective_probe_summary.md').write_text('\n'.join(md), encoding='utf-8')
with open(EVID/'v2g_overall_result_summary.md','a',encoding='utf-8') as f:
    f.write('\n## G4b Selective Probe Screen\n\n')
    f.write(f"Status: `{closeout['status']}`\n\nDecision: `{closeout['decision_label']}`\n\n")
    f.write(closeout['interpretation']+'\n')

print('V2G_G4B_PY_OK', flush=True)
PY
rc=${PIPESTATUS[0]}
set -e
echo "g4b_selective_probe_screen_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then
  echo V2G_G4B_SELECTIVE_PROBE_FAILED
  exit "$rc"
fi
echo V2G_G4B_SELECTIVE_PROBE_OK
