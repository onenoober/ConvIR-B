#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v2h_b_shadow_modulation.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "v2h_b_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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
A_CLOSE=EVID/'v2h_a_risk_coverage_closeout.json'
ALPHAS=[0.1,0.2,0.3,0.5,1.0]

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

def write_json(path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
def safe_div(a,b): return float(a)/float(b) if b else math.nan
def psnr_from_mse(mse): return 99.0 if mse <= 0 else -10.0*math.log10(float(mse))
def qstats(vals):
    vals=sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals: return {'n':0,'mean':math.nan,'p50':math.nan,'p75':math.nan,'p90':math.nan,'p95':math.nan,'max':math.nan}
    def q(p): return vals[min(len(vals)-1,max(0,int(round((len(vals)-1)*p))))]
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'max':vals[-1]}
def upsample(mask64, h, w, device):
    m=torch.as_tensor(mask64.astype(np.float32), device=device).view(1,1,*mask64.shape)
    return (F.interpolate(m, size=(h,w), mode='nearest') >= 0.5)

print('V2H_B_PY_START')
close=json.loads(A_CLOSE.read_text(encoding='utf-8'))
if not close.get('gate_pass'):
    raise RuntimeError('v2h-A gate did not pass; v2h-B not authorized')
d7c_thr=float(close['primary_operating_point']['threshold'])
density_thr=float(close['density_matched_at_primary']['threshold'])
for p in [DATA/'train'/'haze', DATA/'train'/'gt', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C, V2F]:
    if not Path(p).exists(): raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp: raise RuntimeError(f'forbidden path: {p}')

split=json.loads(SPLIT.read_text(encoding='utf-8'))
val_names=sorted(split['splits']['val_inner'])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, 'val_inner', len(val_names), 'd7c_thr', d7c_thr, 'density_thr', density_thr, flush=True)
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
regions=['all','selected','action_positive','action_mid_high','low_adjacent','negative_low_risk','ignore_abstain','isolated_ldhn','low_haze','heavy_haze']
selectors=['d7c_fixed','density_matched','action_oracle']

global_agg={(sel,a):{'base_energy':0.0,'shadow_energy':0.0,'elements':0,'selected_pixels':0,'total_pixels':0} for sel in selectors for a in ALPHAS}
region_agg={(sel,a,r):{'base_energy':0.0,'shadow_energy':0.0,'elements':0,'pixels':0,'selected_pixels':0,'delta_abs_sum':0.0} for sel in selectors for a in ALPHAS for r in regions}
per_lists={(sel,a):{'psnr_gain':[],'negative_delta_l1':[],'negative_touch_rate':[],'isolated_delta_l1':[],'isolated_touch_rate':[],'low_haze_delta_l1':[],'low_haze_touch_rate':[],'action_region_gain':[]} for sel in selectors for a in ALPHAS}

def add_region(sel, alpha, region_name, rmask, smask, residual_sq, residual_abs, factor_sq, h, w):
    key=(sel,alpha,region_name)
    pix=int(rmask.sum().item())
    if pix <= 0: return math.nan
    elem=pix*3
    base=float((residual_sq*rmask).sum().item())
    shadow=float((residual_sq*rmask*factor_sq).sum().item())
    selected_pix=int((rmask & smask).sum().item())
    delta=float((residual_abs*rmask*smask*alpha).sum().item())
    a=region_agg[key]
    a['base_energy']+=base; a['shadow_energy']+=shadow; a['elements']+=elem; a['pixels']+=pix; a['selected_pixels']+=selected_pix; a['delta_abs_sum']+=delta
    return psnr_from_mse(shadow/elem)-psnr_from_mse(base/elem) if elem else math.nan

start=time.time()
ds=v2e.Haze4KPairDataset(val_names, DATA, max_items=0, seed=3407)
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
        action_mid_high=high_need & (~density_low)
        action=action_mid_high | adjacent
        negative=density_low & low_need
        ignore=isolated | low_mid | boundary
        heavy=density_map>=density_q66
        low_haze=density_low
        region64={
            'all':np.ones_like(target_map, dtype=bool),
            'action_positive':action,
            'action_mid_high':action_mid_high,
            'low_adjacent':adjacent,
            'negative_low_risk':negative,
            'ignore_abstain':ignore,
            'isolated_ldhn':isolated,
            'low_haze':low_haze,
            'heavy_haze':heavy,
        }
        selector64={
            'd7c_fixed':pred_map>=d7c_thr,
            'density_matched':density_pred_map>=density_thr,
            'action_oracle':action,
        }
        residual=a0-gt
        residual_sq=residual.float().pow(2)
        residual_abs=residual.float().abs()
        base_energy=float(residual_sq.sum().item())
        elems=int(residual_sq.numel())
        base_psnr=psnr_from_mse(base_energy/elems)
        for sel,mask64 in selector64.items():
            smask=upsample(mask64,h,w,device)
            selected_region64=mask64
            local_regions=dict(region64)
            local_regions['selected']=selected_region64
            total_pix=h*w
            sel_pix=int(smask.sum().item())
            for alpha in ALPHAS:
                factor_sq=(1.0 - alpha*smask.float()).pow(2)
                shadow_energy=float((residual_sq*factor_sq).sum().item())
                g=global_agg[(sel,alpha)]
                g['base_energy']+=base_energy; g['shadow_energy']+=shadow_energy; g['elements']+=elems; g['selected_pixels']+=sel_pix; g['total_pixels']+=total_pix
                per_lists[(sel,alpha)]['psnr_gain'].append(psnr_from_mse(shadow_energy/elems)-base_psnr)
                region_gain_cache={}
                for rname,r64 in local_regions.items():
                    rmask=upsample(r64,h,w,device)
                    rgain=add_region(sel,alpha,rname,rmask,smask,residual_sq,residual_abs,factor_sq,h,w)
                    region_gain_cache[rname]=rgain
                if math.isfinite(region_gain_cache.get('action_positive', math.nan)):
                    per_lists[(sel,alpha)]['action_region_gain'].append(region_gain_cache['action_positive'])
                for rname,prefix in [('negative_low_risk','negative'),('isolated_ldhn','isolated'),('low_haze','low_haze')]:
                    rmask=upsample(local_regions[rname],h,w,device)
                    pix=int(rmask.sum().item())
                    if pix:
                        per_lists[(sel,alpha)][f'{prefix}_touch_rate'].append(safe_div((rmask & smask).sum().item(), pix))
                        per_lists[(sel,alpha)][f'{prefix}_delta_l1'].append(float((residual_abs*rmask*smask*alpha).sum().item())/(pix*3))
        if (idx+1)%100==0: print(f'shadow_progress {idx+1}/{len(ds)} elapsed={time.time()-start:.1f}s', flush=True)

alpha_rows=[]
for sel in selectors:
    for alpha in ALPHAS:
        g=global_agg[(sel,alpha)]
        base_mse=g['base_energy']/g['elements']; shadow_mse=g['shadow_energy']/g['elements']
        row={'selector':sel,'alpha':alpha,'selected_coverage':safe_div(g['selected_pixels'],g['total_pixels']),'base_psnr':psnr_from_mse(base_mse),'shadow_psnr':psnr_from_mse(shadow_mse),'psnr_gain':psnr_from_mse(shadow_mse)-psnr_from_mse(base_mse),'removed_energy_fraction':safe_div(g['base_energy']-g['shadow_energy'],g['base_energy'])}
        alpha_rows.append(row)
write_csv(EVID/'shadow_modulation_by_alpha.csv', alpha_rows)
region_rows=[]
for sel in selectors:
    for alpha in ALPHAS:
        for r in regions:
            a=region_agg[(sel,alpha,r)]
            if a['elements']<=0: continue
            base_mse=a['base_energy']/a['elements']; shadow_mse=a['shadow_energy']/a['elements']
            region_rows.append({'selector':sel,'alpha':alpha,'region':r,'pixels':a['pixels'],'region_coverage':safe_div(a['pixels'],600*256*256),'selector_touch_rate':safe_div(a['selected_pixels'],a['pixels']),'base_region_psnr':psnr_from_mse(base_mse),'shadow_region_psnr':psnr_from_mse(shadow_mse),'region_psnr_gain':psnr_from_mse(shadow_mse)-psnr_from_mse(base_mse),'removed_energy_fraction':safe_div(a['base_energy']-a['shadow_energy'],a['base_energy']),'mean_abs_delta':safe_div(a['delta_abs_sum'],a['elements'])})
write_csv(EVID/'shadow_modulation_by_region.csv', region_rows)
per_rows=[]
for sel in selectors:
    for alpha in ALPHAS:
        row={'selector':sel,'alpha':alpha}
        for key,vals in per_lists[(sel,alpha)].items():
            st=qstats(vals)
            for sk,sv in st.items(): row[f'{key}_{sk}']=sv
        per_rows.append(row)
write_csv(EVID/'shadow_modulation_per_image_tail.csv', per_rows)
# Compact closeout and summary.
def find_alpha(selector, alpha):
    return next(r for r in alpha_rows if r['selector']==selector and abs(float(r['alpha'])-alpha)<1e-9)
def find_region(selector, alpha, region):
    return next(r for r in region_rows if r['selector']==selector and abs(float(r['alpha'])-alpha)<1e-9 and r['region']==region)
d7_03=find_alpha('d7c_fixed',0.3); den_03=find_alpha('density_matched',0.3); oracle_03=find_alpha('action_oracle',0.3)
d7_action_03=find_region('d7c_fixed',0.3,'action_positive')
d7_neg_03=find_region('d7c_fixed',0.3,'negative_low_risk')
d7_iso_03=find_region('d7c_fixed',0.3,'isolated_ldhn')
# Evidence-weighted automatic gate. Final route decision can still refine this.
gate_pass=(d7_03['psnr_gain'] > 0.05 and d7_action_03['region_psnr_gain'] > 0.10 and d7_neg_03['selector_touch_rate'] <= 0.005 and d7_iso_03['selector_touch_rate'] <= 0.03 and d7_03['psnr_gain'] > den_03['psnr_gain'])
closeout={'status':'COMPLETED_GATE_PASS' if gate_pass else 'COMPLETED_GATE_FAIL','decision_label':'V2H_B_SHADOW_MODULATION_PASS_AUTHORIZE_OOF_NOOP_REVIEW' if gate_pass else 'V2H_B_SHADOW_MODULATION_FAIL_PAUSE','locked_haze4k_test_usage':'none','D2':'not_run','F5':'not_run','v3':'not_run','RARM':'not_connected_or_trained','alpha_0_3':{'d7c_fixed':d7_03,'density_matched':den_03,'action_oracle':oracle_03,'d7c_action_region':d7_action_03,'d7c_negative_region':d7_neg_03,'d7c_isolated_region':d7_iso_03},'gate_pass':gate_pass,'next_recommended_stage':'v2h-C OOF stability and v2h-D FAM2 no-op equivalence review only' if gate_pass else 'pause; do not proceed to no-op/RARM'}
write_json(EVID/'shadow_modulation_closeout.json', closeout)
md=['# v2h-B Shadow-Modulation Upper Bound','',f"Status: `{closeout['status']}`",'',f"Decision label: `{closeout['decision_label']}`",'', 'Policy: diagnostic oracle shadow only. No training, no locked test, no D2/F5/v3/RARM.','', '## Alpha 0.3 Summary','', '| Selector | Global PSNR gain | Removed energy | Selected coverage |','| --- | ---: | ---: | ---: |', f"| D7c fixed | {d7_03['psnr_gain']:.6f} | {d7_03['removed_energy_fraction']:.6f} | {d7_03['selected_coverage']:.6f} |", f"| Density matched | {den_03['psnr_gain']:.6f} | {den_03['removed_energy_fraction']:.6f} | {den_03['selected_coverage']:.6f} |", f"| Action oracle | {oracle_03['psnr_gain']:.6f} | {oracle_03['removed_energy_fraction']:.6f} | {oracle_03['selected_coverage']:.6f} |", '', '## D7c Region Touch At Alpha 0.3','', '| Region | Touch rate | Region PSNR gain | Mean abs delta |','| --- | ---: | ---: | ---: |', f"| action_positive | {d7_action_03['selector_touch_rate']:.6f} | {d7_action_03['region_psnr_gain']:.6f} | {d7_action_03['mean_abs_delta']:.8f} |", f"| negative_low_risk | {d7_neg_03['selector_touch_rate']:.6f} | {d7_neg_03['region_psnr_gain']:.6f} | {d7_neg_03['mean_abs_delta']:.8f} |", f"| isolated_ldhn | {d7_iso_03['selector_touch_rate']:.6f} | {d7_iso_03['region_psnr_gain']:.6f} | {d7_iso_03['mean_abs_delta']:.8f} |", '', '## Decision','', closeout['next_recommended_stage'], '']
(EVID/'shadow_modulation_summary.md').write_text('\n'.join(md), encoding='utf-8')
readme=EVID/'README.md'
text=readme.read_text(encoding='utf-8') + '\n## v2h-B Result\n\n' + '\n'.join(md) + '\n'
readme.write_text(text, encoding='utf-8')
print('V2H_B_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "v2h_b_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2H_B_SHADOW_MODULATION_OK | tee -a "$STATUS"; else echo V2H_B_SHADOW_MODULATION_FAILED | tee -a "$STATUS"; fi
exit "$rc"
