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
LOG=$EVID/v2g_g2b_oracle_gain.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "g2b_oracle_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, os, subprocess, time, importlib.util, sys
from pathlib import Path
from PIL import Image
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
v2f=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)
v2e=v2f.v2e
d7c=v2e.d7c
v2d=v2e.v2d
v2b=v2d.v2b

EVID.mkdir(parents=True, exist_ok=True)

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
        fieldnames=fields
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')

def psnr_from_mse(mse):
    mse=float(mse)
    return 99.0 if mse <= 0 else -10.0*math.log10(mse)

def q(vals, p):
    vals=sorted([float(v) for v in vals if v is not None and math.isfinite(float(v))])
    if not vals: return math.nan
    return vals[min(len(vals)-1, max(0, int(round((len(vals)-1)*p))))]

def stats(vals):
    vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals: return {'n':0}
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(vals,.5),'p75':q(vals,.75),'p90':q(vals,.9),'p95':q(vals,.95),'max':max(vals)}

def load_trans(path, h, w):
    img=Image.open(path).convert('L')
    if img.size != (w,h):
        img=img.resize((w,h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)/255.0

def trans_path_for(name):
    p=v2b.v2.label_path(DATA/'train'/'trans', name)
    s=str(p)
    if '/test/' in s.lower() or 'locked' in s.lower():
        raise RuntimeError(f'Forbidden trans path for {name}: {s}')
    return p

def upsample_mask(mask64, h, w, device):
    m=torch.as_tensor(mask64.astype(np.float32), device=device).view(1,1,*mask64.shape)
    return (F.interpolate(m, size=(h,w), mode='nearest') >= 0.5)

print('V2G_G2B_PY_START')
for p in [DATA/'train'/'haze', DATA/'train'/'gt', DATA/'train'/'trans', A0, SPLIT, V2_THRESH, V2B_THRESH, D3, D7C_TOPK]:
    if not Path(p).exists():
        raise FileNotFoundError(str(p))
    sp=str(p).lower()
    if '/test/' in sp or 'locked' in sp:
        # DATA root contains test sibling, but required runtime paths must not point inside it.
        raise RuntimeError(f'Forbidden runtime path: {p}')

split=json.loads(SPLIT.read_text(encoding='utf-8'))
val_names=sorted(split['splits']['val_inner'])
train_names=sorted(split['splits']['train_inner'])
if len(train_names)!=2400 or len(val_names)!=600:
    raise RuntimeError(f'Unexpected split sizes: {len(train_names)}/{len(val_names)}')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, 'val_images', len(val_names), flush=True)
model=v2b.load_model(A0, device)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
density_head=v2d.load_density_head(D3, device)
topk_head=v2e.load_head(D7C_TOPK, device)
for m in [density_head, topk_head]:
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)

density_stats=json.loads(V2_THRESH.read_text(encoding='utf-8'))
target_info=json.loads(V2B_THRESH.read_text(encoding='utf-8'))
q20=float(target_info['quantile']['q20'])
q33=float(target_info['quantile']['q33'])
q66=float(target_info['quantile']['q66'])
q80=float(target_info['quantile']['q80'])
density_q33=float(density_stats['density']['q33'])
density_q66=float(density_stats['density']['q66'])
# Reuse v2f-computed global gradient threshold to keep subtype contract identical.
f1=json.loads((ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/ldhn_target_autopsy_summary.json').read_text(encoding='utf-8'))
grad_p90=float(f1['density_gradient_p90'])
threshold=0.5773006677627563
map_grid=64
blur_radii=[5,9,15]
near_haze_radius=3
blur_kernel=9

def make_record(name, hazy, gt):
    hazy=hazy.unsqueeze(0).to(device)
    gt=gt.unsqueeze(0).to(device)
    padded,h,w=v2b.v2.pad32(hazy)
    a0, context=d7c.convir_a0_context(model, density_head, padded)
    a0=a0[:,:,:h,:w]
    context=context[:,:,:h,:w]
    pred,_=d7c.predict_head(topk_head, context)
    raw_need=v2b.v2.raw_need(a0, gt, blur_kernel)
    target=v2b.make_target(raw_need, target_info, 'quantile')
    density=v2b.v2.normalize(v2b.v2.raw_density(hazy, gt, blur_kernel), density_stats['density']['raw_p1'], density_stats['density']['raw_p99'])
    target_map=v2f.pool_map(target, map_grid)
    density_map=v2f.pool_map(density, map_grid)
    pred_map=v2f.pool_map(pred, map_grid)
    blur_maps=[]
    for radius in blur_radii:
        rn=v2b.v2.raw_need(a0, gt, radius)
        tb=v2b.make_target(rn, target_info, 'quantile')
        blur_maps.append(v2f.pool_map(tb, map_grid))
    return a0, gt, target_map, density_map, pred_map, blur_maps, h, w

def masks_for(target, density, pred, blur_maps):
    stable66=np.logical_and.reduce([tb >= q66 for tb in blur_maps])
    stable80=np.logical_and.reduce([tb >= q80 for tb in blur_maps])
    density_low=density <= density_q33
    ldhn=density_low & (target >= q66)
    boundary_band=density_low & (target >= q66) & (target < q80)
    unstable=ldhn & (~stable66)
    local_high_density=v2f.local_max_2d((density >= density_q66).astype(np.float32), near_haze_radius) > 0
    grad=v2f.gradient_mag(density)
    adjacent=ldhn & (local_high_density | (grad >= grad_p90))
    core=(density_low & (target >= q80)) & stable66
    boundary=ldhn & (boundary_band | unstable)
    isolated=ldhn & (~adjacent)
    pred_high=pred >= threshold
    ldln=density_low & (target <= q33)
    return {
        'all_ldhn': ldhn,
        'ldhn_core': core,
        'ldhn_boundary': boundary,
        'ldhn_adjacent_to_haze': adjacent,
        'ldhn_isolated': isolated,
        'ldhn_unstable': unstable,
        'missed_ldhn': ldhn & (~pred_high),
        'missed_adjacent_ldhn': adjacent & (~pred_high),
        'missed_isolated_ldhn': isolated & (~pred_high),
        'd7c_pred_high': pred_high,
        'd7c_ldhn_hit': ldhn & pred_high,
        'ldln_false_tail_pred_high': ldln & pred_high,
        'low_density_low_need': ldln,
    }

categories=['all_ldhn','ldhn_core','ldhn_boundary','ldhn_adjacent_to_haze','ldhn_isolated','ldhn_unstable','missed_ldhn','missed_adjacent_ldhn','missed_isolated_ldhn','d7c_pred_high','d7c_ldhn_hit','ldln_false_tail_pred_high','low_density_low_need']
agg={c:{'mask_pixels':0,'total_pixels':0,'removed_energy':0.0,'base_energy':0.0,'total_elements':0,'trans_sum':0.0,'veil_sum':0.0,'images_with_support':0,'per_image_gain':[],'per_image_energy_fraction':[],'per_image_coverage':[],'per_image_trans_mean':[]} for c in categories}
per_rows=[]
path_rows=[]
dataset=v2e.Haze4KPairDataset(val_names, DATA, max_items=0, seed=3407)
start=time.time()
with torch.no_grad():
    for idx,(name,hazy,gt_cpu) in enumerate(dataset):
        # Enforce train split only for all paired assets.
        haze_path=DATA/'train'/'haze'/name
        gt_path=v2b.v2.label_path(DATA/'train'/'gt', name)
        tr_path=trans_path_for(name)
        for p in [haze_path,gt_path,tr_path]:
            s=str(p).lower()
            if '/test/' in s or 'locked' in s:
                raise RuntimeError(f'Forbidden path for {name}: {p}')
        a0, gt, target_map, density_map, pred_map, blur_maps, h, w = make_record(name, hazy, gt_cpu)
        masks=masks_for(target_map, density_map, pred_map, blur_maps)
        residual=(a0-gt).float()
        residual_sq=residual.pow(2)
        base_energy=float(residual_sq.sum().item())
        total_elements=int(residual_sq.numel())
        base_mse=base_energy/max(total_elements,1)
        base_psnr=psnr_from_mse(base_mse)
        trans=load_trans(tr_path, h, w)
        total_pix=h*w
        path_rows.append({'name':name,'haze_path':str(haze_path),'gt_path':str(gt_path),'trans_path':str(tr_path),'height':h,'width':w})
        for cat,mask64 in masks.items():
            mask_t=upsample_mask(mask64, h, w, device)
            pix=int(mask_t.sum().item())
            removed=float((residual_sq * mask_t).sum().item()) if pix else 0.0
            oracle_mse=(base_energy-removed)/max(total_elements,1)
            gain=psnr_from_mse(oracle_mse)-base_psnr if pix else 0.0
            mask_np=mask_t.squeeze().detach().cpu().numpy().astype(bool)
            trans_mean=float(trans[mask_np].mean()) if pix else math.nan
            veil_mean=float((1.0-trans[mask_np]).mean()) if pix else math.nan
            energy_frac=removed/base_energy if base_energy>0 else math.nan
            coverage=pix/max(total_pix,1)
            rec={
                'name':name,'category':cat,'mask_pixels':pix,'coverage':coverage,
                'base_psnr':base_psnr,'oracle_psnr_gain':gain,'removed_energy_fraction':energy_frac,
                'trans_mean':trans_mean,'veil_mean':veil_mean,
            }
            per_rows.append(rec)
            a=agg[cat]
            a['mask_pixels']+=pix; a['total_pixels']+=total_pix; a['removed_energy']+=removed; a['base_energy']+=base_energy; a['total_elements']+=total_elements
            if pix:
                a['images_with_support']+=1
                a['trans_sum']+=float(trans[mask_np].sum())
                a['veil_sum']+=float((1.0-trans[mask_np]).sum())
                a['per_image_gain'].append(gain)
                a['per_image_energy_fraction'].append(energy_frac)
                a['per_image_coverage'].append(coverage)
                a['per_image_trans_mean'].append(trans_mean)
        if (idx+1)%50==0:
            print(f'g2b_progress {idx+1}/{len(dataset)} elapsed={time.time()-start:.1f}s', flush=True)

summary_rows=[]
for cat in categories:
    a=agg[cat]
    base_mse=a['base_energy']/max(a['total_elements'],1)
    oracle_mse=(a['base_energy']-a['removed_energy'])/max(a['total_elements'],1)
    summary_rows.append({
        'category':cat,
        'images_with_support':a['images_with_support'],
        'mask_pixels':a['mask_pixels'],
        'coverage':a['mask_pixels']/max(a['total_pixels'],1),
        'removed_energy_fraction_global':a['removed_energy']/max(a['base_energy'],1e-12),
        'global_psnr_gain_if_oracle_corrected':psnr_from_mse(oracle_mse)-psnr_from_mse(base_mse),
        'per_image_psnr_gain_mean':stats(a['per_image_gain']).get('mean',math.nan),
        'per_image_psnr_gain_p50':stats(a['per_image_gain']).get('p50',math.nan),
        'per_image_psnr_gain_p90':stats(a['per_image_gain']).get('p90',math.nan),
        'per_image_energy_fraction_p50':stats(a['per_image_energy_fraction']).get('p50',math.nan),
        'per_image_coverage_p50':stats(a['per_image_coverage']).get('p50',math.nan),
        'trans_mean_weighted':a['trans_sum']/a['mask_pixels'] if a['mask_pixels'] else math.nan,
        'veil_mean_weighted':a['veil_sum']/a['mask_pixels'] if a['mask_pixels'] else math.nan,
        'per_image_trans_mean_p50':stats(a['per_image_trans_mean']).get('p50',math.nan),
        'actionability_hint':{
            'all_ldhn':'diagnostic_overbroad_reference',
            'ldhn_adjacent_to_haze':'candidate_actionable_positive',
            'missed_adjacent_ldhn':'candidate_missed_actionable_positive',
            'ldhn_isolated':'ignore_or_abstain_until_action_gain_justifies',
            'missed_isolated_ldhn':'likely_non_actionable_or_uncertain',
            'ldln_false_tail_pred_high':'false_tail_safety_risk_probe',
            'low_density_low_need':'confident_low_risk_negative_pool'
        }.get(cat,'diagnostic')
    })

write_csv(EVID/'ldhn_oracle_gain_by_region.csv', summary_rows)
write_csv(EVID/'ldhn_oracle_gain_per_image_by_region.csv', per_rows)
write_csv(EVID/'ldhn_missed_region_oracle_gain.csv', [r for r in summary_rows if 'missed' in r['category']])
write_csv(EVID/'ldhn_false_tail_oracle_gain.csv', [r for r in summary_rows if r['category'] in ['ldln_false_tail_pred_high','low_density_low_need','d7c_pred_high']])
write_csv(EVID/'ldhn_physics_consistency_by_bin.csv', summary_rows)
write_csv(EVID/'v2g_g2b_path_contract_resolved.csv', path_rows[:20])

contract={
    'status':'resolved',
    'policy':'val_inner only; train split Haze4K assets only; no locked/test path read',
    'data_dir':str(DATA),
    'split_json':str(SPLIT),
    'checkpoint':str(A0),
    'v2_thresholds':str(V2_THRESH),
    'v2b_thresholds':str(V2B_THRESH),
    'density_artifact':str(D3),
    'd7c_topk_artifact':str(D7C_TOPK),
    'val_inner_count':len(val_names),
    'train_inner_count':len(train_names),
    'map_grid':map_grid,
    'blur_kernel':blur_kernel,
    'stability_blur_radii':blur_radii,
    'candidate_threshold':threshold,
    'density_q33':density_q33,
    'density_q66':density_q66,
    'need_q33':q33,
    'need_q66':q66,
    'need_q80':q80,
    'density_gradient_p90_reused_from_v2f':grad_p90,
    'locked_haze4k_test_usage':'none'
}
write_json(EVID/'g2b_oracle_path_contract.json', contract)
# Pull important rows into JSON by category for easier reading.
summary_by_cat={r['category']:r for r in summary_rows}
closeout={
    'status':'COMPLETED_G2B_ORACLE_GAIN_DIAGNOSTIC',
    'locked_haze4k_test_usage':'none',
    'D2':'not_run','RARM':'not_connected_or_trained','v3':'not_run','F5':'not_run',
    'val_inner_images':len(val_names),
    'key_rows':{k:summary_by_cat[k] for k in ['all_ldhn','ldhn_adjacent_to_haze','ldhn_isolated','missed_ldhn','missed_adjacent_ldhn','missed_isolated_ldhn','ldln_false_tail_pred_high','low_density_low_need']},
    'interpretation':{
        'primary':'Oracle gain should be interpreted with transmission/veil stats: high gain in high-transmission isolated LDHN indicates residual value but weak haze-actionability.',
        'next':'Use these numbers to define actionable/ignore/negative target semantics before any new head training.'
    },
    'elapsed_sec':time.time()-start
}
write_json(EVID/'v2g_g2b_oracle_gain_closeout.json', closeout)
# Update oracle availability and main closeout/summary without deleting earlier fields.
write_json(EVID/'ldhn_oracle_gain_availability.json', {
    'oracle_gain_computed': True,
    'path_contract': str(EVID/'g2b_oracle_path_contract.json'),
    'summary_csv': str(EVID/'ldhn_oracle_gain_by_region.csv'),
    'per_image_csv': str(EVID/'ldhn_oracle_gain_per_image_by_region.csv'),
    'locked_haze4k_test_usage':'none'
})
summary_md=f'''# v2g G2b Oracle-Gain Diagnostic Summary\n\nStatus: `COMPLETED_G2B_ORACLE_GAIN_DIAGNOSTIC`\n\nPolicy: val_inner only; locked Haze4K test, D2, RARM, v3, and F5 were not run.\n\nPath contract: `g2b_oracle_path_contract.json`.\n\n## Key Rows\n\n| Region | Coverage | Removed residual energy | Oracle PSNR gain | Trans mean | Veil mean |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'''
for cat in ['all_ldhn','ldhn_adjacent_to_haze','ldhn_isolated','missed_ldhn','missed_adjacent_ldhn','missed_isolated_ldhn','ldln_false_tail_pred_high','low_density_low_need']:
    r=summary_by_cat[cat]
    summary_md += f"| {cat} | {r['coverage']:.6f} | {r['removed_energy_fraction_global']:.6f} | {r['global_psnr_gain_if_oracle_corrected']:.6f} | {r['trans_mean_weighted']:.6f} | {r['veil_mean_weighted']:.6f} |\n"
summary_md += '''\n## Interpretation\n\nThese rows separate action value from haze actionability. A region can remove residual energy yet still have high transmission/low veil, which means it should not automatically become an RARM-positive target.\n'''
(EVID/'v2g_g2b_oracle_gain_summary.md').write_text(summary_md, encoding='utf-8')
# Append to overall summary.
with open(EVID/'v2g_overall_result_summary.md','a',encoding='utf-8') as f:
    f.write('\n## G2b Oracle Gain\n\n')
    f.write(summary_md)
print('V2G_G2B_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "g2b_oracle_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2G_G2B_ORACLE_OK; else echo V2G_G2B_ORACLE_FAILED; fi
exit "$rc"
