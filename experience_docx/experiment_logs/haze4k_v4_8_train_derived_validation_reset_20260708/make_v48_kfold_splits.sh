#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt
PY=$BASE/envs/convir-cu121/bin/python
ROUTE_ID=haze4k_v4_8_train_derived_validation_reset_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
DATA=$BASE/datasets/Haze4K/Haze4K
STATUS=$EVID/status.txt
LOG=$EVID/make_v48_kfold_splits.log
mkdir -p "$EVID/splits"
{
  echo "split_start v48_kfold $(date --iso-8601=seconds)"
  echo "state=PREFLIGHT_RUNNING"
  echo "data_train_haze=$DATA/train/haze"
  echo "data_train_gt=$DATA/train/gt"
  echo "locked_test_policy=no test directory access"
} | tee -a "$STATUS"
set +e
"$PY" - <<'PYCODE' > "$LOG" 2>&1
import csv, json, math, os, random, statistics, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
BASE=Path('/sda/home/wangyuxin/ConvIR-B')
WORK=BASE/'repos'/'ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt'
ROUTE_ID='haze4k_v4_8_train_derived_validation_reset_20260708'
EVID=WORK/'experience_docx'/'experiment_logs'/ROUTE_ID
DATA=BASE/'datasets'/'Haze4K'/'Haze4K'
HAZE=DATA/'train'/'haze'
GT=DATA/'train'/'gt'
SPLITS=EVID/'splits'
SPLITS.mkdir(parents=True, exist_ok=True)
K=5; SEED=3407; IMG_EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
def label_path_for(image_name):
    stem, ext=os.path.splitext(image_name); candidates=[image_name]
    if '_' in stem:
        candidates += [stem.split('_')[0]+ext, stem.split('_')[0]+'.png']
    for c in candidates:
        p=GT/c
        if p.is_file(): return p
    return None
def load_rgb(path): return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)/255.0
def texture_proxy(arr):
    dy=np.abs(arr[1:,:,:]-arr[:-1,:,:]).mean() if arr.shape[0]>1 else 0.0
    dx=np.abs(arr[:,1:,:]-arr[:,:-1,:]).mean() if arr.shape[1]>1 else 0.0
    return float(dx+dy)
def parse_haze_params(name):
    parts=Path(name).stem.split('_'); vals=[]
    for token in parts[1:3]:
        try: vals.append(float(token))
        except Exception: vals.append(float('nan'))
    while len(vals)<2: vals.append(float('nan'))
    return vals[0], vals[1]
def qbin(values, value, bins=4):
    vals=sorted(v for v in values if not math.isnan(v))
    if not vals or math.isnan(value): return 0
    rank=sum(v<=value for v in vals)-1; rank=max(0,min(rank,len(vals)-1))
    return min(bins-1, int(rank*bins/len(vals)))
start=time.time(); image_names=sorted(p.name for p in HAZE.iterdir() if p.suffix.lower() in IMG_EXT and p.is_file())
rows=[]; missing_label=[]; shape_mismatch=[]
for idx,name in enumerate(image_names,1):
    lp=label_path_for(name)
    if lp is None: missing_label.append(name); continue
    hazy=load_rgb(HAZE/name); gt=load_rgb(lp)
    if hazy.shape!=gt.shape: shape_mismatch.append({'image_id':name,'hazy_shape':list(hazy.shape),'gt_shape':list(gt.shape)}); continue
    p1,p2=parse_haze_params(name); stem=Path(name).stem; base_id=stem.split('_')[0]
    rows.append({'image_id':name,'base_id':base_id,'label':lp.name,'haze_param_1':p1,'haze_param_2':p2,'input_gt_l1':float(np.abs(hazy-gt).mean()),'input_dark_channel_mean':float(np.min(hazy,axis=2).mean()),'input_brightness_mean':float(hazy.mean()),'input_saturation_proxy':float((hazy.max(axis=2)-hazy.min(axis=2)).mean()),'gt_texture_proxy':texture_proxy(gt),'hazy_texture_proxy':texture_proxy(hazy)})
    if idx % 250 == 0: print(f'proxy_progress seen={idx} paired={len(rows)}', flush=True)
groups=defaultdict(list)
for r in rows: groups[r['base_id']].append(r)
proxy_keys=['haze_param_1','haze_param_2','input_gt_l1','input_dark_channel_mean','input_brightness_mean','input_saturation_proxy','gt_texture_proxy','hazy_texture_proxy']
group_rows=[]
for base_id,items in groups.items():
    g={'base_id':base_id,'count':len(items),'image_ids':[r['image_id'] for r in items]}
    for key in proxy_keys:
        vals=[r[key] for r in items if not math.isnan(r[key])]
        g[key]=float(statistics.mean(vals)) if vals else float('nan')
    group_rows.append(g)
all_values={key:[g[key] for g in group_rows if not math.isnan(g[key])] for key in proxy_keys}
rng=random.Random(SEED); strata=defaultdict(list)
for g in group_rows:
    key=(qbin(all_values['input_gt_l1'],g['input_gt_l1'],4),qbin(all_values['input_saturation_proxy'],g['input_saturation_proxy'],4),qbin(all_values['input_dark_channel_mean'],g['input_dark_channel_mean'],4),qbin(all_values['gt_texture_proxy'],g['gt_texture_proxy'],4),qbin(all_values['haze_param_1'],g['haze_param_1'],3),qbin(all_values['haze_param_2'],g['haze_param_2'],3))
    g['stratum']='|'.join(map(str,key)); strata[g['stratum']].append(g)
fold_groups=[[] for _ in range(K)]
for key in sorted(strata):
    items=list(strata[key]); rng.shuffle(items); start_fold=abs(hash(key))%K
    for offset,g in enumerate(items): fold_groups[(start_fold+offset)%K].append(g)
fold_image_ids=[]; all_id_set=set(r['image_id'] for r in rows)
for k,gs in enumerate(fold_groups):
    ids=sorted(img for g in gs for img in g['image_ids']); fold_image_ids.append(ids)
    (SPLITS/f'fold_{k}_val.txt').write_text('\n'.join(ids)+'\n', encoding='utf-8')
    (SPLITS/f'fold_{k}_train.txt').write_text('\n'.join(sorted(all_id_set-set(ids)))+'\n', encoding='utf-8')
seen=set(); leak=[]; base_fold={}; base_leaks=[]
for k,ids in enumerate(fold_image_ids):
    for x in ids:
        if x in seen: leak.append(x)
        seen.add(x); b=Path(x).stem.split('_')[0]
        if b in base_fold and base_fold[b]!=k: base_leaks.append({'base_id':b,'fold_a':base_fold[b],'fold_b':k})
        base_fold[b]=k
balance_rows=[]
for k,ids in enumerate(fold_image_ids):
    s=set(ids); subset=[r for r in rows if r['image_id'] in s]
    out={'fold':k,'val_count':len(subset),'train_count':len(rows)-len(subset),'group_count':len(fold_groups[k])}
    for key in proxy_keys:
        vals=[r[key] for r in subset if not math.isnan(r[key])]
        out[f'{key}_mean']=float(statistics.mean(vals)) if vals else float('nan')
        out[f'{key}_p25']=float(np.percentile(vals,25)) if vals else float('nan')
        out[f'{key}_p75']=float(np.percentile(vals,75)) if vals else float('nan')
    balance_rows.append(out)
with (EVID/'v48_kfold_split_balance.csv').open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(balance_rows[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(balance_rows)
manifest={'route_id':ROUTE_ID,'k':K,'seed':SEED,'source':'Haze4K train/haze and train/gt only','locked_test_touched':False,'test_split_enumerated':False,'train_haze_image_count_seen':len(image_names),'paired_train_count':len(rows),'missing_label_count':len(missing_label),'missing_label_examples':missing_label[:20],'shape_mismatch_count':len(shape_mismatch),'shape_mismatch_examples':shape_mismatch[:5],'base_group_count':len(groups),'stratum_count':len(strata),'proxy_keys':proxy_keys,'folds':[{'fold':k,'val_count':len(ids),'train_count':len(rows)-len(ids),'split_train':str(SPLITS/f'fold_{k}_train.txt'),'split_val':str(SPLITS/f'fold_{k}_val.txt')} for k,ids in enumerate(fold_image_ids)],'runtime_sec':time.time()-start}
(EVID/'v48_kfold_split_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
audit={'route_id':ROUTE_ID,'pass':len(rows)==3000 and not leak and not base_leaks and len(seen)==len(all_id_set),'paired_train_count':len(rows),'total_val_assignments':sum(len(ids) for ids in fold_image_ids),'unique_val_assignments':len(seen),'duplicate_val_assignments':leak[:20],'base_group_leaks':base_leaks[:20],'fold_val_counts':[len(ids) for ids in fold_image_ids],'fold_train_counts':[len(rows)-len(ids) for ids in fold_image_ids],'locked_test_touched':False,'test_split_enumerated':False}
(EVID/'v48_kfold_no_leakage_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True)+'\n', encoding='utf-8')
with (EVID/'v48_kfold_pairing_rows.csv').open('w', newline='', encoding='utf-8') as f:
    fields=['image_id','base_id','label']+proxy_keys; w=csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in rows]
print(json.dumps({'manifest':manifest,'audit':audit}, indent=2, sort_keys=True))
if not audit['pass']: raise SystemExit('split audit failed')
PYCODE
rc=$?
set -e
if [ "$rc" -eq 0 ]; then echo "state=PREFLIGHT_SPLITS_OK" | tee -a "$STATUS"; echo "split_done rc=0 v48_kfold $(date --iso-8601=seconds)" | tee -a "$STATUS"; echo V48_KFOLD_SPLITS_OK; else echo "state=PREFLIGHT_FAILED_ENGINEERING" | tee -a "$STATUS"; echo "split_done rc=$rc v48_kfold $(date --iso-8601=seconds)" | tee -a "$STATUS"; echo V48_KFOLD_SPLITS_FAILED; fi
exit "$rc"
