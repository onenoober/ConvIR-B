#!/usr/bin/env python3
"""C10 formal 5x3 replay for fixed C7c local-alpha profile."""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics
from pathlib import Path
from typing import Any
import numpy as np, torch
import torch.nn.functional as F
import audit_haze4k_v20_c2_outputdiff_router as c2
from audit_haze4k_v20_c2b_multirule_router import fnum, write_csv
from audit_haze4k_v21_c7b_local_alpha_prototype import ALPHAS, C7B_STRONG_GATE, PatchTable, alpha_key, gate_pass, patch_rows_for_image_tensors, summarize_actual_rows
from audit_haze4k_v21_c7c_local_alpha_risk_tighten import PROFILES, choose_profile_from_grid, iter_patches, policy_grid

SEEDS=[3407,3411,2026]
MAX_SEED_SEVERE=60.0

def read_csv(path: Path)->list[dict[str,Any]]:
    with path.open(newline='',encoding='utf-8') as h: return list(csv.DictReader(h))

def seeded_fold_id(name:str, seed:int, folds:int=5)->int:
    return int(hashlib.sha1(f'{seed}:{name}'.encode()).hexdigest()[:8],16)%folds

def profile_def(name:str)->dict[str,Any]:
    for p in PROFILES:
        if p['profile']==name: return p
    raise KeyError(name)

def actions_for_patch_rows(policy_id: str, rows: list[dict[str, Any]]) -> np.ndarray:
    from audit_haze4k_v21_c7b_local_alpha_prototype import FEATURES, apply_policy
    image_rows=[{'name':'one','split':'one','A0_PSNR':0.0}]
    table_rows=[]
    for row in rows:
        rec={'name':'one','pixel_count':1}
        rec.update({feat:row[feat] for feat in FEATURES})
        for alpha in ALPHAS: rec[f'sse_{alpha_key(alpha)}']=0.0
        table_rows.append(rec)
    return apply_policy(PatchTable(table_rows,image_rows), policy_id)

def choose_seed_fold_policies(table:PatchTable, image_rows:list[dict[str,Any]], fixed_profile:str, args:argparse.Namespace):
    pdef=profile_def(fixed_profile)
    policies={}; fold_rows=[]
    for seed in SEEDS:
        folds=np.array([seeded_fold_id(str(r['name']),seed) for r in image_rows],dtype=np.int64)
        policies[seed]={}
        for fold in range(5):
            train_table=table.subset_images(folds!=fold)
            held_table=table.subset_images(folds==fold)
            grid=policy_grid(train_table,args.top_k,args.low_pool_limit,args.high_pool_limit)
            chosen=choose_profile_from_grid(grid,pdef)
            policies[seed][fold]=str(chosen['policy_id'])
            # proxy heldout metric for diagnostics only
            from audit_haze4k_v21_c7b_local_alpha_prototype import summarize_patch_actions, apply_policy
            hact=apply_policy(held_table,str(chosen['policy_id']))
            rec={'seed':seed,'fold':fold,'fixed_profile':fixed_profile,'policy_id':chosen['policy_id'],'train_proxy_severe':chosen.get('severe_loss_per_600'),'train_proxy_hard':chosen.get('hard_bottom25_dPSNR'),'train_proxy_positive':chosen.get('positive_ratio'),'heldout_count':len(held_table.image_rows)}
            rec.update(summarize_patch_actions(held_table,hact))
            fold_rows.append(rec)
            print(f'c10_policy seed={seed} fold={fold} candidates={len(grid)}', flush=True)
    return policies, fold_rows

def eval_actual(args:argparse.Namespace, policies:dict[int,dict[int,str]])->list[dict[str,Any]]:
    _loader,build_convir_net=c2.load_convir_builders(Path(args.convir_its_dir)); build_udpnet=c2.load_udpnet_builder(Path(args.udp_repo))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a0_model=c2.load_a0_model(build_convir_net,Path(args.a0_checkpoint),device)
    udp_model,_=c2.load_udpnet_model(build_udpnet,Path(args.official_checkpoint),device)
    rows=[]; factor=int(args.pad_factor)
    for split in args.splits:
        names=c2.load_split_names(Path(args.split_json),split); depth_split='train' if args.split_json else args.depth_split
        with torch.no_grad():
            for idx,image_name in enumerate(names):
                input_img,label_img,depth=c2.load_sample(Path(args.data_dir),Path(args.depth_cache_dir),image_name,depth_split)
                input_img=input_img.unsqueeze(0).to(device); label_img=label_img.unsqueeze(0).to(device); depth=depth.unsqueeze(0).to(device)
                h,w=input_img.shape[2],input_img.shape[3]; h_pad=((h+factor)//factor)*factor; w_pad=((w+factor)//factor)*factor
                padh=h_pad-h if h%factor!=0 else 0; padw=w_pad-w if w%factor!=0 else 0
                rgb_padded=F.pad(input_img,(0,padw,0,padh),'reflect'); depth_padded=F.pad(depth,(0,padw,0,padh),'reflect')
                a0_pred=c2.infer_one(a0_model,rgb_padded,h,w); udp_pred=c2.infer_one(udp_model,torch.cat([rgb_padded,depth_padded],dim=1),h,w)
                a0_psnr,a0_ssim=c2.metric_pair(a0_pred,label_img,(h_pad,w_pad)); p_rows=patch_rows_for_image_tensors(image_name,split,input_img,depth,a0_pred,udp_pred,int(args.patch_size)); residual=udp_pred-a0_pred
                for seed in SEEDS:
                    fold=seeded_fold_id(image_name,seed); policy_id=policies[seed][fold]; actions=actions_for_patch_rows(policy_id,p_rows)
                    pred=a0_pred.clone(); counts={a:0 for a in ALPHAS}
                    for (_pid,y,y2,x,x2),action in zip(iter_patches(h,w,int(args.patch_size)),actions,strict=False):
                        alpha=ALPHAS[int(action)]; counts[alpha]+=1; pred[...,y:y2,x:x2]=torch.clamp(a0_pred[...,y:y2,x:x2]+alpha*residual[...,y:y2,x:x2],0,1)
                    psnr,ssim=c2.metric_pair(pred,label_img,(h_pad,w_pad))
                    rec={'seed':seed,'name':image_name,'split':split,'fold':fold,'policy_id':policy_id,'A0_PSNR':a0_psnr,'A0_SSIM':a0_ssim,'formal_PSNR':psnr,'formal_SSIM':ssim,'dPSNR':psnr-a0_psnr,'dSSIM':ssim-a0_ssim,'patch_count':sum(counts.values())}
                    for alpha in ALPHAS: rec[f'patch_action_fraction_{alpha_key(alpha)}']=counts[alpha]/max(1,rec['patch_count'])
                    rows.append(rec)
                if (idx+1)%args.print_freq==0: print(f'c10_actual {split} {idx+1}/{len(names)} rows={len(rows)}', flush=True)
                if args.max_images and idx+1>=args.max_images: break
    return rows

def mean_std(vals):
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals)>1 else 0.0)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--patch_rows',type=Path,required=True); ap.add_argument('--image_rows',type=Path,required=True); ap.add_argument('--fixed_profile',default='riskcap36_no075')
    ap.add_argument('--convir_its_dir',required=True); ap.add_argument('--udp_repo',required=True); ap.add_argument('--data_dir',required=True); ap.add_argument('--depth_cache_dir',required=True); ap.add_argument('--a0_checkpoint',required=True); ap.add_argument('--official_checkpoint',required=True); ap.add_argument('--split_json',required=True)
    ap.add_argument('--splits',nargs='+',default=['val_regular','val_hard']); ap.add_argument('--depth_split',default='test'); ap.add_argument('--pad_factor',type=int,default=32); ap.add_argument('--patch_size',type=int,default=128); ap.add_argument('--print_freq',type=int,default=50); ap.add_argument('--max_images',type=int,default=0)
    ap.add_argument('--top_k',type=int,default=900); ap.add_argument('--low_pool_limit',type=int,default=80); ap.add_argument('--high_pool_limit',type=int,default=120); ap.add_argument('--out_dir',type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    patch_rows=read_csv(args.patch_rows); image_rows=read_csv(args.image_rows); table=PatchTable(patch_rows,image_rows)
    policies,fold_rows=choose_seed_fold_policies(table,image_rows,args.fixed_profile,args)
    actual=eval_actual(args,policies)
    seed_rows=[]
    for seed in SEEDS:
        rec={'seed':seed,**summarize_actual_rows([r for r in actual if int(r['seed'])==seed])}; rec['strong_gate_pass']=gate_pass(rec,C7B_STRONG_GATE); seed_rows.append(rec)
    metrics=['mean_dPSNR','hard_bottom25_dPSNR','easy_top25_dPSNR','dSSIM','positive_ratio','nonnegative_ratio','severe_loss_per_600','selected_precision']
    agg={'seed_count':len(seed_rows),'fold_count':len(fold_rows)}
    for m in metrics:
        mean,sd=mean_std([fnum(r[m]) for r in seed_rows]); agg[f'{m}_mean']=mean; agg[f'{m}_std']=sd
    agg['max_seed_severe_loss_per_600']=max(fnum(r['severe_loss_per_600']) for r in seed_rows); agg['all_seed_strong_gate_pass']=all(bool(r['strong_gate_pass']) for r in seed_rows)
    strong_mean={ 'mean_dPSNR':agg['mean_dPSNR_mean'], 'hard_bottom25_dPSNR':agg['hard_bottom25_dPSNR_mean'], 'easy_top25_dPSNR':agg['easy_top25_dPSNR_mean'], 'dSSIM':agg['dSSIM_mean'], 'positive_ratio':agg['positive_ratio_mean'], 'severe_loss_per_600':agg['severe_loss_per_600_mean'] }
    agg['strong_formal_gate_pass']=gate_pass(strong_mean,C7B_STRONG_GATE) and agg['max_seed_severe_loss_per_600']<=MAX_SEED_SEVERE and agg['all_seed_strong_gate_pass']
    decision='C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT' if agg['strong_formal_gate_pass'] else 'C10_FORMAL_5X3_FAIL_NO_LOCKED'
    write_csv(args.out_dir/'v21_c10_formal_fold_proxy.csv',fold_rows,sorted({k for r in fold_rows for k in r})); write_csv(args.out_dir/'v21_c10_formal_per_image.csv',actual,sorted({k for r in actual for k in r})); write_csv(args.out_dir/'v21_c10_formal_seed_summary.csv',seed_rows,sorted({k for r in seed_rows for k in r}))
    payload={'route':'Haze4K-v2.1 SEG-Mix','phase':'C10 Formal 5x3 Fixed Profile Replay','locked_test_touched':False,'fixed_profile':args.fixed_profile,'seeds':SEEDS,'strong_gate':C7B_STRONG_GATE,'max_seed_severe_gate':MAX_SEED_SEVERE,'aggregate':agg,'seed_rows':seed_rows,'decision':decision}
    (args.out_dir/'v21_c10_formal_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Haze4K v2.1 C10 Formal 5x3 Fixed Profile Replay','',f'Decision: `{decision}`','',f'Fixed profile: `{args.fixed_profile}`','', '## Aggregate','']
    for k,v in agg.items(): lines.append(f'- `{k}`: `{v}`')
    lines += ['', '## Seed Summary', '']
    for r in seed_rows: lines.append(f"- seed `{r['seed']}`: mean `{r['mean_dPSNR']}`, hard `{r['hard_bottom25_dPSNR']}`, easy `{r['easy_top25_dPSNR']}`, positive `{r['positive_ratio']}`, severe `{r['severe_loss_per_600']}`, strong `{r['strong_gate_pass']}`")
    lines += ['', 'Locked one-shot is authorized only by `C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT`.']
    (args.out_dir/'v21_c10_formal_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'V21_C10_FORMAL_OK decision={decision} out={args.out_dir}', flush=True); return 0
if __name__=='__main__': raise SystemExit(main())
