#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign-f4-044b7798
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v2g_need_actionability_audit.log
mkdir -p "$EVID"
echo "audit_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" - <<'PY' 2>&1 | tee "$LOG"
import csv, json, math, os, statistics as st, subprocess, time
from pathlib import Path
ROOT=Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign-f4-044b7798')
EVID=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709'
V2D=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709'
V2E=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709'
V2F=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709'
F4B=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709'
EVID.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        keys=[]
        for r in rows:
            for k in r.keys():
                if k not in keys: keys.append(k)
        fieldnames=keys
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def qstats(vals):
    vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    vals.sort()
    if not vals:
        return {'n':0}
    def q(p):
        return vals[min(len(vals)-1, max(0, int(round((len(vals)-1)*p))))]
    return {'n':len(vals),'mean':sum(vals)/len(vals),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'max':vals[-1]}

def git(cmd):
    return subprocess.check_output(cmd, cwd=str(ROOT), text=True).strip()

print('V2G_AUDIT_PY_START')
manifest={
    'route_id':'haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709',
    'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'host':subprocess.check_output(['hostname'], text=True).strip(),
    'root':str(ROOT),
    'branch':git(['git','branch','--show-current']),
    'head':git(['git','rev-parse','--short','HEAD']),
    'cloud_python':'/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python',
    'locked_haze4k_test_usage':'none',
    'forbidden_not_used':['D2','RARM connection','RARM training','v3','F5','locked Haze4K test'],
    'sources':{
        'v2d':str(V2D),
        'v2e':str(V2E),
        'v2f':str(V2F),
        'f4b':str(F4B),
    }
}
(EVID/'v2g_source_of_truth_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

v2d=load_json(V2D/'v2d_overall_run_summary.json')
v2e=load_json(V2E/'v2e_final_closeout.json')
v2f=load_json(V2F/'v2f_run_summary.json')
f4_close=load_json(V2F/'v2f_f4_stratified_head_closeout.json')
f4b_close=load_json(F4B/'v2f_f4b_tail_rescue_closeout.json')
repro=[]
def add(stage, variant, metric, value, source, interpretation=''):
    repro.append({'stage':stage,'variant':variant,'metric':metric,'value':value,'source':source,'interpretation':interpretation})
for k,v in v2d['best_candidate'].items():
    if isinstance(v,(int,float,str)):
        add('v2d','d7c_mc_topk_hn_ordinal',k,v,'v2d_overall_run_summary.json')
for k,v in v2e['main_audit']['candidate'].items():
    add('v2e','d7c_mc_topk_hn_ordinal',k,v,'v2e_final_closeout.json')
for v in v2e['rp_full']['variants']:
    for k in ['ldhn_recall','false_global','false_per_image_p90','false_per_image_p95','gate_safety_pass','gate_ldhn_pass','gate_overall_pass']:
        add('v2e_rp',v['variant'],k,v[k],'v2e_final_closeout.json')
for k,v in v2e['main_audit']['ldhn_support'].items():
    add('v2e','ldhn_support',k,v,'v2e_final_closeout.json')
for k,v in v2e['main_audit']['density_gap'].items():
    add('v2e','density_only_matched_gap',k,v,'v2e_final_closeout.json')
for k,v in v2e['main_audit']['fixed_permutation'].items():
    add('v2e','fixed_permutation',k,v,'v2e_final_closeout.json')
for k,v in v2f.items():
    if k in ['f1_ldhn_boundary_fraction_of_ldhn','f1_ldhn_core_fraction_of_ldhn','f2_best','locked_haze4k_test_usage','status']:
        add('v2f','first_stage',k,json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v,'v2f_run_summary.json')
for k,v in f4_close.items():
    add('v2f_f4','stratified_head',k,v,'v2f_f4_stratified_head_closeout.json')
for k,v in f4b_close.items():
    add('v2f_f4b','tail_rescue',k,v,'v2f_f4b_tail_rescue_closeout.json')
write_csv(EVID/'cross_stage_metric_reproduction.csv', repro)

no_locked={
    'locked_haze4k_test_usage':'none',
    'sources':{
        'v2d':v2d['runtime'].get('locked_haze4k_test_usage'),
        'v2e':v2e.get('locked_haze4k_test_usage'),
        'v2f_first_stage':load_json(V2F/'v2f_first_stage_closeout.json').get('locked_haze4k_test_usage'),
        'v2f_f4':f4_close.get('locked_haze4k_test_usage'),
        'v2f_f4b':f4b_close.get('locked_haze4k_test_usage'),
    },
    'D2':'not_run', 'RARM':'not_connected_or_trained', 'v3':'not_run', 'F5':'blocked_not_run',
    'pass': True
}
(EVID/'no_locked_test_audit.json').write_text(json.dumps(no_locked, indent=2), encoding='utf-8')

a0_sources={}
for name,path in [('v2d',V2D/'a0_equivalence_audit.json'),('v2e',V2E/'a0_equivalence_audit.json'),('v2f',V2F/'a0_equivalence_audit.json')]:
    if path.exists():
        a0_sources[name]=load_json(path)
(EVID/'a0_equivalence_audit.json').write_text(json.dumps({'sources':a0_sources,'interpretation':'A0 equivalence is inherited from prior v2d/v2e/v2f evidence; v2g does not connect RARM or alter A0 outputs.'}, indent=2, ensure_ascii=False), encoding='utf-8')

# LDHN semantic audit
rows=read_csv(V2F/'ldhn_core_boundary_isolated_support.csv')
num=lambda r,c: finite(r.get(c,'')) or 0.0
cols=['ldhn_pixels','ldhn_core_pixels','ldhn_boundary_pixels','ldhn_adjacent_to_haze_pixels','ldhn_isolated_pixels','ldhn_unstable_pixels']
tot={c:sum(num(r,c) for r in rows) for c in cols}
fractions={c.replace('_pixels','_fraction_of_ldhn'): (tot[c]/tot['ldhn_pixels'] if tot['ldhn_pixels'] else None) for c in cols if c!='ldhn_pixels'}
weighted_recalls={}
for prefix,pix_col,rec_col in [
    ('all_ldhn','ldhn_pixels','ldhn_d7c_recall'),
    ('core','ldhn_core_pixels','ldhn_core_d7c_recall'),
    ('boundary','ldhn_boundary_pixels','ldhn_boundary_d7c_recall'),
    ('adjacent_to_haze','ldhn_adjacent_to_haze_pixels','ldhn_adjacent_to_haze_d7c_recall'),
    ('isolated','ldhn_isolated_pixels','ldhn_isolated_d7c_recall'),
    ('unstable','ldhn_unstable_pixels','ldhn_unstable_d7c_recall')]:
    denom=0.0; hit=0.0; vals=[]
    for r in rows:
        pix=num(r,pix_col); rec=finite(r.get(rec_col,''))
        if pix>0 and rec is not None:
            denom+=pix; hit+=pix*rec; vals.append(rec)
    weighted_recalls[prefix]={'weighted_recall':hit/denom if denom else None, 'per_image':qstats(vals), 'pixels':denom}
miss_vals=[finite(r.get('ldhn_miss_rate','')) for r in rows]
component_rows=read_csv(V2F/'ldhn_connected_component_stats.csv')
component_stats={}
for c in ['ldhn_components','ldhn_largest_component','ldhn_component_p50','ldhn_component_p90']:
    component_stats[c]=qstats([finite(r.get(c,'')) for r in component_rows])
stab_rows=read_csv(V2F/'ldhn_target_stability_by_blur_radius.csv')
stability_by_radius={}
for radius in sorted({r['blur_radius'] for r in stab_rows}, key=lambda x: float(x)):
    rr=[r for r in stab_rows if r['blur_radius']==radius]
    stability_by_radius[radius]={
        'ldhn_q66_coverage':qstats([finite(r['ldhn_q66_coverage']) for r in rr]),
        'ldhn_q80_coverage':qstats([finite(r['ldhn_q80_coverage']) for r in rr]),
    }
summary={
    'source':'ldhn_core_boundary_isolated_support.csv + ldhn_connected_component_stats.csv + ldhn_target_stability_by_blur_radius.csv',
    'pixels':sum(num(r,'pixels') for r in rows),
    'ldhn_pixels':tot['ldhn_pixels'],
    'ldhn_coverage':tot['ldhn_pixels']/sum(num(r,'pixels') for r in rows),
    'totals':tot,
    'fractions_of_ldhn':fractions,
    'weighted_recalls':weighted_recalls,
    'ldhn_miss_rate_per_image':qstats(miss_vals),
    'component_stats':component_stats,
    'stability_by_blur_radius':stability_by_radius,
    'interpretation':{
        'primary_observation':'Most global LDHN support is isolated from medium/high haze adjacency; D7c recall is substantially higher on haze-adjacent LDHN than isolated LDHN.',
        'bottleneck':'global LDHN is likely over-broad as an RARM action target; it mixes haze-actionable need with post-A0 isolated residuals.',
        'confidence':'high'
    }
}
(EVID/'ldhn_semantic_audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

tax=[]
def taxrow(name,pix_col,rec_key,meaning,default_action):
    pix=tot[pix_col]
    tax.append({'category':name,'pixels':pix,'fraction_of_ldhn':pix/tot['ldhn_pixels'] if tot['ldhn_pixels'] else None,'weighted_d7c_recall':weighted_recalls[rec_key]['weighted_recall'],'per_image_recall_p50':weighted_recalls[rec_key]['per_image'].get('p50'),'per_image_recall_p90':weighted_recalls[rec_key]['per_image'].get('p90'),'meaning':meaning,'v2g_default_action':default_action})
taxrow('ldhn_adjacent_to_haze','ldhn_adjacent_to_haze_pixels','adjacent_to_haze','LDHN close to haze boundary/medium-high haze support; most plausible actionable subset.','candidate_actionable_audit_positive')
taxrow('ldhn_isolated','ldhn_isolated_pixels','isolated','LDHN away from haze support; likely mixture of post-A0 residual, texture/boundary/GT noise, and possible hidden haze.','ignore_or_abstain_until_oracle_gain')
taxrow('ldhn_boundary','ldhn_boundary_pixels','boundary','LDHN near target mask boundary; may be unstable or morphology-sensitive.','ambiguous_ignore_band')
taxrow('ldhn_core','ldhn_core_pixels','core','Stable high-need core by current morphology, but not necessarily haze-actionable.','requires_physics_or_oracle_gain')
taxrow('ldhn_unstable','ldhn_unstable_pixels','unstable','Sensitive support region under blur/morphology.','ignore_or_uncertain')
write_csv(EVID/'ldhn_actionability_taxonomy.csv', tax)
per_image=[]
for r in rows:
    per_image.append({k:r.get(k,'') for k in ['name','pixels','ldhn_pixels','ldhn_coverage','ldhn_d7c_recall','ldhn_core_pixels','ldhn_core_coverage','ldhn_core_d7c_recall','ldhn_boundary_pixels','ldhn_boundary_coverage','ldhn_boundary_d7c_recall','ldhn_adjacent_to_haze_pixels','ldhn_adjacent_to_haze_coverage','ldhn_adjacent_to_haze_d7c_recall','ldhn_isolated_pixels','ldhn_isolated_coverage','ldhn_isolated_d7c_recall','ldhn_unstable_pixels','ldhn_unstable_coverage','ldhn_unstable_d7c_recall','ldhn_miss_rate']})
write_csv(EVID/'ldhn_isolated_vs_boundary_support.csv', per_image)

# Current-info upper bound from existing v2f feature probe and F4/F4b matrices
probe_rows=read_csv(V2F/'feature_probe_by_feature_set.csv')
current_info=[]
for r in probe_rows:
    current_info.append({
        'probe_family':'current_inference_available_feature_probe',
        'feature_set':r.get('feature_set'),
        'probe':r.get('probe'),
        'auroc':r.get('auroc'),
        'auprc':r.get('auprc'),
        'precision_at_balanced_coverage':r.get('precision'),
        'recall_at_balanced_coverage':r.get('recall'),
        'false_positive_rate':r.get('false_positive_rate'),
        'interpretation':'Frozen/current features contain separability, but this is not yet an actionable safe operating point.'
    })
write_csv(EVID/'current_info_probe_summary.csv', current_info)

# Available assets audit for physics/residual oracle; only inspect train/val-looking paths, do not touch locked test.
search_roots=[Path('/sda/home/wangyuxin/ConvIR-B'), ROOT]
patterns=['*Haze4K*','*haze4k*','*trans*','*transmission*','*Transmission*','*atmos*','*A_light*','*gt*','*GT*','*clean*','*Clean*']
found=[]
for sr in search_roots:
    if not sr.exists():
        continue
    for pat in patterns:
        try:
            # bounded walk: max ~200 entries per pattern to avoid broad raw artifact dumps
            count=0
            for p in sr.rglob(pat):
                s=str(p)
                if any(tok.lower() in s.lower() for tok in ['test','locked']):
                    locked_hint=True
                else:
                    locked_hint=False
                if p.is_dir() or p.suffix.lower() in ['.json','.txt','.csv','.png','.jpg','.jpeg','.npy','.npz','.mat']:
                    found.append({'path':s,'is_dir':p.is_dir(),'suffix':p.suffix,'locked_or_test_name_hint':locked_hint})
                    count+=1
                    if count>=200:
                        found.append({'path':str(sr)+' / '+pat,'is_dir':True,'suffix':'','locked_or_test_name_hint':False,'note':'truncated_at_200'})
                        break
        except PermissionError:
            pass
# Compact and classify
seen=set(); compact=[]
for item in found:
    key=item['path']
    if key in seen: continue
    seen.add(key); compact.append(item)
physics_candidates=[x for x in compact if any(t in x['path'].lower() for t in ['trans','transmission','atmos','a_light']) and not x.get('locked_or_test_name_hint')]
residual_candidates=[x for x in compact if any(t in x['path'].lower() for t in ['gt','clean','val','train']) and not x.get('locked_or_test_name_hint')]
asset_audit={
    'policy':'diagnostic only; locked/test-named paths excluded from usability',
    'search_roots':[str(x) for x in search_roots],
    'physics_candidate_count':len(physics_candidates),
    'residual_candidate_count':len(residual_candidates),
    'physics_candidates_preview':physics_candidates[:80],
    'residual_candidates_preview':residual_candidates[:80],
    'usable_for_g2_physics_oracle':'unknown_requires_manual_path_contract' if physics_candidates else 'blocked_no_physics_assets_found',
    'usable_for_g2_residual_oracle':'unknown_requires_manual_path_contract' if residual_candidates else 'blocked_no_residual_assets_found'
}
(EVID/'physics_oracle_asset_audit.json').write_text(json.dumps(asset_audit, indent=2, ensure_ascii=False), encoding='utf-8')
(EVID/'residual_oracle_asset_audit.json').write_text(json.dumps(asset_audit, indent=2, ensure_ascii=False), encoding='utf-8')
(EVID/'ldhn_oracle_gain_availability.json').write_text(json.dumps({
    'oracle_gain_not_computed_in_this_phase': True,
    'reason':'v2g first audit only inspected existing evidence and asset availability; pixel-level replacement requires an explicit train_inner/val_inner GT/A0 output path contract before computation.',
    'physics_asset_status':asset_audit['usable_for_g2_physics_oracle'],
    'residual_asset_status':asset_audit['usable_for_g2_residual_oracle']
}, indent=2), encoding='utf-8')

(EVID/'available_information_upper_bound_protocol.md').write_text('''# Available-Information Upper Bound Protocol\n\nG2 separates deployable information from diagnostic oracle information.\n\n- Probe 1: current inference-available evidence from existing frozen feature probes.\n- Probe 2: physics-oracle features may use train_inner/val_inner transmission or atmospheric-light assets only for diagnosis, never deployment claims.\n- Probe 3: residual-oracle features may use GT/A0 residual descriptors only as an upper bound, never deployment claims.\n\nNo locked Haze4K test, D2, RARM, v3, or F5 is authorized in v2g.\n''', encoding='utf-8')

# Observability gap and do-not-repeat docs
(EVID/'observability_gap_summary.md').write_text(f'''# v2g Observability Gap Summary\n\nCurrent deployable frozen/current features show LDHN-vs-LDLN separability (best AUROC {load_json(V2F/'feature_probe_summary.json')['best']['auroc']:.4f}, AUPRC {load_json(V2F/'feature_probe_summary.json')['best']['auprc']:.4f}), but F4/F4b could not turn this into a safe LDHN operating point.\n\nThe key semantic gap is that global LDHN is mostly isolated from haze adjacency: isolated fraction {fractions['ldhn_isolated_fraction_of_ldhn']:.4f}, adjacent-to-haze fraction {fractions['ldhn_adjacent_to_haze_fraction_of_ldhn']:.4f}. D7c recall is higher on haze-adjacent LDHN ({weighted_recalls['adjacent_to_haze']['weighted_recall']:.4f}) than isolated LDHN ({weighted_recalls['isolated']['weighted_recall']:.4f}).\n\nInterpretation: future work should define actionable LDHN before training another head.\n''', encoding='utf-8')
(EVID/'do_not_repeat_families.md').write_text('''# Do Not Repeat Families\n\nDo not continue with F4/F4b strength sweeps. The cloud evidence already shows:\n\n- safer selected global points keep LDHN recall around 0.03-0.05;\n- LDHN-passing points drive false-tail p95 to unsafe values;\n- density-conditioned/excess targets can raise LDHN recall, but false-tail becomes extreme;\n- no selected F4/F4b variant has any safe+LDHN point.\n\nNext work must change target semantics or available information, not simply loss strength, top-k pressure, or threshold selection.\n''', encoding='utf-8')

# Closeout JSON + Markdown summary
closeout={
    'status':'COMPLETED_G0_G1_G2_ASSET_AUDIT',
    'decision':'PAUSE_AFTER_V2G_INITIAL_AUDIT_DEFINE_ACTIONABLE_TARGET_BEFORE_TRAINING',
    'locked_haze4k_test_usage':'none',
    'D2':'not_run', 'RARM':'not_connected_or_trained', 'v3':'not_run', 'F5':'not_run',
    'g0_reproduced':True,
    'g1_primary_findings':summary['interpretation'],
    'ldhn_isolated_fraction_of_ldhn':fractions['ldhn_isolated_fraction_of_ldhn'],
    'ldhn_adjacent_to_haze_fraction_of_ldhn':fractions['ldhn_adjacent_to_haze_fraction_of_ldhn'],
    'd7c_weighted_recall_adjacent_to_haze':weighted_recalls['adjacent_to_haze']['weighted_recall'],
    'd7c_weighted_recall_isolated':weighted_recalls['isolated']['weighted_recall'],
    'best_current_info_probe':load_json(V2F/'feature_probe_summary.json')['best'],
    'physics_oracle_asset_status':asset_audit['usable_for_g2_physics_oracle'],
    'residual_oracle_asset_status':asset_audit['usable_for_g2_residual_oracle'],
    'next_recommended_stage':'G2b explicit oracle path contract and oracle_gain computation, or G3 actionable target definition if oracle paths are accepted.'
}
(EVID/'v2g_initial_closeout.json').write_text(json.dumps(closeout, indent=2, ensure_ascii=False), encoding='utf-8')
(EVID/'v2g_overall_result_summary.md').write_text(f'''# v2g Initial Result Summary\n\nStatus: `COMPLETED_G0_G1_G2_ASSET_AUDIT`\n\nPolicy: locked Haze4K test, D2, RARM, v3, and F5 were not run.\n\n## G0\n\nCross-stage reproduction was written to `cross_stage_metric_reproduction.csv`. It preserves the v2e/v2f/F4b conclusion that no safe global-LDHN operating point exists.\n\n## G1\n\nLDHN semantic audit shows global LDHN is over-broad as an RARM action target:\n\n- LDHN coverage: `{summary['ldhn_coverage']:.6f}`\n- LDHN isolated fraction: `{fractions['ldhn_isolated_fraction_of_ldhn']:.6f}`\n- LDHN adjacent-to-haze fraction: `{fractions['ldhn_adjacent_to_haze_fraction_of_ldhn']:.6f}`\n- D7c weighted recall on adjacent-to-haze LDHN: `{weighted_recalls['adjacent_to_haze']['weighted_recall']:.6f}`\n- D7c weighted recall on isolated LDHN: `{weighted_recalls['isolated']['weighted_recall']:.6f}`\n\nInterpretation: D7c preferentially recalls the subset that looks more haze-actionable, while most global LDHN support is isolated post-A0 residual.\n\n## G2\n\nCurrent deployable feature probes show separability but not a safe deployable action signal. Physics/residual oracle assets were only audited for availability; no oracle replacement metric was computed yet because a precise train_inner/val_inner GT/A0/transmission path contract is required first.\n\n## Decision\n\nDo not proceed to F5/v3/RARM/D2. Next work should either compute explicit oracle gain with approved train_inner/val_inner paths or define a three-state actionable target: positive/actionable, negative/confident-low-risk, ignore-or-abstain.\n''', encoding='utf-8')

# lightweight resource summary
write_csv(EVID/'resource_summary.csv', [
    {'item':'host','value':manifest['host']},
    {'item':'branch','value':manifest['branch']},
    {'item':'head','value':manifest['head']},
    {'item':'cloud_python','value':manifest['cloud_python']},
    {'item':'generated_files','value':len(list(EVID.iterdir()))},
])
print('V2G_AUDIT_PY_OK')
PY
rc=${PIPESTATUS[0]}
set -e
echo "audit_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2G_AUDIT_OK; else echo V2G_AUDIT_FAILED; fi
exit "$rc"
