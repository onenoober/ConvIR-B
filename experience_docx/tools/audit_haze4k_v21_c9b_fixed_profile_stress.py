#!/usr/bin/env python3
"""C9b fixed conservative profile stress for C7c local-alpha."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import numpy as np
from audit_haze4k_v20_c2b_multirule_router import fnum, write_csv
from audit_haze4k_v21_c7b_local_alpha_prototype import C7B_STRONG_GATE, gate_pass, score
from audit_haze4k_v21_c9_shifted_strong_validation import build_group_labels, summarize, BIN_MEAN_FLOOR, BIN_POSITIVE_FLOOR, BIN_SEVERE_CAP

def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as h:
        return list(csv.DictReader(h))

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile_per_image', type=Path, required=True)
    ap.add_argument('--image_rows', type=Path, required=True)
    ap.add_argument('--image_feature_rows', type=Path, required=True)
    ap.add_argument('--fixed_profile', default='riskcap36_no075')
    ap.add_argument('--out_dir', type=Path, required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    rows=[r for r in read_csv(args.profile_per_image) if str(r['profile'])==args.fixed_profile]
    image_rows=read_csv(args.image_rows); feature_rows=read_csv(args.image_feature_rows)
    labels_by_dim=build_group_labels(image_rows, feature_rows)
    order=[str(r['name']) for r in image_rows]
    name_labels={dim:{name:labels[i] for i,name in enumerate(order)} for dim,labels in labels_by_dim.items()}
    bin_rows=[]; dim_rows=[]; selected=[]
    for dim, mapping in name_labels.items():
        dim_subset=[]
        for label in sorted(set(mapping.values())):
            subset=[]
            for r in rows:
                if mapping[str(r['name'])]==label:
                    clone=dict(r); clone['dimension']=dim; clone['heldout_bin']=label; clone['fixed_profile']=args.fixed_profile
                    subset.append(clone); selected.append(clone); dim_subset.append(clone)
            brec={'dimension':dim,'heldout_bin':label,'fixed_profile':args.fixed_profile, **summarize(subset)}
            brec['bin_safety_pass']=fnum(brec['mean_dPSNR'])>=BIN_MEAN_FLOOR and fnum(brec['positive_ratio'])>=BIN_POSITIVE_FLOOR and fnum(brec['severe_loss_per_600'])<=BIN_SEVERE_CAP
            bin_rows.append(brec)
        bins=[b for b in bin_rows if b['dimension']==dim]
        drec={'dimension':dim,'fixed_profile':args.fixed_profile, **summarize(dim_subset)}
        drec['min_bin_mean_dPSNR']=min(fnum(b['mean_dPSNR']) for b in bins)
        drec['min_bin_positive_ratio']=min(fnum(b['positive_ratio']) for b in bins)
        drec['max_bin_severe_loss_per_600']=max(fnum(b['severe_loss_per_600']) for b in bins)
        drec['bin_safety_pass_count']=sum(bool(b['bin_safety_pass']) for b in bins)
        drec['dimension_shift_strong_pass']=bool(drec['strong_gate_pass']) and drec['bin_safety_pass_count']==len(bins)
        dim_rows.append(drec)
    write_csv(args.out_dir/'v21_c9b_fixed_profile_selected_per_image.csv', selected, sorted({k for r in selected for k in r}))
    write_csv(args.out_dir/'v21_c9b_fixed_profile_bin_metrics.csv', bin_rows, sorted({k for r in bin_rows for k in r}))
    write_csv(args.out_dir/'v21_c9b_fixed_profile_dimension_summary.csv', dim_rows, sorted({k for r in dim_rows for k in r}))
    all_pass=all(bool(r['dimension_shift_strong_pass']) for r in dim_rows)
    decision='C9B_FIXED_PROFILE_SHIFTED_PASS_START_C10_FORMAL_5X3' if all_pass else 'C9B_FIXED_PROFILE_SHIFTED_FAIL_START_C8_OR_LOCAL_REWORK'
    payload={'route':'Haze4K-v2.1 SEG-Mix','phase':'C9b Fixed Conservative Profile Stress','locked_test_touched':False,'fixed_profile':args.fixed_profile,'strong_gate':C7B_STRONG_GATE,'dimension_rows':dim_rows,'bin_rows':bin_rows,'decision':decision}
    (args.out_dir/'v21_c9b_fixed_profile_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Haze4K v2.1 C9b Fixed Conservative Profile Stress','',f'Decision: `{decision}`','',f'Fixed profile: `{args.fixed_profile}`','', '| Dimension | Pass | Mean | Hard | Easy | dSSIM | Positive | Severe/600 | Min Bin Mean | Min Bin Pos | Max Bin Severe |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in dim_rows:
        lines.append(f"| `{r['dimension']}` | `{r['dimension_shift_strong_pass']}` | `{fnum(r['mean_dPSNR']):.6f}` | `{fnum(r['hard_bottom25_dPSNR']):.6f}` | `{fnum(r['easy_top25_dPSNR']):.6f}` | `{fnum(r['dSSIM']):.8f}` | `{fnum(r['positive_ratio']):.6f}` | `{fnum(r['severe_loss_per_600']):.1f}` | `{fnum(r['min_bin_mean_dPSNR']):.6f}` | `{fnum(r['min_bin_positive_ratio']):.6f}` | `{fnum(r['max_bin_severe_loss_per_600']):.1f}` |")
    lines += ['', 'C10 formal 5x3 is authorized only if every dimension passes. Locked test remains blocked.']
    (args.out_dir/'v21_c9b_fixed_profile_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'V21_C9B_FIXED_PROFILE_OK decision={decision} profile={args.fixed_profile} out={args.out_dir}', flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
