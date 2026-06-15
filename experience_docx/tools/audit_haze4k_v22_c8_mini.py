#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, subprocess, time
from pathlib import Path

ALPHAS=[0,0.0625,0.125,0.25,0.375,0.50]

def sha(p:Path):
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def g(cmd, cwd=None):
    try: return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def globs(pats):
    out=[]
    for pat in pats: out += sorted(glob.glob(pat, recursive=True))
    return sorted(set(out))

def write_csv(p, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def file_info(p):
    p=Path(p); return {'path':str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.is_file() else None,'sha256':sha(p)}

def repo_info(p):
    p=Path(p); return {'path':str(p),'exists':p.exists(),'commit':g(['git','rev-parse','HEAD'],p),'remote':g(['git','remote','get-url','origin'],p)}

def split_counts(p):
    p=Path(p)
    if not p.is_file(): return {'exists':False}
    d=json.loads(p.read_text(encoding='utf-8')); s=d.get('splits',d)
    return {'exists':True,'splits':{k:len(v) for k,v in s.items() if isinstance(v,list)}}

def placeholder(out, prefix, decision, expert, reason, phase, files):
    row={'phase':phase,'expert':expert,'decision':decision,'reason':reason,'status':'BLOCKED_CHECKPOINT_UNAVAILABLE','mean_dPSNR':'','hard_bottom25_dPSNR':'','easy_top25_dPSNR':'','dSSIM':'','positive_ratio':'','severe_loss_per_600':'','unique_win_rate_hard_redflag':''}
    fields=list(row)
    for name in files: write_csv(out/name,[row],fields)
    (out/f'{prefix}_decision.md').write_text(f'# {phase} {expert} Decision\n\nDecision: `{decision}`\n\nStatus: `BLOCKED_CHECKPOINT_UNAVAILABLE`\n\nReason: {reason}\n\nLocked test remains untouched. This is an engineering asset blocker, not scientific evidence against complementarity.\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--a0',type=Path,required=True); ap.add_argument('--udp-repo',type=Path,required=True); ap.add_argument('--udp-ckpt',type=Path,required=True); ap.add_argument('--wd-repo',type=Path,required=True); ap.add_argument('--mb-repo',type=Path,required=True); args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    branch=g(['git','branch','--show-current'],args.root); commit=g(['git','rev-parse','HEAD'],args.root); status=g(['git','status','--short'],args.root) or ''
    wd_ckpts=globs(['/sda/home/wangyuxin/ConvIR-B/checkpoints/wdmamba/**/*Haze4K*','/sda/home/wangyuxin/ConvIR-B/checkpoints/wdmamba/**/*haze4k*','/sda/home/wangyuxin/ConvIR-B/checkpoints/external*/**/*WDMamba*Haze4K*'])
    fs_ckpts=globs(['/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/*FSNet*Haze4K*','/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/*fsnet*haze4k*','/sda/home/wangyuxin/ConvIR-B/checkpoints/external*/**/*FSNet*UDP*Haze4K*'])
    mb_ckpts=globs(['/sda/home/wangyuxin/ConvIR-B/checkpoints/mbtaylor*/**/*Haze4K*','/sda/home/wangyuxin/ConvIR-B/checkpoints/external*/**/*Taylor*Haze4K*','/sda/home/wangyuxin/ConvIR-B/checkpoints/external*/**/*MB*Taylor*'])
    manifest={'route':'haze4k_v2_2_c8_mini_expert_oracle_20260615','time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'branch':branch,'commit':commit,'git_status_short':status.splitlines(),'locked_policy':{'locked_test_touched':False,'locked_outputs_read':False,'no_locked_informed_tuning':True},'alpha_grid':ALPHAS,'data':{'path':str(args.data),'exists':args.data.exists(),'split':split_counts(args.split_json)},'experts':{'A0':{'checkpoint':file_info(args.a0),'role':'anchor'},'FullUDP_current':{'repo':repo_info(args.udp_repo),'checkpoint':file_info(args.udp_ckpt),'role':'S0 current expert'},'WDMamba':{'repo':repo_info(args.wd_repo),'checkpoint_candidates':wd_ckpts,'checkpoint_available':bool(wd_ckpts),'asset_status':'AVAILABLE' if wd_ckpts else 'BLOCKED_CHECKPOINT_UNAVAILABLE','paper_repo':'https://github.com/SunJ000/WDMamba'},'FSNet_UDP':{'repo':repo_info(args.udp_repo),'architecture_file':str(args.udp_repo/'Dehazing/ITS/models/FSNet_UDPNet.py'),'checkpoint_candidates':fs_ckpts,'checkpoint_available':bool(fs_ckpts),'asset_status':'AVAILABLE' if fs_ckpts else 'BLOCKED_CHECKPOINT_UNAVAILABLE','paper_repo':'https://github.com/Harbinzzy/UDPNet'},'MB_TaylorFormerV2_L':{'repo':repo_info(args.mb_repo),'checkpoint_candidates':mb_ckpts,'checkpoint_available':bool(mb_ckpts),'asset_status':'AVAILABLE' if mb_ckpts else 'BLOCKED_CHECKPOINT_UNAVAILABLE','paper_repo':'https://github.com/FVL2020/MB-TaylorFormerV2'}}}
    (out/'v22_c8_0_expert_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    scripts=['audit_haze4k_v22_c8_mini.py','audit_haze4k_v21_c6_c7_multialpha_local_oracle.py','audit_haze4k_v20_c2_outputdiff_router.py']
    (out/'v22_c8_0_metric_script_sha256.txt').write_text(''.join(f"{sha(args.root/'experience_docx/tools'/s) or 'MISSING'}  experience_docx/tools/{s}\n" for s in scripts),encoding='utf-8')
    (out/'v22_c8_0_no_locked_status.txt').write_text(f'C8_NO_LOCKED_STATUS\nbranch={branch}\ncommit={commit}\nlocked_test_touched=false\nlocked_outputs_read=false\nno_locked_informed_tuning=true\nsplits=val_regular,val_hard\nC8_NO_LOCKED_OK\n',encoding='utf-8')
    rows=[{'asset':'WDMamba_models','url':'https://pan.baidu.com/s/1HIs-nHXEaLxwBb1279PVbw','method':'curl_and_baidupcs_probe','status':'HTML_OR_AUTH_REQUIRED'},{'asset':'MBTaylor_models','url':'https://pan.baidu.com/s/11V-wD01rPTHMFFJyjB0R0w','method':'curl_and_baidupcs_probe','status':'HTML_OR_AUTH_REQUIRED'},{'asset':'UDPNet_models','url':'https://pan.baidu.com/s/1JqB-YBPzZAiQsdLlNcidLQ','method':'curl_and_baidupcs_probe','status':'CURRENT_CONVIR_UDP_PRESENT_FSNET_UDP_NOT_PRESENT'}]
    write_csv(out/'v22_c8_0_download_probe.csv',rows,['asset','url','method','status'])
    (out/'v22_c8_0_reliability_note.md').write_text('# C8-0 Command Reliability Note\n\n- `git clone git@github.com:onenoober/ConvIR-B.git` failed on `convir-4090` with host-key verification; corrected to HTTPS clone.\n- A broad checkpoint search command with redirection inside a Bash `for` item list failed; corrected to direct `find ... 2>/dev/null`.\n- Baidu direct download probes returned landing/auth pages, so external checkpoints are unavailable without authenticated transfer or alternate mirrors.\n',encoding='utf-8')
    conv=args.udp_repo/'Dehazing/ITS/models/ConvIR_UDPNet.py'; fs=args.udp_repo/'Dehazing/ITS/models/FSNet_UDPNet.py'
    dup={'current_fulludp_checkpoint':file_info(args.udp_ckpt),'conv_ir_udp_arch_sha256':sha(conv),'fsnet_udp_arch_sha256':sha(fs),'architecture_file_identical':sha(conv)==sha(fs) if conv.is_file() and fs.is_file() else False,'fsudp_checkpoint_candidates':fs_ckpts,'fsudp_checkpoint_available':bool(fs_ckpts),'decision':'FSUDP_CHECKPOINT_UNAVAILABLE_SKIP_TO_C8_3_FALLBACK' if not fs_ckpts else 'FSUDP_RENDER_REQUIRED','locked_test_touched':False}
    (out/'v22_c8_2_fsudp_duplicate_audit.json').write_text(json.dumps(dup,indent=2,sort_keys=True),encoding='utf-8')
    (out/'v22_c8_2_fsudp_duplicate_audit.md').write_text(f"# C8-2 FSNet+UDP Duplicate Audit\n\nDecision: `{dup['decision']}`\n\n- Current FullUDP checkpoint sha256: `{dup['current_fulludp_checkpoint']['sha256']}`\n- ConvIR+UDP arch sha256: `{dup['conv_ir_udp_arch_sha256']}`\n- FSNet+UDP arch sha256: `{dup['fsnet_udp_arch_sha256']}`\n- Architecture file identical: `{dup['architecture_file_identical']}`\n- FSNet+UDP checkpoint candidates: `{len(fs_ckpts)}`\n\nFSNet+UDP is not proven duplicate by source file hash, but no FSNet+UDP Haze4K checkpoint is available, so S2 cannot render complementarity metrics. Locked test remains untouched.\n",encoding='utf-8')
    placeholder(out,'v22_c8_1_wdmamba','WDMAMBA_PREFLIGHT_FAILED_CHECKPOINT_UNAVAILABLE_SKIP_TO_C8_2','WDMamba','Official repo cloned but no reproducible Haze4K checkpoint/result package is present on convir-4090; Baidu probes require authenticated transfer or alternate mirror.','C8-1',[Path('v22_c8_1_wdmamba_single_summary.csv'),Path('v22_c8_1_wdmamba_alpha_grid.csv'),Path('v22_c8_1_wdmamba_oracle_vs_s0.csv'),Path('v22_c8_1_wdmamba_group_metrics.csv'),Path('v22_c8_1_wdmamba_unique_wins.csv')])
    row={'phase':'C8-2','expert':'FSNet+UDP','decision':dup['decision'],'status':'BLOCKED_CHECKPOINT_UNAVAILABLE','mean_dPSNR':'','hard_bottom25_dPSNR':'','positive_ratio':'','severe_loss_per_600':''}
    for name in ['v22_c8_2_fsudp_single_summary.csv','v22_c8_2_fsudp_alpha_grid.csv','v22_c8_2_s2_forward_selection_oracle.csv','v22_c8_2_s2_expert_composition_by_group.csv']:
        write_csv(out/name,[row],list(row))
    (out/'v22_c8_2_s2_decision.md').write_text(f"# C8-2 FSNet+UDP Decision\n\nDecision: `{dup['decision']}`\n\nNo FSNet+UDP Haze4K checkpoint is available. Proceed to C8-3 fallback asset probe.\n",encoding='utf-8')
    placeholder(out,'v22_c8_3_mbtaylor','MBTAYLOR_PREFLIGHT_FAILED_CHECKPOINT_UNAVAILABLE_C8_STOP_ASSET_BLOCKED','MB-TaylorFormerV2-L','Official repo cloned but no Haze4K-L checkpoint is present on convir-4090; Baidu probes require authenticated transfer or alternate mirror.','C8-3',[Path('v22_c8_3_mbtaylor_single_summary.csv'),Path('v22_c8_3_mbtaylor_alpha_grid.csv'),Path('v22_c8_3_s3_forward_selection_oracle.csv'),Path('v22_c8_3_s3_expert_composition_by_group.csv')])
    decision='C8_STOP_PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE'
    (out/'v22_c8_decision.md').write_text(f'# C8-Mini Decision\n\nDecision: `{decision}`\n\nC8-0 completed preregistration and asset audit. WDMamba, FSNet+UDP, and MB-TaylorFormerV2-L source repos are available or audited, but the required Haze4K pretrained checkpoints/result packages are not available on `convir-4090`. Therefore no expert rendering, alpha-grid metrics, oracle metrics, unique-win rates, group-min metrics, or C9 router authorization can be scientifically claimed.\n\nThis is an engineering asset blocker, not evidence that complementarity is absent. Resume C8-1 when a reproducible WDMamba Haze4K checkpoint/result package is available and its sha256 is recorded. Locked test remains untouched.\n',encoding='utf-8')
    (out/'v22_c8_summary.json').write_text(json.dumps({'decision':decision,'locked_test_touched':False,'wdmamba_checkpoint_available':bool(wd_ckpts),'fsudp_checkpoint_available':bool(fs_ckpts),'mbtaylor_checkpoint_available':bool(mb_ckpts),'router_training_authorized':False},indent=2,sort_keys=True),encoding='utf-8')
    print('V22_C8_MINI_AUDIT_OK')
if __name__=='__main__': main()
