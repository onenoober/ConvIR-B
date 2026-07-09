#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v2h_d_fam2_noop_equivalence.log
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$EVID"
echo "v2h_d_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$ROOT"
set +e
{
  "$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency')
evid=root/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709'
c=json.loads((evid/'oof_stability_closeout.json').read_text(encoding='utf-8'))
if not c.get('gate_pass'):
    raise SystemExit('v2h-C gate did not pass; v2h-D not authorized')
print('V2H_D_PREFLIGHT_AUTHORIZED')
PY
  cd "$ROOT/Dehazing/ITS"
  "$PY" "$ROOT/experience_docx/tools/check_haze4k_fam_equivalence.py" \
    --checkpoint "$A0" \
    --candidate_mode fam2_modres \
    --height 256 \
    --width 256 \
    --seed 3407 \
    --device cuda \
    --output "$EVID/fam2_noop_equivalence_256.json"
  "$PY" "$ROOT/experience_docx/tools/check_haze4k_fam_equivalence.py" \
    --checkpoint "$A0" \
    --candidate_mode fam2_modres \
    --height 320 \
    --width 288 \
    --seed 3411 \
    --device cuda \
    --output "$EVID/fam2_noop_equivalence_320x288.json"
  cd "$ROOT"
  "$PY" - <<'PY'
import json, math
from pathlib import Path

ROOT=Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2h-actionable-prior-sufficiency')
EVID=ROOT/'experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709'
files=[EVID/'fam2_noop_equivalence_256.json', EVID/'fam2_noop_equivalence_320x288.json']
items=[json.loads(p.read_text(encoding='utf-8')) for p in files]
max_abs=max(float(item['max_abs_diff']) for item in items)
mean_abs=max(float(item['mean_abs_diff']) for item in items)
fresh_pass=all(item.get('fresh_shared_init',{}).get('pass') for item in items)
equiv_pass=all(item.get('pass') for item in items)
missing_ok=all(all('.modulator.' in key for key in item.get('missing_candidate_keys',[])) for item in items)
param_delta=items[0]['param_delta']
gate_pass=bool(equiv_pass and fresh_pass and missing_ok and max_abs <= 1e-6 and mean_abs <= 1e-7 and param_delta > 0)
closeout={
    'status':'COMPLETED_GATE_PASS' if gate_pass else 'COMPLETED_GATE_FAIL',
    'decision_label':'V2H_D_FAM2_NOOP_EQUIVALENCE_PASS_AUTHORIZE_IMPLEMENTATION_DESIGN_ONLY' if gate_pass else 'V2H_D_FAM2_NOOP_EQUIVALENCE_FAIL_PAUSE',
    'locked_haze4k_test_usage':'none',
    'D2':'not_run',
    'F5':'not_run',
    'v3':'not_run',
    'RARM':'not_connected_or_trained',
    'training':'none',
    'candidate_mode':'fam2_modres',
    'input_checks':[{'shape':item['input_shape'],'max_abs_diff':item['max_abs_diff'],'mean_abs_diff':item['mean_abs_diff'],'pass':item['pass']} for item in items],
    'fresh_shared_init_pass':fresh_pass,
    'missing_candidate_keys_are_modulator_only':missing_ok,
    'param_delta':param_delta,
    'param_delta_pct':items[0]['param_delta_pct'],
    'max_abs_diff':max_abs,
    'mean_abs_diff':mean_abs,
    'gate_pass':gate_pass,
    'next_authorized_stage':'write implementation design for bounded prior-conditioned modulation only; no RARM/training launch yet' if gate_pass else 'pause; do not connect modulation path',
}
(EVID/'fam2_noop_closeout.json').write_text(json.dumps(closeout, indent=2, sort_keys=True)+'\n', encoding='utf-8')
md=[
    '# v2h-D FAM2 No-Op Equivalence Review',
    '',
    f"Status: `{closeout['status']}`",
    '',
    f"Decision label: `{closeout['decision_label']}`",
    '',
    'Policy: no training, no locked Haze4K test, no D2/F5/v3/RARM/adapter training/canary expansion.',
    '',
    '## Result',
    '',
    f"- Candidate mode: `{closeout['candidate_mode']}`",
    f"- Max abs output difference across checks: `{max_abs:.12g}`",
    f"- Max mean abs output difference across checks: `{mean_abs:.12g}`",
    f"- Fresh shared init pass: `{fresh_pass}`",
    f"- Missing candidate keys are modulator-only: `{missing_ok}`",
    f"- Parameter delta: `{param_delta}` (`{closeout['param_delta_pct']:.6f}%`)",
    '',
    '## Decision',
    '',
    closeout['next_authorized_stage'],
    '',
]
(EVID/'fam2_noop_summary.md').write_text('\n'.join(md), encoding='utf-8')
readme=EVID/'README.md'
readme.write_text(readme.read_text(encoding='utf-8').rstrip() + '\n\n## v2h-D Result\n\n' + '\n'.join(md) + '\n', encoding='utf-8')
if not gate_pass:
    raise SystemExit(1)
print('V2H_D_PY_OK')
PY
} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v2h_d_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V2H_D_FAM2_NOOP_EQUIVALENCE_OK | tee -a "$STATUS"; else echo V2H_D_FAM2_NOOP_EQUIVALENCE_FAILED | tee -a "$STATUS"; fi
exit "$rc"
