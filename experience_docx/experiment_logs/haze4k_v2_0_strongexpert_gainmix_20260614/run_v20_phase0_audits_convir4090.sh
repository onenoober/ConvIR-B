#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DTA_EVID=${DTA_EVID:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613}
FULLUDP_EVAL=${FULLUDP_EVAL:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/phase0_official_eval}
STATUS=$EVID/status.txt

mkdir -p "$EVID"
{
  echo "v20_phase0_audits_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "dta_evidence=$DTA_EVID"
  echo "fulludp_eval=$FULLUDP_EVAL"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,80p' | sed 's/^/git_status=/'
  fi
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"

run_one() {
  local name=$1
  shift
  local log="$EVID/${name}.log"
  echo "${name}_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${name}_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
  return "$rc"
}

pids=()
run_one v37_d8_d9_reconciliation \
  "$PY" experience_docx/tools/audit_haze4k_v37_d8_d9_reconciliation.py \
    --dta-evidence-dir "$DTA_EVID" \
    --out-dir "$EVID" &
pids+=($!)

run_one v37_d9_forensic \
  "$PY" experience_docx/tools/audit_haze4k_v37_d9_forensic.py \
    --dta-evidence-dir "$DTA_EVID" \
    --out-dir "$EVID" &
pids+=($!)

run_one v20_candidate_zoo_oracle \
  "$PY" experience_docx/tools/audit_haze4k_v20_candidate_zoo_oracle.py \
    --dta-evidence-dir "$DTA_EVID" \
    --fulludp-eval-dir "$FULLUDP_EVAL" \
    --out-dir "$EVID" &
pids+=($!)

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

if [ "$rc" -eq 0 ]; then
  echo "V20_PHASE0_AUDITS_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V20_PHASE0_AUDITS_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
