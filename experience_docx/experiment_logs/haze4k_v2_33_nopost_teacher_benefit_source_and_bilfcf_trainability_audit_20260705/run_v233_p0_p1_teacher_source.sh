#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
UDP_TABLE=$BASE/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite/experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/phase0_official_eval/udpnet_convir_bucket_compare.csv
WD_TABLE=$BASE/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv
STATUS=$EVID/status.txt
ROUTE_ID=haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705
export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "p0_p1_start $ROUTE_ID $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "udp_table=$UDP_TABLE"
  echo "wdmamba_table=$WD_TABLE"
  echo "python=$PY"
} | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v233_nopost_audit.py --phase p0 --data_dir "$DATA" --checkpoint "$A0" --output_dir "$EVID" > "$EVID/v233_p0.log" 2>&1
rc0=$?
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v233_nopost_audit.py --phase p1 --data_dir "$DATA" --checkpoint "$A0" --output_dir "$EVID" --udp_table "$UDP_TABLE" --wdmamba_table "$WD_TABLE" > "$EVID/v233_p1_teacher_benefit_audit.log" 2>&1
rc1=$?
set -e
echo "p0_p1_done rc0=$rc0 rc1=$rc1 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc0" -eq 0 && "$rc1" -eq 0 ]]; then
  echo "V233_P0_P1_OK" | tee -a "$STATUS"
else
  echo "V233_P0_P1_FAILED" | tee -a "$STATUS"
fi
exit $(( rc0 != 0 ? rc0 : rc1 ))
