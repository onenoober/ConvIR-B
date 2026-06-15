#!/usr/bin/env bash
set -euo pipefail
REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DATA=${DATA:-/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K}
DEPTH_CACHE=${DEPTH_CACHE:-/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf}
A0_CKPT=${A0_CKPT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl}
UDPNET_REPO=${UDPNET_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/UDPNet}
UDPNET_CKPT=${UDPNET_CKPT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt}
PATCH_ROWS=${PATCH_ROWS:-$EVID/v21_c7b_patch_feature_rows.csv}
IMAGE_ROWS=${IMAGE_ROWS:-$EVID/v21_c7b_image_rows.csv}
C10_SUMMARY=${C10_SUMMARY:-$EVID/v21_c10_formal_summary.json}
FIXED_PROFILE=${FIXED_PROFILE:-riskcap36_no075}
STATUS=$EVID/status_locked_one_shot.txt
LOG=$EVID/v21_locked_one_shot.log
SUMMARY=$EVID/v21_locked_one_shot_summary.json
mkdir -p "$EVID"
if [ -e "$STATUS" ] || [ -e "$SUMMARY" ]; then
  echo "V21_LOCKED_ONE_SHOT_REFUSE_EXISTING_OUTPUT status=$STATUS summary=$SUMMARY" | tee -a "$LOG"
  exit 3
fi
echo "===== v21_locked_one_shot_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
 echo "v21_locked_one_shot_start $(date --iso-8601=seconds)"
 echo "remote_root=$REMOTE_ROOT"
 echo "python=$PY"
 echo "patch_rows=$PATCH_ROWS"
 echo "image_rows=$IMAGE_ROWS"
 echo "c10_summary=$C10_SUMMARY"
 echo "fixed_profile=$FIXED_PROFILE"
 echo "locked_test_authorized_by=C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT"
 echo "locked_test_touched=true"
 echo "one_shot=true"
 echo "no_tuning_from_locked=true"
 if [ -f "$REMOTE_ROOT/.codex_source_branch" ]; then sed 's/^/source_branch=/' "$REMOTE_ROOT/.codex_source_branch"; fi
 if [ -f "$REMOTE_ROOT/.codex_source_commit" ]; then sed 's/^/source_commit=/' "$REMOTE_ROOT/.codex_source_commit"; fi
 if [ -f "$REMOTE_ROOT/.codex_source_copy_time" ]; then sed 's/^/source_copy_time=/' "$REMOTE_ROOT/.codex_source_copy_time"; fi
 test -x "$PY" && echo "python_exists=true"
 test -f "$PATCH_ROWS" && echo "patch_rows_exists=true"
 test -f "$IMAGE_ROWS" && echo "image_rows_exists=true"
 test -f "$C10_SUMMARY" && echo "c10_summary_exists=true"
 test -d "$DATA/test" && echo "locked_data_exists=true"
 test -d "$DEPTH_CACHE/test" && echo "locked_depth_cache_exists=true"
 test -f "$A0_CKPT" && sha256sum "$A0_CKPT" | sed 's/^/a0_sha256=/'
 test -f "$UDPNET_CKPT" && sha256sum "$UDPNET_CKPT" | sed 's/^/udpnet_sha256=/'
} | tee -a "$STATUS"
cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v21_locked_one_shot.py \
 --patch_rows "$PATCH_ROWS" \
 --image_rows "$IMAGE_ROWS" \
 --c10_summary "$C10_SUMMARY" \
 --fixed_profile "$FIXED_PROFILE" \
 --convir_its_dir "$REMOTE_ROOT/Dehazing/ITS" \
 --udp_repo "$UDPNET_REPO" \
 --data_dir "$DATA" \
 --data_split test \
 --depth_cache_dir "$DEPTH_CACHE" \
 --depth_split test \
 --a0_checkpoint "$A0_CKPT" \
 --official_checkpoint "$UDPNET_CKPT" \
 --pad_factor 32 \
 --patch_size 128 \
 --print_freq 50 \
 --top_k 900 \
 --low_pool_limit 80 \
 --high_pool_limit 120 \
 --out_dir "$EVID" \
 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v21_locked_one_shot_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo "V21_LOCKED_ONE_SHOT_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"; else echo "V21_LOCKED_ONE_SHOT_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"; fi
exit "$rc"
