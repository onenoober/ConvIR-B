#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle}
EVID=${EVID:-$ROOT/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DATA=${DATA:-/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K}
SPLIT_JSON=${SPLIT_JSON:-$ROOT/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
A0=${A0:-/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl}
UDP_REPO=${UDP_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/UDPNet}
UDP_CKPT=${UDP_CKPT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt}
EXT=${EXT:-/sda/home/wangyuxin/ConvIR-B/repos/external_experts}
WD_REPO=${WD_REPO:-$EXT/WDMamba}
MB_REPO=${MB_REPO:-$EXT/MB-TaylorFormerV2}
LOG=$EVID/v22_c8_full_asset_audit.log
mkdir -p "$EVID" "$EXT"
for s in c8_0 c8_1 c8_2 c8_3; do echo "v22_${s}_start $(date --iso-8601=seconds)" >> "$EVID/status_${s}.txt"; done
{
  echo "===== v22_c8_full_asset_audit_start $(date --iso-8601=seconds) ====="
  echo "root=$ROOT"
  echo "python=$PY"
  echo "locked_test_touched=false"
  cd "$EXT"
  if [ ! -d "$WD_REPO/.git" ]; then git clone --depth 1 https://github.com/SunJ000/WDMamba.git "$WD_REPO"; else git -C "$WD_REPO" fetch --depth 1 origin main || true; fi
  if [ ! -d "$MB_REPO/.git" ]; then git clone --depth 1 https://github.com/FVL2020/MB-TaylorFormerV2.git "$MB_REPO"; else git -C "$MB_REPO" fetch --depth 1 origin main || true; fi
  mkdir -p "$EVID/download_probe"
  for item in \
    'WDMamba_models|https://pan.baidu.com/s/1HIs-nHXEaLxwBb1279PVbw?pwd=98j9' \
    'WDMamba_results|https://pan.baidu.com/s/1VdqpPY-Y1gMmpK4ej37wmg?pwd=y9e6' \
    'MBTaylor_models|https://pan.baidu.com/s/11V-wD01rPTHMFFJyjB0R0w' \
    'UDPNet_models|https://pan.baidu.com/s/1JqB-YBPzZAiQsdLlNcidLQ?pwd=2026'; do
    name=${item%%|*}; url=${item#*|}; safe=$(echo "$name" | tr A-Z a-z)
    set +e
    curl -L --max-time 20 -D "$EVID/download_probe/${safe}_headers.txt" -o "$EVID/download_probe/${safe}_body.html" "$url"
    rc=$?
    set -e
    bytes=$(wc -c < "$EVID/download_probe/${safe}_body.html" 2>/dev/null || echo 0)
    echo "download_probe $name rc=$rc bytes=$bytes"
  done
  cd "$ROOT"
  "$PY" experience_docx/tools/audit_haze4k_v22_c8_mini.py \
    --root "$ROOT" --out "$EVID" --data "$DATA" --split-json "$SPLIT_JSON" \
    --a0 "$A0" --udp-repo "$UDP_REPO" --udp-ckpt "$UDP_CKPT" \
    --wd-repo "$WD_REPO" --mb-repo "$MB_REPO"
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
for s in c8_0 c8_1 c8_2 c8_3; do
  echo "v22_${s}_done rc=$rc $(date --iso-8601=seconds)" >> "$EVID/status_${s}.txt"
  if [ "$rc" -eq 0 ]; then echo "V22_${s^^}_OK $(date --iso-8601=seconds)" >> "$EVID/status_${s}.txt"; else echo "V22_${s^^}_FAILED rc=$rc $(date --iso-8601=seconds)" >> "$EVID/status_${s}.txt"; fi
done
exit "$rc"
