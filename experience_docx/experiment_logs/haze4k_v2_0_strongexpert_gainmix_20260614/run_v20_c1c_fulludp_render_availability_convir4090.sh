#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix-c1}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
UDP_REPO=${UDP_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/UDPNet}
UDP_URL=${UDP_URL:-https://github.com/Harbinzzy/UDPNet.git}
UDP_COMMIT=${UDP_COMMIT:-f925387e690ae6016ffbd4b1cfd8490d75d7a334}
CKPT_DIR=${CKPT_DIR:-/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-$CKPT_DIR/ConvIR_UDPNet_haze4k.ckpt}
EXPECTED_SHA=${EXPECTED_SHA:-6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291}
STATUS=$EVID/status_c1c.txt
LOG=$EVID/v20_c1c_fulludp_render_availability.log
JSON_OUT=$EVID/v20_c1c_fulludp_render_availability.json
MD_OUT=$EVID/v20_c1c_fulludp_render_availability.md

mkdir -p "$EVID" "$CKPT_DIR" "$(dirname "$UDP_REPO")"
{
  echo "v20_c1c_fulludp_render_availability_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "udp_repo=$UDP_REPO"
  echo "udp_url=$UDP_URL"
  echo "udp_commit=$UDP_COMMIT"
  echo "official_ckpt=$OFFICIAL_CKPT"
  echo "expected_sha=$EXPECTED_SHA"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,120p' | sed 's/^/git_status=/'
  fi
} | tee -a "$STATUS"

set +e
{
  set -euo pipefail
  if [ ! -d "$UDP_REPO/.git" ]; then
    git clone "$UDP_URL" "$UDP_REPO"
  fi
  if git -C "$UDP_REPO" cat-file -e "$UDP_COMMIT^{commit}" 2>/dev/null; then
    git -C "$UDP_REPO" fetch --all --tags --prune || printf 'udp_repo_fetch_warning=using_existing_commit\n'
  else
    git -C "$UDP_REPO" fetch --all --tags --prune
  fi
  git -C "$UDP_REPO" checkout "$UDP_COMMIT"
  repo_head=$(git -C "$UDP_REPO" rev-parse HEAD)
  model_file=$UDP_REPO/Dehazing/ITS/models/ConvIR_UDPNet.py
  checkpoint_exists=false
  checkpoint_sha=""
  checkpoint_size=0
  checkpoint_sha_match=false
  if [ -f "$OFFICIAL_CKPT" ]; then
    checkpoint_exists=true
    checkpoint_sha=$(sha256sum "$OFFICIAL_CKPT" | awk '{print $1}')
    checkpoint_size=$(stat -c '%s' "$OFFICIAL_CKPT")
    if [ "$checkpoint_sha" = "$EXPECTED_SHA" ]; then
      checkpoint_sha_match=true
    fi
  fi
  baidupcs_path=$(command -v BaiduPCS-Go || command -v baidupcs-go || command -v baidupcs || true)
  render_ready=false
  decision=C1C_FULLUDP_RENDER_BLOCKED_CHECKPOINT_MISSING
  if [ "$checkpoint_exists" = true ] && [ "$checkpoint_sha_match" = true ] && [ -f "$model_file" ]; then
    render_ready=true
    decision=C1C_FULLUDP_RENDER_READY
  elif [ "$checkpoint_exists" = true ] && [ "$checkpoint_sha_match" != true ]; then
    decision=C1C_FULLUDP_RENDER_BLOCKED_CHECKPOINT_SHA_MISMATCH
  fi
  export C1C_REMOTE_ROOT="$REMOTE_ROOT"
  export C1C_UDP_REPO="$UDP_REPO"
  export C1C_UDP_URL="$UDP_URL"
  export C1C_UDP_COMMIT="$UDP_COMMIT"
  export C1C_REPO_HEAD="$repo_head"
  export C1C_MODEL_FILE="$model_file"
  export C1C_OFFICIAL_CKPT="$OFFICIAL_CKPT"
  export C1C_EXPECTED_SHA="$EXPECTED_SHA"
  export C1C_CHECKPOINT_EXISTS="$checkpoint_exists"
  export C1C_CHECKPOINT_SHA="$checkpoint_sha"
  export C1C_CHECKPOINT_SIZE="$checkpoint_size"
  export C1C_CHECKPOINT_SHA_MATCH="$checkpoint_sha_match"
  export C1C_BAIDUPCS_PATH="$baidupcs_path"
  export C1C_RENDER_READY="$render_ready"
  export C1C_DECISION="$decision"
  export C1C_JSON_OUT="$JSON_OUT"
  export C1C_MD_OUT="$MD_OUT"
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

decision = os.environ["C1C_DECISION"]
payload = {
    "route": "Haze4K-v2.0 StrongExpert-GainMix",
    "phase": "C1c FullUDP render availability audit",
    "locked_test_touched": False,
    "remote_root": os.environ["C1C_REMOTE_ROOT"],
    "udp_repo": os.environ["C1C_UDP_REPO"],
    "udp_url": os.environ["C1C_UDP_URL"],
    "udp_commit_expected": os.environ["C1C_UDP_COMMIT"],
    "udp_commit_actual": os.environ["C1C_REPO_HEAD"],
    "model_file": os.environ["C1C_MODEL_FILE"],
    "model_file_exists": Path(os.environ["C1C_MODEL_FILE"]).is_file(),
    "official_checkpoint": os.environ["C1C_OFFICIAL_CKPT"],
    "expected_sha256": os.environ["C1C_EXPECTED_SHA"],
    "checkpoint_exists": os.environ["C1C_CHECKPOINT_EXISTS"] == "true",
    "checkpoint_sha256": os.environ["C1C_CHECKPOINT_SHA"],
    "checkpoint_size": int(os.environ["C1C_CHECKPOINT_SIZE"]),
    "checkpoint_sha_match": os.environ["C1C_CHECKPOINT_SHA_MATCH"] == "true",
    "baidupcs_path": os.environ["C1C_BAIDUPCS_PATH"],
    "render_ready": os.environ["C1C_RENDER_READY"] == "true",
    "decision": decision,
    "next_action": (
        "render FullUDP/A0 outputs and compute output-difference features"
        if decision == "C1C_FULLUDP_RENDER_READY"
        else "supply/copy the verified ConvIR_UDPNet_haze4k.ckpt to convir-4090 or switch to another reproducible strong expert"
    ),
}
Path(os.environ["C1C_JSON_OUT"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Haze4K v2.0 C1c FullUDP Render Availability",
    "",
    f"Decision: `{payload['decision']}`",
    "",
    "Locked test data was not touched.",
    "",
    "## Checks",
    "",
    f"- UDPNet repo: `{payload['udp_repo']}`",
    f"- Expected commit: `{payload['udp_commit_expected']}`",
    f"- Actual commit: `{payload['udp_commit_actual']}`",
    f"- ConvIR_UDPNet model file exists: `{payload['model_file_exists']}`",
    f"- Official checkpoint path: `{payload['official_checkpoint']}`",
    f"- Checkpoint exists: `{payload['checkpoint_exists']}`",
    f"- Checkpoint sha256: `{payload['checkpoint_sha256']}`",
    f"- Expected sha256: `{payload['expected_sha256']}`",
    f"- Checkpoint sha match: `{payload['checkpoint_sha_match']}`",
    f"- BaiduPCS tool path: `{payload['baidupcs_path']}`",
    f"- Render ready: `{payload['render_ready']}`",
    "",
    "## Interpretation",
    "",
    "- C1b showed A0-PSNR-only deployable proxies are not enough for C2.",
    "- C1c checks whether convir-4090 can render FullUDP outputs for real output-difference features.",
    f"- Next action: {payload['next_action']}.",
]
Path(os.environ["C1C_MD_OUT"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("V20_C1C_FULLUDP_RENDER_AVAILABILITY_OK decision=" + payload["decision"])
PY
} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v20_c1c_fulludp_render_availability_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V20_C1C_FULLUDP_RENDER_AVAILABILITY_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V20_C1C_FULLUDP_RENDER_AVAILABILITY_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
