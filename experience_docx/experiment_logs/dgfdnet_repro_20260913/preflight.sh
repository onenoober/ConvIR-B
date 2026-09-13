#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913
EVID=/sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/ITS/
STATUS=$EVID/status.txt
LOG=$EVID/preflight_retry.log

mkdir -p "$EVID"
printf '%s\n' "PREFLIGHT_RUNNING_RETRY $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$REMOTE_ROOT/Dehazing/ITS"
set +e
{
  printf '%s\n' "REMOTE_ROOT=$REMOTE_ROOT"
  printf '%s\n' "SOURCE_COMMIT=5b389b83a7e82d6f596ddf52e713f55340720ae9"
  printf '%s\n' "PY=$PY"
  "$PY" -V
  "$PY" -m py_compile data/*.py models/*.py eval.py main.py train.py utils.py valid.py vis.py
  "$PY" - <<'PY'
import torch
from models.DGFDNet import build_net
print("IMPORT_OK")
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "AVAILABLE", torch.cuda.is_available())
model = build_net()
print("PARAMS", sum(p.numel() for p in model.parameters()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
x = torch.rand(1, 3, 64, 64, device=device)
with torch.no_grad():
    y = model(x)
print("FORWARD_SHAPE", tuple(y.shape))
print("FORWARD_FINITE", bool(torch.isfinite(y).all().item()))
PY
  test -d "$DATA"
  test -d /sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/hazy
  test -d /sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/gt
  printf '%s\n' "DATA_DIR=$DATA"
  printf '%s\n' "HAZY_COUNT=$(find /sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/hazy -maxdepth 1 -type f | wc -l)"
  printf '%s\n' "GT_COUNT=$(find /sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/gt -maxdepth 1 -type f | wc -l)"
  printf 'DGFD_PREFLIGHT_OK\n'
} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -eq 0 ]]; then
  printf '%s\n' "PREFLIGHT_DONE rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  printf '%s\n' "PREFLIGHT_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
