#!/usr/bin/env bash
set -euo pipefail

PYTHON=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python

for tool in h5dump h5ls matlab octave python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf 'RESIDE_READER_TOOL=%s\n' "$(command -v "$tool")"
  fi
done

for python_bin in "$PYTHON" /usr/bin/python3; do
  [[ -x "$python_bin" ]] || continue
  printf 'RESIDE_READER_PYTHON=%s\n' "$python_bin"
  "$python_bin" - <<'PY'
import importlib.util

for name in ("h5py", "tables", "mat73", "hdf5storage", "cv2", "torch"):
    print(f"RESIDE_READER_MODULE={name}:{bool(importlib.util.find_spec(name))}")
PY
done
echo "RESIDE_MEASUREMENT_READERS_INVENTORY_OK"
