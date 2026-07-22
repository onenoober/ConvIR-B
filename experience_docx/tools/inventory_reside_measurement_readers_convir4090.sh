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

DEPTH_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/OTS_ALPHA/depth
mapfile -t DEPTH_FILES < <(/usr/bin/find "$DEPTH_ROOT" -maxdepth 1 -type f -name '*.mat' -print | /usr/bin/sort)
[[ ${#DEPTH_FILES[@]} -gt 0 ]] || {
  echo "RESIDE_READER_DEPTH_FAILED: no MAT files" >&2
  exit 2
}
DEPTH_SAMPLE=${DEPTH_FILES[0]}
/usr/bin/octave --quiet --no-gui --eval "s=load('$DEPTH_SAMPLE'); f=fieldnames(s); printf('RESIDE_DEPTH_SAMPLE=%s\n','$DEPTH_SAMPLE'); for i=1:numel(f); v=s.(f{i}); printf('RESIDE_DEPTH_FIELD=%s|class=%s|size=%s|finite=%d|min=%.12g|max=%.12g|std=%.12g\n',f{i},class(v),mat2str(size(v)),all(isfinite(v(:))),min(v(:)),max(v(:)),std(double(v(:)))); end;"
echo "RESIDE_DEPTH_OCTAVE_READ_OK"
