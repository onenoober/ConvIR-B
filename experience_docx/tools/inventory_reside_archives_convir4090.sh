#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "RESIDE_INVENTORY_FAILED: missing data root $DATA_ROOT" >&2
  exit 2
fi

echo "RESIDE_INVENTORY_ROOT=$DATA_ROOT"
/bin/df -B1 "$DATA_ROOT"

for tool in 7zz 7z unzip zip zipinfo bsdtar tar unrar-free unrar rar python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf 'RESIDE_TOOL=%s\n' "$(command -v "$tool")"
  fi
done

for path in \
  /usr/local/bin/7zz /usr/local/bin/7z /usr/local/bin/bsdtar /usr/local/bin/unar \
  /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/7zz \
  /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/7z \
  /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/bsdtar \
  /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/unar; do
  if [[ -x "$path" ]]; then
    printf 'RESIDE_FIXED_TOOL=%s\n' "$path"
  fi
done

if [[ -x /usr/bin/dpkg-query ]]; then
  for package in p7zip-full p7zip-rar libarchive-tools unrar-free unar; do
    if /usr/bin/dpkg-query -W -f='${Status}' "$package" 2>/dev/null | /usr/bin/grep -q 'install ok installed'; then
      printf 'RESIDE_INSTALLED_PACKAGE=%s\n' "$package"
    fi
  done
fi

if [[ -x /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python ]]; then
  /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python -c \
    'import importlib.util; print("RESIDE_PYTHON_MODULES=" + ",".join(name for name in ("rarfile","libarchive","py7zr") if importlib.util.find_spec(name)))'
fi

echo "RESIDE_ARCHIVE_CANDIDATES_BEGIN"
/usr/bin/find "$DATA_ROOT" -maxdepth 5 -type f \
  \( -name 'ITS.zip' -o -name 'ITS.z[0-9][0-9]' \
     -o -name 'clear_images.zip' -o -name 'depth.zip.[0-9][0-9][0-9]' \
     -o -name 'OTS.zip.[0-9][0-9][0-9]' -o -name 'SOTS.zip' \) \
  -printf '%s\t%p\n' | /usr/bin/sort -k2,2
echo "RESIDE_ARCHIVE_CANDIDATES_END"

for path in "$DATA_ROOT/RESIDE" "$DATA_ROOT/.RESIDE.prepare-20260722"; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'RESIDE_TARGET_EXISTS=%s\n' "$path"
    /usr/bin/find "$path" -maxdepth 3 -mindepth 1 -printf '%y\t%p\n' | /usr/bin/sort | /usr/bin/head -n 80
  else
    printf 'RESIDE_TARGET_ABSENT=%s\n' "$path"
  fi
done

echo "RESIDE_ARCHIVE_INVENTORY_OK"
