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

STAGE=$DATA_ROOT/.RESIDE.prepare-20260722
if [[ -d "$STAGE" ]]; then
  for path in \
    "$STAGE/official/ITS/train/ITS_clear" \
    "$STAGE/official/ITS/train/ITS_haze" \
    "$STAGE/official/ITS/train/ITS_trans" \
    "$STAGE/official/ITS/val/clear" \
    "$STAGE/official/ITS/val/haze" \
    "$STAGE/official/ITS/val/trans" \
    "$STAGE/official/OTS_ALPHA/clear_images" \
    "$STAGE/official/OTS_ALPHA/depth" \
    "$STAGE/official/OTS_ALPHA/OTS" \
    "$STAGE/official/SOTS/nyuhaze500/gt" \
    "$STAGE/official/SOTS/nyuhaze500/hazy" \
    "$STAGE/official/SOTS/indoor/gt" \
    "$STAGE/official/SOTS/indoor/hazy" \
    "$STAGE/official/SOTS/outdoor/gt" \
    "$STAGE/official/SOTS/outdoor/hazy"; do
    if [[ -d "$path" ]]; then
      printf 'RESIDE_STAGE_FILE_COUNT=%s\t%s\n' \
        "$(/usr/bin/find "$path" -maxdepth 1 -type f -printf x | /usr/bin/wc -c)" "$path"
    else
      printf 'RESIDE_STAGE_DIR_ABSENT=%s\n' "$path"
    fi
  done
  if [[ -f "$STAGE/.combined/depth.full.zip" ]]; then
    printf 'RESIDE_STAGE_DEPTH_COMBINED_BYTES=%s\n' "$(/usr/bin/stat -c %s "$STAGE/.combined/depth.full.zip")"
    /usr/bin/python3 -c '
import collections
import zipfile

path = "/sda/home/wangyuxin/ConvIR-B/datasets/.RESIDE.prepare-20260722/.combined/depth.full.zip"
with zipfile.ZipFile(path) as archive:
    infos = archive.infolist()
    methods = collections.Counter(info.compress_type for info in infos)
    encrypted = sum(bool(info.flag_bits & 1) for info in infos)
    first = infos[0]
    print(f"RESIDE_STAGE_DEPTH_ZIP_ENTRIES={len(infos)}")
    print("RESIDE_STAGE_DEPTH_ZIP_METHODS=" + ",".join(f"{key}:{value}" for key, value in sorted(methods.items())))
    print(f"RESIDE_STAGE_DEPTH_ZIP_ENCRYPTED={encrypted}")
    print(f"RESIDE_STAGE_DEPTH_ZIP_FIRST={first.filename}|method={first.compress_type}|flags={first.flag_bits}")
    try:
        with archive.open(first) as stream:
            stream.read(1)
        print("RESIDE_STAGE_DEPTH_ZIP_FIRST_READ=OK")
    except Exception as error:
        print(f"RESIDE_STAGE_DEPTH_ZIP_FIRST_READ={type(error).__name__}:{error}")
'
  fi
  if [[ -f "$STAGE/ARCHIVE_SHA256SUMS.txt" ]]; then
    printf 'RESIDE_STAGE_SHA256_LINES=%s\n' "$(/usr/bin/wc -l < "$STAGE/ARCHIVE_SHA256SUMS.txt")"
  fi
fi

echo "RESIDE_ARCHIVE_INVENTORY_OK"
