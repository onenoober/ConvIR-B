#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets
STAGE=$DATA_ROOT/.RESIDE.prepare-20260722
FINAL=$DATA_ROOT/RESIDE
MIN_FREE_BYTES=134217728000

fail() {
  echo "RESIDE_PREPARE_FAILED: $*" >&2
  exit 2
}

require_unique() {
  local label=$1
  local name=$2
  local -n output=$3
  mapfile -t output < <(/usr/bin/find "$DATA_ROOT" -maxdepth 5 -type f -name "$name" -print | /usr/bin/sort)
  [[ ${#output[@]} -eq 1 ]] || fail "expected exactly one $label ($name), found ${#output[@]}"
}

require_file() {
  [[ -f "$1" ]] || fail "missing archive part $1"
}

count_files() {
  local path=$1
  local pattern=$2
  /usr/bin/find "$path" -maxdepth 1 -type f -iname "$pattern" -printf x | /usr/bin/wc -c
}

ZIP=/usr/bin/zip
UNZIP=/usr/bin/unzip
[[ -x "$ZIP" ]] || fail "missing required tool $ZIP"
[[ -x "$UNZIP" ]] || fail "missing required tool $UNZIP"

[[ -d "$DATA_ROOT" ]] || fail "missing data root $DATA_ROOT"
[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "refusing to overwrite existing target $FINAL"
[[ ! -e "$STAGE" && ! -L "$STAGE" ]] || fail "staging path already exists: $STAGE"

free_bytes=$(/bin/df --output=avail -B1 "$DATA_ROOT" | /usr/bin/tail -n 1 | /usr/bin/tr -d ' ')
[[ "$free_bytes" =~ ^[0-9]+$ ]] || fail "could not determine free space"
(( free_bytes >= MIN_FREE_BYTES )) || fail "need at least $MIN_FREE_BYTES free bytes; found $free_bytes"

declare -a ITS_ZIPS CLEAR_ZIPS DEPTH_FIRSTS OTS_FIRSTS SOTS_ZIPS
require_unique ITS ITS.zip ITS_ZIPS
require_unique OTS-clear clear_images.zip CLEAR_ZIPS
require_unique OTS-depth depth.zip.001 DEPTH_FIRSTS
require_unique OTS-haze OTS.zip.001 OTS_FIRSTS
require_unique SOTS SOTS.zip SOTS_ZIPS

ITS_ZIP=${ITS_ZIPS[0]}
CLEAR_ZIP=${CLEAR_ZIPS[0]}
DEPTH_FIRST=${DEPTH_FIRSTS[0]}
OTS_FIRST=${OTS_FIRSTS[0]}
SOTS_ZIP=${SOTS_ZIPS[0]}

ITS_DIR=$(/usr/bin/dirname "$ITS_ZIP")
DEPTH_DIR=$(/usr/bin/dirname "$DEPTH_FIRST")
OTS_DIR=$(/usr/bin/dirname "$OTS_FIRST")
for index in 01 02 03 04 05 06 07 08 09 10; do
  require_file "$ITS_DIR/ITS.z$index"
done
for index in 001 002; do
  require_file "$DEPTH_DIR/depth.zip.$index"
done
for index in 001 002 003 004 005 006 007 008 009 010 011; do
  require_file "$OTS_DIR/OTS.zip.$index"
done

echo "RESIDE_PREPARE_ARCHIVES_OK"
printf 'RESIDE_ZIP_TOOL=%s\n' "$ZIP"
printf 'RESIDE_UNZIP_TOOL=%s\n' "$UNZIP"
printf 'RESIDE_FREE_BYTES=%s\n' "$free_bytes"

RAR_EXTRACTOR=
for candidate in /usr/bin/unrar-free /usr/bin/unrar /usr/local/bin/unar; do
  if [[ -x "$candidate" ]]; then
    RAR_EXTRACTOR=$candidate
    break
  fi
done

TOOL_TMP=
cleanup_tools() {
  if [[ -n "$TOOL_TMP" ]]; then
    case "$TOOL_TMP" in
      /tmp/reside-unrar.*) /bin/rm -rf -- "$TOOL_TMP" ;;
      *) echo "RESIDE_TOOL_CLEANUP_REFUSED=$TOOL_TMP" >&2 ;;
    esac
  fi
}
trap cleanup_tools EXIT

if [[ -z "$RAR_EXTRACTOR" ]]; then
  [[ -x /usr/bin/apt-get ]] || fail "no RAR extractor and apt-get is unavailable"
  [[ -x /usr/bin/dpkg-deb ]] || fail "no RAR extractor and dpkg-deb is unavailable"
  TOOL_TMP=$(/usr/bin/mktemp -d /tmp/reside-unrar.XXXXXX)
  (
    cd "$TOOL_TMP"
    /usr/bin/apt-get download unrar-free >/dev/null
  )
  mapfile -t UNRAR_DEBS < <(/usr/bin/find "$TOOL_TMP" -maxdepth 1 -type f -name 'unrar-free_*.deb' -print)
  [[ ${#UNRAR_DEBS[@]} -eq 1 ]] || fail "expected one downloaded unrar-free package, found ${#UNRAR_DEBS[@]}"
  /usr/bin/dpkg-deb -x "${UNRAR_DEBS[0]}" "$TOOL_TMP/root"
  mapfile -t UNRAR_BINS < <(/usr/bin/find "$TOOL_TMP/root" -type f -name 'unrar-free' -perm -u+x -print)
  [[ ${#UNRAR_BINS[@]} -eq 1 ]] || fail "downloaded package did not expose one unrar-free binary"
  RAR_EXTRACTOR=${UNRAR_BINS[0]}
fi
printf 'RESIDE_RAR_EXTRACTOR=%s\n' "$RAR_EXTRACTOR"

if [[ -z "$TOOL_TMP" ]]; then
  TOOL_TMP=$(/usr/bin/mktemp -d /tmp/reside-unrar.XXXXXX)
fi
"$UNZIP" -p "$SOTS_ZIP" SOTS/indoor.rar > "$TOOL_TMP/indoor.rar"
require_file "$TOOL_TMP/indoor.rar"
"$RAR_EXTRACTOR" t "$TOOL_TMP/indoor.rar"
/bin/rm -- "$TOOL_TMP/indoor.rar"
echo "RESIDE_RAR_PREFLIGHT_OK"

/bin/mkdir -p \
  "$STAGE/official" \
  "$STAGE/official/OTS_ALPHA/depth" \
  "$STAGE/official/SOTS" \
  "$STAGE/convir/reside-indoor/train" \
  "$STAGE/convir/reside-indoor/test" \
  "$STAGE/convir/reside-outdoor/train" \
  "$STAGE/convir/reside-outdoor/test" \
  "$STAGE/.sots_outer" \
  "$STAGE/.combined"

echo "RESIDE_SHA256_BEGIN"
{
  /usr/bin/find "$ITS_DIR" -maxdepth 1 -type f \( -name 'ITS.zip' -o -name 'ITS.z[0-9][0-9]' \) -print0
  printf '%s\0' "$CLEAR_ZIP"
  /usr/bin/find "$DEPTH_DIR" -maxdepth 1 -type f -name 'depth.zip.[0-9][0-9][0-9]' -print0
  /usr/bin/find "$OTS_DIR" -maxdepth 1 -type f -name 'OTS.zip.[0-9][0-9][0-9]' -print0
  printf '%s\0' "$SOTS_ZIP"
} | /usr/bin/sort -z | /usr/bin/xargs -0 /usr/bin/sha256sum | /usr/bin/tee "$STAGE/ARCHIVE_SHA256SUMS.txt"
echo "RESIDE_SHA256_OK"

echo "RESIDE_EXTRACT_ITS_BEGIN"
"$ZIP" -q -s 0 "$ITS_ZIP" --out "$STAGE/.combined/ITS.full.zip"
"$UNZIP" -q "$STAGE/.combined/ITS.full.zip" -d "$STAGE/official"
/bin/rm -- "$STAGE/.combined/ITS.full.zip"
echo "RESIDE_EXTRACT_ITS_OK"

echo "RESIDE_EXTRACT_OTS_CLEAR_BEGIN"
"$UNZIP" -q "$CLEAR_ZIP" -d "$STAGE/official/OTS_ALPHA"
echo "RESIDE_EXTRACT_OTS_CLEAR_OK"

echo "RESIDE_EXTRACT_OTS_DEPTH_BEGIN"
/bin/cat "$DEPTH_DIR"/depth.zip.001 "$DEPTH_DIR"/depth.zip.002 > "$STAGE/.combined/depth.full.zip"
"$UNZIP" -q "$STAGE/.combined/depth.full.zip" -d "$STAGE/official/OTS_ALPHA/depth"
/bin/rm -- "$STAGE/.combined/depth.full.zip"
echo "RESIDE_EXTRACT_OTS_DEPTH_OK"

echo "RESIDE_EXTRACT_OTS_HAZE_BEGIN"
/bin/cat \
  "$OTS_DIR"/OTS.zip.001 "$OTS_DIR"/OTS.zip.002 "$OTS_DIR"/OTS.zip.003 \
  "$OTS_DIR"/OTS.zip.004 "$OTS_DIR"/OTS.zip.005 "$OTS_DIR"/OTS.zip.006 \
  "$OTS_DIR"/OTS.zip.007 "$OTS_DIR"/OTS.zip.008 "$OTS_DIR"/OTS.zip.009 \
  "$OTS_DIR"/OTS.zip.010 "$OTS_DIR"/OTS.zip.011 > "$STAGE/.combined/OTS.full.zip"
"$UNZIP" -q "$STAGE/.combined/OTS.full.zip" -d "$STAGE/official/OTS_ALPHA"
/bin/rm -- "$STAGE/.combined/OTS.full.zip"
echo "RESIDE_EXTRACT_OTS_HAZE_OK"

echo "RESIDE_EXTRACT_SOTS_OUTER_BEGIN"
"$UNZIP" -q "$SOTS_ZIP" -d "$STAGE/.sots_outer"
INDOOR_RAR=$STAGE/.sots_outer/SOTS/indoor.rar
OUTDOOR_ZIP=$STAGE/.sots_outer/SOTS/outdoor.zip
require_file "$INDOOR_RAR"
require_file "$OUTDOOR_ZIP"
"$RAR_EXTRACTOR" x -o+ "$INDOOR_RAR" "$STAGE/official/SOTS/"
"$UNZIP" -q "$OUTDOOR_ZIP" -d "$STAGE/official/SOTS"
/bin/rm -- "$INDOOR_RAR" "$OUTDOOR_ZIP"
/bin/rmdir -- "$STAGE/.sots_outer/SOTS" "$STAGE/.sots_outer"
/bin/rmdir -- "$STAGE/.combined"
echo "RESIDE_EXTRACT_SOTS_OK"

ITS_TRAIN_CLEAR=$STAGE/official/ITS/train/ITS_clear
ITS_TRAIN_HAZE=$STAGE/official/ITS/train/ITS_haze
ITS_TRAIN_TRANS=$STAGE/official/ITS/train/ITS_trans
ITS_VAL_CLEAR=$STAGE/official/ITS/val/clear
ITS_VAL_HAZE=$STAGE/official/ITS/val/haze
ITS_VAL_TRANS=$STAGE/official/ITS/val/trans
OTS_CLEAR=$STAGE/official/OTS_ALPHA/clear_images
OTS_DEPTH=$STAGE/official/OTS_ALPHA/depth
OTS_HAZE=$STAGE/official/OTS_ALPHA/OTS
SOTS_INDOOR_GT=$STAGE/official/SOTS/indoor/gt
SOTS_INDOOR_HAZY=$STAGE/official/SOTS/indoor/hazy
SOTS_OUTDOOR_GT=$STAGE/official/SOTS/outdoor/gt
SOTS_OUTDOOR_HAZY=$STAGE/official/SOTS/outdoor/hazy

for directory in \
  "$ITS_TRAIN_CLEAR" "$ITS_TRAIN_HAZE" "$ITS_TRAIN_TRANS" \
  "$ITS_VAL_CLEAR" "$ITS_VAL_HAZE" "$ITS_VAL_TRANS" \
  "$OTS_CLEAR" "$OTS_DEPTH" "$OTS_HAZE" \
  "$SOTS_INDOOR_GT" "$SOTS_INDOOR_HAZY" \
  "$SOTS_OUTDOOR_GT" "$SOTS_OUTDOOR_HAZY"; do
  [[ -d "$directory" ]] || fail "missing extracted directory $directory"
done

declare -A EXPECTED_COUNTS=(
  [its_train_clear]=10000
  [its_train_haze]=100000
  [its_train_trans]=100000
  [its_val_clear]=1000
  [its_val_haze]=10000
  [its_val_trans]=10000
  [ots_clear]=8970
  [ots_depth]=8970
  [ots_haze]=313950
  [sots_outdoor_gt]=492
  [sots_outdoor_hazy]=500
)
declare -A ACTUAL_COUNTS=(
  [its_train_clear]=$(count_files "$ITS_TRAIN_CLEAR" '*.png')
  [its_train_haze]=$(count_files "$ITS_TRAIN_HAZE" '*.png')
  [its_train_trans]=$(count_files "$ITS_TRAIN_TRANS" '*.png')
  [its_val_clear]=$(count_files "$ITS_VAL_CLEAR" '*.png')
  [its_val_haze]=$(count_files "$ITS_VAL_HAZE" '*.png')
  [its_val_trans]=$(count_files "$ITS_VAL_TRANS" '*.png')
  [ots_clear]=$(count_files "$OTS_CLEAR" '*.jpg')
  [ots_depth]=$(count_files "$OTS_DEPTH" '*.mat')
  [ots_haze]=$(count_files "$OTS_HAZE" '*.jpg')
  [sots_outdoor_gt]=$(count_files "$SOTS_OUTDOOR_GT" '*')
  [sots_outdoor_hazy]=$(count_files "$SOTS_OUTDOOR_HAZY" '*')
)

for key in "${!EXPECTED_COUNTS[@]}"; do
  [[ "${ACTUAL_COUNTS[$key]}" == "${EXPECTED_COUNTS[$key]}" ]] || \
    fail "count mismatch for $key: expected ${EXPECTED_COUNTS[$key]}, found ${ACTUAL_COUNTS[$key]}"
done

SOTS_INDOOR_GT_COUNT=$(count_files "$SOTS_INDOOR_GT" '*')
SOTS_INDOOR_HAZY_COUNT=$(count_files "$SOTS_INDOOR_HAZY" '*')
[[ "$SOTS_INDOOR_GT_COUNT" == 50 ]] || fail "SOTS indoor GT count mismatch: expected 50, found $SOTS_INDOOR_GT_COUNT"
[[ "$SOTS_INDOOR_HAZY_COUNT" == 500 ]] || fail "SOTS indoor hazy count mismatch: expected 500, found $SOTS_INDOOR_HAZY_COUNT"

/bin/ln -s ../../../official/ITS/train/ITS_clear "$STAGE/convir/reside-indoor/train/gt"
/bin/ln -s ../../../official/ITS/train/ITS_haze "$STAGE/convir/reside-indoor/train/hazy"
/bin/ln -s ../../../official/ITS/train/ITS_trans "$STAGE/convir/reside-indoor/train/transmission"
/bin/ln -s ../../../official/SOTS/indoor/gt "$STAGE/convir/reside-indoor/test/gt"
/bin/ln -s ../../../official/SOTS/indoor/hazy "$STAGE/convir/reside-indoor/test/hazy"
/bin/ln -s ../../../official/OTS_ALPHA/clear_images "$STAGE/convir/reside-outdoor/train/gt"
/bin/ln -s ../../../official/OTS_ALPHA/OTS "$STAGE/convir/reside-outdoor/train/hazy"
/bin/ln -s ../../../official/OTS_ALPHA/depth "$STAGE/convir/reside-outdoor/train/depth"
/bin/ln -s ../../../official/SOTS/outdoor/gt "$STAGE/convir/reside-outdoor/test/gt"
/bin/ln -s ../../../official/SOTS/outdoor/hazy "$STAGE/convir/reside-outdoor/test/hazy"

{
  echo "RESIDE dataset organization"
  echo "official/ITS: official updated ITS with train and val clear/haze/transmission"
  echo "official/OTS_ALPHA: official RESIDE-beta OTS clear/depth/haze"
  echo "official/SOTS: official RESIDE-Standard indoor and outdoor tests"
  echo "convir/reside-indoor: ConvIR-compatible ITS train plus SOTS-indoor test"
  echo "convir/reside-outdoor: ConvIR-compatible OTS train plus SOTS-outdoor test"
  echo "ITS_TRAIN_CLEAR=${ACTUAL_COUNTS[its_train_clear]}"
  echo "ITS_TRAIN_HAZE=${ACTUAL_COUNTS[its_train_haze]}"
  echo "ITS_TRAIN_TRANS=${ACTUAL_COUNTS[its_train_trans]}"
  echo "ITS_VAL_CLEAR=${ACTUAL_COUNTS[its_val_clear]}"
  echo "ITS_VAL_HAZE=${ACTUAL_COUNTS[its_val_haze]}"
  echo "ITS_VAL_TRANS=${ACTUAL_COUNTS[its_val_trans]}"
  echo "OTS_CLEAR=${ACTUAL_COUNTS[ots_clear]}"
  echo "OTS_DEPTH=${ACTUAL_COUNTS[ots_depth]}"
  echo "OTS_HAZE=${ACTUAL_COUNTS[ots_haze]}"
  echo "SOTS_INDOOR_GT=$SOTS_INDOOR_GT_COUNT"
  echo "SOTS_INDOOR_HAZY=$SOTS_INDOOR_HAZY_COUNT"
  echo "SOTS_OUTDOOR_GT=${ACTUAL_COUNTS[sots_outdoor_gt]}"
  echo "SOTS_OUTDOOR_HAZY=${ACTUAL_COUNTS[sots_outdoor_hazy]}"
} | /usr/bin/tee "$STAGE/DATASET_LAYOUT.txt"

[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "target appeared during extraction: $FINAL"
/bin/mv -- "$STAGE" "$FINAL"

[[ -d "$FINAL/official/ITS" ]] || fail "final ITS directory missing after promotion"
[[ -d "$FINAL/official/OTS_ALPHA" ]] || fail "final OTS directory missing after promotion"
[[ -d "$FINAL/official/SOTS" ]] || fail "final SOTS directory missing after promotion"
[[ -L "$FINAL/convir/reside-indoor/train/hazy" ]] || fail "final indoor compatibility link missing"
[[ -L "$FINAL/convir/reside-outdoor/train/hazy" ]] || fail "final outdoor compatibility link missing"

echo "RESIDE_FINAL_ROOT=$FINAL"
echo "RESIDE_INDOOR_CONVIR_ROOT=$FINAL/convir/reside-indoor"
echo "RESIDE_OUTDOOR_CONVIR_ROOT=$FINAL/convir/reside-outdoor"
echo "RESIDE_DATASETS_PREPARE_OK"
