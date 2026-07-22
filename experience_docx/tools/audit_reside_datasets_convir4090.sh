#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets
FINAL=$DATA_ROOT/RESIDE
STAGE=$DATA_ROOT/.RESIDE.prepare-20260722

fail() {
  echo "RESIDE_AUDIT_FAILED: $*" >&2
  exit 2
}

count_files() {
  local path=$1
  local pattern=$2
  /usr/bin/find "$path" -maxdepth 1 -type f -iname "$pattern" -printf x | /usr/bin/wc -c
}

check_count() {
  local label=$1
  local path=$2
  local pattern=$3
  local expected=$4
  local actual
  [[ -d "$path" ]] || fail "missing directory $path"
  actual=$(count_files "$path" "$pattern")
  [[ "$actual" == "$expected" ]] || fail "$label expected $expected, found $actual"
  printf 'RESIDE_AUDIT_COUNT=%s\t%s\n' "$actual" "$label"
}

[[ -d "$FINAL" ]] || fail "missing final dataset root $FINAL"
[[ ! -e "$STAGE" && ! -L "$STAGE" ]] || fail "staging directory still exists: $STAGE"
[[ -f "$FINAL/ARCHIVE_SHA256SUMS.txt" ]] || fail "missing archive SHA-256 manifest"
[[ "$(/usr/bin/wc -l < "$FINAL/ARCHIVE_SHA256SUMS.txt")" == 26 ]] || fail "archive SHA-256 manifest does not contain 26 lines"
[[ -f "$FINAL/DATASET_LAYOUT.txt" ]] || fail "missing dataset layout record"
[[ -f "$FINAL/PAIRING_VALIDATION.txt" ]] || fail "missing pairing validation record"
/bin/grep -q '^RESIDE_PAIRING_VALIDATION_OK$' "$FINAL/PAIRING_VALIDATION.txt" || fail "pairing validation marker missing"

check_count ITS_TRAIN_CLEAR "$FINAL/official/ITS/train/ITS_clear" '*.png' 10000
check_count ITS_TRAIN_HAZE "$FINAL/official/ITS/train/ITS_haze" '*.png' 100000
check_count ITS_TRAIN_TRANS "$FINAL/official/ITS/train/ITS_trans" '*.png' 100000
check_count ITS_VAL_CLEAR "$FINAL/official/ITS/val/clear" '*.png' 1000
check_count ITS_VAL_HAZE "$FINAL/official/ITS/val/haze" '*.png' 10000
check_count ITS_VAL_TRANS "$FINAL/official/ITS/val/trans" '*.png' 10000
check_count OTS_CLEAR "$FINAL/official/OTS_ALPHA/clear_images" '*.jpg' 8970
check_count OTS_DEPTH "$FINAL/official/OTS_ALPHA/depth" '*.mat' 8970
check_count OTS_HAZE "$FINAL/official/OTS_ALPHA/OTS" '*.jpg' 313950
check_count SOTS_INDOOR_GT "$FINAL/official/SOTS/indoor/gt" '*.png' 50
check_count SOTS_INDOOR_HAZY "$FINAL/official/SOTS/indoor/hazy" '*.png' 500
check_count SOTS_OUTDOOR_GT "$FINAL/official/SOTS/outdoor/gt" '*.png' 492
check_count SOTS_OUTDOOR_HAZY "$FINAL/official/SOTS/outdoor/hazy" '*.png' 500

for link in \
  "$FINAL/convir/reside-indoor/train/gt" \
  "$FINAL/convir/reside-indoor/train/hazy" \
  "$FINAL/convir/reside-indoor/train/transmission" \
  "$FINAL/convir/reside-indoor/test/gt" \
  "$FINAL/convir/reside-indoor/test/hazy" \
  "$FINAL/convir/reside-outdoor/train/gt" \
  "$FINAL/convir/reside-outdoor/train/hazy" \
  "$FINAL/convir/reside-outdoor/train/depth" \
  "$FINAL/convir/reside-outdoor/test/gt" \
  "$FINAL/convir/reside-outdoor/test/hazy"; do
  [[ -L "$link" ]] || fail "expected symbolic link: $link"
  [[ -d "$link" ]] || fail "symbolic link does not resolve to a directory: $link"
  printf 'RESIDE_AUDIT_LINK=%s\t%s\n' "$link" "$(/usr/bin/readlink "$link")"
done

printf 'RESIDE_AUDIT_FINAL_BYTES=%s\n' "$(/usr/bin/du -sb "$FINAL" | /usr/bin/cut -f1)"
printf 'RESIDE_AUDIT_FREE_BYTES=%s\n' "$(/bin/df --output=avail -B1 "$DATA_ROOT" | /usr/bin/tail -n 1 | /usr/bin/tr -d ' ')"
echo "RESIDE_AUDIT_FINAL_ROOT=$FINAL"
echo "RESIDE_DATASETS_AUDIT_OK"
