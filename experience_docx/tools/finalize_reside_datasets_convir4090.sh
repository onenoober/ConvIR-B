#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets
STAGE=$DATA_ROOT/.RESIDE.prepare-20260722
FINAL=$DATA_ROOT/RESIDE
PYTHON=/usr/bin/python3
UNZIP=/usr/bin/unzip

fail() {
  echo "RESIDE_FINALIZE_FAILED: $*" >&2
  exit 2
}

count_files() {
  local path=$1
  local pattern=$2
  /usr/bin/find "$path" -maxdepth 1 -type f -iname "$pattern" -printf x | /usr/bin/wc -c
}

expect_count() {
  local label=$1
  local path=$2
  local pattern=$3
  local expected=$4
  local actual
  [[ -d "$path" ]] || fail "missing $label directory: $path"
  actual=$(count_files "$path" "$pattern")
  [[ "$actual" == "$expected" ]] || fail "$label count mismatch: expected $expected, found $actual"
}

[[ -x "$PYTHON" ]] || fail "missing required Python: $PYTHON"
[[ -x "$UNZIP" ]] || fail "missing required unzip: $UNZIP"
[[ -d "$STAGE" ]] || fail "missing staging directory: $STAGE"
[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "refusing to overwrite final target: $FINAL"
[[ -f "$STAGE/ARCHIVE_SHA256SUMS.txt" ]] || fail "missing archive SHA-256 manifest"
[[ "$(/usr/bin/wc -l < "$STAGE/ARCHIVE_SHA256SUMS.txt")" == 26 ]] || fail "archive SHA-256 manifest must contain 26 lines"

ITS_TRAIN_CLEAR=$STAGE/official/ITS/train/ITS_clear
ITS_TRAIN_HAZE=$STAGE/official/ITS/train/ITS_haze
ITS_TRAIN_TRANS=$STAGE/official/ITS/train/ITS_trans
ITS_VAL_CLEAR=$STAGE/official/ITS/val/clear
ITS_VAL_HAZE=$STAGE/official/ITS/val/haze
ITS_VAL_TRANS=$STAGE/official/ITS/val/trans
OTS_CLEAR=$STAGE/official/OTS_ALPHA/clear_images
OTS_DEPTH=$STAGE/official/OTS_ALPHA/depth
OTS_HAZE=$STAGE/official/OTS_ALPHA/OTS
SOTS_ROOT=$STAGE/official/SOTS
SOTS_INDOOR=$SOTS_ROOT/indoor
SOTS_INDOOR_RAW=$SOTS_ROOT/nyuhaze500
SOTS_INDOOR_GT=$SOTS_INDOOR/gt
SOTS_INDOOR_HAZY=$SOTS_INDOOR/hazy
SOTS_OUTDOOR_GT=$SOTS_ROOT/outdoor/gt
SOTS_OUTDOOR_HAZY=$SOTS_ROOT/outdoor/hazy
SOTS_OUTER=$STAGE/.sots_outer
INDOOR_RAR=$SOTS_OUTER/SOTS/indoor.rar
OUTDOOR_ZIP=$SOTS_OUTER/SOTS/outdoor.zip

expect_count its_train_clear "$ITS_TRAIN_CLEAR" '*.png' 10000
expect_count its_train_haze "$ITS_TRAIN_HAZE" '*.png' 100000
expect_count its_train_trans "$ITS_TRAIN_TRANS" '*.png' 100000
expect_count its_val_clear "$ITS_VAL_CLEAR" '*.png' 1000
expect_count its_val_haze "$ITS_VAL_HAZE" '*.png' 10000
expect_count its_val_trans "$ITS_VAL_TRANS" '*.png' 10000
expect_count ots_clear "$OTS_CLEAR" '*.jpg' 8970
expect_count ots_depth "$OTS_DEPTH" '*.mat' 8970
expect_count ots_haze "$OTS_HAZE" '*.jpg' 313950
[[ -f "$INDOOR_RAR" ]] || fail "missing nested indoor RAR"
[[ -f "$OUTDOOR_ZIP" ]] || fail "missing nested outdoor ZIP"
[[ ! -e "$SOTS_INDOOR" ]] || fail "SOTS indoor final name already exists before finalize"
[[ -d "$SOTS_INDOOR_RAW" ]] || fail "missing empty RAR staging directory"
[[ "$(/usr/bin/find "$SOTS_INDOOR_RAW" -type f -printf x | /usr/bin/wc -c)" == 0 ]] || fail "RAR staging directory contains files from failed extractor"
echo "RESIDE_FINALIZE_STATE_OK"

TOOL_TMP=$(/usr/bin/mktemp -d /tmp/reside-7zip.XXXXXX)
cleanup_tools() {
  case "$TOOL_TMP" in
    /tmp/reside-7zip.*) /bin/rm -rf -- "$TOOL_TMP" ;;
    *) echo "RESIDE_TOOL_CLEANUP_REFUSED=$TOOL_TMP" >&2 ;;
  esac
}
trap cleanup_tools EXIT
(
  cd "$TOOL_TMP"
  /usr/bin/apt-get download p7zip-full >/dev/null
)
mapfile -t P7ZIP_DEBS < <(/usr/bin/find "$TOOL_TMP" -maxdepth 1 -type f -name 'p7zip-full_*.deb' -print)
[[ ${#P7ZIP_DEBS[@]} -eq 1 ]] || fail "expected one p7zip-full package, found ${#P7ZIP_DEBS[@]}"
/usr/bin/dpkg-deb -x "${P7ZIP_DEBS[0]}" "$TOOL_TMP/root"
mapfile -t SEVEN_ZIP_BINS < <(/usr/bin/find "$TOOL_TMP/root" -type f -path '*/p7zip/7z' -perm -u+x -print)
[[ ${#SEVEN_ZIP_BINS[@]} -eq 1 ]] || fail "downloaded p7zip-full package did not expose one 7z binary"
SEVEN_ZIP=${SEVEN_ZIP_BINS[0]}
printf 'RESIDE_FINALIZE_7ZIP=%s\n' "$SEVEN_ZIP"

"$SEVEN_ZIP" t -bb0 "$INDOOR_RAR" >/dev/null
echo "RESIDE_FINALIZE_RAR_TEST_OK"
"$SEVEN_ZIP" x -y -aoa -bb0 "$INDOOR_RAR" "-o$SOTS_ROOT" >/dev/null
expect_count sots_indoor_raw_gt "$SOTS_INDOOR_RAW/gt" '*.png' 50
expect_count sots_indoor_raw_hazy "$SOTS_INDOOR_RAW/hazy" '*.png' 500
/bin/mv -- "$SOTS_INDOOR_RAW" "$SOTS_INDOOR"

"$UNZIP" -q "$OUTDOOR_ZIP" -d "$SOTS_ROOT"
expect_count sots_indoor_gt "$SOTS_INDOOR_GT" '*.png' 50
expect_count sots_indoor_hazy "$SOTS_INDOOR_HAZY" '*.png' 500
expect_count sots_outdoor_gt "$SOTS_OUTDOOR_GT" '*.png' 492
expect_count sots_outdoor_hazy "$SOTS_OUTDOOR_HAZY" '*.png' 500
/bin/rm -- "$INDOOR_RAR" "$OUTDOOR_ZIP"
/bin/rmdir -- "$SOTS_OUTER/SOTS" "$SOTS_OUTER"
echo "RESIDE_FINALIZE_SOTS_OK"

PAIR_REPORT=$STAGE/PAIRING_VALIDATION.txt
"$PYTHON" - \
  "$ITS_TRAIN_CLEAR" "$ITS_TRAIN_HAZE" "$ITS_TRAIN_TRANS" \
  "$ITS_VAL_CLEAR" "$ITS_VAL_HAZE" "$ITS_VAL_TRANS" \
  "$OTS_CLEAR" "$OTS_HAZE" "$OTS_DEPTH" \
  "$SOTS_INDOOR_GT" "$SOTS_INDOOR_HAZY" \
  "$SOTS_OUTDOOR_GT" "$SOTS_OUTDOOR_HAZY" > "$PAIR_REPORT" <<'PY'
import pathlib
import sys

paths = [pathlib.Path(value) for value in sys.argv[1:]]

def stems(path):
    return {item.stem for item in path.iterdir() if item.is_file()}

def hazy_ids(path):
    return {item.stem.split("_", 1)[0] for item in path.iterdir() if item.is_file()}

def require(label, observed, expected):
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unexpected or missing:
        raise SystemExit(
            f"{label}: unexpected={len(unexpected)} first={unexpected[:5]}; "
            f"missing={len(missing)} first={missing[:5]}"
        )
    print(f"{label}_PAIRED_IDS={len(observed)}")

(its_train_clear, its_train_haze, its_train_trans,
 its_val_clear, its_val_haze, its_val_trans,
 ots_clear, ots_haze, ots_depth,
 sots_indoor_gt, sots_indoor_hazy,
 sots_outdoor_gt, sots_outdoor_hazy) = paths

require("ITS_TRAIN_HAZE", hazy_ids(its_train_haze), stems(its_train_clear))
require("ITS_TRAIN_TRANS", hazy_ids(its_train_trans), stems(its_train_clear))
require("ITS_VAL_HAZE", hazy_ids(its_val_haze), stems(its_val_clear))
require("ITS_VAL_TRANS", hazy_ids(its_val_trans), stems(its_val_clear))
require("OTS_HAZE", hazy_ids(ots_haze), stems(ots_clear))
if stems(ots_depth) != stems(ots_clear):
    raise SystemExit("OTS_DEPTH: depth and clear scene ID sets differ")
print(f"OTS_DEPTH_PAIRED_IDS={len(stems(ots_depth))}")
require("SOTS_INDOOR_HAZE", hazy_ids(sots_indoor_hazy), stems(sots_indoor_gt))
require("SOTS_OUTDOOR_HAZE", hazy_ids(sots_outdoor_hazy), stems(sots_outdoor_gt))
print("RESIDE_PAIRING_VALIDATION_OK")
PY
/bin/grep -q '^RESIDE_PAIRING_VALIDATION_OK$' "$PAIR_REPORT" || fail "pairing validation marker missing"
/bin/cat "$PAIR_REPORT"

for link in \
  "$STAGE/convir/reside-indoor/train/gt" \
  "$STAGE/convir/reside-indoor/train/hazy" \
  "$STAGE/convir/reside-indoor/train/transmission" \
  "$STAGE/convir/reside-indoor/test/gt" \
  "$STAGE/convir/reside-indoor/test/hazy" \
  "$STAGE/convir/reside-outdoor/train/gt" \
  "$STAGE/convir/reside-outdoor/train/hazy" \
  "$STAGE/convir/reside-outdoor/train/depth" \
  "$STAGE/convir/reside-outdoor/test/gt" \
  "$STAGE/convir/reside-outdoor/test/hazy"; do
  [[ ! -e "$link" && ! -L "$link" ]] || fail "compatibility path already exists: $link"
done

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

/bin/cat > "$STAGE/DATASET_LAYOUT.txt" <<'EOF'
RESIDE dataset organization
official/ITS: official updated ITS with train and val clear/haze/transmission
official/OTS_ALPHA: official RESIDE-beta OTS clear/depth/haze
official/SOTS: official RESIDE-Standard indoor and outdoor tests
convir/reside-indoor: ConvIR-compatible ITS train plus SOTS-indoor test
convir/reside-outdoor: ConvIR-compatible OTS train plus SOTS-outdoor test
ITS_TRAIN_CLEAR=10000
ITS_TRAIN_HAZE=100000
ITS_TRAIN_TRANS=100000
ITS_VAL_CLEAR=1000
ITS_VAL_HAZE=10000
ITS_VAL_TRANS=10000
OTS_CLEAR=8970
OTS_DEPTH=8970
OTS_HAZE=313950
SOTS_INDOOR_GT=50
SOTS_INDOOR_HAZY=500
SOTS_OUTDOOR_GT=492
SOTS_OUTDOOR_HAZY=500
EOF

[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "final target appeared during finalize"
/bin/rmdir -- "$STAGE/.combined"
/bin/mv -- "$STAGE" "$FINAL"

for link in \
  "$FINAL/convir/reside-indoor/train/hazy" \
  "$FINAL/convir/reside-indoor/test/hazy" \
  "$FINAL/convir/reside-outdoor/train/hazy" \
  "$FINAL/convir/reside-outdoor/test/hazy"; do
  [[ -L "$link" && -d "$link" ]] || fail "final compatibility link invalid: $link"
done

echo "RESIDE_FINAL_ROOT=$FINAL"
echo "RESIDE_INDOOR_CONVIR_ROOT=$FINAL/convir/reside-indoor"
echo "RESIDE_OUTDOOR_CONVIR_ROOT=$FINAL/convir/reside-outdoor"
echo "RESIDE_DATASETS_FINALIZE_OK"
