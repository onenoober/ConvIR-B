#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/sda/home/wangyuxin/ConvIR-B/datasets
STAGE=$DATA_ROOT/.RESIDE.prepare-20260722
FINAL=$DATA_ROOT/RESIDE
PYTHON=/usr/bin/python3
PAIR_REPORT=$STAGE/PAIRING_VALIDATION.txt
LAYOUT_REPORT=$STAGE/DATASET_LAYOUT.txt
PUBLISHED=0
CREATED_LINKS=()

fail() {
  echo "RESIDE_PUBLISH_FAILED: $*" >&2
  exit 2
}

cleanup() {
  local path
  if [[ "$PUBLISHED" == 0 ]]; then
    for path in "${CREATED_LINKS[@]}"; do
      [[ -L "$path" ]] && /bin/rm -- "$path"
    done
    [[ -f "$PAIR_REPORT.tmp" ]] && /bin/rm -- "$PAIR_REPORT.tmp"
    [[ -f "$LAYOUT_REPORT.tmp" ]] && /bin/rm -- "$LAYOUT_REPORT.tmp"
    [[ -f "$PAIR_REPORT" ]] && /bin/rm -- "$PAIR_REPORT"
    [[ -f "$LAYOUT_REPORT" ]] && /bin/rm -- "$LAYOUT_REPORT"
  fi
}
trap cleanup EXIT

[[ -x "$PYTHON" ]] || fail "missing required Python: $PYTHON"
[[ -d "$STAGE" ]] || fail "missing exact staging directory: $STAGE"
[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "refusing to overwrite final target: $FINAL"
[[ -f "$STAGE/ARCHIVE_SHA256SUMS.txt" ]] || fail "missing archive SHA-256 manifest"
[[ "$(/usr/bin/wc -l < "$STAGE/ARCHIVE_SHA256SUMS.txt")" == 26 ]] || fail "archive SHA-256 manifest must contain 26 lines"
[[ -f "$DATA_ROOT/SOTS.zip" ]] || fail "original SOTS.zip is missing"
[[ ! -e "$PAIR_REPORT" ]] || fail "pairing report already exists: $PAIR_REPORT"
[[ ! -e "$LAYOUT_REPORT" ]] || fail "layout report already exists: $LAYOUT_REPORT"

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
SOTS_OUTER=$STAGE/.sots_outer
INDOOR_RAR=$SOTS_OUTER/SOTS/indoor.rar
OUTDOOR_ZIP=$SOTS_OUTER/SOTS/outdoor.zip
[[ -f "$INDOOR_RAR" && -f "$OUTDOOR_ZIP" ]] || fail "nested SOTS archive copies are not in the expected paths"

"$PYTHON" - \
  "$ITS_TRAIN_CLEAR" "$ITS_TRAIN_HAZE" "$ITS_TRAIN_TRANS" \
  "$ITS_VAL_CLEAR" "$ITS_VAL_HAZE" "$ITS_VAL_TRANS" \
  "$OTS_CLEAR" "$OTS_HAZE" "$OTS_DEPTH" \
  "$SOTS_INDOOR_GT" "$SOTS_INDOOR_HAZY" \
  "$SOTS_OUTDOOR_GT" "$SOTS_OUTDOOR_HAZY" "$OUTDOOR_ZIP" > "$PAIR_REPORT.tmp" <<'PY'
import collections
import pathlib
import sys
import zipfile

paths = [pathlib.Path(value) for value in sys.argv[1:-1]]
outdoor_zip_path = pathlib.Path(sys.argv[-1])
image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def checked_files(label, path, expected_count, allowed_suffixes):
    if not path.is_dir():
        raise SystemExit(f"{label}: missing directory {path}")
    files = [item for item in path.iterdir() if item.is_file()]
    if len(files) != expected_count:
        raise SystemExit(f"{label}: expected {expected_count} files, found {len(files)}")
    suffix_counts = collections.Counter(item.suffix.lower() for item in files)
    unexpected = sorted(suffix for suffix in suffix_counts if suffix not in allowed_suffixes)
    if unexpected:
        raise SystemExit(f"{label}: unexpected suffixes {unexpected}")
    distribution = ",".join(f"{suffix or '<none>'}:{count}" for suffix, count in sorted(suffix_counts.items()))
    print(f"{label}_FILES={len(files)}")
    print(f"{label}_SUFFIXES={distribution}")
    return files

def keyed(files):
    return {item.stem for item in files}

def grouped(files):
    return collections.Counter(item.stem.split("_", 1)[0] for item in files)

def require_grouping(label, derived_files, reference_files, expected_per_reference):
    reference_ids = keyed(reference_files)
    if len(reference_ids) != len(reference_files):
        raise SystemExit(f"{label}: duplicate reference stems")
    counts = grouped(derived_files)
    unexpected = sorted(set(counts) - reference_ids)
    missing = sorted(reference_ids - set(counts))
    wrong = sorted((key, value) for key, value in counts.items() if key in reference_ids and value != expected_per_reference)
    if unexpected or missing or wrong:
        raise SystemExit(
            f"{label}: unexpected={len(unexpected)} first={unexpected[:5]}; "
            f"missing={len(missing)} first={missing[:5]}; "
            f"wrong_multiplicity={len(wrong)} first={wrong[:5]}"
        )
    print(f"{label}_PAIRED_IDS={len(counts)}")
    print(f"{label}_FILES_PER_ID={expected_per_reference}")

def require_grouping_set(label, derived_files, reference_files):
    reference_ids = keyed(reference_files)
    if len(reference_ids) != len(reference_files):
        raise SystemExit(f"{label}: duplicate reference stems")
    counts = grouped(derived_files)
    unexpected = sorted(set(counts) - reference_ids)
    missing = sorted(reference_ids - set(counts))
    if unexpected or missing:
        raise SystemExit(
            f"{label}: unexpected={len(unexpected)} first={unexpected[:5]}; "
            f"missing={len(missing)} first={missing[:5]}"
        )
    multiplicities = collections.Counter(counts.values())
    distribution = ",".join(f"{key}:{value}" for key, value in sorted(multiplicities.items()))
    print(f"{label}_PAIRED_IDS={len(counts)}")
    print(f"{label}_FILES_PER_ID_DISTRIBUTION={distribution}")

(its_train_clear_path, its_train_haze_path, its_train_trans_path,
 its_val_clear_path, its_val_haze_path, its_val_trans_path,
 ots_clear_path, ots_haze_path, ots_depth_path,
 sots_indoor_gt_path, sots_indoor_hazy_path,
 sots_outdoor_gt_path, sots_outdoor_hazy_path) = paths

its_train_clear = checked_files("ITS_TRAIN_CLEAR", its_train_clear_path, 10000, {".png"})
its_train_haze = checked_files("ITS_TRAIN_HAZE", its_train_haze_path, 100000, {".png"})
its_train_trans = checked_files("ITS_TRAIN_TRANS", its_train_trans_path, 100000, {".png"})
its_val_clear = checked_files("ITS_VAL_CLEAR", its_val_clear_path, 1000, {".png"})
its_val_haze = checked_files("ITS_VAL_HAZE", its_val_haze_path, 10000, {".png"})
its_val_trans = checked_files("ITS_VAL_TRANS", its_val_trans_path, 10000, {".png"})
ots_clear = checked_files("OTS_CLEAR", ots_clear_path, 8970, {".jpg", ".jpeg"})
ots_haze = checked_files("OTS_HAZE", ots_haze_path, 313950, {".jpg", ".jpeg"})
ots_depth = checked_files("OTS_DEPTH", ots_depth_path, 8970, {".mat"})
sots_indoor_gt = checked_files("SOTS_INDOOR_GT", sots_indoor_gt_path, 50, image_suffixes)
sots_indoor_hazy = checked_files("SOTS_INDOOR_HAZY", sots_indoor_hazy_path, 500, image_suffixes)
sots_outdoor_gt = checked_files("SOTS_OUTDOOR_GT", sots_outdoor_gt_path, 492, image_suffixes)
sots_outdoor_hazy = checked_files("SOTS_OUTDOOR_HAZY", sots_outdoor_hazy_path, 500, image_suffixes)
with zipfile.ZipFile(outdoor_zip_path) as archive:
    infos = [info for info in archive.infolist() if not info.is_dir()]
    archive_gt_names = [pathlib.PurePosixPath(info.filename).name for info in infos
                        if pathlib.PurePosixPath(info.filename).parent.name.lower() == "gt"]
    archive_hazy_names = [pathlib.PurePosixPath(info.filename).name for info in infos
                          if pathlib.PurePosixPath(info.filename).parent.name.lower() == "hazy"]
if set(archive_gt_names) != {item.name for item in sots_outdoor_gt}:
    raise SystemExit("SOTS_OUTDOOR_GT: extracted names differ from nested official ZIP")
if set(archive_hazy_names) != {item.name for item in sots_outdoor_hazy}:
    raise SystemExit("SOTS_OUTDOOR_HAZY: extracted names differ from nested official ZIP")
print(f"SOTS_OUTDOOR_ARCHIVE_GT_ENTRIES={len(archive_gt_names)}")
print(f"SOTS_OUTDOOR_ARCHIVE_GT_UNIQUE_NAMES={len(set(archive_gt_names))}")
print(f"SOTS_OUTDOOR_ARCHIVE_HAZY_ENTRIES={len(archive_hazy_names)}")
print(f"SOTS_OUTDOOR_ARCHIVE_HAZY_UNIQUE_NAMES={len(set(archive_hazy_names))}")

require_grouping("ITS_TRAIN_HAZE", its_train_haze, its_train_clear, 10)
require_grouping("ITS_TRAIN_TRANS", its_train_trans, its_train_clear, 10)
require_grouping("ITS_VAL_HAZE", its_val_haze, its_val_clear, 10)
require_grouping("ITS_VAL_TRANS", its_val_trans, its_val_clear, 10)
require_grouping("OTS_HAZE", ots_haze, ots_clear, 35)
if keyed(ots_depth) != keyed(ots_clear):
    raise SystemExit("OTS_DEPTH: depth and clear scene ID sets differ")
print(f"OTS_DEPTH_PAIRED_IDS={len(keyed(ots_depth))}")
require_grouping("SOTS_INDOOR_HAZE", sots_indoor_hazy, sots_indoor_gt, 10)
require_grouping_set("SOTS_OUTDOOR_HAZE", sots_outdoor_hazy, sots_outdoor_gt)
print("RESIDE_PAIRING_VALIDATION_OK")
PY
/bin/grep -q '^RESIDE_PAIRING_VALIDATION_OK$' "$PAIR_REPORT.tmp" || fail "pairing validation marker missing"
/bin/mv -- "$PAIR_REPORT.tmp" "$PAIR_REPORT"
/bin/cat "$PAIR_REPORT"

/bin/mkdir -p \
  "$STAGE/convir/reside-indoor/train" "$STAGE/convir/reside-indoor/test" \
  "$STAGE/convir/reside-outdoor/train" "$STAGE/convir/reside-outdoor/test"

create_link() {
  local target=$1
  local path=$2
  [[ ! -e "$path" && ! -L "$path" ]] || fail "compatibility path already exists: $path"
  /bin/ln -s "$target" "$path"
  CREATED_LINKS+=("$path")
  [[ -L "$path" && -d "$path" ]] || fail "compatibility link does not resolve: $path"
}

create_link ../../../official/ITS/train/ITS_clear "$STAGE/convir/reside-indoor/train/gt"
create_link ../../../official/ITS/train/ITS_haze "$STAGE/convir/reside-indoor/train/hazy"
create_link ../../../official/ITS/train/ITS_trans "$STAGE/convir/reside-indoor/train/transmission"
create_link ../../../official/SOTS/indoor/gt "$STAGE/convir/reside-indoor/test/gt"
create_link ../../../official/SOTS/indoor/hazy "$STAGE/convir/reside-indoor/test/hazy"
create_link ../../../official/OTS_ALPHA/clear_images "$STAGE/convir/reside-outdoor/train/gt"
create_link ../../../official/OTS_ALPHA/OTS "$STAGE/convir/reside-outdoor/train/hazy"
create_link ../../../official/OTS_ALPHA/depth "$STAGE/convir/reside-outdoor/train/depth"
create_link ../../../official/SOTS/outdoor/gt "$STAGE/convir/reside-outdoor/test/gt"
create_link ../../../official/SOTS/outdoor/hazy "$STAGE/convir/reside-outdoor/test/hazy"

/bin/cat > "$LAYOUT_REPORT.tmp" <<'EOF'
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
/bin/mv -- "$LAYOUT_REPORT.tmp" "$LAYOUT_REPORT"

[[ "$(/usr/bin/find "$SOTS_OUTER" -mindepth 1 ! -type d -printf x | /usr/bin/wc -c)" == 2 ]] || fail "unexpected non-directory content under $SOTS_OUTER"
[[ -d "$STAGE/.combined" ]] || fail "missing combined-archive staging directory"
[[ "$(/usr/bin/find "$STAGE/.combined" -mindepth 1 -printf x | /usr/bin/wc -c)" == 0 ]] || fail "combined-archive staging directory is not empty"

/bin/rmdir -- "$STAGE/.combined"
/bin/rm -- "$INDOOR_RAR" "$OUTDOOR_ZIP"
/bin/rmdir -- "$SOTS_OUTER/SOTS" "$SOTS_OUTER"
echo "RESIDE_TEMPORARY_ARCHIVE_COPIES_REMOVED=2"

[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || fail "final target appeared during publish: $FINAL"
/bin/mv -T -- "$STAGE" "$FINAL"
PUBLISHED=1

for path in \
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
  [[ -L "$path" && -d "$path" ]] || fail "final compatibility link invalid: $path"
done

echo "RESIDE_FINAL_ROOT=$FINAL"
echo "RESIDE_INDOOR_CONVIR_ROOT=$FINAL/convir/reside-indoor"
echo "RESIDE_OUTDOOR_CONVIR_ROOT=$FINAL/convir/reside-outdoor"
echo "RESIDE_DATASETS_PUBLISH_OK"
