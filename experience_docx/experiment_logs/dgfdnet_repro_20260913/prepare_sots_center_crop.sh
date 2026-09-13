#!/usr/bin/env bash
set -euo pipefail

PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
SRC=/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor
ROOT=/sda/home/wangyuxin/ConvIR-B/datasets/DGFDNet_SOTS_center_crop_20260913
EVID=/sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913
STATUS=$EVID/status.txt
LOG=$EVID/prepare_sots_center_crop.log

mkdir -p "$ROOT/ITS" "$ROOT/SOTS/indoor/gt"
if [[ ! -e "$ROOT/SOTS/indoor/hazy" ]]; then
  ln -s "$SRC/hazy" "$ROOT/SOTS/indoor/hazy"
fi
printf '%s\n' "DERIVED_DATA_RESUME_ROOT=$ROOT" | tee -a "$STATUS"
set +e
ROOT="$ROOT" SRC="$SRC" "$PY" - <<'PY' 2>&1 | tee "$LOG"
import os
from pathlib import Path
from PIL import Image

src = Path(os.environ["SRC"])
root = Path(os.environ["ROOT"])
hazy_dir = src / "hazy"
gt_dir = src / "gt"
out_dir = root / "SOTS" / "indoor" / "gt"
names = sorted(p.name for p in hazy_dir.iterdir() if p.is_file())
for hazy_name in names:
    prefix = hazy_name.split("_")[0]
    output_path = out_dir / f"{prefix}.png"
    if output_path.exists():
        continue
    with Image.open(hazy_dir / hazy_name) as hazy, Image.open(gt_dir / f"{prefix}.png") as gt:
        width, height = hazy.size
        left = (gt.width - width) // 2
        top = (gt.height - height) // 2
        cropped = gt.crop((left, top, left + width, top + height))
        cropped.save(output_path)
print("DERIVED_HAZY_LINK", root / "SOTS" / "indoor" / "hazy")
count = len(list(out_dir.glob("*.png")))
print("DERIVED_GT_COUNT", count)
if count != 50:
    raise RuntimeError(f"Expected 50 derived gt files, found {count}")
print("DERIVED_GT_SIZE", Image.open(next(out_dir.glob("*.png"))).size)
print("DGFD_SOTS_CENTER_CROP_PREP_OK")
PY
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -ne 0 ]]; then
  printf '%s\n' "DERIVED_DATA_PREP_FAILED rc=$rc" | tee -a "$STATUS"
  exit "$rc"
fi
printf '%s\n' "DERIVED_DATA_PREP_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
