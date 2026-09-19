#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B
REPO="$ROOT/repos/dehazeformer-repro-20260915"
EVID="$ROOT/experience_docx/experiment_logs/dehazeformer_repro_20260915"
PY="$ROOT/envs/convir-cu121/bin/python"
SOTS="$ROOT/datasets/RESIDE/official/SOTS/indoor"
FIXTURE="$ROOT/datasets/RESIDE/derived/dehazeformer_smoke_20260915"
SAVE_ROOT="$ROOT/checkpoints/engineering-only/dehazeformer_smoke_20260915"
RESULT_ROOT="$ROOT/outputs/dehazeformer_smoke_20260915"
STATUS="$EVID/status.txt"
LOG="$EVID/smoke_test.log"
MODEL=dehazeformer-s
SOURCE_COMMIT=5af7c34d6f8e0784a88a2132a7add296bf99ca9c

mkdir -p "$EVID"
printf 'PREFLIGHT_RUNNING run=dehazeformer_smoke_20260915 time=%s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"

run_body() {
  printf 'run_id=dehazeformer_smoke_20260915\n'
  printf 'route_class=new_external_reproduction_smoke\n'
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'repo=%s\npython=%s\ndata=%s\ncheckpoint_root=%s\nresult_root=%s\n' "$REPO" "$PY" "$FIXTURE" "$SAVE_ROOT" "$RESULT_ROOT"
  printf 'locked_test_policy=engineering-only single public SOTS fixture; no benchmark claim\n'
  printf 'checkpoint_policy=random initialization for pipeline validation only\n'

  test -x "$PY"
  test -f "$REPO/test.py"
  grep -q "source_commit=$SOURCE_COMMIT" "$REPO/SOURCE_PROVENANCE.txt"
  test -f "$SOTS/hazy/1400_1.png"
  test -f "$SOTS/clear/1400.png"

  GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -1 | cut -d, -f1 | tr -d ' ')"
  test -n "$GPU_ID"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  printf 'selected_physical_gpu=%s\n' "$GPU_ID"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader

  "$PY" - <<'PY'
import cv2
import numpy
import PIL
import skimage
import torch
import torchvision
from pytorch_msssim import ssim

print(f"python_torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_device_count={torch.cuda.device_count()}")
print(f"visible_gpu={torch.cuda.get_device_name(0)}")
print(f"numpy={numpy.__version__}")
print(f"pillow={PIL.__version__}")
print(f"opencv={cv2.__version__}")
print(f"skimage={skimage.__version__}")
print("DEHAZEFORMER_IMPORT_PREFLIGHT_OK")
PY

  mkdir -p "$FIXTURE/RESIDE-IN/test/GT" "$FIXTURE/RESIDE-IN/test/hazy"
  mkdir -p "$SAVE_ROOT/indoor" "$RESULT_ROOT"
  FIXTURE="$FIXTURE" SOTS="$SOTS" "$PY" - <<'PY'
import os
from pathlib import Path
from PIL import Image

sots = Path(os.environ["SOTS"])
fixture = Path(os.environ["FIXTURE"]) / "RESIDE-IN" / "test"
hazy_path = sots / "hazy" / "1400_1.png"
gt_path = sots / "clear" / "1400.png"
with Image.open(hazy_path) as hazy_img, Image.open(gt_path) as gt_img:
    hazy = hazy_img.convert("RGB")
    gt = gt_img.convert("RGB")
    if hazy.size != (620, 460) or gt.size != (640, 480):
        raise RuntimeError(f"unexpected source sizes: hazy={hazy.size}, gt={gt.size}")
    left = (gt.width - hazy.width) // 2
    top = (gt.height - hazy.height) // 2
    cropped_gt = gt.crop((left, top, left + hazy.width, top + hazy.height))
    hazy.save(fixture / "hazy" / "1400_1.png")
    cropped_gt.save(fixture / "GT" / "1400_1.png")
    print(f"fixture_hazy_size={hazy.size}")
    print(f"fixture_gt_size={cropped_gt.size}")
    print(f"gt_crop_box={(left, top, left + hazy.width, top + hazy.height)}")
print("DEHAZEFORMER_FIXTURE_OK")
PY

  cd "$REPO"
  CHECKPOINT="$SAVE_ROOT/indoor/$MODEL.pth" "$PY" - <<'PY'
import os
import torch
import torch.nn as nn
from models import dehazeformer_s

torch.manual_seed(3407)
model = dehazeformer_s()
parameter_count = sum(p.numel() for p in model.parameters())
trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
wrapped = nn.DataParallel(model)
checkpoint = {"state_dict": wrapped.state_dict(), "engineering_only": True, "seed": 3407}
torch.save(checkpoint, os.environ["CHECKPOINT"])

model = model.cuda().eval()
sample = torch.zeros(1, 3, 64, 64, device="cuda")
with torch.no_grad():
    output = model(sample)
if output.shape != sample.shape:
    raise RuntimeError(f"shape mismatch: input={tuple(sample.shape)} output={tuple(output.shape)}")
if not torch.isfinite(output).all():
    raise RuntimeError("non-finite model output")
print(f"parameter_count={parameter_count}")
print(f"trainable_parameter_count={trainable_parameter_count}")
print(f"forward_input_shape={tuple(sample.shape)}")
print(f"forward_output_shape={tuple(output.shape)}")
print("DEHAZEFORMER_MODEL_FORWARD_OK")
PY

  "$PY" test.py \
    --model "$MODEL" \
    --dataset RESIDE-IN \
    --exp indoor \
    --num_workers 0 \
    --data_dir "$FIXTURE" \
    --save_dir "$SAVE_ROOT" \
    --result_dir "$RESULT_ROOT"

  result_dir="$RESULT_ROOT/RESIDE-IN/$MODEL"
  test -f "$result_dir/imgs/1400_1.png"
  result_csv_count="$(find "$result_dir" -maxdepth 1 -type f -name '*.csv' | wc -l)"
  test "$result_csv_count" -eq 1
  result_csv="$(find "$result_dir" -maxdepth 1 -type f -name '*.csv' -print -quit)"
  test "$(wc -l < "$result_csv")" -eq 1
  printf 'result_image=%s\n' "$result_dir/imgs/1400_1.png"
  printf 'result_csv=%s\n' "$result_csv"
  printf 'result_csv_row=%s\n' "$(sed -n '1p' "$result_csv")"
  printf 'metrics_warning=random-weight engineering values are scientifically meaningless\n'
  printf 'DEHAZEFORMER_SMOKE_TEST_OK\n'
}

set +e
run_body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -eq 0 ]]; then
  printf 'COMPLETED_GATE_PASS run=dehazeformer_smoke_20260915 rc=0 time=%s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
  printf 'DEHAZEFORMER_SMOKE_RUN_OK\n'
else
  printf 'PREFLIGHT_FAILED_ENGINEERING run=dehazeformer_smoke_20260915 rc=%s time=%s\n' "$rc" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
  printf 'DEHAZEFORMER_SMOKE_RUN_FAILED rc=%s\n' "$rc"
  exit "$rc"
fi
