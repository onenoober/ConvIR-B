#!/usr/bin/env bash
set -euo pipefail

# Run this script from local WSL after `ssh convir-4090` works.
# It creates the convir-4090 route workspace under /sda/home/wangyuxin/ConvIR-B,
# clones the GitHub DTA branch, configures a Python env, and streams required
# non-dataset support files from dehaze1.

TARGET=${TARGET:-convir-4090}
SOURCE=${SOURCE:-dehaze1}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REPO_URL=${REPO_URL:-https://github.com/onenoober/ConvIR-B.git}
BRANCH=${BRANCH:-codex/haze4k-dta-lowgate}
REMOTE_ROOT=$BASE/repos/ConvIR-B-dta-lowgate
ENV_ROOT=$BASE/envs/convir-cu128
CKPT_DIR=$BASE/checkpoints/official/Haze4K
DEPTH_DIR=$BASE/depth_cache/depth_anything_v2_small_hf
DATA_DIR=$BASE/datasets/Haze4K/Haze4K
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_lowgate_20260610
STATUS=$EVID/convir4090_setup_status.txt

SRC_A0=${SRC_A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
SRC_DEPTH=${SRC_DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}

echo "setup_start convir4090 $(date --iso-8601=seconds)"
ssh -o BatchMode=yes "$TARGET" 'printf "TARGET_SSH_OK host=%s user=%s home=%s\n" "$(hostname)" "$(whoami)" "$HOME"'
ssh -o BatchMode=yes "$SOURCE" 'printf "SOURCE_SSH_OK host=%s user=%s home=%s\n" "$(hostname)" "$(whoami)" "$HOME"'

ssh "$TARGET" "bash -s" <<REMOTE
set -euo pipefail
mkdir -p "$BASE"/{repos,envs,checkpoints/official/Haze4K,depth_cache,datasets/Haze4K,logs,tmp}
if [ -d "$REMOTE_ROOT/.git" ]; then
  cd "$REMOTE_ROOT"
  if [ -n "\$(git status --short)" ]; then
    echo "REMOTE_REPO_DIRTY $REMOTE_ROOT" >&2
    git status --short >&2
    exit 3
  fi
  git fetch origin "$BRANCH"
  git switch "$BRANCH" || git switch -c "$BRANCH" "origin/$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_ROOT"
fi
mkdir -p "$EVID"
{
  echo "setup_remote_dirs_ok \$(date --iso-8601=seconds)"
  echo "base=$BASE"
  echo "remote_root=$REMOTE_ROOT"
  echo "env_root=$ENV_ROOT"
  echo "checkpoint_dir=$CKPT_DIR"
  echo "depth_dir=$DEPTH_DIR"
  echo "data_dir=$DATA_DIR"
  cd "$REMOTE_ROOT"
  echo "branch=\$(git branch --show-current)"
  echo "commit=\$(git rev-parse --short HEAD)"
} | tee -a "$STATUS"
REMOTE

echo "sync_a0_start $(date --iso-8601=seconds)"
ssh "$SOURCE" "tar -C \"$(dirname "$SRC_A0")\" -cf - \"$(basename "$SRC_A0")\"" \
  | ssh "$TARGET" "tar -C \"$CKPT_DIR\" -xf -"
ssh "$TARGET" "sha256sum '$CKPT_DIR/$(basename "$SRC_A0")' | tee -a '$STATUS'"
ssh "$TARGET" "cp -f '$CKPT_DIR/$(basename "$SRC_A0")' '$CKPT_DIR/haze4k-base.pkl'"
echo "sync_a0_done $(date --iso-8601=seconds)"

echo "sync_depth_start $(date --iso-8601=seconds)"
ssh "$TARGET" "mkdir -p '$DEPTH_DIR'"
ssh "$SOURCE" "tar -C \"$(dirname "$SRC_DEPTH")\" -cf - \"$(basename "$SRC_DEPTH")\"" \
  | ssh "$TARGET" "tar -C \"$(dirname "$DEPTH_DIR")\" -xf -"
ssh "$TARGET" "find '$DEPTH_DIR' -type f -name '*.npy' | wc -l | awk '{print \"depth_npy_count=\" \$1}' | tee -a '$STATUS'; du -sh '$DEPTH_DIR' | tee -a '$STATUS'"
echo "sync_depth_done $(date --iso-8601=seconds)"

ssh "$TARGET" "bash -s" <<REMOTE
set -euo pipefail
if [ ! -x "$ENV_ROOT/bin/python" ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create -y -p "$ENV_ROOT" python=3.10
  elif [ -x "\$HOME/miniconda3/bin/conda" ]; then
    "\$HOME/miniconda3/bin/conda" create -y -p "$ENV_ROOT" python=3.10
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$ENV_ROOT"
  else
    echo "FAILED_INFRA_NO_PYTHON_OR_CONDA" | tee -a "$STATUS"
    exit 4
  fi
fi
"$ENV_ROOT/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_ROOT/bin/python" -m pip install \
  numpy pillow opencv-python-headless scikit-image pytorch-msssim tensorboard transformers
"$ENV_ROOT/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || \
  "$ENV_ROOT/bin/python" -m pip install torch torchvision
"$ENV_ROOT/bin/python" -m pip install -e "$REMOTE_ROOT/pytorch-gradual-warmup-lr"
cd "$REMOTE_ROOT"
"$ENV_ROOT/bin/python" -m py_compile \
  Dehazing/ITS/models/ConvIR.py \
  Dehazing/ITS/main.py \
  Dehazing/ITS/train.py \
  Dehazing/ITS/valid.py \
  Dehazing/ITS/eval.py \
  Dehazing/ITS/data/data_load.py \
  Dehazing/ITS/data/data_augment.py \
  experience_docx/tools/check_haze4k_dta_preflight.py \
  experience_docx/tools/eval_haze4k_checkpoint_compare.py
"$ENV_ROOT/bin/python" - <<'PY'
import sys, importlib.util
print("python_exe=", sys.executable)
for name in ["torch", "torchvision", "numpy", "PIL", "cv2", "skimage", "pytorch_msssim", "transformers", "tensorboard", "warmup_scheduler"]:
    print(f"pkg_{name}=", bool(importlib.util.find_spec(name)))
try:
    import torch
    print("torch_version=", torch.__version__)
    print("torch_cuda=", torch.version.cuda)
    print("cuda_available=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_name=", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_probe_error=", repr(exc))
PY
echo "CONVIR4090_ENV_CONFIG_OK" | tee -a "$STATUS"
REMOTE

ssh "$TARGET" "bash -lc 'cat \"$STATUS\"; echo CONVIR4090_SETUP_OK'"
echo "setup_done convir4090 $(date --iso-8601=seconds)"
