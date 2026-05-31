# Haze4K Reproduction Runbook

Date: 2026-05-31

Status: baseline bring-up notes for ConvIR dehazing on Haze4K.

## Scope

- Task: image dehazing only.
- Model entrypoint: `Dehazing/ITS/main.py`.
- Target baseline: ConvIR-B on Haze4K, official table PSNR about 34.15 and SSIM about 0.99.
- Training policy: run training on the cloud server only.

## Cloud Paths

Current server alias:

```bash
ssh autodl-dehaze3
```

Observed paths:

```bash
CODE_ROOT=/root/autodl-tmp/workspace/ConvIR-B
ITS_ROOT=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS
DATA_ROOT=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PRETRAINED_BASE=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
```

The Haze4K copy in `Dehaze-Net` uses this layout:

```text
HAZE4K/
  train/
    haze/
    gt/
  test/
    haze/
    gt/
```

This differs from the ConvIR README `IN/GT` layout. The loader now accepts both
`IN/GT` and `haze/gt`.

## Environment

The official ConvIR README lists PyTorch 1.8.1 and torchvision 0.9.1, but that
stack is not compatible with the required CUDA 12.8 / RTX 5090 runtime. Use the
dedicated `convir-cu128` environment: it keeps the ConvIR dependencies and uses
the official PyTorch cu128 wheel stack available on the server.

Environment creation on `autodl-dehaze3`:

```bash
/root/miniconda3/bin/conda create -y -n convir-cu128 --clone py310
$PYTHON -m pip install tensorboard einops scikit-image pytorch-msssim opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYTHON -m pip install -e /root/autodl-tmp/workspace/ConvIR-B/pytorch-gradual-warmup-lr
```

Verified package/runtime facts on 2026-05-31:

```bash
$PYTHON - <<'PY'
import torch
import torchvision
print(torch.__version__)       # 2.11.0+cu128
print(torchvision.__version__) # 0.26.0+cu128
print(torch.version.cuda)      # 12.8
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 5090
PY
```

## Pretrained Evaluation

Run from the cloud server:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS
$PYTHON main.py \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --test_model /root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
```

Use this as the first baseline check before changing the model.

Result on `autodl-dehaze3`, log
`results/ConvIR/logs/haze4k_base_eval_20260531-194703.log`:

```text
The average PSNR is 34.14 dB
The average SSIM is 0.98971 dB
Average time: 0.084050
```

This matches the official ConvIR-B Haze4K reference within rounding tolerance
against the reported 34.15 / 0.99.

## Training Command

Use after pretrained evaluation and GPU visibility are confirmed:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS
$PYTHON main.py \
  --mode train \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 1000 \
  --print_freq 100 \
  --num_worker 8 \
  --save_freq 20 \
  --valid_freq 20
```

Keep `results/ConvIR/Training-Results/Best.pkl` as the best validation PSNR
checkpoint and `Final.pkl` as the final checkpoint.

## First Gate

Do not begin model modifications until:

- Haze4K dataloader can read both train and test splits.
- The ConvIR-B pretrained checkpoint loads.
- Full Haze4K test evaluation completes.
- The reproduced metric is recorded with command, checkpoint, environment, and
  dataset path.
