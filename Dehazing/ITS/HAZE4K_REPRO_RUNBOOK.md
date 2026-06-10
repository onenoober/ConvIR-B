# Haze4K Reproduction Runbook

Date: 2026-06-10

Status: baseline bring-up notes for ConvIR dehazing on Haze4K.

## Scope

- Task: image dehazing only.
- Model entrypoint: `Dehazing/ITS/main.py`.
- Target baseline: ConvIR-B on Haze4K, official table PSNR about 34.15 and SSIM about 0.99.
- Training policy: run training on the cloud server only.

## Cloud Paths

Current server alias:

```bash
ssh dehaze1
```

Observed paths:

```bash
CODE_ROOT=/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor
ITS_ROOT=/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/ITS
DATA_ROOT=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PRETRAINED_BASE=/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/pretrained_models/haze4k-base.pkl
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
```

On the current `dehaze1` server, the existing checkpoint copy is still at the
legacy path `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
For a new server, place or download `haze4k-base.pkl` under the anchor checkout
path above, or pass its explicit path with `--test_model` / `--init_model`.

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
legacy stack is not the current cloud runtime. Use the dedicated `convir-cu128`
environment or recreate it from `py310` on the new server.

Current environment authority:

- full guide: `../../experience_docx/CLOUD_PY310_ENVIRONMENT.md`;
- evidence root: `../../experience_docx/experiment_logs/cloud_py310_environment_20260610/`;
- current verified stack: Python `3.10.13`, PyTorch `2.11.0+cu128`, torchvision
  `0.26.0+cu128`, torch CUDA `12.8`, cuDNN `91900`;
- current GPU/driver: NVIDIA GeForce RTX 4090, driver `595.58.03`, host CUDA
  `13.2`.

Environment creation on a future AutoDL server when `py310` already contains the
cu128 stack:

```bash
/root/miniconda3/bin/conda create -y -n convir-cu128 --clone py310
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
$PYTHON -m pip install tensorboard==2.20.0 einops==0.8.2 scikit-image==0.25.2 pytorch-msssim==1.0.0 opencv-python==4.6.0.66 -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYTHON -m pip install -e /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/pytorch-gradual-warmup-lr
```

Do not preserve the old editable `warmup_scheduler` path from
`/root/autodl-tmp/workspace/ConvIR-B`; reinstall it from the current GitHub
checkout or route branch.

## Pretrained Evaluation

Run from the cloud server:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/ITS
$PYTHON main.py \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --test_model "$PRETRAINED_BASE"
```

Use this as the first baseline check before changing the model.

Historical result on `autodl-dehaze3`, log
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
cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/ITS
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

## Training Smoke

For a baseline bring-up smoke, keep artifacts isolated from the official
pretrained evaluation by using a separate model name:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/ITS
$PYTHON main.py \
  --model_name ConvIR-Haze4K-smoke \
  --mode train \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 2 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 1 \
  --valid_freq 1
```

Historical smoke result on `autodl-dehaze3`, log
`results/ConvIR-Haze4K-smoke/logs/train_smoke_20260531-201109.log`:

```text
Epoch 1 valid PSNR: 20.96 dB
Epoch 2 valid PSNR: 20.31 dB
Loss content and FFT loss stayed finite.
```

Generated checkpoint artifacts:

```text
results/ConvIR-Haze4K-smoke/Training-Results/Best.pkl
results/ConvIR-Haze4K-smoke/Training-Results/Final.pkl
results/ConvIR-Haze4K-smoke/Training-Results/model.pkl
results/ConvIR-Haze4K-smoke/Training-Results/model_1.pkl
results/ConvIR-Haze4K-smoke/Training-Results/model_2.pkl
```

`Best.pkl` was then evaluated with:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/ITS
$PYTHON main.py \
  --model_name ConvIR-Haze4K-smoke \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --test_model results/ConvIR-Haze4K-smoke/Training-Results/Best.pkl
```

Smoke `Best.pkl` test result, log
`results/ConvIR-Haze4K-smoke/logs/test_best_smoke_20260531-201541.log`:

```text
The average PSNR is 20.96 dB
The average SSIM is 0.90390 dB
Average time: 0.083701
```

This is diagnostic only. It proves train, valid, checkpoint save, checkpoint
load, and test are wired correctly; it is not a from-scratch reproduction claim.

## First Gate

Do not begin model modifications until:

- Haze4K dataloader can read both train and test splits.
- The ConvIR-B pretrained checkpoint loads.
- Full Haze4K test evaluation completes.
- The reproduced metric is recorded with command, checkpoint, environment, and
  dataset path.
