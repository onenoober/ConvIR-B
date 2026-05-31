# Haze4K ConvIR-B Pretrained Baseline

Date: 2026-05-31

Status: reproduced pretrained evaluation baseline on `autodl-dehaze3`.

## Scope

- Task: image dehazing, Haze4K.
- Model: ConvIR-B.
- Checkpoint: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Data root: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Test split: 1000 hazy images, 1000 GT images.

## Command

```bash
cd /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS
/root/miniconda3/envs/convir-cu128/bin/python main.py \
  --model_name ConvIR-Haze4K-pretrained-baseline-20260531 \
  --mode test \
  --version base \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --test_model /root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
```

## Result

| Metric | Local result | Official ConvIR-B Haze4K reference |
| --- | --- | --- |
| PSNR | 34.14 dB | 34.15 dB |
| SSIM | 0.98971 | 0.99 |
| Average per-image time | 0.083973 s | not listed |
| Peak GPU memory | 1329 MiB | not listed |
| Wall time | 118 s | not listed |

The reproduced pretrained baseline matches the official table within rounding
tolerance.

## Environment

- Python: `/root/miniconda3/envs/convir-cu128/bin/python`
- PyTorch: `2.11.0+cu128`
- Torchvision: `0.26.0+cu128`
- CUDA runtime: `12.8`
- GPU: `NVIDIA GeForce RTX 5090`

## Local Evidence Files

- `haze4k_base_pretrained_eval_20260531-205801.log`
- `haze4k_base_pretrained_eval_20260531-205801_meta.txt`
- `haze4k_base_pretrained_eval_20260531-205801_nvidia_smi.csv`

## Notes

- The first wrapper attempt failed before evaluation because `/usr/bin/time`
  is not installed in the container. The successful run used shell timestamp
  timing and completed with model return code `0`.
- The train split currently reports 3001 hazy files and 3000 GT files. This
  does not affect this test-only pretrained baseline, but must be audited before
  any training run.
