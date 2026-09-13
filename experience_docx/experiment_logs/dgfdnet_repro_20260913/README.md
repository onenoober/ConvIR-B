# DGFDNet Reproduction: First ITS Test

Date: 2026-09-13

Status: `PREFLIGHT_DONE_CHECKPOINT_PENDING`

## Scope

- Source: `Dilizlr/DGFDNet` main at commit `5b389b83a7e82d6f596ddf52e713f55340720ae9`
- Runtime host: `convir-4090`
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913`
- Dataset argument: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/ITS/`
- Effective test set: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/`
- Expected checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/dgfdnet_its.pth`

## Preflight

- Python 3.10.20, PyTorch 2.5.1+cu121, CUDA available.
- `models.DGFDNet` imported successfully.
- Parameter count: `2,083,311`.
- Synthetic CUDA forward: `(1, 3, 64, 64) -> (1, 3, 64, 64)`, finite output.
- Official ITS/SOTS layout: 500 hazy files and 50 gt files; the 500-to-50 pairing is expected for SOTS indoor.

## Test Command

Run on `convir-4090` after the checkpoint is uploaded:

```bash
bash /sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913/test_its.sh
```

Pass a different checkpoint path as the first argument when needed. The script records `status.txt` and the test log in this directory.
