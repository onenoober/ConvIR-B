# DGFDNet Reproduction Summary

Date: 2026-09-13

Status: `PREFLIGHT_DONE_CHECKPOINT_PENDING`

This is an external baseline reproduction route for
[Dilizlr/DGFDNet](https://github.com/Dilizlr/DGFDNet), separate from the
ConvIR-B Haze4K architecture and route families.

## Current State

- Source snapshot: main commit `5b389b83a7e82d6f596ddf52e713f55340720ae9`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913`.
- Cloud environment: Python `3.10.20`, PyTorch `2.5.1+cu121`, CUDA available.
- Model import and synthetic CUDA forward passed.
- Official ITS/SOTS indoor data layout passed with 500 hazy and 50 gt files.
- Official checkpoint is pending user upload; no quality metric has been claimed.

## Next Gate

Run the fixed ITS test script after the checkpoint path is available. Record
checkpoint hash, checkpoint-load result, mean PSNR, mean SSIM, average inference
time, raw log, and the final status marker in
`experiment_logs/dgfdnet_repro_20260913/`.
