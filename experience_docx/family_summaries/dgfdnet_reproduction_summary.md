# DGFDNet Reproduction Summary

Date: 2026-09-13

Status: `COMPLETED_OFFICIAL_ALIGNED_TEST_SUCCESS_DATA_DIMENSION_REPAIR`

This is an external baseline reproduction route for
[Dilizlr/DGFDNet](https://github.com/Dilizlr/DGFDNet), separate from the
ConvIR-B Haze4K architecture and route families.

## Current State

- Source snapshot: main commit `5b389b83a7e82d6f596ddf52e713f55340720ae9`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913`.
- Cloud environment: Python `3.10.20`, PyTorch `2.5.1+cu121`, CUDA available.
- Model import and synthetic CUDA forward passed.
- Official ITS/SOTS indoor data layout passed with 500 hazy and 50 gt files.
- Official checkpoint loaded strictly with zero missing/unexpected keys. SHA-256:
  `c7287eec2f0e90d751b349b9246f869dbd45fa7d3b77cf6487f2ecf00b980a04`.
- The unmodified official evaluator fails because all SOTS indoor hazy images
  are `620x460` while gt images are `640x480`.
- Center-crop alignment was selected by a structural audit and completed all
  500 samples through the original official entrypoint: PSNR `42.19 dB`, SSIM
  `0.99661`, average time `0.030670 s`.

## Evidence Boundary

The metric is valid for the declared center-crop derived-data route. The raw
official failure, engineering-fix attempts, successful official-aligned log,
checkpoint hash, and per-image CSV are retained in
`experiment_logs/dgfdnet_repro_20260913/`. Do not compare it to an unqualified
official result without matching the same data alignment.
