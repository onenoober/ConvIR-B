# Haze4K v4.3 SDFM+GST Evidence README

Route id: `haze4k_v4_3_sdfm_gst_20260707`

Branch: `codex/haze4k-v4-3-sdfm-gst`

Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-3-sdfm-gst`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

## Policy

- Locked Haze4K test is blocked for preflight, training, and train-side audit.
- Official training validation is disabled with `--valid_freq 999`.
- Checkpoints and raw runtime artifacts stay cloud-only.

## Stage Authorization

Stage 1 training may launch only after `V4_A3_SDFM_GST_PREFLIGHT_OK`.

Stage 2 train-side audit may run only after `V4_A3_SDFM_GST_ADAPTER5_NOTEST_TRAIN_OK` and cloud-local `Final.pkl` exist.

## Stage 0 Result

`V4_A3_SDFM_GST_PREFLIGHT_OK` at 2026-07-07T23:22:55+08:00.

Summary: official checkpoint hash matched, official strict load passed, A3 partial-load accepted only `SFAD_*` missing keys, adapter-only scope exposed only SDFM/GST modules, synthetic and train-crop no-op deltas were both `0.0`, and locked test remained untouched.

## Stage 1 Result

`V4_A3_SDFM_GST_ADAPTER5_NOTEST_TRAIN_OK` at 2026-07-07T23:29:51+08:00. Training used seed `3407`, adapter-only scope, `--valid_freq 999`, and produced cloud-only `Final.pkl`. No validation PSNR was emitted.

## Stage 2 Result

First audit attempt failed due an engineering issue in module-stat collection on unpadded original-size images. The script was corrected and the same train-only audit was rerun.

`V4_A3_TRAIN128_AUDIT_OK` at 2026-07-07T23:39:21+08:00.

Train-only audit summary:

- mean delta PSNR: `-0.0457435846`
- median delta PSNR: `-0.0459899902`
- p5/p95 delta PSNR: `-0.3481691360` / `0.2201377869`
- positive ratio: `0.3984375000`
- mean delta SSIM: `-0.0001366269`
- locked test touched: `false`

Decision: stop current A3 combined route; density aux and DCFSB are not authorized from this base.
