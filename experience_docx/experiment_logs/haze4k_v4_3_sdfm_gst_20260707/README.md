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
