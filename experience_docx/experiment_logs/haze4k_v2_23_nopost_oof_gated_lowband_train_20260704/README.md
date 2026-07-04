# Haze4K v2.23 NoPost OOF Gated Lowband Train Evidence

Status: PLANNED

This route is an OOF train-derived screen for the v2.22 gated-lowband module.
It trains only `nopost_gated_lowband_policy.*` and does not touch locked Haze4K
test data.

## Initial Policy

- Runtime only on `convir-4090`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Train/eval split: existing train-derived `oof_fold` CSV.
- Locked test touched: `false`.

Closeout decision is pending cloud OOF evidence.
