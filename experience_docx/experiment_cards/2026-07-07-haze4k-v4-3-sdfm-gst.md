# Haze4K v4.3 SDFM+GST Route Card

Date: 2026-07-07

Branch: `codex/haze4k-v4-3-sdfm-gst`

Route id: `haze4k_v4_3_sdfm_gst_20260707`

Base: `codex/haze4k-official-arch-anchor` at `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`

## Fixed v4 Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

## Route Identity

This is the combined v4 ablation route authorized by A1 SDFM-only and A2 GST-only train-side mechanism signals. It combines only those two mechanisms and starts from the immutable official architecture anchor.

Allowed in this route:

- SDFM at the 1/2 and 1/4 FAM fusion points.
- GST at the two decoder skip transfers.
- Official Haze4K checkpoint partial-load with only `SFAD_*` missing keys.
- Five-epoch adapter-only no-test training screen after preflight passes.

Not allowed in this route:

- Density auxiliary heads, DCFSB, selector probes, teacher/distillation, canary expansion, or locked-test checkpoint selection.
- Any command that enumerates Haze4K `test` before a later written gate authorizes one fixed locked-test confirmation.

## Metric Contract

Stage 0 preflight must verify official checkpoint hash, official strict load, partial-load missing keys only under `SFAD_*`, adapter-only trainable prefixes `SFAD_GST1`, `SFAD_GST2`, `SFAD_SDFM1`, and `SFAD_SDFM2`, exact no-op output equality against A0, non-collapsed SDFM/GST statistics, and locked-test untouched.

Stage 1 training uses seed `3407`, adapter-only scope, official A0 initialization, and `--valid_freq 999`.

Stage 2 train-only audit compares A3 Final against A0 on sorted first 128 images from Haze4K `train/haze`. It is train-fit/mechanism evidence only.

## Current Status

`PREFLIGHT_PASSED`: Stage 0 passed on `convir-4090` at 2026-07-07T23:22:55+08:00.

Preflight summary:

- train count: `3000/3000`
- total parameters: `9,079,247`
- added parameters: `448,582`
- partial-load missing keys: `48`, all `SFAD_*`
- adapter-only trainable prefixes: `SFAD_GST1`, `SFAD_GST2`, `SFAD_SDFM1`, `SFAD_SDFM2`
- no-op max abs synthetic vs A0: `0.0`
- no-op max abs train crop vs A0: `0.0`
- locked test touched: `false`
- test split enumerated: `false`
