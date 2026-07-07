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

`COMPLETED_TRAIN_SIDE_FAIL`: A3 SDFM+GST completed preflight, five-epoch no-test adapter training, and train-only 128-image audit, but failed the train-side mechanism screen.

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

Training screen:

- seed: `3407`
- epochs: `5`
- scope: `adapter_only`
- validation: disabled with `--valid_freq 999`; no locked-test PSNR was produced
- final checkpoint: cloud-only `Final.pkl`

Audit note: first audit attempt failed due an engineering bug in module-stat collection on unpadded original-size images. The script was fixed to read stats from the padded candidate forward, then the same train-only first128 audit was rerun successfully.

Train-only audit summary:

- sample policy: sorted first 128 files from Haze4K `train/haze`; train-fit/mechanism sanity only
- mean delta PSNR: `-0.0457435846`
- median delta PSNR: `-0.0459899902`
- p5/p95 delta PSNR: `-0.3481691360` / `0.2201377869`
- positive ratio: `0.3984375000`
- mean delta SSIM: `-0.0001366269`
- worst/best delta PSNR: `-0.6070098877` / `0.4700622559`

Decision: stop the current A3 combined route. Do not launch density auxiliary or DCFSB phases from this failed combined base. Locked test remains blocked.
