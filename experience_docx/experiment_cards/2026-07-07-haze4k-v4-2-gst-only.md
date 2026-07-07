# Haze4K v4.2 GST-only Route Card

Date: 2026-07-07

Branch: `codex/haze4k-v4-2-gst-only`

Route id: `haze4k_v4_2_gst_only_20260707`

Base: `codex/haze4k-official-arch-anchor` at `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`

## Fixed v4 Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

These two pain points are fixed for v4 and must not be replaced by a new story.

## Route Identity

This is a new v4 ablation route: GST-only. It tests whether guided skip transfer can make the official ConvIR-B skip path less vulnerable to polluted shallow features while preserving high-frequency detail.

Allowed in this route:

- Add GST modules only at the two decoder skip transfers.
- Partial-load the official Haze4K checkpoint with only `SFAD_GST*` missing keys.
- Run adapter-only training screens on Haze4K train data with default validation disabled.

Not allowed in this route:

- SDFM, DCFSB, density auxiliary heads, selector probes, teacher/distillation, canary expansion, or locked-test checkpoint selection.
- Any command that enumerates Haze4K `test` before a written gate authorizes locked-test use.

## Method

`SFADGSTConvIR` preserves the official encoder/decoder/FAM/SCM path and inserts two neutral-initialized `GuidedSkipTransfer` adapters:

- `SFAD_GST1`: 1/2-resolution skip transfer from `res2` into the second decoder stage.
- `SFAD_GST2`: full-resolution skip transfer from `res1` into the final decoder stage.

Each GST block computes low-pass and high-pass skip/decoder features, predicts a spatial-channel transfer gate, and applies:

`skip_out = skip + alpha * gate * (clean_skip - skip)`

with `alpha=0` at initialization. Therefore a successful partial-load must be a strict no-op against A0 before training.

## Metric Contract

Stage 0 preflight must pass all of these checks before training:

- Official checkpoint hash equals `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Official A0 strict load passes.
- A2 partial-load has no unexpected or shape-mismatched keys, and all missing keys start with `SFAD_GST`.
- Adapter-only train scope exposes only `SFAD_GST1` and `SFAD_GST2` parameters.
- Synthetic and one train crop outputs match A0 with max absolute delta `<= 1e-7`.
- Gate statistics are finite and not collapsed at initialization.
- Haze4K locked test is not enumerated or touched.

Stage 1 training screen:

- Five adapter-only epochs, seed `3407`, official A0 init, `--valid_freq 999` to prevent default Haze4K test validation.
- Output checkpoint remains cloud-only; do not sync weights to GitHub.

Stage 2 train-side audit:

- Compare A2 Final against A0 on sorted first 128 images from Haze4K `train/haze` only.
- This is train-fit and mechanism sanity evidence, not generalization evidence.
- Record mean/median/p5/p95 delta PSNR, mean delta SSIM, positive ratio, and GST module statistics.

## Current Status

`COMPLETED_TRAIN_SIDE_SIGNAL`: A2 GST-only completed preflight, five-epoch no-test adapter training, and train-only 128-image audit.

Preflight summary:

- train count: `3000/3000`
- total parameters: `8,878,283`
- added parameters: `247,618`
- partial-load missing keys: `26`, all `SFAD_GST*`
- adapter-only trainable prefixes: `SFAD_GST1`, `SFAD_GST2`
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

Train-only audit summary:

- sample policy: sorted first 128 files from Haze4K `train/haze`; train-fit/mechanism sanity only
- mean delta PSNR: `0.0303175598`
- median delta PSNR: `0.0237083435`
- p5/p95 delta PSNR: `-0.2722772598` / `0.3557992935`
- positive ratio: `0.5546875000`
- mean delta SSIM: `0.0000538551`
- worst/best delta PSNR: `-0.5011405945` / `0.7117614746`

Decision: continue to A3 SDFM+GST from the immutable official architecture anchor. A2 is a mechanism/trainability signal, not a quality gate pass or generalization claim.
