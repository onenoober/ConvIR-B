# Haze4K v4.0 Baseline Lock Evidence

Date: 2026-07-07

Status: COMPLETED_PREFLIGHT_PASS

## Read First

- Route card: `../../experiment_cards/2026-07-07-haze4k-v4-0-baseline-lock.md`
- Central index: `../../EXPERIMENT_INDEX.md`
- Protocol package: `../../../docs/ai_text_packages/haze4k_v4_sfad/`

## Purpose

Initialize the v4 SFAD route and lock the A0 ConvIR-B baseline contract before any SDFM-GST or DCFSB model edits.

## Fixed Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

These two pain points are fixed for v4; color/airlight correction is intentionally out of scope for the main route.

## Current Contract

- Source branch: `codex/haze4k-official-arch-anchor`.
- Source commit: `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Runtime host: `convir-4090`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Locked test: blocked for selection.

## Planned Primary Files

| File | Use |
| --- | --- |
| `run_v4_a0_preflight.sh` | Durable cloud preflight script. |
| `v4_a0_preflight.log` | Cloud stdout/stderr. |
| `v4_a0_preflight.json` | Structured resource, load, shape, and no-locked-test result. |
| `status.txt` | Start/end markers. |

## Decision

Pending A0 preflight and internal metric-contract lock.

## A0 Preflight Result

Status: `COMPLETED_PREFLIGHT_PASS`.

| Check | Result |
| --- | --- |
| Branch | `codex/haze4k-v4-0-baseline-lock` |
| Commit | `243e3ea1b043daf3d4101c000bc4a33f98f2c1d7` |
| Python | `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python` |
| Torch | `2.5.1+cu121` |
| GPU | NVIDIA GeForce RTX 4090 |
| Haze4K train count | `3000` hazy / `3000` gt |
| Checkpoint sha256 | `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088` |
| Strict load | pass, missing `[]`, unexpected `[]` |
| Parameter count | `8,630,665` |
| Synthetic output shapes | `[[1,3,64,64], [1,3,128,128], [1,3,256,256]]` |
| One train-crop output shapes | `[[1,3,64,64], [1,3,128,128], [1,3,256,256]]` |
| One train-crop multiscale L1 | `0.009162645787000656` |
| Forbidden experimental state keys | `[]` |
| Locked test touched | `false` |
| Test split enumerated | `false` |

Decision: `A0_PREFLIGHT_PASS`. Continue to A1 SDFM-only setup only after preserving this metric/resource contract.
