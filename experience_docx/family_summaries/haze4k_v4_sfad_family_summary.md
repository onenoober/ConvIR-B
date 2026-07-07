# Haze4K v4 SFAD Family Summary

Date: 2026-07-07

Status: open; A3 SDFM+GST is authorized, locked test remains blocked.

## Fixed Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

These two pain points are fixed for v4 and should not be replaced by later route narratives.

## Current Read

- A0 baseline lock passed official checkpoint/resource preflight from the immutable official architecture anchor. It established the Haze4K train data count, strict official checkpoint load, parameter count `8,630,665`, one train-crop loss, and locked-test block.
- A1 SDFM-only passed neutral partial-load/no-op preflight with `200,964` added parameters and produced a train-only first128 mechanism signal: mean delta PSNR `+0.0184446`, median `+0.0330057`, positive ratio `0.5703125`, mean delta SSIM `-0.0000368`.
- A2 GST-only passed neutral partial-load/no-op preflight with `247,618` added parameters and produced a train-only first128 mechanism signal: mean delta PSNR `+0.0303176`, median `+0.0237083`, positive ratio `0.5546875`, mean delta SSIM `+0.0000539`.

A1 and A2 are mechanism/trainability signals only. They are not promotion evidence and do not authorize locked-test use.

## Decision

Continue to A3 SDFM+GST from `github/codex/haze4k-official-arch-anchor` at `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.

A3 must preserve the v4 pain points, combine only the A1 SDFM and A2 GST mechanisms, keep neutral no-op initialization, partial-load only new `SFAD_*` keys, disable default validation during training screens with `--valid_freq 999`, and keep Haze4K locked test blocked.

## Stop/Reopen Rules

- Do not continue A1 or A2 by simply adding epochs, seeds, canary expansion, selector probes, or locked-test access.
- Do not introduce density auxiliary or DCFSB until A3 evidence is written and authorizes the next phase.
- Do not use any route that touched default Haze4K test validation as scientific evidence.
