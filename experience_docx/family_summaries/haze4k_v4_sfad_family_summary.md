# Haze4K v4 SFAD Family Summary

Date: 2026-07-07

Status: current v4 SFAD route stopped after A3 train-side failure; locked test remains blocked.

## Fixed Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

These two pain points are fixed for v4 and should not be replaced by later route narratives.

## Current Read

- A0 baseline lock passed official checkpoint/resource preflight from the immutable official architecture anchor. It established the Haze4K train data count, strict official checkpoint load, parameter count `8,630,665`, one train-crop loss, and locked-test block.
- A1 SDFM-only passed neutral partial-load/no-op preflight with `200,964` added parameters and produced a train-only first128 mechanism signal: mean delta PSNR `+0.0184446`, median `+0.0330057`, positive ratio `0.5703125`, mean delta SSIM `-0.0000368`.
- A2 GST-only passed neutral partial-load/no-op preflight with `247,618` added parameters and produced a train-only first128 mechanism signal: mean delta PSNR `+0.0303176`, median `+0.0237083`, positive ratio `0.5546875`, mean delta SSIM `+0.0000539`.
- A3 SDFM+GST passed neutral partial-load/no-op preflight with `448,582` added parameters and exact A0 output equality, but the train-only first128 audit failed: mean delta PSNR `-0.0457436`, median `-0.0459900`, positive ratio `0.3984375`, mean delta SSIM `-0.0001366`.

A1 and A2 remain mechanism/trainability signals only. A3 shows that the naive combination is not additive and should not be used as a base for larger phases. None of these results authorize locked-test use.

## Decision

Stop the current v4 SFAD route after A3. Do not launch density auxiliary or DCFSB phases from the failed A3 combined base. Any future v4 continuation needs a new written route identity and metric contract from the official anchor, not a simple A3 expansion.

## Stop/Reopen Rules

- Do not continue A1 or A2 by simply adding epochs, seeds, canary expansion, selector probes, or locked-test access.
- Do not introduce density auxiliary or DCFSB from the failed A3 combined base.
- Do not use any route that touched default Haze4K test validation as scientific evidence.
