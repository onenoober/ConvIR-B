# Haze4K v4 SFAD/DCFSB Family Summary

Date: 2026-07-08

Status: after-A3 restart completed; v4.6 DCFSB-bottleneck adapter4 is the current internal-positive candidate; locked test remains blocked.

## Fixed Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

These two pain points are fixed for v4 and should not be replaced by later route narratives.

## Current Read

- A0 baseline lock passed official checkpoint/resource preflight from the immutable official architecture anchor.
- A1 SDFM-only and A2 GST-only were train-side mechanism-positive, but A3 SDFM+GST showed a strong negative non-additive interaction and remains stopped.
- v4.4 bottleneck diagnosis rechecked A1/A2/A3 on trainfit128 and internal_holdout256. It found first128 was too pessimistic for A3, but A3 still underperformed A1/A2 and had strong negative interaction, so A3 expansion stayed blocked.
- v4.5 SDC-Lite from the official anchor failed internal256: mean delta PSNR `-0.009711`, positive ratio `0.437500`, R-response correlations negative, locked test untouched.
- v4.6 DCFSB-bottleneck from the official anchor passed the internal stage gate after bracket probes. The selected `adapter4` candidate reached internal256 mean delta PSNR `+0.044404`, positive ratio `0.625000`, p5 delta PSNR `-0.216141`, mean high-frequency L1 delta `-0.00000057`, and high-gate std `0.038074`. Locked test was not touched or enumerated.

## Decision

Keep v4.6 DCFSB-bottleneck `adapter4` as the current v4 candidate. This is an internal train-derived candidate-positive result, not a promotion-ready result. Any locked-test, broader validation, or code integration requires a separate written gate.

## Stop/Reopen Rules

- Do not continue A1/A2/A3 by simply adding epochs, seeds, canary expansion, selector probes, density auxiliary, or locked-test access.
- Do not introduce DCFSB from the failed A3 combined base; v4.6 succeeded only as an independent route from the official anchor.
- Do not use any route that touched default Haze4K test validation as scientific evidence.
- Do not tune from the v4.6 result on locked test. Choose any further checkpoint/epoch/variant on train-derived/internal validation first.
