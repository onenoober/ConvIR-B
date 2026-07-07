# Haze4K v4 Source Anchors After A3

Date: 2026-07-08

Authoritative evidence source: GitHub `main` at `a3202c6e1516d21c3b2bf05cb29500ac8d6c8176` or later.

Runtime/raw-output source: `convir-4090`.

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

Fixed v4 pain points:

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

Current route state:

- A1 SDFM-only and A2 GST-only are train-side mechanism signals only.
- A3 SDFM+GST failed the train-only first128 gate.
- Locked Haze4K test remains blocked.
- Do not continue from A3 by adding density auxiliary loss, DCFSB, longer training, seed sweep, canary expansion, or locked-test access.

The v4.4 route is audit-only. It diagnoses A3's negative interaction using existing A0/A1/A2/A3 checkpoints.
