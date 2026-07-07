# Haze4K v4.4 Bottleneck Diagnosis Evidence README

Route id: `haze4k_v4_4_bottleneck_diagnosis_20260708`

Branch: `codex/haze4k-v4-4-bottleneck-diagnosis`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Policy: audit-only, Haze4K train-derived splits only, locked test blocked.

## Result

`V4_4_BOTTLENECK_DIAGNOSIS_OK` at 2026-07-08T00:35:20+08:00.

Internal256: A1 `0.060228`, A2 `0.066864`, A3 `0.028960`, additive expectation `0.127093`, interaction `-0.098133`.

Decision: independent v4.5 and v4.6 are authorized from the official anchor; A3 extension remains blocked.
