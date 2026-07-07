# Haze4K v4.5 SDC-Lite Route Card

Date: 2026-07-08

Branch: `codex/haze4k-v4-5-sdc-lite`

Status: planned; launch only after v4.4 diagnosis.

Route identity: new architecture route from the official anchor, not an A3 continuation.

Design: shared `R_1_2`, SDFM at 1/2 only, no `SDFM_1_4`, no `GST_1_2`; optional full-resolution GST only if written stage gate authorizes it.

Training policy: adapter-only on `haze4k_train_adapter_train.txt`, with `internal_holdout256` audit. Default validation must remain disabled.

Initial gate: internal-holdout mean delta PSNR >= `+0.03`, positive ratio >= `0.53`, `R_1_2_std >= 0.10`, p5 delta PSNR >= `-0.25`, and positive correlation between `R_1_2` and haze/A0-error proxy.
