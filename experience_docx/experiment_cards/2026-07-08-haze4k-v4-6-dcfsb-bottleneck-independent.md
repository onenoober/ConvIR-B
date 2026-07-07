# Haze4K v4.6 DCFSB-Bottleneck Independent Route Card

Date: 2026-07-08

Branch: `codex/haze4k-v4-6-dcfsb-bottleneck-independent`

Status: planned; launch only after v4.4 diagnosis.

Route identity: independent frequency route from the official anchor, not an A3 continuation.

Design: insert a single neutral-initialized DCFSB module after `Encoder[2]` and before `Decoder[0]`. Do not change FAM, skip paths, loss, or density maps in the first version.

Training policy: adapter-only on `haze4k_train_adapter_train.txt`, with `internal_holdout256` audit. Default validation must remain disabled.

Required evidence: `band_error_report_internal_holdout.csv`, `dcfsb_module_stats.jsonl`, and `failure_atlas_dcfsb_bottleneck.md`.
