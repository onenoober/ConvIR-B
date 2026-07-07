# Haze4K v4.2 GST-only Evidence README

Route id: `haze4k_v4_2_gst_only_20260707`

Branch: `codex/haze4k-v4-2-gst-only`

Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-2-gst-only`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

## Policy

- Locked Haze4K test is blocked for A2 preflight, training, and train-side audit.
- Official training validation is disabled for the adapter screen with `--valid_freq 999` because the repository default validation reads Haze4K `test`.
- Checkpoints, images, arrays, and raw inference outputs are cloud-only and must not be committed.

## Planned Files

- `run_v4_a2_gst_preflight.sh`: Stage 0 preflight.
- `v4_a2_gst_preflight.json`: preflight result summary.
- `v4_a2_gst_preflight.log`: full preflight stdout/stderr.
- `run_v4_a2_gst_adapter5_notest.sh`: five-epoch adapter-only no-test training screen.
- `train_ConvIR-Haze4K-v4A2-GST-adapter-notest-seed3407-20260707.log`: train stdout/stderr.
- `audit_v4_a2_train128_final_vs_a0.sh`: train-only 128-image audit.
- `a2_train128_compare_final_vs_a0.json`: compact train-side comparison.
- `a2_train128_per_image_final_vs_a0.csv`: per-image train-side comparison.
- `a2_train128_module_stats_final.jsonl`: averaged GST statistics.
- `status.txt`: durable status markers.

## Stage Authorization

Stage 1 training may launch only after `V4_A2_GST_PREFLIGHT_OK` is present in `status.txt`.

Stage 2 train-side audit may run only after `V4_A2_GST_ADAPTER5_NOTEST_TRAIN_OK` and a cloud-local `Final.pkl` exist.

Locked-test evaluation remains blocked unless a later route card update explicitly authorizes one fixed checkpoint and one locked-test command.

## Stage 0 Result

`V4_A2_GST_PREFLIGHT_OK` at 2026-07-07T23:02:55+08:00.

Summary: official checkpoint hash matched, official strict load passed, A2 partial-load accepted only `SFAD_GST*` missing keys, adapter-only scope exposed only GST modules, synthetic and train-crop no-op deltas were both `0.0`, and locked test remained untouched.
