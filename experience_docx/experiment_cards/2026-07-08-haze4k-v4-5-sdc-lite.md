# Haze4K v4.5 SDC-Lite Route Card

Date: 2026-07-08

Branch: `codex/haze4k-v4-5-sdc-lite`

Status: stage gate failed; not promoted.

Route identity: new architecture route from the official anchor, not an A3 continuation.

Design: shared `R_1_2`, SDFM at 1/2 only, no `SDFM_1_4`, no `GST_1_2`; optional full-resolution GST only if written stage gate authorizes it.

Training policy: adapter-only on `haze4k_train_adapter_train.txt`, with `internal_holdout256` audit. Default validation must remain disabled.

Initial gate: internal-holdout mean delta PSNR >= `+0.03`, positive ratio >= `0.53`, `R_1_2_std >= 0.10`, p5 delta PSNR >= `-0.25`, and positive correlation between `R_1_2` and haze/A0-error proxy.

## Preflight Gate

Status: passed; adapter-5 training authorized.

- Branch/commit: `codex/haze4k-v4-5-sdc-lite` / `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Checkpoint SHA256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- Split contract: adapter_train `2744`, internal_holdout256 `256`, overlap `0`
- Init equivalence to A0: synthetic max abs `0.0`, train-crop max abs `0.0`
- Added params: `42562`
- Trainable prefixes: `['SFAD_SDC']`
- Missing partial-load keys are restricted to new SDC module: `11`
- Locked test touched/enumerated: `False` / `False`

Next authorized phase: run `run_v4_5_sdc_lite_adapter5_notest.sh` with `--valid_freq 999`, then audit trainfit128 and internal_holdout256.

## Audit Closeout

Status: stage gate failed; route not promoted.

- trainfit128 mean dPSNR `-0.014244`, positive ratio `0.484375`
- internal256 mean dPSNR `-0.009711`, positive ratio `0.437500`
- internal256 p5 dPSNR `-0.180751`, R std mean `0.082352`
- R correlations are negative: input L1 `-0.467570`, A0 L1 `-0.409864`
- Locked test touched/enumerated: `False` / `False`
- Raw per-image/module tables and checkpoints remain cloud-only.

Decision: do not extend v4.5 immediately; proceed to v4.6 DCFSB-bottleneck independent route from the official anchor.
