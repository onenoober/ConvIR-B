# Haze4K DTA-v2 Calibrated Evidence

Date: 2026-06-11

Status: `IN_PROGRESS_CLOUD_QUEUE_PENDING`

This directory stores text-only evidence for `codex/haze4k-dta-v2-calibrated`,
the calibrated confidence-gated DTA route for Innovation 1. Checkpoints, model
weights, datasets, images, `.npy` depth caches, and raw inference outputs are not
committed.

## Runtime Contract

- Cloud host: `convir-4090`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.

## Planned Text Artifacts

- `setup_convir4090_dta_v2.sh`
- `run_dta_v2_depth_transmission_audit_convir4090.sh`
- `run_dta_v2_make_oof_splits_convir4090.sh`
- `run_dta_v2_preflight_convir4090.sh`
- `run_dta_v2_train_eval_convir4090.sh`
- `dta_v2_depth_transmission_audit.log`
- `dta_v2_depth_transmission_audit/dta_depth_transmission_audit_summary.json`
- `dta_v2_oof_splits.log`
- `dta_v2_haze4k_oof_splits_seed3407.json`
- `dta_v2_preflight.log`
- `dta_v2_preflight.json`
- train/eval logs and `scout_eval_compare_*.json/csv` for normal, zero, shuffle, invert, and adapter-neighbors runs.

## Execution Order

1. Sync branch to `convir-4090` and run setup/static py_compile there.
2. Run depth-transmission audit and OOF split generation.
3. Run DTA-v2 preflight with supervised transmission/physics probe.
4. Run adapter-only normal depth on train-derived split, then zero/shuffle/invert controls.
5. Run adapter-neighbors only after the adapter-only/control evidence is available.
6. Sync text evidence back to GitHub after every completed cloud stage.


## 2026-06-11 First-Stage Cloud Evidence

- convir-4090 setup completed at commit `2460b21` on `codex/haze4k-dta-v2-calibrated`; remote py_compile passed for DTA-v2 code and tools.
- Dataset inventory at setup: train haze `3001` files by raw `find`, train gt `3000`, train trans `3000`, test haze `1000`; the Haze4K loader filters image extensions and matched OOF split generation found `3000` train images.
- OOF split generation passed: `5` folds with `600` validation images each, saved as `dta_v2_haze4k_oof_splits_seed3407.json`.
- DTA-v2 preflight passed on GPU 1: partial A0 load had `602` loaded keys, `25` missing keys all under `DTA.`, unexpected keys `[]`.
- Synthetic no-op equivalence passed with max absolute diff `0.0`.
- Real-batch supervised probe passed: content `0.01675373`, rank `0.69314724`, TV `0.0`, trans `0.01417094`, phys `0.02130603`, total `0.01783682`, DTA grad sum `0.66677364`, nonzero DTA grad tensors `8`.
- Depth-transmission audit is currently running in tmux session `dta_v2_audit`; partial progress reached train `250/3000` at the first sync.
