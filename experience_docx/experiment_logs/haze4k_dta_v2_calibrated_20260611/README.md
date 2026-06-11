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


## 2026-06-11 Adapter-Only Fold0 OOF20 Controls

Four adapter-only OOF20 jobs ran concurrently on fold0 train/val. Evaluation used
all `600` images from `fold0_val`, followed by full-fold `t_pred` quality audits.

| Depth mode | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate mean | Stage3 gate mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `+0.106894` | `+0.099160` | `+0.091081` | `-0.0000075` | `56` | `102` | `0.077984` | `0.921685` | `0.013002` | `0.050007` |
| `normal` | `+0.106010` | `+0.104724` | `+0.087849` | `-0.0000045` | `56` | `98` | `0.088812` | `0.920188` | `0.005675` | `0.052407` |
| `shuffle` | `+0.098391` | `+0.095590` | `+0.084815` | `+0.0000089` | `55` | `90` | `0.084428` | `0.921052` | `0.011980` | `0.053491` |
| `zero` | `+0.095529` | `+0.091814` | `+0.085666` | `+0.0000107` | `52` | `88` | `0.079434` | `0.922128` | `0.013502` | `0.055354` |

Interpretation: adapter-only DTA-v2 is positive on fold0 OOF20 for all four
modes, with calibrated `invert` barely best on mean dPSNR and `normal` best on
hard bottom-25. The small spread versus zero/shuffle means image-quality gains
cannot yet be attributed solely to correct depth; however, the full-fold result
is strong enough to continue the predeclared adapter-neighbors experiment.
