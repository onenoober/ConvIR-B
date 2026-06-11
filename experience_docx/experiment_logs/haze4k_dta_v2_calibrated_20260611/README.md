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


## 2026-06-11 Adapter-Only Fold0 Scout5 Controls

Four adapter-only scout5 jobs ran concurrently on convir-4090 GPUs 1-4 using
OOF `fold0_train` for training and the first `128` images from `fold0_val` for
comparison. All jobs completed train, A0 comparison, and post-run `t_pred`
quality audit.

| Depth mode | Role | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Spearman(depth,-log(t)) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | calibrated true-depth from audit | `+0.058864` | `+0.035617` | `+0.083542` | `-0.0000009` | `10` | `20` | `0.076015` | `0.922318` | `+0.898767` |
| `normal` | wrong raw orientation control | `+0.064201` | `+0.041953` | `+0.083473` | `-0.0000019` | `10` | `20` | `0.090006` | `0.918616` | `-0.898766` |
| `shuffle` | mismatched-depth control | `+0.035514` | `-0.013165` | `+0.099094` | `+0.0000336` | `10` | `18` | `0.084645` | `0.919693` | `-0.415015` |
| `zero` | no-depth control | `+0.024335` | `-0.028063` | `+0.080075` | `+0.0000468` | `12` | `17` | `0.079062` | `0.921572` | n/a |

Interpretation: the route is trainable and positive on this small internal
diagnostic, but mechanism attribution is not clean yet because the wrong raw
orientation is slightly higher than calibrated `invert` on 128 images. Zero and
shuffle controls improve mostly easy samples while hurting hard bottom-25,
which keeps depth-mechanism evidence open for the 20-epoch/full-fold run.
