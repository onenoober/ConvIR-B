# Haze4K DTA-v2 Calibrated Evidence

Date: 2026-06-11

Status: `IN_PROGRESS_ADAPTER_NEIGHBORS_F0_SYNCED_OOF_NEXT`

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

## Text Artifacts

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
- train/eval logs, `scout_eval_compare_*.json/csv`, and
  `dta_v2_tpred_quality_*.json/csv` for normal, zero, shuffle, invert, and
  adapter-neighbors runs.

## Execution Order

1. Sync branch to `convir-4090` and run setup/static py_compile there.
2. Run depth-transmission audit and OOF split generation.
3. Run DTA-v2 preflight with supervised transmission/physics probe.
4. Run adapter-only normal depth on train-derived split, then zero/shuffle/invert controls.
5. Run adapter-neighbors only after the adapter-only/control evidence is available.
6. Sync text evidence back to GitHub after every completed cloud stage.

## Completed Cloud Evidence

- convir-4090 setup/static checks passed.
- OOF split generation passed with five train-derived folds of `600` validation
  images each.
- DTA-v2 preflight passed: partial A0 load `602` loaded keys, `25` missing keys
  all under `DTA.`, no-op max abs diff `0.0`, finite supervised
  transmission/physics losses, and DTA grad sum `0.66677364`.
- Depth-transmission audit produced `4000` rows with `0` errors and found the
  cached depth direction is reversed; primary calibrated runs therefore use
  `--dta_depth_mode invert`.
- Adapter-only fold0 scout5 and OOF20 controls completed and were synced in the
  previous evidence commit.
- Adapter-neighbors fold0 OOF20 controls completed on convir-4090 GPUs 1-4 and
  are synced here.

## Adapter-Neighbors Fold0 OOF20 Result

Evaluation used all `600` fold0 validation images. Compared with the
adapter-only result, releasing neighboring FAM/Conv layers reduced mean gain,
made easy/top samples negative, increased worst regressions, and collapsed the
recorded DTA gate means to near zero in the `t_pred` audit. This scope is not
the current promotion candidate; continue OOF expansion with `adapter_only`.

| Depth mode | Mean dPSNR | Hard bottom-25 | Easy top-25 | dSSIM | Strong regressions | Worst regressions | t_l1 | Spearman(t_pred,t_gt) | Stage2 gate mean | Stage3 gate mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `invert` | `+0.015092` | `+0.009731` | `-0.063870` | `-0.0001458` | `54` | `144` | `0.077901` | `0.921738` | `0.000074` | `0.000072` |
| `normal` | `+0.015129` | `+0.008829` | `-0.062361` | `-0.0001398` | `56` | `142` | `0.088921` | `0.920107` | `0.000074` | `0.000072` |
| `shuffle` | `+0.009656` | `+0.003892` | `-0.072763` | `-0.0001245` | `53` | `145` | `0.084493` | `0.921071` | `0.000072` | `0.000070` |
| `zero` | `+0.007218` | `+0.001740` | `-0.074593` | `-0.0001274` | `53` | `146` | `0.079408` | `0.922233` | `0.000074` | `0.000074` |

Next internal queue: keep locked Haze4K test blocked and run the predeclared
multi-fold/multi-control OOF expansion, starting with adapter-only folds `1-4`
for `invert`, `normal`, `shuffle`, and `zero`.
