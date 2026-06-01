# Haze4K FAM2 Bounded Gamma Stop20 Scout

Date: 2026-06-01

Status: completed diagnostic; bounded gamma failed preservation.

## Read First

- Route card: `../../experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Primary Files

| File | Use |
| --- | --- |
| `scout_eval_compare_seed3407_stop20_best.json` | Best checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_last.json` | Last checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_best_vs_last.json` | Stability check. |
| `scout_eval_bucket_analysis_seed3407_stop20_best.json` | Best checkpoint bucket analysis. |
| `modulation_bucket_analysis_seed3407_stop20_best.json` | Modulation behavior by bucket. |
| `scout_eval_per_image_seed3407_stop20_best.csv` | Best per-image deltas. |
| `fam2_modres_gamma_bounded_train_stop20_seed3407.log` | Training log. |
| `run_gamma_only_stop20.sh` | Reproducibility command. |

## Key Result

Best mean PSNR was `-0.0271 dB` versus original. The hard bucket improved by
`+0.8054 dB`, but easy top 25% dropped `-1.2740 dB` and strong-reference
regressions reached `181/250`.
