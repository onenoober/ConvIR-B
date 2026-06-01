# Haze4K Hard-Aware Frequency Loss Stop20 Scout

Date: 2026-06-01

Status: completed diagnostic; exact `hard_fft_lambda=0.02` failed.

## Read First

- Route card: `../../experiment_cards/2026-06-01-haze4k-hardfreq-loss.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Primary Files

| File | Use |
| --- | --- |
| `hardfreq_loss_preflight_seed3407.json` | Mechanical preflight output. |
| `hardfreq_loss_preflight_seed3407.manual.json` | Manual preflight copy. |
| `hardfreq_loss_train_stop20_seed3407.log` | Training log. |
| `scout_eval_compare_seed3407_stop20_best.json` | Best checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_last.json` | Last checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_best_vs_last.json` | Stability check. |
| `scout_eval_bucket_analysis_seed3407_stop20_best.json` | Best checkpoint bucket analysis. |
| `scout_eval_per_image_seed3407_stop20_best.csv` | Best per-image deltas. |
| `run_hardfreq_loss_stop20.sh` | Reproducibility command. |

## Key Result

Best mean PSNR was `-0.2127 dB`, while hard bottom 25% improved by
`+0.5999 dB`. Easy top 25% dropped `-1.2363 dB`, strong-reference regressions
were `188/250`, and Best-vs-Last stability was `-0.6922 dB`. Decision label:
`FAIL_STOP_HARDFFT_LAMBDA_002`.
