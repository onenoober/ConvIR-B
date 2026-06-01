# Haze4K FAM2-Only Stop20 Scout

Date: 2026-05-31

Status: completed diagnostic; preservation failed.

## Read First

- Route card: `../../experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Primary Files

| File | Use |
| --- | --- |
| `scout_eval_compare_seed3407_stop20.json` | Main original-vs-FAM2 comparison. |
| `scout_eval_bucket_analysis_seed3407_stop20.json` | Hard/medium/easy bucket analysis. |
| `scout_eval_per_image_seed3407_stop20.csv` | Per-image PSNR/SSIM deltas. |
| `fam2_modres_train_stop20_seed3407.log` | FAM2 training log. |
| `original_train_stop20_seed3407.log` | Matched original training log. |
| `run_matched_stop20.sh` | Reproducibility command. |
| `status.txt` | Cloud status transcript. |

## Key Result

Mean PSNR improved by `+0.1739 dB`, and the hard bottom 25% improved by
`+0.8159 dB`. The route still failed preservation with easy top 25% at
`-0.2860 dB` and strong-reference regressions at `138/250`.
