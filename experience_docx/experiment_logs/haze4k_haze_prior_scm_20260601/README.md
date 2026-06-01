# Haze4K Haze-Prior SCM Hard-Aux Stop20 Scout

Date: 2026-06-01

Status: completed diagnostic; exact route not promoted.

## Read First

- Route card: `../../experiment_cards/2026-06-01-haze4k-haze-prior-scm.md`
- Central index: `../../EXPERIMENT_INDEX.md`
- AI text package: `../../../docs/ai_text_packages/2026-06-01-haze4k-haze-prior-scm/`

## Primary Files

| File | Use |
| --- | --- |
| `preflight_synthetic_seed3407.json` | Synthetic mechanical preflight. |
| `preflight_real_batch_seed3407.json` | Real-batch mechanical preflight. |
| `scout_eval_compare_seed3407_stop20_best.json` | Best checkpoint vs matched original-SCM control. |
| `scout_eval_compare_seed3407_stop20_last.json` | Last checkpoint vs matched original-SCM control. |
| `scout_eval_bucket_analysis_seed3407_stop20_best.json` | Best checkpoint bucket analysis. |
| `scout_eval_per_image_seed3407_stop20_best.csv` | Best per-image deltas. |
| `run_haze_prior_scm_hardaux_stop20.sh` | Reproducibility command. |
| `status.txt` | Cloud status transcript. |

## Key Result

Best mean PSNR was `-0.3789 dB`, while hard bottom 25% improved by
`+0.3501 dB`. Easy top 25% dropped `-1.6511 dB`, and strong-reference
regressions were `185/250`. Decision label:
`NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`.
