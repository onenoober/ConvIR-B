# Haze4K FAM2 Confidence-Gated Gamma Stop20 Scout

Date: 2026-06-01

Status: completed diagnostic; global gain positive, preservation not safe.

## Read First

- Route card: `../../experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Primary Files

| File | Use |
| --- | --- |
| `scout_eval_compare_seed3407_stop20_best.json` | Best checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_last.json` | Last checkpoint vs original. |
| `scout_eval_compare_seed3407_stop20_best_vs_last.json` | Stability check. |
| `scout_eval_bucket_analysis_seed3407_stop20_best.json` | Best checkpoint bucket analysis. |
| `proxy_separability_seed3407.json` | Deployable proxy separability summary. |
| `proxy_separability_seed3407.csv` | Per-image proxy table. |
| `modulation_bucket_analysis_seed3407_stop20_best.json` | Modulation behavior by bucket. |
| `fam2_modres_gamma_conf_gated_train_stop20_seed3407.log` | Training log. |
| `run_conf_gate_stop20.sh` | Reproducibility command. |

## Key Result

Best mean PSNR improved by `+0.4523 dB`, with hard bottom 25% at `+0.9380 dB`.
The route still failed because strong-reference regressions remained `121/250`
and the deployable gate did not become selective enough.
