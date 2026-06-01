# Haze4K Stop20 Original Noise Floor

Date: 2026-06-01

Status: completed baseline audit.

## Read First

- Route card: `../../experiment_cards/2026-06-01-haze4k-stop20-noise-floor.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Primary Files

| File | Use |
| --- | --- |
| `original_seed_noise_stop20.json` | Main multi-seed noise summary. |
| `original_seed_noise_per_image.csv` | Per-image seed variance table. |
| `original_train_stop20_seed2027.log` | Seed 2027 training log. |
| `original_train_stop20_seed8675.log` | Seed 8675 training log. |
| `original_train_stop20_seed2027.interrupted-20260601-131628.log` | Interrupted run transcript retained for traceability. |
| `run_original_multiseed_stop20.sh` | Reproducibility command. |
| `tmux.out` | Cloud terminal transcript. |

## Key Result

Stop20 seed mean PSNR sample std is `0.2206 dB`, and hard bucket std is
`0.4551 dB`. Single-seed stop20 improvements need this noise floor before any
promotion claim.
