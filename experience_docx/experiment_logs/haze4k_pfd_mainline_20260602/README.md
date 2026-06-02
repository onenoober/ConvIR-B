# Haze4K PFD-ConvIR Mainline

Date: 2026-06-02

Status: completed gated stop20 scout; B1 failed preservation gate.

## Read First

- Route card: `../../experiment_cards/2026-06-02-haze4k-pfd-convir-mainline-plan.md`
- Central index: `../../EXPERIMENT_INDEX.md`

## Files

| File | Use |
| --- | --- |
| `preflight_pfd_v0.json` | Stage 0 pair audit, checkpoint load, and zero-init equivalence. |
| `A1_stop20_seed3407.log` | Matched ConvIR-B A1 stop20 train log. |
| `B1_rhfd_stop20_seed3407.log` | PFD B1 stop20 train log. |
| `scout_eval_compare_seed3407_B1_vs_A1_best.json` | B1 vs A1 per-image comparison summary. |
| `scout_eval_bucket_analysis_seed3407_B1_vs_A1_best.json` | Hard/easy bucket summary for B1. |
| `scout_eval_per_image_seed3407_B1_vs_A1_best.csv` | Per-image PSNR/SSIM deltas for B1. |
| `gate_B1_stop20.json` | Automatic B1 gate result. |
| `diagnostic_seed3407_B1_vs_A1_best/` | Fixed diagnostic sidecar output directory when backfilled. |
| `run_pfd_mainline_stop20.sh` | Gated stop20 run script. |
| `status.txt` | Chronological run status stream. |
| `tmux.out` | Tmux command transcript. |

## Key Result

Stage 0 passed. `A1_stop20` and `B1_stop20` both completed. B1 improved hard
samples but failed the overall preservation gate:

- global mean PSNR delta: `-0.0885 dB`
- easy top-25% mean delta: `-0.3345 dB`
- strong-reference regressions: `137/250`
- severe regressions: `434/1000`

Decision: keep as diagnostic only; stop before B2/B3.

## Required Backfill

The A1/B1 training and evaluation ran with `save_image=False`. Before this route
is used as a complete closure package, backfill the fixed diagnostic sidecar for
`seed3407_B1_vs_A1_best` and record visual notes. The updated run script starts
that pack in the background after future `compare_and_gate` evaluations, so it
does not interrupt training.
