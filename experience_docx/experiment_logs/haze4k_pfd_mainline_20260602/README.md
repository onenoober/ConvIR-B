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
| `gate_B1_stop20_with_diagnostic.json` | B1 gate result after attaching the completed diagnostic sidecar. |
| `diagnostic_seed3407_B1_vs_A1_best/` | Text-only fixed diagnostic sidecar evidence retained for Git sync. |
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

## Diagnostic Backfill

The A1/B1 training and evaluation ran with `save_image=False`, so the fixed
diagnostic sidecar was backfilled directly from checkpoints, per-image CSV, and
bucket JSON on `autodl-dehaze3`.

Retained text evidence:

- `diagnostic_seed3407_B1_vs_A1_best/diagnostic_summary.json`
- `diagnostic_seed3407_B1_vs_A1_best/sample_manifest.csv`
- `diagnostic_seed3407_B1_vs_A1_best/output_safety_stats.csv`
- `diagnostic_seed3407_B1_vs_A1_best/pfd_branch_stats.csv`
- `diagnostic_seed3407_B1_vs_A1_best/pfd_branch_stats_by_category.json`
- `diagnostic_seed3407_B1_vs_A1_best/visual_notes_filled.md`
- `diagnostic_seed3407_B1_vs_A1_best/direct_sidecar_status_20260602.txt`

Raw sample PNG files and `visual_panel_20.png` are artifact outputs only and are
not part of the GitHub text-evidence sync.

## Diagnostic Conclusion

The sidecar confirms that B1 failure is not metric-only. Catastrophic rows have
visible brightness/color/range failures, and output safety stats show large
candidate shifts in luma, RGB mean, and out-of-range ratios.

RHFD activity is low in absolute magnitude but not selective enough across hard
gain, easy regression, preserved, and catastrophic groups. B1 acts like a broad
feature residual adapter rather than a preservation-aware hard-case
intervention. Do not launch B2/B3 from this B1 as-is.
