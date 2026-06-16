# Haze4K v2.8 NH-HAZE Official-Weight Evaluation Evidence

Status: `PLANNED`

- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-8-nhhaze-official-weights.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Runtime host: `convir-4090`
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`
- WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`
- Weight protocol: NH-HAZE-specific ConvIR-B and WDMamba checkpoints
- Haze4K locked test touched: `false`
- NH-HAZE alpha tuning: `false`

## Scope

This route is the formal NH-HAZE official-weight evaluation replacing the
over-broad reading of v2.7. The endpoint row `alpha=1.0` compares
`WDMamba_NH` against `A0_NH`. The shrinkage alpha grid is diagnostic and must
not be used to select a tuned NH-HAZE alpha.

## Planned Primary Files

- `v28_nhhaze_official_weights_summary.json`
- `v28_nhhaze_official_weights_alpha_grid.csv`
- `v28_nhhaze_official_weights_compact_alpha_comparison.csv`
- `v28_nhhaze_official_weights_per_image.csv`
- `v28_nhhaze_official_weights_group_metrics.csv`
- `v28_nhhaze_official_weights_group_min.csv`
- `v28_nhhaze_dataset_preflight.json`
- `v28_final_audit.json`
- `v28_decision.md`
