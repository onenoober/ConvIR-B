# Haze4K v2.7 NH-HAZE Transfer Evidence

Status: `COMPLETED_GATE_FAIL`

- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-7-nhhaze-transfer.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Decision: `V27_NHHAZE_FIXED_WD0375_TRANSFER_NOT_SUPPORTED`
- Runtime host: `convir-4090`
- Source snapshot commit: `1adb61a`
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- Primary fixed profile: `WD0375 = A0 + 0.375 * (WDMamba - A0)`
- Haze4K locked test touched: `false`
- NH-HAZE tuning: `false`

## Dataset Preflight

NH-HAZE is a flat paired PNG dataset with `55` pairs named `<id>_hazy.png` and
`<id>_GT.png`. All pairs are `1600x1200`; no GT files are missing and no size
mismatches were found.

## Primary Result

The fixed Haze4K-selected `WD0375` row did not transfer to NH-HAZE:

- count: `55`
- mean/hard/easy dPSNR: `-0.018157` / `-0.003815` / `-0.042949`
- dSSIM: `+0.00887693`
- positive/nonnegative ratio: `0.472727` / `0.472727`
- severe: `13/55` (`141.82/600`)
- worst dPSNR: `-0.750659`

Full WDMamba alpha `1.0` was worse: mean/hard/easy `-0.187173` /
`-0.095121` / `-0.364553`, positive
`0.363636`, severe `26/55`
(`283.64/600`), worst `-2.029044`.

The diagnostic grid shows that larger residual scales increasingly damage
NH-HAZE. Alpha `0.125` is only near-zero (`+0.000960` mean, easy `-0.001208`,
positive `0.472727`) and is not a tuned candidate.

## Primary Files

- `v27_nhhaze_wdmamba_transfer_summary.json`
- `v27_nhhaze_wdmamba_transfer_alpha_grid.csv`
- `v27_nhhaze_wdmamba_transfer_compact_alpha_comparison.csv`
- `v27_nhhaze_wdmamba_transfer_per_image.csv`
- `v27_nhhaze_wdmamba_transfer_group_metrics.csv`
- `v27_nhhaze_wdmamba_transfer_group_min.csv`
- `v27_nhhaze_dataset_preflight.json`
- `v27_nhhaze_dataset_pairs.csv`
- `v27_final_audit.json`
- `v27_decision.md`

## Operational Notes

The first launch failed before evaluation because an rsynced worktree `.git`
pointer referenced the local WSL gitdir; this engineering failure is archived in
`failed_launch_20260616_git_pointer/`. A first successful run with an outdated
source-commit status marker is archived in
`superseded_success_20260616_commit_metadata_fix/`. The final clean rerun used
source snapshot `1adb61a` and completed all shards with rc `0`.
