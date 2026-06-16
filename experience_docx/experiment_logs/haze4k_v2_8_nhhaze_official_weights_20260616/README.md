# Haze4K v2.8 NH-HAZE Official-Weight Evaluation Evidence

Status: `COMPLETED_GATE_FAIL`

- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-8-nhhaze-official-weights.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`
- Runtime host: `convir-4090`
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- Weight protocol: NH-HAZE-specific ConvIR-B and WDMamba checkpoints
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`
- WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`
- A0 data argument: `NHR`
- WDMamba DENet blocks: `4`
- Primary endpoint: `alpha=1.0` WDMamba_NH relative to A0_NH
- Diagnostic inherited alpha: `A0_NH + 0.375 * (WDMamba_NH - A0_NH)`
- Haze4K locked test touched: `false`
- NH-HAZE alpha tuning: `false`

## Result

Decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`

Final audit: `v28_final_audit.json` reports `ok=true`, `55` unique image ids,
three shard manifests with `19/18/18` rows, complete seven-alpha grid, Haze4K
locked untouched, and NH-HAZE alpha tuning disabled.

Primary rows:

- A0_NH mean PSNR: `26.104707`.
- WDMamba_NH endpoint alpha `1.0`: mean/hard/easy dPSNR
  `-2.197751/-1.093919/-2.327606`, dSSIM `-0.06118223`, positive `0.054545`,
  severe `52/55`, worst `-4.730593`.
- Inherited alpha `0.375`: mean/hard/easy dPSNR
  `-0.133584/+0.122348/-0.155758`, dSSIM `-0.00598572`, positive `0.309091`,
  severe `26/55`, worst `-0.956678`.
- Alpha `0.125` is diagnostic-positive only:
  `+0.086285/+0.124708/+0.082227`, dSSIM `+0.00025417`, positive `0.781818`,
  severe `0/55`.

Interpretation: under the available NH-specific checkpoints and this paired
test protocol, WDMamba_NH is not a stronger endpoint than ConvIR-B A0_NH, and
the Haze4K inherited `alpha=0.375` shrinkage profile is not supported on
NH-HAZE. The positive `alpha=0.125` row must not be used as a tuned NH-HAZE
claim without a separate validation or OOF protocol.

## Primary Files

- `v28_nhhaze_official_weights_summary.json`
- `v28_nhhaze_official_weights_alpha_grid.csv`
- `v28_nhhaze_official_weights_compact_alpha_comparison.csv`
- `v28_nhhaze_official_weights_per_image.csv`
- `v28_nhhaze_official_weights_group_metrics.csv`
- `v28_nhhaze_official_weights_group_min.csv`
- `v28_decision.md`
