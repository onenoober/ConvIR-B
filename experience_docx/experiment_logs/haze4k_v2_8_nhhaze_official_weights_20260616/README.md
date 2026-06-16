# Haze4K v2.8 NH-HAZE Official-Weight Evaluation Evidence

Status: `COMPLETED_AUDIT_RELABELED_MIXED_SPLIT_INVALID_FOR_OFFICIAL_BENCHMARK`

- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-8-nhhaze-official-weights.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Original decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`
- Audit decision: `V28_ALL55_MIXED_SPLIT_REPRO_INVALID_FOR_OFFICIAL_BENCHMARK`
- Split audit: `experience_docx/experiment_logs/haze4k_v2_8b_nhhaze_official_test_split_audit_20260616/`
- Runtime host: `convir-4090`
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- Weight protocol: NH-HAZE-specific ConvIR-B and WDMamba checkpoints
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`
- WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`
- A0 data argument: `NHR`
- WDMamba DENet blocks: `4`
- Evaluated scope: all `55` flat NH-HAZE pairs, later audited as a mixed
  official train/validation/test aggregate
- Primary endpoint in this mixed aggregate: `alpha=1.0` WDMamba_NH relative to A0_NH
- Diagnostic inherited alpha in this mixed aggregate: `A0_NH + 0.375 * (WDMamba_NH - A0_NH)`
- Haze4K locked test touched: `false`
- NH-HAZE alpha tuning: `false`

## Result

Audit decision: `V28_ALL55_MIXED_SPLIT_REPRO_INVALID_FOR_OFFICIAL_BENCHMARK`

Final engineering audit: `v28_final_audit.json` reports `ok=true`, `55` unique
image ids, three shard manifests with `19/18/18` rows, complete seven-alpha
grid, Haze4K locked untouched, and NH-HAZE alpha tuning disabled.

Scientific split audit: v2.8 evaluated the flat local directory as all `55`
paired images. That mixes official-style `01-45` train, `46-50` validation, and
`51-55` test images, so the all-55 aggregate is invalid for official NH-HAZE
benchmark reproduction.

Mixed all-55 rows, retained only as diagnostic/audit evidence:

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

Correct official-test aggregation is recorded in v2.8b. On `51-55`, A0_NH is
`20.6636/0.7968` and WDMamba_NH is `20.8307/0.8182`, aligning with the ConvIR-B
README NH-HAZE base result (`20.66/0.802`) and the WDMamba checkpoint name
`NH_20.83.pth`.

Interpretation: this v2.8 all-55 aggregate must not be used to claim that
WDMamba_NH is weaker than A0_NH on the official NH-HAZE test set, nor to claim
that inherited `alpha=0.375` fails on the official NH-HAZE test set. Use v2.8b
for official-test discussion. The all-55 result is now only a mixed-split
cautionary diagnostic.

## Primary Files

- `v28_nhhaze_official_weights_summary.json`
- `v28_nhhaze_official_weights_alpha_grid.csv`
- `v28_nhhaze_official_weights_compact_alpha_comparison.csv`
- `v28_nhhaze_official_weights_per_image.csv`
- `v28_nhhaze_official_weights_group_metrics.csv`
- `v28_nhhaze_official_weights_group_min.csv`
- `v28_decision.md`
