# Haze4K v2.0 StrongExpert-GainMix Evidence

Date: 2026-06-14

Status: `C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES`

Route card: `experience_docx/experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Locked test: blocked and untouched for C0/C1/C1b.

## C0 Completion

C0a completed on `convir-4090` at `2026-06-15T00:08:18+08:00` with marker:

```text
V20_PHASE0_AUDITS_OK
```

Candidate-zoo decision:

```text
C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED
```

FullUDP global endpoint remains unsafe on the available 600-image internal
validation evidence:

- all mean dPSNR `+0.062005`;
- hard bottom-25 `+0.685523`;
- easy top-25 `-0.686496`;
- mean dSSIM `-0.00031039`;
- severe regressions `252/600`.

A0-preserving endpoint oracle opens strong-model capacity:

- mean dPSNR `+0.741695`;
- hard bottom-25 `+1.110910`;
- easy top-25 `+0.397112`;
- mean dSSIM `+0.00022958`;
- positive/intervention ratio `0.53`;
- nonnegative ratio `1.0`;
- worst `0/600`.

Because strict positive coverage is only `0.53`, C0a does not authorize direct
router promotion. It authorizes C1 risk/correctability mapping first.

Completed outputs:

- `v20_candidate_zoo_manifest.json`
- `v20_candidate_zoo_per_image_metrics.csv`
- `v20_candidate_zoo_single_expert_summary.csv`
- `v20_candidate_zoo_alpha_grid.csv`
- `v20_candidate_zoo_oracle_grid.csv`
- `v20_candidate_zoo_oracle_composition.csv`
- `v20_candidate_zoo_failure_bins.csv`
- `v20_candidate_zoo_decision.md`


## C1/C1b Risk And Deployability Audit

C1 completed on `convir-4090` at `2026-06-15T00:15:00+08:00` using only
existing 600-image internal-validation FullUDP/A0 metrics. It did not touch
locked test data.

The best C1 simple policy was:

```text
val_hard_and_name_param_2_le_1.39
```

Its headline metrics were mean dPSNR `+0.184478`, hard bottom-25
`+0.345854`, easy top-25 `0.0`, dSSIM `+0.0000116`, nonnegative ratio
`0.943333`, and severe regressions `32/600`. However, this row is a risk-map
signal only: it uses validation split membership and filename-derived haze
metadata, and its all-sample positive ratio is only `0.15`. It is not a
deployable router.

C1b corrected that issue by excluding split labels and filename-derived
parameters, using only deployable A0-PSNR proxy thresholds plus 5-fold held-out
threshold replay. C1b decision:

```text
C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES
```

C1b OOF replay metrics:

- mean dPSNR `+0.170433`;
- hard bottom-25 `+0.622967`;
- easy top-25 `0.0`;
- dSSIM `-0.00004448`;
- positive ratio `0.17`;
- selected precision `0.653846`;
- nonnegative ratio `0.91`;
- severe regressions `46/600`.

The strict positive-ratio gate fails, and the abstention-aware proxy gate also
fails because dSSIM is negative and the deployable proxy is too close to the
severe-tail limit. Therefore C2 router training is not authorized from the
current metric-only features. The efficient next step is to reacquire/render
FullUDP outputs on `convir-4090` and compute real FullUDP-A0 output-difference,
depth, texture, and artifact features before any C2 router claim.

Completed C1/C1b outputs:

- `v20_c1_summary.json`
- `v20_c1_decision.md`
- `v20_c1_feature_auc.csv`
- `v20_c1_simple_policy_grid.csv`
- `v20_c1_strong_expert_gain_risk_bins.csv`
- `v20_c1b_summary.json`
- `v20_c1b_decision.md`
- `v20_c1b_deployable_policy_grid.csv`
- `v20_c1b_oof_fold_metrics.csv`

## Parallel Evidence Hygiene Outputs

Completed outputs:

- `v37_d8_d9_reconciliation_audit.json`
- `v37_d8_d9_reconciliation_audit.md`
- `v37_d8_d9_reconciliation_inconsistencies.csv`
- `v37_d9_forensic_bucket_summary.csv`
- `v37_d9_forensic_top_regressions.csv`
- `v37_d9_forensic_feature_drift.csv`
- `v37_d9_forensic_summary.json`
- `v37_d9_forensic_summary.md`

These audits are evidence hygiene and failure attribution only. They are not a
DTA-v3.7 locked-test repair path.

Reconciliation result:

```text
D8_METRICS_USABLE_METADATA_RECONCILIATION_REQUIRED
```

D9 forensic result:

```text
D9_FORENSIC_COMPLETE_NO_TUNING
```
