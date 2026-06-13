# Haze4K DTA-v3.7 U-TQS-Mix Evidence

Date: 2026-06-13

Status: `PHASE_A_TABLE_ONLY_COMPLETE_PASS_SOFT_ORACLE_HEADROOM`

Route card: `experience_docx/experiment_cards/2026-06-13-haze4k-dta-v3-7-u-tqs-mix.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Selected host: `convir-4090`.
- Host decision: `convir-4090` reachable and mostly idle; `convir-5090` SSH failed with `Permission denied (publickey,password)`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Source OOF table: `experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/v36_formal_oof_per_image_action_table.csv`.
- Source selector errors: `experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/formal_hrcs/v36_selector_error_table.csv`.
- Locked test: untouched and blocked unless formal strict pass seals a fixed policy.

## Route Decision

DTA-v3.7 is the new DTA mainline. The route abandons hard reject as the main
strategy and moves to an A0-preserving utility-aware soft action mixture with
transmission, airlight, quality, and uncertainty signals.

## Required Phase A Artifacts

- `status.txt`
- `run_dta_v3_7_phase_a_convir4090.sh`
- `v37_phase_a.log`
- `v37_positive_loss_budget_report.csv`
- `v37_soft_action_bank_oracle_grid.csv`
- `v37_false_reject_false_accept_taxonomy.csv`
- `v37_feature_ablation_auc_report.csv`
- `v37_tA_quality_uncertainty_preflight.json`
- `v37_phase_a_summary.json`

## Phase A Gate

Pass requires at least one soft action oracle row with coverage `>=0.95` that
passes mean, hard, dSSIM, positive-ratio, true-vs-controls, worst, and max outer
worst strict gates. The Phase A soft-alpha metrics are table-only linear-delta
proxies; real blended-output verification is required in Phase C before any
promotion claim.

## Phase A Completion

Phase A completed on `convir-4090` at `2026-06-13T11:44:49+08:00` with marker:

```text
DTA_V3_7_U_TQS_MIX_PHASE_A_OK rows=27000 soft_rows=18 strict_soft=13 gate=PASS_SOFT_ORACLE_HEADROOM
```

Primary row:

| Bank | Utility | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | intervention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_L2_L3_L1_full` | `max_dpsnr` | `+0.143298` | `+0.121101` | `+0.00002551` | `0.6623` | `0.00` | `0.00` | `0.6623` |

Budget and preflight findings:

- L3 hard-reject positive loss is `32.73/600`, about `8.77x` the strict budget.
- L1 hard-reject positive loss is `29.87/600`, about `6.79x` the strict budget.
- Transmission GT/pred/uncertainty columns are present; explicit airlight GT and NR-IQA features are not yet present.
- Current deployable severe-risk AUC remains weak at about `0.608`, so Phase B must add T/A/Q/U feature separability.

Decision: `PHASE_A_PASS_SOFT_ORACLE_HEADROOM`; proceed to Phase B.


## Phase B TQS Plan

Run script: `run_dta_v3_7_phase_b_tqs_convir4090.sh`.

Outputs:

- `status_phase_b_tqs.txt`
- `v37_phase_b_tqs.log`
- `v37_tqs_policy_nested_report.csv`
- `v37_tqs_policy_aggregate.csv`
- `v37_tqs_policy_action_table.csv`
- `v37_tqs_feature_group_ablation.csv`
- `v37_tqs_summary.json`

This is a train-derived table-only policy diagnostic. Deployable feature groups
must not use `trans_gt`; diagnostic `trans_gt` rows are reported only to measure
how much separability physical supervision could add.
