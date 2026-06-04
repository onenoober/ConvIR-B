# Haze4K DPGA v1.3 HSDF Diagnostics

Date: 2026-06-04

Status: v1.3A and v1.3B intenal diagnostics completed; v1.3B failed the regular+hard gate. Training/testing execute only on AutoDL `autodl-dehaze4`.

## Purpose

This directory holds the runnable text commands and expected intermediate results for `ConvIR-Dehaze-v1.3-HSDF`.

v1.3A tests the loss-mask and hard-aware sampler first:

- revised hard-selective tail mask;
- hard-region reconstruction weighting;
- train-derived `val_regular` and `val_hard` splits;
- no locked Haze4K test.

v1.3B was authorized only because v1.3A missed the hard gate while preserving safety. It adds a deployable hard gate on the bottleneck DPGA adapter and still uses only train-derived `val_regular`/`val_hard` gates. v1.3B did not pass and does not authorize locked Haze4K test.

## Expected Outputs

| File | Producer |
| --- | --- |
| `intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json` | `run_make_v13_hard_splits.sh` |
| `intermediates/dpga_v13_tail_mask_audit_by_bucket.csv` | `run_v13_intermediate_audits.sh` |
| `intermediates/dpga_v13_val_split_audit.csv` | `run_v13_intermediate_audits.sh` |
| `intermediates/dpga_v13_hard_proxy_auc.csv` | `run_v13_intermediate_audits.sh` |
| `train_ConvIR-Haze4K-DPGA-v1.3A-HSDF-lossmask-hardaware-shallow-scale0p25-hardw1p2-t0p01-seed3407-20260604.log` | `run_dpga_v13a_train.sh` |
| `v13a_eval_regular_hard/dpga_v13_gate_eval_regular_and_hard.json` | `run_eval_dpga_v13a_regular_hard.sh` |
| `runtime_diagnostics_val_regular/dpga_v13_runtime_ablation_on_val_inner.csv` | `run_dpga_v13_runtime_ablation_val_regular.sh` |
| `train_ConvIR-Haze4K-DPGA-v1.3B-HSDF-hardgated-bottleneck-shallow-scale0p25-bneckmax0p05-gatelambda0p01-seed3407-20260604.log` | `run_dpga_v13b_train.sh` |
| `v13b_eval_regular_hard/dpga_v13b_gate_eval_regular_and_hard.json` | `run_eval_dpga_v13b_regular_hard.sh` |
| `runtime_diagnostics_v13b_val_regular/dpga_v13b_runtime_ablation_on_val_inner.csv` | `run_dpga_v13b_runtime_ablation_val_regular.sh` |
| `train_ConvIR-Haze4K-DPGA-v1.3B-HSDF-hardgated-bottleneck-shallow-scale0p25-bneckmax0p05-gatelambda0p01-seed3407-20260604.gate_restore_bug_stopped_20260604T1645.log` | archived early v1.3B run stopped before gate supervision fix |
| `runtime_diagnostics*_val_regular.scale1_bug_20260604T1725/` | archived runtime-ablation outputs generated with incorrect module scale `1.0` |

## Decision Boundary

Passing v1.3A does not unlock Haze4K test. It decides whether loss-mask correction is enough, or whether v1.3B should add the hard-gated bottleneck expert.

v1.3A did not pass: Best `val_regular` mean was `+0.026333 dB` and Best `val_hard` hard bottom-25 was `+0.022099 dB`. Safety remained acceptable, so v1.3B was run as the next intenal diagnostic.

v1.3B also did not pass: Best `val_regular` mean was `+0.025839 dB`, Best `val_hard` hard bottom-25 was `+0.023642 dB`, positive ratio was `0.586667`, and strong regression ratio was `0.200000`. Corrected route-scale runtime ablation shows bottleneck-only mean delta was only about `+0.000824 dB`, so the hard-gated bottleneck did not add useful capacity.

Locked Haze4K test remains blocked. Stop this exact HSDF bottleneck route unless a separately justified mechanism is written first.
