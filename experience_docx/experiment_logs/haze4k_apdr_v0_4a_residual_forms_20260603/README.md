# APDR-v0.4A Residual-Form Failure-Branch Diagnostics

Date: 2026-06-03

Status: completed AutoDL diagnostics after LowFieldNet-v1 Gate B failure.

## Purpose

LowFieldNet-v1 passed no-op/cache/freeze checks but failed overfit32 for sigma
`3.0` and `7.0`. Free-parameter low succeeded, so the next question is which
residual expression can recover learnability under the same cached full-image
`M_safe`, train-calibrated correctability, and delta-supervised loss.

These diagnostics do not authorize stop20. They only decide which residual
expression is worth a later Gate C train128/mini-val check.

The overfit32 branch uses full-image cached tensors (`crop_size=4096`) so
ID-embedding and basis-mixture fields are not confounded by missing crop
coordinates. Later trainable versions must reintroduce coordinate-correct crop
sampling before Gate C.

## Branches

| Branch | Question |
| --- | --- |
| ID embedding | Can the same cache/gate/loss express the target when image identity supplies the parameters? |
| Basis mixture | Can low-dimensional learned global bases reduce image-to-field difficulty? |
| Basis + local | Can global bases plus a small local residual recover learnability? |
| Physics veil | Can a scalar veil map plus RGB haze vector learn the target under stronger structure? |

## Commands

Run sequentially on AutoDL:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_residual_forms_sigma3_sequence.sh
```

Or run one branch:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_id_embedding_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_basis_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_basis_local_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_veil_sigma3.sh
```

## Expected Artifacts

Each branch writes:

- `lowfield_overfit32_summary_*.json`
- `lowfield_overfit32_per_image_*.csv`
- `lowfield_overfit32_history_*.csv`
- `opened_closed_groups_*.csv`
- `lowfield_amplitude_audit_*.json`
- `cache_usage_audit_*.csv`
- `preflight_noop_cache_*.json`
- `lowfield_overfit32_*.log`

The tool may write `tensor_cache/` on AutoDL. Do not sync or commit that cache.

## Results

| Branch | Verdict | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ID embedding | pass | `0.7807` | `0.9712` | `0.9720` | `+1.2022 dB` | `+0.3935 dB` | `0 / 0` |
| Basis mixture | fail | `0.0717` | `0.3214` | `0.0870` | `+0.0834 dB` | `+0.0648 dB` | `0 / 0` |
| Basis + local | fail | `0.0791` | `0.3813` | `0.0979` | `+0.0920 dB` | `+0.0771 dB` | `0 / 0` |
| Physics veil | fail | `0.0437` | `0.2582` | `0.0375` | `+0.0490 dB` | `-0.0018 dB` | `0 / 0` |

## Decision

The cache/gate/loss path is valid because ID embedding passes. Pure
image-feature mapping remains the blocker: basis-style expressions improve over
LowFieldNet-v1 but remain far below Gate B. Do not run Gate C or stop20 from
these residual forms.

The best next residual-form idea is not another randomly initialized dense
predictor. If this route continues, initialize/derive global bases from the
successful ID/free-parameter targets, then train only image-to-basis weights
plus a very small local correction under the same Gate A/B protocol.
