# APDR-v0.4A LowFieldNet Gate A/B

Date: 2026-06-03

Status: sigma `3.0` and sigma `7.0` Gate A/B completed on AutoDL; both overfit32 runs failed.

## Purpose

This preflight checks whether a deployable LowFieldNet can learn the cached
low-frequency delta target under frozen ConvIR-B, frozen APDR-v0.2RC `M_safe`,
and frozen train-calibrated correctability.

It is deliberately separate from `Dehazing/ITS/train.py`: local work only
compiles the tool, and AutoDL runs the overfit32 diagnostic.

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_lowfield_gate_ab_20260603/run_apdr_v0_4a_lowfield_overfit32_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_lowfield_gate_ab_20260603/run_apdr_v0_4a_lowfield_overfit32_sigma7.sh
```

## Expected Artifacts

- `preflight_noop_cache_*.json`
- `lowfield_overfit32_summary_*.json`
- `lowfield_overfit32_per_image_*.csv`
- `lowfield_overfit32_history_*.csv`
- `opened_closed_groups_*.csv`
- `lowfield_amplitude_audit_*.json`
- `cache_usage_audit_*.csv`
- `lowfield_overfit32_*.log`
- `status.txt`

The script may write a local `tensor_cache/` directory on AutoDL for cache
roundtrip auditing. Do not sync or commit that cache.

Run sigma `7.0` after sigma `3.0` if sigma `3.0` fails Gate B. This separates a
general LowFieldNet mapping failure from a sharper-target sigma `3.0` failure.

## Gate

Gate A/B passes only if no-op/cache exactness and overfit32 learnability pass:

```text
initial output max_abs_diff vs J0 <= 1e-6
cached mask/target crop max diff <= 1e-8
backbone/selector trainable params = 0
weighted delta L1 drop >= 0.50
pred-target corr >= 0.50
oracle recovery >= 0.30
hard train gain >= +0.30 dB
easy train gain >= -0.010 dB
strong/severe regressions = 0
```

## Results

| Target | No-op/cache/freeze | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigma `3.0` | pass | `0.0211` | `0.0072` | `0.0210` | `+0.0191 dB` | `-0.0037 dB` | `0 / 0` |
| sigma `7.0` | pass | `0.0227` | `0.0107` | `0.0222` | `+0.0192 dB` | `-0.0042 dB` | `0 / 0` |

Decision: Gate A passed for both targets, but Gate B failed for both. Do not
run stop20 from LowFieldNet-v1. The cache/freeze/no-op protocol is sound; the
failure is deployable feature-to-lowfield mapping rather than sigma choice.
