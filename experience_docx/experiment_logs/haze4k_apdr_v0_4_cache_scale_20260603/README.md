# APDR-v0.4 Cache Scale Diagnostic

Date: 2026-06-03

Status: completed diagnostic; cache and lowpass scale gates passed.

Route card: `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md`

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_cache_scale_20260603/run_apdr_v0_4_cache_scale_train128.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4-cclf-diagnostics
```

## Artifacts

- `cache_scale_summary_apdr_v0_4_cache_scale_train128_seed3407.json`
- `cache_scale_per_image_apdr_v0_4_cache_scale_train128_seed3407.csv`
- `cache_roundtrip_audit_apdr_v0_4_cache_scale_train128_seed3407.csv`
- `cache_manifest_apdr_v0_4_cache_scale_train128_seed3407.json`
- `cache_scale_apdr_v0_4_cache_scale_train128_seed3407.log`
- `status.txt`

The tensor cache under `tensor_cache/train128/` is a raw intermediate and is intentionally not synced to GitHub.

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| cached crop mask max diff | `0.0` | `<= 1e-8` | pass |
| cached crop low sigma `3` max diff | `0.0` | `<= 1e-8` | pass |
| low sigma `3` mean gain | `+0.7804 dB` | decision-grade | pass |
| low sigma `3` hard bottom-25 gain | `+1.4176 dB` | decision-grade | pass |
| low sigma `3` easy top-25 gain | `+0.2977 dB` | non-regressive | pass |
| color mean gain | `+0.3224 dB` | diagnostic only | signal |

## Decision

Decision label: `PASS_CACHE_AND_LOW_SCALE`.

Use cached full-image `M_safe` and lowpass targets for any deployable v0.4A follow-up. The cache protocol is exact on the audited train128 subset, and sigma `3` is the strongest lowpass setting in this sweep.
