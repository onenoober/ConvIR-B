# APDR-v0.4 Free-Parameter Color Diagnostic

Date: 2026-06-03

Status: completed diagnostic; gate failed.

Route card: `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md`

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_freeparam_color_20260603/run_apdr_v0_4_freeparam_color_32.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4-cclf-diagnostics
```

## Artifacts

- `freeparam_lowcolor_apdr_v0_4_freeparam_color_32_seed3407.json`
- `freeparam_lowcolor_history_apdr_v0_4_freeparam_color_32_seed3407.csv`
- `freeparam_lowcolor_per_image_apdr_v0_4_freeparam_color_32_seed3407.csv`
- `freeparam_apdr_v0_4_freeparam_color_32_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted target L1 drop | `-1.0683` | `>= 0.80` | fail |
| oracle gain recovery | `1.0973` | `>= 0.80` | pass |
| residual/target correlation | `0.3899` | `>= 0.70` | fail |
| hard bottom-25 output gain | `+0.3169 dB` | `>= +0.35 dB` | fail |
| easy top-25 output gain | `+0.0053 dB` | non-regressive | pass |
| strong-reference regressions | `1` | `0` | fail |
| severe regressions | `2` | `0` | fail |

## Decision

Decision label: `FAIL_COLOR_BRANCH_BLOCKED`.

Do not merge color correction into the next v0.4A route. Keep color as a diagnostic/rework item only.
