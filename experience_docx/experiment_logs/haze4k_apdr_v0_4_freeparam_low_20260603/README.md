# APDR-v0.4 Free-Parameter Low Diagnostic

Date: 2026-06-03

Status: completed diagnostic; strict configured gate failed, but low-field target/application signal is strong.

Route card: `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md`

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_freeparam_low_20260603/run_apdr_v0_4_freeparam_low_32.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4-cclf-diagnostics
```

## Artifacts

- `freeparam_lowcolor_apdr_v0_4_freeparam_low_32_seed3407.json`
- `freeparam_lowcolor_history_apdr_v0_4_freeparam_low_32_seed3407.csv`
- `freeparam_lowcolor_per_image_apdr_v0_4_freeparam_low_32_seed3407.csv`
- `freeparam_apdr_v0_4_freeparam_low_32_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted target L1 drop | `0.6762` | `>= 0.80` | fail |
| oracle gain recovery | `1.0938` | `>= 0.80` | pass |
| residual/target correlation | `0.9322` | `>= 0.70` | pass |
| hard bottom-25 output gain | `+1.2484 dB` | positive hard gain | pass |
| easy top-25 output gain | `+0.4411 dB` | non-regressive | pass |
| strong-reference regressions | `0` | `0` | pass |
| severe regressions | `0` | `0` | pass |

## Decision

Decision label: `LOW_FIELD_SIGNAL_STRONG_STRICT_LOSS_GATE_FAIL`.

Do not mark the combined v0.4 CCLF route as passed, because the configured loss-drop gate failed. However, recovery, correlation, hard gain, and safety are strong enough to justify a separate v0.4A low-field-only implementation card with a reviewed sanity gate.
