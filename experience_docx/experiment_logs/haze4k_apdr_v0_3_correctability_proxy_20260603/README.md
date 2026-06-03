# APDR-v0.3 Correctability Proxy

Date: 2026-06-03

Status: completed diagnostic, strict gate failed but signal is strong.

## Scope

This tabular diagnostic tests whether deployable full-image statistics from
hazy input, APDR anchor, priors, and `M_safe` can predict low-frequency oracle
correctability.

Target:

```text
low_oracle_gain = PSNR(J0 + M_safe * lowpass(Delta_star), GT) - PSNR(J0, GT)
```

Labels:

- positive: `low_oracle_gain >= +0.10 dB`;
- negative: `low_oracle_gain <= +0.01 dB`;
- middle: ignored for AUC but retained for rank checks.

The proxy is a small tabular MLP trained on a fixed 70/30 split of Haze4K test
rows. It is not a final model component and no checkpoint is saved.

## Command

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_correctability_proxy_20260603/run_apdr_v0_3_correctability_proxy_test.sh
```

## Artifacts

- `correctability_proxy_apdr_v0_3_correctability_proxy_test_seed3407.json`
- `correctability_proxy_per_image_apdr_v0_3_correctability_proxy_test_seed3407.csv`
- `audit_apdr_v0_3_correctability_proxy_test_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| held-out AUC | `1.0000` | `>= 0.80` | pass |
| held-out Spearman(score, low gain) | `0.9691` | `>= 0.45` | pass |
| easy top-25% mean score | `0.1137` | `<= 0.10` | fail |
| oracle-positive hard mean score | `0.9856` | `>= 0.50` | pass |

Held-out context:

| Metric | Value |
| --- | ---: |
| valid positives | `134` |
| valid negatives | `100` |
| valid positive-hard count | `67` |
| valid mean low oracle gain | `+0.3803 dB` |

Threshold sanity on the full per-image CSV:

| Proxy threshold | Easy top-25% open rate | Positive-hard recall | Negative false-open |
| --- | ---: | ---: | ---: |
| `0.50` | `0.0438` | `1.0000` | `0.0000` |
| `0.70` | `0.0319` | `0.9950` | `0.0000` |
| `0.90` | `0.0199` | `0.9900` | `0.0000` |
| `0.95` | `0.0120` | `0.9850` | `0.0000` |

## Decision

Decision label: `PASS_SIGNAL_CORRECTABILITY_PROXY_NEEDS_CALIBRATION`.

The strict gate is marked failed because the unthresholded easy mean score is
slightly above the configured `0.10` ceiling. However, ranking quality is very
strong: held-out AUC is `1.0`, Spearman is `0.9691`, and simple thresholds
strongly suppress easy openings while retaining oracle-positive hard cases.

This supports a future CorrectabilityOpen branch or thresholded proxy, but only
after the residual expression path and full-mask crop protocol are fixed.
