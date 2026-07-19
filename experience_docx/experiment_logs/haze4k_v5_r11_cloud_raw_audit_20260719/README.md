# R11 Cloud Raw Evidence Audit

This directory archives the compact post-hoc read-only audit of R11 at evidence
cutoff `github/main@893ba97790ad19d745ff676f5bbf28bd37395d50`. The formal R11
terminal remains `COMPLETED_GATE_FAIL /
R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP / NONE`.

Primary files:

- `cloud_audit_closeout.json`: audit identity, reproduction status and decision.
- `r11_cloud_raw_audit_conclusion.json`: single scientific interpretation.
- `r11_cloud_raw_audit_summary.json`: exact bootstrap reproduction and principal
  post-hoc distribution statistics.
- `r11_cloud_raw_audit_cache_identity.json`: 768-unit manifest/hash audit.
- `r11_cloud_raw_audit_action_summary.csv`: positive/negative selected-action
  utility and harm.
- `r11_cloud_raw_audit_risk_coverage.csv`: descriptive score-ranked utility and
  tail behavior.
- `r11_cloud_raw_audit_label_sensitivity.csv`: descriptive eligibility-label
  changes around the frozen `+0.005 dB` local utility threshold.
- `r11_cloud_raw_audit_severe_images.csv`: nine formally severe R11 image rows
  with reconstructed action diagnostics.
- `r11_cloud_raw_audit_spatial_summary.csv`: descriptive fixed 8x8 position
  counts; no semantic or fog-severity labels.

Key result: all official bootstrap outputs reproduce exactly (maximum absolute
difference `0.0`); all 768 cache units (14,236,968,704 bytes) match their
manifest hashes. The audit supports but narrows the bottleneck to weak
candidate-conditioned relative signed-utility ranking plus uncalibrated
action-conditional downside. It authorizes `NONE`.

Cloud-only artifacts retained under
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r11_cloud_raw_audit_20260719/`:
the source 384-row per-image table, 49,152-row tile/action table, cache tensors,
runtime/status logs and audit scripts. Confirmation, canary and locked test were
not accessed.
