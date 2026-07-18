#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/audits/haze4k_v5_r4b_three_action_setwise_utility_risk_20260718/r4b-a1-cloud-audit-20260718-r1
FILES=(
  cloud_evidence_manifest.json
  provenance_and_identity_audit.json
  official_metric_reproduction.json
  github_cloud_discrepancy.csv
  fold_seed_operator_stability.csv
  per_sample_distribution_summary.json
  severe_hard_case_summary.csv
  risk_coverage_reanalysis.csv
  action_margin_and_label_stability.csv
  subgroup_failure_summary.csv
  engineering_integrity_audit.json
  updated_bottleneck_assessment.json
  prior_plan_reassessment.json
  cloud_audit_closeout.json
)

total=0
for name in "${FILES[@]}"; do
  path="$ROOT/$name"
  test -f "$path"
  bytes=$(stat -c '%s' "$path")
  total=$((total + bytes))
done
test "$total" -le 60000

for name in "${FILES[@]}"; do
  printf 'CLOUD_AUDIT_FILE_BEGIN %s\n' "$name"
  /bin/cat "$ROOT/$name"
  printf 'CLOUD_AUDIT_FILE_END %s\n' "$name"
done
printf 'REMOTE_R4B_A1_CLOUD_AUDIT_EXPORT_OK total_bytes=%s\n' "$total"
