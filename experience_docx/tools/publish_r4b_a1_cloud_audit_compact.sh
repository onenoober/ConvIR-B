#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
ROUTE_ID=haze4k_v5_r4b_three_action_setwise_utility_risk_20260718
RUN_ID=r4b-a1-setwise-screen-r1
SOURCE="$BASE/audits/$ROUTE_ID/r4b-a1-cloud-audit-20260718-r1"

digest=$(printf '%s\0%s' "$ROUTE_ID" "$RUN_ID" | sha256sum | cut -c1-16)
prefix="${ROUTE_ID:0:32}-${RUN_ID:0:24}"
prefix="${prefix:0:56}"
REMOTE_REPO="$BASE/repos/$prefix-$digest"
DESTINATION="$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID"
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

test -d "$SOURCE"
test -d "$DESTINATION"
for name in "${FILES[@]}"; do
  source_path="$SOURCE/$name"
  destination_path="$DESTINATION/$name"
  test -f "$source_path"
  test ! -e "$destination_path"
  /bin/cp --preserve=mode,timestamps "$source_path" "$destination_path"
  /usr/bin/cmp -s "$source_path" "$destination_path"
  printf 'PUBLISHED %s SHA256=%s\n' "$name" "$(sha256sum "$destination_path" | cut -d' ' -f1)"
done
printf 'REMOTE_R4B_A1_CLOUD_AUDIT_COMPACT_PUBLISH_OK\n'
