#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B
BRANCH=codex/haze4k-v5-r14-cross-route-cloud-evidence-audit-20260720
BASE_COMMIT=2667d59ed5537147818be566d9f2a3d5ec8944b1
CLOUD=${ROOT}/runs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720/r14-cross-route-audit-r2
REPO=${ROOT}/repos/ConvIR-B-r14-evidence-sync-r2-2667d59ed
DEST=${REPO}/experience_docx/experiment_logs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720
FILES=(
  input_identity.json
  raw_contract_checks.json
  official_result_reproduction.json
  a1_regret_attribution.json
  a2_label_region_stability.json
  a3_risk_observability.json
  a4_target_alignment.json
  a5_external_directional_reference.json
  key_raw_findings.json
  cloud_audit_closeout.json
  scientific_conclusion.json
  status.txt
  runtime.log
)

if [[ -e "${REPO}" ]]; then
  printf 'R14_SYNC_FRESH_REPO_REQUIRED\n' >&2
  exit 2
fi
/usr/bin/git clone --branch "${BRANCH}" --single-branch git@github.com:onenoober/ConvIR-B.git "${REPO}" >/dev/null 2>&1
if ! /usr/bin/git -C "${REPO}" merge-base --is-ancestor "${BASE_COMMIT}" HEAD; then
  printf 'R14_SYNC_BASE_COMMIT_NOT_ANCESTOR\n' >&2
  exit 3
fi
for name in "${FILES[@]}"; do
  if [[ ! -f "${CLOUD}/${name}" ]]; then
    printf 'R14_SYNC_SOURCE_MISSING\t%s\n' "${name}" >&2
    exit 4
  fi
  /usr/bin/cp "${CLOUD}/${name}" "${DEST}/${name}"
done
/usr/bin/git -C "${REPO}" add "experience_docx/experiment_logs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720"
for name in $(/usr/bin/git -C "${REPO}" diff --cached --name-only); do
  case "${name}" in
    *.json|*.md|*.txt|*.log) ;;
    *) printf 'R14_SYNC_FORBIDDEN_SUFFIX\t%s\n' "${name}" >&2; exit 5 ;;
  esac
done
/usr/bin/git -C "${REPO}" diff --cached --check
/usr/bin/git -C "${REPO}" commit -m "Archive R14 compact cloud evidence" >/dev/null
/usr/bin/git -C "${REPO}" push origin "${BRANCH}" >/dev/null
printf 'R14_CLOUD_EVIDENCE_SYNC_OK\t%s\n' "$(/usr/bin/git -C "${REPO}" rev-parse HEAD)"
