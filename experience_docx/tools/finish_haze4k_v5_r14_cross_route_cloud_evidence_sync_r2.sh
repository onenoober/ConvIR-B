#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B
REPO=${ROOT}/repos/ConvIR-B-r14-evidence-sync-r2-2667d59ed
BRANCH=codex/haze4k-v5-r14-cross-route-cloud-evidence-audit-20260720
PREFIX=experience_docx/experiment_logs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720/

if [[ ! -d "${REPO}/.git" ]]; then
  printf 'R14_SYNC_REPO_MISSING\n' >&2
  exit 2
fi
/usr/bin/git -C "${REPO}" config user.name Codex
/usr/bin/git -C "${REPO}" config user.email codex@openai.com
COUNT=0
for name in $(/usr/bin/git -C "${REPO}" diff --cached --name-only); do
  case "${name}" in
    ${PREFIX}*.json|${PREFIX}*.md|${PREFIX}*.txt|${PREFIX}*.log) ;;
    *) printf 'R14_SYNC_STAGED_PATH_FORBIDDEN\t%s\n' "${name}" >&2; exit 3 ;;
  esac
  COUNT=$((COUNT + 1))
done
if [[ "${COUNT}" -ne 13 ]]; then
  printf 'R14_SYNC_STAGED_COUNT_MISMATCH\t%s\n' "${COUNT}" >&2
  exit 4
fi
/usr/bin/git -C "${REPO}" diff --cached --check
/usr/bin/git -C "${REPO}" commit -m "Archive R14 compact cloud evidence" >/dev/null
/usr/bin/git -C "${REPO}" push origin "${BRANCH}" >/dev/null
printf 'R14_CLOUD_EVIDENCE_SYNC_OK\t%s\n' "$(/usr/bin/git -C "${REPO}" rev-parse HEAD)"
