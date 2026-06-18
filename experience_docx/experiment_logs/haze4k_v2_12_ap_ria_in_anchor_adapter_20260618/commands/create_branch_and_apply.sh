#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-codex/haze4k-v2-12-ap-ria-in-anchor-adapter}"
PATCH="${2:-ap_ria_v212.patch}"
REMOTE="${REMOTE:-github}"
ANCHOR="${ANCHOR:-${REMOTE}/codex/haze4k-official-arch-anchor}"

git fetch "${REMOTE}" '+refs/heads/*:refs/remotes/'"${REMOTE}"'/*'
git switch --detach "${ANCHOR}"
git switch -c "${BRANCH}"
git apply "${PATCH}"
git status --short
echo AP_RIA_BRANCH_APPLY_OK
