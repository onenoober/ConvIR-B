#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
ROUTE_BRANCH=codex/haze4k-v5-v4a-conditional-safety-audit-20260714
ROUTE_COMMIT=${ROUTE_COMMIT:-}
REPO_URL=${REPO_URL:-git@github.com:onenoober/ConvIR-B.git}
REMOTE_REPO=$BASE/repos/ConvIR-B-v4a-conditional-safety-audit-20260714
V3Z_ROOT=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
V3S_ROOT=$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b
V3P_ROOT=$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea

clone_snapshot() {
  local target=$1
  local commit=$2
  test ! -e "$target"
  git clone "$REPO_URL" "$target"
  git -C "$target" checkout --detach "$commit"
  test "$(git -C "$target" rev-parse HEAD)" = "$commit"
  test -z "$(git -C "$target" status --porcelain)"
}

test ! -e "$REMOTE_REPO"
git clone --branch "$ROUTE_BRANCH" "$REPO_URL" "$REMOTE_REPO"
if [ -n "$ROUTE_COMMIT" ]; then
  test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$ROUTE_COMMIT"
fi
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"

clone_snapshot "$V3Z_ROOT" 3caddcc5265732e5be77e3404119a28cb28c11e6
clone_snapshot "$V3S_ROOT" 2860f580bb25cc75ec9ade56378af6d77f5c8d8b
clone_snapshot "$V3P_ROOT" 555fd008e29f02128564f2fad41d0095ee44f5ea

test -x "$BASE/envs/convir-cu121/bin/python"
test -f "$REMOTE_REPO/experience_docx/tools/run_chd_rm_v4a_a0r.sh"
test -f "$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a0r_reconstruct.py"
printf 'V4A_CLOUD_ROUTE_HEAD=%s\n' "$(git -C "$REMOTE_REPO" rev-parse HEAD)"
echo V4A_CLOUD_SETUP_OK
