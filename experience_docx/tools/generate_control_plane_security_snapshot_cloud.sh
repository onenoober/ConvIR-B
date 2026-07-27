#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_SECURITY_SNAPSHOT_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-control-plane-security-hardening-v1
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-control-plane-security-snapshot.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools=$work/repo/experience_docx/tools
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$candidate" --write >/dev/null
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$candidate" --check >/dev/null

printf 'CONTROL_PLANE_SECURITY_SNAPSHOT_JSON_BEGIN\n'
"$python" -m json.tool "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json"
printf 'CONTROL_PLANE_SECURITY_SNAPSHOT_JSON_END\n'
printf 'CONTROL_PLANE_SECURITY_SNAPSHOT_OK candidate=%s gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate"
