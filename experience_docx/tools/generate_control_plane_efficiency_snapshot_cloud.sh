#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_EFFICIENCY_SNAPSHOT_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-control-plane-efficiency-v1
base=ed884b4ebd3b0f733381093058f3f694c454f390
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-control-plane-snapshot.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- experience_docx/experiment_logs

tools=$work/repo/experience_docx/tools
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$candidate" --write >/dev/null
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$candidate" --check >/dev/null

printf 'CONTROL_PLANE_EFFICIENCY_SNAPSHOT_JSON_BEGIN\n'
cat "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json"
printf 'CONTROL_PLANE_EFFICIENCY_SNAPSHOT_JSON_END\n'
printf 'CONTROL_PLANE_EFFICIENCY_SNAPSHOT_OK candidate=%s protected_data_access=0 experiment_launches=0\n' \
  "$candidate"
