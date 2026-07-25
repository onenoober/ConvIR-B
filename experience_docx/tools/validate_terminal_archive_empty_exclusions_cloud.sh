#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'TERMINAL_ARCHIVE_EMPTY_EXCLUSIONS_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/terminal-archive-empty-exclusions-v1
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/terminal-archive-empty-exclusions.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools=$work/repo/experience_docx/tools
"$python" -m py_compile "$tools/prepare_terminal_archive.py" "$tools/tests/test_prepare_terminal_archive.py"
PYTHONPATH="$tools:$tools/tests" "$python" -m unittest -v test_prepare_terminal_archive

git -C "$work/repo" diff --check HEAD^ HEAD
git -C "$work/repo" diff --quiet
printf 'TERMINAL_ARCHIVE_EMPTY_EXCLUSIONS_CLOUD_OK candidate=%s model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' "$candidate"
