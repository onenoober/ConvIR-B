#!/usr/bin/env bash
set -euo pipefail

repo=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-mcp-main
validated=12801a8b63af39a46f672c94af7481887abf79eb
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python

test -d "$repo/.git"
test -z "$(/usr/bin/git -C "$repo" status --porcelain)"
before=$(/usr/bin/git -C "$repo" rev-parse HEAD)
/usr/bin/git -C "$repo" fetch --quiet --no-tags github \
  refs/heads/main:refs/remotes/github/main
target=$(/usr/bin/git -C "$repo" rev-parse refs/remotes/github/main)
/usr/bin/git -C "$repo" merge-base --is-ancestor "$validated" "$target"
/usr/bin/git -C "$repo" checkout --quiet --detach "$target"
test -z "$(/usr/bin/git -C "$repo" status --porcelain)"

tools=$repo/experience_docx/tools
"$python" -m py_compile \
  "$tools/experiment_assistant_contract.py" \
  "$tools/experiment_assistant_datasets.py" \
  "$tools/experiment_assistant_archive.py" \
  "$tools/experiment_assistant_snapshot.py" \
  "$tools/experiment_assistant_runner.py" \
  "$tools/experiment_assistant_transport.py" \
  "$tools/convir_experiment_assistant_mcp.py"

printf 'EXPERIMENT_ASSISTANT_PHASE4_CLOUD_ACTIVATED before=%s target=%s validated=%s experiment_launches=0 dataset_access=0 github_writes=0\n' \
  "$before" "$target" "$validated"
