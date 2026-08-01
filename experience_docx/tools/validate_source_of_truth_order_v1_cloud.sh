#!/bin/bash
set -euo pipefail

# Read-only cloud validation of the source-of-truth routing contract.
BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
MAIN_SHA=bd73a099803dcca6ded09401ea7568ebb0e6ba71
TMP_ROOT=$(mktemp -d)
REPO=$TMP_ROOT/repo
trap 'rm -rf "$TMP_ROOT"' EXIT

REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/main")
read -r REMOTE_SHA REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_SHA" = "$MAIN_SHA"
test "$REMOTE_REF" = refs/heads/main
/usr/bin/git clone --quiet --no-checkout --depth 1 "$REMOTE_URL" "$REPO"
/usr/bin/git -C "$REPO" fetch --quiet --depth 1 origin "$MAIN_SHA"
/usr/bin/git -C "$REPO" checkout --quiet --detach "$MAIN_SHA"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"

"$PYTHON" "$REPO/experience_docx/tools/policy_snapshot.py" \
  --repo "$REPO" --rules-commit "$MAIN_SHA" --check
"$PYTHON" -m py_compile "$REPO/experience_docx/tools/policy_snapshot.py"
"$PYTHON" "$REPO/experience_docx/tools/validate_source_of_truth_order_v1.py" "$REPO"
printf 'SOURCE_OF_TRUTH_ORDER_CLOUD_OK\n'
