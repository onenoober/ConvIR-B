#!/usr/bin/env bash
set -euo pipefail

PY='/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python'
GITHUB='git@github.com:onenoober/ConvIR-B.git'
CANDIDATE='codex/science-fastpath-v1-20260718'
R3_BRANCH='codex/haze4k-v5-r3-proposal-first-acv-20260717'
R3_COMMIT='4875e7715e202952abc43b41256f70d469be34bd'
R3_ROUTE='haze4k_v5_r3_proposal_first_acv_20260717'
R3_RECEIPT='1a3197afc453964482aa75f13dcda642d823dd6580a98ec674e90f38663eaca5'
TMP="$(mktemp -d /tmp/science-fastpath-acceptance.XXXXXX)"
trap 'rm -rf -- "$TMP"' EXIT

git clone --quiet --single-branch --branch "$CANDIDATE" "$GITHUB" "$TMP/candidate"
git clone --quiet --single-branch --branch "$R3_BRANCH" "$GITHUB" "$TMP/r3"
TESTED_COMMIT="$(git -C "$TMP/candidate" rev-parse HEAD)"

set +e
"$PY" -m unittest discover \
  -s "$TMP/candidate/experience_docx/tools/tests" \
  -p 'test_*.py' -v 2>&1 | tee "$TMP/tests.log"
TEST_RC=${PIPESTATUS[0]}
set -e
test "$TEST_RC" -eq 0

"$PY" "$TMP/candidate/experience_docx/tools/prepare_terminal_archive.py" \
  --source-repo "$TMP/r3" \
  --source-ref "$R3_COMMIT" \
  --route-id "$R3_ROUTE" \
  --closeout "experience_docx/experiment_logs/$R3_ROUTE/r3_a2_acv_full_oof_closeout.json" \
  --contract experience_docx/experiment_cards/2026-07-18-haze4k-v5-r3-proposal-first-acv-a2.md \
  --receipt "$R3_RECEIPT" \
  --audit-only --existing-archive \
  --report "$TMP/r3_a2_audit.json"

"$PY" - "$TMP/r3_a2_audit.json" "$TMP/tests.log" <<'PY'
import json
import re
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tests = Path(sys.argv[2]).read_text(encoding="utf-8")
assert audit["status"] == "TERMINAL_SOURCE_AUDIT_OK"
assert len(audit["result_files"]) == 9
assert all(audit["checks"].values())
matches = re.findall(r"Ran ([0-9]+) tests?", tests)
assert matches and int(matches[-1]) >= 100
print(
    "SCIENCE_FASTPATH_ACCEPTANCE_ASSERTIONS_OK "
    f"tests={matches[-1]} real_result_files={len(audit['result_files'])}"
)
PY

printf 'SCIENCE_FASTPATH_ACCEPTANCE_OK tested_commit=%s\n' "$TESTED_COMMIT"
