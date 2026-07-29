#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_PHASE2_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-phase2
baseline=3e9e8368a6383e6319e58b8993d1bb065fe5f56b
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-evidence-review-phase2.XXXXXX)
case "$work" in
  /tmp/convir-evidence-review-phase2.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$baseline^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$baseline" "$candidate")
expected=$'experience_docx/tools/convir_evidence_catalog.py\nexperience_docx/tools/tests/test_convir_evidence_catalog.py\nexperience_docx/tools/validate_convir_evidence_review_phase2_cloud.sh'
[[ "$changed" == "$expected" ]]

tools=$work/repo/experience_docx/tools
tests=$tools/tests
"$python" -m py_compile \
  "$tools/convir_evidence_catalog.py" \
  "$tests/test_convir_evidence_catalog.py"

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 120 "$stdout" >&2 || true
  tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 277

summary=$work/summary.json
entries=$work/entries.json
loose=$work/loose.json
PYTHONPATH="$tools" "$python" "$tools/convir_evidence_catalog.py" \
  --repo "$work/repo" --commit "$candidate" summary >"$summary"
PYTHONPATH="$tools" "$python" "$tools/convir_evidence_catalog.py" \
  --repo "$work/repo" --commit "$candidate" entries \
  --coverage indexed --term haze4k --limit 5 >"$entries"
PYTHONPATH="$tools" "$python" "$tools/convir_evidence_catalog.py" \
  --repo "$work/repo" --commit "$candidate" entries \
  --coverage unindexed --term run_v2f_f4b_tail_rescue_matrix.sh \
  --limit 1 >"$loose"
PYTHONPATH="$tools" "$python" - "$summary" "$entries" "$loose" "$candidate" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_bytes())
entries = json.loads(Path(sys.argv[2]).read_bytes())
loose = json.loads(Path(sys.argv[3]).read_bytes())
candidate = sys.argv[4]
assert summary["ok"] is True
header = summary["header"]
assert header["snapshot_commit"] == candidate
assert header["scientific_completeness"] == "not_assessed"
assert header["excluded_sources"] == ["route_branches", "cloud_runtime"]
assert header["terminal_index"]["record_count"] == 55
assert header["terminal_index"]["route_count"] == 54
assert header["terminal_index"]["schema_counts"] == {"1": 47, "2": 8}
assert header["terminal_index"]["unmodeled_record_count"] == 0
assert header["terminal_index"]["terminal_resolution_counts"] == {
    "AMBIGUOUS_LEGACY": 1, "VALID_CHAIN": 8, "VALID_SINGLE": 45,
}
assert header["terminal_index"]["sha256"] == "97ff689d00fb13d2af0c4fe2c2f6ce9c2c7e7310c3b9ab1ca8fb5cc7c7025130"
tree = header["experiment_log_tree"]
assert tree["tree_oid"] == "feff0048a4e7123306330e9b0b2025b48a2fe12b"
assert tree["tracked_file_count"] == 4011
assert tree["catalog_entry_count"] == 232
assert tree["directory_count"] == 231
assert tree["indexed_directory_count"] == 54
assert tree["unindexed_directory_count"] == 177
assert tree["loose_file_count"] == 1
assert entries["ok"] is True
assert entries["total_count"] > entries["returned_count"] > 0
assert entries["returned_count"] <= 5
assert loose["ok"] is True
assert loose["total_count"] == loose["returned_count"] == 1
assert loose["entries"][0]["record_kind"] == "loose_file"
assert loose["entries"][0]["file_name"] == "run_v2f_f4b_tail_rescue_matrix.sh"
assert loose["entries"][0]["terminal_assessment"] == "NOT_ASSESSED"
assert len(Path(sys.argv[1]).read_bytes()) <= 32768
assert len(Path(sys.argv[2]).read_bytes()) <= 32768
assert len(Path(sys.argv[3]).read_bytes()) <= 32768
PY

git -C "$work/repo" diff --check "$baseline" "$candidate"
git -C "$work/repo" diff --quiet
printf 'CONVIR_EVIDENCE_REVIEW_PHASE2_CLOUD_OK candidate=%s baseline=%s tests=%s records=55 routes=54 entries=232 directories=231 indexed=54 unindexed=177 loose=1 model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$baseline" "$test_count"
