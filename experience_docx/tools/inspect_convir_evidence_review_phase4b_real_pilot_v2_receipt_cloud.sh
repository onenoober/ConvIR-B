#!/usr/bin/env bash
set -euo pipefail

python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
receipt=/sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review/receipts/convir-evidence-review-phase4b-real-pilot-v2.json

"$python" - "$receipt" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
info = path.lstat()
assert stat.S_ISREG(info.st_mode)
assert not path.is_symlink()
assert info.st_size <= 16384
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    data = os.read(fd, 16385)
finally:
    os.close(fd)
assert len(data) == info.st_size
assert len(data) <= 16384
receipt = json.loads(data)
assert receipt["schema_version"] == 2
assert receipt["receipt_id"] == "convir-evidence-review-phase4b-real-pilot-v2-receipt"
assert receipt["pilot_id"] == "convir-evidence-review-phase4b-real-pilot-v2"
assert receipt["github_binding"] == {
    "snapshot_commit": "e4ddd62ef1e6b45bec6f70b5197ef6a72de43531",
    "catalog_sha256": "d13cdfd1a13b15f6f085155dfc77630145ee539ea7bb9143d3be88db6dbebff2",
    "terminal_record_sha256": "7c896f414cb3f9d1feb07e9b8817685b3fcfea6e7225bbb887ab073e740c4530",
}
state = receipt["state"]
if state == "PHASE4B_REAL_PILOT_PASS":
    assert receipt["accepted"] is True
    outcome = receipt["outcome"]
    assert outcome["summary_calls"] == 1
    assert outcome["query_calls"] == 1
    assert outcome["query_entries"] == 1
    assert outcome["matched_count"] >= 1
    assert outcome["temporary_workspace_removed"] is True
    assert all(value == 0 for value in receipt["access_observed"].values())
else:
    assert receipt["accepted"] is False
sha256 = hashlib.sha256(data).hexdigest()
inventory_sha256 = receipt.get("outcome", {}).get("inventory_sha256", "none")
print(
    "CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V2_RECEIPT_OK "
    f"state={state} accepted={str(receipt['accepted']).lower()} "
    f"candidate={receipt.get('candidate_commit') or 'unknown'} "
    f"inventory_sha256={inventory_sha256} receipt_sha256={sha256} schema_version=2"
)
PY
