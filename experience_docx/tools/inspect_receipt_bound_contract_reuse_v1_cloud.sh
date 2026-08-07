#!/usr/bin/env bash
set -euo pipefail

python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
"$python" - <<'PY'
import json
from pathlib import Path

needle = b"/tmp/convir-receipt-bound-contract-reuse."
processes = []
for child in Path("/proc").iterdir():
    if not child.name.isdigit():
        continue
    try:
        raw = (child / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in raw:
        processes.append({
            "pid": int(child.name),
            "cmdline": raw.rstrip(b"\0").replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )[:1024],
        })
workspaces = sorted(
    str(path) for path in Path("/tmp").glob(
        "convir-receipt-bound-contract-reuse.*"
    ) if path.is_dir() and not path.is_symlink()
)
value = {
    "schema_version": 1,
    "active_processes": processes,
    "temporary_workspaces": workspaces,
    "clean_retry_state": not processes and not workspaces,
    "gpu_access": 0,
    "experiment_launches": 0,
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
print(
    "RECEIPT_BOUND_CONTRACT_REUSE_V1_INSPECTION_"
    + ("CLEAN" if value["clean_retry_state"] else "ACTIVE")
)
PY
