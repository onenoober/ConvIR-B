#!/usr/bin/env bash
set -euo pipefail

PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
: "${REMOTE_REPO:?REMOTE_REPO is required}"
exec "$PY" "$REMOTE_REPO/experience_docx/tools/route_lifecycle.py"
