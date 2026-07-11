#!/usr/bin/env bash
set -euo pipefail
MODE=smoke exec "$(dirname "$0")/run_v3m_a1_local_actuation.sh"
