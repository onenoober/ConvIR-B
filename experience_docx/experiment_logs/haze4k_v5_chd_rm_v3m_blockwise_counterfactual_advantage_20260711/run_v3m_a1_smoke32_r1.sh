#!/usr/bin/env bash
set -euo pipefail
MODE=smoke SMOKE_OUTPUT_DIR=a1_smoke32_r1 SMOKE_TAG=v3m_a1_smoke32_r1 \
  exec "$(dirname "$0")/run_v3m_a1_local_actuation.sh"
