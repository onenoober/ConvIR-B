#!/usr/bin/env bash
set -euo pipefail
export A1X_S0_CLOSEOUT_FILENAME=a1x_v3_s0_r2_closeout.json
export A1X_REFERENCE_SOURCE="$REMOTE_REPO/experience_docx/tools/a1x_v3_a1c_exact_half_reference.py"
export A1X_RUNNER_RELPATH=experience_docx/tools/run_chd_rm_v4a_a1x_accessibility_v3_s0_r2.sh
exec "$REMOTE_REPO/experience_docx/tools/run_chd_rm_v4a_a1x_accessibility_v3_s0.sh"
