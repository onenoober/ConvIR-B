#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4a-low-field-only}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603"
STATUS="$LOG_DIR/status.txt"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "start_residual_forms_sigma3_sequence root=$ROOT $(date --iso-8601=seconds)" | tee -a "$STATUS"
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_id_embedding_sigma3.sh || true
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_basis_sigma3.sh || true
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_basis_local_sigma3.sh || true
bash experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/run_apdr_v0_4a_veil_sigma3.sh || true
echo "complete_residual_forms_sigma3_sequence $(date --iso-8601=seconds)" | tee -a "$STATUS"
