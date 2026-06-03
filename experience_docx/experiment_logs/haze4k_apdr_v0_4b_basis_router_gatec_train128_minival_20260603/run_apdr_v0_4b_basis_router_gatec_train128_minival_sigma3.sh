#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-derived-lowfield-basis}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603}
TAG=${TAG:-apdr_v0_4b_basis_router_gatec_train128_minival_sigma3_seed3407}
STAGE_LABEL=${STAGE_LABEL:-APDR-v0.4B basis-only coefficient router Gate C train128 mini-val256}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-basis_router_gatec_train128_minival}
STATUS_NAME=${STATUS_NAME:-v0_4b_basis_router_gatec_train128_minival}
BASIS_NUM_IMAGES=${BASIS_NUM_IMAGES:-0}
TRAIN_COUNT=${TRAIN_COUNT:-128}
EVAL_COUNT=${EVAL_COUNT:-256}
FIT_COUNT=${FIT_COUNT:-0}
LOW_SIZE=${LOW_SIZE:-32}
K_VALUES=${K_VALUES:-32}
STEPS=${STEPS:-2000}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

export ROOT LOG_DIR TAG STAGE_LABEL ARTIFACT_PREFIX STATUS_NAME
export BASIS_NUM_IMAGES TRAIN_COUNT EVAL_COUNT FIT_COUNT LOW_SIZE K_VALUES STEPS PROGRESS_FREQ

bash "$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/run_apdr_v0_4b_basis_router_gateb_sigma3.sh"
