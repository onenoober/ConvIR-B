#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}"

"${PYTHON}" experience_docx/tools/smoke_ap_ria_model.py --device cpu --version base --data Haze4K --height 64 --width 64
echo AP_RIA_SMOKE_OK
