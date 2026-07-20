#!/usr/bin/env bash
set -euo pipefail

branch="codex/engineering-repair-card-normalization-20260720"
python_bin="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
runtime_root="/sda/home/wangyuxin/ConvIR-B/runtime"
repo="$(mktemp -d "${runtime_root}/engineering-repair-card-normalization.XXXXXX")"

cleanup() {
  rm -rf -- "${repo}"
}
trap cleanup EXIT

/usr/bin/git clone --quiet --filter=blob:none --branch "${branch}" --single-branch \
  git@github.com:onenoober/ConvIR-B.git "${repo}"

PYTHONPATH="${repo}/experience_docx/tools" "${python_bin}" -m unittest \
  discover -s "${repo}/experience_docx/tools/tests" \
  -p "test_validate_engineering_repair.py" -v

printf '%s\n' "ENGINEERING_REPAIR_CARD_NORMALIZATION_CLOUD_OK"
