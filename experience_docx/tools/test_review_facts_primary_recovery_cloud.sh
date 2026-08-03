#!/usr/bin/env bash
set -euo pipefail

branch="codex/review-facts-primary-recovery-v1"
python_bin="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
runtime_root="/sda/home/wangyuxin/ConvIR-B/runtime"
repo="$(mktemp -d "${runtime_root}/review-facts-primary-recovery.XXXXXX")"

cleanup() {
  rm -rf -- "${repo}"
}
trap cleanup EXIT

/usr/bin/git clone --quiet --filter=blob:none --branch "${branch}" --single-branch \
  git@github.com:onenoober/ConvIR-B.git "${repo}"

test_root="${repo}/experience_docx/tools/tests"
test_path="${repo}/experience_docx/tools:${test_root}"

PYTHONPATH="${test_path}" "${python_bin}" -m unittest discover \
  -s "${test_root}" -p "test_prepare_terminal_archive.py" -v
PYTHONPATH="${test_path}" "${python_bin}" -m unittest discover \
  -s "${test_root}" -p "test_convir_evidence_catalog.py" -v
PYTHONPATH="${test_path}" "${python_bin}" -m unittest discover \
  -s "${test_root}" -p "test_convir_evidence_cloud_inventory.py" -v

printf '%s\n' "REVIEW_FACTS_PRIMARY_RECOVERY_CLOUD_OK"
