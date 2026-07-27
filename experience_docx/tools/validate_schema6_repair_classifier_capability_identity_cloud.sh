#!/usr/bin/env bash
set -euo pipefail

branch=codex/schema6-repair-classifier-capability-identity-fix
baseline=f195ddd3b6a22048dd8910eb7d15336c1053003f
github=git@github.com:onenoober/ConvIR-B.git
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
work=$(/usr/bin/mktemp -d "${runtime_root}/schema6-repair-classifier.XXXXXX")

cleanup() {
  case "${work}" in
    "${runtime_root}"/schema6-repair-classifier.*)
      /bin/rm -rf -- "${work}"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "${work}" >&2
      ;;
  esac
}
trap cleanup EXIT

/usr/bin/git clone --quiet --filter=blob:none --branch "${branch}" --single-branch \
  "${github}" "${work}/repo"
candidate=$(/usr/bin/git -C "${work}/repo" rev-parse HEAD)
/usr/bin/git -C "${work}/repo" cat-file -e "${baseline}^{commit}"
/usr/bin/git -C "${work}/repo" merge-base --is-ancestor "${baseline}" "${candidate}"
test -z "$(/usr/bin/git -C "${work}/repo" status --porcelain)"

changed=$(/usr/bin/git -C "${work}/repo" diff --name-only "${baseline}" "${candidate}")
expected=$'experience_docx/tools/tests/test_validate_engineering_repair.py\nexperience_docx/tools/validate_engineering_repair.py\nexperience_docx/tools/validate_schema6_repair_classifier_capability_identity_cloud.sh'
[[ "${changed}" == "${expected}" ]]

"${python}" -m py_compile \
  "${work}/repo/experience_docx/tools/validate_engineering_repair.py" \
  "${work}/repo/experience_docx/tools/tests/test_validate_engineering_repair.py"

PYTHONPATH="${work}/repo/experience_docx/tools:${work}/repo/experience_docx/tools/tests" \
  "${python}" -m unittest discover \
  -s "${work}/repo/experience_docx/tools/tests" \
  -p 'test_validate_engineering_repair.py' -v

/usr/bin/git -C "${work}/repo" diff --check "${baseline}" "${candidate}"
/usr/bin/git -C "${work}/repo" diff --quiet
printf 'SCHEMA6_REPAIR_CLASSIFIER_CAPABILITY_IDENTITY_CLOUD_OK candidate=%s baseline=%s model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "${candidate}" "${baseline}"
