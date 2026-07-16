#!/usr/bin/env bash
set -euo pipefail

readonly branch="codex/command-transport-v1-20260716"
readonly ref="refs/heads/${branch}"
readonly remote_url="git@github.com:onenoober/ConvIR-B.git"
readonly shared_repo="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor"
readonly cloud_python="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
readonly evidence_root="/sda/home/wangyuxin/ConvIR-B/command_transport_validation"

for required in /usr/bin/awk /usr/bin/cp /usr/bin/git /usr/bin/grep /usr/bin/mkdir \
  /usr/bin/mktemp /usr/bin/rm /usr/bin/ssh /usr/bin/tee /usr/bin/timeout \
  /bin/bash /bin/false "${cloud_python}"; do
  if [[ ! -x "${required}" ]]; then
    printf 'required program missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if ! /usr/bin/git -C "${shared_repo}" rev-parse --git-dir >/dev/null 2>&1; then
  printf 'shared repository missing: %s\n' "${shared_repo}" >&2
  exit 2
fi

/usr/bin/mkdir -p "${evidence_root}/passed"
readonly attempt_root=$(/usr/bin/mktemp -d "${evidence_root}/attempt.XXXXXX")
readonly status_file="${attempt_root}/status.txt"
readonly log_file="${attempt_root}/cloud_test.log"
candidate_commit=UNRESOLVED
validation_state=RUNNING
work_root=""
printf 'state=RUNNING\ncandidate_branch=%s\ncandidate_commit=%s\nmodel_calls=0\n' \
  "${branch}" "${candidate_commit}" >"${status_file}"
cleanup() {
  exit_code=$?
  if [[ -n "${work_root}" && "${work_root}" == "${evidence_root}/work."* \
    && -d "${work_root}" ]]; then
    /usr/bin/rm -rf -- "${work_root}"
  fi
  if [[ "${validation_state}" != "PASS" ]]; then
    printf 'state=FAILED\ncandidate_branch=%s\ncandidate_commit=%s\nexit_code=%s\nmodel_calls=0\n' \
      "${branch}" "${candidate_commit}" "${exit_code}" >"${status_file}"
  fi
}
trap cleanup EXIT

export GIT_TERMINAL_PROMPT=0
export GCM_INTERACTIVE=never
export SSH_ASKPASS=/bin/false
export GIT_SSH_COMMAND="/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=30"
remote_line=$(/usr/bin/timeout 60 /usr/bin/git ls-remote --exit-code "${remote_url}" "${ref}")
read -r candidate_commit returned_ref <<<"${remote_line}"
if [[ ! "${candidate_commit}" =~ ^[0-9a-f]{40}$ || "${returned_ref}" != "${ref}" ]]; then
  printf 'candidate ref malformed: %s\n' "${remote_line}" >&2
  exit 2
fi
readonly pass_receipt="${evidence_root}/passed/${candidate_commit}.status"
if [[ -e "${pass_receipt}" ]]; then
  printf 'candidate already passed validation: %s\n' "${pass_receipt}" >&2
  exit 3
fi
printf 'state=RUNNING\ncandidate_branch=%s\ncandidate_commit=%s\nmodel_calls=0\n' \
  "${branch}" "${candidate_commit}" >"${status_file}"

work_root=$(/usr/bin/mktemp -d "${evidence_root}/work.XXXXXX")
{
  printf 'candidate_branch=%s\n' "${branch}"
  printf 'candidate_commit=%s\n' "${candidate_commit}"
  printf 'model_calls=0\n'
  /usr/bin/timeout 120 /usr/bin/git clone --shared --no-checkout \
    "${shared_repo}" "${work_root}/repo"
  /usr/bin/git -C "${work_root}/repo" remote add candidate "${remote_url}"
  /usr/bin/timeout 120 /usr/bin/git -C "${work_root}/repo" fetch --no-tags candidate \
    "+${ref}:refs/remotes/candidate/${branch}"
  fetched_commit=$(/usr/bin/git -C "${work_root}/repo" rev-parse \
    "refs/remotes/candidate/${branch}^{commit}")
  if [[ "${fetched_commit}" != "${candidate_commit}" ]]; then
    printf 'fetched commit mismatch: %s != %s\n' "${fetched_commit}" "${candidate_commit}" >&2
    exit 4
  fi
  /usr/bin/git -C "${work_root}/repo" checkout --detach "${candidate_commit}"
  checked_out=$(/usr/bin/git -C "${work_root}/repo" rev-parse HEAD)
  if [[ "${checked_out}" != "${candidate_commit}" ]]; then
    printf 'checkout mismatch: %s != %s\n' "${checked_out}" "${candidate_commit}" >&2
    exit 4
  fi
  /bin/bash -n "${work_root}/repo/experience_docx/tools/convir_remote_script.sh"
  /bin/bash -n "${work_root}/repo/experience_docx/tools/validate_command_transport.sh"
  "${cloud_python}" -m py_compile \
    "${work_root}/repo/experience_docx/tools/convirctl.py" \
    "${work_root}/repo/experience_docx/tools/tests/test_convirctl.py"
  "${cloud_python}" -m unittest discover -v \
    -s "${work_root}/repo/experience_docx/tools/tests" -p 'test_convirctl.py'
  "${cloud_python}" -m unittest discover -v \
    -s "${work_root}/repo/experience_docx/tools/tests" -p 'test_convir_ops_mcp.py'
  printf 'model_calls=0\n'
  printf 'COMMAND_TRANSPORT_OK\n'
} 2>&1 | /usr/bin/tee "${log_file}"

if ! /usr/bin/grep -q '^COMMAND_TRANSPORT_OK$' "${log_file}"; then
  printf 'success marker missing\n' >&2
  exit 5
fi
test_count=$(/usr/bin/awk '/^Ran [0-9]+ tests? in / {total += $2} END {print total + 0}' "${log_file}")
if [[ "${test_count}" -le 0 ]]; then
  printf 'test count missing\n' >&2
  exit 5
fi
printf 'state=PASS\ncandidate_commit=%s\ntests=%s\nmodel_calls=0\nmarker=COMMAND_TRANSPORT_OK\n' \
  "${candidate_commit}" "${test_count}" >"${status_file}"
/usr/bin/cp -- "${status_file}" "${pass_receipt}"
validation_state=PASS
printf 'status_file=%s\nCOMMAND_TRANSPORT_VALIDATION_OK\n' "${status_file}"
