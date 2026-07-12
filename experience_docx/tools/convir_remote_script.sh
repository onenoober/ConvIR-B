#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/remote_script.sh" >&2
  exit 2
fi

source_script=$1
if [[ ! -f "$source_script" ]]; then
  echo "CONVIR_REMOTE_SCRIPT_FAILED missing_file=$source_script" >&2
  exit 2
fi

normalized_script=$(mktemp)
trap 'rm -f "$normalized_script"' EXIT

LC_ALL=C sed '1s/^\xEF\xBB\xBF//' "$source_script" | tr -d '\r' > "$normalized_script"

if [[ ! -s "$normalized_script" ]]; then
  echo "CONVIR_REMOTE_SCRIPT_FAILED empty_script=$source_script" >&2
  exit 2
fi

bash -n "$normalized_script"

if ssh convir-4090 'bash -s' < "$normalized_script"; then
  echo CONVIR_REMOTE_SCRIPT_OK
else
  rc=$?
  echo "CONVIR_REMOTE_SCRIPT_FAILED rc=$rc" >&2
  exit "$rc"
fi
