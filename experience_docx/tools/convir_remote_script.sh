#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/remote_script.sh" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(/usr/bin/dirname -- "$0")" && pwd)
exec /usr/bin/python3 "${script_dir}/convirctl.py" \
  remote-script --script "$1"
