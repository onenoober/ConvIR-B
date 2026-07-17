#!/usr/bin/env bash
set -euo pipefail

repo_root=/sda/home/wangyuxin/ConvIR-B/repos
run_root=/sda/home/wangyuxin/ConvIR-B/runs

printf 'START_UNKNOWN_INSPECTION_BEGIN\n'
printf 'SESSIONS_BEGIN\n'
/usr/bin/tmux list-sessions -F 'session=#{session_name} created=#{session_created} attached=#{session_attached}' 2>/dev/null \
  | /usr/bin/sed -n '/^session=convir-/p' || true
printf 'SESSIONS_END\n'

printf 'RECENT_REPOS_BEGIN\n'
while IFS= read -r -d '' repo; do
  head=$(/usr/bin/git -C "$repo" rev-parse HEAD 2>/dev/null || printf unavailable)
  branch=$(/usr/bin/git -C "$repo" branch --show-current 2>/dev/null || printf unavailable)
  dirty=$(/usr/bin/git -C "$repo" status --porcelain 2>/dev/null | /usr/bin/wc -l || printf unavailable)
  printf 'repo=%s head=%s branch=%s dirty_entries=%s\n' "$repo" "$head" "$branch" "$dirty"
done < <(/usr/bin/find "$repo_root" -mindepth 1 -maxdepth 1 -type d -mmin -60 -print0 \
  | /usr/bin/sort -z)
printf 'RECENT_REPOS_END\n'

printf 'RECENT_OUTPUTS_BEGIN\n'
while IFS= read -r -d '' output; do
  identity="$output/control/lifecycle_identity.json"
  status="$output/status.txt"
  heartbeat="$output/heartbeat.json"
  identity_sha=absent
  status_bytes=0
  heartbeat_bytes=0
  [[ ! -f $identity ]] || identity_sha=$(/usr/bin/sha256sum "$identity" | /usr/bin/awk '{print $1}')
  [[ ! -f $status ]] || status_bytes=$(/usr/bin/stat -c %s "$status")
  [[ ! -f $heartbeat ]] || heartbeat_bytes=$(/usr/bin/stat -c %s "$heartbeat")
  printf 'output=%s identity_sha256=%s status_bytes=%s heartbeat_bytes=%s\n' \
    "$output" "$identity_sha" "$status_bytes" "$heartbeat_bytes"
  /usr/bin/find "$output/control" -maxdepth 1 -type f -name '*_closeout.json' \
    -printf 'output_closeout=%p bytes=%s\n' 2>/dev/null | /usr/bin/sort || true
done < <(/usr/bin/find "$run_root" -mindepth 2 -maxdepth 2 -type d -mmin -60 -print0 \
  | /usr/bin/sort -z)
printf 'RECENT_OUTPUTS_END\n'
printf 'START_UNKNOWN_INSPECTION_OK\n'
