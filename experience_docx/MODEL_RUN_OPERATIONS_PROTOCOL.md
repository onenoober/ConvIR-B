# Model Run Operations Protocol

Date: 2026-07-16

## Launch Order

```text
exact route commit -> card/manifest operation -> dynamic cloud preflight ->
tracked runner -> bounded observation -> typed closeout -> compact archive
```

Local WSL remains syntax/compile-only. Before launch, verify exact route HEAD,
clean/fresh or exact-continuation workspace, runner/assets, prior closeout, data
role, locked-test policy, GPU floor, new/exact-resume output, status, heartbeat,
log and closeout paths. Never substitute a commit, data, split, checkpoint,
threshold, output, or stage silently.

## Runner

One tracked runner must use `set -euo pipefail` and the explicit cloud Python;
run integrated identity/input/no-op/shape/finite checks before expensive work;
write progress and heartbeat to `status.txt`; capture stdout/stderr; reject
forbidden data/locked-test access; and write one closeout bound to route id, run
id, route commit, runner SHA-256, evidence role, and allowed terminal tuple.

Resume only at predeclared complete units with matching hashes. An incomplete
unit restarts. Resume cannot reveal confirmation results early or change the
scientific contract.

## Observe And Stop

`convir_route_finish` observes one window: `short` is 30 seconds and `standard`
is 60 seconds. A healthy active run may be observed again. Stale heartbeat or a
dead session without closeout gets one engineering inspection and then stops;
never create a watcher loop or task per poll.

Failure classes stay distinct: command/transport, preflight/resource,
engineering/runtime, evidence/closeout, and scientific/safety gate. Use one
deterministic command correction and one engineering repair cycle per root
cause. `START_STATE_UNKNOWN` requires one inspection and no blind retry.
Resource wait may retry the unchanged prelaunch plan.

Raw logs, checkpoints, images, arrays, predictions and large tables remain in
the cloud run root. Only curated compact text evidence is eligible for Git.
