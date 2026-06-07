# v1.8 Remote Access Blocker

Date: 2026-06-06 05:09 +08:00

State label: `FAILED_INFRA_REMOTE_ACCESS_BLOCKED_FOR_MONITOR_SYNC`

This is a monitoring/sync infrastructure blocker, not a scientific route
result and not evidence that the cloud queue failed.

## Observed State

- Repository `AGENTS.md` declares `dehaze1` as
  `root@connect.bjb1.seetacloud.com:49601`.
- Current WSL `~/.ssh/config` resolves `dehaze1` to
  `root@connect.bjb2.seetacloud.com:40921` using
  `~/.ssh/id_ed25519_seetacloud`.
- `ssh dehaze1` failed with `Connection refused` on
  `connect.bjb2.seetacloud.com:40921`.
- `ssh-keyscan -p 49601 -t ed25519 connect.bjb1.seetacloud.com` succeeded and
  returned ED25519 host fingerprint
  `SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc`.
- The existing WSL `known_hosts` entries for
  `[connect.bjb1.seetacloud.com]:49601` do not match that host key, so normal
  strict SSH produced `REMOTE HOST IDENTIFICATION HAS CHANGED`.
- A temporary pinned-host-key probe to
  `root@connect.bjb1.seetacloud.com:49601` failed authentication with
  `Permission denied (publickey,password)`.
- WSL private-key probes with `id_ed25519_seetacloud` and `id_ed25519` both
  failed authentication.
- Windows-side probes with `id_ed25519`, `id_rsa`, `key.pem`, and
  `devbox_ed25519` also failed authentication.

## Operational Decision

- Do not disable host-key checking permanently.
- Do not remove or replace `known_hosts` entries without confirming the current
  cloud endpoint from the provider/control panel.
- Do not run model tests, training, eval, inference, or runtime validation
  locally. The repository rule still requires runtime work on `dehaze1`.
- Keep the v1.8 monitor heartbeat active. Once remote access is restored, the
  next action is to refresh `v18_progress`, sync text evidence from the cloud,
  and let `v18_eval_repair` finish missing evals if the main queue has ended.

## Last Locally Synced Queue Snapshot

The latest successful cloud sync before this blocker was generated at
`2026-06-06T04:56:52+08:00`.

- `v18_execution_queue=ACTIVE`
- `v18_eval_repair=ACTIVE`
- `v18_domain_adaptation_q5=NOT_ACTIVE`
- Q1 corrected router: completed, gate failed, queue continued.
- Q2 data/domain preflight: completed, `missing_count=0`.
- Q5 data/domain adaptation coverage: completed,
  `DOMAIN_POLICY_GATE_FAIL_CONTINUE_V18_QUEUE`.
- Q3 seed status from local `v18_progress` refresh:
  - repair pending: `3407`, `2026`
  - evidence complete: `929`, `123`, `777`
  - running at last sync: `1701`
- pending at last sync: `2222`, `3141`, `4242`, `5151`

## Recovery

Recovery timestamp: `2026-06-06 10:28 +08:00`

State label: `RECOVERED_INFRA_ACCESS_RESTORED`

- The user confirmed that the active server moved to
  `root@connect.bjb1.seetacloud.com:16124` with the same private key.
- Local WSL `~/.ssh/config` and repository `AGENTS.md` were updated to that
  endpoint.
- Strict host-key probing and authenticated SSH succeeded against the new
  endpoint.
- Patched v1.8 runtime support files were re-synced to
  `/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue`.
- `v18_execution_queue` and `v18_eval_repair` were relaunched in tmux without
  rerunning completed seeds.
- `seed_1701` resumed from
  `Dehazing/ITS/results/ConvIR-Haze4K-v1.8-BiDPFM1-fusion-neighbor-seed1701-20260606/Training-Results/model.pkl`
  at saved epoch `6`, and monitor output at `2026-06-06T10:30:22+08:00`
  showed active training on the RTX 4090 (`11000 MiB`, `99%` GPU utilization).
- The refreshed `v18_progress` artifacts now classify `3407` and `2026` as
  `EVAL_FAILED_ENGINEERING_REPAIR_PENDING`, separating those eval import/path
  failures from scientific route results.
