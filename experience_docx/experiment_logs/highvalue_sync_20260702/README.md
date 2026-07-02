# High-Value Three-Way Evidence Sync 2026-07-02

Status: `THREE_WAY_SYNC_COMPACT_EVIDENCE_ONLY`

This audit compared the local editing workspace, the GitHub `main` archive, and
the cloud runtime tree at `/sda/home/wangyuxin/ConvIR-B/`.

Roles after this sync:

- GitHub `main`: durable reader-facing archive for compact evidence,
  governance docs, route cards, decisions, summary/status files, aggregate
  tables, and small reproducibility scripts.
- Cloud `convir-4090`: runtime source for training/evaluation outputs, raw
  tables, checkpoints, datasets, images, arrays, and dirty route worktrees.
- Local WSL: editing and sync staging only; the dirty route workspace is not a
  GitHub sync authority.

Included in this sync:

- DTA-v3.7 compact phase evidence for Phase A through D7 plus missing v37
  reproducibility tools referenced by the route README.
- v2.0 StrongExpert D8/D9 reconciliation and forensic files that were already
  referenced by README/card/family docs but missing from GitHub.
- C13-F full-600 summary JSON files; per-image C13 outputs remain excluded.

Explicitly not synced:

- Cloud C11 locked decision, because it still says
  `LOCKED_C11_SELECTOR_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER`; GitHub `main`
  correctly records `LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED_DO_NOT_PROMOTE_OVER_WD0375`.
- Cloud C13/v26/v27 README and decision files that are older than GitHub
  `main` closeouts.
- DTA per-image/action tables, selected-image/action tables, raw feature tables,
  images, checkpoints, weights, datasets, arrays, archives, and large runtime
  logs.

Operational note:

- PowerShell-to-WSL stdin scripts should be piped through `tr -d '\r'` before
  `bash`; this audit hit the known CRLF boundary failure and used the corrected
  form from `COMMAND_RELIABILITY_PROTOCOL.md`.
