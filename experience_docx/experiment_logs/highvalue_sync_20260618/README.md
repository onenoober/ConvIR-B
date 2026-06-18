# High-Value Local Evidence Sync 2026-06-18

Status: `TEXT_EVIDENCE_SYNC_NO_LARGE_RAW`

This sync copied only high-value, GitHub-readable local evidence into the
durable repository archive. The cloud server remains the runtime/training/test
location; GitHub is the reading and record authority.

Included:

- governance/navigation updates: experiment index, README, branch route index,
  official anchor policy, cloud environment guide, DTA family summary;
- DTA-v3.7 and official-anchor route cards needed by the current index;
- C13 diagnostic closeout docs, status files, command script, leaderboard,
  grouped summaries, failure taxonomy, and diagnostic tool code.

Excluded by design:

- checkpoints, weights, datasets, arrays, images, archives, PDFs, and raw
  inference outputs;
- large per-image/raw CSV tables above the normal text-evidence threshold;
- superseded or duplicate raw logs when a decision/summary file already records
  the useful outcome.

Useful source worktrees inspected:

- `/home/ubuntu/workspace/ConvIR-B`
- `/home/ubuntu/workspace/ConvIR-B-c13-diagnostics`

Primary route evidence completed by this sync:

- `experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615/`
