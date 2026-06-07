# Haze4K v1.8 Execution Queue Evidence

Route card:
`experience_docx/experiment_cards/2026-06-06-haze4k-convir-v1-8-execution-queue.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

Status: `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`

## Purpose

This evidence directory is the final text archive for the post-diagnosis
execution queue. The queue was designed to continue through all declared
experiments even when an earlier independent experiment failed, and this
directory now contains the completed Q1-Q5 evidence plus post-queue repair
outputs for the early `3407/2026` eval import-path failure.

## Runtime Contract

- Runtime server: `dehaze1`
- Remote workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue`
- Python: `/root/miniconda3/envs/convir-cu128/bin/python`
- Locked Haze4K test touched: `false`
- Queue launcher: `run_v18_execution_queue.sh`
- Resume launcher: `run_v18_execution_queue_resume.sh`
- Q5 launcher: `run_v18_domain_adaptation_q5.sh`
- Post-queue repair launcher: `repair_v18_missing_eval_and_aggregate.sh`

## Final Decision

Decision label: `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`

Final route result:

- Q1 corrected router gate failed:
  OOF mean `+0.0557 dB`, hard bottom-25 `+0.3324 dB`, easy top-25
  `-0.1049 dB`, worst ratio `0.0783`; heldout mean `+0.2140 dB`, hard
  bottom-25 `+0.4173 dB`, easy top-25 `-0.0601 dB`, worst ratio `0.0717`.
- Q2 data/domain preflight completed with `3000` rows and `missing_count=0`.
- Q5 recorded `REAL_DOMAIN_DATA_BLOCKED_NO_CANDIDATE_DATA` and
  `DOMAIN_POLICY_GATE_FAIL_CONTINUE_V18_QUEUE`; its best Haze4K internal
  domain-conditioned policy was positive on mean PSNR but unsafe on tail gates.
- Q3/Q4 finished negative after repaired evidence closeout: all `10/10` seeds
  are `SEED_EVIDENCE_COMPLETE`, all `10/10` selected checkpoint labels are
  `Best`, and all `10/10` seed decisions are
  `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`.

The repaired 10-seed aggregate in
`v18_multiseed_aggregate/v18_multiseed_aggregate_summary.json` records:

```text
regular mean PSNR delta mean -0.05399 dB, CI95 half-width 0.00794 dB
regular easy-top25 mean -0.04441 dB
regular mean SSIM delta -0.0000961
regular strong-regression ratio mean 0.508
regular worst-count mean 85.2
hard mean PSNR delta mean -0.09085 dB, CI95 half-width 0.01434 dB
hard hard-bottom25 mean -0.12387 dB
hard mean SSIM delta -0.0001369
hard strong-regression ratio mean 0.532
hard worst-count mean 85.7
```

Only `n_ge_5` passes in the written multi-seed screen. Every quality and safety
gate is `false`.

## Engineering Note

The first completed Q3 seed evals failed because
`eval_haze4k_checkpoint_compare.py` only added the current working directory to
`sys.path`, so running it from the repository root could not import
`Dehazing/ITS/data`. The tool was patched to resolve the repo root and
`Dehazing/ITS` from `__file__`, and
`repair_v18_missing_eval_and_aggregate.sh` reran only the missing `3407/2026`
evals plus the aggregate. Both repaired seeds remained negative after repair,
so the import-path bug was an engineering failure only, not the scientific
reason for route failure.

## Remote Access Note

The `2026-06-06 05:09 +08:00` remote-access blocker is preserved in
`remote_access_blocker_20260606_0509.md`. After the user confirmed the updated
endpoint `connect.bjb1.seetacloud.com:16124`, the queue resumed without
rerunning completed seeds. Final remote closeout at `2026-06-06T14:28:47+08:00`
confirmed no active tmux sessions, no related train/eval/repair processes, and
idle GPU.

## Key Files

| Path | Meaning |
| --- | --- |
| `status.txt` | Durable queue plus repair heartbeat log, including the final `repair_done` closeout. |
| `v18_router_policy_fixed/v18_router_best_policy_summary.json` | Authoritative corrected Q1 router summary. |
| `v18_domain_data_preflight/v18_domain_data_preflight_summary.json` | Q2 split/domain preflight summary. |
| `v18_domain_adaptation_q5/v18_domain_adaptation_summary.json` | Q5 domain/data coverage summary and real-domain blocker inventory. |
| `seed_*/v18_seed*_multimetric_checkpoint_selection.json` | Per-seed checkpoint selection and negative-decision evidence. |
| `v18_multiseed_aggregate/v18_multiseed_aggregate_summary.json` | Final 10-seed aggregate and gate result. |
| `v18_progress/v18_progress_summary.json` | Read-only queue/repair completion summary refreshed after final sync. |
| `v18_eval_repair/repaired_seeds.txt` | Explicit list of repaired engineering-failure seeds (`3407`, `2026`). |

## Directory Contents

| Path | Meaning |
| --- | --- |
| `v18_router_policy/` | Original Q1 table run retained as invalid diagnostic evidence because derived `v18_alpha_*` labels leaked into the feature set. |
| `v18_router_policy_fixed/` | Corrected Q1 router rerun and the authoritative router decision artifacts. |
| `v18_domain_data_preflight/` | Q2 Haze4K train-derived split metadata and image-stat audit. |
| `v18_domain_adaptation_q5/` | Q5 real-domain data inventory plus Haze4K internal domain-conditioned policy diagnostic. |
| `seed_<seed>/` | Per-seed train logs, eval compare JSON/CSV, and multi-metric checkpoint-selection evidence. |
| `v18_eval_repair/` | Post-queue repair watcher log and repaired seed list. |
| `v18_multiseed_aggregate/` | Final 10-seed aggregate metrics and summary. |
| `v18_progress/` | Read-only queue progress/closeout summary for machine-readable monitoring. |
