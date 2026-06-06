# ConvIR-Dehaze-v1.8-ExecutionQueue

Date: 2026-06-06

Status: completed on `dehaze1` with post-queue eval repair finished on the
updated `connect.bjb1.seetacloud.com:16124` endpoint; final closeout confirmed
no active tmux sessions or related train/eval processes.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: DPGA/UDP expert-bank follow-up plus data/domain preflight.
- Primary objective: execute the post-report experiment plan without stopping
  the queue after an independent route failure.
- Execution environment: cloud server `dehaze1`; local WSL checkout is editing
  and compile/syntax-only.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/`.
- Branch or isolated workspace:
  `codex/haze4k-v1-7-risk-controlled-expert-mix` local source copied to
  `/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue`.
- Locked Haze4K test policy: blocked. v1.8 uses only train-derived
  `train_inner`, `val_regular`, and `val_hard` unless a later route card
  explicitly authorizes a one-shot locked confirmation.

## Report Corrections Applied

The uploaded diagnosis is consistent with the repository evidence, but the
execution plan is tightened before launch:

- Treat FAM/HardFreq/HazePrior/APDR as closed unless new evidence appears; do
  not spend runtime there in v1.8.
- Keep the two active scientific lines: A0+UDP expert routing and DPGA/UDP
  depth-prior fusion capacity.
- Do not rely on single-PSNR `Best.pkl` selection. Evaluate `model_5`,
  `model_10`, `model_15`, `model_20`, `Best`, and `Final`, then select a
  decision checkpoint with multi-metric regular+hard evidence.
- Do not stop the full queue when one independent experiment fails. Each item
  writes its own state and the queue continues.
- Preserve intermediate outputs before long training: router policy grids,
  feature AUC, per-image policy tables, domain/data preflight tables, per-seed
  checkpoint-selection JSON/CSV, and multi-seed aggregates.
- The report also asks for a data/domain-adaptation line. The cloud data
  inventory currently exposes Haze4K only, with no visible `real_haze` target
  domain. v1.8 therefore adds an auditable Q5: record the real-domain data
  blocker and run a Haze4K internal domain-conditioned A0/UDP alpha-policy
  diagnostic from existing train-derived tables. This does not replace a later
  real-domain adaptation run once data exists.

## Baseline Contract

- Baseline expert `E0`: official ConvIR-B A0 checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Hard expert `E1`: official UDPNet ConvIR checkpoint
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Data root:
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- Split JSON:
  `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- Baseline comparison: every trained candidate is evaluated against A0 on
  `val_regular` and `val_hard`.

## Experiment Queue

| ID | Experiment | Runtime type | Required outputs | Continue rule |
| --- | --- | --- | --- | --- |
| V18-Q1 | A0/UDP router policy grid from the existing v1.7 3000-row feature table | table-only | `v18_router_policy_grid.csv`, feature AUC, OOF and heldout per-image policy CSVs, summary JSON | Always continue after recording pass/fail. |
| V18-Q2 | Haze4K data/domain preflight for train-derived splits | metadata/light image audit | per-image domain table and summary JSON | Always continue after recording pass/fail. |
| V18-Q3 | BiDPFM1 `fusion_neighbor` partial-unfreeze stop20, 10 seeds | training + eval | per-seed train logs, regular/hard compare JSON/CSV for candidate checkpoints, multi-metric checkpoint selection | Continue through all seeds even when earlier seeds fail or gate fail. |
| V18-Q4 | Multi-seed aggregate and decision | table-only | seed aggregate CSV/JSON with CI-style summary | Always write final queue state. |
| V18-Q5 | Data/domain adaptation coverage | data inventory + table-only domain policy | real-domain data inventory, Haze4K internal domain group summary, OOF/heldout per-image domain-policy CSVs, summary JSON | Run independently from Q3 when possible; missing real-domain data is a recorded blocker, not a stop condition. |

## DPGA/UDP Partial-Unfreeze Contract

- Starting point: v1.4B BiDPFM1 mechanism, because adapter-only failed but
  zero-init and component diagnostics were already validated.
- Primary variable: `--dpga_train_scope fusion_neighbor`.
- Active adapters: `dpfm1`.
- Fusion mode: `udp_bi`.
- Neighbor unfreeze scope: existing code scope `FAM1`, `FAM2`, `SCM1`, `SCM2`,
  `Convs.0`, `Convs.1` plus DPGA parameters.
- Training budget: stop20 per seed, no locked test.
- Seeds: `3407 2026 929 123 777 1701 2222 3141 4242 5151`.
- Checkpoint evidence: evaluate `model_5`, `model_10`, `model_15`, `model_20`,
  `Best`, and `Final` on both internal splits when present.

## Gates

### Router Table Gate

Pass only if the selected table-only router satisfies both:

```text
OOF mean_delta >= +0.20 dB
OOF hard_bottom25_delta >= +0.35 dB
OOF easy_top25_delta >= 0
OOF SSIM_delta >= 0
OOF worst_ratio <= 0.04
OOF strong_ratio <= 0.08
heldout mean_delta >= +0.15 dB
heldout hard_bottom25_delta >= +0.25 dB
heldout easy_top25_delta >= -0.02 dB
heldout worst_ratio <= 0.05
heldout strong_ratio <= 0.10
```

This gate does not authorize locked test; it only decides whether a later
router route is worth a separate locked-test card.

### Partial-Unfreeze Multi-Seed Screen

The 10-seed queue must run regardless of the 5-seed screen. A useful screen
requires:

```text
n >= 5
regular mean PSNR delta CI lower >= +0.05 dB
hard-bottom25 PSNR delta CI lower >= +0.10 dB
easy-top25 mean >= -0.02 dB
regular/hard SSIM means >= 0
regular strong regression ratio mean <= 0.16
hard worst-count mean <= 8
```

If the screen fails, continue the remaining scheduled seeds and mark the route
as negative or inconclusive after the 10-seed aggregate.

## Cloud Run Contract

- Remote workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue`.
- Python: `/root/miniconda3/envs/convir-cu128/bin/python`.
- Runtime launcher:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/run_v18_execution_queue.sh`.
- Monitor script:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/monitor_v18_execution_queue.sh`.
- Domain-adaptation Q5 launcher:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/run_v18_domain_adaptation_q5.sh`.
- Post-queue eval repair launcher:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/repair_v18_missing_eval_and_aggregate.sh`.
- tmux session: `v18_execution_queue`.
- Optional parallel Q5 tmux session: `v18_domain_adaptation_q5`.
- Optional post-queue repair tmux session: `v18_eval_repair`.
- Status file:
  `experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606/status.txt`.

## Decision

Decision label: `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`.

All declared v1.8 items were completed on `dehaze1` without early stop. Q1, Q2,
and Q5 wrote independent results; Q3/Q4 finished after post-queue repair
regenerated the missing `3407/2026` eval evidence caused by the early import-path
bug. Locked Haze4K test remained untouched throughout.

## Final Cloud Result

Cloud execution started on `dehaze1` in tmux session `v18_execution_queue` at
`2026-06-06T01:50:49+08:00`.

Q1 initial router analysis completed, but its feature-selection code
accidentally allowed derived `v18_alpha_*` labels into the candidate feature
set. That first directory is retained as invalid diagnostic evidence only.
The corrected `v18_router_policy_fixed/` rerun excludes derived labels and is
the authoritative Q1 result.

Corrected Q1 decision: `ROUTER_GATE_FAIL_CONTINUE_OTHER_V18_EXPERIMENTS`.
Best corrected table policy used features
`udp_a0_luma_shift_mean,input_edge_mean,bright_channel_mean`, alpha `1.0`,
and nominal coverage target `0.2`. It failed OOF and heldout gates:

```text
OOF mean_delta +0.0557 dB
OOF hard_bottom25_delta +0.3324 dB
OOF easy_top25_delta -0.1049 dB
OOF worst_ratio 0.0783
Heldout mean_delta +0.2140 dB
Heldout hard_bottom25_delta +0.4173 dB
Heldout easy_top25_delta -0.0601 dB
Heldout worst_ratio 0.0717
```

Q2 data/domain preflight completed with `3000` train-derived rows and
`missing_count=0`. It confirmed the expected split profile: `val_hard` has
lower A0 PSNR, brighter/lower-saturation/sky-proxy-heavy images, and stronger
UDP hard-expert movement (`udp_delta_psnr_mean +0.4260 dB`) while
`val_regular` remains unsafe for UDP-only (`udp_delta_psnr_mean -0.3020 dB`).

Q5 completed at `2026-06-06T03:01:20+08:00`. It recorded
`REAL_DOMAIN_DATA_BLOCKED_NO_CANDIDATE_DATA`: no visible real-haze candidate
dataset existed under the current cloud workspace roots; only Haze4K was
present under `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`. The best
Haze4K internal domain-conditioned policy was `sky_tertiles` with
`min_group_size=40`, but it failed gates despite positive means:

```text
OOF mean_delta +0.4192 dB
OOF hard_bottom25_delta +0.2620 dB
OOF easy_top25_delta +0.5224 dB
OOF worst_ratio 0.1363
OOF strong_ratio 0.1208
Heldout mean_delta +0.3706 dB
Heldout hard_bottom25_delta +0.3613 dB
Heldout easy_top25_delta +0.5519 dB
Heldout worst_ratio 0.1733
Heldout strong_ratio 0.1268
```

Decision: `DOMAIN_POLICY_GATE_FAIL_CONTINUE_V18_QUEUE`. The result is useful
domain evidence and a real-domain data blocker, not a deployable policy.

Q3 engineering note: seed `3407` and `2026` reached training completion, but
their first checkpoint eval commands failed with `ModuleNotFoundError: data`.
This was an eval-tool import-path bug, not a model result. The eval tool was
patched to resolve `Dehazing/ITS` from `__file__`, and a post-queue repair
script was added so missing per-seed eval JSON and the aggregate are regenerated
after the active training queue finishes. `v18_eval_repair` started at
`2026-06-06T03:08:58+08:00`, resumed after the endpoint recovery, and completed
at `2026-06-06T13:38:33+08:00`.

Monitoring/sync note: at `2026-06-06 05:09 +08:00`, cloud monitoring and
evidence sync became blocked by remote-access infrastructure, not by a model
failure. The current WSL `dehaze1` alias resolved to
`connect.bjb2.seetacloud.com:40921` and refused the connection, while the
AGENTS-declared endpoint `connect.bjb1.seetacloud.com:49601` presented a new
ED25519 host key and then rejected all available local keys under a temporary
pinned-host-key probe. The local evidence file
`remote_access_blocker_20260606_0509.md` records the details. Do not fall back
to local runtime commands; resume cloud monitoring and sync only after
`dehaze1` access is restored.

Recovery note: at `2026-06-06T10:28:51+08:00`, access was restored on the
user-confirmed `dehaze1` endpoint `root@connect.bjb1.seetacloud.com:16124`.
`AGENTS.md` and local `~/.ssh/config` were updated to match, the patched
v1.8 monitor/progress tooling was re-synced to
`/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue`, and the queue was
restarted in tmux without rerunning completed seeds. `seed_1701` resumed from
`Training-Results/model.pkl` at saved epoch `6`, `v18_eval_repair` resumed its
wait loop, and GPU monitoring at `2026-06-06T10:30:22+08:00` showed active
training (`11000 MiB`, `99%` utilization). The updated
`v18_progress/v18_progress_seeds.csv` now separates the early `3407` and `2026`
eval import/path failures as engineering states
`EVAL_FAILED_ENGINEERING_REPAIR_PENDING`, not scientific gate results.

Q3/Q4 final result is negative after the full 10-seed queue plus repair:

- all `10/10` seeds are `SEED_EVIDENCE_COMPLETE`;
- all `10/10` selected checkpoint labels are `Best`;
- all `10/10` seed decisions are
  `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`;
- repaired seeds `3407` and `2026` stayed negative after eval recovery, so the
  original `ModuleNotFoundError: data` was an engineering failure only, not the
  cause of the final scientific result.

The repaired multi-seed aggregate in
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

Gate result: only `n_ge_5=true`; every quality/safety gate is `false`.

Final remote closeout at `2026-06-06T14:28:47+08:00` confirmed:

```text
v18_execution_queue=NOT_ACTIVE
v18_eval_repair=NOT_ACTIVE
v18_domain_adaptation_q5=NOT_ACTIVE
GPU utilization 0%, memory 1 / 24564 MiB
no related train/eval/repair process
```

This route satisfies the report requirement to continue every declared
experiment even after independent failures. The final scientific conclusion is
negative only after Q1-Q5 completion and repaired Q3/Q4 evidence closeout, not
because the queue stopped early.
