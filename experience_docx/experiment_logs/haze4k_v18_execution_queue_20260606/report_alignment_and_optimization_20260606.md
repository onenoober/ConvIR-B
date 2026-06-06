# v1.8 Report Alignment And Optimization Note

Date: 2026-06-06

Status: interim note after re-reading the uploaded DOCX
`C:/Users/Administrator/Downloads/ConvIR 去雾实验根因诊断与路线评估报告.docx`
against the live v1.8 queue evidence.

## Main Alignment

The report's highest-priority conclusions are already reflected in the current
v1.8 execution design:

- closed low-value families stay closed: FAM, HardFreq, HazePrior, and APDR are
  not reopened inside v1.8;
- route selection is limited to the two scientifically active lines:
  DPGA/depth-prior capacity and A0/UDP expert utilization;
- locked Haze4K test remains blocked;
- multi-metric checkpoint selection replaces reliance on training-time
  `Best.pkl`;
- the queue continues after independent gate failures instead of stopping at
  the first negative result;
- intermediate evidence is preserved during runtime (`v18_progress`, router
  tables, domain/data preflight, per-seed compare JSON/CSV, per-seed selection
  JSON/CSV);
- Q3 uses the report-requested multi-seed design rather than a single-seed
  conclusion path;
- Q5 records the real-domain data blocker explicitly instead of silently
  dropping the domain-adaptation line.

## Corrections Already Enforced

The report's two strongest experimental-design criticisms are now handled
inside the repository workflow:

1. Test-based model selection:
   early routes that selected `Best.pkl` from `test/` are treated as legacy
   evidence only. v1.8 uses train-derived `train_inner`, `val_regular`, and
   `val_hard` plus a separate locked-test block.
2. Single-metric checkpoint gate:
   v1.8 evaluates `model_5`, `model_10`, `model_15`, `model_20`, `Best`, and
   `Final` on both internal splits and writes a separate multimetric selection
   JSON/CSV instead of trusting average PSNR alone.

## Live Queue Evidence That Changes The Report Context

The report recommended that all later decisions distinguish engineering failures
from scientific failures. The live queue now has concrete evidence for that:

- `seed_3407` and `seed_2026` are not negative model results; their first evals
  failed from an import/path bug and are tracked as
  `EVAL_FAILED_ENGINEERING_REPAIR_PENDING`;
- resumed `seed_1701` completed train, full eval, and checkpoint selection
  after remote access recovery, and still landed at
  `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`;
- the queue then advanced into fresh `seed_2222` training without rerunning any
  completed seeds;
- `v18_eval_repair` is intentionally waiting for the main queue to finish before
  repairing only the missing eval evidence and rebuilding the multiseed
  aggregate.

## Optimization Deferred On Purpose

The report also suggests several additional optimizations that are reasonable in
principle but should **not** be injected into the active v1.8 queue mid-run,
because they would change the route contract after part of the seeds have
already finished:

- changing warmup/scheduler stepping semantics;
- introducing EMA/SWA;
- replacing the current router family with a stronger token/alpha-map router;
- launching Bayesian hyperparameter search over adapter/loss/router settings;
- changing unfreeze scope beyond the predeclared `fusion_neighbor` contract.

These are candidates for a later route only, not silent mid-queue edits.

## Engineering Observation For The Next Route

`seed_2222` training emitted the PyTorch warning that
`lr_scheduler.step()` was called before `optimizer.step()`. A remote static
check confirmed that the current code intentionally pre-steps the warmup
scheduler before epoch 1, which changes the initial LR from `0.0` to
`3.33333e-05`. This is a legitimate future cleanup item, but changing it now
would make later v1.8 seeds non-comparable with earlier ones, so it is deferred
to a new route or an explicitly documented protocol revision.

## Bottom Line

The uploaded report is directionally correct, and the active v1.8 queue already
implements most of its highest-value corrections. The right action now is not
to mutate the route again mid-queue, but to keep the queue running, keep
engineering failures isolated from scientific results, and finish the declared
10-seed plus repair/aggregate workflow before deciding the next route.
