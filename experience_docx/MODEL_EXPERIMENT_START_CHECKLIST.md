# Model Experiment Start Checklist

Date: 2026-06-10

Status: checklist for starting and governing a model experiment.

## 0. Universal Route Framing Gate

Before starting any new route, method, architecture change, teacher/distillation
plan, selector, loss, or runtime experiment, complete this route framing gate:

- Fact-source decision: name the authoritative evidence sources that will ground
  the task. Use GitHub `main` or the named GitHub branch for route memory and
  current decisions; use cloud runtime state for active/raw outputs; use local
  files only for editing, syntax checks, sync staging, or local-safety checks.
- Route-identity decision: label the work as a new route, continuation, rescue,
  ablation, reproducibility audit, or evidence sync. Write the consequences of
  that label before launch.
- Forbidden-flow list: record which tempting actions are not allowed for this
  route, such as expanding a failed canary, launching selector probes, increasing
  epochs/folds/samples, changing loss weights, touching locked test, or treating
  a diagnostic phase as a promotion gate.
- Resource-preflight list: identify required cloud branch/commit, workspace,
  explicit Python path, dataset/split, checkpoint, teacher/expert asset, output
  root, command script, status/log file, and tmux/session name before launch.
- Metric-contract list: define baseline, exact sample/crop/split pairing,
  metric direction, pass/fail thresholds, and what later phase each gate
  authorizes. For every formal decision gate, also record its gate type,
  analysis unit, threshold/margin source, `PASS`/`INCONCLUSIVE`/`FAIL` meaning,
  and allowed scientific claim according to the canonical Gate Policy in
  `EXPERIMENT_GOVERNANCE_PROTOCOL.md`. Do not interpret a result if
  base/before/after metrics were computed on different data views.
- Transport plan: choose the stable command/sync pattern up front from
  `COMMAND_RELIABILITY_QUICKSTART.md`. Prefer `tar`/`scp`/`rsync` or stable
  script bodies through WSL/SSH with explicit success markers; avoid ad hoc
  nested PowerShell/WSL/SSH quoting. Use `COMMAND_RELIABILITY_PROTOCOL.md` only
  for failed or unfamiliar command-boundary cases.
- Archive plan: decide which files belong on the route branch and which compact
  evidence belongs on GitHub `main`. Use a clean `github/main` worktree for
  evidence sync and stage explicit paths only.

If any item is unknown, do not launch a cloud runtime command yet. Fill the
route card or evidence README first, or classify the missing item as an
engineering blocker.

## 0B. Route Source And Anchor Compliance Gate

After the route identity is known, choose the correct starting source before
code edits or cloud runs:

- New model-structure or architecture routes in this repository must start from
  `github/codex/haze4k-official-arch-anchor`, not from a dirty worktree or an
  unrelated experimental branch.
- Continuations, rescues, ablations, audits, selectors, losses, adapters,
  data-policy changes, and fine-tuning routes start from the named parent
  GitHub branch/commit authorized by the current index, route card, or family
  summary. Do not reset them to the official anchor unless the framing gate says
  the work is a new model-structure route.
- Evidence sync and archival tasks start from a clean `github/main` worktree and
  restore only the explicitly allowed compact evidence paths from the route
  branch.
- Record the starting branch, starting commit, parent/source rationale, locked
  test policy, and any forbidden continuations in the route card or evidence
  README before launch.
- Keep `github/codex/haze4k-official-arch-anchor` unchanged except for
  documentation, command reliability, or text-evidence maintenance.
- Create or update the route card with the checkpoint path/hash, strict or
  partial load contract, new-module initialization rule when applicable,
  locked-test policy, cloud workspace, output root, command script, status file,
  and evidence root.
- Mark the route invalid for anchor or parent-route comparison if the required
  source gate was skipped or the starting source cannot be reconstructed.

## 1. Define Objective And Assumptions

- Name the new project objective in one sentence.
- List what is known.
- List what is unknown.
- Mark assumptions that still need evidence.
- Identify the baseline, target metric, and constraints that matter for the
  first decision.

## 2. Create A Documentation Map

Create or choose documents for:

- current state;
- experiment log;
- artifact manifest;
- runbook;
- workflow commands;
- analysis commands;
- dated experiment cards.

Write where each fact belongs before facts start accumulating.

## 3. Set Repository Boundaries

- Create a branch or isolated workspace from the source chosen in the Route
  Source And Anchor Compliance Gate. New model-structure routes use the official
  anchor; continuations, rescues, audits, and evidence syncs use their named
  parent or clean `github/main` source.
- Check version-control status before edits.
- Identify unrelated local changes and leave them untouched.
- Decide what can be committed and what must remain external.
- Keep reference entrypoints stable until an experiment card says otherwise.

## 4. Verify Data And Metrics

- Confirm dataset ownership, location, and allowed use.
- Confirm split definitions.
- Confirm pairing or label integrity.
- Confirm preprocessing and decoding.
- Confirm metric code and expected direction.
- Confirm sample counts and missing-file handling.
- Save a small text-only audit result.

## 5. Verify Runtime

- Confirm Python or runtime version.
- Confirm core dependencies.
- Confirm hardware availability.
- Confirm storage paths.
- Confirm checkpoint read/write.
- Confirm logging.
- Confirm resume behavior.
- Confirm evaluation can run from saved artifacts.
- Record durable dependency or environment facts in the runbook.

## 6. Establish Baseline

- Run reference evaluation if available.
- Run a minimal no-change smoke.
- Run the first fair baseline if needed.
- Record baseline config and artifacts.
- Define matched gate references for later routes.
- Do not modify the model before this is complete unless the first task is only
  repository bring-up.

### ConvIR-B Baseline Minimum

For this repository, baseline establishment means pretrained-checkpoint
evaluation before any training or model edits:

- download the official checkpoint from the root `README.md` links;
- record local checkpoint path, file size, and sha256 hash;
- run each target task's repository evaluation command;
- use `--version base` or the task folder's base-equivalent setting for
  ConvIR-B;
- record official-reference PSNR/SSIM and local PSNR/SSIM;
- record dataset split and verified sample count;
- record inference output directory, average latency, and peak GPU memory;
- save or export per-sample PSNR where possible;
- inspect saved outputs for obvious artifacts and list example filenames;
- label the baseline as accepted only after reproduction gaps are explained.

## 7. Define First Failure Inventory

Collect the smallest useful evidence for:

- average quality;
- subgroup quality;
- per-sample wins and losses;
- runtime and memory;
- training stability;
- obvious failure cases;
- data or label issues.

Convert observations into candidate failure modes. Do not jump directly to a
solution.

For ConvIR-B restoration tasks, include per-sample PSNR deltas, worst-10%
samples, strong-reference regressions, texture or edge errors when measurable,
frequency-domain loss/error when relevant, and runtime or memory outliers.

## 8. Choose The First Route

Use this filter:

- Does it target one failure mode?
- Can it be tested cheaply first?
- Does it change one primary variable?
- Does it have an early hard gate?
- Does it measure the claimed mechanism?
- Does it protect already-good cases?
- Does failure teach what not to try next?

If not, rewrite the route.

Write the first route as "fixed budget under ConvIR-B constraints": FLOPs <=
ConvIR-B +5%, latency <= matched runtime baseline +10%, peak memory <= matched
runtime baseline +10% and fitting the current GPU, with matched 5/20/80/full
epoch gates.

## 9. Launch Discipline

Before launch:

- freeze the config;
- record the command or job spec;
- record the expected artifact paths;
- record gate times or steps;
- record stop rules;
- record who can approve scope changes.

During launch:

- monitor without changing scope;
- record infrastructure failures separately from scientific failures;
- stop only at written gates or clear runtime failure;
- do not replace the run with a reduced version and call it equivalent.

For ConvIR-B, use successive halving by default: smoke, 5 epochs, 20 epochs,
80 epochs, then full budget. A candidate reaches the next stage only when the
written quality, mechanism, preservation, and cost gates all pass or when the
card says why the next stage is still informative.

## 10. After The Run

- Record final status.
- Record metrics and mechanism checks.
- Label the result precisely.
- Update artifact retention.
- Write what the result rules in or rules out.
- If evidence must be shared, create a compact text-only review package.
- Audit source/local/remote parity for any published evidence package.
- Create the next card only after the decision is clear.
