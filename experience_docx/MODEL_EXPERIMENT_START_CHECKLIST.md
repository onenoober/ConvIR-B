# Model Experiment Start Checklist

Date: 2026-05-31

Status: checklist for starting and governing a model experiment.

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

- Create a branch or isolated workspace for the task.
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
ConvIR-B +5%, latency <= local baseline +10%, peak memory <= local baseline
+10% and fitting the current GPU, with matched 5/20/80/full epoch gates.

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
