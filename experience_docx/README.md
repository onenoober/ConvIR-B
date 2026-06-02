# Generic Experiment Protocol

Date: 2026-05-31

Status: model experiment governance package.

## Purpose

Use this package when starting or reorganizing a model experiment.

It defines the reusable logic and constraints for model experimentation:

- baseline-first setup;
- one authoritative place per fact;
- repository hygiene and isolated work;
- reference entrypoint stability;
- route cards before expensive work;
- one primary variable per first trial;
- cheap preflight before long runs;
- fair matched-budget comparisons;
- sample-size discipline;
- early stop and promotion gates;
- mechanism-specific metrics;
- preservation and regression checks;
- deployability and leakage controls;
- text-only evidence package policy;
- artifact retention and cleanup boundaries;
- explicit decision labels.

For this repository, start with `CONVIR_B_EXECUTION_GUIDE.md`. The generic
files define the rules; the ConvIR-B guide binds those rules to the official
pretrained checkpoints, repository commands, baseline metrics, and fixed-budget
promotion gates.

For consolidated Haze4K route outcomes and GitHub-readable evidence locations,
start with `EXPERIMENT_INDEX.md`.

For future `codex/*` experiment branches, follow
`BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` so evidence returns to `main` without
merging diagnostic experiment code.

## Files

| File | Use |
| --- | --- |
| `EXPERIMENT_GOVERNANCE_PROTOCOL.md` | General rules and constraints for running model experiments. |
| `ROUTE_DESIGN_FRAMEWORK.md` | Route families and the questions each route must answer. |
| `EXPERIMENT_CARD_TEMPLATE.md` | Blank route/experiment card for a new candidate. |
| `MODEL_EXPERIMENT_START_CHECKLIST.md` | Checklist for starting and governing a model experiment. |
| `CONVIR_B_EXECUTION_GUIDE.md` | Project-specific baseline-first and fixed-budget guide for ConvIR-B. |
| `EXPERIMENT_INDEX.md` | Consolidated Haze4K route outcomes, retained branches, and text evidence roots. |
| `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | Required evidence-only sync workflow for future GitHub experiment branches. |

## Use Sequence

1. read `EXPERIMENT_INDEX.md` if you need the current Haze4K route state;
2. read `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` before syncing any route branch
   back to GitHub `main`;
3. read `CONVIR_B_EXECUTION_GUIDE.md` for the current repository;
4. download the official pretrained checkpoint for each target task;
5. run evaluation first and record the local ConvIR-B baseline;
6. explain any reproduction gap against the root `README.md` result table;
7. copy `EXPERIMENT_CARD_TEMPLATE.md` only after the baseline is trustworthy;
8. fill the card with fixed-budget gates, mechanism metrics, and stop rules;
9. treat every route as unproven until it passes its own written gates.

## Current ConvIR-B Route State

The baseline package now exists and the current Haze4K route state is tracked
in `EXPERIMENT_INDEX.md`. Use that index as the first stop before starting or
judging another route.

As of 2026-06-02:

- PFD mainline B1 improved hard cases but failed the preservation gate.
- B1 surgery `scale=0.70` is the best preservation-first diagnostic candidate,
  but it is a no-training checkpoint surgery result, not a trained replacement.
- SafeRHFD-v2 pfd-only training completed and failed its automatic gate.
- No trained Haze4K replacement route is promotion-ready.
