# Generic Experiment Protocol

Date: 2026-06-01

Status: model experiment governance package.

## Purpose

Use this package when starting or reorganizing a model experiment.

It defines the reusable logic and constraints for model experimentation:

- baseline-first setup;
- one authoritative place per fact;
- repository hygiene and isolated work;
- cloud-first execution boundaries;
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
- completion GitHub synchronization;
- artifact retention and cleanup boundaries;
- explicit decision labels.

For this repository, start with `CONVIR_B_EXECUTION_GUIDE.md`. The generic
files define the rules; the ConvIR-B guide binds those rules to the official
pretrained checkpoints, repository commands, baseline metrics, and fixed-budget
promotion gates.

## Files

| File | Use |
| --- | --- |
| `EXPERIMENT_GOVERNANCE_PROTOCOL.md` | General rules and constraints for running model experiments. |
| `ROUTE_DESIGN_FRAMEWORK.md` | Route families and the questions each route must answer. |
| `EXPERIMENT_CARD_TEMPLATE.md` | Blank route/experiment card for a new candidate. |
| `MODEL_EXPERIMENT_START_CHECKLIST.md` | Checklist for starting and governing a model experiment. |
| `CONVIR_B_EXECUTION_GUIDE.md` | Project-specific baseline-first and fixed-budget guide for ConvIR-B. |

## Use Sequence

1. read `CONVIR_B_EXECUTION_GUIDE.md` for the current repository;
2. download the official pretrained checkpoint for each target task;
3. run evaluation first on the documented cloud server and record the matched
   ConvIR-B baseline;
4. explain any reproduction gap against the root `README.md` result table;
5. copy `EXPERIMENT_CARD_TEMPLATE.md` only after the baseline is trustworthy;
6. fill the card with fixed-budget gates, mechanism metrics, and stop rules;
7. treat every route as unproven until it passes its own written gates;
8. after each completed experiment, push scoped code, docs, and curated text
   evidence to the GitHub experiment branch and verify the remote paths.

## Current ConvIR-B Priority

The first useful work is not model modification. The first useful work is a
cloud-recorded baseline package with official checkpoints, PSNR/SSIM,
per-sample deltas, latency, peak GPU memory, and saved output-quality notes.
Only then can a route claim a real positive return.
