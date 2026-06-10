# Generic Experiment Protocol

Date: 2026-06-10

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
- explicit decision labels;
- mandatory clean-route starts from the official architecture anchor.

For this repository, start with `CONVIR_B_EXECUTION_GUIDE.md`. The generic
files define the rules; the ConvIR-B guide binds those rules to the official
pretrained checkpoints, repository commands, baseline metrics, and fixed-budget
promotion gates.

For consolidated Haze4K route outcomes and GitHub-readable evidence locations,
start with `EXPERIMENT_INDEX.md`. For future ConvIR-B/Haze4K model changes, the
mandatory start point is `github/codex/haze4k-official-arch-anchor`; do not
modify that anchor directly.

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
| `EXPERIMENT_INDEX.md` | Consolidated Haze4K route outcomes, family verdicts, evidence-strength labels, retained branches, and text evidence roots. |
| `family_summaries/` | Family-level verdicts, evidence summaries, do-not-repeat notes, and reopen conditions. |
| `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | Required evidence-only sync workflow for future GitHub experiment branches. |
| `OFFICIAL_ARCH_ANCHOR_POLICY.md` | Mandatory clean-route and immutability rules for the official ConvIR-B/Haze4K anchor. |
| `CLOUD_PY310_ENVIRONMENT.md` | Current `dehaze1` py310/cu128 environment audit and future-server install guide. |

## Use Sequence

1. for any new ConvIR-B/Haze4K model route, start from
   `github/codex/haze4k-official-arch-anchor` in a new `codex/<route>` branch or
   isolated worktree;
2. read `OFFICIAL_ARCH_ANCHOR_POLICY.md` and satisfy its mandatory clean-route
   procedure before editing code;
3. read `EXPERIMENT_INDEX.md` if you need the current Haze4K route state,
   family verdict, evidence-strength level, or route reopening condition;
4. read the corresponding `family_summaries/` file before reopening a stopped
   route family or proposing a follow-up within an active family;
5. read `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` before syncing any route branch
   back to GitHub `main`;
6. read `CONVIR_B_EXECUTION_GUIDE.md` for the current repository;
7. download the official pretrained checkpoint for each target task;
8. run evaluation first and record the local ConvIR-B baseline;
9. explain any reproduction gap against the root `README.md` result table;
10. copy `EXPERIMENT_CARD_TEMPLATE.md` only after the baseline is trustworthy;
11. fill the card with fixed-budget gates, mechanism metrics, and stop rules;
12. treat every route as unproven until it passes its own written gates.

## Current ConvIR-B Priority

The first useful work is not model modification. The first useful work is a
local baseline package with official checkpoints, PSNR/SSIM, per-sample deltas,
latency, peak GPU memory, and saved output-quality notes. Only then can a route
claim a real positive return.
