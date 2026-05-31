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

## Files

| File | Use |
| --- | --- |
| `EXPERIMENT_GOVERNANCE_PROTOCOL.md` | General rules and constraints for running model experiments. |
| `ROUTE_DESIGN_FRAMEWORK.md` | Route families and the questions each route must answer. |
| `EXPERIMENT_CARD_TEMPLATE.md` | Blank route/experiment card for a new candidate. |
| `MODEL_EXPERIMENT_START_CHECKLIST.md` | Checklist for starting and governing a model experiment. |

## Use Sequence

1. copy the directory as-is;
2. fill a new `EXPERIMENT_CARD_TEMPLATE.md` copy with project-specific facts;
3. define the new project's documentation map;
4. verify baseline, data, metrics, and runtime before modifying the model;
5. treat every route as unproven until it passes its own written gates.
