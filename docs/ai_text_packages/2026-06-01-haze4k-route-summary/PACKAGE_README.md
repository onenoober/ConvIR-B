# Haze4K Route Summary Text Package

Date: 2026-06-01

Status: compact AI-readable route summary.

## Purpose

This package is the shortest stable entry point for reading the Haze4K route
sequence without checking every branch. It summarizes the route decisions and
points to the consolidated evidence now available on `main`.

## Contents

| File | Use |
| --- | --- |
| `ROUTE_DECISION_MATRIX.md` | Compact route-by-route result and decision table. |
| `EVIDENCE_MANIFEST.md` | Links from each route to its card and evidence root. |

## Source Of Truth

Detailed evidence remains in:

- `experience_docx/EXPERIMENT_INDEX.md`
- `experience_docx/experiment_cards/`
- `experience_docx/experiment_logs/`

This package intentionally avoids copying large per-image CSV tables. It keeps
the AI-facing reading path short while the full text evidence remains reachable
from the manifest.

## Boundary

Included: Markdown summaries and pointers to text evidence.

Excluded: checkpoints, model weights, image outputs, datasets, arrays, and raw
inference artifacts.
