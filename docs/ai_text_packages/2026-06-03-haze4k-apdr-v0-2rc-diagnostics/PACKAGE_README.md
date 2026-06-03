# Haze4K APDR-v0.2RC Diagnostics Sync Package

Date: 2026-06-03

Status: GitHub-readable sync package for APDR-v0.2RC follow-up diagnostics.

## Purpose

This package records the code and text-only experiment entry points prepared
after the APDR-v0.2RC conservative-budget replay. The replay showed that the
single global budget can close easy and strong-reference images, but that the
same score cannot remain both a calibrated hard probability and a conservative
action budget.

The next synchronized work is diagnostic, not promotion:

- run oracle-on-fail analysis for the selected v0.2RC action mask;
- preserve intermediate CSV/JSON tables that explain the BCE failure;
- only consider residual stop20 scouts after selector/oracle gates justify it;
- keep checkpoints and training outputs on AutoDL, outside GitHub.

## Source Branch

- Branch: `codex/haze4k-apdr-v0-2rc-oracle-diagnostic`
- Local isolated worktree:
  `/home/ubuntu/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic`
- Cloud execution target: AutoDL `autodl-dehaze3`
- Default cloud root:
  `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic`

## Contents

| File | Use |
| --- | --- |
| `SYNC_MANIFEST.md` | Exact list of changed code, scripts, and artifact boundaries. |
| `DIAGNOSTIC_PLAN.md` | Short gate-based plan for oracle, BCE, leakage, coverage, and residual diagnostics. |

## Boundary

Included in GitHub: Python source, shell launchers, Markdown summaries, and
future text evidence such as `.json`, `.csv`, `.log`, `.txt`, and `.out`.

Excluded from GitHub: checkpoints, model weights, image outputs, datasets,
NumPy arrays, raw inference artifacts, and generated selector checkpoints such
as `selector_checkpoint_*.pkl`.
