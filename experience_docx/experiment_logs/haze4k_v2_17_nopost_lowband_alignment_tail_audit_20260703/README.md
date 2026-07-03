# Haze4K v2.17 NoPost Lowband Alignment Tail Audit

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/2026-07-03-haze4k-v2-17-nopost-lowband-alignment-tail-audit.md`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Source branch:
`codex/haze4k-v2-17-nopost-lowband-alignment-tail-audit`

Locked Haze4K test: untouched and blocked.

## Plan

Adopt the review recommendation exactly:

> Close WLDB-A as a concrete form; do not close NoPost lowband.

Run no-training audits first:

- R1 WLDB-A postmortem;
- R2 capacity-ladder oracle;
- R3 tail-objective audit only if R2 keeps internal feature lowband open.

No training is launched in R1/R2/R3.

## Current Decision

Pending cloud execution.
