# Official Architecture Anchor Policy

Date: 2026-06-10

Status: active policy for `codex/haze4k-official-arch-anchor`.

## Purpose

This branch is the clean ConvIR-B official architecture anchor for Haze4K work.
It preserves the official ConvIR-B model architecture while keeping the project
utilities that have already been validated for Haze4K data, pretrained
checkpoint loading, and A0-vs-candidate evidence generation.

## Immutable Anchor Rule

Do not make architecture experiments directly on this branch. Future model
changes must start from a new `codex/<route>` branch or isolated worktree.

Allowed changes on the anchor branch are limited to:

- command reliability or documentation fixes that do not change model behavior;
- compatibility fixes for already validated baseline workflows;
- evidence sync for anchor validation, excluding checkpoints, datasets, images,
  arrays, archives, and raw inference outputs.

Not allowed on the anchor branch:

- adding new trainable model modules;
- changing ConvIR-B forward behavior;
- enabling FAM/APDR/DPGA/PFD/UDP variants;
- tuning thresholds, losses, gates, adapters, or data selection policies;
- running locked-test selection or checkpoint selection from this branch.

## Baseline Contract

The anchor uses the official ConvIR-B architecture for `Dehazing/ITS` and the
validated Haze4K pretrained checkpoint contract recorded in:

- `experience_docx/baseline_logs/haze4k_pretrained_20260531/README.md`
- `experience_docx/experiment_logs/haze4k_stop20_noise_floor_20260601/README.md`

The default architecture is `--arch official_convir` with `--fam_mode original`.
The legacy alias `--arch convir` is accepted only for compatibility with older
scripts and maps to the same official ConvIR-B architecture.

Upstream source comparison is recorded in
`experiment_logs/haze4k_official_arch_anchor_20260610/source_audit.txt`. The
only `ConvIR.py` delta from upstream is the compatibility wrapper that rejects
non-original `fam_mode`; the existing `layers.py` dataset-dispatch cleanup is
Haze4K-equivalent to upstream and preserves the successful Haze4K records.
