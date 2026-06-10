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
This is a blocking rule: if the current branch is
`codex/haze4k-official-arch-anchor` and the requested work changes model
behavior, runtime entrypoints, data policy, losses, selectors, gates, adapters,
or evaluation tooling, stop and create a route branch first.

Allowed changes on the anchor branch are limited to:

- command reliability or documentation fixes that do not change model behavior;
- text-evidence sync for anchor validation, excluding checkpoints, datasets,
  images, arrays, archives, and raw inference outputs;
- non-runtime compatibility notes or scripts that do not touch protected
  model/runtime paths. Any compatibility change to protected paths requires
  explicit user approval and a dedicated maintenance branch or pull request.

Not allowed on the anchor branch:

- adding new trainable model modules;
- changing ConvIR-B forward behavior;
- enabling FAM/APDR/DPGA/PFD/UDP variants;
- tuning thresholds, losses, gates, adapters, or data selection policies;
- running locked-test selection or checkpoint selection from this branch.

## Mandatory Clean Route Procedure

Every future ConvIR-B/Haze4K model modification must satisfy all of these
requirements before the first code edit or cloud run:

1. Start from `github/codex/haze4k-official-arch-anchor` in a new
   `codex/<route>` branch or isolated worktree.
2. Confirm the starting anchor commit and record it in the route card.
3. Leave this anchor branch unchanged except for documentation, command
   reliability, or text-evidence maintenance.
4. Define whether `haze4k-base.pkl` is loaded strictly or partially. If partial
   loading is needed, record the missing/new keys and initialization rule.
5. Declare the Haze4K data path, checkpoint path/hash, locked-test policy, cloud
   workspace, output root, command script, status file, and evidence root.
6. Run preflight and training/evaluation only on `dehaze1`; local WSL remains
   edit and syntax/static-check only.
7. Sync text evidence back into `experience_docx/` and GitHub before closing the
   run.

A route that skips any item above is invalid for comparison against the official
anchor and must be relabeled as setup work or rerun from a compliant branch.

## Enforcement References

- Agent-level rule: `AGENTS.md`, section `Official Architecture Anchor And
  Clean Route Gate`.
- Governance rule: `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, section `Official
  Anchor Clean Route Rule`.
- Start gate: `MODEL_EXPERIMENT_START_CHECKLIST.md`, section `0. Official
  Anchor Compliance Gate`.
- GitHub Actions guard: `.github/workflows/official-anchor-guard.yml` blocks
  protected runtime/model path edits that target this branch. Configure GitHub
  branch protection to require this check for hard server-side enforcement.

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
