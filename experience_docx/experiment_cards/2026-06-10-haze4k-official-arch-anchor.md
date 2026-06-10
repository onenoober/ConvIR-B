# Haze4K Official ConvIR-B Architecture Anchor

Date: 2026-06-10

Status: completed

## Scope

- Project: ConvIR-B Haze4K.
- Model family: official ConvIR-B architecture anchor.
- Dataset or task: Haze4K dehazing.
- Primary objective: create a clean, immutable GitHub branch for future routes
  to branch from, without carrying failed route implementations into the base.
- Main metric: checkpoint load integrity and runtime preflight pass.
- Secondary metrics: parameter count, output shapes, finite train-batch loss,
  CLI compatibility, locked-test untouched.
- Execution environment: `dehaze1` cloud runtime.
- Artifact root: `experience_docx/experiment_logs/haze4k_official_arch_anchor_20260610/`.
- Branch or isolated workspace: `codex/haze4k-official-arch-anchor`.
- Review package location: same evidence root.

## Baseline Contract

- Baseline implementation: official ConvIR-B architecture in `Dehazing/ITS/models/ConvIR.py`.
- Baseline checkpoint: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Checkpoint hash: `sha256:6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088` from the reproduced baseline record.
- Evaluation entrypoint: `Dehazing/ITS/main.py --mode test --version base --data Haze4K`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train --version base --data Haze4K`.
- Dataset and split: Haze4K under `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`; validation preflight uses train data only and does not touch locked test.
- Reproduced baseline result: PSNR `34.14 dB`, SSIM `0.98971`, average time `0.083973 s`, peak memory `1329 MiB` in `baseline_logs/haze4k_pretrained_20260531/README.md`.
- Known reproduction gap: none beyond official table rounding; train split count mismatch remains an audit item before formal training.
- Reference entrypoints that must remain stable: `--arch official_convir`, `--fam_mode original`, `--init_model`, `--learning_rate` with legacy `--leaning_rate` alias.
- Checkpoint/export/resume contract: official checkpoint loads strictly into the anchor; route branches must use explicit partial-load rules for new modules.
- Source audit: `experiment_logs/haze4k_official_arch_anchor_20260610/source_audit.txt`; `ConvIR.py` matches upstream architecture except for an `original`-only compatibility wrapper, and the existing `layers.py` dataset-dispatch cleanup is Haze4K-equivalent.

## Change

- Removed FAM/FAM2 experiment behavior from the anchor model path by restoring official ConvIR-B architecture semantics.
- Kept a compatibility `fam_mode='original'` argument while rejecting non-original modes.
- Added `--init_model` strict official checkpoint initialization to match successful A1/Haze4K records.
- Added `--learning_rate` and legacy `--leaning_rate` compatibility.
- Removed APDR/DPGA hard imports from the default Haze4K entrypoint; variant routes must live on route branches.
- Made Haze4K compare tooling lazy-load APDR/DPGA so official-anchor use is not blocked by missing experimental modules.

## Preflight Plan

| Check | Pass line | Result |
| --- | --- | --- |
| syntax/static | Python compile succeeds for changed Haze4K code | pass locally: `py_compile`, `bash -n`, `git diff --check` |
| checkpoint strict load | `haze4k-base.pkl` loads with no missing/unexpected keys | pass on `dehaze1` |
| official architecture cleanliness | no FAM modulator/APDR/DPGA keys in official model state dict | pass: forbidden keys `[]` |
| synthetic forward | three output scales are finite and shaped as expected | pass: `[1,3,64,64]`, `[1,3,128,128]`, `[1,3,256,256]` |
| Haze4K train-batch forward | finite loss on one train crop without touching test split | pass: multiscale L1 `0.009162915870547295` |
| CLI compatibility | `--learning_rate` and `--leaning_rate` are both accepted | pass |
| source audit | official architecture path has no experimental FAM/APDR/DPGA/PFD behavior | pass, see `source_audit.txt` |

## Gates

The anchor is valid: all preflight checks passed on `dehaze1`, and no locked-test
data was touched.

## Decision

`OFFICIAL_ANCHOR_PREFLIGHT_OK`. Use `codex/haze4k-official-arch-anchor` as the
immutable official ConvIR-B architecture anchor. Future architecture edits must
start from a new `codex/<route>` branch or isolated worktree.
