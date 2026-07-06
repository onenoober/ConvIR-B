# Haze4K v2.40 Teacher Residual Alignment Atlas Evidence

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-40-teacher-residual-alignment-atlas.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-40-teacher-residual-alignment-atlas`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v240_teacher_residual_alignment_atlas.sh`
- `v240_teacher_residual_alignment_atlas_summary.json`
- `v240_closeout.json`

Cloud-only runtime/raw evidence:

- `v240_teacher_residual_alignment_atlas_per_image.csv`
- `v240_alignment_predictability_per_fold.csv`

## Metric Contract

P0 is a diagnostic-only atlas on the same `600` train-derived full-image
same-context images used by v2.37/v2.38/v2.39. It reads existing A0, GT,
WDMamba, and ConvIR-L tensor caches and existing raw per-image CSVs. It does
not run training, bridge/generator work, canary80, or locked test.

Diagnostic completion requires matched cache coverage for all `600` images,
summary JSON, closeout JSON, and `locked_test_touched=false`.

## Result

Pending P0 completion.
