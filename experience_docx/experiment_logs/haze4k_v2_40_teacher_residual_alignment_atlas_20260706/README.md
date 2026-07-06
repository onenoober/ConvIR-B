# Haze4K v2.40 Teacher Residual Alignment Atlas Evidence

Status: `COMPLETED_DIAGNOSTIC`

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
- `v240_alignment_predictability_per_fold.csv`
- `v240_manual_review_decision.json`
- `v240_closeout.json`

Cloud-only runtime/raw evidence:

- `v240_teacher_residual_alignment_atlas_per_image.csv`

## Metric Contract

P0 is a diagnostic-only atlas on the same `600` train-derived full-image
same-context images used by v2.37/v2.38/v2.39. It reads existing A0, GT,
WDMamba, and ConvIR-L tensor caches and existing raw per-image CSVs. It does
not run training, bridge/generator work, canary80, or locked test.

Diagnostic completion requires matched cache coverage for all `600` images,
summary JSON, closeout JSON, and `locked_test_touched=false`.

## Result

Decision:
`V240_COMPLETE_SELECTOR_ALPHA_BLOCKED_V241_STAGE0_DESIGN_OPEN`.

P0 completed on `600/600` train-derived full-image same-context images. The
first cloud attempt skipped predictability because `sklearn` was unavailable;
that output was archived under
`engineering_incomplete_missing_sklearn_20260706_2311/` and the route was
rerun with the `numpy_weighted_ridge_fallback` predictability backend.

Key findings:

| Metric | WDMamba | ConvIR-L |
| --- | ---: | ---: |
| anti-aligned rate all / hard / easy / strong-ref | `0.0033 / 0.0000 / 0.0067 / 0.0067` | `0.1033 / 0.1867 / 0.0333 / 0.0333` |
| alpha-safe-upper p05 / median | `0.5031 / 1.5130` | `-0.8716 / 1.5017` |
| easy alpha-safe-upper p01 / p05 | `0.0138 / 0.2869` | `-0.5871 / 0.1424` |
| headroom MSE mean / p05 | `+2.2589e-4 / -6.3274e-5` | `+6.2325e-5 / -1.6750e-4` |
| overshoot-risk median / p95 | `1.0696 / 1.4853` | `0.6229 / 1.0284` |

Unsafe overlap is low, which argues against a single shared A0/headroom-only
failure mode:

- useful alpha pair (`WDMamba alpha=0.125`, `ConvIR-L alpha=0.25`): WDMamba
  unsafe `0.0083`, ConvIR-L unsafe `0.1267`, shared unsafe `0.0050`, Jaccard
  `0.0385`;
- low alpha pair (`WDMamba alpha=0.03125`, `ConvIR-L alpha=0.015625`): shared
  unsafe `0.0017`, Jaccard `0.0149`.

Runtime-visible feature predictability remains insufficient for reopening a
selector route:

- WDMamba anti-alignment base rate is only `0.0033`; OOF AUPRC `0.0030`,
  recall at FPR0.05 `0.0`;
- WDMamba alpha-safe-upper `<0.02` base rate `0.0050`; OOF AUPRC `0.0069`,
  recall at FPR0.05 `0.0`;
- ConvIR-L anti-alignment is more common (`0.1033`) and partially predictable
  (AUROC/AUPRC `0.7123/0.2176`), but recall at FPR0.05 is only `0.1129`, so it
  is not a deployable unsafe/no-op selector.

Interpretation:

- Do not reopen WDMamba/ConvIR-L alpha sweeps, richer target-only selector
  tuning, M0 bridge/generator, P5 projection, canary80, or locked test.
- WDMamba residuals are mostly GT-aligned with rare fragile tail cases, so the
  evidence supports only a new `v2.41` route-card/design and Stage-0 preflight
  for an A0-proximal, GT-risk-controlled supervised residual architecture from
  `github/codex/haze4k-official-arch-anchor`.
- v2.41 canary training is not authorized by this README alone; it requires its
  own route card, strict partial-load/zero-init evidence, cloud preflight, and
  written canary gate.
