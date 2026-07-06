# Haze4K v2.40 Teacher Residual Alignment Atlas

Date: 2026-07-06

Branch:
`codex/haze4k-v2-40-teacher-residual-alignment-atlas`

Route identity: diagnostic/root-cause audit after v2.38 WDMamba micro-alpha,
v2.38B richer target-only separability, and v2.39 ConvIR-L same-family teacher
alpha all failed deployable gates.

## Primary Question

Are WDMamba and ConvIR-L teacher residual failures mainly caused by residual
anti-alignment, low A0 headroom/overshoot, teacher-specific artifacts, or
runtime-feature identifiability limits?

## Fact Sources

- GitHub `main` `experience_docx/EXPERIMENT_INDEX.md`.
- GitHub `main`
  `experience_docx/family_summaries/nopost_lowband_family_summary.md`.
- v2.37 cloud raw per-image WDMamba alpha sweep CSV.
- v2.38 cloud raw per-image WDMamba micro-alpha sweep CSV.
- v2.38B cloud raw runtime-visible feature manifest.
- v2.39 cloud raw per-image ConvIR-L alpha sweep CSV.
- Existing A0/GT/WDMamba/ConvIR-L full-image tensor caches on `convir-4090`.

## Not Allowed

- No WDMamba or ConvIR-L alpha continuation beyond reading existing CSVs.
- No selector threshold tuning, small classifier rescue, or bridge/generator
  launch.
- No P5 masked free-tensor projection.
- No canary80.
- No locked test.
- No direct WDMamba-on-256-crop route.
- No 256 crop-input/full-image-slice target.
- No S5-only BILFCF continuation.

## Metric Contract

P0 is a diagnostic-only atlas on the same `600` train-derived full-image
same-context images used by v2.37/v2.38/v2.39. For each image and teacher,
it computes:

- `E0 = A0 - GT`, `D = teacher - A0`;
- MSE headroom and full-teacher PSNR delta vs A0;
- `<E0, D>`, alignment cosine, `alpha_mse_opt`, and
  `alpha_safe_upper = -2 * <E0, D> / ||D||^2`;
- luma/chroma and low/high-frequency teacher-residual energy;
- overshoot risk `||D|| / ||E0||`;
- WDMamba/ConvIR-L unsafe overlap at low and useful alpha pairs;
- fold-out predictability of alignment labels from runtime-visible v2.38B
  features only, excluding GT/teacher-delta/score/leak columns.

The audit passes as a diagnostic if all `600` images have matched A0, GT,
WDMamba, and ConvIR-L tensors, the locked-test flag remains false, and summary
and closeout JSON files are written. It does not authorize v2.41 automatically;
v2.41 requires manual review of the atlas.

## Result

Status: `PLANNED`.

Decision pending P0 completion.
