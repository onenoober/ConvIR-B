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

Status: `COMPLETED_DIAGNOSTIC`.

Decision:
`V240_COMPLETE_SELECTOR_ALPHA_BLOCKED_V241_STAGE0_DESIGN_OPEN`.

P0 completed on `600/600` train-derived full-image same-context images. WDMamba
teacher residuals were mostly GT-aligned (`anti_aligned_rate_all=0.0033`) with
positive alpha-safe margin in most images (`alpha_safe_upper p05=0.5031`), but
rare easy/strong-reference tail cases remain enough to keep no-selector alpha
continuation blocked. ConvIR-L showed much higher teacher-specific
anti-alignment (`0.1033` all, `0.1867` hard) and unsafe rate at useful alpha
(`0.1267`). WDMamba/ConvIR-L unsafe overlap was low (`Jaccard=0.0385` at the
useful alpha pair), arguing against a single shared teacher-independent unsafe
mode.

Runtime-visible feature predictability did not authorize selector work:
WDMamba anti-alignment/alpha-safe-tail labels were too rare and had recall
`0.0` at FPR0.05, while ConvIR-L anti-alignment reached AUROC/AUPRC
`0.7123/0.2176` but recall at FPR0.05 was only `0.1129`.

Manual review keeps all selector/alpha/bridge/generator/P5/canary80/locked-test
continuations blocked. It opens only a separate v2.41 Stage-0 design/preflight
route from `github/codex/haze4k-official-arch-anchor` for an A0-proximal,
GT-risk-controlled supervised residual architecture.
