# HAZE4K SafeRHFD-v2 Stage-Scale Sweep 20260602

Status: complete; strict robustness gate failed for all 11 candidates.

Pointers:

- Route card:
  `experience_docx/experiment_cards/2026-06-02-haze4k-saferhfd-v2-stage-scale.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/pfd_rhfd_family_summary.md`

## Purpose

This sweep tests independent RHFD2/RHFD1 final-conv scales for the existing
B1-Surgery candidate. It performs no training and evaluates each generated
checkpoint against A0 on the Haze4K test split.

## Expected Outputs

- `scout_eval_compare_saferhfd_v2_*.json`
- `scout_eval_per_image_saferhfd_v2_*.csv`
- `scout_eval_bucket_analysis_saferhfd_v2_*.json`
- `stage_scale_summary.json`
- `stage_scale_summary.csv`
- `stage_scale_stdout.log`
- `status.txt`

## Result

No candidate passed the strict gate in `stage_scale_summary.json`.

The best failed diagnostic candidate was `RHFD2=0.50, RHFD1=0.70`:

- mean PSNR delta `+0.00779 dB`;
- hard bottom-25% delta `+0.02228 dB`;
- easy top-25% delta `+0.00693 dB`;
- severe regressions `1`, so the severe-regression gate failed;
- hard median delta `-0.00136 dB`;
- hard positive ratio `0.44`;
- hard delta excluding top-1 hard gain `-0.00164 dB`.

The top-1 gain remained `508_0.72_1.04.png` for every non-isolated positive
candidate. Removing that top hard gain left the hard bucket negative for all
candidate scales, so the stage-wise calibration did not remove the
single-sample dependency risk.

Decision: `FAIL_STRICT_ROBUSTNESS_GATE`. Do not promote this route or start
B2/B3 from this evidence.

## Artifact Boundary

Commit text-only evidence only. Do not commit generated checkpoints, visual
panels, datasets, or raw image outputs.
