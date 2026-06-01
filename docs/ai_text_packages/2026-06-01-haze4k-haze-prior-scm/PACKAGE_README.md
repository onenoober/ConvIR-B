# Haze4K Haze-Prior SCM Text Package

Date: 2026-06-01

This package contains the GitHub-readable text evidence for the ConvIR-B
Haze4K Haze-Prior SCM stop20 scout.

## Verdict

`haze_prior SCM + hard_aux` is not promoted to 80 epochs from this scout.

The exact configuration is mechanically valid and the prior branch becomes
active, but the Best checkpoint loses to the matched `original SCM + hard_aux`
control and violates preservation gates through strong/easy regressions.

## File Map

- `RESULT_SUMMARY.md`: compact decision summary and key numbers.
- `route_card.md`: full experiment card with protocol, gates, evidence, and
  decision.
- `evidence/preflight_synthetic_seed3407.json`: synthetic neutral-init and
  gradient preflight.
- `evidence/preflight_real_batch_seed3407.json`: real Haze4K batch preflight.
- `evidence/scout_eval_compare_seed3407_stop20_best.json`: Best checkpoint
  paired comparison.
- `evidence/scout_eval_bucket_analysis_seed3407_stop20_best.json`: Best
  checkpoint bucket/regression analysis.
- `evidence/scout_eval_compare_seed3407_stop20_last.json`: Last checkpoint
  paired comparison.
- `evidence/scout_eval_bucket_analysis_seed3407_stop20_last.json`: Last
  checkpoint bucket/regression analysis.
- `evidence/scout_eval_per_image_seed3407_stop20_best.csv`: Best checkpoint
  per-image metrics.
- `evidence/scout_eval_per_image_seed3407_stop20_last.csv`: Last checkpoint
  per-image metrics.
- `evidence/run_haze_prior_scm_hardaux_stop20.sh`: cloud run script.

## Exclusions

This package intentionally excludes checkpoints, datasets, images, tensors, and
generated image outputs. It is meant for text-only audit and AI-readable review.
