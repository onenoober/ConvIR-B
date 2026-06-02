# Haze4K SafeRHFD-v2 Train

Date: 2026-06-02

Status: completed cloud stop20 run; automatic gate failed.

## Purpose

Train B1-v2 Selective SafeRHFD as a preservation-aware replacement for the
failed B1 RHFD full-model route.

This run is independent from the concurrent stage-scale surgery sweep:

- code/root: `ConvIR-B-saferhfd-v2-train`
- model name: `ConvIR-Haze4K-B1v2-SafeRHFD-pfdonly-stop20-seed3407-20260602`
- output log dir: `experience_docx/experiment_logs/haze4k_saferhfd_v2_train_20260602`
- comparison target: A0 official ConvIR-B Haze4K checkpoint

## Files

| File | Use |
| --- | --- |
| `preflight_saferhfd_v2.json` | SafeRHFD zero-init equivalence and trainable-parameter check. |
| `train_ConvIR-Haze4K-B1v2-SafeRHFD-pfdonly-stop20-seed3407-20260602.log` | Cloud stop20 training log. |
| `scout_eval_compare_saferhfd_v2_pfdonly_stop20_seed3407_vs_a0.json` | A0-vs-candidate evaluation summary. |
| `scout_eval_bucket_analysis_saferhfd_v2_pfdonly_stop20_seed3407_vs_a0.json` | Hard/medium/easy bucket analysis. |
| `scout_eval_per_image_saferhfd_v2_pfdonly_stop20_seed3407_vs_a0.csv` | Per-image PSNR/SSIM deltas. |
| `gate_saferhfd_v2_pfdonly_stop20_seed3407_vs_a0.json` | Automatic gate result. |
| `status.txt` | Chronological cloud run status. |
| `run_saferhfd_v2_pfdonly_stop20.sh` | Exact run script. |

## Result

The run finished successfully, but did not pass the promotion gate.

Key metrics versus A0 official ConvIR-B:

- mean PSNR delta: `+0.00568 dB`
- mean SSIM delta: `+0.000024`
- hard bottom-25% mean delta: `+0.04890 dB`
- easy top-25% mean delta: `-0.00920 dB`
- strong-reference regressions: `70 / 250`
- severe regressions: `18 / 1000`
- worst10 mean delta: `-0.16355 dB`

Decision: keep as diagnostic evidence. The route is safer than the original B1
full-model RHFD failure, but the gain is below the replacement threshold and
strong-reference regressions are too high.

## Gate

The automatic gate compares against A0:

- global mean PSNR delta `>= +0.02 dB`
- mean SSIM delta `>= 0`
- hard bottom-25% delta `>= +0.08 dB`
- easy top-25% delta `>= -0.02 dB`
- strong-reference regressions `<= 30 / 250`
- severe regressions `<= 20 / 1000`
- worst10 mean delta `> -0.50 dB`

Manual visual artifact review is still required before promotion.

Gate checks failed on:

- global mean PSNR delta: required `>= +0.02 dB`, observed `+0.00568 dB`
- hard bottom-25% delta: required `>= +0.08 dB`, observed `+0.04890 dB`
- strong-reference regressions: required `<= 30 / 250`, observed `70 / 250`
