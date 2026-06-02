# HAZE4K B1-Surgery Preserve Sweep 20260602

Date: 2026-06-02

Status: completed no-training diagnostic; `scale=0.70` selected as the
preservation-first surgery candidate.

## Purpose

B1-Surgery tests whether the RHFD branch learned in B1 can be reused without carrying over B1's full-backbone fine-tuning regressions.

The surgery checkpoint rule is:

- Backbone: A0 official ConvIR-B checkpoint.
- Added branches: `PFD_RHFD1` and `PFD_RHFD2` copied from B1 `Best.pkl`.
- Branch strength: scale only the RHFD final `body.4` conv weight and bias.
- No training is performed.

## Gate

The minimum pass line used for this sweep:

- mean PSNR delta vs A0: `>= 0.000 dB`
- mean SSIM delta vs A0: `>= -0.00005`
- hard bottom-25% mean PSNR delta: `>= +0.03 dB`
- easy top-25% mean PSNR delta: `>= -0.02 dB`
- severe regressions (`delta <= -0.20 dB`): `<= 50 / 1000`
- strong-reference regressions (`delta <= -0.05 dB` in A0 top-25%): `<= 50 / 250`

## Result

`scale=0.70` is the recommended preservation-first candidate.

| Scale | Mean PSNR delta | Hard delta | Easy delta | Severe | Strong reg | Global reg <= -0.05 | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | +0.00019 | +0.00056 | +0.00005 | 0 | 1 | 2 | fail hard gain |
| 0.10 | +0.00109 | +0.00282 | +0.00127 | 0 | 0 | 0 | fail hard gain |
| 0.20 | +0.00440 | +0.01403 | +0.00313 | 0 | 0 | 0 | fail hard gain |
| 0.30 | +0.00566 | +0.01763 | +0.00437 | 0 | 0 | 2 | fail hard gain |
| 0.50 | +0.00815 | +0.02508 | +0.00627 | 0 | 0 | 4 | fail hard gain |
| 0.70 | +0.01064 | +0.03317 | +0.00782 | 0 | 0 | 9 | pass |
| 1.00 | +0.01268 | +0.03888 | +0.00980 | 0 | 9 | 31 | pass, higher regression |

Although `scale=1.00` also passes the minimum gate, it introduces more global, easy, and strong-reference regressions. Use `scale=0.70` as the primary SafeRHFD candidate and keep `scale=1.00` as high-gain backup evidence.

## Diagnostic

The selected diagnostic pack is:

- `diagnostic_b1_surgery_s0.70_vs_a0/`
- selected samples: 38
- visual panel: generated remotely and reviewed locally, not committed
- branch and safety stats: committed as text-only evidence

Visual inspection found no B1-style brightness/color/range collapse. Candidate-original differences are small and mostly localized; worst regressions are metric-level or mild texture/residual-haze shifts rather than catastrophic output drift.

## Files

| File | Use |
| --- | --- |
| `surgery_sweep_summary.json` | Structured sweep summary, gates, passing scales, and recommendation. |
| `surgery_sweep_summary.csv` | Scale-by-scale sweep table. |
| `scout_eval_compare_b1_surgery_s0.05_vs_a0.json` | A0-vs-surgery summary for `scale=0.05`. |
| `scout_eval_compare_b1_surgery_s0.10_vs_a0.json` | A0-vs-surgery summary for `scale=0.10`. |
| `scout_eval_compare_b1_surgery_s0.20_vs_a0.json` | A0-vs-surgery summary for `scale=0.20`. |
| `scout_eval_compare_b1_surgery_s0.30_vs_a0.json` | A0-vs-surgery summary for `scale=0.30`. |
| `scout_eval_compare_b1_surgery_s0.50_vs_a0.json` | A0-vs-surgery summary for `scale=0.50`. |
| `scout_eval_compare_b1_surgery_s0.70_vs_a0.json` | A0-vs-surgery summary for selected `scale=0.70`. |
| `scout_eval_compare_b1_surgery_s1.00_vs_a0.json` | A0-vs-surgery summary for backup `scale=1.00`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.05_vs_a0.json` | Bucket analysis for `scale=0.05`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.10_vs_a0.json` | Bucket analysis for `scale=0.10`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.20_vs_a0.json` | Bucket analysis for `scale=0.20`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.30_vs_a0.json` | Bucket analysis for `scale=0.30`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.50_vs_a0.json` | Bucket analysis for `scale=0.50`. |
| `scout_eval_bucket_analysis_b1_surgery_s0.70_vs_a0.json` | Bucket analysis for selected `scale=0.70`. |
| `scout_eval_bucket_analysis_b1_surgery_s1.00_vs_a0.json` | Bucket analysis for backup `scale=1.00`. |
| `scout_eval_per_image_b1_surgery_s0.05_vs_a0.csv` | Per-image deltas for `scale=0.05`. |
| `scout_eval_per_image_b1_surgery_s0.10_vs_a0.csv` | Per-image deltas for `scale=0.10`. |
| `scout_eval_per_image_b1_surgery_s0.20_vs_a0.csv` | Per-image deltas for `scale=0.20`. |
| `scout_eval_per_image_b1_surgery_s0.30_vs_a0.csv` | Per-image deltas for `scale=0.30`. |
| `scout_eval_per_image_b1_surgery_s0.50_vs_a0.csv` | Per-image deltas for `scale=0.50`. |
| `scout_eval_per_image_b1_surgery_s0.70_vs_a0.csv` | Per-image deltas for selected `scale=0.70`. |
| `scout_eval_per_image_b1_surgery_s1.00_vs_a0.csv` | Per-image deltas for backup `scale=1.00`. |
| `diagnostic_b1_surgery_s0.70_vs_a0/diagnostic_summary.json` | Selected diagnostic pack summary. |
| `diagnostic_b1_surgery_s0.70_vs_a0/output_safety_stats.csv` | Output safety statistics for selected samples. |
| `diagnostic_b1_surgery_s0.70_vs_a0/pfd_branch_stats.csv` | PFD branch activity statistics. |
| `diagnostic_b1_surgery_s0.70_vs_a0/pfd_branch_stats_by_category.json` | Branch statistics grouped by diagnostic category. |
| `diagnostic_b1_surgery_s0.70_vs_a0/sample_manifest.csv` | Selected sample manifest. |
| `diagnostic_b1_surgery_s0.70_vs_a0/visual_notes_filled.md` | Filled visual review notes. |
| `diagnostic_b1_surgery_s0.70_vs_a0/visual_notes_template.md` | Visual review template. |
| `status.txt` | Chronological status for the sweep. |
| `sweep_status.txt` | Initial sweep status. |
| `sweep_stdout.log` | Initial sweep command output. |
| `sweep_extension_status.txt` | Extension sweep status. |
| `sweep_extension_stdout.log` | Extension sweep command output. |

## Artifact Boundary

Commit text-only evidence:

- `surgery_sweep_summary.json`
- `surgery_sweep_summary.csv`
- scale-specific compare JSON, bucket JSON, and per-image CSV files listed
  above;
- status and stdout text logs listed above;
- diagnostic CSV/JSON/Markdown files under
  `diagnostic_b1_surgery_s0.70_vs_a0/`.

Do not commit:

- surgery checkpoints under `Dehazing/ITS/results/PFD-B1-surgery-*`
- `visual_panel_20.png`
- `samples/*.png`
- datasets or raw inference artifacts
