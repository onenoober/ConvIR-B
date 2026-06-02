# HAZE4K B1-Surgery Preserve Sweep 20260602

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

## Artifact Boundary

Commit text-only evidence:

- `surgery_sweep_summary.json`
- `surgery_sweep_summary.csv`
- `scout_eval_compare_*.json`
- `scout_eval_bucket_analysis_*.json`
- `scout_eval_per_image_*.csv`
- `sweep*_status.txt`
- `sweep*_stdout.log`
- diagnostic CSV/JSON/Markdown files

Do not commit:

- surgery checkpoints under `Dehazing/ITS/results/PFD-B1-surgery-*`
- `visual_panel_20.png`
- `samples/*.png`
- datasets or raw inference artifacts
