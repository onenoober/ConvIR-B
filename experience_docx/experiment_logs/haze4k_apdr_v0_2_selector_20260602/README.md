# Haze4K APDR-ConvIR v0.2 Selector Logs

Status: completed cloud selector-only preflight; failed selector gate.

Route card:
`experience_docx/experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2-selector.md`

This directory contains text evidence from the APDR-v0.2 selector-only route.
Residual training was not launched because the selector-only gate failed.

Boundary:

- train only the APDR full-scale selector/context parameters;
- freeze the residual head and force zero residual output;
- compare selector scores against A0 difficulty buckets on the full Haze4K
  test split;
- keep checkpoints, image outputs, datasets, arrays, and raw inference
  artifacts out of Git.

Text artifacts:

| File | Purpose |
| --- | --- |
| `run_apdr_v0_2_selector_preflight.sh` | Reproducible AutoDL command wrapper. |
| `preflight_apdr_v0_2_arch.json` | Architecture, load, zero-init, and finite-backward preflight. |
| `selector_summary_apdr_v0_2_selector_seed3407.json` | Selector-only training, calibration, test metrics, and gate summary. |
| `selector_per_image_apdr_v0_2_selector_seed3407.csv` | Per-image A0 PSNR, selector scores, spatial BCE, and zero-output diff. |
| `selector_history_apdr_v0_2_selector_seed3407.csv` | Selector loss history. |
| `gate_apdr_v0_2_selector_seed3407.json` | Predeclared selector-only gate. |
| `selector_preflight_apdr_v0_2_selector_seed3407.log` | Cloud stdout/stderr from selector preflight. |
| `status.txt` | Timestamped cloud status stream. |
| `launcher.out` | Detached cloud launcher transcript when applicable. |

## Result Summary

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-2`.

Architecture preflight passed with Haze4K `3000/3000` train pairs and
`1000/1000` test pairs. APDR-v0.2 added `64,575` parameters over official
ConvIR-B, and zero-init equivalence stayed exact at `max_abs_diff = 0.0`.

Selector-only calibration used all `3000` Haze4K train images:

- RMSE q50/q90: `0.0105008` / `0.0219539`.
- Pixel-error q70/q90: `0.0100390` / `0.0211587`.
- Spatial tau: `0.0111198`.

Full-test selector gate:

| Gate | Observed | Result |
| --- | ---: | --- |
| hard/easy `H_img` ratio | `1.00245` | fail |
| Spearman(`H_img`, A0 PSNR) | `-0.35373` | fail |
| AUC hard vs easy by `H_img` | `0.768624` | pass |
| strong-reference mean `H_img` | `0.0212881` | pass |
| spatial BCE | `2.06398 -> 0.729276` | pass |
| zero-residual max diff vs A0 | `0.0` | pass |

Decision: `FAIL_STOP_APDR_V0_2_SELECTOR_ONLY`. Spatial risk learned, but the
image-level hard selector stayed nearly flat: hard bottom-25% mean `H_img` was
`0.0213403`, easy top-25% mean `H_img` was `0.0212881`. Do not launch the
residual stage from this selector.
