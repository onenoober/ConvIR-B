# Haze4K APDR-ConvIR v0.2R Selector Logs

Status: completed cloud selector-only preflight; failed selector gate.

Route card:
`experience_docx/experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2r-selector.md`

This directory contains text evidence from the APDR-v0.2R selector-only route.
Residual training was not launched because the selector-only gate failed.

Boundary:

- Phase A trains only the full-scale decoupled global hard router;
- Phase B freezes the global router and trains only the full-scale spatial
  risk gate;
- residual output is force-zero, so APDR output must match A0 exactly;
- checkpoints, image outputs, datasets, arrays, and raw inference artifacts stay
  out of Git.

Expected text artifacts:

| File | Purpose |
| --- | --- |
| `run_apdr_v0_2r_selector_preflight.sh` | Reproducible AutoDL command wrapper. |
| `preflight_apdr_v0_2r_arch.json` | Architecture, load, zero-init, and finite-backward preflight. |
| `selector_summary_apdr_v0_2r_selector_seed3407.json` | Selector-only training, calibration, test metrics, and gate summary. |
| `selector_per_image_apdr_v0_2r_selector_seed3407.csv` | Per-image A0 PSNR, `z_img`, `B_img`, spatial BCE, and zero-output diff. |
| `selector_history_apdr_v0_2r_selector_seed3407.csv` | Phase A/B selector loss history. |
| `gate_apdr_v0_2r_selector_seed3407.json` | Predeclared selector-only gate. |
| `selector_preflight_apdr_v0_2r_selector_seed3407.log` | Cloud stdout/stderr from selector preflight. |
| `status.txt` | Timestamped cloud status stream. |
| `launcher.out` | Detached cloud launcher transcript when applicable. |

## Result Summary

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-2r-fullimage-router`.

Architecture preflight passed with Haze4K `3000/3000` train pairs and
`1000/1000` test pairs. APDR-v0.2R added `85,119` parameters over official
ConvIR-B, and zero-init equivalence stayed exact at `max_abs_diff = 0.0`.

Selector-only full-test gate:

| Gate | Observed | Result |
| --- | ---: | --- |
| zero-residual max diff vs A0 | `0.0` | pass |
| deterministic hard BCE | `1.70804 -> 0.630292` | fail |
| AUC hard vs easy by `z_img` | `0.97664` | pass |
| Spearman(`z_img`, A0 PSNR) | `-0.74664` | pass |
| mean `B_img` hard bottom-25% | `0.782352` | pass |
| mean `B_img` easy top-25% | `0.146157` | fail |
| hard/easy `B_img` ratio | `5.35281` | pass |
| spatial BCE | `2.06208 -> 0.733602` | pass |
| full-test mean spatial BCE | `0.757404` | pass |

Decision: `FAIL_STOP_APDR_V0_2R_SELECTOR_ONLY`. The full-image router learned a
strong hard/easy ranking, but the train-calibrated budget remains too open on
easy/strong-reference images. Do not launch residual training from this selector.
