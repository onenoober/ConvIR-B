# Haze4K APDR-ConvIR v0.2R Selector Logs

Status: planned cloud selector-only preflight.

Route card:
`experience_docx/experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2r-selector.md`

This directory is reserved for text evidence from the APDR-v0.2R selector-only
route. Residual training must not be launched unless the selector-only gate
passes.

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
