# Haze4K APDR-ConvIR v0.2RC Budget Logs

Status: planned cloud conservative-budget replay and oracle ceiling preflight.

Route card:
`experience_docx/experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2rc-budget.md`

This directory is reserved for text evidence from the APDR-v0.2RC route.
Residual training must not be launched unless the conservative-budget replay
and oracle residual ceiling gates both pass.

Boundary:

- rerun APDR-v0.2R selector-only on AutoDL to collect train and test `z_img`;
- choose conservative budget mapping using train scores only;
- evaluate test once for the replay gate;
- compute oracle residual ceiling only after the replay gate passes, unless
  explicitly requested for diagnostics;
- keep checkpoints, image outputs, datasets, arrays, and raw inference artifacts
  out of Git.

Expected text artifacts:

| File | Purpose |
| --- | --- |
| `run_apdr_v0_2rc_budget_preflight.sh` | Reproducible AutoDL command wrapper. |
| `preflight_apdr_v0_2rc_arch.json` | Architecture, load, zero-init, and finite-backward preflight. |
| `budget_summary_apdr_v0_2rc_budget_seed3407.json` | Selector replay, budget calibration, gate, and optional oracle summary. |
| `budget_train_scores_apdr_v0_2rc_budget_seed3407.csv` | Train split `z_img` and A0-risk targets used for budget selection. |
| `budget_test_scores_apdr_v0_2rc_budget_seed3407.csv` | Test split `z_img`, A0 PSNR, spatial BCE, and zero-output diff. |
| `budget_candidates_apdr_v0_2rc_budget_seed3407.csv` | Candidate budget maps with train/test metrics. |
| `oracle_per_image_apdr_v0_2rc_budget_seed3407.csv` | Optional oracle residual ceiling per-image results. |
| `gate_apdr_v0_2rc_budget_seed3407.json` | Predeclared replay and oracle gates. |
| `budget_preflight_apdr_v0_2rc_budget_seed3407.log` | Cloud stdout/stderr from budget preflight. |
| `status.txt` | Timestamped cloud status stream. |
| `launcher.out` | Detached cloud launcher transcript when applicable. |
