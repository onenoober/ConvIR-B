# Haze4K APDR-ConvIR v0.2RC Budget Logs

Status: completed cloud conservative-budget replay; replay gate failed.

Route card:
`experience_docx/experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2rc-budget.md`

This directory is reserved for text evidence from the APDR-v0.2RC route.
Residual training must not be launched unless the conservative-budget replay
and oracle residual ceiling gates both pass.

## Result

- Cloud run: AutoDL `autodl-dehaze3`, tmux session
  `apdr_v0_2rc_budget_20260602`.
- Source snapshot: GitHub commit
  `6f7bf1dd2badaf6bc14aa14b1c4e091e08ff1f02`.
- Gate label: `FAIL_STOP_APDR_V0_2RC_BUDGET_CALIBRATION`.
- Candidate grid: `489` budget maps; `33` passed train constraints; `0`
  passed the full held-out replay gate.
- Selected train-only candidate: `platt_tau-1.4839_t1_g4`.
- Train constraints passed: hard mean `0.373883`, easy mean `0.008297`,
  hard/easy ratio `45.0604`, calibration BCE `0.540913`.
- Held-out positives: zero-output diff `0.0`, AUC `0.97664`, Spearman
  `-0.74664`, hard mean budget `0.378346`, easy/strong-reference mean budget
  `0.002531`, hard/easy ratio `149.481`.
- Held-out failure: calibration BCE `1.619142` versus required `<= 0.55`.
- Oracle residual ceiling was not evaluated because replay gate failed.

Decision: do not launch residual training from v0.2RC. The route successfully
closes easy images, but the single-head conservative budget does not remain a
well-calibrated action probability on held-out data.

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
| `oracle_per_image_apdr_v0_2rc_budget_seed3407.csv` | Not produced; replay gate failed before oracle. |
| `gate_apdr_v0_2rc_budget_seed3407.json` | Predeclared replay and oracle gates. |
| `budget_preflight_apdr_v0_2rc_budget_seed3407.log` | Cloud stdout/stderr from budget preflight. |
| `status.txt` | Timestamped cloud status stream. |
| `launcher.out` | Detached cloud launcher transcript when applicable. |
