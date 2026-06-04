# Haze4K DPGA-v1.1 Tail-Control Diagnostics

Date: 2026-06-04

Status: completed diagnostic route on `autodl-dehaze4`; locked Haze4K test
remains blocked.

## Purpose

This directory holds the first DPGA-v1.1 tail-control intermediate-result
programs. They are designed to run independently of training and answer two
questions before any new long run:

- Which DPGA insertion point causes gain or strong-reference tail risk?
- Does reducing the effective DPGA scale preserve mean gain while lowering
  worst/strong regressions?

## Scripts

- `run_make_internal_val_split.sh`: writes a fixed Haze4K
  `train_inner`/`val_inner` split JSON under `internal_val/`.
- `run_dpga_runtime_diagnostics.sh`: evaluates existing DPGA-Lite `Best.pkl`
  and `Final.pkl` with module ablation and scale sweep runtime controls.
- `run_decide_dpga_v1_1_training.sh`: parses the diagnostic CSVs and writes
  a machine-readable DPGA-v1.1 training decision.
- `run_dpga_v1_1_tail_control_train.sh`: launches the selected DPGA-v1.1
  Tail-Control training run on `train_inner`, with checkpoint selection on
  `val_inner`.
- `run_watch_and_launch_dpga_v1_1.sh`: waits for diagnostics to finish, writes
  the decision, and launches training only when the decision gate allows it.
- `run_eval_dpga_v1_1_val_inner.sh`: evaluates the v1.1 Best/Final
  checkpoints against A0 on `val_inner` only.
- `run_watch_dpga_v1_1_posttrain.sh`: waits for v1.1 `Final.pkl`, then runs
  the `val_inner` gate.
- `run_analyze_dpga_v1_1_val_failure.sh`: turns the v1.1 gate failure into
  grouped hard/easy/strong/worst intermediate evidence.
- `run_decide_dpga_v1_2_training.sh`: decides whether the hard-gain follow-up
  is justified by the v1.1 failure shape.
- `run_dpga_v1_2_hard_gain_train.sh`: launches the next small-step cloud run
  only when the v1.2 decision allows it.
- `run_watch_dpga_v1_2_posttrain.sh`: waits for v1.2 `Final.pkl`, then runs
  the same `val_inner` gate.

## Expected Outputs

- `internal_val/haze4k_train_inner_val_inner_seed3407.json`
- `runtime_diagnostics/dpga_module_ablation_best_final.csv`
- `runtime_diagnostics/dpga_scale_sweep_best_final.csv`
- `runtime_diagnostics/dpga_module_ablation_per_image.csv`
- `runtime_diagnostics/dpga_scale_sweep_per_image.csv`
- `runtime_diagnostics/dpga_runtime_variants_summary.json`
- `v1_1_decision/dpga_v1_1_training_decision.json`
- `v1_1_decision/dpga_v1_1_training_decision.md`
- `train_ConvIR-Haze4K-DPGA-v1.1-tail-control-*.log`
- `v1_1_val_inner_eval/gate_dpga_v1_1_val_inner.json`
- `v1_1_failure_analysis/dpga_v1_1_val_inner_failure_analysis.json`
- `v1_2_decision/dpga_v1_2_training_decision.json`
- `train_ConvIR-Haze4K-DPGA-v1.2-hard-gain-*.log`
- `v1_2_val_inner_eval/gate_dpga_v1_2_val_inner.json`

These outputs are text-only and safe to sync after the cloud run completes.

## Final Results

| Stage | Result | Notes |
| --- | --- | --- |
| Runtime diagnostics | selected shallow-only scale-control path | module ablation and scale sweep reduced the original DPGA-Lite tail risk enough to justify v1.1 training |
| v1.1 `val_inner` gate | fail | Best mean `+0.037036 dB` passed; hard bottom-25% `+0.023367 dB` missed the `+0.030 dB` gate |
| v1.2 `val_inner` gate | fail | Best mean improved to `+0.042656 dB`; hard bottom-25% reached only `+0.026225 dB`; worst `<= -0.20 dB` regressions rose to `16/300` |

Decision: `STOP_DPGA_SCALE_ONLY_TAIL_CONTROL`. Do not run locked Haze4K test
for v1.1 or v1.2. A future DPGA follow-up needs a new diagnostic or mechanism
that directly improves hard bottom-25% gain without increasing worst-tail
regressions.

## Training Guardrails

DPGA-v1.1 training must not start from manual test-set Best selection. The
watcher only uses the completed runtime diagnostics to choose a starting
adapter/scale configuration, then trains on `train_inner` and validates on
`val_inner`. Locked Haze4K test evaluation remains a later step after the
checkpoint/config is fixed.
