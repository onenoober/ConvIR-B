# Haze4K UDP-Lite v1.4B BiDPFM1 Evidence Root

Date: 2026-06-04

Status: v1.4B adapter-only completed; internal regular+hard gate failed; locked Haze4K test blocked.

This directory is for ConvIR-Dehaze-v1.4B-BiDPFM1 text evidence only. The WSL
checkout is for editing and compile/syntax-only checks. Run zero-init, runtime
audits, training, evaluation, inference, and post-run watchers on `dehaze1`.

## Artifacts

- `run_v14b_runtime_component_matrix.sh`: no-training v1.4A component matrix for
  `dpfm1_channel_only`, `dpfm1_cross_only`, `dpfm1_all`, `dpfm4_only`,
  `dpfm1_plus_dpfm4`, `dpfm1_plus_dpfm2`, and full `dpfm1+2+4`.
- `run_v14b_zero_init_equivalence.sh`: BiDPFM1 A0-equivalence and projection
  gradient liveness preflight.
- `run_v14b_train_adapter_only.sh`: v1.4B adapter-only cloud training launcher.
- `run_v14b_eval_regular_hard.sh`: Best/Final `val_regular` and `val_hard`
  evaluation plus v1.4B continue/locked-test gate.
- `preflight/`: expected zero-init JSON/log output.
- `v14b_runtime_component_matrix/`: expected no-training matrix output.
- `v14b_eval_regular_hard/`: Best/Final regular/hard eval and gate output.

## Route Boundary

- Active route: `ConvIR-Dehaze-v1.4B-BiDPFM1`.
- Active adapters: `dpfm1` only.
- Fusion mode: `udp_bi`.
- Disabled route components: `dpfm2`, `dpfm4`, `agf1`, `agf2`, legacy DPGA,
  hard gate, APDR/output residual, and locked Haze4K test.
- Train scope: `active_adapter_only`, intended to train `DPGA_prior_encoder.stem`
  plus `DPGA_dpfm1` while leaving ConvIR-B frozen.

## Run Order

1. Run `run_v14b_runtime_component_matrix.sh` independently from training to
   finish the v1.4A no-training component audit.
2. Run `run_v14b_zero_init_equivalence.sh` and require `equivalence_pass=true`
   plus nonzero BiDPFM1 zero-projection gradients.
3. Launch `run_v14b_train_adapter_only.sh` only after preflight passes.
4. Run `run_v14b_eval_regular_hard.sh` for Best/Final on `val_regular` and
   `val_hard`.
5. Do not run locked Haze4K test unless the gate JSON has
   `locked_test_allowed=true`.

## Decision Boundary

The first decision must come from internal `val_regular` and `val_hard`. If the
route reaches only `continue_allowed=true`, proceed to a separate v1.4C
full-resolution fusion-neighbor adapter route; do not run locked test.

## Current Preflight Result

- `preflight/v14b_zero_init_equivalence.json`: `equivalence_pass=true`,
  `max_abs_diff=0.0`, and nonzero gradients for `DPGA_dpfm1.channel.mlp.2`,
  `DPGA_dpfm1.cross_rgb_from_depth.project`, and
  `DPGA_dpfm1.cross_depth_from_rgb.project`.
- `v14b_runtime_component_matrix/*/v14b_runtime_component_matrix.csv`:
  `DPFM1+4` improves mean on both splits (`+0.035215 dB` regular,
  `+0.031952 dB` hard) but raises `val_regular` worst count to `8` and strong
  regression ratio to `0.24`, so the first v1.4B training run remains
  BiDPFM1-only.

## Run Outcome

- v1.4B adapter-only training completed on `dehaze1` at
  `2026-06-05T00:18:16+08:00`.
- Internal Best/Final eval completed at `2026-06-05T00:25:33+08:00`.
- Gate result: `continue_allowed=false`, `locked_test_allowed=false`.
- Best `val_regular`: mean `+0.028624 dB`, positive ratio `0.536667`, worst
  count `17`, strong regression ratio `0.280000`, SSIM delta `-0.000007778`.
- Best `val_hard`: mean `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, worst
  count `8`, SSIM delta `-0.000012043`.
- Decision: `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`; do not run locked Haze4K
  test or rerun BiDPFM1-only scale/gate tuning.
