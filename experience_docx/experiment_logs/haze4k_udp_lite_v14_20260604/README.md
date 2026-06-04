# Haze4K UDP-Lite v1.4 Evidence Root

Date: 2026-06-04

Status: v1.4A adapter-only completed; internal gate failed; locked Haze4K test blocked.

This directory is for ConvIR-Dehaze-v1.4-UDP-Lite text evidence only.
The WSL checkout is for editing and compile/syntax-only checks. Run zero-init,
smoke, training, evaluation, inference, and module audits on `dehaze1`.

## Artifacts

- `v14_udpnet_repro_audit.md`: external UDPNet ConvIR repository/checkpoint/protocol audit.
- `preflight/v14_zero_init_equivalence.json`: A0 equivalence and projection-gradient liveness for UDP-Lite.
- `train_ConvIR-Haze4K-v1.4A-UDP-Lite-DPFM123-adapter-only-seed3407-20260604.log`: v1.4A train log.
- `v14a_eval_regular_hard/v14a_gate_eval_regular_and_hard.json`: Best/Final internal regular+hard gate summary.
- `v14a_eval_regular_hard/scout_eval_compare_*_vs_a0.json`: Best/Final regular/hard A0 comparison summaries.
- `v14a_intermediates/*/v14_depth_fusion_module_ablation_val.csv`: DGCA/DPFM scale ablation on `val_regular` and `val_hard`.
- `v14a_intermediates/*/v14_depth_quality_failure_audit.csv`: depth/prior/failure signatures for hard and regression cases.
- `v14_locked_selection_protocol.md`: fixed validation and locked-test policy.

## Run Outcome

- v1.4A launched on `dehaze1` at `2026-06-04T22:02:25+08:00` and finished at epoch `20`.
- `Best.pkl`, `Final.pkl`, `model.pkl`, and `model_20.pkl` were produced in the cloud results directory.
- Best gate failed: `val_regular` mean `+0.028294 dB`, positive ratio `0.586667`, worst count `19`; `val_hard` mean `+0.020340 dB`, hard bottom-25 `+0.022275 dB`.
- Module ablation: `DPFM1-only` is safer/stronger than full `DPFM1+2+4`; `DPFM2-only` is negative on both splits.
- Decision: `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`; locked Haze4K test remains blocked.

## Run Order

1. Run `run_v14_udpnet_repro_audit.sh` to inspect the external UDPNet repo.
2. Run `run_v14_zero_init_equivalence.sh` and require `equivalence_pass=true`.
3. Launch v1.4A only after zero-init and split/depth cache paths are verified.
4. Evaluate Best/Final on `val_regular` and `val_hard`; do not use locked test.
5. Run `run_v14_intermediate_audits.sh` after a checkpoint exists.

## Decision Boundary

The first route decision must come from internal `val_regular` and `val_hard`.
Locked Haze4K test remains blocked because v1.4A failed the written internal gates.
