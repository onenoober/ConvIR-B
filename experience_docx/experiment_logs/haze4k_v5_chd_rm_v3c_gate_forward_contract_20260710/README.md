# v3c Gate Forward Contract

Status: `COMPLETED_GATE_PASS`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3c-gate-forward-contract.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3c addresses the v3b blocker by adding an explicit gate-producing forward
contract for `fam2_d7c_noop` entrypoints. It remains no-training and no-RARM.

## Evidence

- `run_v3c_gate_forward_contract.sh`
- `v3c_gate_forward_contract.log`
- `v3c_static_source_contract.json`
- `v3c_gate_forward_contract_summary.json`
- `v3c_gate_forward_contract_per_image.csv`
- `forbidden_flow_audit.json`
- `v3c_gate_forward_contract_closeout.json`
- `status.txt`

## Result

Decision:
`V3C_GATE_FORWARD_CONTRACT_PASS_AUTHORIZE_NO_TRAINING_ENTRYPOINT_PREFLIGHT_ONLY`

The cloud audit passed:

- samples: `16` internal val-inner images;
- nontrivial D7c gate images: `16/16`;
- D7c coverage mean/min/max:
  `0.3246304675703868` / `0.015908146277070045` /
  `0.6701125502586365`;
- output max absolute diff: `0.0`;
- PSNR/SSIM max absolute deltas: `0.0` / `0.0`;
- source forward-contract check: pass;
- modulation stats include D7c gate stats;
- training/RARM/adapter/canary/locked-test: not used.

This resolves the v3b entrypoint-contract blocker only. RARM/training remains
blocked until a separate written decision authorizes the next phase.
