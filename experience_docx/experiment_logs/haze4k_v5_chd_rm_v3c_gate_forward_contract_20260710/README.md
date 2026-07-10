# v3c Gate Forward Contract

Status: `PENDING_CLOUD_NO_TRAINING_PREFLIGHT`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3c-gate-forward-contract.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3c addresses the v3b blocker by adding an explicit gate-producing forward
contract for `fam2_d7c_noop` entrypoints. It remains no-training and no-RARM.

## Expected Evidence

- `run_v3c_gate_forward_contract.sh`
- `v3c_gate_forward_contract.log`
- `v3c_static_source_contract.json`
- `v3c_gate_forward_contract_summary.json`
- `v3c_gate_forward_contract_per_image.csv`
- `forbidden_flow_audit.json`
- `v3c_gate_forward_contract_closeout.json`
- `status.txt`

## Current Scope

Cloud no-training preflight only. RARM/training/adapter/canary/locked-test
remain blocked.
