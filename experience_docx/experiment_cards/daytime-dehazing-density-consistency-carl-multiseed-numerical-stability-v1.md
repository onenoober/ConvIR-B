# Daytime Dehazing Density Consistency CARL Multiseed Numerical Stability v1

Status: PLANNED

- Route id: daytime-dehazing-density-consistency-carl-multiseed-numerical-stability-v1
- First operation: DAYTIME_DEHAZING_DENSITY_CONSISTENCY_CARL_MULTISEED_NUMERICAL_STABILITY_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-density-consistency-carl-multiseed-numerical-stability-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The receipt-bound rerun reached 24 of 41 units and ended FAILED_ENGINEERING at the frozen three-seed summary because at least one raw grouped contrast was non-finite. Its archived closeout SHA-256 is 2c3b36264e14288dcc905716fd0922dccb4eb2a5eb14c947636425335e0625b6, with decision null, authorizes=NONE, scientific-data touch, and no protected-data touch. The closeout does not identify the first non-finite tensor, seed, arm, or scene, so this contract makes no such claim and never reads partial scientific results. The user-authorized formal amendment changes one operational fixed factor: CUDA TF32 moves from unspecified framework-default behavior to explicit disablement for matmul and cuDNN. The route adds observation-only fail-fast finite assertions over prediction, component and total loss, gradient, parameter, optimizer, EMA teacher, evaluation prediction, and PSNR state. The protected-data-free engineering contract exercises the first frozen seed through all four arms for 1000 steps each, totaling 4000 representative synthetic iterations under the fixed 900-second limit, before scientific work is eligible. Every scientific cell retains its full 2000-step trajectory through the same asserted code. The scientific 2x2 intervention, five frozen seed blocks, data roles, optimizer, learning rate, budgets, losses and weights, metrics, thresholds, uncertainty, sequential boundaries, and 41-unit ledger remain unchanged. No non-finite value may be filtered, imputed, clipped, replaced, or excluded. This route tests one falsifiable stability repair and cannot establish the cause if another non-finite failure occurs. Development evidence cannot authorize confirmation, sealed-final access, promotion, deployment, publication, or another repair route.
