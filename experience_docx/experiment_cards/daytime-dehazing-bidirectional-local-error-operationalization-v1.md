# Daytime Dehazing Bidirectional Local Error Operationalization v1

Status: PLANNED

- Route id: daytime-dehazing-bidirectional-local-error-operationalization-v1
- First operation: DAYTIME_DEHAZING_BIDIRECTIONAL_LOCAL_ERROR_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-bidirectional-local-error-operationalization-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Use the one remaining adjacent S1 attempt to test whether the sequence-1 failure was specific to rigid equal-area regions, five discrete local strengths, and observation-specific maps rather than to the program's core bidirectional local-error assumption. The unchanged official ConvIR-B baseline is evaluated on the same 450 development-screening clear scenes and exactly two nested haze observations per scene. Sequence 2 constructs one shared 16-region partition from hazy images and frozen official outputs only, aligns its boundaries to two-observation consensus content and residual discontinuities, and gives local and global comparators the same dense 1001-point residual-strength grid. Ground truth remains a privileged offline selection signal only. The primary estimand is held-out cross-observation transfer over a target-specific in-sample global oracle; same-observation utility, spatial permutation, shifted-target artifact, strength-tolerance stability, negative tail, and near-clear fidelity are decisive controls. The route performs no training, model change, observable predictor fitting, confirmation, NH-Haze, sealed-final access, or deployment. PASS authorizes only S2 contract authoring; a valid FAIL closes the local-restoration-need family; validity or precision insufficiency remains inconclusive only.
