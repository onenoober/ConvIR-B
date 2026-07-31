# Daytime Dehazing Local Restoration Need Qualification v1

Status: PLANNED

- Route id: daytime-dehazing-local-restoration-need-qualification-v1
- First operation: DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-local-restoration-need-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Use only program-local development-screening scenes to decide whether a fixed dehazing baseline has a repeatable and practically useful spatial residual-strength calibration problem rather than a measurement, aggregation, context, or global-difficulty artifact. The privileged diagnostic uses five local residual strengths with two-way 8-by-8 macroblock cross-fitting inside 4-by-4 cell cores and compares pooled evaluation-pixel SSE with a stronger 1001-point scene-global scalar grid. Two outcome-blind haze observations test exact direction-and-strength repeatability and cross-observation action-map transfer. A spatially permuted action-map control, shifted-target selection null, explicit unresolved label, near-clear exposure/damage/conditional-mitigation decomposition, negative-tail control, S0 canonical clear identity, frozen input roster, and compact endpoint/saturation/edge/clipping diagnostics constrain alternative explanations. The route trains no network, tests no observable signal, proposes no post-processing method, and cannot access confirmation, NH-Haze, or sealed-final images. PASS authorizes only authoring an S2 mechanism-discovery contract; a valid FAIL rejects this first operationalization without spending the predeclared second S1 attempt automatically.
