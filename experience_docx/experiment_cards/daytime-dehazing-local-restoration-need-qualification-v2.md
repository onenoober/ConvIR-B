# Daytime Dehazing Local Restoration Need Qualification v2

Status: PLANNED

- Route id: daytime-dehazing-local-restoration-need-qualification-v2
- First operation: DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-local-restoration-need-qualification-v2.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Use only program-local development-screening scenes to decide whether a fixed dehazing baseline has a repeatable and practically useful spatial residual-strength calibration problem rather than a measurement, aggregation, context, or global-difficulty artifact. This revised S1 keeps the v1 scene, variant, crop, estimand, uncertainty, and decision rules, but repairs the pre-inference validity veto by consuming the outcome-blind RESIDE pairing-v2 ledger exactly: 300 clear scenes, 600 nested haze observations, fixed relative paths, file SHA-256, decoded RGB SHA-256, dimensions, and scoring geometry. The privileged diagnostic uses five local residual strengths with two-way 8-by-8 macroblock cross-fitting inside 4-by-4 cell cores and compares pooled evaluation-pixel SSE with a stronger 1001-point scene-global scalar grid. The official ConvIR-B model now runs on each complete padded image before the frozen square scoring crop is extracted, matching the authorized pairing-v2 input contract and the device-aware synthetic capability fixture. A spatially permuted action-map control, shifted-target selection null, explicit unresolved label, near-clear exposure/damage/conditional-mitigation decomposition, negative-tail control, and compact endpoint/saturation/edge/clipping diagnostics constrain alternative explanations. The route trains no network, tests no observable signal, proposes no post-processing method, and cannot access confirmation, NH-Haze, or sealed-final images. PASS authorizes only authoring an S2 mechanism-discovery contract; a valid scientific FAIL rejects this first operationalization without spending the predeclared second S1 attempt automatically.
