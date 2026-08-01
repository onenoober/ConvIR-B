# Daytime Dehazing Local Restoration Need Qualification v3

Status: PLANNED

- Route id: daytime-dehazing-local-restoration-need-qualification-v3
- First operation: DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-local-restoration-need-qualification-v3.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Requalify the unchanged S1 question without repeating ConvIR-B inference. The route consumes only SHA-bound compact v2 terminal, summary, gate, and 450-scene CSV evidence plus the typed S1 validity-review PASS. It preserves every historical metric, threshold, scientific gate outcome, and terminal. The revised decision contract separates affirmative shifted-target null artifact evidence, which is a validity veto, from failure to certify the null below the unchanged margins, which is inconclusive-only and cannot hide a decisive utility, repeatability, or fidelity FAIL. It adds scene-grouped uncertainty for edge and interior contrasts and explicitly records that the archived CSV lacks scoring-crop side, so padded full-image area cannot be relabeled as a crop-size result. The route reads no images, arrays, weights, checkpoints, protected data, or model code and performs no inference or training. PASS authorizes only an S2 contract; FAIL preserves the predeclared second S1 operationalization; INCONCLUSIVE authorizes only another bounded S1 validity or precision review.
