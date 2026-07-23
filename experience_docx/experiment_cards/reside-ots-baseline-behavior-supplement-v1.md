# Official ConvIR-B OTS baseline behavior complete-scene supplement

Status: PLANNED

- Route id: reside-ots-baseline-behavior-supplement-v1
- First operation: OTS_BASELINE_BEHAVIOR_SUPPLEMENT
- Program contract: experience_docx/research_programs/reside_ots_baseline_behavior_supplement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-baseline-behavior-supplement-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The direct parent ended INCONCLUSIVE solely because a long-edge-320 resize made the short edge too small for an internal official ConvIR-B reflection pad in 218 aspect-ratio-extreme scenes. Parent and dataset identities matched, the GPU contract passed, and 407 scenes completed, but those diagnostic outcomes cannot substitute for the frozen 625-scene population. This one supplement recomputes all 625 scenes and all three frozen variants with the sole change that aspect ratio is preserved while the short edge is at least 64 pixels; the nominal 320-pixel long-edge cap may be exceeded only as needed to meet that minimum. An extreme 64x1024 GPU no-data fixture now exercises the exact official forward path. Model, population, variants, regions, estimand, margins, negative control, uncertainty, and terminal rules remain unchanged.
