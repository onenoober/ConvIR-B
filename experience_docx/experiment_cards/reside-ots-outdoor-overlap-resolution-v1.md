# RESIDE OTS outdoor source-overlap resolution

Status: PLANNED

- Route id: reside-ots-outdoor-overlap-resolution-v1
- First operation: OTS_OVERLAP_RESOLUTION
- Program contract: experience_docx/research_programs/reside_ots_outdoor_measurement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-outdoor-overlap-resolution-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

RESIDE OTS is the most suitable mainstream outdoor pool, but the prior qualification conservatively blocked it because only 458 of 500 provenance-expected Haze4K outdoor sources were mapped. This read-only source-identity operation first reproduces the archived strict mappings, then requires a fixed multi-view matcher to recover every known OTS source before resolving all 42 remaining sources with unique, mutual, thresholded matches. Passing creates an exact exclusion asset and authorizes only a subsequent outdoor measurement design on a deterministic 1,000-scene definition subset and 1,000-scene validation subset. It does not transport the failed indoor q gate or authorize training, inference, or evaluation.
