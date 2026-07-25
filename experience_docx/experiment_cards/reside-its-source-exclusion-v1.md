# RESIDE ITS scene-level source exclusion qualification

Status: PLANNED

- Route id: reside-its-source-exclusion-v1
- First operation: ITS_SOURCE_EXCLUSION
- Program contract: experience_docx/research_programs/reside_its_source_exclusion_v1.json
- Experiment spec: experience_docx/experiment_specs/reside-its-source-exclusion-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The current Haze4K continuous-utility route has never used ITS outcomes for selection, but that fact alone does not make all ITS scenes unseen by the Haze4K checkpoint. This source-identity-only route audits every canonical Haze4K source group against every ITS clear scene with a frozen multi-view matcher and conservative family exclusion. It publishes an exact reusable exclusion list only if all 500 provenance-expected indoor source relationships are recovered while positive and negative controls pass. No model, inference, outcome metric, threshold tuning, or protected data is used.
