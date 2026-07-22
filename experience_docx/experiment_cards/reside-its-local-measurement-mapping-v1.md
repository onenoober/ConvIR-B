# RESIDE ITS local measurement physical-mapping supplement

Status: PLANNED

- Route id: reside-its-local-measurement-mapping-v1
- First operation: MEASUREMENT_SUPPLEMENT_ONLY
- Program contract: experience_docx/research_programs/reside_local_measurement_mapping.json
- Experiment spec: experience_docx/experiment_specs/reside-its-local-measurement-mapping-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived first attempt measured zero scenes because it required haze and transmission filename stems to match, although the qualified ITS layout guarantees only same-scene groups of ten. This adjacent second attempt preserves the deterministic 1,000-scene definition sample, all 1,000 validation scenes, three severities, scene-level independence, q metric, negative control, and thresholds. It changes only the mapping: each selected transmission is paired to the unique minimum-RMSE haze among the ten same-scene candidates under the synthetic atmospheric model. Passing authorizes only official ConvIR-B indoor baseline behavior measurement; outdoor and real-haze calibration remain mandatory independent stages.
