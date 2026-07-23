# RESIDE OTS synthetic-outdoor local optical-thickness qualification

Status: PLANNED

- Route id: reside-ots-outdoor-measurement-v1
- First operation: OTS_OUTDOOR_CALIBRATE
- Program contract: experience_docx/research_programs/reside_ots_outdoor_measurement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-outdoor-measurement-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The direct parent resolved all 42 expected Haze4K outdoor source relationships and authorized an 8,006-scene OTS pool after a conservative 964-scene exclusion union. OTS is a mainstream outdoor synthetic dehazing source with 35 official haze variants and one MATLAB v7.3/HDF5 depth field per clear scene. This route uses a deterministic 1,000-scene subset: 375 definition scenes freeze the two filename-token roles and 625 disjoint validation scenes supply formal scene-level uncertainty. Every selected scene uses all 35 nested variants. Depth is read with the already available cloud Octave executable, while all statistics remain scene-level. The estimand is synthetic optical-thickness measurement qualification only, not real-haze restoration demand.
