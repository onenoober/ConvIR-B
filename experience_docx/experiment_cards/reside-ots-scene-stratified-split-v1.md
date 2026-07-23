# RESIDE OTS scene-stratified role and haze-budget design

Status: PLANNED

- Route id: reside-ots-scene-stratified-split-v1
- First operation: OTS_SCENE_STRATIFIED_SPLIT
- Program contract: experience_docx/research_programs/reside_ots_data_pool_design.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-scene-stratified-split-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The authorized OTS exclusion union leaves 8,006 source-independent outdoor clear scenes, but using all 280,210 associated hazy images would be unnecessarily expensive. This route freezes scene roles before model work: 4,000 training scenes, 500 model-development validation scenes, 1,000 measurement-definition scenes, 1,000 measurement-validation scenes, and 1,506 reserve scenes. Allocation uses only clear-image content and paired-depth descriptors, never model outcomes. Training and model-development roles use five fixed haze-parameter grid points per scene; both measurement roles retain access to all 35 variants so later measurement validity is not constrained by this compute reduction. Passing authorizes only an outdoor measurement-definition route and reusable data-role manifests, not model selection or training.
