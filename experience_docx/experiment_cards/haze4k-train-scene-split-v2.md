# Haze4K 750-scene train and internal-development split qualification

Status: PLANNED

- Route id: haze4k-train-scene-split-v2
- First operation: HAZE4K_TRAIN_SCENE_SPLIT_QUALIFY
- Program contract: experience_docx/research_programs/haze4k_local_error_qualification_v2.json
- Experiment spec: experience_docx/experiment_specs/haze4k-train-scene-split-v2.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived train-only audit established 750 canonical clear scenes, each with four nested haze variants, rather than 3,000 independent scenes. This evidence-backed reopen validates a deterministic 600-scene training and 150-scene internal-development split while retaining all 3,000 hazy observations. It accesses only Haze4K train, never official test or any model checkpoint. Passing authorizes a separate fixed-official-weight local-error measurement route on the 150 internal-development scenes.
