# Outcome-blind Haze4K official-test scene grouping and 100/150 role isolation

Status: PLANNED

- Route id: haze4k-test-scene-split-v1
- First operation: HAZE4K_TEST_SCENE_SPLIT_QUALIFY
- Program contract: experience_docx/research_programs/haze4k_test_local_error_replication_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-scene-split-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The official ConvIR-B checkpoint was trained on Haze4K train and its aggregate official-test metric was previously reproduced, so train-only local-error prevalence cannot answer whether the same constructs appear on unseen Haze4K scenes. This qualification performs one complete but outcome-blind official-test structure census, freezes a deterministic 100-scene development-screening and 150-scene candidate-confirmation partition, and materializes physically separated role assets. It loads no model or checkpoint and computes no restoration result or metric. The full test is baseline-exposed but remains candidate-unseen for this research route.
