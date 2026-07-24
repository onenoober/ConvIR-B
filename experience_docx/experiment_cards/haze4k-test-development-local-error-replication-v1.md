# Haze4K test development local-error cross-synthetic replication

Status: PLANNED

- Route id: haze4k-test-development-local-error-replication-v1
- First operation: HAZE4K_TEST_DEVELOPMENT_LOCAL_ERROR_REPLICATE
- Program contract: experience_docx/research_programs/haze4k_test_local_error_measurement_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-development-local-error-replication-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The outcome-blind Haze4K official-test census created disjoint 100-scene development and 150-scene candidate-confirmation assets. This route delivers only the 100-scene asset and performs zero training with the fixed official Haze4K ConvIR-B. It separately tests a SOTS-consistent pattern—signed overshoot exists, high-demand under-recovery is not prevalent, and overshoot is not high-demand specific—and the stricter prevalence of constructs repeatable in at least three of four variants. Pattern replication does not automatically qualify an oracle or module.
