# Haze4K development bounded local-action oracle

Status: PLANNED

- Route id: haze4k-test-bounded-local-action-oracle-v1
- First operation: HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE
- Program contract: experience_docx/research_programs/haze4k_test_bounded_local_action_oracle_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-bounded-local-action-oracle-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The fixed official Haze4K ConvIR-B exhibited strict local errors repeatable across at least three of four haze variants in 50/100 isolated development scenes, with a Wilson lower bound of 0.4038. This authorizes one non-deployable oracle, not module design. The oracle freezes the residual-scale actions weaken 0.8, keep 1.0, and strengthen 1.2. Its decisive contrast is deliberately stringent: a GT-selected 32-pixel spatial action field must beat a GT-selected per-image best uniform action drawn from the identical bounded set. This isolates spatial selection headroom from ordinary whole-image strength calibration. Candidate-confirmation data, training, proxy fitting, and structure selection remain prohibited.
