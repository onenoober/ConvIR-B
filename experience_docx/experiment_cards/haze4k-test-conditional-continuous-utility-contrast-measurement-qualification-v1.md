# Haze4K conditional continuous-utility measurement qualification

Status: PLANNED

- Route id: haze4k-test-conditional-continuous-utility-contrast-measurement-qualification-v1
- First operation: HAZE4K_TEST_CONDITIONAL_CONTINUOUS_UTILITY_PRECISION_PILOT
- Program contract: experience_docx/research_programs/haze4k_conditional_continuous_utility_measurement_qualification_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-conditional-continuous-utility-contrast-measurement-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived taper and content-aligned routes precisely rejected hard-action regret targets but did not directly test whether their underlying continuous utility contrasts are stable. This orthogonal Stage-1 route changes only measurement_target. Before the full 100-scene qualification is materialized, the first operation runs a fixed 32-scene variance-only precision supplement. It withholds the primary mean and uses the conservative maximum of a one-sided chi-square and scene-bootstrap SD upper bound. Only a bound supporting the frozen 0.025 dB precision distance at n=100 authorizes the full operation; otherwise the route stops as BLOCKED_PRECISION. Model, checkpoint, actions, metric, taper carrier, grouping, data role, and protected-data prohibitions remain frozen.
