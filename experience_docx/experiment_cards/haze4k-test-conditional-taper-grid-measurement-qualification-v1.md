# Haze4K conditional taper-grid measurement qualification

Status: PLANNED

- Route id: haze4k-test-conditional-taper-grid-measurement-qualification-v1
- First operation: HAZE4K_TEST_CONDITIONAL_TAPER_GRID_MEASUREMENT_QUALIFICATION
- Program contract: experience_docx/research_programs/haze4k_conditional_taper_grid_measurement_qualification_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-conditional-taper-grid-measurement-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The completed parent conditional grid route precisely failed only the equal-weight 32-pixel box projection target: mean worst-held-out-grid regret was 0.142335 dB, its one-sided 95 percent interval was [0.125254, 0.159921] dB, and achieved precision was adequate. This orthogonal Stage-1 route changes only grid_definition by replacing equal source-grid weights with a fixed strictly-positive half-sample raised-cosine taper normalized per pixel as a partition of unity. It keeps the same model, data, three hard actions, four grid origins, held-out hard-tile oracle, estimand, 0.10 dB margin, scene bootstrap, controls, and safety gates. The archived box projection is recomputed from the same inference and must reproduce the parent mean regret within 1e-8 dB. Boundary, origin, clipping, and taper-minus-box results are diagnostic only. No terminal authorizes Stage 2.
