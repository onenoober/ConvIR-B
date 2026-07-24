# Haze4K conditional content-aligned region measurement qualification

Status: PLANNED

- Route id: haze4k-test-conditional-content-aligned-region-measurement-qualification-v1
- First operation: HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_MEASUREMENT_QUALIFICATION
- Program contract: experience_docx/research_programs/haze4k_conditional_content_aligned_region_measurement_qualification_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-conditional-content-aligned-region-measurement-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived taper route precisely failed: mean worst-held-out-partition regret was 0.134003 dB and its one-sided 95 percent lower bound 0.118104 dB remained above the frozen 0.10 dB maximum, while matched box reproduction and achieved precision passed. Its same-origin diagnostics did not support simple boundary tapering as sufficient and left mismatch of the fixed content-agnostic spatial unit unresolved. This orthogonal Stage-1 route changes only grid_definition to a deterministic source-only edge-watershed region unit. It keeps the official model, 100 development scenes, 400 nested haze variants, three hard actions, four phases, RGB-MSE/PSNR target, 0.10 dB margin, uncertainty, controls, and safety gates frozen. The taper path is recomputed from the same inference and must reproduce its archived mean within 1e-8 dB. No terminal authorizes Stage 2.
