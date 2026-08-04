# Daytime Dehazing Density Consistency CARL Multiseed Factorial v1

Status: PLANNED

- Route id: daytime-dehazing-density-consistency-carl-multiseed-factorial-v1
- First operation: DAYTIME_DEHAZING_DENSITY_CONSISTENCY_CARL_MULTISEED_FACTORIAL_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-density-consistency-carl-multiseed-factorial-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The r4 development-screening factorial passed only for scene contrasts clipped to [-0.25,+0.25] dB, and every primary, main-effect, interaction, and density-stratum estimate reached the upper clipping boundary under one training seed. This orthogonal precision-model route repeats the unchanged 2x2 CARL-by-density-consistency intervention with five predeclared paired seed blocks, raw finite scene PSNR differences as primary and mechanism estimands, and a nonbinding futility look after three seeds. It tests whether the combined signal is stable across seeds and whether the CARL main effect, consistency main effect, and interaction can each be classified as material, equivalent-to-small, or unresolved. Architecture, official initialization, Haze4K roles, per-arm training budget, batch shape, optimizer, loss definitions and weights, inference path, and protected-data prohibitions remain unchanged. Development evidence cannot authorize confirmation, sealed-final access, promotion, deployment, or publication.
