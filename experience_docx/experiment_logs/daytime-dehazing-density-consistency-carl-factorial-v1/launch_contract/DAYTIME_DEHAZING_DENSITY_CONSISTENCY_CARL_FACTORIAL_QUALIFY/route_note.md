# Daytime Dehazing Density Consistency CARL Factorial v1

Status: PLANNED

- Route id: daytime-dehazing-density-consistency-carl-factorial-v1
- First operation: DAYTIME_DEHAZING_DENSITY_CONSISTENCY_CARL_FACTORIAL_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-density-consistency-carl-factorial-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

C2 is the evidence-bound orthogonal training-objective route selected after C1's decisive FAIL of observable spatial conditioning. It tests only two fixed losses in a 2x2 factorial: haze-density consistency and contrast-assisted reconstruction loss (CARL). All four arms retain the official ConvIR-B architecture, the same official Haze4K initialization, complete trainable model scope, data roles, scene grouping, augmentation, optimizer, step budget, seeds, evaluation route, and inference path. CARL is a fixed K=4 transfer variant of Cheng et al. rather than a reproduction of its K=5 FFA-Net/RESIDE experiment. Its VGG-19 feature extractor is bound to an exact local state-dict identity and runtime download or cache resolution is prohibited. A single terminal-only comparison family covers the combined effect, both factorial main effects, interaction, three severity-stratum combined effects, and the worst-stratum safety check. Validity failures veto the result, precision is inconclusive-only, and no development result can authorize promotion, deployment, confirmation, sealed-final evaluation, post-processing, or a result-adaptive extension.
