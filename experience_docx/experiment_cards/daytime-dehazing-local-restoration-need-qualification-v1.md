# Daytime Dehazing Local Restoration Need Qualification v1

Status: PLANNED

- Route id: daytime-dehazing-local-restoration-need-qualification-v1
- First operation: DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY
- Program contract: experience_docx/research_programs/daytime_dehazing_spatially_adaptive_restoration_v1.json
- Experiment spec: experience_docx/experiment_specs/daytime-dehazing-local-restoration-need-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Use only program-local development-screening scenes to decide whether bidirectional local restoration need is a repeatable and useful dehazing-domain phenomenon rather than metric noise or global image difficulty. The fixed official ConvIR-B output is perturbed only as a privileged offline diagnostic: within 4-by-4 spatial cells, a five-level residual-strength grid is selected and evaluated by opposite checkerboard pixel halves, repeated over two outcome-blind haze observations, and compared with the optimistic best single global strength for the same scene. A circularly shifted-target selection control tests measurement artifact, while direction stability, within-scene weaken/strengthen coexistence, keep regions, and near-clear damage mitigation test the stated failure mode. The route trains no network, does not test observability, proposes no post-processing method, and cannot access confirmation, NH-Haze, or sealed-final images. PASS authorizes only authoring an S2 mechanism-discovery contract.
