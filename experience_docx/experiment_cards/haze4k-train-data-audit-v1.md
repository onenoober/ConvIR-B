# Haze4K train-only pairing, scene grouping, and data-role audit

Status: PLANNED

- Route id: haze4k-train-data-audit-v1
- First operation: HAZE4K_TRAIN_DATA_AUDIT
- Program contract: experience_docx/research_programs/haze4k_local_error_qualification.json
- Experiment spec: experience_docx/experiment_specs/haze4k-train-data-audit-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Audit only Haze4K train, never the official test split, before any local-error measurement or model work. The official ConvIR-B test metric has been reproduced previously, but no candidate, threshold, proxy, or module decision in this program may use test outcomes. This operation verifies loader-compatible pairing, canonical clear-scene grouping, image alignment, filename-token syntax without assigning physical semantics, and a deterministic 2,400/600 train/internal-development split. Passing authorizes only fixed-official-weight local-error measurement on the internal-development scenes.
