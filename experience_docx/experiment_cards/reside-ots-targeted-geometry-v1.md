# RESIDE OTS targeted geometric source resolution

Status: PLANNED

- Route id: reside-ots-targeted-geometry-v1
- First operation: OTS_TARGETED_GEOMETRY
- Program contract: experience_docx/research_programs/reside_ots_outdoor_measurement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-targeted-geometry-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The full 542-query route found 39 new bidirectional mappings and exposed three uncounted relationships. Two are high-confidence mutual matches to OTS IDs already present in the parent exclusion union: Haze4K test 69 to OTS 0354 and Haze4K train 977 to OTS 1964. The remaining Haze4K train 937 query has pHash distance zero and correlation 0.999979 to OTS 0230 but a near-tied global-correlation competitor. This orthogonal route uses a frozen RootSIFT and RANSAC geometry test only on these targets, fixed controls, and a bounded candidate shortlist. It resolves source relationships first and deduplicates OTS IDs second; it does not require every relationship to contribute a new ID.
