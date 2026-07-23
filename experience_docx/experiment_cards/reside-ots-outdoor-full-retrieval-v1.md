# RESIDE OTS full-query outdoor source retrieval

Status: PLANNED

- Route id: reside-ots-outdoor-full-retrieval-v1
- First operation: OTS_FULL_RETRIEVAL
- Program contract: experience_docx/research_programs/reside_ots_outdoor_measurement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-outdoor-full-retrieval-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The parent source-overlap route reproduced 458 strict Haze4K-to-OTS mappings and 492 SOTS-outdoor exclusions, but showed that the remaining provenance shortfall of 42 is embedded within 542 strict-unmatched Haze4K source groups. This orthogonal source-identity route searches all 542 groups rather than assuming a known 42-query subset. The frozen matcher must recover all 458 known positive controls, admit no SOTS-indoor negative control, identify exactly 42 unique mutual OTS matches outside the 920 verified exclusions, and retain at least 6,000 eligible OTS scenes. Passing authorizes only a separately specified outdoor measurement design; no training, inference, dehazing evaluation, or transport of the failed indoor q gate is permitted.
