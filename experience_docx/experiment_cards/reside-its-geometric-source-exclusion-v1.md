# RESIDE ITS local-geometric scene-level source exclusion qualification

Status: PLANNED

- Route id: reside-its-geometric-source-exclusion-v1
- First operation: ITS_GEOMETRIC_SOURCE_EXCLUSION
- Program contract: experience_docx/research_programs/reside_its_geometric_source_exclusion_v1.json
- Experiment spec: experience_docx/experiment_specs/reside-its-geometric-source-exclusion-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived global pHash-correlation census was internally valid but recovered only 6 of the 500 provenance-expected Haze4K-to-ITS relationships. This orthogonal route changes only the source-identity method: a bounded multi-index ORB retrieval stage cheaply recalls local candidates, a wide ORB homography stage ranks them, and the already successful OTS bidirectional RootSIFT-RANSAC rule makes the final decision. The prior six relationships, fixed transformed identities, qualified OTS pairs, and all SOTS-outdoor scenes are frozen controls. Thresholds and shortlist sizes are fixed before results and cannot be changed to force 500. No model, outcome, training, or protected data is accessed.
