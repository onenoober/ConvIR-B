# RESIDE ITS cost-bounded targeted direct relationship completion

Status: PLANNED

- Route id: reside-its-targeted-direct-geometry-v1
- First operation: ITS_TARGETED_DIRECT_RELATIONSHIP
- Program contract: experience_docx/research_programs/reside_its_targeted_direct_relationship_v1.json
- Experiment spec: experience_docx/experiment_specs/reside-its-targeted-direct-geometry-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived local-geometric census directly resolved 426 of 500 provenance-expected Haze4K-to-ITS relationships, but its exact 16-bit descriptor subkey retrieval returned no geometrically eligible candidate for 573 of the 574 unresolved queries and its later ITS-to-ITS closure exceeded its frozen bound. This adjacent, single-attempt route isolates the highest-value unresolved question: whether a multi-probe LSH retriever can place the known direct source in a fixed top-64 shortlist and then complete only the 574 unresolved Haze4K-to-ITS relationships with the already qualified direct geometry rule. It inherits the 426 relationships diagnostically, forbids ITS-to-ITS propagation, adds spatial support against localized false matches, and stops before formal search unless exact recall, positive, negative and runtime qualification gates pass. The control phase is bounded at 900 seconds and the entire application at 1,800 seconds.
