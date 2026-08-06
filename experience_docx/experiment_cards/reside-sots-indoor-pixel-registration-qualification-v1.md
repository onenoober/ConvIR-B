# RESIDE SOTS Indoor Model-Free Pixel Registration Qualification v1

Status: PLANNED

- Route id: reside-sots-indoor-pixel-registration-qualification-v1
- First operation: QUALIFY_PIXEL_REGISTRATION
- Program contract: experience_docx/research_programs/reside_sots_indoor_asset_qualification_v3.json
- Experiment spec: experience_docx/experiment_specs/reside-sots-indoor-pixel-registration-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

Resolve only the pixel-registration question explicitly left open by the archived geometry-v2 PASS. Bind its terminal record and exact summary, closeout, conclusion, GT, hazy, and combined identities. Apply the frozen ten-pixel center crop to each clear image, then census all 500 mapped pairs with one fixed luminance-gradient spatial score, an integer displacement search, a deterministic cyclic wrong-clear control, fixed eight-pixel shift controls, and a synthetic known-shift engineering fixture. Treat each of the 50 clear scenes as independent and its ten hazy variants as nested. PASS requires broad zero-displacement support and no contradictory pair. FAIL is allowed only when controls are valid and a stable nonzero displacement recurs across enough pairs and scenes. All other outcomes are INCONCLUSIVE. Do not load a model or checkpoint, train, infer, compute restoration metrics, access protected data, display or archive images, or publish per-pair evidence.
