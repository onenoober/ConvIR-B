# Official ConvIR-B OTS baseline behavior complete-scene supplement

Status: PLANNED

- Route id: reside-ots-baseline-behavior-supplement-v1
- First operation: OTS_BASELINE_BEHAVIOR_SUPPLEMENT
- Program contract: experience_docx/research_programs/reside_ots_baseline_behavior_supplement.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-baseline-behavior-supplement-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The direct parent ended INCONCLUSIVE solely because a long-edge-320 resize made the short edge too small for an internal official ConvIR-B reflection pad in 218 aspect-ratio-extreme scenes. Parent and dataset identities matched, the GPU contract passed, and 407 scenes completed, but those diagnostic outcomes cannot substitute for the frozen 625-scene population. This one supplement recomputes all 625 scenes and all three frozen variants while preserving the parent's aspect-ratio-preserving content resize exactly. The sole change reflect-extends the bottom and right model canvas to at least 256 pixels per dimension and then to a multiple of 32; prediction is cropped back to the unpadded content before GT, depth, tau, region, PSNR, SSIM or mismatch measurement. The 256 model-canvas minimum follows directly from the unchanged production graph: a 1/4-scale feature is pooled by 8 before ReflectionPad2d(7), so the pooled dimension must be at least 8. Synthetic 32x320 and 320x32 content fixtures cover both exact formal aspect-ratio boundaries and exercise 256x320 and 320x256 CUDA model canvases. Model, content resolution, population, variants, regions, estimand, margins, negative control, uncertainty, and terminal rules remain unchanged.
