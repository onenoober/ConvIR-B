# Official ConvIR-B OTS synthetic-outdoor baseline behavior measurement

Status: PLANNED

- Route id: reside-ots-baseline-behavior-v1
- First operation: OTS_BASELINE_BEHAVIOR_MEASURE
- Program contract: experience_docx/research_programs/reside_ots_baseline_behavior.json
- Experiment spec: experience_docx/experiment_specs/reside-ots-baseline-behavior-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The parent route qualified the official OTS tau=beta*depth field on 625 independent validation scenes and authorized only a separately frozen official-baseline behavior design. This route reuses those exact 625 scene IDs, fixes three official beta severities (0.04, 0.10, 0.20) at airlight 0.90, and runs the immutable official Haze4K ConvIR-B checkpoint once per variant. The model output never defines tau. The primary scene-level mismatch indicator requires materially weaker relative error correction in the true high-tau than low-tau region and separation from a 180-degree spatially rotated tau control in at least two of three severities. Full-image PSNR/SSIM and low-tau excess-error harm are secondary. All inference is development-screening only and cannot establish a causal mechanism or real-haze behavior.
