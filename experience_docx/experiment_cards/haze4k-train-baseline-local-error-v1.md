# Haze4K train internal-development baseline local-error measurement

Status: PLANNED

- Route id: haze4k-train-baseline-local-error-v1
- First operation: HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_MEASURE
- Program contract: experience_docx/research_programs/haze4k_local_error_qualification_v3.json
- Experiment spec: experience_docx/experiment_specs/haze4k-train-baseline-local-error-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived train audit and deterministic split qualification established 750 canonical clear scenes and a candidate-unseen 150-scene internal-development role containing 600 nested haze variants. This route performs no training and uses the fixed official Haze4K ConvIR-B checkpoint only on those 150 scenes. It preregisters under-recovery, signed overshoot, and low-demand harm without assuming a monotonic haze-demand direction. Scene repeatability and scene-level uncertainty decide whether a separate non-deployable keep, weaken, or strengthen oracle is warranted. Official test is absent.
