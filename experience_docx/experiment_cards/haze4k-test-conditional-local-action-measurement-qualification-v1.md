# Haze4K conditional local-action grid-stability measurement qualification

Status: PLANNED

- Route id: haze4k-test-conditional-local-action-measurement-qualification-v1
- First operation: HAZE4K_TEST_CONDITIONAL_LOCAL_ACTION_MEASUREMENT_QUALIFICATION
- Program contract: experience_docx/research_programs/haze4k_conditional_local_action_measurement_qualification_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-conditional-local-action-measurement-qualification-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The completed parent route found that a cross-haze-variant invariant action field was not measurement-qualified: mean regret was 1.0269 dB, its one-sided 95 percent upper bound was 1.1720 dB, and SSIM/color tail safety did not pass. This orthogonal route therefore changes the measurement target rather than relaxing any gate. It treats each haze variant as an observed condition and tests only whether source-grid tile-aggregated continuous utilities retain value on a held-out grid origin within that same condition. The source fields are piecewise-constant tile summaries projected by fixed geometry, never raw per-pixel GT utilities, so the held-out tile oracle cannot be reconstructed algebraically. No terminal authorizes stage 2: PASS and INCONCLUSIVE return only to stage 1, while FAIL authorizes nothing. The closest archived scene SD makes n=100 formally underpowered for a 0.05 dB planning half-width, so this is declared descriptive capacity and any positive terminal is only a conditional grid-stability screen pass.
