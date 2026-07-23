# Official OTS ConvIR-B signed high-demand overshoot measurement on SOTS-Outdoor

Status: PLANNED

- Route id: sots-ots-signed-overshoot-v1
- First operation: SOTS_OTS_SIGNED_OVERSHOOT_MEASURE
- Program contract: experience_docx/research_programs/sots_ots_signed_overshoot.json
- Experiment spec: experience_docx/experiment_specs/sots-ots-signed-overshoot-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The completed paired-GT correction measurement rejected the claim that high-demand regions are corrected less strongly, but its unsigned MSE ratio cannot distinguish accurate recovery from crossing beyond GT. This orthogonal measurement-target route uses the same fixed OTS checkpoint and complete SOTS population, defines high-demand tiles only from paired input error, and measures signed projection beyond GT with both relative and absolute materiality guards. It performs no training, threshold selection or protected-data access.
