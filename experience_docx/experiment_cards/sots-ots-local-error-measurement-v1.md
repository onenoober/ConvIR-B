# Official OTS ConvIR-B local paired-GT error measurement on SOTS-Outdoor

Status: PLANNED

- Route id: sots-ots-local-error-measurement-v1
- First operation: SOTS_OTS_LOCAL_ERROR_MEASURE
- Program contract: experience_docx/research_programs/sots_ots_local_error.json
- Experiment spec: experience_docx/experiment_specs/sots-ots-local-error-measurement-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The asset audit qualified one exact official OTS checkpoint and one complete SOTS-Outdoor copy. This route is not a retry of the failed OTS tau claim: it changes the target domain, baseline identity and measurement target, using paired hazy-to-GT tile error rather than depth times beta. It evaluates all 500 hazy variants nested within 492 clear-source groups, requires global model competence, and tests a model-output-independent local correction imbalance against a 180-degree spatial rotation control. The result is development-screening diagnostic evidence only; SOTS cannot later be relabeled as unseen confirmation evidence.
