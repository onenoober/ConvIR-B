# Official OTS checkpoint and SOTS-Outdoor asset qualification

Status: PLANNED

- Route id: sots-ots-asset-audit-v1
- First operation: SOTS_OTS_ASSET_AUDIT
- Program contract: experience_docx/research_programs/sots_ots_asset_audit.json
- Experiment spec: experience_docx/experiment_specs/sots-ots-asset-audit-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The next scientific measurement must bind one exact ots-base.pkl file and one exact SOTS-Outdoor paired dataset. This route performs only a deterministic checkpoint hash, image-file census, source-group pairing check and aggregate dataset digest. It does not load the model, run inference, compute restoration metrics or access protected confirmation evidence. Passing authorizes only authoring the separately frozen SOTS local-error measurement contract.
