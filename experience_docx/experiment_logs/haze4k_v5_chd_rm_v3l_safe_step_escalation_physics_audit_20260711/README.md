# v3l Safe-Step Escalation And Physics Audit

Status: `CLOSED_B_WEAK_STOP`.

Route card:
`experience_docx/experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3l-safe-step-escalation-physics-audit.md`

This route starts after v3k's provisional diagnosis. It first freezes canonical
context direct-head operators and verifies deterministic replay, then audits
oracle step-size headroom and privileged transmission-map risk. A0 and A1
passed, but B found transmission-only risk too weak to justify a physics policy.

Locked Haze4K test access, canary expansion, backbone training, route-confirm
strategy selection, and new model search are not authorized.

## A0 Contract

- Fit only context direct heads for fixed seeds `3407` and `3408`.
- Save direct-head artifacts under cloud-only `cloud_only_artifacts/`.
- Sync only artifact path/SHA manifests and compact summaries to GitHub.
- Replay gate: exact row identity/order, max per-image PSNR delta difference
  `<= 1e-6`, exact severe set at `<= -0.2 dB`, max direct tensor replay
  difference `<= 1e-7`, and stable artifact SHA.

## Expected A0 Outputs

- `v3l_a0_canonical_operator_closeout.json`
- `v3l_a0_canonical_operator_artifact_manifest.json`
- `v3l_a0_canonical_operator_artifact_manifest.csv`
- `v3l_a0_probe_training_history.csv`
- cloud-only replay/tensor equivalence CSVs
- cloud-only `cloud_only_artifacts/*.pt`

## A1 Contract

- Read only frozen A0 cloud-only artifacts; do not refit heads.
- Compute OOF oracle step-size upper bounds at image, block, and pixel
  granularity.
- Compare oracle policies against fixed `alpha=0.125` on OOF for both `D_ref`
  and `D_rep`.
- Treat route-confirm output as confirm-audit-only; it is not allowed to select
  alpha, risk thresholds, or any deployable policy.
- Do not authorize canary, locked test, confidence/router training, or new model
  search.

## A1 Result

- Decision:
  `V3L_A1_ORACLE_GRANULARITY_PASS_AUTHORIZE_B_PHYSICS_RISK_AUDIT_ONLY`.
- OOF dual-operator pass policies: image oracle, 16x16 block oracle, 32x32 block
  oracle, and pixel scalar oracle.
- Fixed `alpha=0.125` remains the only non-oracle safe reference; larger fixed
  alphas improve mean but reintroduce severe tails.

## B Contract

- Audit actual Haze4K physics metadata availability first.
- Use privileged transmission maps only; do not assume airlight/beta/depth files
  exist.
- Gate B on OOF only for both `D_ref` and `D_rep`; route-confirm is
  confirm-audit-only.
- Do not train a deployable t/A estimator or choose canary thresholds in B.

## B Result

- Decision:
  `V3L_B_PRIVILEGED_TRANSMISSION_RISK_WEAK_STOP_NO_PHYSICS_POLICY`.
- Haze4K exposes `train/trans` and `test/trans`; no airlight/beta/depth/atmos
  metadata were found.
- Privileged transmission features failed the pre-registered direct-severe OOF
  AUC gate on both frozen operators: best direct-severe AUC was about `0.635`
  for `D_ref` and `0.631` for `D_rep`, below the `0.65` threshold.
- Low-optimal-alpha and wrong-or-harmful signals were only moderate, so they do
  not rescue the failed direct-severe gate.

No next stage is authorized. Canary, locked test, confidence/router training,
deployable physics-estimator training, and larger direct-head/model search all
remain blocked.
