# CHD-RM v2h Actionable Prior Sufficiency

Status: `PLANNED`

Decision label: `PLANNED_V2H_ACTIONABLE_PRIOR_SUFFICIENCY_AUDIT`

Route card: `experience_docx/experiment_cards/haze4k-chd-rm-v2h-actionable-prior-sufficiency.md`.

Central index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.

## Purpose

v2h tests whether D7c is sufficient as a conservative actionable prior before any no-op RARM or bounded adapter work. It is not a training route.

## Authorized Stages

- v2h-A: D7c risk-coverage calibration under the v2g three-state target.
- v2h-B: D7c shadow-modulation oracle upper-bound audit only if v2h-A passes.
- v2h-C: OOF stability only if A/B justify continuing.
- v2h-D: FAM2-only no-op equivalence only if A/B justify continuing.

## Primary Files

- `v2h_route_decision.md`
- `v2h_gate_definition.md`
- `v2h_source_of_truth_manifest.json`
- `no_locked_test_audit.json`
- `run_v2h_a_d7c_risk_coverage.sh`
- `status.txt`

## Policy

No locked Haze4K test, D2, F5, v3, RARM connection/training, adapter training, or new selective probe training is authorized by this route.
