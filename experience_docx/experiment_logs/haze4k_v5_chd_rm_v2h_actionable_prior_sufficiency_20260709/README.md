# CHD-RM v2h Actionable Prior Sufficiency

Status: `COMPLETED_GATE_PASS`

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

## v2h-A Result

# v2h-A D7c Risk-Coverage Calibration

Status: `COMPLETED_GATE_PASS`

Decision label: `V2H_A_D7C_RISK_COVERAGE_PASS_AUTHORIZE_SHADOW`

Policy: train-calib selects thresholds; val-inner is report-only. No locked test, D2, F5, v3, RARM, adapter training, or new probe training was run.

## Primary Operating Point

| Score | Coverage | Action recall | Low-adj recall | Negative false | Negative false p95 | Isolated hit | Density action recall | Density negative false |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D7c fixed | 0.302695 | 0.548312 | 0.155904 | 0.002974 | 0.047619 | 0.022366 | 0.448391 | 0.047786 |

## Decision

v2h-B shadow-modulation only

