# v3j Bounded Safe-Correction Audit Evidence

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3j-bounded-safe-correction-audit.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`

Status: planned on 2026-07-11.

This route is a no-training diagnostic audit unless v3j-A and v3j-B gates
authorize a later stage. It uses train-derived Haze4K splits only and does not
touch locked test.

## Planned Stages

| Stage | Question | Gate |
| --- | --- | --- |
| v3j-A bounded action-space audit | Does the bounded output-residual actuator retain privileged teacher gain? | Primary bounded projection must stably beat hard D7c without tail regression. |
| v3j-B direct correction OOF diagnostic | Can tiny heads directly regress bounded residuals and replay safely? | Same replay gate as v3j-A, with OOF training only. |

## Primary Files

- `v3j_route_decision.md`
- `v3j_source_of_truth_manifest.json`
- `v3j_forbidden_flow_audit.json`
- `no_locked_test_audit.json`
- `bounded_action_space_definition.md`
- `run_v3j_a_bounded_action_audit.sh`
