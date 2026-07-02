# DTA-v3.7 D8/D9 Reconciliation Audit

Decision: `D8_METRICS_USABLE_METADATA_RECONCILIATION_REQUIRED`

## D8 status-derived scope

- folds: `[0, 1, 2, 3, 4]`
- seeds: `[3407, 3411, 2026]`
- expected tasks: `15`
- outputdiff group starts/dones: `15` / `15`
- raw D1 full 5x3 status claim: `True`

## Metric bottom line

- D8 summary strict pass: `True`
- D9 locked decision: `D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING`

## Inconsistencies

- `raw_d1_full_5x3_run`: status `True` vs artifact `False` (metadata); D8 status says raw D1 5x3 was run; summary retained D7 value.
- `summary_phase`: status `D8_fixed_formal_confirmation` vs artifact `D7_fixed_outputdiff_confirmation` (metadata); D8 summary phase should not retain the D7 confirmation label.
- `aggregate_phase`: status `D8_fixed_formal_confirmation` vs artifact `D7_fixed_outputdiff_confirmation` (metadata); D8 aggregate phase should not retain the D7 confirmation label.
- `outer_groups`: status `15` vs artifact `4` (metadata); Broader D8 scope is 5 folds x 3 seeds; current artifact records D7 outer group count.
- `aggregate_outer_groups`: status `15` vs artifact `4` (metadata); Broader D8 aggregate should record 15 fold/seed groups.

## Interpretation

- The D8 metric values remain usable as a train-derived formal confirmation record, but the summary/aggregate metadata retained D7 labels and should not be read as a clean route-state source without this audit.
- The D9 locked one-shot remains a failed confirmation and must not be used for threshold, feature, action, or checkpoint tuning.
